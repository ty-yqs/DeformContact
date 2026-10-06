import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, TAGConv, knn
import torch
import torch.nn.functional as F

from utils.pos_encoding import to_log_freq


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(num_heads)])

    def forward(self, x_resting, x_rigid, batch_resting=None, batch_rigid=None):
        # Batch.from_data_list concatenates every sample's nodes onto one node
        # axis, so without the mask a resting node can attend to another
        # sample's rigid nodes. That is harmless when each sample has its own
        # resting geometry, but when the whole dataset shares one rest mesh
        # (e.g. the HRA retina) the resting queries are identical across
        # samples and every sample in the batch collapses onto the same output.
        # batch_resting / batch_rigid are the usual PyG ``.batch`` vectors.
        mask = None
        if batch_resting is not None and batch_rigid is not None:
            mask = batch_resting[:, None] != batch_rigid[None, :]

        outputs = []
        for head in self.attention_heads:
            scores = torch.mm(head(x_resting), head(x_rigid).transpose(0, 1))
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            if mask is not None:
                # A row with every entry masked would softmax to NaN.
                attn_weights = torch.nan_to_num(attn_weights)
            output = torch.mm(attn_weights, x_rigid)
            outputs.append(output)

        return torch.cat(outputs, dim=-1)

class GraphNet(nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim, encoder_layers, decoder_layers, dropout_rate, knn_k, backbone,use_mha, num_mha_heads,mode, mha_masked=False, skip_pos=False, skip_freqs=3):
        super(GraphNet, self).__init__()

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.backbone = backbone

        self.conv_layers_resting = nn.ModuleList()
        self.conv_layers_rigid = nn.ModuleList()
        self.use_mha = use_mha
        self.mha_masked = mha_masked

        self.dropout_rate = dropout_rate
        self.knn_k = knn_k
        self.mode = mode

        conv_layer = GATConv if self.backbone == "GATConv" else GCNConv if self.backbone == "GCNConv" else TAGConv

        input_dims_resting = input_dims.copy()
        input_dims_rigid = input_dims.copy()

        for _ in range(self.encoder_layers):
            self.conv_layers_resting.append(conv_layer(input_dims_resting[0], hidden_dim))
            input_dims_resting[0] = hidden_dim 

        for _ in range(self.encoder_layers):
            self.conv_layers_rigid.append(conv_layer(input_dims_rigid[1], hidden_dim))
            input_dims_rigid[1] = hidden_dim  

        # Decoder
        decoder = []
        if self.use_mha:
            input_dim_decoder = hidden_dim * (num_mha_heads+1)
        else:
            input_dim_decoder = hidden_dim *2
        for _ in range(self.decoder_layers):
            decoder.append(nn.Linear(input_dim_decoder, hidden_dim))
            decoder.append(nn.ReLU())  
            decoder.append(nn.Dropout(self.dropout_rate))
            input_dim_decoder = hidden_dim
        decoder.append(nn.Linear(hidden_dim, output_dim))
        self.decoder = nn.Sequential(*decoder)
        self.multihead_attention = MultiHeadAttention(hidden_dim, num_heads=num_mha_heads)

        # Optional high-frequency skip; see forward() for why it is built in the
        # contact frame. Only constructed when asked for, so checkpoints trained
        # without it still load.
        self.skip_pos = skip_pos
        self.skip_freqs = skip_freqs
        if self.skip_pos:
            skip_in_dim = 3 * (1 + 2 * int(skip_freqs))  # to_log_freq(x, n, 1)
            self.skip_encoder = nn.Sequential(
                nn.Linear(skip_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
            # Zero the output layer so the branch starts as an exact no-op: a
            # skip_pos run then begins from the same function as the stock model
            # and any difference is attributable to what the branch learns.
            nn.init.zeros_(self.skip_encoder[-1].weight)
            nn.init.zeros_(self.skip_encoder[-1].bias)

    def forward(self, graph_resting, graph_rigid):
        # For resting graph
        x_resting = graph_resting.x
        for conv in self.conv_layers_resting:
            x_resting = F.relu(conv(x_resting, graph_resting.edge_index))
            x_resting = F.dropout(x_resting, p=self.dropout_rate, training=self.training)

        # For rigid graph
        x_rigid = graph_rigid.x
        for conv in self.conv_layers_rigid:
            x_rigid = F.relu(conv(x_rigid, graph_rigid.edge_index))
            x_rigid = F.dropout(x_rigid, p=self.dropout_rate, training=self.training)

        

        batch_resting = batch_rigid = None
        if self.mha_masked:
            # Plain (unbatched) Data has no .batch attribute.
            batch_resting = getattr(graph_resting, "batch", None)
            batch_rigid = getattr(graph_rigid, "batch", None)

        pooled_features = self.multihead_attention(
            x_resting, x_rigid, batch_resting, batch_rigid
        )

        x_combined = torch.cat([x_resting, pooled_features], dim=-1)


        # Pass through the decoder to get the deformed positions
        x_out = self.decoder(x_combined)

        if self.skip_pos:
            # A direct path from the positional encoding to the output.
            #
            # The naive form of this is useless for two reasons, both fixed by
            # putting the encoding in the contact frame:
            #   * graph_resting.x is identical for every sample (the HRA sets
            #     share one rest mesh), so a branch computed from it alone emits
            #     a fixed field and can never move the bump onto the needle;
            #   * the trunk encodes with to_log_freq(x, 3, 1), whose highest band
            #     is 4 rad per unit -- a 1.6 mm wavelength at length_scale=1000,
            #     coarser than the ~0.3 mm dent it has to represent. The skip
            #     therefore gets its own, wider band budget (skip_freqs).
            # Coordinates are relative to the tip, so the target becomes one
            # learned shape shared by every sample instead of a bump that sits
            # somewhere new in absolute coordinates each time.
            rel = graph_resting.pos - self._needle_tip(graph_resting, graph_rigid)
            x_out = x_out + self.skip_encoder(to_log_freq(rel, self.skip_freqs, 1))

        # Building a graph with deformed positions
        deformed_graph = graph_resting.clone()
        if  self.mode == "res":
            deformed_graph.pos += x_out
        elif self.mode == "rec":
            deformed_graph.pos = x_out

        return deformed_graph

    @staticmethod
    def _needle_tip(graph_resting, graph_rigid):
        """Per-resting-node needle tip, in graph units.

        The rigid graph is a query set built as ``unit_sphere * radius + tip``,
        so each sample's rigid centroid is that sample's tip, up to the sphere's
        own centroid offset -- a constant of the point set that the skip branch
        can absorb. Unbatched graphs fall back to the global centroid.
        """
        batch_rigid = getattr(graph_rigid, "batch", None)
        if batch_rigid is None:
            return graph_rigid.pos.mean(dim=0, keepdim=True)

        pos = graph_rigid.pos
        dtype, device = pos.dtype, pos.device
        n_graphs = int(batch_rigid.max()) + 1
        tip = torch.zeros(n_graphs, pos.size(-1), dtype=dtype, device=device)
        count = torch.zeros(n_graphs, dtype=dtype, device=device)
        tip.index_add_(0, batch_rigid, pos)
        count.index_add_(0, batch_rigid, torch.ones_like(batch_rigid, dtype=dtype))
        tip = tip / count.clamp(min=1).unsqueeze(-1)

        batch_resting = getattr(graph_resting, "batch", None)
        return tip[:1] if batch_resting is None else tip[batch_resting]

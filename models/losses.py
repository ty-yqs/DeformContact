import torch
import torch.nn as nn

class GradientConsistencyLoss(nn.Module):
    def __init__(self):
        super(GradientConsistencyLoss, self).__init__()

    def forward(self, pred_graphs_batched, soft_rest_graphs_batched):
        node_diffs = pred_graphs_batched.pos - soft_rest_graphs_batched.pos
        node_pos_rest = soft_rest_graphs_batched.pos
        node_pos_pred = pred_graphs_batched.pos

        edge_diffs_rest = node_pos_rest[soft_rest_graphs_batched.edge_index[1]] - node_pos_rest[soft_rest_graphs_batched.edge_index[0]]
        edge_diffs_pred = node_pos_pred[pred_graphs_batched.edge_index[1]] - node_pos_pred[pred_graphs_batched.edge_index[0]]

        cross_shape_diffs = edge_diffs_rest - edge_diffs_pred

        loss = cross_shape_diffs.norm(p=2, dim=-1).sum() / len(soft_rest_graphs_batched.edge_index[0])  # Scale by the number of edges

        return loss


def displacement_weight(target_disp, scheme="none", strength=1.0, ref=None,
                        tip_dist=None, sigma=None, base=1.0, clip=10.0):
    """Per-node multiplier, shape ``(N, 3)``, for the displacement loss.

    ``target_disp`` and ``ref``/``tip_dist``/``sigma`` are all in graph units
    (``length_scale`` x metres), the same space the loss is computed in.

    Plain L1 over the 5389-node retina is dominated by the 57% of nodes that
    displace less than 0.5 um: on the trained baseline those nodes carry 67% of
    the loss while everything above 30 um -- the contact peak, the only part
    that matters physically -- carries 0.15%. The optimizer is near-indifferent
    to the peak, and the model reproduces only 39% of it. Two schemes push
    back:

    ``magnitude``  ``base + strength * |gt| / ref``, clipped. ``base`` is the
        weight of a static node; ``base=1`` is a mild tilt, ``base=0`` hands
        the loss almost entirely to the moving nodes. At ``strength=1`` the
        peak's share goes 0.15% -> 0.36% (no real change); at ``strength=10``
        -> 1.7%; with ``base=0`` -> 5.8%.
    ``contact``    ``base + strength * exp(-d_tip / sigma)``. Measured to be
        nearly useless on this dataset -- only 0.59% of nodes sit within 1 mm
        of the tip, and they are not the high-displacement ones (share goes
        0.15% -> 0.27% even at ``strength=5``). Kept for comparison.

    ``both`` sums the two. ``strength=0`` with ``base=1`` reproduces the
    unweighted loss exactly, which is the default.

    Measured outcome -- reweighting is NOT the lever here. Three 100-epoch runs
    on the same split, differing only in this weighting (lambda_gradient was
    scaled to hold the gradient/displacement balance at the unweighted run's
    0.92, so the comparison is not confounded by the smoother):

        run                        peak bin ratio   overall norm_err
        unweighted                        0.422             0.740
        magnitude base=1 strength=10      0.378             0.794
        magnitude base=0 strength=1       0.470             1.228

    39x more of the loss on the peak bought 11% on the peak, while ``base=0``
    let the 57% static bulk drift to 15.6x over-prediction and pushed the model
    past the zero baseline (norm_err > 1). The contact scheme moved nothing
    (0.15% -> 0.27%). So the under-prediction is not an optimization-incentive
    problem: the peak is largely predictable from the commanded press depth
    alone (corr 0.897, ~20% spread within a depth), so the information is
    present in the inputs and the model is failing to represent it. Look at the
    decoder/output parameterisation, not the loss.
    """
    weight = base * torch.ones_like(target_disp)
    if scheme in ("magnitude", "both"):
        if not ref:
            raise ValueError("magnitude weighting needs ref (graph units)")
        magnitude = target_disp.norm(dim=-1, keepdim=True) / ref
        weight = weight + strength * magnitude.clamp(max=clip)
    if scheme in ("contact", "both"):
        if tip_dist is None or not sigma:
            raise ValueError("contact weighting needs tip_dist and sigma")
        weight = weight + strength * torch.exp(-tip_dist[:, None] / sigma)
    return weight


class WeightedL1Loss(nn.Module):
    """L1 normalised by the total weight.

    Dividing by ``weight.sum()`` rather than by the element count keeps the
    loss on the same scale as ``nn.L1Loss`` when the weights are all ~1, so
    ``lambda_gradient`` does not have to be retuned when weighting is turned on.
    """

    def forward(self, pred, target, weight):
        return ((pred - target).abs() * weight).sum() / weight.sum()
    

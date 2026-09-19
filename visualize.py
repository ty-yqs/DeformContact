from utils.visualization import *
import os

from loaders.dataset_loader import load_dataset
from configs.config import Config
from torch_geometric.data import Batch

if __name__ == "__main__":
    config = Config("configs/everyday.json")

    out_dir = "visualizations"
    os.makedirs(out_dir, exist_ok=True)

    _, dataloader_val = load_dataset(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch_idx, (
            obj_names,
            soft_rest_graphs,
            soft_def_graphs,
            meta_data,
            rigid_graphs,
        ) in enumerate(dataloader_val):
            soft_rest_graphs_batched = Batch.from_data_list(soft_rest_graphs).to(device)
            rigid_graphs_batched = Batch.from_data_list(rigid_graphs).to(device)
            soft_def_graphs_batched = Batch.from_data_list(soft_def_graphs).to(device)

            for indx in range(config.dataloader.batch_size):
                tag = f"{obj_names[indx]}_{batch_idx:03d}_{indx:03d}"

                visualize_deformations_normals_colors(
                    soft_rest_graphs[indx],
                    soft_def_graphs_batched[indx],
                    save_path=os.path.join(out_dir, f"{tag}_normals_colors.png"),
                )
                visualize_deformation_field(
                    soft_rest_graphs[indx].pos.cpu(),
                    soft_def_graphs_batched[indx].pos.cpu(),
                    rigid_graphs[indx].pos.cpu(),
                    meta_data["force_vector"][indx],
                    save_path=os.path.join(out_dir, f"{tag}_deform_field.png"),
                )
                visualize_merged_graphs(
                    soft_rest_graphs[indx],
                    soft_def_graphs_batched[indx],
                    rigid_graphs[indx],
                    soft_def_graphs_batched[indx],
                    save_path=os.path.join(out_dir, f"{tag}_merged.png"),
                )

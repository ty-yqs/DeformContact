import argparse
import os
import time

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from utils.visualization import *
from configs.config import Config
from models.model_loader import load_model
from models.losses import GradientConsistencyLoss
from loaders.dataset_loader import load_dataset
from utils.graph_utils import *

visualize = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a DeformContact model from a local weights file."
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        required=True,
        help="Path to the local model weights file (.pth).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the config file (.json) used to train the model.",
    )
    return parser.parse_args()


def eval_runtime(config_path, weights_path):
    config = Config(config_path)
    _, dataloader_val = load_dataset(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    total_data_load_time = 0.0
    total_inference_time = 0.0

    with torch.no_grad():
        for batch_idx, (
            obj_name,
            soft_rest_graphs,
            soft_def_graphs,
            meta_data,
            rigid_graphs,
        ) in enumerate(dataloader_val):

            start_time = time.time()

            soft_rest_graphs_batched = Batch.from_data_list(soft_rest_graphs).to(device)
            rigid_graphs_batched = Batch.from_data_list(rigid_graphs).to(device)

            end_time = time.time()
            total_data_load_time += end_time - start_time

            start_time = time.time()

            predictions = model(soft_rest_graphs_batched, rigid_graphs_batched)

            end_time = time.time()
            total_inference_time += end_time - start_time

    avg_data_load_time = total_data_load_time / len(dataloader_val)
    avg_inference_time = total_inference_time / len(dataloader_val)

    print(f"Average Data Loading Time per Batch: {avg_data_load_time:.6f} seconds")
    print(f"Average Inference Time per Batch: {avg_inference_time:.6f} seconds")


def eval(config_path, weights_path):
    config = Config(config_path)
    _, dataloader_val = load_dataset(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    # Accumulate metrics per object so each object category is reported
    # separately. A single validation batch may mix several objects, so the
    # batched predictions are split back into per-sample results via `.batch`.
    obj_metrics = {}

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

            predictions = model(soft_rest_graphs_batched, rigid_graphs_batched)

            node_batch = soft_def_graphs_batched.batch
            edge_index = soft_def_graphs_batched.edge_index

            for sample_idx, obj in enumerate(obj_names):
                node_mask = node_batch == sample_idx
                pred_pos = predictions.pos[node_mask]
                gt_pos = soft_def_graphs_batched.pos[node_mask]

                loss_mse = criterion_mse(pred_pos, gt_pos).item()
                loss_mae = criterion_mae(pred_pos, gt_pos).item()

                # Per-sample gradient consistency, matching GradientConsistencyLoss
                # but restricted to the edges belonging to this sample only.
                edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                if edge_mask.sum() > 0:
                    src = edge_index[0][edge_mask]
                    dst = edge_index[1][edge_mask]
                    edge_diffs_gt = soft_def_graphs_batched.pos[dst] - soft_def_graphs_batched.pos[src]
                    edge_diffs_pred = predictions.pos[dst] - predictions.pos[src]
                    cross_shape_diffs = (edge_diffs_gt - edge_diffs_pred).norm(p=2, dim=-1)
                    loss_consistency = cross_shape_diffs.sum().item() / edge_mask.sum().item()
                else:
                    loss_consistency = 0.0

                if obj not in obj_metrics:
                    obj_metrics[obj] = {
                        "mse_sum": 0.0,
                        "mae_sum": 0.0,
                        "consistency_sum": 0.0,
                        "count": 0,
                        "errors": [],
                    }
                m = obj_metrics[obj]
                m["mse_sum"] += loss_mse
                m["mae_sum"] += loss_mae
                m["consistency_sum"] += loss_consistency
                m["count"] += 1
                m["errors"].append((pred_pos - gt_pos).cpu().numpy())

            if visualize:
                for indx in range(config.dataloader.batch_size):

                    output_folder = "./outputs/{}".format(config.dataset.obj_list[0])
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    rigid_mesh = o3d.geometry.TriangleMesh()
                    rigid_mesh.vertices = o3d.utility.Vector3dVector(
                        meta_data["rigid_mesh_vertices"][indx]
                    )
                    rigid_mesh.triangles = o3d.utility.Vector3iVector(
                        meta_data["rigid_mesh_triangles"][indx]
                    )
                    rigid_mesh_path = os.path.join(
                        "./outputs/", meta_data["sample_path"][indx] + "_rigid.obj"
                    )
                    o3d.io.write_triangle_mesh(rigid_mesh_path, rigid_mesh)

                    soft_mesh = o3d.geometry.TriangleMesh()
                    soft_mesh.triangles = o3d.utility.Vector3iVector(
                        meta_data["soft_rest_mesh_triangles"][indx]
                    )

                    # For resting
                    soft_mesh.vertices = o3d.utility.Vector3dVector(
                        soft_rest_graphs[indx].pos.cpu().numpy()
                    )
                    resting_mesh_path = os.path.join(
                        "./outputs/", meta_data["sample_path"][indx] + "_resting.obj"
                    )
                    o3d.io.write_triangle_mesh(resting_mesh_path, soft_mesh)

                    # For gt
                    soft_mesh.vertices = o3d.utility.Vector3dVector(
                        soft_def_graphs_batched[indx].pos.cpu().numpy()
                    )
                    gt_mesh_path = os.path.join(
                        "./outputs/", meta_data["sample_path"][indx] + "_gt.obj"
                    )
                    o3d.io.write_triangle_mesh(gt_mesh_path, soft_mesh)

                    # For prediction
                    soft_mesh.vertices = o3d.utility.Vector3dVector(
                        predictions[indx].pos.cpu().numpy()
                    )
                    pred_mesh_path = os.path.join(
                        "./outputs/", meta_data["sample_path"][indx] + "_pred.obj"
                    )
                    o3d.io.write_triangle_mesh(pred_mesh_path, soft_mesh)
                    # visualize_deformations_normals_colors(soft_rest_graphs[indx])
                    # visualize_deformations_normals_colors(soft_def_graphs_batched[indx])
                    # visualize_deformations_normals_colors(predictions[indx])

                    # visualize_deformations_normals_colors(soft_rest_graphs[indx], soft_def_graphs_batched[indx])
                    # visualize_deformation_field(soft_rest_graphs[indx].pos.cpu(), predictions[indx].pos.cpu(),rigid_graphs[indx].pos.cpu(), meta_data['force_vector'][indx])
                    # visualize_merged_graphs(soft_rest_graphs[indx], soft_def_graphs_batched[indx], rigid_graphs[indx],predictions[indx])

    print("=" * 60)
    for obj in sorted(obj_metrics.keys()):
        m = obj_metrics[obj]
        n = m["count"]
        avg_mse = m["mse_sum"] / n
        avg_mae = m["mae_sum"] / n
        avg_consistency = m["consistency_sum"] / n
        errors = np.concatenate(m["errors"])
        variance_of_error = np.var(errors)
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(np.abs(errors))

        print(f"Object: {obj} ({n} samples)")
        print(f"  MSE:          {avg_mse:.6f}")
        print(f"  RMSE:         {rmse:.6f}")
        print(f"  MAE:          {avg_mae:.6f}")
        print(f"  Consistency:  {avg_consistency:.6f}")
        print(f"  Max Error:    {max_error:.6f}")
        print(f"  Variance:     {variance_of_error:.6f}")
        print("-" * 60)

    # Aggregate all objects into an overall summary.
    total_count = sum(m["count"] for m in obj_metrics.values())
    if total_count > 0:
        all_errors = np.concatenate(
            [np.concatenate(m["errors"]) for m in obj_metrics.values()]
        )
        overall_mse = sum(m["mse_sum"] for m in obj_metrics.values()) / total_count
        overall_mae = sum(m["mae_sum"] for m in obj_metrics.values()) / total_count
        overall_consistency = (
            sum(m["consistency_sum"] for m in obj_metrics.values()) / total_count
        )
        overall_rmse = np.sqrt(np.mean(all_errors**2))
        overall_variance = np.var(all_errors)
        overall_max_error = np.max(np.abs(all_errors))

        print("=" * 60)
        print(f"Total ({total_count} samples, {len(obj_metrics)} objects)")
        print(f"  MSE:          {overall_mse:.6f}")
        print(f"  RMSE:         {overall_rmse:.6f}")
        print(f"  MAE:          {overall_mae:.6f}")
        print(f"  Consistency:  {overall_consistency:.6f}")
        print(f"  Max Error:    {overall_max_error:.6f}")
        print(f"  Variance:     {overall_variance:.6f}")
        print("=" * 60)


if __name__ == "__main__":

    args = parse_args()
    eval(args.config, args.weights)
    # eval_runtime(args.config, args.weights)

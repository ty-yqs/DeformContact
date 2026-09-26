"""Evaluate a DeformContact checkpoint on ``dataset/hra_dataset_large``.

Same metric definitions as ``eval.py`` but reported in micrometres, bucketed by
``regime`` and by commanded press depth, and always next to the zero-displacement
baseline. That baseline matters here: 33% of the mesh sits below the clamped
band and never moves, so the mean absolute displacement over all 5389 nodes is
only ~2 um -- an MAE printed without the baseline cannot tell a trained model
from a dead one.

Also reports inference speed. The model forward is timed under
``torch.cuda.synchronize`` with a warmup -- without the sync the asynchronous
kernel launches never reach the clock and the rate comes out 10-100x too fast,
and without the warmup the first CUDA call's context init and cudnn autotuning
get amortised into the result. ``fps`` is batched throughput, so it is an upper
bound on what a controller would see; the single-sample latency is
``ms_per_sample``.

Usage:
    python eval_hra.py -w runs/hra_large/model_weights.pth -c runs/hra_large/config.json
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from configs.config import Config
from loaders.dataset_loader import load_dataset
from models.model_loader import load_model

UM_PER_UNIT = 1e3  # graph units are length_scale x metres (0.001 m -> 1 um)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", "-w", required=True,
                        help="Path to the .pth checkpoint.")
    parser.add_argument("--config", "-c", required=True,
                        help="Config the checkpoint was trained with. Use the "
                             "config.json saved next to the weights, not the "
                             "source config: length_scale and the split must match.")
    parser.add_argument("--split", default="val", choices=["val", "train", "all"])
    parser.add_argument("--limit", type=int, default=None,
                        help="cap batches per split (smoke runs)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="batches to run before the speed clock starts; the "
                             "first CUDA call pays for context init and cudnn "
                             "autotuning, which is not steady-state inference")
    parser.add_argument("--device", default=None)
    parser.add_argument("--report", default=None, help="write a JSON report here")
    parser.add_argument("--per-case-csv", default=None,
                        help="write per-case metrics here")
    return parser.parse_args()


def unpack(batch, device):
    obj_names, soft_rest_graphs, soft_def_graphs, meta_data, rigid_graphs = batch
    return (
        obj_names,
        Batch.from_data_list(soft_rest_graphs).to(device),
        Batch.from_data_list(soft_def_graphs).to(device),
        meta_data,
        Batch.from_data_list(rigid_graphs).to(device),
    )


def evaluate(model, loader, clamp_scale, device, limit=None, warmup=0):
    """Per-sample metrics plus an inference-speed profile.

    Returns ``(records, timing)``. Lengths are in um; times are in seconds and
    measure the model forward only -- the data prep is accumulated separately
    so the per-sample metric loop below cannot pollute the throughput number.
    """
    model.eval()
    records = []

    if device.type == "cuda":
        sync = lambda: torch.cuda.synchronize(device)
    else:
        sync = lambda: None

    forward_time, prep_time, metric_time = 0.0, 0.0, 0.0
    timed_batches = timed_samples = timed_nodes = 0
    wall_start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if limit is not None and batch_idx >= limit:
                break

            t0 = time.perf_counter()
            obj_names, soft_rest, soft_def, meta_data, rigid = unpack(batch, device)
            sync()  # .to(device) is async too; it belongs in the prep time
            t1 = time.perf_counter()

            predictions = model(soft_rest, rigid)
            sync()
            t2 = time.perf_counter()

            # Batches before `warmup` still run, they just are not counted.
            timed = batch_idx >= warmup
            if timed:
                prep_time += t1 - t0
                forward_time += t2 - t1
                timed_batches += 1
                timed_samples += len(obj_names)
                timed_nodes += int(soft_def.pos.shape[0])

            pred, gt = predictions.pos, soft_def.pos
            node_batch = soft_def.batch
            edge_index = soft_def.edge_index
            # Nodes in the clamped band are held fixed by the scene and never
            # move, so they dilute every all-node average.
            active = soft_rest.pos[:, 2] > (
                torch.as_tensor(meta_data["clamp_z_m"], device=device)[node_batch]
                * clamp_scale
            )

            for sample_idx, case_id in enumerate(meta_data["case_id"]):
                node_mask = node_batch == sample_idx
                # The model predicts absolute positions; the error against the
                # ground truth is already translation invariant, but the zero
                # baseline and norm_err are only meaningful in displacement
                # space.
                rest = soft_rest.pos[node_mask]
                err = pred[node_mask] - gt[node_mask]
                target = gt[node_mask] - rest
                active_mask = active[node_mask]

                edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                if edge_mask.sum() > 0:
                    src, dst = edge_index[0][edge_mask], edge_index[1][edge_mask]
                    edge_err = (
                        (gt[dst] - gt[src]) - (pred[dst] - pred[src])
                    ).norm(p=2, dim=-1)
                    consistency = edge_err.mean().item()
                else:
                    consistency = 0.0

                abs_err = err.abs()
                abs_target = target.abs()
                record = {
                    "case_id": case_id,
                    "regime": meta_data["regime"][sample_idx],
                    "press_depth_um": meta_data["press_depth_m"][sample_idx] * 1e6,
                    "tilt_deg": meta_data["tilt_deg"][sample_idx],
                    "site": "{}_{}".format(
                        meta_data["site_x_mm"][sample_idx],
                        meta_data["site_y_mm"][sample_idx],
                    ),
                    "n_nodes": int(node_mask.sum()),
                    "mae_um": abs_err.mean().item() * UM_PER_UNIT,
                    "rmse_um": err.pow(2).mean().sqrt().item() * UM_PER_UNIT,
                    "max_um": abs_err.max().item() * UM_PER_UNIT,
                    "zero_base_um": abs_target.mean().item() * UM_PER_UNIT,
                    "norm_err": (abs_err.sum() / abs_target.sum()).item(),
                    "consistency_um": consistency * UM_PER_UNIT,
                }
                if active_mask.any():
                    a_err, a_gt = abs_err[active_mask], abs_target[active_mask]
                    record["active_mae_um"] = a_err.mean().item() * UM_PER_UNIT
                    record["active_zero_base_um"] = a_gt.mean().item() * UM_PER_UNIT
                    record["active_norm_err"] = (
                        (a_err.sum() / a_gt.sum()).item() if a_gt.sum() else float("nan")
                    )
                records.append(record)

            if timed:
                metric_time += time.perf_counter() - t2

    wall_time = time.perf_counter() - wall_start

    def rate(total, per_second):
        return total / per_second if per_second else float("nan")

    timing = {
        "device": str(device),
        "warmup_batches": warmup,
        "batches": timed_batches,
        "samples": timed_samples,
        "nodes": timed_nodes,
        "forward_time_s": forward_time,
        "prep_time_s": prep_time,
        "metric_time_s": metric_time,
        "wall_time_s": wall_time,
        "ms_per_batch": 1000 * rate(forward_time, timed_batches),
        "ms_per_sample": 1000 * rate(forward_time, timed_samples),
        "fps": rate(timed_samples, forward_time),
        "nodes_per_s": rate(timed_nodes, forward_time),
        # Forward + data prep + the per-sample metric loop, over the same timed
        # batches. This is a floor on the pipeline, not the model's rate.
        "e2e_fps": rate(timed_samples, forward_time + prep_time + metric_time),
    }
    return records, timing


def summarize(records):
    if not records:
        return None

    def mean(key):
        values = [r[key] for r in records if r.get(key) is not None]
        return float(np.mean(values)) if values else float("nan")

    # RMSE and max are recombined from per-sample values so they stay honest.
    rmse = float(np.sqrt(np.mean([r["rmse_um"] ** 2 for r in records])))
    summary = {
        "n_samples": len(records),
        "mae_um": mean("mae_um"),
        "rmse_um": rmse,
        "max_um": float(np.max([r["max_um"] for r in records])),
        "zero_base_um": mean("zero_base_um"),
        "norm_err": mean("norm_err"),
        "consistency_um": mean("consistency_um"),
        "active_mae_um": mean("active_mae_um"),
        "active_zero_base_um": mean("active_zero_base_um"),
        "active_norm_err": mean("active_norm_err"),
    }
    summary["beats_zero_baseline"] = bool(
        summary["mae_um"] < summary["zero_base_um"]
    )
    return summary


def print_block(title, summary, indent="  "):
    if summary is None:
        return
    print("{}{} ({} samples)".format(indent, title, summary["n_samples"]))
    print("{}{:<22}{:>12.4f}".format(indent, "MAE (um)", summary["mae_um"]))
    print("{}{:<22}{:>12.4f}".format(indent, "RMSE (um)", summary["rmse_um"]))
    print("{}{:<22}{:>12.4f}".format(indent, "Max error (um)", summary["max_um"]))
    print("{}{:<22}{:>12.4f}".format(indent, "zero baseline (um)", summary["zero_base_um"]))
    print("{}{:<22}{:>12.4f}".format(indent, "norm_err", summary["norm_err"]))
    print("{}{:<22}{:>12.4f}".format(indent, "consistency (um)", summary["consistency_um"]))
    print("{}{:<22}{:>12.4f}".format(indent, "MAE, active (um)", summary["active_mae_um"]))
    print("{}{:<22}{:>12.4f}".format(indent, "zero base, active (um)", summary["active_zero_base_um"]))
    print("{}{:<22}{:>12.4f}".format(indent, "norm_err, active", summary["active_norm_err"]))


def print_speed(timing, indent="  "):
    if not timing or not timing["samples"]:
        print("{}inference speed: not measured (no batch past warmup)".format(indent))
        return
    batch = timing["samples"] / timing["batches"]
    print(
        "{}{} batches / {} samples, batch {:.1f}, {} warmup".format(
            indent, timing["batches"], timing["samples"], batch,
            timing["warmup_batches"],
        )
    )
    print("{}{:<30}{:>12.3f}".format(indent, "model forward (ms/batch)", timing["ms_per_batch"]))
    print("{}{:<30}{:>12.3f}".format(indent, "model forward (ms/sample)", timing["ms_per_sample"]))
    print("{}{:<30}{:>12.2f}".format(indent, "throughput (fps)", timing["fps"]))
    print("{}{:<30}{:>12.0f}".format(indent, "throughput (nodes/s)", timing["nodes_per_s"]))
    print("{}{:<30}{:>12.2f}".format(indent, "data prep (ms/batch)", 1000 * timing["prep_time_s"] / timing["batches"]))
    print("{}{:<30}{:>12.2f}".format(indent, "pipeline fps (incl. metrics)", timing["e2e_fps"]))


def group_by(records, key):
    groups = {}
    for record in records:
        groups.setdefault(record[key], []).append(record)
    return groups


def report(split_name, records, timing=None):
    print("=" * 64)
    summary = summarize(records)
    if summary is None:
        print("[{}] no samples".format(split_name))
        return None
    print("[{}]".format(split_name))
    print_block("overall", summary)
    print("-" * 64)
    print_speed(timing)

    print("-" * 64)
    # README: snap_through is bistable and path dependent, so an average over
    # mixed regimes is not a meaningful number.
    for regime, group in sorted(group_by(records, "regime").items()):
        print_block("regime = {}".format(regime), summarize(group))

    print("-" * 64)
    buckets = group_by(records, "press_depth_um")
    for depth in sorted(buckets):
        print_block("press depth = {:.0f} um".format(depth), summarize(buckets[depth]))

    print("-" * 64)
    worst = sorted(records, key=lambda r: r["mae_um"], reverse=True)[:5]
    print("  worst 5 cases by MAE")
    for record in worst:
        print(
            "    {:<40} {:>8.3f} um  (target {:>7.3f})  {}".format(
                record["case_id"], record["mae_um"], record["zero_base_um"],
                record["regime"],
            )
        )
    print("=" * 64)
    return summary


def main():
    args = parse_args()
    config = Config(args.config)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataloader_train, dataloader_val = load_dataset(config)
    model = load_model(config).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    clamp_scale = float(config.dataset.length_scale or 1.0)

    splits = []
    if args.split in ("val", "all"):
        splits.append(("val", dataloader_val))
    if args.split in ("train", "all"):
        splits.append(("train", dataloader_train))

    all_summaries, all_records, all_speeds = {}, {}, {}
    for name, loader in splits:
        records, timing = evaluate(
            model, loader, clamp_scale, device, limit=args.limit, warmup=args.warmup
        )
        all_records[name] = records
        all_speeds[name] = timing
        all_summaries[name] = report(name, records, timing)

    if args.per_case_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.per_case_csv)), exist_ok=True)
        fieldnames = sorted({k for records in all_records.values() for r in records for k in r})
        with open(args.per_case_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["split"] + fieldnames)
            writer.writeheader()
            for name, records in all_records.items():
                for record in records:
                    writer.writerow(dict(record, split=name))
        print("wrote per-case metrics to {}".format(args.per_case_csv))

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(
                {
                    "weights": args.weights,
                    "config": args.config,
                    "summaries": all_summaries,
                    "speed": all_speeds,
                },
                handle,
                indent=4,
            )
        print("wrote report to {}".format(args.report))


if __name__ == "__main__":
    main()

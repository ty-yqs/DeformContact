"""Train DeformContact on ``dataset/hra_dataset_large``.

Same loss and loop as ``train.py``, but driven by a config path passed on the
command line, checkpoints written to a real output directory (no wandb
required), and per-epoch diagnostics reported in micrometres.

Usage:
    python train_hra.py --config configs/hra_large.json --output-dir runs/hra_large
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

from configs.config import Config
from loaders.dataset_loader import load_dataset
from models.losses import GradientConsistencyLoss, WeightedL1Loss, displacement_weight
from models.model_loader import load_model

REQUIRED_CONFIG = [
    "dataset.root_dir",
    "dataset.split_ratio",
    "dataloader.batch_size",
    "training.n_epochs",
    "training.learning_rate",
    "network.input_dims",
]

# A node at this displacement gets `1 + strength` times the weight of a static
# node. 30 um is roughly where the baseline model's predictions start to fall
# behind the ground truth (it reproduces only 39% above that).
DEFAULT_WEIGHT_REF_UM = 30.0
DEFAULT_CONTACT_SIGMA_MM = 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", "-c", default="configs/hra_large.json")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="default: config.training.output_dir")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-gradient", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--loss-weighting", default=None,
                        choices=["none", "magnitude", "contact", "both"],
                        help="weight the displacement loss towards the contact "
                             "peak. 'none' (the default) is plain L1 and "
                             "reproduces the unweighted run exactly.")
    parser.add_argument("--loss-weight-strength", type=float, default=None,
                        help="weight of a node at the reference displacement "
                             "(or at zero distance from the tip)")
    parser.add_argument("--loss-weight-base", type=float, default=None,
                        help="weight of a static node; 1.0 keeps the bulk "
                             "constrained, 0.0 hands the loss to the peaks")
    parser.add_argument("--device", default=None, help="cuda / cpu / cuda:1 ...")
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="cap training batches per epoch (smoke runs)")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="cap validation batches per epoch (smoke runs)")
    parser.add_argument("--wandb", action="store_true",
                        help="also log to Weights & Biases (off by default)")
    return parser.parse_args()


def build_config(args):
    updates = {}
    if args.epochs is not None:
        updates.setdefault("training", {})["n_epochs"] = args.epochs
    if args.lr is not None:
        updates.setdefault("training", {})["learning_rate"] = args.lr
    if args.lambda_gradient is not None:
        updates.setdefault("training", {})["lambda_gradient"] = args.lambda_gradient
    if args.batch_size is not None:
        updates.setdefault("dataloader", {})["batch_size"] = args.batch_size
    if args.loss_weighting is not None:
        updates.setdefault("training", {})["loss_weighting"] = args.loss_weighting
    if args.loss_weight_strength is not None:
        updates.setdefault("training", {})["loss_weight_strength"] = args.loss_weight_strength
    if args.loss_weight_base is not None:
        updates.setdefault("training", {})["loss_weight_base"] = args.loss_weight_base

    config = Config(args.config, updates=updates or None)
    config.validate(REQUIRED_CONFIG)
    return config


def unpack(batch, device):
    """The shared 5-tuple -> batched PyG graphs on ``device``."""
    obj_names, soft_rest_graphs, soft_def_graphs, meta_data, rigid_graphs = batch
    return (
        obj_names,
        Batch.from_data_list(soft_rest_graphs).to(device),
        Batch.from_data_list(soft_def_graphs).to(device),
        meta_data,
        Batch.from_data_list(rigid_graphs).to(device),
    )


def run_epoch(model, loader, device, criterion_mse, criterion_grad, lambda_gradient,
              optimizer=None, limit_batches=None, collect_stats=False, weighting=None):
    """One pass. Trains when ``optimizer`` is given, otherwise evaluates.

    Returns ``(loss, stats)`` where the loss is weighted by sample count (a
    plain mean over batches would over-weight the last, partial batch) and
    ``stats`` holds micrometre diagnostics for this dataset.

    ``weighting`` is the kwargs for ``displacement_weight``, or None for the
    unweighted loss.
    """
    training = optimizer is not None
    model.train(training)

    total_loss, total_samples = 0.0, 0
    sum_abs_err, sum_abs_gt, sum_nodes = 0.0, 0.0, 0

    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        obj_names, soft_rest, soft_def, meta_data, rigid = unpack(batch, device)

        with torch.set_grad_enabled(training):
            predictions = model(soft_rest, rigid)
            predictions.pos = predictions.pos - soft_rest.pos
            soft_def.pos = soft_def.pos - soft_rest.pos

            if weighting is None:
                loss_mse = criterion_mse(predictions.pos, soft_def.pos)
            else:
                tip = meta_data["needle_tip_mm"].to(device)[soft_rest.batch]
                weight = displacement_weight(
                    soft_def.pos,
                    tip_dist=(soft_rest.pos - tip).norm(dim=-1),
                    **weighting,
                )
                loss_mse = criterion_mse(predictions.pos, soft_def.pos, weight)
            loss_consistency = criterion_grad(predictions, soft_def)
            loss = loss_mse + lambda_gradient * loss_consistency

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_samples = len(obj_names)
        total_loss += loss.item() * batch_samples
        total_samples += batch_samples

        if collect_stats:
            with torch.no_grad():
                err = (predictions.pos - soft_def.pos).abs()
                sum_abs_err += err.sum().item()
                sum_abs_gt += soft_def.pos.abs().sum().item()
                sum_nodes += err.numel()

    if total_samples == 0:
        return float("nan"), {}

    stats = {}
    if collect_stats and sum_nodes:
        # Working units are config.dataset.length_scale x metres; report um.
        stats["mae_um"] = sum_abs_err / sum_nodes * 1e3
        stats["zero_base_um"] = sum_abs_gt / sum_nodes * 1e3
        stats["norm_err"] = sum_abs_err / sum_abs_gt if sum_abs_gt else float("nan")

    return total_loss / total_samples, stats


def main():
    args = parse_args()
    config = build_config(args)

    seed = args.seed if args.seed is not None else (config.training.seed or 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = args.output_dir or config.training.output_dir or "runs/hra"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataloader_train, dataloader_val = load_dataset(config)
    print(
        "dataset={} train={} val={} | device={} | output={}".format(
            config.dataset.name,
            len(dataloader_train.dataset),
            len(dataloader_val.dataset),
            device,
            output_dir,
        )
    )

    model = load_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    scheme = (config.training.loss_weighting or "none").lower()
    weighting = None
    if scheme == "none":
        criterion_mse = nn.L1Loss()
    else:
        strength = config.training.loss_weight_strength
        ref_um = config.training.loss_weight_ref_um
        sigma_mm = config.training.loss_contact_sigma_mm
        base = config.training.loss_weight_base
        strength = 1.0 if strength is None else float(strength)
        ref_um = DEFAULT_WEIGHT_REF_UM if ref_um is None else float(ref_um)
        sigma_mm = DEFAULT_CONTACT_SIGMA_MM if sigma_mm is None else float(sigma_mm)
        base = 1.0 if base is None else float(base)
        weighting = {
            "scheme": scheme,
            "strength": strength,
            # um -> graph units: the working unit is length_scale x metres,
            # i.e. mm, and 1 mm = 1e3 um.
            "ref": ref_um * 1e-3,
            "sigma": sigma_mm,
            "base": base,
        }
        criterion_mse = WeightedL1Loss()
        print(
            "loss weighting: scheme={} base={} strength={} ref={} um sigma={} mm".format(
                scheme, base, strength, ref_um, sigma_mm
            )
        )

    criterion_grad = GradientConsistencyLoss()
    lambda_gradient = config.training.lambda_gradient

    weights_path = os.path.join(output_dir, "model_weights.pth")
    config_path = os.path.join(output_dir, "config.json")

    use_wandb = args.wandb
    if use_wandb:
        import wandb

        wandb.init(
            project="DeformContact",
            name="{}-{}".format(config.dataset.name, random.randint(1000, 9999)),
        )

    best_val = float("inf")
    for epoch in range(config.training.n_epochs):
        train_loss, train_stats = run_epoch(
            model, dataloader_train, device, criterion_mse, criterion_grad,
            lambda_gradient, optimizer=optimizer, limit_batches=args.limit_batches,
            collect_stats=True, weighting=weighting,
        )
        val_loss, val_stats = run_epoch(
            model, dataloader_val, device, criterion_mse, criterion_grad,
            lambda_gradient, limit_batches=args.max_val_batches, collect_stats=True,
            weighting=weighting,
        )

        print(
            "Epoch {}/{} - train {:.6f} - val {:.6f} | "
            "val MAE {:.3f} um (zero baseline {:.3f} um, norm_err {:.3f})".format(
                epoch + 1, config.training.n_epochs, train_loss, val_loss,
                val_stats.get("mae_um", float("nan")),
                val_stats.get("zero_base_um", float("nan")),
                val_stats.get("norm_err", float("nan")),
            )
        )

        if use_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "train_mae_um": train_stats.get("mae_um"),
                    "val_mae_um": val_stats.get("mae_um"),
                    "val_norm_err": val_stats.get("norm_err"),
                }
            )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), weights_path)
            # Saved next to the weights so eval_hra.py gets the exact config --
            # including length_scale and the split -- the checkpoint was fit with.
            config.save(config_path)
            print("  saved {} (val {:.6f})".format(weights_path, val_loss))

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

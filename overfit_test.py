"""Can the network represent this target at all?

Trains on a handful of samples the model sees on every step, for long enough
that the loss should reach ~0. If it cannot fit those few samples, the failure
is representational -- the architecture cannot express the target -- and no
amount of loss reweighting, schedule tuning or extra data will fix it. If it
can, the failure is optimization on the full set and the loss is the place to
look.

This is the question ``hra_dataset_fundus`` needs answered. On the full split
the model reaches norm_err 6.5 on the *training* sites -- it never fits data it
has seen for 63 epochs -- and its output varies by only 2% across a 36 mm mesh,
i.e. it emits a near-constant field regardless of where the needle is. Both the
"objective prefers zero" explanation and the "architecture cannot localise"
explanation predict a stuck model, and they call for opposite fixes.

By default the samples with the largest peaks are chosen: the target is
0.024% of nodes moving > 30 um on a mesh whose median edge is 0.63 mm, so if
the model cannot fit even the most prominent dents it certainly cannot fit the
faint ones.

Usage:
    python overfit_test.py --config configs/hra_fundus.json --samples 4
    python overfit_test.py --config configs/hra_fundus.json --samples 4 --lambda-gradient 0
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from configs.config import Config
from loaders.collate import collate_fn
from loaders.dataset_loader import load_dataset
from models.losses import GradientConsistencyLoss
from models.model_loader import load_model
from train_hra import unpack

UM_PER_UNIT = 1e3


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", "-c", default="configs/hra_fundus.json")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--samples", "-n", type=int, default=4,
                        help="how many samples to fit")
    parser.add_argument("--pick", default="peak", choices=["peak", "first"],
                        help="'peak' takes the largest dents, 'first' the first n")
    parser.add_argument("--focus", default="all", choices=["all", "moving"],
                        help="'moving' drops the static bulk from the "
                             "displacement loss, so only nodes whose target "
                             "exceeds --focus-um contribute. On this data 5388 "
                             "of 5389 nodes are static, and their pull towards "
                             "zero is applied by 5388 votes to the one node "
                             "that moves. This separates 'cannot represent the "
                             "dent' from 'the bulk outvotes the dent'.")
    parser.add_argument("--focus-um", type=float, default=1.0,
                        help="target magnitude above which a node counts as moving")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=None,
                        help="default: config.training.learning_rate")
    parser.add_argument("--lambda-gradient", type=float, default=None,
                        help="default: config.training.lambda_gradient")
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def pick_samples(dataset, n, how):
    """Indices of the samples to fit, plus their peak displacements."""
    if how == "first":
        picked = list(range(min(n, len(dataset))))
    else:
        peaks = []
        for i in range(len(dataset)):
            _, rest, defo, _, _ = dataset[i]
            peaks.append(((defo.pos - rest.pos).norm(dim=-1).max().item(), i))
        peaks.sort(reverse=True)
        picked = [i for _, i in peaks[:n]]
    return picked


def displacement_term(pred, target, focus, focus_um):
    """L1 on the displacement, optionally restricted to the moving nodes.

    The restricted form keeps ``nn.L1Loss``'s scale (a mean over the elements
    that count) so ``--lambda-gradient`` means the same thing either way.
    """
    if focus == "all":
        return nn.functional.l1_loss(pred, target)
    moving = (target.norm(dim=-1) > focus_um * 1e-3).float()[:, None]
    if moving.sum() == 0:
        return nn.functional.l1_loss(pred, target)
    return ((pred - target).abs() * moving).sum() / moving.sum()


def probe(model, loader, lambda_gradient, device, focus="all", focus_um=1.0):
    """Loss and per-sample peak reproduction on the fitted samples."""
    crit_grad = GradientConsistencyLoss()
    model.eval()
    total, rows = 0.0, []
    with torch.no_grad():
        for batch in loader:
            _, rest, defo, meta, rigid = unpack(batch, device)
            pred = model(rest, rigid).pos - rest.pos
            gt = defo.pos - rest.pos
            total += (displacement_term(pred, gt, focus, focus_um)
                      + lambda_gradient * crit_grad(_as_graph(pred, rest),
                                                    _as_graph(gt, rest))).item()
            nb = rest.batch
            for s, case_id in enumerate(meta["case_id"]):
                m = nb == s
                pm, gm = pred[m].norm(dim=-1), gt[m].norm(dim=-1)
                rows.append({
                    "case_id": case_id,
                    "peak_gt_um": gm.max().item() * UM_PER_UNIT,
                    "peak_pred_um": pm[gm.argmax()].item() * UM_PER_UNIT,
                    "max_pred_um": pm.max().item() * UM_PER_UNIT,
                    # 0 means the same vector at every node: the model is not
                    # using position at all, which is the failure seen on the
                    # full split.
                    "flatness": (pm.std() / (pm.mean() + 1e-12)).item(),
                    "gt_flatness": (gm.std() / (gm.mean() + 1e-12)).item(),
                    "mae_um": (pred[m] - gt[m]).abs().mean().item() * UM_PER_UNIT,
                    "zero_base_um": gt[m].abs().mean().item() * UM_PER_UNIT,
                })
    return total / max(len(loader), 1), rows


def _as_graph(displacement, like):
    """GradientConsistencyLoss takes graphs, so rewrap a displacement tensor."""
    graph = like.clone()
    graph.pos = displacement
    return graph


def main():
    args = parse_args()
    config = Config(args.config)
    lr = args.lr if args.lr is not None else config.training.learning_rate
    lam = (args.lambda_gradient if args.lambda_gradient is not None
           else config.training.lambda_gradient)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_loader, val_loader = load_dataset(config)
    source = train_loader.dataset if args.split == "train" else val_loader.dataset
    indices = pick_samples(source, args.samples, args.pick)
    loader = DataLoader(
        Subset(source, indices), batch_size=len(indices), shuffle=False,
        collate_fn=collate_fn,
    )

    print("config={} split={} samples={} device={}".format(
        args.config, args.split, len(indices), device))
    print("epochs={} lr={} lambda_gradient={} focus={}".format(
        args.epochs, lr, lam, args.focus
        + (" (|target| > {} um)".format(args.focus_um) if args.focus == "moving" else "")))
    print("fitting: {}".format(
        ", ".join(source[i][3]["case_id"] for i in indices)))
    print()

    model = load_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    crit_grad = GradientConsistencyLoss()

    header = "{:>7}{:>12}  {}".format("epoch", "loss", "peak pred/gt per sample")
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            _, rest, defo, meta, rigid = unpack(batch, device)
            pred = model(rest, rigid)
            pred.pos = pred.pos - rest.pos
            defo.pos = defo.pos - rest.pos
            loss = displacement_term(pred.pos, defo.pos, args.focus,
                                     args.focus_um) + lam * crit_grad(pred, defo)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()

        if epoch % args.report_every == 0 or epoch == args.epochs:
            _, rows = probe(model, loader, lam, device, args.focus, args.focus_um)
            ratios = "  ".join(
                "{:.3f}".format(r["peak_pred_um"] / r["peak_gt_um"]) for r in rows
            )
            print("{:>7}{:>12.2e}  {}".format(epoch, epoch_loss, ratios))

    loss, rows = probe(model, loader, lam, device, args.focus, args.focus_um)
    print()
    print("final loss {:.3e}".format(loss))
    print("{:<34}{:>10}{:>11}{:>8}{:>9}{:>8}{:>11}".format(
        "case", "peak gt", "peak pred", "ratio", "flatness", "sharp", "norm_err"))
    for r in rows:
        print("{:<34}{:>10.2f}{:>11.3f}{:>8.3f}{:>9.3f}{:>8.3f}{:>11.2f}".format(
            r["case_id"], r["peak_gt_um"], r["peak_pred_um"],
            r["peak_pred_um"] / r["peak_gt_um"], r["flatness"],
            r["flatness"] / r["gt_flatness"] if r["gt_flatness"] else float("nan"),
            r["mae_um"] / r["zero_base_um"] if r["zero_base_um"] else float("nan"),
        ))
    print()
    print("'sharp' is the prediction's spatial contrast as a fraction of the "
          "target's; a real fit needs it near 1.")
    print()
    worst = min(r["peak_pred_um"] / r["peak_gt_um"] for r in rows)
    worst_ne = max(
        r["mae_um"] / r["zero_base_um"] if r["zero_base_um"] else float("inf")
        for r in rows
    )
    sharp = min(
        r["flatness"] / r["gt_flatness"] if r["gt_flatness"] else 0.0 for r in rows
    )
    # The peak ratio alone cannot tell a fit from an inflated field: scaling a
    # near-constant output up until it reaches the peak sets the ratio to 1.0
    # while the other 5380-odd nodes stay wrong. So a fit additionally has to
    # beat the zero baseline (norm_err < 1) and carry the target's contrast.
    if worst > 0.8 and worst_ne < 1.0 and sharp > 0.5:
        print("FITS: peaks reproduced, field beats the zero baseline, and the "
              "output carries the target's spatial contrast. The architecture "
              "can express this target; the full-split stall is optimization.")
    elif worst > 0.8:
        print("INFLATED, NOT FITTED: the peak is reached, but norm_err is "
              "{:.1f} (a real fit needs < 1) and the field carries only {:.0%} "
              "of the target's spatial contrast. Scaling a near-constant field "
              "until its maximum touches the peak is not a fit -- the model "
              "still does not represent the target." .format(worst_ne, sharp))
    elif worst > 0.2:
        print("PARTIAL: peaks are reproduced but not fitted. Look at the loss "
              "landscape and the readout, not raw capacity.")
    elif args.focus == "all":
        print("CANNOT FIT: the model fails on samples it has seen {} times. "
              "Rerun with --focus moving before concluding: with 5388 of 5389 "
              "nodes static, the bulk's pull towards zero outvotes the dent by "
              "count, and that alone can produce this." .format(args.epochs))
    else:
        print("CANNOT FIT: the model fails on samples it has seen {} times even "
              "with the static bulk removed from the loss. The failure is "
              "representational -- no reweighting can help."
              .format(args.epochs))


if __name__ == "__main__":
    main()

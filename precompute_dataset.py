"""Precompute and cache the everyday_deform dataset graphs to disk.

Running this once (single process, no DataLoader workers) builds every sample's
graphs and saves them under ``config.dataset.cache_dir``. After that, the
dataset's ``__getitem__`` loads the cached tensors instead of reading PLY files
and building meshes with Open3D, so training can safely use ``num_workers > 0``
without the intermittent ``received 0 items of ancdata`` crash (fork + Open3D
is what kills the workers).

Usage:
    python precompute_dataset.py [path/to/config.json]
"""
import os
import sys
import time

from configs.config import Config
from loaders.dataset_loader import build_dataset


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/everyday.json"
    config = Config(config_path)

    if not config.dataset.cache_dir:
        print("config.dataset.cache_dir is empty; set it to enable caching.")
        sys.exit(1)

    for split in ("train", "val"):
        ds = build_dataset(config, split)
        n = len(ds)
        print(f"Precomputing {split}: {n} samples -> {ds.cache_dir}")
        t0 = time.time()
        for i in range(n):
            ds[i]  # computes on first access, then caches
            if (i + 1) % 500 == 0 or (i + 1) == n:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{n} ({rate:.1f} samples/s)")
        print(f"  done in {time.time() - t0:.1f}s")

    print("Precompute finished.")


if __name__ == "__main__":
    main()

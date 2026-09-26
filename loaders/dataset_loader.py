from loaders.everyday_deform import EverydayDeformDataset
from loaders.hra_retina import HraRetinaDataset
from torch.utils.data import DataLoader
from loaders.collate import collate_fn


def build_dataset(config, split):
    """Build one split of whichever dataset ``config.dataset.name`` selects."""
    if config.dataset.name == "everyday":
        return EverydayDeformDataset(
            obj_list=config.dataset.obj_list,
            root_dir=config.dataset.root_dir,
            n_points=config.dataset.n_points,
            graph_method=config.dataset.graph_method,
            neigbor_k=config.dataset.neigbor_k,
            neigbor_radius=config.dataset.neigbor_radius,
            sphere_radius=config.dataset.sphere_radius,
            force_max=config.dataset.force_max,
            split=split,
            cache_dir=config.dataset.cache_dir,
        )

    if config.dataset.name == "hra":
        return HraRetinaDataset(
            root_dir=config.dataset.root_dir,
            n_points=config.dataset.n_points,
            n_points_mode=config.dataset.n_points_mode,
            graph_method=config.dataset.graph_method,
            neigbor_k=config.dataset.neigbor_k,
            neigbor_radius=config.dataset.neigbor_radius,
            sphere_radius=config.dataset.sphere_radius,
            rigid_n_points=config.dataset.rigid_n_points,
            force_max=config.dataset.force_max,
            length_scale=config.dataset.length_scale,
            press_depth_max=config.dataset.press_depth_max,
            condition=config.dataset.condition,
            split=split,
            split_ratio=config.dataset.split_ratio,
            split_seed=config.dataset.split_seed,
            split_mode=config.dataset.split_mode,
            manifest_name=config.dataset.manifest_name,
            filter_status=config.dataset.filter_status,
            filter_hold_settled=config.dataset.filter_hold_settled,
            filter_min_force=config.dataset.filter_min_force,
            exclude_regression=config.dataset.exclude_regression,
            preload=config.dataset.preload,
            cache_dir=config.dataset.cache_dir,
        )

    raise ValueError(f"Unknown dataset name: {config.dataset.name}")


def load_dataset(config):
    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")

    num_workers = int(config.dataloader.num_workers or 0)

    dataloader_val = DataLoader(
        val_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    dataloader_train = DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return dataloader_train, dataloader_val

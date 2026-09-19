from loaders.everyday_deform import EverydayDeformDataset
from torch.utils.data import DataLoader
from loaders.collate import collate_fn


def load_dataset(config):


    if config.dataset.name == "everyday":
        train_dataset = EverydayDeformDataset(
            obj_list=config.dataset.obj_list,
            root_dir=config.dataset.root_dir,
            n_points=config.dataset.n_points,
            graph_method=config.dataset.graph_method,
            neigbor_k=config.dataset.neigbor_k,
            neigbor_radius=config.dataset.neigbor_radius,
            sphere_radius =config.dataset.sphere_radius,
            force_max=config.dataset.force_max,
            split="train",
            cache_dir=config.dataset.cache_dir,
        )

        val_dataset = EverydayDeformDataset(
            obj_list=config.dataset.obj_list,
            root_dir=config.dataset.root_dir,
            n_points=config.dataset.n_points,
            graph_method=config.dataset.graph_method,
            neigbor_k=config.dataset.neigbor_k,
            neigbor_radius=config.dataset.neigbor_radius,
            sphere_radius =config.dataset.sphere_radius,
            force_max=config.dataset.force_max,
            split="val",
            cache_dir=config.dataset.cache_dir,
        )
    else:
        raise ValueError(f"Unknown dataset name: {config.dataset.name}")

    dataloader_val = DataLoader(
        val_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    dataloader_train = DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return dataloader_train, dataloader_val

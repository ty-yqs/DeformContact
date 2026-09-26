import json

class Config:
    def __init__(self,path = 'configs/everyday.json', updates=None):
        self.dataset = DatasetConfig() 
        self.visualization = VisualizationConfig() 
        self.dataloader = DataLoaderConfig() 
        self.training = TrainingConfig()
        self.network = NetworkConfig()  

        with open(path, 'r') as f:
            defaults = json.load(f)
        if defaults:
            self._load_defaults(defaults)
        if updates:
            self._apply_updates(updates)

    def _load_defaults(self, defaults):
        for key, value in defaults.items():
            if hasattr(self, key):
                for sub_key, sub_value in value.items():
                    if hasattr(getattr(self, key), sub_key):
                        setattr(getattr(self, key), sub_key, sub_value)

    def _apply_updates(self, updates):
        for key, value in updates.items():
            if hasattr(self, key):
                for sub_key, sub_value in value.items():
                    if hasattr(getattr(self, key), sub_key):
                        setattr(getattr(self, key), sub_key, sub_value)
    def save(self, file_path):
        config_dict = {
            "dataset": self.dataset.__dict__,
            "visualization": self.visualization.__dict__,
            "dataloader": self.dataloader.__dict__,
            "training": self.training.__dict__,
            "network": self.network.__dict__
        }

        with open(file_path, 'w') as config_file:
            json.dump(config_dict, config_file, indent=4)

    def validate(self, required):
        """Raise if any ``"section.attr"`` in ``required`` is still None.

        ``_load_defaults`` silently drops JSON keys that have no matching
        attribute, so a typo in a config file shows up later as a confusing
        TypeError (e.g. ``range(None)``). Call this right after loading.
        """
        missing = []
        for path in required:
            section, _, attr = path.partition(".")
            value = getattr(getattr(self, section, None), attr, None)
            if value is None:
                missing.append(path)
        if missing:
            raise ValueError(
                "Config is missing required value(s): {}. Either the config "
                "file omits them or they are not declared on the sub-config "
                "class (unknown JSON keys are dropped silently).".format(
                    ", ".join(sorted(missing))
                )
            )

class DatasetConfig:
    def __init__(self):
        self.name = None
        self.root_dir = None
        self.obj_list = None
        self.n_points = None
        self.neigbor_radius = None
        self.sphere_radius = None
        self.neigbor_k = None
        self.force_max = None
        self.graph_method = None
        self.cache_dir = None
        # HRA retina dataset (configs/hra_large.json). Everything below is
        # ignored by the everyday loader.
        self.n_points_mode = None
        self.length_scale = None
        self.rigid_n_points = None
        self.press_depth_max = None
        self.condition = None
        self.split_ratio = None
        self.split_seed = None
        self.split_mode = None
        self.manifest_name = None
        self.filter_status = None
        self.filter_hold_settled = None
        self.filter_min_force = None
        self.exclude_regression = None
        self.preload = None

class VisualizationConfig:
    def __init__(self):
        self.rigid_radius_contact = None
        self.rigid_radius_deform = None
        self.colors = {
            "contact_rigid": [0, 0, 1],
            "deform_rigid": [1, 1, 0],
            "soft_rest_pcd": [1, 0, 0],
            "soft_def_pcd": [0, 1, 0],
            "lineset": [0.5, 0.5, 0.5],
            "vector": [0, 0, 0]
        }

class DataLoaderConfig:
    def __init__(self):
        self.batch_size = None
        self.shuffle = None
        self.num_workers = None

class TrainingConfig:
    def __init__(self):
        self.n_epochs = None
        self.learning_rate = None
        self.model_save_path =None
        self.lambda_gradient = None
        self.lambda_deformable = None
        self.seed = None
        self.output_dir = None
        self.log_every = None

class NetworkConfig:
    def __init__(self):
        self.input_dims = []
        self.hidden_dim = None
        self.output_dim = None
        self.encoder_layers = None
        self.decoder_layers = None
        self.dropout_rate = None
        self.knn_k = None
        self.backbone = None
        self.use_mha = None
        self.num_mha_heads=None
        self.mode = None
        self.mha_masked = False

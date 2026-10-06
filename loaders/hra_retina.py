"""Dataset for the HRA retina needle-indentation sets.

Two generations live side by side and they are not interchangeable:

``hra_dataset_large``
    The scene clamped the *fundus* (rigid below ``clamp_z_m``), so every one of
    its 1409 samples landed in the clamped band -- they pressed the ora serrata
    rim, not the fundus -- and 51% of them were ``snap_through``.
``hra_dataset_fundus``
    The clamp was inverted (rigid above ``clamp_z_m``), putting all 2800 sites
    in the deformable cup floor, and all of them are ``elastic``. This is the
    generation the README describes as correct.

Both share one rest mesh (``rest.npy``, 5389 nodes, byte-identical between the
two) with a per-case displacement field (``retina_displacement.npy``). Unlike
``everyday_deform`` there is no ``.ply`` per sample, no per-object
sub-directory, and **no triangle connectivity** -- the ``.msh`` the README
points at is not part of the dataset. Graphs are therefore built from bare
points with ``construct_graph_kdtree`` (scipy cKDTree; the ``pyg_nn.knn_graph``
path needs ``torch-cluster``, which is not installed).

The class returns the same 5-tuple as ``EverydayDeformDataset`` so that
``loaders.collate.collate_fn``, the loss in ``train.py`` and the metric code in
``eval.py`` all keep working unchanged.
"""
import csv
import json
import os

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch.utils.data import Dataset

from loaders.common import _feature_rigid
from utils.pointcloud_utils import construct_graph_kdtree
from utils.pos_encoding import to_log_freq

# Tilt and azimuth both orient the needle, and on hra_dataset_large the four
# tilt-0 azimuth cases of a site were bit-identical replays, so grouping the
# split by site kept them together. On hra_dataset_fundus that premise is gone:
# only 8% of the tilt-0 azimuth pairs are replays (median difference 11.5 um,
# max 81.7 um), because the bevel roll follows the azimuth. Site grouping is
# still what the split must do -- adjacent sites are 1.35 mm apart while the
# dent is only ~0.75 mm wide, so a per-sample split would put neighbouring
# views of the same contact geometry on both sides.
REGIME_ELASTIC = "elastic"


def fibonacci_sphere(n):
    """Deterministic, near-uniform ``(n, 3)`` point set on the unit sphere."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
        axis=1,
    )


def read_manifest(path, status="ok", hold_settled=True, min_force=0.0,
                  exclude_regression=True, min_disp_ratio=None):
    """Filtered manifest rows.

    The README's recommended gates: keep ``status == "ok"``, drop cases whose
    hold-phase force is still ringing (``hold_settled != 1``), drop the
    degenerate zero-contact cases, and drop the regression case. Rejected rows
    have empty strings in every numeric column, so they must be filtered before
    any ``float()``.

    ``min_disp_ratio`` additionally drops the "grab and release" cases: the
    driver's own gate compares ``max_disp_m``, which is the peak over the whole
    1201-frame trajectory, so a case that reaches the commanded depth during
    the press and then lets go before the sample frame still passes it. Those
    give one pose two different depths, i.e. an ambiguous regression target.
    The ratio is sample-frame ``max |retina_displacement| / press_depth``, which
    is not a manifest column, so this reads one ``.npy`` per surviving candidate
    (~0.6 s over the 2800 fundus cases). ``None`` skips it entirely.
    """
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    kept = []
    for row in rows:
        if status is not None and row["status"] != status:
            continue
        if exclude_regression and row["case_id"].startswith("regression"):
            continue
        if hold_settled and row["hold_settled"] != "1":
            continue
        if min_force is not None and float(row["force_n"]) <= min_force:
            continue
        kept.append(row)

    if min_disp_ratio is not None:
        root = os.path.dirname(os.path.abspath(path))
        gated = []
        for row in kept:
            disp = np.load(
                os.path.join(root, row["case_id"], "retina_displacement.npy")
            )
            peak = float(np.linalg.norm(disp, axis=1).max())
            if peak / (float(row["press_depth_um"]) * 1e-6) >= min_disp_ratio:
                gated.append(row)
        kept = gated
    return kept


def _site_groups(rows, site_keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in site_keys), []).append(row)
    return groups


def _pick_val_sites(groups, sequence, target_val):
    """Take the prefix of ``sequence`` closest to ``target_val``, then swap.

    Whole sites hold 24..36 cases each, so the prefix alone can only land within
    ~1 point of the ratio. Trade one val group for one unselected group at a
    time while that strictly improves the count; the error decreases every pass,
    so this terminates.
    """
    running, best_cut, best_err = 0, 0, float("inf")
    for cut, key in enumerate(sequence, start=1):
        running += len(groups[key])
        error = abs(running - target_val)
        if error < best_err:
            best_cut, best_err = cut, error
    val_set = set(sequence[:best_cut])

    def val_count():
        return sum(len(groups[k]) for k in val_set)

    while True:
        current = val_count()
        best_swap, swap_err = None, abs(current - target_val)
        candidates = [k for k in sequence if k not in val_set]
        # Walk the selected keys in ``sequence`` order, not set order: a set of
        # string tuples iterates in an order that depends on PYTHONHASHSEED, so
        # the first strictly-improving swap found would differ between
        # processes and hand two runs different (both legal) val splits.
        for out_key in [k for k in sequence if k in val_set]:
            for in_key in candidates:
                error = abs(
                    current - len(groups[out_key]) + len(groups[in_key]) - target_val
                )
                if error < swap_err - 1e-9:
                    swap_err, best_swap = error, (out_key, in_key)
        if best_swap is None:
            break
        val_set.discard(best_swap[0])
        val_set.add(best_swap[1])
    return val_set


def _split_from_val_set(groups, val_set):
    train = sorted([r for k, v in groups.items() if k not in val_set for r in v],
                   key=lambda r: r["case_id"])
    val = sorted([r for k in val_set for r in groups[k]], key=lambda r: r["case_id"])
    return train, val


def stratified_site_split(rows, split_ratio=0.8, seed=0,
                          site_keys=("site_x_mm", "site_y_mm"),
                          regime_key="regime"):
    """Split rows 4:1, holding out whole needle sites, stratified by regime.

    Every site carries many (tilt, azimuth, press depth) combinations, so a
    per-sample split would put the same site in train and val at many different
    angles. Grouping by site fixes that, but on ``hra_dataset_large`` the
    per-site elastic fraction ranged from 0.0 to 1.0, so an unstratified site
    shuffle swung the regime mix by ~17 points. This walks the site list from
    both ends of the elastic-fraction ordering so the running val mix tracks the
    global one.

    Note the ``seed`` is unused: the walk is fully determined by the elastic
    fractions. On a dataset where every case has the same regime -- such as
    ``hra_dataset_fundus``, which is 100% elastic -- the fractions all tie and
    the ordering collapses to site coordinates, which hands val a *contiguous
    band* of sites rather than a random sample. Use ``random_site_split`` there.

    Returns ``(train_rows, val_rows)``, both sorted by ``case_id``.
    """
    groups = _site_groups(rows, site_keys)
    target_val = (1.0 - split_ratio) * len(rows)
    ordered = sorted(
        groups.items(),
        key=lambda kv: (sum(r[regime_key] == REGIME_ELASTIC for r in kv[1]) / len(kv[1]), kv[0]),
    )

    # Alternate low-elastic / high-elastic groups so the extremes cancel out.
    sequence, low, high = [], 0, len(ordered) - 1
    while low <= high:
        sequence.append(ordered[low][0])
        low += 1
        if low <= high:
            sequence.append(ordered[high][0])
            high -= 1

    return _split_from_val_set(groups, _pick_val_sites(groups, sequence, target_val))


def random_site_split(rows, split_ratio=0.8, seed=0,
                      site_keys=("site_x_mm", "site_y_mm")):
    """Split rows 4:1, holding out whole needle sites chosen at random.

    The right protocol whenever the stratification key has no signal -- e.g.
    ``hra_dataset_fundus``, where ``regime`` is ``elastic`` for all 2800 cases,
    so ``stratified_site_split`` degenerates into a coordinate-ordered walk and
    val comes out as a mid-depth band (train spanning z -18.07..-14.98 mm but
    val only -16.98..-15.18). Drawing the sites with ``seed`` instead gives val
    sites the same spatial distribution as train sites, which is what "hold out
    20%" is supposed to mean.

    Returns ``(train_rows, val_rows)``, both sorted by ``case_id``.
    """
    groups = _site_groups(rows, site_keys)
    target_val = (1.0 - split_ratio) * len(rows)
    keys = sorted(groups)
    sequence = [keys[i] for i in np.random.RandomState(seed).permutation(len(keys))]
    return _split_from_val_set(groups, _pick_val_sites(groups, sequence, target_val))



def random_split(rows, split_ratio=0.8, seed=0):
    """Plain seeded shuffle, per sample. Diagnostic baseline for the split."""
    order = np.random.RandomState(seed).permutation(len(rows))
    cut = int(round(split_ratio * len(rows)))
    train_idx, val_idx = set(order[:cut].tolist()), set(order[cut:].tolist())
    train = [r for i, r in enumerate(rows) if i in train_idx]
    val = [r for i, r in enumerate(rows) if i in val_idx]
    return (sorted(train, key=lambda r: r["case_id"]),
            sorted(val, key=lambda r: r["case_id"]))


def unit_or_zero(vector):
    """Unit vector, or zeros when the input is null/zero (no contact)."""
    if vector is None:
        return torch.zeros(3, dtype=torch.float32)
    tensor = torch.tensor(vector, dtype=torch.float32)
    norm = tensor.norm()
    return tensor / norm if norm > 0 else tensor


class HraRetinaDataset(Dataset):
    """One static end state per case: shared rest mesh + displacement field.

    The rest mesh and the graph edges are identical for every sample, so they
    are built once in ``__init__`` and only sliced in ``__getitem__``.

    Units: ``length_scale`` converts the dataset's SI metres into graph units.
    This is not cosmetic -- at SI scale ``to_log_freq`` collapses (the 5389-node
    rest mesh spans a rank-6/21 encoding, because the coordinates are so close
    to zero that sin/cos are all in their linear regime) and the displacement
    target is ~5e-5, three orders of magnitude below the decoder's initial
    output. ``1000.0`` gives millimetres, where the encoding has full rank.
    """

    def __init__(
        self,
        root_dir,
        n_points=-1,
        n_points_mode="global",
        graph_method="knn",
        neigbor_k=6,
        neigbor_radius=1.5e-3,
        sphere_radius=7.5e-4,
        rigid_n_points=256,
        force_max=8.4255e-06,
        length_scale=1000.0,
        press_depth_max=7.5e-5,
        condition="press_depth",
        split="train",
        split_ratio=0.8,
        split_seed=0,
        split_mode="site",
        manifest_name="manifest.csv",
        filter_status="ok",
        filter_hold_settled=True,
        filter_min_force=0.0,
        filter_min_disp_ratio=None,
        exclude_regression=True,
        use_needle_pose=False,
        preload=True,
        cache_dir=None,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.root_dir = root_dir
        self.split = split
        self.length_scale = float(length_scale)
        self.press_depth_max = float(press_depth_max)
        self.force_max = float(force_max)
        self.condition = condition
        self.n_points = n_points
        self.use_needle_pose = bool(use_needle_pose)
        # None -> use the case's physical needle tip radius.
        self.sphere_radius = None if sphere_radius is None else float(sphere_radius)

        rows = read_manifest(
            os.path.join(root_dir, manifest_name),
            status=filter_status,
            hold_settled=filter_hold_settled,
            min_force=filter_min_force,
            exclude_regression=exclude_regression,
            min_disp_ratio=filter_min_disp_ratio,
        )
        if split_mode == "site":
            train_rows, val_rows = stratified_site_split(rows, split_ratio, split_seed)
        elif split_mode == "site_random":
            train_rows, val_rows = random_site_split(rows, split_ratio, split_seed)
        elif split_mode == "random":
            train_rows, val_rows = random_split(rows, split_ratio, split_seed)
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")
        self.rows = train_rows if split == "train" else val_rows
        if not self.rows:
            raise ValueError(f"HRA {split} split is empty ({len(rows)} cases selected)")

        self._assert_split_is_clean(train_rows, val_rows, split_mode)

        # --- shared geometry, built once -------------------------------------
        rest = np.load(os.path.join(root_dir, "rest.npy"))
        if rest.ndim != 2 or rest.shape[1] != 3:
            raise ValueError(f"rest.npy has unexpected shape {rest.shape}")
        rest = (rest * self.length_scale).astype(np.float32)
        self.rest_pos = torch.from_numpy(rest)

        if n_points == -1:
            self.node_idx = None
            self.edge_index = construct_graph_kdtree(rest, k=neigbor_k)
        elif n_points_mode == "global":
            # One fixed node set for the whole dataset keeps edge_index,
            # soft_x and the metric comparable across samples.
            self.node_idx = self._global_subset(rest, int(n_points))
            self.edge_index = construct_graph_kdtree(rest[self.node_idx], k=neigbor_k)
        else:
            self.node_idx = None  # chosen per sample around the needle tip
            self.edge_index = None

        self._tree = None
        if n_points != -1 and n_points_mode != "global":
            self._tree = cKDTree(rest)

        sel = self.rest_pos if self.node_idx is None else self.rest_pos[self.node_idx]
        self.rest_sel = sel
        self.soft_x = to_log_freq(sel, 3, 1)

        # The rigid graph is a query set around the needle tip, not a physical
        # body: at the true needle radius (0.15 mm) its 256 points are almost
        # indistinguishable once encoded, so the default is one mesh median
        # edge (0.75 mm). It only translates between samples, hence one edge
        # index built on the unit sphere.
        sphere = fibonacci_sphere(int(rigid_n_points))
        self.rigid_dirs = torch.from_numpy(sphere).float()
        self.rigid_edge_index = construct_graph_kdtree(sphere * sphere_radius, k=neigbor_k)

        self.disp = self._load_displacements() if preload else None

    # -- setup helpers --------------------------------------------------------

    def _assert_split_is_clean(self, train_rows, val_rows, split_mode):
        train_ids = {r["case_id"] for r in train_rows}
        val_ids = {r["case_id"] for r in val_rows}
        if train_ids & val_ids:
            raise AssertionError("train and val share cases")
        if split_mode in ("site", "site_random"):
            train_sites = {(r["site_x_mm"], r["site_y_mm"]) for r in train_rows}
            val_sites = {(r["site_x_mm"], r["site_y_mm"]) for r in val_rows}
            if train_sites & val_sites:
                raise AssertionError(
                    f"{len(train_sites & val_sites)} needle site(s) appear in both splits"
                )

    def _global_subset(self, rest, n_points):
        """Deterministic spread-out subset, shared by every sample."""
        if n_points >= len(rest):
            return np.arange(len(rest))
        rng = np.random.RandomState(0)
        return np.sort(rng.choice(len(rest), size=n_points, replace=False))

    def _case_dir(self, row):
        return os.path.join(self.root_dir, row["case_id"])

    def _load_displacements(self):
        array = np.empty((len(self.rows), len(self.rest_sel), 3), dtype=np.float32)
        for i, row in enumerate(self.rows):
            disp = np.load(os.path.join(self._case_dir(row), "retina_displacement.npy"))
            if self.node_idx is not None:
                disp = disp[self.node_idx]
            array[i] = disp * self.length_scale
        return array

    def _read_meta(self, row):
        with open(os.path.join(self._case_dir(row), "meta.json")) as handle:
            return json.load(handle)

    @staticmethod
    def _bevel_dir(meta):
        """Unit vector giving the bevel's roll about the needle axis.

        ``needle_direction`` fixes the needle's *axis* but says nothing about
        how the needle is rolled about that axis. At ``tilt_deg == 0`` the four
        azimuth cases of a site therefore share one direction vector
        bit-for-bit while their targets differ by up to 67.7 um (median 8.6) --
        889 of the 2799 fundus cases (31.8%) collapse onto 2140 distinct inputs
        without this feature. The bevel tip sits one needle radius off the
        axis, so ``needle_centre - needle_tip`` projected perpendicular to the
        axis recovers the roll directly: its magnitude is the needle radius
        (155.5 um, constant across cases) and its direction rotates with the
        azimuth, separating all 230 degenerate input groups.

        Returns a zero vector if the field is missing or degenerate, which
        makes the feature inert rather than crashing on an older dataset.
        """
        centre = meta.get("needle_centre_position_m")
        tip = meta.get("needle_tip_position_m")
        axis = meta.get("needle_direction")
        if centre is None or tip is None or axis is None:
            return torch.zeros(3, dtype=torch.float32)
        axis = np.asarray(axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm == 0.0:
            return torch.zeros(3, dtype=torch.float32)
        axis = axis / norm
        offset = np.asarray(centre, dtype=np.float64) - np.asarray(tip, dtype=np.float64)
        lateral = offset - float(offset @ axis) * axis
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm == 0.0:
            return torch.zeros(3, dtype=torch.float32)
        return torch.tensor(lateral / lateral_norm, dtype=torch.float32)

    def _build_meta_data(self, row, meta):
        """Fixed schema: every tensor key must have the same shape per sample,
        because ``collate_fn`` stacks tensor-valued keys unconditionally."""
        # ``_feature_rigid`` (loaders/common.py) broadcasts ``force`` as a
        # scalar and ``force_vector`` as a 3-vector. In everyday those are the
        # applied force, i.e. a control input. Here the simulated contact force
        # is an output, so feeding it in would leak the label; use the control
        # inputs instead, which are the exact analogues.
        if self.condition == "force":
            scalar = float(meta["force"]) / self.force_max
            direction = unit_or_zero(meta["force_vector"])
        else:
            scalar = float(meta["press_depth_m"]) / self.press_depth_max
            direction = unit_or_zero(meta["needle_direction"])

        has_contact = float(meta["force"]) > 0.0
        return {
            # consumed by _feature_rigid
            "force": scalar,
            "force_vector": direction,
            # Consumed only when ``use_needle_pose`` is set; see _bevel_dir.
            "needle_bevel_dir": self._bevel_dir(meta),
            # reporting / conditioning
            "contact_normal": unit_or_zero(meta["contact_normal"]),
            "clamp_z_m": float(meta["clamp_z_m"]),
            # The rigid graph is built around this point, so it is also the
            # contact centre -- the loss uses it to weight the contact region.
            "needle_tip_mm": torch.tensor(
                meta["needle_tip_position_m"], dtype=torch.float32
            ) * self.length_scale,
            "force_n": float(meta["force"]),
            "has_contact": has_contact,
            "press_depth_m": float(meta["press_depth_m"]),
            "tilt_deg": float(meta["tilt_deg"]),
            "azimuth_deg": float(meta["azimuth_deg"]),
            "site_x_mm": float(row["site_x_mm"]),
            "site_y_mm": float(row["site_y_mm"]),
            "regime": row["regime"],
            "case_id": row["case_id"],
        }

    # -- Dataset interface ----------------------------------------------------

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        meta = self._read_meta(row)

        if self.disp is not None:
            disp = torch.from_numpy(self.disp[idx])
        else:
            disp = torch.from_numpy(
                np.load(os.path.join(self._case_dir(row), "retina_displacement.npy"))
                * self.length_scale
            ).float()

        tip = torch.tensor(meta["needle_tip_position_m"], dtype=torch.float32)
        tip = tip * self.length_scale

        if self.node_idx is not None:
            rest_sel, disp_sel = self.rest_sel, disp
            edge_index = self.edge_index
            soft_x = self.soft_x
        elif self.n_points == -1:
            rest_sel, disp_sel = self.rest_sel, disp
            edge_index = self.edge_index
            soft_x = self.soft_x
        else:
            _, nearest = self._tree.query(tip.numpy(), k=int(self.n_points))
            keep = np.sort(np.atleast_1d(nearest))
            rest_sel = self.rest_sel[keep]
            disp_sel = disp[keep]
            # Per-sample connectivity: rest and deformed still share one edge
            # index, which GradientConsistencyLoss relies on.
            edge_index = construct_graph_kdtree(rest_sel.numpy(), k=6)
            soft_x = to_log_freq(rest_sel, 3, 1)

        def_pos = rest_sel + disp_sel

        soft_rest_graph = Data(x=soft_x, edge_index=edge_index, pos=rest_sel)
        soft_def_graph = Data(
            x=to_log_freq(def_pos, 3, 1), edge_index=edge_index, pos=def_pos
        )

        radius = self._rigid_radius(meta)
        rigid_pos = self.rigid_dirs * radius + tip
        meta_data = self._build_meta_data(row, meta)
        rigid_x = _feature_rigid(meta_data, to_log_freq(rigid_pos, 3, 1))
        if self.use_needle_pose:
            # Appended here rather than inside _feature_rigid, which everyday
            # also calls and whose 25-wide layout its pins depend on.
            rigid_x = torch.cat(
                [rigid_x, meta_data["needle_bevel_dir"].repeat(rigid_pos.shape[0], 1)],
                dim=1,
            )
        rigid_graph = Data(
            x=rigid_x,
            edge_index=self.rigid_edge_index,
            pos=rigid_pos,
        )

        return "retina", soft_rest_graph, soft_def_graph, meta_data, rigid_graph

    def _rigid_radius(self, meta):
        if self.sphere_radius is not None:
            return float(self.sphere_radius) * self.length_scale
        return float(meta["needle_radius_m"]) * self.length_scale

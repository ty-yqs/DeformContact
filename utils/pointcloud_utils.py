import open3d as o3d
import numpy as np
import torch
from scipy import spatial
import torch_geometric.nn as pyg_nn


def construct_graph(point_cloud, k=None, radius=None):

    if radius is not None:
        edge_index = pyg_nn.radius_graph(point_cloud, r=radius, batch=None, loop=False)
    else:
        edge_index = pyg_nn.knn_graph(point_cloud, k=k, batch=None, loop=False)
    return edge_index


def construct_graph_kdtree(points, k=None, radius=None, undirected=True, workers=-1):
    """Build an ``edge_index`` from a bare point set using scipy's cKDTree.

    ``construct_graph`` above routes through ``pyg_nn.knn_graph`` /
    ``pyg_nn.radius_graph``, which need the optional ``torch-cluster``
    extension. That extension is not installed here, so ``construct_graph``
    raises ImportError. This variant has no compiled dependency, so it also
    works for meshes given as bare points with no triangle connectivity (the
    HRA dataset ships no ``.msh``).

    Args:
        points: (N, 3) torch.Tensor or numpy array.
        k: neighbours per point, *excluding* the point itself.
        radius: radius for a ball query instead of ``k``.
        undirected: emit both edge directions.
        workers: cKDTree query workers (``-1`` = all cores).

    Returns:
        (2, E) torch.long edge_index.
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    points = np.ascontiguousarray(points, dtype=np.float64)
    n = len(points)
    tree = spatial.cKDTree(points)

    if radius is not None:
        neighbours = tree.query_ball_point(points, r=radius, workers=workers)
        sizes = [len(lst) for lst in neighbours]
        src = np.repeat(np.arange(n, dtype=np.int64), sizes)
        dst = np.fromiter(
            (j for lst in neighbours for j in lst), dtype=np.int64, count=sum(sizes)
        )
        keep = src != dst
        src, dst = src[keep], dst[keep]
    else:
        k = min(int(k), n - 1)
        # cKDTree always reports the query point itself first, so ask for k + 1.
        _, idx = tree.query(points, k=k + 1, workers=workers)
        dst = np.asarray(idx)[:, 1:].reshape(-1).astype(np.int64)
        src = np.repeat(np.arange(n, dtype=np.int64), k)

    if undirected:
        src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
        order = np.lexsort((dst, src))
        src, dst = src[order], dst[order]

    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


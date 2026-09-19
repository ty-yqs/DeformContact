

import open3d as o3d
import numpy as np
from utils.graph_utils import *
from pyquaternion import Quaternion


# The filament offscreen renderer reliably handles only a few thousand line
# segments; denser graph edge sets render as a black image. Subsample edges
# beyond this count so the visualization stays intact.
_MAX_RENDER_LINES = 5000


def _subsample_lineset(geom, max_lines=_MAX_RENDER_LINES):
    lines = np.asarray(geom.lines)
    if len(lines) <= max_lines:
        return geom
    step = int(np.ceil(len(lines) / max_lines))
    new_geom = o3d.geometry.LineSet()
    new_geom.points = geom.points
    new_geom.lines = o3d.utility.Vector2iVector(lines[::step])
    if geom.has_colors():
        colors = np.asarray(geom.colors)
        new_geom.colors = o3d.utility.Vector3dVector(colors[::step])
    return new_geom


def _render_offscreen(geometries, save_path, width=1600, height=1200,
                      point_size=6.0, line_width=2.0,
                      background=(1.0, 1.0, 1.0, 1.0), side_view=False):
    """Render a list of o3d geometries to a PNG file without needing a display."""
    import open3d.visualization.rendering as rendering

    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(list(background))

    all_points = []
    for i, geom in enumerate(geometries):
        mat = rendering.MaterialRecord()
        if isinstance(geom, o3d.geometry.LineSet):
            geom = _subsample_lineset(geom)
            mat.shader = "unlitLine"
            mat.line_width = line_width
            # The filament renderer ignores per-line colors, so use the uniform
            # color as the material base color.
            if geom.has_colors():
                c = np.asarray(geom.colors)[0]
                mat.base_color = [float(c[0]), float(c[1]), float(c[2]), 1.0]
            else:
                mat.base_color = [0.2, 0.2, 0.2, 1.0]
        else:  # PointCloud
            mat.shader = "defaultUnlit"
            mat.point_size = point_size
        renderer.scene.add_geometry(f"geom_{i}", geom, mat)
        all_points.append(np.asarray(geom.points))

    pts = np.concatenate(all_points, axis=0)
    vmin, vmax = pts.min(axis=0), pts.max(axis=0)
    center = (vmin + vmax) / 2.0
    span = float((vmax - vmin).max())
    if span < 1e-6:
        span = 1.0
    if side_view:
        eye = center + np.array([0.0, span * 0.7, span * 2.0])
    else:
        eye = center + np.array([0.0, 0.0, span * 2.0])
    renderer.scene.camera.look_at(center, eye, [0.0, 1.0, 0.0])

    img = renderer.render_to_image()
    o3d.io.write_image(save_path, img)
    del renderer


def visualize_deformation_field(soft_rest_graph, soft_def_graph, rigid_graph, force_vector, save_path=None):

    soft_rest_mesh_np = soft_rest_graph.detach().numpy()
    soft_def_mesh_np = soft_def_graph.detach().numpy()
    rigid_graph_np = rigid_graph.detach().numpy()
    force_vector_np = force_vector.detach().numpy()

    deformation_magnitudes = np.linalg.norm(soft_def_mesh_np - soft_rest_mesh_np, axis=1)

    pcd_rest = o3d.geometry.PointCloud()
    pcd_def = o3d.geometry.PointCloud()
    pcd_rigid = o3d.geometry.PointCloud()
    pcd_rest.points = o3d.utility.Vector3dVector(soft_rest_mesh_np)
    pcd_def.points = o3d.utility.Vector3dVector(soft_def_mesh_np)
    pcd_rigid.points = o3d.utility.Vector3dVector(rigid_graph_np)

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.concatenate((soft_rest_mesh_np, soft_def_mesh_np)))
    n = len(soft_rest_mesh_np)
    lineset.lines = o3d.utility.Vector2iVector([(i, i + n) for i in range(n)])


    direction = force_vector_np / np.linalg.norm(force_vector_np)
    scaled_direction = -direction * 0.1 

    start_points = rigid_graph_np
    end_points = rigid_graph_np + scaled_direction
    vector_lineset = o3d.geometry.LineSet()
    vector_lineset.points = o3d.utility.Vector3dVector(np.vstack([start_points, end_points]))
    n_rigid = len(start_points)
    vector_lineset.lines = o3d.utility.Vector2iVector([(i, i + n_rigid) for i in range(n_rigid)])
    vector_lineset.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.9] for _ in range(n_rigid)])

    pcd_rigid.paint_uniform_color([0.0, 0, 0.8])
    pcd_rest.paint_uniform_color([0, 0.8, 0])
    pcd_def.paint_uniform_color([0.8, 0.0, 0])
    lineset.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for _ in range(n)])

    geometries = [pcd_rest, pcd_def, lineset, pcd_rigid, vector_lineset]
    if save_path is not None:
        _render_offscreen(geometries, save_path)
    else:
        o3d.visualization.draw_geometries(geometries)


def visualize_merged_graphs(soft_rest_graph, soft_def_graph=None, rigid_graph=None, pred_graph=None, save_path=None):
    soft_rest_points_np = soft_rest_graph.pos.cpu().numpy()
    soft_rest_edge_index_np = soft_rest_graph.edge_index.t().cpu().numpy().astype(np.int32)

    x_min = np.min(soft_rest_points_np[:, 0])
    x_max = np.max(soft_rest_points_np[:, 0])
    translation = (x_max - x_min)  *1.5

    soft_rest_pcd = o3d.geometry.PointCloud()
    soft_rest_pcd.points = o3d.utility.Vector3dVector(soft_rest_points_np)
    soft_rest_pcd.paint_uniform_color([0, 0.8, 0])

    soft_rest_lines = o3d.geometry.LineSet()
    soft_rest_lines.points = o3d.utility.Vector3dVector(soft_rest_points_np)
    soft_rest_lines.lines = o3d.utility.Vector2iVector(soft_rest_edge_index_np)

    geometries = [soft_rest_pcd, soft_rest_lines]

    rigid_points_np = rigid_graph.pos.cpu().numpy() 
    rigid_edge_index_np = rigid_graph.edge_index.t().cpu().numpy().astype(np.int32)
    
    rigid_pcd = o3d.geometry.PointCloud()
    rigid_pcd.points = o3d.utility.Vector3dVector(rigid_points_np)
    rigid_pcd.paint_uniform_color([0.5, 0.5, 0.8])  # Blue-ish for the rigid mesh
    
    rigid_lines = o3d.geometry.LineSet()
    rigid_lines.points = o3d.utility.Vector3dVector(rigid_points_np)
    rigid_lines.lines = o3d.utility.Vector2iVector(rigid_edge_index_np)
    

    # coor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    geometries.extend([rigid_pcd, rigid_lines])


    soft_def_points_np = soft_def_graph.pos.cpu().numpy() + [translation, 0, 0]
    soft_def_edge_index_np = soft_def_graph.edge_index.t().cpu().numpy().astype(np.int32)
    
    soft_def_pcd = o3d.geometry.PointCloud()
    soft_def_pcd.points = o3d.utility.Vector3dVector(soft_def_points_np)
    soft_def_pcd.paint_uniform_color([0.8, 0.8, 0]) 
    
    soft_def_lines = o3d.geometry.LineSet()
    soft_def_lines.points = o3d.utility.Vector3dVector(soft_def_points_np)
    soft_def_lines.lines = o3d.utility.Vector2iVector(soft_def_edge_index_np)
    
    geometries.extend([soft_def_pcd, soft_def_lines])


    rigid_points_np = rigid_graph.pos.cpu().numpy() + [translation, 0, 0]
    rigid_edge_index_np = rigid_graph.edge_index.t().cpu().numpy().astype(np.int32)
    
    rigid_pcd = o3d.geometry.PointCloud()
    rigid_pcd.points = o3d.utility.Vector3dVector(rigid_points_np)
    rigid_pcd.paint_uniform_color([0.5, 0.5, 0.8]) 
    
    rigid_lines = o3d.geometry.LineSet()
    rigid_lines.points = o3d.utility.Vector3dVector(rigid_points_np)
    rigid_lines.lines = o3d.utility.Vector2iVector(rigid_edge_index_np)
    

    geometries.extend([rigid_pcd, rigid_lines])
    

    if pred_graph:
        pred_points_np = pred_graph.pos.cpu().numpy() + [2*translation, 0, 0]
        pred_edge_index_np = pred_graph.edge_index.t().cpu().numpy().astype(np.int32)
        
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_points_np)
        pred_pcd.paint_uniform_color([0.8, 0, 0])  
        
        pred_lines = o3d.geometry.LineSet()
        pred_lines.points = o3d.utility.Vector3dVector(pred_points_np)
        pred_lines.lines = o3d.utility.Vector2iVector(pred_edge_index_np)
        
        geometries.extend([pred_pcd, pred_lines])


        rigid_points_np = rigid_graph.clone().pos.cpu().numpy() + [2*translation, 0, 0]
        rigid_edge_index_np = rigid_graph.clone().edge_index.t().cpu().numpy().astype(np.int32)
        
        rigid_pcd = o3d.geometry.PointCloud()
        rigid_pcd.points = o3d.utility.Vector3dVector(rigid_points_np)
        rigid_pcd.paint_uniform_color([0.5, 0.5, 0.8])  
        
        rigid_lines = o3d.geometry.LineSet()
        rigid_lines.points = o3d.utility.Vector3dVector(rigid_points_np)
        rigid_lines.lines = o3d.utility.Vector2iVector(rigid_edge_index_np)
        
        geometries.extend([rigid_pcd, rigid_lines])

    if save_path is not None:
        _render_offscreen(geometries, save_path)
    else:
        o3d.visualization.draw_geometries(geometries)


def map_deformation_to_color(deformation_values):
    min_val, max_val = np.min(deformation_values), np.max(deformation_values)
    normalized_values = (deformation_values - min_val) / (max_val - min_val)
    

    colors = np.ones((normalized_values.shape[0], 3)) - np.expand_dims(normalized_values, axis=1)
    
    return colors

def visualize_deformations_normals_colors(soft_rest_graph, soft_def_graph=None, save_path=None):
    soft_rest_points_np = soft_rest_graph.pos.cpu().numpy()
    soft_rest_edge_index_np = soft_rest_graph.edge_index.t().cpu().numpy().astype(np.int32)

    x_min = np.min(soft_rest_points_np[:, 0])
    x_max = np.max(soft_rest_points_np[:, 0])
    translation = (x_max - x_min)  *1.5

    soft_rest_pcd = o3d.geometry.PointCloud()
    soft_rest_pcd.points = o3d.utility.Vector3dVector(soft_rest_points_np)
    soft_rest_pcd.estimate_normals()    

    soft_rest_lines = o3d.geometry.LineSet()
    soft_rest_lines.points = o3d.utility.Vector3dVector(soft_rest_points_np)
    soft_rest_lines.lines = o3d.utility.Vector2iVector(soft_rest_edge_index_np)
    

    geometries = [soft_rest_pcd, soft_rest_lines]
    if soft_def_graph:

        soft_def_points_np = soft_def_graph.pos.cpu().numpy() + [translation, 0, 0]
        soft_def_edge_index_np = soft_def_graph.edge_index.t().cpu().numpy().astype(np.int32)

        soft_def_pcd = o3d.geometry.PointCloud()
        soft_def_pcd.points = o3d.utility.Vector3dVector(soft_def_points_np)
        soft_def_pcd.estimate_normals() 

        soft_def_lines = o3d.geometry.LineSet()
        soft_def_lines.points = o3d.utility.Vector3dVector(soft_def_points_np)
        soft_def_lines.lines = o3d.utility.Vector2iVector(soft_def_edge_index_np)

        geometries.extend([soft_def_pcd, soft_def_lines]) 

    if save_path is not None:
        _render_offscreen(geometries, save_path, side_view=True)
    else:
        show_from_side(geometries)



def show_from_center(objs):
    vis = o3d.visualization.Visualizer()

    vis.create_window()
    [vis.add_geometry(obj) for obj in objs]
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
    pose = np.eye(4)
    pose[2, 3] = 1
    cam.extrinsic = pose
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
    vis.get_render_option().point_size = 2
    vis.run()
    vis.destroy_window()


def show_from_side(objs):
    vis = o3d.visualization.Visualizer()

    vis.create_window()
    [vis.add_geometry(obj) for obj in objs]
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
    top_rot = Quaternion(axis=(1.0, 0.0, 0.0), degrees=200)
    pose = np.eye(4)
    pose[2, 3] = 2.0

    pose[0:3, 0:3] = top_rot.rotation_matrix
    cam.extrinsic = pose
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
    vis.get_render_option().point_size = 8
    vis.run()
    vis.destroy_window()


import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc.geometry import Geometry, trimesh, pointcloud

from diag_utils import (
    create_ddf, poll_keyboard, compute_newton_dir,
    vector_section, flag_label, build_pt_voronoi, set_d_hat,
)

# engine + feature
engine, world, ddf = create_ddf(__file__)

# geometry
tri_verts = np.array([
    [-1.0, 0.0, -0.5],
    [ 1.0, 0.0, -0.5],
    [ 0.0, 0.0,  1.0],
], dtype=np.float64)
tri_faces = np.array([[0, 1, 2]], dtype=np.int32)
tri_sc = trimesh(tri_verts, tri_faces)
tri_sc.vertices().create("d_hat", 0.10)
tri_sc.vertices().create("thickness", 0.0)

point_pos = np.array([[0.0, 0.5, 0.0]], dtype=np.float64)
pt_sc = pointcloud(point_pos)
pt_sc.vertices().create("d_hat", 0.10)
pt_sc.vertices().create("thickness", 0.0)

R = Geometry()

# polyscope
ps.init()
ps.set_ground_plane_height(-2.0)
ps.set_up_dir("y_up")

tri_mesh = ps.register_surface_mesh("triangle", tri_verts, tri_faces)
tri_mesh.set_edge_width(1.5)
tri_mesh.set_transparency(0.6)

pt_cloud = ps.register_point_cloud("query_point", point_pos)
pt_cloud.set_radius(0.04)

closest_cloud = ps.register_point_cloud("closest_point",
    np.array([[0, 0, 0]], dtype=np.float64))
closest_cloud.set_radius(0.03)

dist_line = ps.register_curve_network("distance_line",
    np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64),
    np.array([[0, 1]], dtype=np.int32))
dist_line.set_radius(0.005)

vr_v, vr_f, vr_c = build_pt_voronoi(tri_verts[0], tri_verts[1], tri_verts[2])
if vr_v is not None:
    vor_mesh = ps.register_surface_mesh("voronoi_planes", vr_v, vr_f)
    vor_mesh.set_transparency(0.3)
    vor_mesh.add_color_quantity("region", vr_c, defined_on='faces', enabled=True)

newton_cloud = ps.register_point_cloud("newton_vectors",
    np.array([point_pos[0], tri_verts[0], tri_verts[1], tri_verts[2]]))
newton_cloud.set_radius(0.01)

# state
move_speed = 0.05
d_hat = 0.10

def on_update():
    global point_pos, pt_sc, R, move_speed, d_hat

    dx, dy, dz = poll_keyboard(move_speed)
    if dx != 0 or dy != 0 or dz != 0:
        point_pos[0] += np.array([dx, dy, dz])
        pos_view = view(pt_sc.positions())
        pos_view[0] = point_pos[0].reshape(3, 1)
        pt_cloud.update_point_positions(point_pos)

    ddf.compute_point_triangle_distance(R, pt_sc, tri_sc)

    # read results
    dist2_attr   = R.instances().find("dist2")
    flag_attr    = R.instances().find("flag")
    coord_attr   = R.instances().find("coord")
    grad_attr    = R.instances().find("dist2/grad")
    hess_attr    = R.instances().find("dist2/hess")
    barrier_attr = R.instances().find("barrier")
    bg_attr      = R.instances().find("barrier/grad")
    bh_attr      = R.instances().find("barrier/hess")

    dist2_val   = float(dist2_attr.view().flatten()[0]) if dist2_attr else None
    flag_val    = flag_attr.view().reshape(-1, 4)[0]    if flag_attr  else None
    coord_val   = coord_attr.view().reshape(-1, 4)[0]   if coord_attr else None
    barrier_val = float(barrier_attr.view().flatten()[0]) if barrier_attr else None

    if coord_val is not None:
        verts = np.array([point_pos[0], tri_verts[0], tri_verts[1], tri_verts[2]])
        c = coord_val.astype(np.float64)
        cp = c[0] * verts[0] + c[1] * verts[1] + c[2] * verts[2] + c[3] * verts[3]
        closest_cloud.update_point_positions(cp.reshape(1, 3))
        dist_line.update_node_positions(np.array([point_pos[0], cp], dtype=np.float64))

    g_arr  = grad_attr.view().flatten()[:12].astype(np.float64) if grad_attr else None
    d_arr  = compute_newton_dir(g_arr, hess_attr, flag_val, 4)
    bg_arr = bg_attr.view().flatten()[:12].astype(np.float64) if bg_attr else None
    bd_arr = compute_newton_dir(bg_arr, bh_attr, flag_val, 4)

    all_pos = np.array([point_pos[0], tri_verts[0], tri_verts[1], tri_verts[2]])
    newton_cloud.update_point_positions(all_pos)
    labels = ["P", "T0", "T1", "T2"]

    _, move_speed = imgui.DragFloat("Move Speed", move_speed, 0.001, 0.001, 1.0)
    changed_dh, d_hat = imgui.DragFloat("d_hat", d_hat, 0.001, 0.001, 1.0)
    if changed_dh:
        set_d_hat(pt_sc, d_hat)
        set_d_hat(tri_sc, d_hat)

    # Distance
    if imgui.CollapsingHeader("Distance", imgui.ImGuiTreeNodeFlags_DefaultOpen):
        imgui.Text(f"Point: ({point_pos[0][0]:.3f}, {point_pos[0][1]:.3f}, {point_pos[0][2]:.3f})")
        if dist2_val is not None:
            imgui.Text(f"dist2 : {dist2_val:.6f}")
            imgui.Text(f"dist  : {np.sqrt(max(dist2_val, 0)):.6f}")
        if flag_val is not None:
            f = flag_val.flatten()
            imgui.Text(f"flag  : [{f[0]},{f[1]},{f[2]},{f[3]}] -> {flag_label(f)}")
        if coord_val is not None:
            c = coord_val.flatten().astype(np.float64)
            imgui.Text(f"coord : [{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},{c[3]:.4f}]")

    # Barrier
    if imgui.CollapsingHeader("Barrier", imgui.ImGuiTreeNodeFlags_DefaultOpen):
        if barrier_val is not None:
            imgui.Text(f"barrier : {barrier_val:.6e}")

    vector_section("gX(D)", "gD_table", g_arr, labels, newton_cloud)
    vector_section("dX(D)", "dD_table", d_arr, labels, newton_cloud)
    vector_section("gX(B)", "gB_table", bg_arr, labels, newton_cloud)
    vector_section("dX(B)", "dB_table", bd_arr, labels, newton_cloud)

    if imgui.CollapsingHeader("Info"):
        imgui.Text("Voronoi: blue=edge wall, orange=vertex wall")
        imgui.Text("Controls: W/S (+-Z)  A/D (+-X)  Q/E (+-Y)")

ps.set_user_callback(on_update)
ps.show()

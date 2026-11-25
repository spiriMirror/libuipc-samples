import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc.geometry import Geometry, linemesh

from diag_utils import (
    create_ddf, poll_keyboard, compute_newton_dir,
    vector_section, flag_label, build_edge_voronoi,
    rot_x, rot_y, rot_z, set_d_hat,
)

# engine + feature
engine, world, ddf = create_ddf(__file__)

# geometry
# Edge A: static, along X axis
ea_verts = np.array([[-1.0, 0.0, 0.0],
                     [ 1.0, 0.0, 0.0]], dtype=np.float64)
ea_edges = np.array([[0, 1]], dtype=np.int32)
ea_sc = linemesh(ea_verts, ea_edges)
ea_sc.vertices().create("d_hat", 0.10)
ea_sc.vertices().create("thickness", 0.0)
ea_rest = linemesh(ea_verts.copy(), ea_edges.copy())

# Edge B: defined by center, half-length, and rotation angles
eb_half = np.array([0.0, 0.0, 1.0], dtype=np.float64)
eb_center = np.array([0.0, 0.5, 0.0], dtype=np.float64)
eb_rot_deg = [0.0, 0.0, 0.0]
eb_edges = np.array([[0, 1]], dtype=np.int32)

def rebuild_eb():
    R = rot_z(np.radians(eb_rot_deg[2])) @ rot_y(np.radians(eb_rot_deg[1])) @ rot_x(np.radians(eb_rot_deg[0]))
    d = R @ eb_half
    return np.array([eb_center - d, eb_center + d], dtype=np.float64)

eb_verts = rebuild_eb()
eb_sc = linemesh(eb_verts.copy(), eb_edges.copy())
eb_sc.vertices().create("d_hat", 0.10)
eb_sc.vertices().create("thickness", 0.0)
eb_rest = linemesh(eb_verts.copy(), eb_edges.copy())

R_geo = Geometry()

# polyscope
ps.init()
ps.set_ground_plane_height(-2.0)
ps.set_up_dir("y_up")

ps_ea = ps.register_curve_network("edge_a", ea_verts, ea_edges)
ps_ea.set_radius(0.008)

ps_eb = ps.register_curve_network("edge_b", eb_verts, eb_edges)
ps_eb.set_radius(0.008)

cp_cloud = ps.register_point_cloud("closest_points",
    np.zeros((2, 3), dtype=np.float64))
cp_cloud.set_radius(0.025)

dist_line = ps.register_curve_network("distance_line",
    np.zeros((2, 3), dtype=np.float64),
    np.array([[0, 1]], dtype=np.int32))
dist_line.set_radius(0.004)

vr_a_v, vr_a_f = build_edge_voronoi(ea_verts[0], ea_verts[1])
vor_a = None
if vr_a_v is not None:
    vor_a = ps.register_surface_mesh("voronoi_edge_a", vr_a_v, vr_a_f)
    vor_a.set_transparency(0.3)
    vor_a.set_color((0.3, 0.6, 1.0))

vr_b_v, vr_b_f = build_edge_voronoi(eb_verts[0], eb_verts[1])
vor_b = None
if vr_b_v is not None:
    vor_b = ps.register_surface_mesh("voronoi_edge_b", vr_b_v, vr_b_f)
    vor_b.set_transparency(0.3)
    vor_b.set_color((1.0, 0.6, 0.3))

newton_cloud = ps.register_point_cloud("newton_vectors",
    np.array([ea_verts[0], ea_verts[1], eb_verts[0], eb_verts[1]]))
newton_cloud.set_radius(0.01)

# state
move_speed = 0.05
d_hat = 0.10

def sync_eb():
    """Recompute eb_verts from center/rotation and push to SC + polyscope."""
    global eb_verts
    eb_verts = rebuild_eb()
    pos_view = view(eb_sc.positions())
    for i in range(2):
        pos_view[i] = eb_verts[i].reshape(3, 1)
    ps_eb.update_node_positions(eb_verts)

def on_update():
    global eb_center, move_speed, d_hat

    dx, dy, dz = poll_keyboard(move_speed)
    moved = (dx != 0 or dy != 0 or dz != 0)
    if moved:
        eb_center += np.array([dx, dy, dz])

    _, move_speed = imgui.DragFloat("Move Speed", move_speed, 0.001, 0.001, 1.0)
    changed_dh, d_hat = imgui.DragFloat("d_hat", d_hat, 0.001, 0.001, 1.0)
    if changed_dh:
        set_d_hat(ea_sc, d_hat)
        set_d_hat(eb_sc, d_hat)
    imgui.Text("Edge B rotation (deg)")
    changed_rx, eb_rot_deg[0] = imgui.DragFloat("rot X", eb_rot_deg[0], 0.5, -180.0, 180.0)
    changed_ry, eb_rot_deg[1] = imgui.DragFloat("rot Y", eb_rot_deg[1], 0.5, -180.0, 180.0)
    changed_rz, eb_rot_deg[2] = imgui.DragFloat("rot Z", eb_rot_deg[2], 0.5, -180.0, 180.0)

    if moved or changed_rx or changed_ry or changed_rz:
        sync_eb()
        if vor_b is not None:
            vr_b_new, _ = build_edge_voronoi(eb_verts[0], eb_verts[1])
            if vr_b_new is not None:
                vor_b.update_vertex_positions(vr_b_new)

    ddf.compute_edge_edge_distance(R_geo, ea_sc, eb_sc, ea_rest, eb_rest)

    dist2_attr     = R_geo.instances().find("dist2")
    flag_attr      = R_geo.instances().find("flag")
    coord_attr     = R_geo.instances().find("coord")
    grad_attr      = R_geo.instances().find("dist2/grad")
    hess_attr      = R_geo.instances().find("dist2/hess")
    eps_x_attr     = R_geo.instances().find("eps_x")
    ek_attr        = R_geo.instances().find("e_k")
    ek_grad_attr   = R_geo.instances().find("e_k/grad")
    ek_hess_attr   = R_geo.instances().find("e_k/hess")
    barrier_attr   = R_geo.instances().find("barrier")
    bg_attr        = R_geo.instances().find("barrier/grad")
    bh_attr        = R_geo.instances().find("barrier/hess")

    dist2_val   = float(dist2_attr.view().flatten()[0]) if dist2_attr else None
    flag_val    = flag_attr.view().reshape(-1, 4)[0]    if flag_attr  else None
    coord_val   = coord_attr.view().reshape(-1, 4)[0]   if coord_attr else None
    eps_x_val   = float(eps_x_attr.view().flatten()[0]) if eps_x_attr else None
    ek_val      = float(ek_attr.view().flatten()[0])    if ek_attr    else None
    barrier_val = float(barrier_attr.view().flatten()[0]) if barrier_attr else None

    if coord_val is not None:
        c = coord_val.astype(np.float64)
        cp_a = c[0] * ea_verts[0] + c[1] * ea_verts[1]
        cp_b = c[2] * eb_verts[0] + c[3] * eb_verts[1]
        cp_cloud.update_point_positions(np.array([cp_a, cp_b]))
        dist_line.update_node_positions(np.array([cp_a, cp_b]))

    g_arr    = grad_attr.view().flatten()[:12].astype(np.float64)    if grad_attr    else None
    mg_arr   = ek_grad_attr.view().flatten()[:12].astype(np.float64) if ek_grad_attr else None
    bg_arr   = bg_attr.view().flatten()[:12].astype(np.float64)      if bg_attr      else None
    d_arr    = compute_newton_dir(g_arr, hess_attr, flag_val, 4)
    dek_arr  = compute_newton_dir(mg_arr, ek_hess_attr, flag_val, 4)
    bd_arr   = compute_newton_dir(bg_arr, bh_attr, flag_val, 4)

    all_pos = np.array([ea_verts[0], ea_verts[1], eb_verts[0], eb_verts[1]])
    newton_cloud.update_point_positions(all_pos)
    labels = ["Ea0", "Ea1", "Eb0", "Eb1"]

    # Distance
    if imgui.CollapsingHeader("Distance", imgui.ImGuiTreeNodeFlags_DefaultOpen):
        imgui.Text(f"Edge B: ({eb_verts[0][0]:.2f},{eb_verts[0][1]:.2f},{eb_verts[0][2]:.2f})"
                   f" -> ({eb_verts[1][0]:.2f},{eb_verts[1][1]:.2f},{eb_verts[1][2]:.2f})")
        if dist2_val is not None:
            imgui.Text(f"dist2 : {dist2_val:.6f}")
            imgui.Text(f"dist  : {np.sqrt(max(dist2_val, 0)):.6f}")
        if flag_val is not None:
            f = flag_val.flatten()
            imgui.Text(f"flag  : [{f[0]},{f[1]},{f[2]},{f[3]}] -> {flag_label(f)}")
        if coord_val is not None:
            c = coord_val.flatten().astype(np.float64)
            imgui.Text(f"coord : [{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},{c[3]:.4f}]")

    # Mollifier
    if imgui.CollapsingHeader("Mollifier", imgui.ImGuiTreeNodeFlags_DefaultOpen):
        if ek_val is not None:
            active = "YES" if ek_val < 1.0 else "no"
            imgui.Text(f"e_k    : {ek_val:.6f}  (active: {active})")
        if eps_x_val is not None:
            imgui.Text(f"eps_x  : {eps_x_val:.6e}")

    # Barrier
    if imgui.CollapsingHeader("Barrier", imgui.ImGuiTreeNodeFlags_DefaultOpen):
        if barrier_val is not None:
            imgui.Text(f"barrier : {barrier_val:.6e}")

    vector_section("gX(D)",   "g_table",   g_arr,   labels, newton_cloud)
    vector_section("gX(e_k)", "mg_table",  mg_arr,  labels, newton_cloud)
    vector_section("dX(e_k)", "dek_table", dek_arr, labels, newton_cloud)
    vector_section("dX(D)",   "d_table",   d_arr,   labels, newton_cloud)
    vector_section("gX(B)",   "gB_table",  bg_arr,  labels, newton_cloud)
    vector_section("dX(B)",   "dB_table",  bd_arr,  labels, newton_cloud)

    if imgui.CollapsingHeader("Info"):
        imgui.Text("Voronoi: blue=edge A, orange=edge B")
        imgui.Text("Controls: W/S (+-Z)  A/D (+-X)  Q/E (+-Y)")

ps.set_user_callback(on_update)
ps.show()

import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc.geometry import Geometry, pointcloud

from diag_utils import create_ddf, poll_keyboard, compute_newton_dir, vector_section, set_d_hat

# engine + feature
engine, world, ddf = create_ddf(__file__)

# geometry
pos_a = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
sc_a = pointcloud(pos_a)
sc_a.vertices().create("d_hat", 0.10)
sc_a.vertices().create("thickness", 0.0)

pos_b = np.array([[0.0, 0.5, 0.0]], dtype=np.float64)
sc_b = pointcloud(pos_b)
sc_b.vertices().create("d_hat", 0.10)
sc_b.vertices().create("thickness", 0.0)

R = Geometry()

# polyscope
ps.init()
ps.set_ground_plane_height(-2.0)
ps.set_up_dir("y_up")

cloud_a = ps.register_point_cloud("point_a", pos_a)
cloud_a.set_radius(0.04)

cloud_b = ps.register_point_cloud("point_b", pos_b)
cloud_b.set_radius(0.04)

dist_line = ps.register_curve_network("distance_line",
    np.zeros((2, 3), dtype=np.float64),
    np.array([[0, 1]], dtype=np.int32))
dist_line.set_radius(0.004)

newton_cloud = ps.register_point_cloud("newton_vectors",
    np.array([pos_a[0], pos_b[0]]))
newton_cloud.set_radius(0.01)

# state
move_speed = 0.05
d_hat = 0.10

def on_update():
    global pos_b, move_speed, d_hat

    dx, dy, dz = poll_keyboard(move_speed)
    if dx != 0 or dy != 0 or dz != 0:
        pos_b[0] += np.array([dx, dy, dz])
        pos_view = view(sc_b.positions())
        pos_view[0] = pos_b[0].reshape(3, 1)
        cloud_b.update_point_positions(pos_b)

    ddf.compute_point_point_distance(R, sc_a, sc_b)

    dist2_attr   = R.instances().find("dist2")
    grad_attr    = R.instances().find("dist2/grad")
    hess_attr    = R.instances().find("dist2/hess")
    barrier_attr = R.instances().find("barrier")
    bg_attr      = R.instances().find("barrier/grad")
    bh_attr      = R.instances().find("barrier/hess")

    dist2_val   = float(dist2_attr.view().flatten()[0])   if dist2_attr   else None
    barrier_val = float(barrier_attr.view().flatten()[0]) if barrier_attr else None

    dist_line.update_node_positions(np.array([pos_a[0], pos_b[0]]))

    g_arr  = grad_attr.view().flatten()[:6].astype(np.float64) if grad_attr else None
    d_arr  = compute_newton_dir(g_arr, hess_attr, None, 2)
    bg_arr = bg_attr.view().flatten()[:6].astype(np.float64) if bg_attr else None
    bd_arr = compute_newton_dir(bg_arr, bh_attr, None, 2)

    all_pos = np.array([pos_a[0], pos_b[0]])
    newton_cloud.update_point_positions(all_pos)
    labels = ["A", "B"]

    _, move_speed = imgui.DragFloat("Move Speed", move_speed, 0.001, 0.001, 1.0)
    changed_dh, d_hat = imgui.DragFloat("d_hat", d_hat, 0.001, 0.001, 1.0)
    if changed_dh:
        set_d_hat(sc_a, d_hat)
        set_d_hat(sc_b, d_hat)

    # Distance
    if imgui.CollapsingHeader("Distance", imgui.ImGuiTreeNodeFlags_DefaultOpen):
        imgui.Text(f"A: ({pos_a[0][0]:.3f}, {pos_a[0][1]:.3f}, {pos_a[0][2]:.3f})")
        imgui.Text(f"B: ({pos_b[0][0]:.3f}, {pos_b[0][1]:.3f}, {pos_b[0][2]:.3f})")
        if dist2_val is not None:
            imgui.Text(f"dist2 : {dist2_val:.6f}")
            imgui.Text(f"dist  : {np.sqrt(max(dist2_val, 0)):.6f}")

    # Barrier
    if imgui.CollapsingHeader("Barrier", imgui.ImGuiTreeNodeFlags_DefaultOpen):
        if barrier_val is not None:
            imgui.Text(f"barrier : {barrier_val:.6e}")

    vector_section("gX(D)", "gD_table", g_arr, labels, newton_cloud)
    vector_section("dX(D)", "dD_table", d_arr, labels, newton_cloud)
    vector_section("gX(B)", "gB_table", bg_arr, labels, newton_cloud)
    vector_section("dX(B)", "dB_table", bd_arr, labels, newton_cloud)

    if imgui.CollapsingHeader("Info"):
        imgui.Text("Controls: W/S (+-Z)  A/D (+-X)  Q/E (+-Y)")

ps.set_user_callback(on_update)
ps.show()

"""Example 92 -- Stiff-GIPC set_case5: twisting bar with animated soft constraints.

Port of Stiff-GIPC gl_main.cu set_case5: high_mat.msh bar (E=1e4, nu=0.48,
rho=1000), gravity OFF, both ends (|x| > 0.5) driven by soft position
constraints rotating in opposite directions about the X axis
(angular velocity pi/5 rad/s; x<0 end clockwise, x>0 end counterclockwise).
Stiff runs this case with P_type=1, so MAS is enabled here
(linear_system/fem_preconditioner = "mas").

Global parameters follow the established Stiff-GIPC alignment (see 88/89).

Usage:
  python main.py                  # GUI with run/stop
  python main.py --headless [N]   # N frames (default 100)
"""
import os, sys

import numpy as np
import uipc
from uipc import Logger, Timer, Transform, Vector3, builtin, view, Animation
from uipc.core import Engine, World, Scene
from uipc.geometry import (SimplicialComplexIO, flip_inward_triangles, ground,
                           label_surface, label_triangle_orient)
from uipc.constitution import ElasticModuli, StableNeoHookean, SoftPositionConstraint

from asset_dir import AssetDir

HEADLESS = "--headless" in sys.argv
_positional = [a for a in sys.argv[1:] if not a.startswith("--")]
N_FRAMES = int(_positional[0]) if _positional else 100

Logger.set_level(Logger.Level.Info if HEADLESS else Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.01
config["gravity"] = [[0.0], [0.0], [0.0]]  # case5: no gravity
config["contact"]["d_hat_relative"] = 1e-3
config["newton"]["velocity_tol_relative"] = 1e-2
config["contact"]["eps_velocity_relative"] = 1e-2
config["linear_system"]["tol_rate"] = 1e-4
config["newton"]["transrate_tol"] = 10
config["newton"]["semi_implicit"]["enable"] = 1
config["newton"]["semi_implicit"]["beta_tol"] = 1e-2
config["newton"]["semi_implicit"]["K_min"] = 6
# Stiff P_type=1 -> MAS
config["linear_system"]["fem_preconditioner"] = "mas"
scene = Scene(config)

scene.contact_tabular().default_model(0.2, 1e8)
default_contact = scene.contact_tabular().default_element()

snh = StableNeoHookean()
spc = SoftPositionConstraint()

bar_mesh = SimplicialComplexIO().read(f"{AssetDir.tetmesh_path()}/high_mat.msh")
label_surface(bar_mesh)
label_triangle_orient(bar_mesh)
bar_mesh = flip_inward_triangles(bar_mesh)

snh.apply_to(bar_mesh, ElasticModuli.youngs_poisson(1e4, 0.48), 1e3)
spc.apply_to(bar_mesh, 100.0)
default_contact.apply_to(bar_mesh)
bar_obj = scene.objects().create("twisting_bar")
bar_obj.geometries().create(bar_mesh)

# constrained end vertices (|x| > 0.5, Stiff rule with eps=1e-4)
rest_pos = np.asarray(bar_mesh.positions().view()).reshape(-1, 3)
EPS = 1e-4
END_MASK = np.abs(rest_pos[:, 0]) > 0.5 - EPS
print(f"soft constraint num: {END_MASK.sum()}")

ANGULAR_VEL = np.pi / 5  # rad/s
ROT_PER_STEP = None      # filled per frame below

animator = scene.animator()


def twist(info: Animation.UpdateInfo):
    geo: SimplicialComplex = info.geo_slots()[0].geometry()
    rest_geo: SimplicialComplex = info.rest_geo_slots()[0].geometry()
    rest = np.asarray(rest_geo.positions().view()).reshape(-1, 3)

    ic_view = view(geo.vertices().find(builtin.is_constrained))
    ap_view = view(geo.vertices().find(builtin.aim_position))

    # absolute twist angle this frame; x<0 end one way, x>0 end the other
    theta = ANGULAR_VEL * info.dt() * info.frame()
    th = np.where(rest[:, 0] < 0, theta, -theta)
    c, s = np.cos(th), np.sin(th)
    aim = rest.copy()
    aim[:, 1] = rest[:, 1] * c - rest[:, 2] * s
    aim[:, 2] = rest[:, 1] * s + rest[:, 2] * c

    ap_view[:] = aim.reshape(-1, 3, 1)  # Vector3 attribute view is (N,3,1)
    ic_view[:] = END_MASK.astype(np.int32)


animator.insert(bar_obj, twist)

ground_obj = scene.objects().create("ground")
ground_obj.geometries().create(ground(-1.0))

world.init(scene)

# --------------------------------------------------------------------------
if HEADLESS:
    import time
    frame_ms = []
    bar_geo_id = bar_obj.geometries().ids()[0]
    end_vid = int(np.nonzero(END_MASK)[0][0])  # first constrained end vertex
    for _ in range(N_FRAMES):
        t0 = time.perf_counter()
        world.advance()
        world.retrieve()
        frame_ms.append((time.perf_counter() - t0) * 1e3)
        if world.frame() % 10 == 0:
            slot, _ = scene.geometries().find(bar_geo_id)
            pos = np.asarray(
                slot.geometry().vertices().find("position").view()).reshape(-1, 3)
            c = pos.mean(axis=0)
            e = pos[end_vid]
            print(f"track f{world.frame()} centroid=({c[0]:.4f},{c[1]:.4f},{c[2]:.4f})"
                  f" end_vid{end_vid}=({e[0]:.4f},{e[1]:.4f},{e[2]:.4f})",
                  flush=True)
    import statistics
    print(f"TOTAL frames={N_FRAMES} mean={statistics.mean(frame_ms):.1f}ms "
          f"median={statistics.median(frame_ms):.1f}ms", flush=True)
    Timer.report()
else:
    import polyscope as ps
    from polyscope import imgui
    from uipc.gui import SceneGUI

    sgui = SceneGUI(scene)
    ps.init()
    tri_surf, _, _ = sgui.register()
    tri_surf.set_edge_width(1.0)
    ps.set_ground_plane_height(-1.0)
    ps.set_ground_plane_height_mode('manual')

    run = False


    def on_update():
        global run
        imgui.Text(f'frame: {world.frame()}')
        if imgui.Button('stop' if run else 'run'):
            run = not run
        if run:
            world.advance()
            world.retrieve()
            sgui.update()


    ps.set_user_callback(on_update)
    ps.show()

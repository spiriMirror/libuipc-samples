"""Example 89 -- single FEM bunny on the ground with the MAS preconditioner.

Cross-project MAS parity scene (vs Stiff-GIPC set_case7):
  - FEM bunny2.msh, scale 0.2, translate (0, -0.65, 0), SNK E=1e5, nu=0.49,
    rho=1000 (Stiff's SNK parametrization)
  - MAS preconditioner via config linear_system/fem_preconditioner = "mas"
    (Stiff P_type=1); auto-partitions all FEM geometries internally
  - floor y=-1, dt=0.01, g=-9.8, mu=0.4, kappa=1e8 (= Stiff raw Kappa 1e4
    after libuipc's dt^2 scaling), d_hat_relative=1e-3,
    velocity_tol_relative=1e-2, eps_velocity_relative=1e-2, tol_rate=1e-4,
    semi-implicit early exit on (Stiff's beta rule)

Usage:
  python main.py                  # GUI with run/stop
  python main.py --headless [N]   # N frames (default 100), structured summary

Set WB_LOG=Info for per-solve logs. Set UIPC_BENCHMARK_TIMERS=1 only for a
separate synchronized stage diagnostic.
"""
import os, sys, time
from pathlib import Path

import numpy as np
import uipc
from uipc import Logger, Timer, Transform, Vector3, view
from uipc.core import Engine, World, Scene
from uipc.geometry import (SimplicialComplexIO, flip_inward_triangles, ground,
                           label_surface, label_triangle_orient)
from uipc.constitution import ElasticModuli, StableNeoHookean
from uipc.unit import MPa

from asset_dir import AssetDir

sys.path.append(str(Path(__file__).resolve().parents[1]))
from benchmark_utils import (configure_benchmark_timers, emit_benchmark_result,
                             report_timers_if_enabled, snapshot_frame_stats)

HEADLESS = "--headless" in sys.argv
_positional = [a for a in sys.argv[1:] if not a.startswith("--")]
N_FRAMES = int(_positional[0]) if _positional else 100

configure_benchmark_timers()
Logger.set_level(getattr(Logger.Level, os.environ.get("WB_LOG", "Warn")))

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.01
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["d_hat_relative"] = 1e-3
config["newton"]["velocity_tol_relative"] = 1e-2
config["contact"]["eps_velocity_relative"] = 1e-2
config["linear_system"]["tol_rate"] = 1e-4
config["newton"]["transrate_tol"] = 10
config["newton"]["semi_implicit"]["enable"] = 1
config["newton"]["semi_implicit"]["beta_tol"] = 1e-2
config["newton"]["semi_implicit"]["K_min"] = 6
# NO_MAS=1 runs the diagonal-preconditioner baseline instead
if os.environ.get("NO_MAS") != "1":
    config["linear_system"]["fem_preconditioner"] = "mas"
# NO_GRAPH=1 disables PCG graph replay (plain launches) for A/B benchmarking
if os.environ.get("NO_GRAPH") == "1":
    config["linear_system"]["use_cuda_graph"] = 0
scene = Scene(config)

# kappa: Stiff applies raw Kappa=1e4; libuipc scales the barrier by dt^2
scene.contact_tabular().default_model(0.4, 1e8)
default_contact = scene.contact_tabular().default_element()

# --- FEM bunny -------------------------------------------------------------
t = Transform.Identity()
t.translate(Vector3.Values([0.0, -0.65, 0.0]))
t.scale(0.2)
bunny_mesh = SimplicialComplexIO(t).read(f"{AssetDir.tetmesh_path()}/bunny2.msh")
label_surface(bunny_mesh)
label_triangle_orient(bunny_mesh)
bunny_mesh = flip_inward_triangles(bunny_mesh)

StableNeoHookean().apply_to(
    bunny_mesh, ElasticModuli.youngs_poisson(1e5, 0.49), mass_density=1e3)
default_contact.apply_to(bunny_mesh)
bunny = scene.objects().create("bunny")
bunny.geometries().create(bunny_mesh)

# --- floor (Stiff's built-in floor is y=-1) ---------------------------------
ground_obj = scene.objects().create("ground")
ground_obj.geometries().create(ground(-1.0))

world.init(scene)

# --------------------------------------------------------------------------
if HEADLESS:
    bunny_geo_id = bunny.geometries().ids()[0]
    frame_ms = []
    frame_stats = []
    final_centroid = None
    for _ in range(N_FRAMES):
        t0 = time.perf_counter()
        world.advance()
        world.retrieve()
        frame_ms.append((time.perf_counter() - t0) * 1e3)
        frame_stats.append(snapshot_frame_stats(engine))
        if world.frame() % 10 == 0:
            slot, _ = scene.geometries().find(bunny_geo_id)
            pos = np.asarray(
                slot.geometry().vertices().find("position").view()).reshape(-1, 3)
            c = pos.mean(axis=0)
            final_centroid = c
            print(f"track f{world.frame()} centroid=({c[0]:.4f},{c[1]:.4f},{c[2]:.4f})",
                  flush=True)
    if final_centroid is None:
        slot, _ = scene.geometries().find(bunny_geo_id)
        pos = np.asarray(
            slot.geometry().vertices().find("position").view()).reshape(-1, 3)
        final_centroid = pos.mean(axis=0)
    emit_benchmark_result(
        frame_ms,
        frame_stats,
        observables={
            "final_frame": int(world.frame()),
            "bunny_centroid": [float(value) for value in final_centroid],
        },
    )
    report_timers_if_enabled()
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

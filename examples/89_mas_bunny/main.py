"""Example 89 -- single FEM bunny on the ground with the MAS preconditioner.

Cross-project MAS parity scene (vs Stiff-GIPC set_case7):
  - FEM bunny2.msh, scale 0.2, translate (0, -0.65, 0), SNK E=1e7, nu=0.49,
    rho=1000 (Stiff's SNK parametrization)
  - MAS preconditioner via mesh_partition(bunny, 16) (Stiff P_type=1)
  - floor y=-1, dt=0.01, g=-9.8, mu=0.4, kappa=1e8 (= Stiff raw Kappa 1e4
    after libuipc's dt^2 scaling), d_hat_relative=1e-3,
    velocity_tol_relative=1e-2, eps_velocity_relative=1e-2, tol_rate=1e-4,
    semi-implicit early exit on (Stiff's beta rule)

Usage:
  python main.py                  # GUI with run/stop
  python main.py --headless [N]   # N frames (default 100), logs per-solve
                                  # PCG iteration counts + a summary line
"""
import os, sys

import numpy as np
import uipc
from uipc import Logger, Timer, Transform, Vector3, view
from uipc.core import Engine, World, Scene
from uipc.geometry import (SimplicialComplexIO, flip_inward_triangles, ground,
                           label_surface, label_triangle_orient, mesh_partition)
from uipc.constitution import ElasticModuli, StableNeoHookean
from uipc.unit import MPa

from asset_dir import AssetDir

HEADLESS = "--headless" in sys.argv
_positional = [a for a in sys.argv[1:] if not a.startswith("--")]
N_FRAMES = int(_positional[0]) if _positional else 100

# headless mode needs the per-solve iteration lines
Logger.set_level(Logger.Level.Info if HEADLESS else Logger.Level.Warn)

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
config["newton"]["min_iter"] = 6
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
    bunny_mesh, ElasticModuli.youngs_poisson(1e7, 0.49), mass_density=1e3)
# NO_MAS=1 runs the diagonal-preconditioner baseline instead
if os.environ.get("NO_MAS") != "1":
    mesh_partition(bunny_mesh, 16)  # activates the MAS preconditioner
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
    for _ in range(N_FRAMES):
        world.advance()
        world.retrieve()
        if world.frame() % 10 == 0:
            slot, _ = scene.geometries().find(bunny_geo_id)
            pos = np.asarray(
                slot.geometry().vertices().find("position").view()).reshape(-1, 3)
            c = pos.mean(axis=0)
            print(f"track f{world.frame()} centroid=({c[0]:.4f},{c[1]:.4f},{c[2]:.4f})",
                  flush=True)
    print(f"BENCH DONE frames={N_FRAMES}", flush=True)
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

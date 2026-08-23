"""Example 90 -- Stiff-GIPC set_case1: alternating ABD/FEM cube stack.

Port of Stiff-GIPC gl_main.cu set_case1: a 4x4 grid of cube layers, 8 layers
interleaved FEM (E=1e4) / ABD (E=1e5 -> the ABD constitution mapping uses
100 MPa, see example 88 notes), falling onto the floor y=-1.

Stiff parameters: cube = stiff_cube.msh (0.4-size, y-center 0.3), scale 0.4,
layer spacing 0.4, grid spacing 0.2 centered at origin; FEM layers at
y = 0.8-0.4k, ABD layers at y = 0.6-0.4k (k=0..3); E_FEM=1e4, nu=0.49,
rho=1000; P_type=0 (diagonal preconditioner in Stiff).

Global parameters follow the established Stiff-GIPC alignment (see 88/89):
dt=0.01, g=-9.8, mu=0.2, kappa=1e8 (= Stiff raw 1e4 after dt^2 scaling),
d_hat_relative=1e-3, velocity_tol_relative=1e-2, eps_velocity_relative=1e-2,
tol_rate=1e-4, semi-implicit on with K_min=6.

Usage:
  python main.py                  # GUI with run/stop
  python main.py --headless [N]   # N frames (default 100)
"""
import os, sys

import numpy as np
import uipc
from uipc import Logger, Timer, Transform, Vector3, view
from uipc.core import Engine, World, Scene
from uipc.geometry import (SimplicialComplexIO, flip_inward_triangles, ground,
                           label_surface, label_triangle_orient)
from uipc.constitution import (AffineBodyConstitution, ElasticModuli,
                               StableNeoHookean)
from uipc.unit import MPa

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
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["d_hat_relative"] = 1e-3
config["newton"]["velocity_tol_relative"] = 1e-2
config["contact"]["eps_velocity_relative"] = 1e-2
config["linear_system"]["tol_rate"] = 1e-4
config["newton"]["transrate_tol"] = 10
config["newton"]["semi_implicit"]["enable"] = 1
config["newton"]["semi_implicit"]["beta_tol"] = 1e-2
config["newton"]["semi_implicit"]["K_min"] = 6
scene = Scene(config)

# kappa: Stiff applies raw Kappa=1e4; libuipc scales the barrier by dt^2
scene.contact_tabular().default_model(0.2, 1e8)
default_contact = scene.contact_tabular().default_element()

abd = AffineBodyConstitution()
snh = StableNeoHookean()

tetmesh_path = AssetDir.tetmesh_path()

DIST = 0.2
SCALE = 0.4
COUNT = 4
COUNT_Y = 4
FEM_TOP = 0.8   # Stiff fem_height = -0.8 with negated offsets
ABD_TOP = 0.6   # Stiff abd_height = -0.6 with negated offsets


def make_cube(x, y, z, kind):
    t = Transform.Identity()
    t.translate(Vector3.Values([x, y, z]))
    t.scale(SCALE)
    mesh = SimplicialComplexIO(t).read(f"{tetmesh_path}/stiff_cube.msh")
    label_surface(mesh)
    label_triangle_orient(mesh)
    mesh = flip_inward_triangles(mesh)
    if(kind == "abd"):
        abd.apply_to(mesh, 100 * MPa, 1e3)
    else:
        snh.apply_to(mesh, ElasticModuli.youngs_poisson(1e4, 0.49), 1e3)
    default_contact.apply_to(mesh)
    obj = scene.objects().create(f"{kind}_cube")
    obj.geometries().create(mesh)
    return obj


fem_obj = None
for k in range(COUNT_Y):
    for i in range(COUNT):
        for j in range(COUNT):
            x = i * DIST - DIST * (COUNT - 1) / 2.0
            z = j * DIST - DIST * (COUNT - 1) / 2.0
            fem_obj = make_cube(x, FEM_TOP - 2 * DIST * k, z, "fem")
            make_cube(x, ABD_TOP - 2 * DIST * k, z, "abd")

ground_obj = scene.objects().create("ground")
ground_obj.geometries().create(ground(-1.0))

world.init(scene)

# --------------------------------------------------------------------------
if HEADLESS:
    fem_geo_id = fem_obj.geometries().ids()[0]
    import time
    frame_ms = []
    for _ in range(N_FRAMES):
        t0 = time.perf_counter()
        world.advance()
        world.retrieve()
        frame_ms.append((time.perf_counter() - t0) * 1e3)
        if world.frame() % 10 == 0:
            slot, _ = scene.geometries().find(fem_geo_id)
            pos = np.asarray(
                slot.geometry().vertices().find("position").view()).reshape(-1, 3)
            c = pos.mean(axis=0)
            print(f"track f{world.frame()} centroid=({c[0]:.4f},{c[1]:.4f},{c[2]:.4f})",
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

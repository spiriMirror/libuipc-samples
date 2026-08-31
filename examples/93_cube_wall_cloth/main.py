"""Example 93 -- Stiff-GIPC set_case6: cube wall falling onto a pinned cloth.

Port of Stiff-GIPC gl_main.cu set_case6: an 8x8x15 wall of interleaved ABD
cube layers (Stiff E=1e6 and 5e4, but Stiff forces ABD Young's to 1e7
internally, so both groups map to the same 100 MPa ABD constitution here;
see example 88 notes) rises from y~1.05 and falls onto a cloth pinned at
its |x|>1.5 sides (cloth_high.obj, scale 1.5 then translate y+0.35 in
Stiff's transform order, i.e. v' = 1.5*(v + (0,0.35,0))).

Stiff parameters retained for the cube wall are stiff_cube.msh, scale 0.3,
spacing 0.15, relative_dhat=1e-3, and P_type=1 (MAS). The cloth uses stretch
E=5e4, shear E=1e1, nu=0.49, one-sided thickness r=1e-3, rho=200,
case-specific strain_rate=10000, and bending E=3e4.

Global parameters follow the established Stiff-GIPC alignment (see 88/89).

Usage:
  python main.py                  # GUI with run/stop
  python main.py --headless [N]   # N frames (default 100)
"""
import os, sys

import numpy as np
import uipc
from uipc import Logger, Timer, Transform, Vector3, builtin, view
from uipc.core import Engine, World, Scene
from uipc.geometry import (SimplicialComplexIO, flip_inward_triangles, ground,
                           label_surface, label_triangle_orient)
from uipc.constitution import (AffineBodyConstitution, ElasticModuli2D,
                               StrainLimitingBaraffWitkinShell,
                               DiscreteShellBending)
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
# Stiff P_type=1 -> MAS
config["linear_system"]["fem_preconditioner"] = "mas"
scene = Scene(config)

scene.contact_tabular().default_model(0.2, 1e8)
default_contact = scene.contact_tabular().default_element()

abd = AffineBodyConstitution()
slbws = StrainLimitingBaraffWitkinShell()
dsb = DiscreteShellBending()

tetmesh_path = AssetDir.tetmesh_path()

# --- ABD cube wall (Stiff case6: two interleaved layer sets) ---------------
SCALE = 0.3
DIST = SCALE / 2
COUNT = 8
COUNT_Y = 15
FEM_BASE = 1.2           # global_offset(1.0) + 1 - 0.8
ABD_BASE = FEM_BASE - DIST  # 1.05


def make_cube(x, y, z, name):
    t = Transform.Identity()
    t.translate(Vector3.Values([x, y, z]))
    t.scale(SCALE)
    mesh = SimplicialComplexIO(t).read(f"{tetmesh_path}/stiff_cube.msh")
    label_surface(mesh)
    label_triangle_orient(mesh)
    mesh = flip_inward_triangles(mesh)
    abd.apply_to(mesh, 100 * MPa, 1e3)
    default_contact.apply_to(mesh)
    obj = scene.objects().create(name)
    obj.geometries().create(mesh)
    return obj


last_cube = None
for k in range(COUNT_Y):
    for i in range(COUNT):
        for j in range(COUNT):
            x = i * DIST - DIST * (COUNT - 1) / 2.0
            z = j * DIST - DIST * (COUNT - 1) / 2.0
            last_cube = make_cube(x, ABD_BASE + 2 * DIST * k, z, "abd_hard")
            last_cube = make_cube(x, FEM_BASE + 2 * DIST * k, z, "abd_soft")

# --- cloth (pinned at |x| > 1.5 sides) --------------------------------------
t = Transform.Identity()
t.scale(1.5)
t.translate(Vector3.Values([0.0, 0.35, 0.0]))  # Stiff order: v' = 1.5*(v+o)
cloth_mesh = SimplicialComplexIO(t).read(
    f"{AssetDir.trimesh_path()}/cloth_high.obj")
label_surface(cloth_mesh)

cloth_stretch = ElasticModuli2D.youngs_poisson(5e4, 0.49)
cloth_shear = ElasticModuli2D.youngs_poisson(1e1, 0.49)
slbws.apply_to(cloth_mesh,
               stretch_moduli=cloth_stretch,
               shear_moduli=cloth_shear,
               mass_density=200,
               thickness=0.001,
               strain_rate=10000)
dsb.apply_to(cloth_mesh, 3e4, 0.49)

pos = np.asarray(cloth_mesh.positions().view()).reshape(-1, 3)
eps = 1e-4
is_fixed = cloth_mesh.vertices().find(builtin.is_fixed)
if is_fixed is None:
    is_fixed = cloth_mesh.vertices().create(builtin.is_fixed, 0)
is_fixed_view = view(is_fixed)
is_fixed_view[:] = (np.abs(pos[:, 0]) > 1.5 - eps).astype(np.int32)

default_contact.apply_to(cloth_mesh)
cloth_obj = scene.objects().create("cloth")
cloth_obj.geometries().create(cloth_mesh)

ground_obj = scene.objects().create("ground")
ground_obj.geometries().create(ground(-1.0))

world.init(scene)

# --------------------------------------------------------------------------
if HEADLESS:
    import time
    frame_ms = []
    cloth_geo_id = cloth_obj.geometries().ids()[0]
    for _ in range(N_FRAMES):
        t0 = time.perf_counter()
        world.advance()
        world.retrieve()
        frame_ms.append((time.perf_counter() - t0) * 1e3)
        if world.frame() % 10 == 0:
            slot, _ = scene.geometries().find(cloth_geo_id)
            p = np.asarray(
                slot.geometry().vertices().find("position").view()).reshape(-1, 3)
            print(f"track f{world.frame()} cloth_min_y={p[:, 1].min():.4f}",
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

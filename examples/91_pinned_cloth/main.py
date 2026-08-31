"""Example 91 -- Stiff-GIPC set_case4: pinned cloth (hanging by two corners).

Port of Stiff-GIPC gl_main.cu set_case4: cloth_high.obj rotated 90 deg about
the X axis (hanging vertically), scaled 0.6, lifted to y=+1; the two top
corners (max-y edge vertices at min/max x) are pinned. Stiff runs this case
with P_type=0 (diagonal preconditioner), so MAS is left off here.

The cloth uses the shared example 88 material preset: stretch E=5e4,
shear E=1e1, nu=0.49, one-sided thickness r=1e-3, rho=200,
strain_rate=100, and bending E=3e4. Transform order matches Stiff exactly:
rotate, then scale, then translate.

Global parameters follow the established Stiff-GIPC alignment (see 88/89).

Usage:
  python main.py                  # GUI with run/stop
  python main.py --headless [N]   # N frames (default 100)
"""
import os, sys

import numpy as np
import uipc
from uipc import Logger, Timer, Transform, Vector3, AngleAxis, builtin, view
from uipc.core import Engine, World, Scene
from uipc.geometry import (SimplicialComplexIO, ground, label_surface)
from uipc.constitution import (ElasticModuli2D, StrainLimitingBaraffWitkinShell,
                               DiscreteShellBending)

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

scene.contact_tabular().default_model(0.2, 1e8)
default_contact = scene.contact_tabular().default_element()

slbws = StrainLimitingBaraffWitkinShell()
dsb = DiscreteShellBending()

# --- cloth: rotate +90 deg about X, scale 0.6, translate y+1 (Stiff order) --
t = Transform.Identity()
t.translate(Vector3.Values([0.0, 1.0, 0.0]))
t.scale(0.6)
t.rotate(AngleAxis(np.pi / 2, Vector3.UnitX()))
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
               strain_rate=100)
dsb.apply_to(cloth_mesh, 3e4, 0.49)

# pin the two top corners: max-y vertices that sit at min/max x (Stiff rule)
pos = np.asarray(cloth_mesh.positions().view()).reshape(-1, 3)
eps = 1e-4
max_y, min_x, max_x = pos[:, 1].max(), pos[:, 0].min(), pos[:, 0].max()
is_fixed = cloth_mesh.vertices().find(builtin.is_fixed)
if is_fixed is None:
    is_fixed = cloth_mesh.vertices().create(builtin.is_fixed, 0)
is_fixed_view = view(is_fixed)
pinned = 0
for i in range(pos.shape[0]):
    if pos[i, 1] > max_y - eps and (pos[i, 0] < min_x + eps
                                    or pos[i, 0] > max_x - eps):
        is_fixed_view[i] = 1
        pinned += 1
print(f"fixed vertex num: {pinned}")

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
    for _ in range(N_FRAMES):
        t0 = time.perf_counter()
        world.advance()
        world.retrieve()
        frame_ms.append((time.perf_counter() - t0) * 1e3)
        if world.frame() % 10 == 0:
            print(f"frame {world.frame()}: {frame_ms[-1]:.1f}ms", flush=True)
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

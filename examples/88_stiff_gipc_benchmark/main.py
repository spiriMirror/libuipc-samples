"""Example 88 -- Stiff-GIPC set_case2 benchmark scene, with GUI visualization.

Scene (Stiff-GIPC gl_main.cu set_case2):
  - ABD  bunny: bunny2.msh, scale 0.2, translate (0, +0.5, 0), rho=1000
    (Stiff forces ABD Young's to 1e7; the ABD shape stiffness is the global
    parms.kappa=1e8, matched here by AffineBodyConstitution(100 MPa))
  - FEM  bunny: bunny2.msh, scale 0.2, translate (0, -0.65, 0),
    E=1e4, nu=0.49, rho=1000 (StableNeoHookean ~ Stiff's SNK parametrization)
  - cloth: cloth_high.obj (4225 verts, x,z in [-1,1] at y=0), t=1e-3, rho=200,
    cloth E=5e4, nu=0.49, strain_rate=100; bending matched by value:
    Stiff bendStiff = E_bend*t^3/(24*(1-nu^2)) = 5.48e-3 with E_bend=1e8
    -> libuipc DiscreteShellBending with E=5e7 gives E*t^3/(12*(1-nu^2)) = 5.48e-3
  - ground: floor y=-1 (Stiff's 4 side "walls" at x=-1/z=-1 are self-canceling
    duplicate half-planes and are omitted)

Global parameters (Stiff-GIPC Assets/scene/parameterSetting.txt):
  dt=0.01, gravity=(0,-9.8,0), friction mu=0.2, relative_dhat=1e-3,
  Newton threshold 1e-2*diag*dt, PCG tol_rate=1e-4, friction slip
  1e-2*diag*dt per step; kappa pinned: Stiff raw 1e4 == libuipc 1e8/dt^2.

Usage:
  python main.py                  # GUI: run/stop button, live per-frame ms
  python main.py --headless [N]   # benchmark: N frames (default 250), no GUI

Both modes write tracked-body centroids (ABD bunny center, FEM bunny
centroid) to output/examples/88_stiff_gipc_benchmark/traj.csv and print a
timing summary when the run completes, for cross-project comparison.

Env knobs: WB_TIMER=1 enables Timer reports, WB_LOG=Info sets log level.
"""
import os, sys, time
import statistics

import numpy as np
import uipc
from uipc import Logger, Timer, Transform, Vector3, view
from uipc.core import Engine, World, Scene
from uipc.geometry import SimplicialComplexIO, label_surface, label_triangle_orient, flip_inward_triangles, ground
from uipc.constitution import (AffineBodyConstitution, StableNeoHookean, ElasticModuli,
                               ElasticModuli2D, StrainLimitingBaraffWitkinShell,
                               DiscreteShellBending)
from uipc.unit import MPa

from asset_dir import AssetDir

# --------------------------------------------------------------------------
# args & logging
# --------------------------------------------------------------------------
HEADLESS = "--headless" in sys.argv
_positional = [a for a in sys.argv[1:] if not a.startswith("--")]
N_FRAMES = int(_positional[0]) if _positional else 250

Timer.enable_all() if os.environ.get("WB_TIMER", "0") == "1" else None
Logger.set_level(getattr(Logger.Level, os.environ.get("WB_LOG", "Warn")))

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

# --------------------------------------------------------------------------
# scene config (Stiff-GIPC-aligned; do not retune -- benchmark comparability)
# --------------------------------------------------------------------------
config = Scene.default_config()
config["dt"] = 0.01
config["gravity"] = [[0.0], [-9.8], [0.0]]  # Stiff uses -9.8
config["contact"]["d_hat_relative"] = 1e-3
config["newton"]["velocity_tol_relative"] = 1e-2
config["contact"]["eps_velocity_relative"] = 1e-2
config["linear_system"]["tol_rate"] = 1e-4
config["newton"]["transrate_tol"] = 10
# Stiff-GIPC semi-implicit early exit (their newton loop: beta=(1-alpha)*beta
# from iter>=Kmin=6, exit when beta<=1e-2) -- absorbs hard pile-up frames
config["newton"]["semi_implicit"]["enable"] = 1
config["newton"]["semi_implicit"]["beta_tol"] = 1e-2
config["newton"]["min_iter"] = 6
scene = Scene(config)

abd = AffineBodyConstitution()
snh = StableNeoHookean()
slbws = StrainLimitingBaraffWitkinShell()
dsb = DiscreteShellBending()

# kappa: libuipc scales the barrier by dt^2 in the incremental potential while
# Stiff-GIPC applies Kappa raw; equivalent: 1e4 / 1e-4 = 1e8
scene.contact_tabular().default_model(0.2, 1e8)
default_contact = scene.contact_tabular().default_element()


def process_tet(sc):
    label_surface(sc)
    label_triangle_orient(sc)
    return flip_inward_triangles(sc)


def vec3(x, y, z):
    v = Vector3.Zero()
    v[0], v[1], v[2] = x, y, z
    return v


def read_tet_transformed(path, offset, scale):
    # bake scale+translation into the vertices at read time (like the other
    # samples); using the instance transform's scale would instead create an
    # 80% compression strain w.r.t. the unscaled rest shape
    t = Transform.Identity()
    t.translate(offset)
    t.scale(scale)
    return SimplicialComplexIO(t).read(path)


io = SimplicialComplexIO()
tetmesh_path = AssetDir.tetmesh_path()
trimesh_path = AssetDir.trimesh_path()

# --- ABD bunny (load first, like Stiff-GIPC's ABD-before-FEM rule) ---------
abd_obj = scene.objects().create("abd_bunny")
abd_mesh = process_tet(read_tet_transformed(f"{tetmesh_path}/bunny2.msh",
                                            vec3(0.0, 0.5, 0.0), 0.2))
abd.apply_to(abd_mesh, 100 * MPa, 1e3)
default_contact.apply_to(abd_mesh)
abd_obj.geometries().create(abd_mesh)

# --- FEM bunny -------------------------------------------------------------
fem_obj = scene.objects().create("fem_bunny")
fem_mesh = process_tet(read_tet_transformed(f"{tetmesh_path}/bunny2.msh",
                                            vec3(0.0, -0.65, 0.0), 0.2))
snh.apply_to(fem_mesh, ElasticModuli.youngs_poisson(1e4, 0.49), 1e3)
default_contact.apply_to(fem_mesh)
fem_obj.geometries().create(fem_mesh)

# --- cloth -----------------------------------------------------------------
cloth_obj = scene.objects().create("cloth")
cloth_mesh = io.read(f"{trimesh_path}/cloth_high.obj")
label_surface(cloth_mesh)
cloth_moduli = ElasticModuli2D.youngs_poisson(5e4, 0.49)
slbws.apply_to(cloth_mesh,
               stretch_moduli=cloth_moduli,
               shear_moduli=cloth_moduli,
               mass_density=200,
               thickness=0.001,
               strain_rate=100)
dsb.apply_to(cloth_mesh, 5e7, 0.49)  # kappa_bend = 5.48e-3, matches Stiff
cloth_obj.geometries().create(cloth_mesh)

# --- ground (floor y=-1) ----------------------------------------------------
ground_obj = scene.objects().create("ground")
ground_obj.geometries().create(ground(-1.0))

world.init(scene)

# --------------------------------------------------------------------------
# tracked points: ABD bunny center (via its transform), FEM bunny centroid
# --------------------------------------------------------------------------
abd_geo_id = abd_obj.geometries().ids()[0]
fem_geo_id = fem_obj.geometries().ids()[0]
abd_local_center = np.asarray(
    abd_mesh.vertices().find("position").view()).reshape(-1, 3).mean(axis=0)
fem_local_rest = np.asarray(
    fem_mesh.vertices().find("position").view()).reshape(-1, 3)


def abd_world_center():
    slot, _ = scene.geometries().find(abd_geo_id)
    g = slot.geometry()
    M = np.asarray(view(g.transforms())[0]).reshape(4, 4)
    return M[:3, :3] @ abd_local_center + M[:3, 3]


def fem_world_center():
    slot, _ = scene.geometries().find(fem_geo_id)
    g = slot.geometry()
    pos = np.asarray(g.vertices().find("position").view()).reshape(-1, 3)
    return pos.mean(axis=0)


traj = []
frame_ms = []


def step_frame():
    """Advance one frame, record timing + centroids. Returns elapsed ms."""
    t0 = time.perf_counter()
    world.advance()
    world.retrieve()
    dt_ms = (time.perf_counter() - t0) * 1e3
    frame_ms.append(dt_ms)
    c_abd = abd_world_center()
    c_fem = fem_world_center()
    traj.append([world.frame(), *c_abd, *c_fem])
    return dt_ms


def report_and_save():
    np.savetxt(f"{workspace}/traj.csv",
               np.asarray(traj),
               delimiter=",",
               header="frame,abd_cx,abd_cy,abd_cz,fem_cx,fem_cy,fem_cz",
               comments="")
    print(f"TOTAL frames={len(frame_ms)} mean={statistics.mean(frame_ms):.1f}ms "
          f"median={statistics.median(frame_ms):.1f}ms")
    print(f"traj saved to {workspace}/traj.csv")
    Timer.report()


# --------------------------------------------------------------------------
# entry: headless benchmark or GUI
# --------------------------------------------------------------------------
if HEADLESS:
    for i in range(N_FRAMES):
        dt_ms = step_frame()
        c = traj[-1]
        print(f"frame {i}: {dt_ms:.1f}ms"
              f"  abd=({c[1]:.3f},{c[2]:.3f},{c[3]:.3f})"
              f"  fem=({c[4]:.3f},{c[5]:.3f},{c[6]:.3f})", flush=True)
    report_and_save()
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
    finished = False
    last_ms = 0.0


    def on_update():
        global run, finished, last_ms

        imgui.Text(f'frame: {world.frame()} / {N_FRAMES}')
        imgui.Text(f'last frame: {last_ms:.1f} ms')
        if frame_ms:
            imgui.Text(f'mean: {statistics.mean(frame_ms):.1f} ms')

        if imgui.Button('stop' if run else 'run'):
            run = not run

        if world.frame() >= N_FRAMES and not finished:
            run = False
            finished = True
            report_and_save()

        if finished:
            imgui.Text('Benchmark finished!')
        elif run:
            last_ms = step_frame()
            sgui.update()


    ps.set_user_callback(on_update)
    ps.show()

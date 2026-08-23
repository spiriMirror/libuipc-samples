"""Example 88 -- two FEM bunnies (MAS vs diagonal preconditioner) + cloth.

Derived from the Stiff-GIPC set_case2 benchmark: the original ABD bunny is
replaced by a second FEM bunny, so the scene runs two identical FEM bunnies
side by side -- one preconditioned by MAS, one by the diagonal fallback.
This exercises the mixed-partition path: MAS activates scene-wide, the
unpartitioned bunny (and the cloth) automatically get the internal
block-Jacobi fallback inside FEMMASPreconditioner.

Scene:
  - FEM bunny "mas":  bunny2.msh, scale 0.2, translate (0, +0.5, 0),
    E=1e7, nu=0.49, rho=1000 (StableNeoHookean ~ Stiff's SNK parametrization),
    mesh_partition(mesh, 16) -> MAS preconditioner on its vertices
  - FEM bunny "diag": same mesh/params at (0, -0.65, 0), no mesh_part ->
    diagonal (block-Jacobi) fallback inside the same global PCG
  - cloth: cloth_high.obj (4225 verts, x,z in [-1,1] at y=0), t=1e-3, rho=200,
    stretch E=1e4, shear E=1e3, nu=0.40, strain_rate=100;
    bending matched by value: Stiff bendStiff = E_bend*t^3/(24*(1-nu^2))
    = 5.48e-3 with E_bend=1e8
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

Both modes write the tracked-body centroids (diag bunny, MAS bunny) to
output/examples/88_stiff_gipc_benchmark/traj.csv and print a timing summary
when the run completes, for cross-project comparison.

Env knobs: WB_TIMER=1 enables Timer reports, WB_LOG=Info sets log level,
NO_MAS=1 turns all mesh_partition off (all-diagonal A/B baseline),
ALL_MAS=1 additionally partitions the lower bunny (full-FEM MAS coverage).
"""
import os, sys, time
import statistics

import numpy as np
import uipc
from uipc import Logger, Timer, Transform, Vector3, view
from uipc.core import Engine, World, Scene
from uipc.geometry import (SimplicialComplexIO, label_surface, label_triangle_orient,
                           flip_inward_triangles, ground, mesh_partition)
from uipc.constitution import (StableNeoHookean, ElasticModuli,
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


def make_fem_bunny(offset, use_mas):
    mesh = process_tet(read_tet_transformed(f"{tetmesh_path}/bunny2.msh", offset, 0.2))
    snh.apply_to(mesh, ElasticModuli.youngs_poisson(1e7, 0.49), 1e3)
    if use_mas:
        # writes the mesh_part vertex attribute -> FEMMASPreconditioner picks
        # these vertices up; everything unpartitioned gets the diagonal
        # fallback inside the same global PCG
        mesh_partition(mesh, 16)
    default_contact.apply_to(mesh)
    obj = scene.objects().create("fem_bunny_mas" if use_mas else "fem_bunny_diag")
    obj.geometries().create(mesh)
    return obj


# --- FEM bunny (MAS preconditioner), upper ----------------------------------
# NO_MAS=1 disables mesh_partition (all-diagonal baseline for A/B);
# ALL_MAS=1 additionally partitions the lower bunny (full-FEM MAS coverage)
_no_mas = os.environ.get("NO_MAS") == "1"
_all_mas = os.environ.get("ALL_MAS") == "1"
mas_obj = make_fem_bunny(vec3(0.0, 0.5, 0.0), use_mas=(not _no_mas))

# --- FEM bunny (diagonal preconditioner unless ALL_MAS=1), lower ------------
diag_obj = make_fem_bunny(vec3(0.0, -0.65, 0.0), use_mas=(_all_mas and not _no_mas))

# --- cloth -----------------------------------------------------------------
cloth_obj = scene.objects().create("cloth")
cloth_mesh = io.read(f"{trimesh_path}/cloth_high.obj")
label_surface(cloth_mesh)
cloth_stretch = ElasticModuli2D.youngs_poisson(1e4, 0.40)
cloth_shear = ElasticModuli2D.youngs_poisson(1e3, 0.40)
slbws.apply_to(cloth_mesh,
               stretch_moduli=cloth_stretch,
               shear_moduli=cloth_shear,
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
# tracked bodies: per-bunny world centroids
# --------------------------------------------------------------------------
diag_geo_id = diag_obj.geometries().ids()[0]
mas_geo_id = mas_obj.geometries().ids()[0]


def world_centroid(geo_id):
    slot, _ = scene.geometries().find(geo_id)
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
    c_diag = world_centroid(diag_geo_id)
    c_mas = world_centroid(mas_geo_id)
    traj.append([world.frame(), *c_diag, *c_mas])
    return dt_ms


def report_and_save():
    np.savetxt(f"{workspace}/traj.csv",
               np.asarray(traj),
               delimiter=",",
               header="frame,diag_cx,diag_cy,diag_cz,mas_cx,mas_cy,mas_cz",
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
              f"  diag=({c[1]:.3f},{c[2]:.3f},{c[3]:.3f})"
              f"  mas=({c[4]:.3f},{c[5]:.3f},{c[6]:.3f})", flush=True)
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

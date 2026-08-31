"""Wrecking balls sample with GUI and reproducible headless benchmark modes.

Usage:
  python main.py                  # interactive GUI
  python main.py --headless 120   # pure-ABD contact benchmark
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path

import uipc
from uipc import view
from uipc import Vector3, Transform, Logger, Quaternion, AngleAxis, Timer
from uipc.core import Engine, World, Scene
from uipc.geometry import SimplicialComplex, SimplicialComplexIO, ground, label_surface, label_triangle_orient, flip_inward_triangles
from uipc.constitution import AffineBodyConstitution
from uipc.unit import MPa, GPa

from asset_dir import AssetDir

sys.path.append(str(Path(__file__).resolve().parents[1]))
from benchmark_utils import (configure_benchmark_timers, emit_benchmark_result,
                             report_timers_if_enabled, snapshot_frame_stats)


HEADLESS = "--headless" in sys.argv
_positional = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
N_FRAMES = int(_positional[0]) if _positional else 120


def process_surface(sc: SimplicialComplex):
    label_surface(sc)
    label_triangle_orient(sc)
    sc = flip_inward_triangles(sc)
    return sc


configure_benchmark_timers()
Logger.set_level(getattr(Logger.Level, os.environ.get("WB_LOG", "Warn")))
workspace = AssetDir.output_path(__file__)
folder = AssetDir.folder(__file__)

engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.02
config["gravity"] = [[0.0], [-9.81], [0.0]]
# --- Stiff-GIPC set_case3 alignment: scene-diagonal-relative parameters ---
# gap = relative_dhat * scene_diagonal (Stiff-GIPC stores dHat = rel^2 * diag^2)
config["contact"]["d_hat_relative"] = 1e-3
# newton exit: max displacement < velocity_tol_relative * diag * dt
config["newton"]["velocity_tol_relative"] = 1e-2
# Stiff-GIPC pcg_solver_threshold = 1e-4
config["linear_system"]["tol_rate"] = 1e-4
# Stiff-GIPC friction slip threshold: sqrt(fDhat)*dt = 1e-2*diag*dt per step
config["contact"]["eps_velocity_relative"] = 1e-2
config["newton"]["transrate_tol"] = 10
print(config)

scene = Scene(config)
abd = AffineBodyConstitution()
# Stiff-GIPC: friction_rate = 0.2.
# kappa note: libuipc scales the barrier by dt^2 in the incremental potential while
# Stiff-GIPC applies Kappa raw, so libuipc kappa = Stiff Kappa / dt^2 = 1e4 / 1e-4 = 1e8.
scene.contact_tabular().default_model(0.2, 1e8)
default_contact = scene.contact_tabular().default_element()

io = SimplicialComplexIO()

scene_json_path = Path(AssetDir.asset_path()) / "sim_data" / "wrecking_ball.json"
with open(scene_json_path, "r", encoding="utf-8") as f:
    wrecking_ball_scene = json.load(f)

tetmesh_dir = AssetDir.tetmesh_path()

cube = io.read(f"{tetmesh_dir}/cube.msh")
cube = process_surface(cube)
ball = io.read(f"{tetmesh_dir}/ball.msh")
ball = process_surface(ball)
link = io.read(f"{tetmesh_dir}/link.msh")
link = process_surface(link)

cube_obj = scene.objects().create("cubes")
ball_obj = scene.objects().create("balls")
link_obj = scene.objects().create("links")


def build_mesh(desc, obj: uipc.core.Object, mesh: SimplicialComplex):
    t = Transform.Identity()
    position = Vector3.Zero()
    if "position" in desc:
        position[0] = desc["position"][0]
        position[1] = desc["position"][1]
        position[2] = desc["position"][2]
        t.translate(position)

    if "rotation" in desc:
        rotation = Vector3.Zero()
        rotation[0] = desc["rotation"][0]
        rotation[1] = desc["rotation"][1]
        rotation[2] = desc["rotation"][2]
        rotation *= np.pi / 180
        q = (
            AngleAxis(rotation[2][0], Vector3.UnitZ())
            * AngleAxis(rotation[1][0], Vector3.UnitY())
            * AngleAxis(rotation[0][0], Vector3.UnitX())
        )
        t.rotate(q)

    is_fixed = 0
    if "is_dof_fixed" in desc:
        is_fixed = desc["is_dof_fixed"]

    this_mesh = mesh.copy()
    # constitution/contact first: they create the instance attributes
    # (is_fixed, transforms); per-object density from the scene description
    # (Stiff-GIPC alignment: cubes 1000, links/ball 7680)
    abd.apply_to(this_mesh, 100 * MPa, desc.get("density", 1e3))
    default_contact.apply_to(this_mesh)
    view(this_mesh.transforms())[0] = t.matrix()
    is_fixed_attr = this_mesh.instances().find("is_fixed")
    view(is_fixed_attr)[0] = is_fixed
    obj.geometries().create(this_mesh)


for obj in wrecking_ball_scene:
    if obj["mesh"] == "link.msh":
        build_mesh(obj, link_obj, link)
    elif obj["mesh"] == "ball.msh":
        build_mesh(obj, ball_obj, ball)
    elif obj["mesh"] == "cube.msh":
        build_mesh(obj, cube_obj, cube)

ground_obj = scene.objects().create("ground")
ground_obj.geometries().create(ground(-1.0))

world.init(scene)

if HEADLESS:
    ball_geo_id = ball_obj.geometries().ids()[0]
    local_center = np.asarray(
        ball.vertices().find("position").view()).reshape(-1, 3).mean(axis=0)
    frame_ms = []
    frame_stats = []
    final_center = None
    for i in range(N_FRAMES):
        t0 = time.perf_counter()
        world.advance()
        world.retrieve()
        frame_ms.append((time.perf_counter() - t0) * 1e3)
        frame_stats.append(snapshot_frame_stats(engine))

        slot, _ = scene.geometries().find(ball_geo_id)
        transform = np.asarray(view(slot.geometry().transforms())[0]).reshape(4, 4)
        final_center = transform[:3, :3] @ local_center + transform[:3, 3]
        if i % 25 == 0 or i == N_FRAMES - 1:
            print(
                f"frame {i}: ball_center=({final_center[0]:.3f},"
                f"{final_center[1]:.3f},{final_center[2]:.3f})",
                flush=True,
            )

    emit_benchmark_result(
        frame_ms,
        frame_stats,
        observables={
            "final_frame": int(world.frame()),
            "ball_center": [float(value) for value in final_center],
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
    tri_surf.set_edge_width(1)

    run = False

    def on_update():
        global run
        if imgui.Button("run & stop"):
            run = not run

        if run:
            world.advance()
            world.retrieve()

        sgui.update()

    ps.set_user_callback(on_update)
    ps.show()

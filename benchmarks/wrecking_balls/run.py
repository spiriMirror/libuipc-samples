import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import uipc
from uipc import view
from uipc import Vector3, Transform, Logger, AngleAxis, Timer
from uipc.core import Engine, World, Scene
from uipc.geometry import (
    SimplicialComplex,
    SimplicialComplexIO,
    ground,
    label_surface,
    label_triangle_orient,
    flip_inward_triangles,
)
from uipc.constitution import AffineBodyConstitution
from uipc.unit import MPa, GPa
from uipc.stats import SimulationStats

from asset_dir import AssetDir

METHODS: tuple[str, ...] = ("stackless_bvh", "info_stackless_bvh")


def process_surface(sc: SimplicialComplex) -> SimplicialComplex:
    label_surface(sc)
    label_triangle_orient(sc)
    return flip_inward_triangles(sc)


def build_mesh(desc: dict, obj: uipc.core.Object, mesh: SimplicialComplex) -> None:
    t = Transform.Identity()
    if "position" in desc:
        position = Vector3.Zero()
        position[0] = desc["position"][0]
        position[1] = desc["position"][1]
        position[2] = desc["position"][2]
        t.translate(position)

    if "rotation" in desc:
        rotation = Vector3.Zero()
        rotation[0] = desc["rotation"][0]
        rotation[1] = desc["rotation"][1]
        rotation[2] = desc["rotation"][2]
        rotation *= np.pi / 180.0
        q = (
            AngleAxis(rotation[2][0], Vector3.UnitZ())
            * AngleAxis(rotation[1][0], Vector3.UnitY())
            * AngleAxis(rotation[0][0], Vector3.UnitX())
        )
        t.rotate(q)

    is_fixed = int(desc.get("is_dof_fixed", 0))
    this_mesh = mesh.copy()
    view(this_mesh.transforms())[0] = t.matrix()
    view(this_mesh.instances().find("is_fixed"))[0] = is_fixed
    obj.geometries().create(this_mesh)


def create_scene(method: str) -> Scene:
    config = Scene.default_config()
    config["dt"] = 0.01
    config["contact"]["d_hat"] = 0.01
    config["newton"]["transrate_tol"] = 10
    config["newton"]["velocity_tol"] = 1
    config["collision_detection"]["method"] = method

    scene = Scene(config)
    scene.contact_tabular().default_model(0.02, 10 * GPa)
    default_contact = scene.contact_tabular().default_element()
    abd = AffineBodyConstitution()

    io = SimplicialComplexIO()
    scene_json_path = Path(AssetDir.asset_path()) / "sim_data" / "wrecking_ball.json"
    with open(scene_json_path, "r", encoding="utf-8") as f:
        wrecking_ball_scene = json.load(f)

    tetmesh_dir = Path(AssetDir.tetmesh_path())
    cube = process_surface(io.read(str(tetmesh_dir / "cube.msh")))
    ball = process_surface(io.read(str(tetmesh_dir / "ball.msh")))
    link = process_surface(io.read(str(tetmesh_dir / "link.msh")))

    cube_obj = scene.objects().create("cubes")
    ball_obj = scene.objects().create("balls")
    link_obj = scene.objects().create("links")

    abd.apply_to(cube, 100 * MPa)
    default_contact.apply_to(cube)
    abd.apply_to(ball, 100 * MPa)
    default_contact.apply_to(ball)
    abd.apply_to(link, 100 * MPa)
    default_contact.apply_to(link)

    for entry in wrecking_ball_scene:
        if entry["mesh"] == "link.msh":
            build_mesh(entry, link_obj, link)
        elif entry["mesh"] == "ball.msh":
            build_mesh(entry, ball_obj, ball)
        elif entry["mesh"] == "cube.msh":
            build_mesh(entry, cube_obj, cube)

    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground(-1.0))

    return scene


def run_case(method: str, num_frames: int, base_output: Path) -> Path:
    case_output = base_output / method
    case_output.mkdir(parents=True, exist_ok=True)
    engine = Engine("cuda", str(case_output))
    world = World(engine)
    scene = create_scene(method)
    world.init(scene)

    stats = SimulationStats()
    for _ in range(num_frames):
        world.advance()
        world.retrieve()
        stats.collect()

    stats_dir = case_output / "stats"
    stats.summary_report(output_dir=str(stats_dir), workspace=str(case_output))
    timer_frames_json = case_output / "timer_frames.json"
    stats.save_timer_frames_json(timer_frames_json)
    return timer_frames_json


def run_method_subprocess(
    method: str, num_frames: int, base_output: Path
) -> Path:
    cmd = [
        sys.executable,
        __file__,
        "--frames",
        str(num_frames),
        "--method",
        method,
    ]
    print(f"[benchmark] running method: {method}")
    subprocess.run(cmd, check=True)
    return base_output / method / "timer_frames.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run no-GUI wrecking_ball benchmark and compare collision broadphase methods."
    )
    parser.add_argument("--frames", type=int, default=300, help="Number of frames per method.")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=("stackless_bvh", "info_stackless_bvh"),
        help="Run a single method only (internal use).",
    )
    args = parser.parse_args()

    Logger.set_level(Logger.Level.Info)
    Timer.enable_all()

    base_output = Path(AssetDir.output_path(__file__))

    if args.method is not None:
        run_case(args.method, args.frames, base_output)
        return

    print(f"[benchmark] output: {base_output}")
    print(f"[benchmark] frames per method: {args.frames}")
    comparison_map = {
        method: run_method_subprocess(method, args.frames, base_output)
        for method in METHODS
    }

    nway_dir = base_output / "nway_compare"
    SimulationStats.create_comparison(
        comparison_map=comparison_map,
        output_dir=str(nway_dir),
        metric="duration",
        align="union",
    )
    print(f"[benchmark] N-way report written to: {nway_dir}")


if __name__ == "__main__":
    main()

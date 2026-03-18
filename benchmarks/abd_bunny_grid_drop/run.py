import argparse
import subprocess
import sys
from pathlib import Path

from uipc import Logger, Timer, Transform, Vector3, view
from uipc.core import Engine, World, Scene
from uipc.geometry import SimplicialComplexIO, label_surface, label_triangle_orient, flip_inward_triangles
from uipc.constitution import AffineBodyConstitution
from uipc.stats import SimulationStats
from uipc.unit import MPa, GPa

from asset_dir import AssetDir


GRID_X = 10
GRID_Z = 10
GRID_SPACING = 1.2
DROP_HEIGHT = 8.0
SCALE = 0.1
METHODS: tuple[str, ...] = ("stackless_bvh", "info_stackless_bvh")


def create_scene(method: str) -> Scene:
    config = Scene.default_config()
    config["dt"] = 0.01
    config["contact"]["d_hat"] = 0.01
    config["newton"]["transrate_tol"] = 10
    config["newton"]["velocity_tol"] = 1
    config["collision_detection"]["method"] = method
    scene = Scene(config)

    scene.contact_tabular().default_model(0.2, 10 * GPa, enable=False)
    default_contact = scene.contact_tabular().default_element()
    abd = AffineBodyConstitution()

    transform = Transform.Identity()
    transform.scale(SCALE)
    io = SimplicialComplexIO(transform)
    bunny_path = f"{AssetDir.tetmesh_path()}/bunny0.msh"
    bunny = io.read(bunny_path)
    label_surface(bunny)
    label_triangle_orient(bunny)
    bunny = flip_inward_triangles(bunny)

    abd.apply_to(bunny, 100 * MPa)
    default_contact.apply_to(bunny)
    bunny.instances().resize(GRID_X * GRID_Z)

    transforms = view(bunny.transforms())
    for iz in range(GRID_Z):
        for ix in range(GRID_X):
            idx = iz * GRID_X + ix
            t = Transform.Identity()
            x = (ix - (GRID_X - 1) * 0.5) * GRID_SPACING
            z = (iz - (GRID_Z - 1) * 0.5) * GRID_SPACING
            p = Vector3.Zero()
            p[0] = x
            p[1] = DROP_HEIGHT
            p[2] = z
            t.translate(p)
            transforms[idx] = t.matrix()

    bunnies = scene.objects().create("bunny_grid")
    bunnies.geometries().create(bunny)
    return scene


def run_case(method: str, num_frames: int, base_output: Path) -> Path:
    output = base_output / method
    output.mkdir(parents=True, exist_ok=True)

    engine = Engine("cuda", str(output))
    world = World(engine)
    world.init(create_scene(method))

    stats = SimulationStats()
    for _ in range(num_frames):
        world.advance()
        world.retrieve()
        stats.collect()

    stats_dir = output / "stats"
    stats.summary_report(output_dir=str(stats_dir), workspace=str(output))
    timer_frames_json = output / "timer_frames.json"
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
        description="Run no-GUI benchmark for 10x10 ABD bunny grid drop and compare broadphase methods."
    )
    parser.add_argument("--frames", type=int, default=100, help="Number of frames per method.")
    parser.add_argument("--method", type=str, default=None, choices=("stackless_bvh", "info_stackless_bvh"))
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

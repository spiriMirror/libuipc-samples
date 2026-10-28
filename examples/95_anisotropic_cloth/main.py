"""Compare shared, a-stiff and b-stiff cloth under identical biaxial traction.

No external mesh assets are needed. Build pyuipc with the directional
StrainLimitingBaraffWitkinShell overload before running this example.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from uipc import Engine, Logger, Scene, World, builtin, view
from uipc.constitution import (
    DiscreteShellBending,
    ElasticModuli2D,
    FiniteElementExternalForce,
    StrainLimitingBaraffWitkinShell,
)
from uipc.geometry import label_surface, trimesh

sys.path.append(str(Path(__file__).resolve().parents[1]))
from benchmark_utils import (
    configure_benchmark_timers,
    emit_benchmark_result,
    report_timers_if_enabled,
    snapshot_frame_stats,
)

CASES = ("shared", "a_stiff", "b_stiff")
COLORS = ((0.65, 0.65, 0.7), (0.9, 0.35, 0.2), (0.2, 0.5, 0.9))


def grid(resolution):
    """Unit square in XY, +Z normals: rest axes a=world X, b=world Y."""
    x, y = np.meshgrid(np.linspace(0, 1, resolution + 1),
                       np.linspace(0, 1, resolution + 1))
    vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    cells = []
    for j in range(resolution):
        for i in range(resolution):
            v = j * (resolution + 1) + i
            cells.extend(((v, v + 1, v + resolution + 2),
                          (v, v + resolution + 2, v + resolution + 1)))
    return vertices, np.asarray(cells, dtype=np.int32)


def create_scene(args):
    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [0.0], [0.0]]
    # This is an isolated material comparison without contact or gravity loads.
    config["contact"]["enable"] = False
    config["newton"]["velocity_tol"] = 1e-5
    scene = Scene(config)
    shell = StrainLimitingBaraffWitkinShell()
    bending = DiscreteShellBending()
    loading = FiniteElementExternalForce()
    soft = ElasticModuli2D.youngs_poisson(args.soft_young, args.soft_poisson)
    stiff = ElasticModuli2D.youngs_poisson(args.stiff_young, args.stiff_poisson)
    shear = ElasticModuli2D.youngs_poisson(10.0, 0.25)
    local, triangles = grid(args.resolution)
    fixed = np.isclose(local[:, 0], 0) | np.isclose(local[:, 1], 0)
    right = np.isclose(local[:, 0], 1)
    top = np.isclose(local[:, 1], 1)
    # Integrate uniform line traction (N/m) over the right and top boundaries.
    forces = np.zeros_like(local)
    forces[right, 0] = args.traction / args.resolution
    forces[top, 1] = args.traction / args.resolution
    forces[right & (np.isclose(local[:, 1], 0) | top), 0] *= 0.5
    forces[top & (np.isclose(local[:, 0], 0) | right), 1] *= 0.5
    forces[fixed] = 0
    records = {}
    for index, name in enumerate(CASES):
        vertices = local + np.array([1.6 * index, 0, 0])
        mesh = trimesh(vertices, triangles)
        if name == "shared":
            shell.apply_to(mesh, stretch_moduli=soft, shear_moduli=shear,
                           thickness=0.001, mass_density=200, strain_rate=100)
        else:
            shell.apply_to(
                mesh,
                stretch_a_moduli=stiff if name == "a_stiff" else soft,
                stretch_b_moduli=stiff if name == "b_stiff" else soft,
                shear_moduli=shear,
                thickness=0.001,
                mass_density=200,
                strain_rate=100,
            )
        # Bending and shear are identical in all three sheets.
        bending.apply_to(mesh, 3e4, 0.3)
        view(mesh.vertices().find(builtin.is_fixed)).reshape(-1)[fixed] = 1
        loading.apply_to(mesh, np.zeros(3))
        view(mesh.vertices().find("external_force")).reshape(-1, 3)[:] = forces
        label_surface(mesh)
        obj = scene.objects().create(name)
        slot, _ = obj.geometries().create(mesh)
        records[name] = {"slot": slot, "rest": vertices.copy(), "fixed": fixed}
    return scene, records, triangles


def measurements(records, resolution):
    result = {}
    for name, record in records.items():
        mesh = record["slot"].geometry()
        positions = np.array(mesh.positions().view()).reshape(-1, 3)
        if not np.isfinite(positions).all():
            raise RuntimeError(f"{name}: non-finite cloth positions")
        rows = positions.reshape(resolution + 1, resolution + 1, 3)
        a_stretch = np.linalg.norm(np.diff(rows, axis=1), axis=2) * resolution
        b_stretch = np.linalg.norm(np.diff(rows, axis=0), axis=2) * resolution
        pin_error = np.max(np.linalg.norm(
            positions[record["fixed"]] - record["rest"][record["fixed"]], axis=1))
        attrs = mesh.triangles()
        ka = float(np.asarray(attrs.find("lambda").view()).flat[0])
        kb_attr = attrs.find("stretch_stiffness_b")
        kb = ka if kb_attr is None else float(np.asarray(kb_attr.view()).flat[0])
        result[name] = {
            "Ka": ka, "Kb": kb,
            "mean_strain_a": float(a_stretch.mean() - 1),
            "mean_strain_b": float(b_stretch.mean() - 1),
            "max_pin_error": float(pin_error),
        }
    return result


def validate(result):
    for name, data in result.items():
        if data["max_pin_error"] > 1e-8:
            raise RuntimeError(f"{name}: pinned vertices moved")
    shared = result["shared"]
    if min(shared["mean_strain_a"], shared["mean_strain_b"]) <= 0.005:
        raise RuntimeError("Reference cloth has not stretched enough for comparison")
    for name, axis in (("a_stiff", "a"), ("b_stiff", "b")):
        key = f"mean_strain_{axis}"
        if result[name][key] >= 0.75 * shared[key]:
            raise RuntimeError(f"{name}: stiff direction did not reduce extension")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", type=int, nargs="?", const=120, metavar="FRAMES")
    parser.add_argument("--validate", action="store_true",
                        help="Check directional response after at least 80 headless frames")
    parser.add_argument("--resolution", type=int, default=16, help="Cells per side")
    parser.add_argument("--soft-young", type=float, default=2e3)
    parser.add_argument("--soft-poisson", type=float, default=0.25)
    parser.add_argument("--stiff-young", type=float, default=2e5)
    parser.add_argument("--stiff-poisson", type=float, default=0.3)
    parser.add_argument("--traction", type=float, default=10., help="Edge traction in N/m")
    parser.add_argument("--output", type=Path, default=(
        Path(__file__).resolve().parents[2] / "output/examples/95_anisotropic_cloth"))
    args = parser.parse_args()
    if args.resolution < 2 or (args.headless is not None and args.headless < 1):
        parser.error("resolution must be >= 2 and headless frames must be positive")
    if not all(np.isfinite(value) and value > 0 for value in
               (args.soft_young, args.stiff_young, args.traction)):
        parser.error("Young parameters and traction must be positive and finite")
    if not all(np.isfinite(value) and -1 < value < 1 for value in
               (args.soft_poisson, args.stiff_poisson)):
        parser.error("Poisson parameters must be finite and strictly between -1 and 1")
    if args.validate and (args.headless is None or args.headless < 80):
        parser.error("--validate requires --headless with at least 80 frames")
    soft_k = args.soft_young / (1 - args.soft_poisson**2)
    stiff_k = args.stiff_young / (1 - args.stiff_poisson**2)
    if args.validate and stiff_k < 10 * soft_k:
        parser.error("--validate requires at least a 10:1 stiff/soft coefficient ratio")
    if "stretch_a_moduli" not in (StrainLimitingBaraffWitkinShell.apply_to.__doc__ or ""):
        parser.error("Installed pyuipc lacks directional cloth support; build the updated libuipc source")
    args.output.mkdir(parents=True, exist_ok=True)
    Logger.set_level(Logger.Level.Warn)
    if args.headless is not None:
        configure_benchmark_timers()
    scene, records, triangles = create_scene(args)
    engine = Engine("cuda", str(args.output))
    world = World(engine)
    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("Cloth scene initialization failed")

    def advance():
        world.advance()
        world.retrieve()
        if not world.is_valid():
            raise RuntimeError(f"Simulation failed at frame {world.frame()}")

    if args.headless is not None:
        frame_ms = []
        frame_stats = []
        for frame in range(args.headless):
            start = time.perf_counter()
            advance()
            frame_ms.append((time.perf_counter() - start) * 1e3)
            frame_stats.append(snapshot_frame_stats(engine))
            if (frame + 1) % 20 == 0:
                print(f"frame {world.frame()}/{args.headless}", flush=True)
        result = measurements(records, args.resolution)
        if args.validate:
            validate(result)
        report = {"frames": world.frame(), "validated": args.validate, "cases": result}
        (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez(args.output / "cloth_positions.npz", triangles=triangles, **{
            name: np.array(record["slot"].geometry().positions().view()).reshape(-1, 3)
            for name, record in records.items()
        })
        print(json.dumps(report, indent=2))
        emit_benchmark_result(frame_ms, frame_stats, observables=report)
        report_timers_if_enabled()
        return

    import polyscope as ps
    from polyscope import imgui

    ps.init()
    ps.set_ground_plane_mode("none")
    surfaces = {}
    for name, color in zip(CASES, COLORS):
        surface = ps.register_surface_mesh(name, records[name]["rest"], triangles)
        surface.set_color(color)
        surface.set_edge_width(0.6)
        surfaces[name] = surface
        ps.register_point_cloud(f"{name} pins",
                                records[name]["rest"][records[name]["fixed"]],
                                color=(0.1, 0.1, 0.1), radius=0.003)
    ps.look_at((2.1, 0.5, 5.5), (2.1, 0.5, 0.0))
    running = False

    def update():
        nonlocal running
        imgui.Text("Left: shared | Middle: a stiff (X) | Right: b stiff (Y)")
        imgui.Text("Identical top/right traction; left/bottom edges pinned.")
        imgui.Text(f"Frame: {world.frame()}")
        if imgui.Button("Pause" if running else "Run"):
            running = not running
        if running:
            advance()
            for name, surface in surfaces.items():
                surface.update_vertex_positions(
                    np.array(records[name]["slot"].geometry().positions().view()).reshape(-1, 3))
        for name, data in measurements(records, args.resolution).items():
            imgui.Text(f"{name}: a={100 * data['mean_strain_a']:.2f}%, "
                       f"b={100 * data['mean_strain_b']:.2f}%")

    ps.set_user_callback(update)
    ps.show()


if __name__ == "__main__":
    main()

"""Example 94 -- a four-finger URDF hand picks and places a free apple.

The scene directly exercises libuipc: 17 quasi-static ABD hand links follow
per-substep SoftTransformConstraint targets, while an unconstrained ABD apple
responds only to gravity, contact and friction. The hand holds, approaches,
closes all four fingertips, lifts, transfers, lowers, releases and retreats.

Usage:
  python main.py                  # interactive Polyscope playback
  python main.py --headless 500   # run and numerically verify the full motion
  python main.py --headless 50 --write-surfaces
"""

import argparse
import json
import sys
import time

import numpy as np
import trimesh as trimesh_library

import uipc
from uipc import Animation, Logger, Transform, Vector3, builtin, view
from uipc.constitution import (
    AffineBodyConstitution,
    SoftTransformConstraint,
)
from uipc.core import Engine, Scene, SceneIO, World
from uipc.geometry import (
    UrdfIO,
    flip_inward_triangles,
    ground,
    label_surface,
    label_triangle_orient,
    tetmesh,
)
from uipc.unit import MPa

from asset_dir import CASE_DIR, output_dir, urdf_path


FPS = 30
SUBSTEPS = 4
DT = 1.0 / (FPS * SUBSTEPS)
FIRST_FRAME = 1
LAST_FRAME = 500
APPLE_RADII = np.array([0.0395, 0.036, 0.0395])
APPLE_START = np.array([0.0, APPLE_RADII[1] + 0.001, 0.0])
APPLE_DESTINATION = np.array([0.45, APPLE_RADII[1] + 0.001, 0.0])
TIP_NAMES = ("fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip")
# p_native = BASIS @ p_blender. Native/Polyscope uses +Y as up.
BASIS = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("frames", nargs="?", type=int, default=LAST_FRAME)
    parser.add_argument(
        "--write-surfaces",
        action="store_true",
        help="write OBJ surfaces at stage boundaries (slower)",
    )
    result = parser.parse_args()
    if not FIRST_FRAME <= result.frames <= LAST_FRAME:
        parser.error(f"frames must be in [{FIRST_FRAME}, {LAST_FRAME}]")
    return result


def scene_config():
    config = Scene.default_config()
    config["dt"] = DT
    config["gravity"] = [[0.0], [-9.81], [0.0]]
    config["contact"]["constitution"] = "ipc"
    config["contact"]["d_hat"] = 1e-3
    # Loaded sample-87 collision meshes contain two known zero-volume/open
    # fragments. UrdfIO and contact support them; the original assets stay intact.
    config["sanity_check"]["enable"] = 0
    config["extras"]["strict_mode"]["enable"] = 1
    # Strong position drives and a light free object share the global solve.
    # These are accuracy settings, not modified materials or artificial damping.
    # config["newton"]["semi_implicit"]["enable"] = 0
    config["newton"]["velocity_tol"] = 1e-2
    config["newton"]["velocity_tol_relative"] = 0.0
    config["linear_system"]["tol_rate"] = 1e-6
    config["line_search"]["max_iter"] = 32
    return config


def apply_hand_constitutions(controller, contact):
    abd = AffineBodyConstitution()
    drive = SoftTransformConstraint()
    links = {}
    for slot in controller.links():
        geometry = slot.geometry()
        name = geometry.meta().find("urdf/name").view()[0]
        links[name] = slot
        abd.apply_to(geometry, 100 * MPa, mass_density=5e3)
        view(geometry.instances().find(builtin.is_dynamic))[:] = 0
        drive.apply_to(geometry, np.array([1e4, 1e4]))
        view(geometry.instances().find(builtin.is_constrained))[:] = 1
        contact.apply_to(geometry)
    return links


def create_apple(scene, contact):
    # A symmetric star tetrahedralization keeps the visual center, volume
    # centroid and grasp center coincident. Reusing ball.msh here would offset
    # its volume centroid by about 5.6 mm and make the free object roll before
    # the hand arrives.
    surface = trimesh_library.creation.icosphere(subdivisions=2, radius=1.0)
    positions = np.asarray(surface.vertices, dtype=np.float64) * APPLE_RADII
    center = len(positions)
    positions = np.vstack((positions, np.zeros(3)))
    cells = np.column_stack(
        (
            np.full(len(surface.faces), center, dtype=np.int32),
            np.asarray(surface.faces, dtype=np.int32),
        )
    )
    p = positions[cells]
    signed = np.linalg.det(
        np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
    )
    cells[signed < 0, 2:4] = cells[signed < 0, 3:1:-1]
    if np.any(signed == 0):
        raise ValueError("Degenerate procedural apple tetrahedron")
    mesh = tetmesh(positions, cells)
    label_surface(mesh)
    label_triangle_orient(mesh)
    mesh = flip_inward_triangles(mesh)
    AffineBodyConstitution().apply_to(mesh, 100 * MPa, mass_density=820)
    contact.apply_to(mesh)
    transform = Transform.Identity()
    transform.translate(Vector3.Values(APPLE_START.tolist()))
    view(mesh.transforms())[0] = transform.matrix()
    apple = scene.objects().create("apple")
    slot, _ = apple.geometries().create(mesh)
    return apple, slot


class HandTrajectory:
    def __init__(self, controller, plan):
        self.controller = controller
        self.open = {name: float(value) for name, value in plan["open"].items()}
        self.closed = {name: float(value) for name, value in plan["closed"].items()}
        available = set(
            controller.revolute_joints().geometry().instances().find("name").view()
        )
        if set(self.open) != available or set(self.closed) != available:
            raise ValueError(
                "The grasp plan must define every URDF revolute joint exactly once"
            )
        chains = (
            ("0", "1", "2", "3"),
            ("4", "5", "6", "7"),
            ("8", "9", "10", "11"),
            ("12", "13", "14", "15"),
        )
        self.four_fingers_commanded = all(
            any(abs(self.closed[name] - self.open[name]) > 1e-3 for name in chain)
            for chain in chains
        )
        if not self.four_fingers_commanded:
            raise ValueError(
                "Every finger chain must move between open and closed poses"
            )
        grasp_blender = np.asarray(plan["grasp_center_blender"], dtype=float)
        # UrdfController.move_root() accepts URDF/Z-up coordinates and performs
        # its own BASIS conversion. Only world-space apple coordinates need the
        # inverse map when constructing root targets.
        contact_root = BASIS.T @ APPLE_START - grasp_blender
        destination_root = BASIS.T @ APPLE_DESTINATION - grasp_blender
        self.stages = (
            (1, contact_root + [0, 0.25, 0.28], self.open),
            (50, contact_root + [0, 0.25, 0.28], self.open),
            (110, contact_root + [0, 0, 0.13], self.open),
            (150, contact_root, self.open),
            (195, contact_root, self.closed),
            (250, contact_root + [0, 0, 0.22], self.closed),
            (325, destination_root + [0, 0, 0.22], self.closed),
            (380, destination_root, self.closed),
            (420, destination_root, self.open),
            (465, destination_root + [0, 0, 0.22], self.open),
            (500, destination_root + [0, 0.20, 0.25], self.open),
        )

    def sample(self, frame):
        if frame <= self.stages[0][0]:
            return self.stages[0][1], self.stages[0][2]
        for left, right in zip(self.stages, self.stages[1:]):
            if frame <= right[0]:
                t = np.clip((frame - left[0]) / (right[0] - left[0]), 0.0, 1.0)
                t = t * t * (3.0 - 2.0 * t)
                root = (1.0 - t) * left[1] + t * right[1]
                angles = {
                    name: (1.0 - t) * left[2][name] + t * right[2][name]
                    for name in self.open
                }
                return root, angles
        return self.stages[-1][1], self.stages[-1][2]

    def apply(self, frame, attr):
        root, angles = self.sample(frame)
        self.controller.move_root(root, np.zeros(3))
        for name, angle in angles.items():
            self.controller.rotate_to(name, angle)
        self.controller.apply_to(attr=attr)

    def animate(self, info: Animation.UpdateInfo):
        output_frame = FIRST_FRAME + info.frame() / SUBSTEPS
        self.apply(output_frame, builtin.aim_transform)


def build_world():
    Logger.set_level(Logger.Level.Warn)
    workspace = output_dir()
    engine = Engine("cuda", str(workspace))
    world = World(engine)
    scene = Scene(scene_config())
    tabular = scene.contact_tabular()
    tabular.default_model(0.45, 1e9)
    default_contact = tabular.default_element()
    hand_contact = tabular.create("robot_hand")
    tabular.insert(hand_contact, hand_contact, 0.0, 0.0, False)
    tabular.insert(hand_contact, default_contact, 0.8, 1e9, True)

    hand = scene.objects().create("robot_hand")
    urdf_config = UrdfIO.default_config()
    urdf_config["load_visual_mesh"] = False
    controller = UrdfIO(urdf_config).read(hand, str(urdf_path()))
    links = apply_hand_constitutions(controller, hand_contact)

    with (CASE_DIR / "grasp_plan.json").open(encoding="utf-8") as stream:
        trajectory = HandTrajectory(controller, json.load(stream))
    trajectory.apply(FIRST_FRAME, builtin.transform)
    trajectory.apply(FIRST_FRAME, builtin.aim_transform)
    scene.animator().insert(hand, trajectory.animate)

    apple, apple_slot = create_apple(scene, default_contact)
    table = scene.objects().create("table")
    table_mesh = ground(0.0)
    default_contact.apply_to(table_mesh)
    table.geometries().create(table_mesh)

    hand_points = np.concatenate([world_points(slot) for slot in links.values()])
    apple_points = world_points(apple_slot, surface_only=True)
    print(
        "initial bounds: "
        f"hand={hand_points.min(axis=0)}..{hand_points.max(axis=0)} "
        f"apple={apple_points.min(axis=0)}..{apple_points.max(axis=0)}",
        flush=True,
    )
    if hand_points[:, 1].min() <= 0.0 or apple_points[:, 1].min() <= 0.0:
        raise ValueError("The initial hand/apple geometry crosses the y=0 table")

    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("World initialization failed")
    world.retrieve()
    return engine, world, scene, trajectory, links, apple, apple_slot


def world_points(slot, surface_only=False):
    geometry = slot.geometry()
    points = np.asarray(geometry.positions().view()).reshape(-1, 3)
    if surface_only:
        attribute = geometry.vertices().find(builtin.is_surf)
        if attribute is not None:
            points = points[np.asarray(attribute.view()).reshape(-1) != 0]
    transform = np.asarray(geometry.transforms().view()).reshape(-1, 4, 4)[0]
    return points @ transform[:3, :3].T + transform[:3, 3]


def surface_mesh(slot):
    geometry = slot.geometry()
    points = world_points(slot)
    faces = np.asarray(geometry.triangles().topo().view()).reshape(-1, 3)
    attribute = geometry.triangles().find(builtin.is_surf)
    if attribute is not None:
        faces = faces[np.asarray(attribute.view()).reshape(-1) != 0]
    return trimesh_library.Trimesh(vertices=points, faces=faces, process=False)


def surface_distance(left_slot, right_slot):
    """Conservative bidirectional vertex-to-triangle surface distance."""
    left, right = surface_mesh(left_slot), surface_mesh(right_slot)
    left_to_right = trimesh_library.proximity.closest_point_naive(right, left.vertices)[
        1
    ]
    right_to_left = trimesh_library.proximity.closest_point_naive(left, right.vertices)[
        1
    ]
    return float(min(left_to_right.min(), right_to_left.min()))


def phase_name(frame):
    for boundary, name in (
        (50, "hold"),
        (110, "approach"),
        (150, "descend"),
        (195, "close"),
        (250, "lift"),
        (325, "transfer"),
        (380, "lower"),
        (420, "release"),
        (465, "rise"),
        (500, "retreat"),
    ):
        if frame <= boundary:
            return name
    return "complete"


def run_headless(args, engine, world, scene, trajectory, links, apple_slot):
    output = output_dir()
    writer = SceneIO(scene)
    centers = []
    phases = {}
    contacts = {}
    started = time.perf_counter()

    def record(frame):
        apple = world_points(apple_slot, surface_only=True)
        center = apple.mean(axis=0)
        centers.append(center)
        if frame in (1, 50, 150, 195, 250, 325, 380, 420, 465, 500):
            phases[str(frame)] = {
                "phase": phase_name(frame),
                "apple_center": center.tolist(),
                "apple_min_y": float(apple[:, 1].min()),
            }
        if frame in (195, 250, 325):
            contacts[str(frame)] = {
                name: surface_distance(apple_slot, links[name]) for name in TIP_NAMES
            }
        if args.write_surfaces and frame in (1, 150, 195, 250, 325, 380, 420, 500):
            writer.write_surface(str(output / f"surface_{frame:04d}.obj"))

    record(FIRST_FRAME)
    for frame in range(FIRST_FRAME + 1, args.frames + 1):
        for _ in range(SUBSTEPS):
            world.advance()
            if not world.is_valid():
                raise RuntimeError(f"Simulation failed before output frame {frame}")
        world.retrieve()
        record(frame)
        if frame % 25 == 0 or frame in (2, args.frames):
            center = centers[-1]
            print(
                f"frame={frame} phase={phase_name(frame)} "
                f"apple=({center[0]:.5f},{center[1]:.5f},{center[2]:.5f})",
                flush=True,
            )

    centers = np.asarray(centers)
    apple_geometry = apple_slot.geometry()
    apple_fixed = np.asarray(
        apple_geometry.instances().find(builtin.is_fixed).view()
    ).reshape(-1)
    apple_constrained = apple_geometry.instances().find(builtin.is_constrained)
    result = {
        "success": True,
        "requested_output_frames": args.frames,
        "world_substeps": int(world.frame()),
        "dt": DT,
        "fps": FPS,
        "substeps_per_output_frame": SUBSTEPS,
        "elapsed_seconds": time.perf_counter() - started,
        "apple_is_unconstrained": bool(
            np.all(apple_fixed == 0)
            and (
                apple_constrained is None
                or np.all(np.asarray(apple_constrained.view()).reshape(-1) == 0)
            )
        ),
        "four_fingers_commanded": trajectory.four_fingers_commanded,
        "full_validation_performed": args.frames == LAST_FRAME,
        "phases": phases,
        "fingertip_surface_distances": contacts,
        "final_apple_center": centers[-1].tolist(),
    }
    errors = []
    if args.frames == LAST_FRAME:
        final_speed = np.linalg.norm(np.diff(centers[-16:], axis=0) * FPS, axis=1)
        result.update(
            apple_lift_m=float(centers[249, 1] - centers[149, 1]),
            apple_transport_m=float(abs(centers[324, 0] - centers[149, 0])),
            final_apple_rms_speed_m_s=float(np.sqrt(np.mean(final_speed**2))),
        )
        if result["apple_lift_m"] <= 0.15:
            errors.append("The apple was not lifted")
        if result["apple_transport_m"] <= 0.35:
            errors.append("The apple was not transported")
        if abs(centers[-1, 0] - APPLE_DESTINATION[0]) >= 0.05:
            errors.append("The apple missed the destination")
        if not 0.0 < phases["500"]["apple_min_y"] < 0.01:
            errors.append("The apple is not resting on the table")
        if result["final_apple_rms_speed_m_s"] >= 0.01:
            errors.append("The released apple has not settled")
        for frame in (250, 325):
            if any(value >= 0.0015 for value in contacts[str(frame)].values()):
                errors.append(
                    f"Not all four fingertips carry the apple at frame {frame}: {contacts[str(frame)]}"
                )
        result["four_fingertip_carry_verified"] = not any(
            "four fingertips" in error for error in errors
        )
    result["success"] = not errors
    result["errors"] = errors
    (output / "validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output / "trajectory.npz", apple_centers=centers)
    print("ROBOT_HAND_GRASP_APPLE " + json.dumps(result), flush=True)
    if errors:
        raise AssertionError("; ".join(errors))
    return result


def run_gui(world, scene):
    import polyscope as ps
    from polyscope import imgui
    from uipc.gui import SceneGUI

    scene_gui = SceneGUI(scene)
    ps.init()
    surface, _, _ = scene_gui.register()
    surface.set_material("ceramic")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_height(0.0)
    ps.set_ground_plane_height_mode("manual")
    state = {"running": False, "output_frame": FIRST_FRAME}

    def update():
        imgui.Text(f"output frame: {state['output_frame']}/{LAST_FRAME}")
        imgui.Text(f"phase: {phase_name(state['output_frame'])}")
        if imgui.Button("pause" if state["running"] else "run"):
            state["running"] = not state["running"]
        if state["running"] and state["output_frame"] < LAST_FRAME:
            for _ in range(SUBSTEPS):
                world.advance()
            world.retrieve()
            state["output_frame"] += 1
            scene_gui.update()
        if state["output_frame"] >= LAST_FRAME:
            state["running"] = False

    ps.set_user_callback(update)
    ps.show()


def main():
    args = arguments()
    engine, world, scene, trajectory, links, _apple, apple_slot = build_world()
    if args.headless:
        run_headless(args, engine, world, scene, trajectory, links, apple_slot)
    else:
        run_gui(world, scene)


if __name__ == "__main__":
    main()

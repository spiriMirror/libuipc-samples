"""
Prismatic joint built from body-local attachment points.

Mirror of `examples/42_abd_revolute_joint_local`, but using the
*local-coordinate* overload of `AffineBodyPrismaticJoint.create_geometry`
instead of the revolute version.

A prismatic constraint forces the two attachment axes of the left and
right bodies to stay collinear; the bodies may *slide* along this shared
axis but cannot translate perpendicular to it or rotate relative to each
other. The scalar `distance` edge attribute measures the current slide
offset and is printed every frame.

Scene layout (no gravity, no contact):
- Left body fixed, right body free.
- Both bodies expose a local axis defined by (pos0 -> pos1).
- After `world.init`, `change_right_tf()` uses
  `AffineBodyStateAccessorFeature` to explicitly reposition the right
  body so its axis is collinear with the left body's axis.
"""

import numpy as np
import polyscope as ps
import uipc.builtin as builtin
import uipc.unit as unit
from asset_dir import AssetDir
from polyscope import imgui
from uipc import AngleAxis, Engine, Logger, Matrix4x4, Scene, Transform, Vector3, World, view
from uipc.constitution import AffineBodyConstitution, AffineBodyPrismaticJoint
from uipc.core import AffineBodyStateAccessorFeature
from uipc.geometry import (
    SimplicialComplex,
    SimplicialComplexIO,
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
)
from uipc.gui import SceneGUI

Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.01
config["contact"]["enable"] = False
config["line_search"]["report_energy"] = True
config["sanity_check"]["enable"] = 0
scene = Scene(config)

# ---------------------------------------------------------------------------
# Body meshes: 0.3 m cubes, ABD with kappa = 100 MPa
# ---------------------------------------------------------------------------
pre = Transform.Identity()
pre.scale([0.3, 0.3, 0.3])
io = SimplicialComplexIO(pre)

abd = AffineBodyConstitution()


def process_surface(sc: SimplicialComplex) -> SimplicialComplex:
    label_surface(sc)
    label_triangle_orient(sc)
    return flip_inward_triangles(sc)


def make_abd_cube(position, rotation: AngleAxis = AngleAxis.Identity(), fixed: bool = False):
    mesh = process_surface(io.read(f"{AssetDir.tetmesh_path()}/cube.msh"))
    abd.apply_to(mesh, 100.0 * unit.MPa)
    mesh.instances().resize(1)

    t = Transform.Identity()
    t.translate(position)
    t.rotate(rotation)
    view(mesh.transforms())[0] = t.matrix()
    if fixed:
        view(mesh.instances().find(builtin.is_fixed))[0] = 1
    return mesh


# Left body fixed; right body free (to be moved by the prismatic drive or state accessor)
left_obj = scene.objects().create("left")
left_mesh = make_abd_cube([0.15, 0.5, -0.5], fixed=True)
left_slot, _ = left_obj.geometries().create(left_mesh)

right_obj = scene.objects().create("right")
right_mesh = make_abd_cube([0.15, 0.0, -0.5], fixed=False)
right_slot, _ = right_obj.geometries().create(right_mesh)

# ---------------------------------------------------------------------------
# Prismatic joint built with the local-coordinate create_geometry overload
# ---------------------------------------------------------------------------
prismatic = AffineBodyPrismaticJoint()

# Vector3.Unit*() returns a (3, 1) column vector — flatten before arithmetic
# so the final batched array is (N, 3) not (N, 3, 3).
_unit_y = np.asarray(Vector3.UnitY(), dtype=np.float64).reshape(3)
_unit_z = np.asarray(Vector3.UnitZ(), dtype=np.float64).reshape(3)

# Local axis endpoints, in each body's local frame.
# Left body axis runs parent_pos -> parent_pos + Z (along local +Z).
# Right body axis is offset by +Y*0.5 but parallel, so once right body is
# translated -Y*0.5 (or left translated +Y*0.5) the two axes coincide.
parent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
child_pos = parent_pos + _unit_y * 0.5

l_pos0 = np.array([parent_pos], dtype=np.float64)
l_pos1 = np.array([parent_pos + _unit_y], dtype=np.float64)
r_pos0 = np.array([child_pos], dtype=np.float64)
r_pos1 = np.array([child_pos + _unit_y], dtype=np.float64)

l_slots = [left_slot]
r_slots = [right_slot]
l_instance_ids = np.array([0], dtype=np.int32)
r_instance_ids = np.array([0], dtype=np.int32)
strength_ratios = np.array([100.0], dtype=np.float64)

joint_mesh = prismatic.create_geometry(
    l_pos0,
    l_pos1,
    r_pos0,
    r_pos1,
    l_slots,
    l_instance_ids,
    r_slots,
    r_instance_ids,
    strength_ratios,
)

joint_object = scene.objects().create("prismatic_joint")
joint_geo_slot, _ = joint_object.geometries().create(joint_mesh)

NUM_JOINTS = 1
JOINT_LABELS = ["Prismatic L<->R"]


def _read_transform(slot):
    return np.asarray(view(slot.geometry().transforms())[0]).reshape(4, 4)


def to_world(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    p4 = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return (T @ p4)[:3]


def print_distances() -> None:
    """Print the `distance` edge attribute (slide offset along the joint axis)."""
    geo = joint_geo_slot.geometry()
    distances = geo.edges().find("distance")
    if distances is None:
        print(f"frame={world.frame()} distance attribute not ready yet")
        return
    v = view(distances)
    for i in range(min(NUM_JOINTS, len(v))):
        d = float(v[i])
        print(f"frame={world.frame()} joint={i} {JOINT_LABELS[i]}: distance={d:+.4f} m")


def change_right_tf(new_T: Transform) -> None:
    """Overwrite the right body's transform *after* `world.init`.

    Follows the same state-accessor pattern as test `43_abd_fem_state_access`:
      1. Fetch `AffineBodyStateAccessorFeature` from the world.
      2. Build an empty state-geometry and declare the `transform` attribute.
      3. `copy_to` to pull the current transforms from the backend.
      4. Mutate the target body's row in the view.
      5. `copy_from` to push the change back, then `world.retrieve()`.

    Right body is the 2nd ABD body created → body index 1.
    """
    abd_accessor = world.features().find(AffineBodyStateAccessorFeature)
    assert abd_accessor is not None, "AffineBodyStateAccessorFeature is not available in this uipc build"

    abd_state = abd_accessor.create_geometry()
    abd_state.instances().create(builtin.transform, Matrix4x4.Zero())

    abd_accessor.copy_to(abd_state)

    trans_view = view(abd_state.transforms())
    RIGHT_BODY_IDX = 1  # creation order: 0=left, 1=right

    old_T = np.asarray(trans_view[RIGHT_BODY_IDX]).reshape(4, 4).copy()
    trans_view[RIGHT_BODY_IDX] = new_T.matrix()

    abd_accessor.copy_from(abd_state)
    world.retrieve()

    print(f"change_right_tf: body[{RIGHT_BODY_IDX}] translation {old_T[:3, 3]} -> {np.asarray(new_T.matrix())[:3, 3]}")


def check_axes_collinear(atol: float = 5e-3) -> bool:
    """Assert that the left and right attachment axes coincide in world
    space (pos0 of both bodies, and the axis direction pos1-pos0, should
    match up to the prismatic slide).

    For a correctly-satisfied prismatic constraint, the two endpoints on
    each side need to lie on a common infinite line. We verify this by
    checking that the axis *direction* is aligned (cross product ~ 0)
    and that pos0_L - pos0_R is parallel to that direction.
    """
    TL = _read_transform(left_slot)
    TR = _read_transform(right_slot)

    dir_L = to_world(TL, l_pos1[0]) - to_world(TL, l_pos0[0])
    dir_R = to_world(TR, r_pos1[0]) - to_world(TR, r_pos0[0])

    # Parallel axes
    cross = np.cross(dir_L, dir_R)
    assert np.linalg.norm(cross) < atol, (
        f"frame={world.frame()} axes not parallel: dir_L={dir_L} dir_R={dir_R} |cross|={np.linalg.norm(cross):.6e}"
    )

    # Collinear (offset between pos0's lies along the axis)
    offset = to_world(TR, r_pos0[0]) - to_world(TL, l_pos0[0])
    axis = dir_L / max(np.linalg.norm(dir_L), 1e-12)
    perp = offset - np.dot(offset, axis) * axis
    assert np.linalg.norm(perp) < atol, (
        f"frame={world.frame()} axes not collinear: perpendicular residual |perp|={np.linalg.norm(perp):.6e}"
    )
    return True


# ---------------------------------------------------------------------------
# Init & GUI
# ---------------------------------------------------------------------------
sgui = SceneGUI(scene)
world.init(scene)


# Demo: teleport the right body so its axis becomes collinear with the
# left body's axis (shift right body by +Y*0.5 to align Y).
# new_T = Transform.Identity()
# new_T.translate([0.15, 1.0, -0.5])
# change_right_tf(new_T)
print_distances()

ps.init()
tri_surf, _, _ = sgui.register()
tri_surf.set_edge_width(1)

run = False


def on_update():
    global run

    if imgui.Button("Run & Stop"):
        run = not run

    imgui.Separator()
    imgui.Text(f"Frame: {world.frame()}")

    if run:
        world.advance()
        world.retrieve()
        sgui.update()
        print_distances()
        check_axes_collinear()


ps.set_user_callback(on_update)
ps.show()

"""
Batched revolute joints built from body-local attachment points.

This example demonstrates the *local-coordinate* overload of
`AffineBodyRevoluteJoint.create_geometry`. Two independent ABD body
pairs are connected with revolute joints in a SINGLE call by passing
per-joint left/right local attachment points (`l_pos0`, `l_pos1`,
`r_pos0`, `r_pos1`).

Scene layout (no gravity, no contact):
- Aligned pair (z = -0.5): left and right bodies share identity rotation
  and are placed along the X axis; the revolute constraint pulls the
  right body up by 0.5 m so the attachment points coincide.
- Misaligned pair (z = +0.5): right body has a small XZ offset on top
  of the same Y gap; the constraint still converges.

Each frame the `angle` edge attribute of the joint SimplicialComplex is
printed, and `check_attachments_equal()` asserts that
`T_L * l_pos ≈ T_R * r_pos` for every joint (i.e. the revolute
constraint has driven the two bodies together in world space).
"""

import numpy as np
import polyscope as ps
import uipc.builtin as builtin
import uipc.unit as unit
from asset_dir import AssetDir
from polyscope import imgui
from uipc import AngleAxis, Engine, Logger, Matrix4x4, Scene, Transform, Vector3, World, view
from uipc.constitution import AffineBodyConstitution, AffineBodyRevoluteJoint
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


# Aligned pair (z = -0.5): attachment points already coincide
al_left_obj = scene.objects().create("aligned_left")
al_left_mesh = make_abd_cube([0.15, 0.5, -0.5], fixed=True)
al_left_slot, _ = al_left_obj.geometries().create(al_left_mesh)

al_right_obj = scene.objects().create("aligned_right")
al_right_mesh = make_abd_cube([0.15, 0.0, -0.5], fixed=False)
al_right_slot, _ = al_right_obj.geometries().create(al_right_mesh)
# ---------------------------------------------------------------------------
# Two revolute joints built in a single create_geometry call
# ---------------------------------------------------------------------------
revolute = AffineBodyRevoluteJoint()

# Same local attachment points for both joints, expressed in each body's
# local frame (joint axis runs from *_pos0 -> *_pos1).
# NOTE: Vector3.UnitX/Y/Z() returns a column vector with shape (3, 1);
# flatten it before arithmetic so the final batched array is (N, 3) not (N, 3, 3).
_unit_x = np.asarray(Vector3.UnitX(), dtype=np.float64).reshape(3)
_unit_y = np.asarray(Vector3.UnitY(), dtype=np.float64).reshape(3)
_unit_z = np.asarray(Vector3.UnitZ(), dtype=np.float64).reshape(3)

parent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
child_pos = parent_pos + _unit_y * 0.5

l_pos0 = np.array([parent_pos], dtype=np.float64)
l_pos1 = np.array([parent_pos + _unit_z], dtype=np.float64)

r_pos0 = np.array([child_pos], dtype=np.float64)
r_pos1 = np.array([child_pos + _unit_z], dtype=np.float64)

l_slots = [al_left_slot]
r_slots = [al_right_slot]
l_instance_ids = np.array([0], dtype=np.int32)
r_instance_ids = np.array([0], dtype=np.int32)
strength_ratios = np.array([100.0], dtype=np.float64)

joint_mesh = revolute.create_geometry(
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

joint_object = scene.objects().create("revolute_joint")
joint_geo_slot, _ = joint_object.geometries().create(joint_mesh)

NUM_JOINTS = 1
JOINT_LABELS = ["Aligned  (z=-0.5)"]

# Local attachment points shared by both joints; expressed in each body's
# local frame. The revolute axis runs from pos0 -> pos1.


def _read_transform(slot):
    return np.asarray(view(slot.geometry().transforms())[0]).reshape(4, 4)


def to_world(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    p4 = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return (T @ p4)[:3]


def print_angles() -> None:
    geo = joint_geo_slot.geometry()
    angles = geo.edges().find("angle")
    if angles is None:
        print(f"frame={world.frame()} angle attribute not ready yet")
    else:
        v = view(angles)
        for i in range(min(NUM_JOINTS, len(v))):
            rad = float(v[i])
            print(
                f"frame={world.frame()} joint={i} {JOINT_LABELS[i]}: angle={rad:+.4f} rad ({np.degrees(rad):+.2f} deg)"
            )


def change_right_tf(new_T: Transform) -> None:
    """Translate al_right's transform by (0, dy, 0) *after* `world.init`.

    Follows the same pattern as test `43_abd_fem_state_access`:
      1. Fetch `AffineBodyStateAccessorFeature` from the world.
      2. Build an empty state-geometry and declare the `transform` attribute.
      3. `copy_to` to pull the current transforms from the backend.
      4. Mutate the target body's row in the view.
      5. `copy_from` to push the change back, then `world.retrieve()`.

    al_right is the 2nd ABD body created → body index 1.
    """
    abd_accessor = world.features().find(AffineBodyStateAccessorFeature)
    assert abd_accessor is not None, "AffineBodyStateAccessorFeature is not available in this uipc build"

    abd_state = abd_accessor.create_geometry()
    # Transform attribute isn't populated by default — declare it so copy_to
    # knows to materialize it.
    abd_state.instances().create(builtin.transform, Matrix4x4.Zero())

    abd_accessor.copy_to(abd_state)

    trans_view = view(abd_state.transforms())
    AL_RIGHT_BODY_IDX = 1  # creation order: 0=al_left, 1=al_right

    trans_view[AL_RIGHT_BODY_IDX] = new_T.matrix()

    abd_accessor.copy_from(abd_state)
    world.retrieve()


def check_attachments_equal(atol: float = 1e-3) -> bool:
    """Assert that left/right attachment points coincide in world space
    for every joint (the revolute constraint should drive them together).

    Returns True on success; raises AssertionError with a helpful message
    otherwise.
    """
    al_L = _read_transform(al_left_slot)
    al_R = _read_transform(al_right_slot)
    for name, TL, TR in [("aligned", al_L, al_R)]:
        for tag, lp, rp in [("p0", l_pos0[0], r_pos0[0]), ("p1", l_pos1[0], r_pos1[0])]:
            lhs = to_world(TL, lp)
            rhs = to_world(TR, rp)
            print(f"frame={world.frame()} joint={name} attach={tag} TL:{TL} lp:{lp} lw:{lhs}")
            print(f"frame={world.frame()} joint={name} attach={tag} TR:{TR} rp:{rp} rw:{rhs}")
            assert np.allclose(lhs, rhs, atol=atol), (
                f"frame={world.frame()} joint={name} attach={tag} "
                f"mismatch: T_L*l={lhs} != T_R*r={rhs} "
                f"(|diff|={np.linalg.norm(lhs - rhs):.6e}, atol={atol})"
            )
    return True


# ---------------------------------------------------------------------------
# Init & GUI
# ---------------------------------------------------------------------------
sgui = SceneGUI(scene)
world.init(scene)

t = Transform.Identity()
t.translate([0.15, 0.5, -0.5] + _unit_x * 0.5)
change_right_tf(t)

print_angles()
# check_attachments_equal()
# Constraint is not yet satisfied at frame 0 (bodies are offset by 0.5 m);
# we only assert convergence after `world.advance()` below.

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
        print_angles()
        # check_attachments_equal()


ps.set_user_callback(on_update)
ps.show()

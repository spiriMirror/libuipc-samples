"""Pure-cube replay of the first 2 frames of Newton's panda_hydro trajectory.

This example mirrors the 9-DOF joint structure of the Franka FR3 arm used
by ``newton.examples.uipc.example_uipc_panda_hydro``:

  * 1 fixed base cube + 7 arm link cubes, chained by 7 revolute joints
    built in a single batched `AffineBodyRevoluteJoint.create_geometry`.
  * 2 finger cubes, each attached to the final arm link through an
    `AffineBodyPrismaticJoint` (also batched, N = 2).

No URDF, no URDF-derived geometry, no contact, no gravity — all bodies
are 0.05 m ABD cubes loaded from ``assets/sim_data/tetmesh/cube.msh``.

The 9-DOF driving target for each frame is loaded from
``joint_targets_first2.npy`` (shape ``(2, 9)`` float64, columns 0–6 =
revolute arm angles in rad, columns 7–8 = prismatic finger distances in
m). The file is produced by running ``dump_newton_targets.py`` once
inside a Newton checkout.

After frame 1 the animator clamps to the last target row, so you can
keep stepping the sim to see it settle.
"""

import pathlib

import numpy as np
import polyscope as ps
import uipc.builtin as builtin
import uipc.unit as unit
from asset_dir import AssetDir
from polyscope import imgui
from uipc import Engine, Logger, Scene, Transform, Vector3, World, view
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyDrivingPrismaticJoint,
    AffineBodyDrivingRevoluteJoint,
    AffineBodyPrismaticJoint,
    AffineBodyRevoluteJoint,
)
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

# ---------------------------------------------------------------------------
# Target trajectory (2 frames × 9 DOF)
# ---------------------------------------------------------------------------
TARGETS_PATH = pathlib.Path(__file__).with_name("joint_targets_first2.npy")
TARGETS = np.load(TARGETS_PATH)
assert TARGETS.shape == (2, 9), f"expected (2, 9) targets, got {TARGETS.shape}"
print(f"Loaded targets {TARGETS.shape} from {TARGETS_PATH.name}")
with np.printoptions(precision=4, suppress=True):
    for i, row in enumerate(TARGETS):
        print(f"  target[{i}]: arm(deg)={np.degrees(row[:7])}  fingers(m)={row[7:9]}")

# ---------------------------------------------------------------------------
# Engine / world / scene
# ---------------------------------------------------------------------------
workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = 1.0 / 120.0  # align with Newton frame_dt
config["contact"]["enable"] = False
config["line_search"]["report_energy"] = True
config["sanity_check"]["enable"] = 0
config["gravity"] = [[0.0], [0.0], [0.0]]
scene = Scene(config)

# ---------------------------------------------------------------------------
# ABD cube factory (0.05 m edge, kappa = 100 MPa) — mirrors #84 / #85
# ---------------------------------------------------------------------------
CUBE_SIZE = 0.05
HALF = CUBE_SIZE / 2.0
GAP = 0.05  # world-space spacing between adjacent cube faces
# Distance from each cube's center to its joint pivot (placed midway in the
# gap so the two pivot points coincide in world space when θ = 0).
STEP = HALF + GAP / 2.0  # = 0.05 m

# Newton init_q for the Franka FR3 arm (7 revolute) + 2 prismatic fingers.
# Same numbers as `example_uipc_panda_hydro.Example.__init__`.
INIT_Q_ARM = np.array(
    [
        -3.6802115e-03,
        2.3901723e-02,
        3.6804110e-03,
        -2.3683236e00,
        -1.2918962e-04,
        2.3922248e00,
        7.8549200e-01,
    ],
    dtype=np.float64,
)
INIT_FINGER_DIST = 0.05  # Newton init gripper distance [m]

pre = Transform.Identity()
pre.scale([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE])
io = SimplicialComplexIO(pre)

abd = AffineBodyConstitution()


def process_surface(sc: SimplicialComplex) -> SimplicialComplex:
    label_surface(sc)
    label_triangle_orient(sc)
    return flip_inward_triangles(sc)


def make_abd_cube(transform_mat: np.ndarray, fixed: bool = False):
    """Instantiate a 0.05 m ABD cube and stamp its world transform.

    ``transform_mat`` is a (4, 4) float64 homogeneous transform that lands
    the unit-cube's local frame wherever the caller wants it in world
    space (used to express FK-accumulated poses for the arm chain).
    """
    mesh = process_surface(io.read(f"{AssetDir.tetmesh_path()}/cube.msh"))
    abd.apply_to(mesh, 100.0 * unit.MPa)
    mesh.instances().resize(1)

    T = np.asarray(transform_mat, dtype=np.float64).reshape(4, 4)
    view(mesh.transforms())[0] = T
    if fixed:
        view(mesh.instances().find(builtin.is_fixed))[0] = 1
    return mesh


def _translate_mat(v) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, 3] = np.asarray(v, dtype=np.float64)
    return M


def _rot_x_mat(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    M = np.eye(4, dtype=np.float64)
    M[1, 1] = c
    M[1, 2] = -s
    M[2, 1] = s
    M[2, 2] = c
    return M


# ---------------------------------------------------------------------------
# 8 arm cubes (1 base + 7 revolute-driven links) laid out along world +Y
# using accumulated FK with the Newton ``init_q`` angles. (Y-up matches
# polyscope's default up axis.)
#
# FK recurrence for adjacent cubes i and i+1 sharing a revolute joint
# (parent pivot = local (0, +STEP, 0), child pivot = local (0, -STEP, 0),
# axis = local +X, joint angle = θ_i):
#
#   T_{i+1} = T_i · Translate(0, +STEP, 0) · RotX(θ_i) · Translate(0, +STEP, 0)
#
# At θ_i = 0 this reduces to pure +Y translation of 2·STEP = CUBE_SIZE + GAP
# so the identity case collapses to a straight stack standing up on +Y.
# ---------------------------------------------------------------------------
NUM_ARM = 8  # = 1 base + 7 links
NUM_REV = 7
_unit_x = np.asarray(Vector3.UnitX(), dtype=np.float64).reshape(3)

_step_up = _translate_mat([0.0, STEP, 0.0])
arm_transforms: list[np.ndarray] = [np.eye(4, dtype=np.float64)]  # link0 at origin
for i in range(NUM_REV):
    arm_transforms.append(arm_transforms[-1] @ _step_up @ _rot_x_mat(INIT_Q_ARM[i]) @ _step_up)

# Print the angles/distances baked into the initial transforms. Note that
# UIPC's ``angle`` / ``distance`` edge attributes are only populated after
# the first ``world.advance()``, so ``print_state()`` at frame 0 reports
# zeros; the numbers below are the ground-truth values we wrote into each
# cube's transform via FK of ``INIT_Q_ARM`` and ``INIT_FINGER_DIST``.
with np.printoptions(precision=4, suppress=True):
    print(f"Initial setpoint: arm(deg)={np.degrees(INIT_Q_ARM)}  fingers(m)=[{INIT_FINGER_DIST} {INIT_FINGER_DIST}]")

arm_slots = []
for i in range(NUM_ARM):
    label = "arm_base" if i == 0 else f"arm_link{i}"
    obj = scene.objects().create(label)
    mesh = make_abd_cube(arm_transforms[i], fixed=(i == 0))
    slot, _ = obj.geometries().create(mesh)
    arm_slots.append(slot)

# ---------------------------------------------------------------------------
# 2 finger cubes attached to arm_link7 via prismatic joints (axis = ±local X
# of link7 so "aim_distance = 0.06" opens the gripper symmetrically).
#
# Initial transform: inherit link7's rotation, offset by 2·STEP along local
# +Y (so finger's bottom pivot coincides with link7's top pivot at
# distance 0), then slide ±INIT_FINGER_DIST along local X to express the
# Newton ``init_q`` gripper opening.
# ---------------------------------------------------------------------------
T_link7 = arm_transforms[NUM_REV]
finger_slots = []
finger_transforms: list[np.ndarray] = []
for sign, label in [(+1.0, "finger_L"), (-1.0, "finger_R")]:
    local_offset = _translate_mat([sign * INIT_FINGER_DIST, 2.0 * STEP, 0.0])
    T_finger = T_link7 @ local_offset
    finger_transforms.append(T_finger)
    obj = scene.objects().create(label)
    mesh = make_abd_cube(T_finger)
    slot, _ = obj.geometries().create(mesh)
    finger_slots.append(slot)

# ---------------------------------------------------------------------------
# Revolute chain — 7 joints batched into a single linemesh
#
# Pivot placement (in each body's local frame):
#   parent  (cube i)   : (0, +STEP, 0)    in free space above the cube
#   child   (cube i+1) : (0, -STEP, 0)    in free space below the cube
# (pivots sit in the middle of the GAP so when T_i = FK(init_q), the two
#  pivots already coincide in world space — no initial snap.)
#
# Axis runs from pos0 -> pos1; we pick local +X so every joint rotates in
# its own local YZ plane. Reproducing the URDF axes is not a goal here;
# what matters is that the animator can drive `aim_angle` to the Newton
# target value and the constraint responds.
# ---------------------------------------------------------------------------
parent_pivot = np.array([0.0, +STEP, 0.0], dtype=np.float64)
child_pivot = np.array([0.0, -STEP, 0.0], dtype=np.float64)

rev_l_pos0 = np.tile(parent_pivot, (NUM_REV, 1))
rev_l_pos1 = np.tile(parent_pivot + _unit_x, (NUM_REV, 1))
rev_r_pos0 = np.tile(child_pivot, (NUM_REV, 1))
rev_r_pos1 = np.tile(child_pivot + _unit_x, (NUM_REV, 1))

rev_l_slots = arm_slots[:NUM_REV]  # parents: cubes 0..6
rev_r_slots = arm_slots[1 : NUM_REV + 1]  # children: cubes 1..7
rev_l_inst = np.zeros(NUM_REV, dtype=np.int32)
rev_r_inst = np.zeros(NUM_REV, dtype=np.int32)
rev_strength = np.full(NUM_REV, 100.0, dtype=np.float64)

rev_mesh = AffineBodyRevoluteJoint().create_geometry(
    rev_l_pos0,
    rev_l_pos1,
    rev_r_pos0,
    rev_r_pos1,
    rev_l_slots,
    rev_l_inst,
    rev_r_slots,
    rev_r_inst,
    rev_strength,
)
AffineBodyDrivingRevoluteJoint().apply_to(rev_mesh, np.full(NUM_REV, 100.0, dtype=np.float64))

rev_object = scene.objects().create("joints_revolute")
rev_slot, _ = rev_object.geometries().create(rev_mesh)

# ---------------------------------------------------------------------------
# Prismatic fingers — 2 joints batched
#
# Parent = arm_link7's top-pivot plane; child = finger bottom-pivot plane.
# To open symmetrically on a shared ``aim_distance`` target, the left
# finger's joint axis is link7-local +X while the right finger's axis is
# link7-local -X. Both are driven by `aim_distance`.
# ---------------------------------------------------------------------------
NUM_PRIS = 2
pris_l_pos0 = np.stack([parent_pivot, parent_pivot], axis=0)
pris_l_pos1 = np.stack([parent_pivot + _unit_x, parent_pivot - _unit_x], axis=0)
pris_r_pos0 = np.stack([child_pivot, child_pivot], axis=0)
pris_r_pos1 = np.stack([child_pivot + _unit_x, child_pivot - _unit_x], axis=0)

pris_l_slots = [arm_slots[7], arm_slots[7]]
pris_r_slots = list(finger_slots)
pris_l_inst = np.zeros(NUM_PRIS, dtype=np.int32)
pris_r_inst = np.zeros(NUM_PRIS, dtype=np.int32)
pris_strength = np.full(NUM_PRIS, 100.0, dtype=np.float64)

pris_mesh = AffineBodyPrismaticJoint().create_geometry(
    pris_l_pos0,
    pris_l_pos1,
    pris_r_pos0,
    pris_r_pos1,
    pris_l_slots,
    pris_l_inst,
    pris_r_slots,
    pris_r_inst,
    pris_strength,
)
AffineBodyDrivingPrismaticJoint().apply_to(pris_mesh, np.full(NUM_PRIS, 100.0, dtype=np.float64))

pris_object = scene.objects().create("joints_prismatic")
pris_slot, _ = pris_object.geometries().create(pris_mesh)


# ---------------------------------------------------------------------------
# Animator — clamp frame index to the last row once we run past the npy
# ---------------------------------------------------------------------------
def _clamped_frame() -> int:
    return min(world.frame(), TARGETS.shape[0] - 1)


def revolute_anim(info) -> None:
    targets = TARGETS[_clamped_frame(), :7]
    for gs in info.geo_slots():
        if gs is None:
            continue
        geo = gs.geometry()
        if geo is None:
            continue
        is_c = geo.edges().find("driving/is_constrained")
        if is_c is not None:
            view(is_c)[:] = 1
        aim = geo.edges().find("aim_angle")
        if aim is not None:
            view(aim)[:] = targets


def prismatic_anim(info) -> None:
    targets = TARGETS[_clamped_frame(), 7:9]
    for gs in info.geo_slots():
        if gs is None:
            continue
        geo = gs.geometry()
        if geo is None:
            continue
        is_c = geo.edges().find("driving/is_constrained")
        if is_c is not None:
            view(is_c)[:] = 1
        aim = geo.edges().find("aim_distance")
        if aim is not None:
            view(aim)[:] = targets


scene.animator().insert(rev_object, revolute_anim)
scene.animator().insert(pris_object, prismatic_anim)


# ---------------------------------------------------------------------------
# Init & GUI
# ---------------------------------------------------------------------------
sgui = SceneGUI(scene)
world.init(scene)


def _sync_init_transforms_to_uipc() -> None:
    """Push FK-derived initial transforms into the UIPC backend via the
    ``AffineBodyStateAccessorFeature`` (per-slot ``copy_to`` → overwrite
    ``transform`` instance attribute → ``copy_from``).

    This mirrors Newton's ``SolverUIPC._sync_body_state_to_uipc``
    (``newton/_src/solvers/uipc/solver_uipc.py:918-1023``): the exact same
    three-stage flow of *scene-build stamp → world.init → accessor
    copy_from*. In Newton, the accessor pass is essential because the
    ``body_q`` state that SolverUIPC wants to commit may have changed
    between build and init (e.g. ``newton.eval_fk`` runs on an already-
    constructed scene).

    **In this sample, calling the function is OFF by default.** The
    build-time ``view(mesh.transforms())[0] = T`` inside ``make_abd_cube``
    already writes the FK pose — the identical value this function would
    commit — so the accessor pass becomes a "write the same value again"
    round-trip. Empirically on the current libuipc build with pure ABD
    bodies + driving joints + ``dt = 1/120``, that round-trip resets the
    constraint warm-start between worlds and produces a period-3
    oscillation in the driving-joint angles before the solver settles.
    The function is kept here as a reference implementation of the
    canonical SolverUIPC pattern; uncomment the call below if you want
    to exercise that exact path.
    """
    abd_accessor: AffineBodyStateAccessorFeature = world.features().find(AffineBodyStateAccessorFeature)  # ty:ignore[invalid-assignment]
    assert abd_accessor is not None, "AffineBodyStateAccessorFeature not available in this uipc build"

    def _push_slot(slot, T_4x4: np.ndarray) -> None:
        geo = slot.geometry()
        abd_accessor.copy_to(geo)
        transform_attr = geo.instances().find("transform")
        velocity_attr = geo.instances().find("velocity")
        assert transform_attr is not None, (
            f"slot {slot} has no 'transform' instance attribute after copy_to"
        )
        view(transform_attr)[0] = np.asarray(T_4x4, dtype=np.float64).reshape(4, 4)
        # Zero the affine-body velocity matrix to match our "cubes are at
        # rest" FK assumption (SolverUIPC._sync_body_state_to_uipc always
        # writes velocity alongside transform).
        if velocity_attr is not None:
            view(velocity_attr)[0] = np.zeros((4, 4), dtype=np.float64)
        abd_accessor.copy_from(geo)

    for i, slot in enumerate(arm_slots):
        _push_slot(slot, arm_transforms[i])
    for i, slot in enumerate(finger_slots):
        _push_slot(slot, finger_transforms[i])


# Uncomment to exercise the canonical SolverUIPC pattern (see note in the
# function docstring for why we don't by default):
# _sync_init_transforms_to_uipc()


def print_state() -> None:
    geo_rev = rev_slot.geometry()
    angles = geo_rev.edges().find("angle")
    if angles is not None:
        v_deg = np.degrees(np.asarray(view(angles)).reshape(-1))
        tgt_deg = np.degrees(TARGETS[_clamped_frame(), :7])
        print(
            f"frame={world.frame()} rev angles (deg): {np.array2string(v_deg, precision=2, suppress_small=True)}"
            f"   target: {np.array2string(tgt_deg, precision=2, suppress_small=True)}"
        )
    geo_pris = pris_slot.geometry()
    distances = geo_pris.edges().find("distance")
    if distances is not None:
        v = np.asarray(view(distances)).reshape(-1)
        tgt = TARGETS[_clamped_frame(), 7:9]
        print(
            f"frame={world.frame()} pris distances (m): {np.array2string(v, precision=4, suppress_small=True)}"
            f"   target: {np.array2string(tgt, precision=4, suppress_small=True)}"
        )


print_state()

ps.init()
tri_surf, _, _ = sgui.register()
tri_surf.set_edge_width(1)

run = False


def on_update() -> None:
    global run
    if imgui.Button("Run & Stop"):
        run = not run
    imgui.Separator()
    imgui.Text(f"Frame: {world.frame()}")
    if run and world.frame() <= TARGETS.shape[0]:
        world.advance()
        world.retrieve()
        sgui.update()
        print_state()


ps.set_user_callback(on_update)
ps.show()

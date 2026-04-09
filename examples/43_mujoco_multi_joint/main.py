"""
MuJoCo + libuipc multi-joint chain coupling example.

Chain topology:  [Body0 (free)] --revolute-- [Body1]

MuJoCo provides the full 7x7 configuration-dependent mass matrix (M(q))
and predicted displacement for a free-revolute chain.
libuipc handles IPC contact resolution (ground plane).

DOF layout (7 total):
  0-5: FreeJoint (TransX/Y/Z, RotX/Y/Z) on Body 0
  6:   Revolute (around X axis) between Body 0 and Body 1
"""

import mujoco
import numpy as np
import polyscope as ps
import uipc.builtin as builtin
from asset_dir import AssetDir
from polyscope import imgui
from uipc import Animation, Engine, Logger, Scene, Timer, Transform, Vector3, Vector12, World, view
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyFreeJoint,
    AffineBodyRevoluteJoint,
    ExternalArticulationConstraint,
)
from uipc.geometry import SimplicialComplex, SimplicialComplexIO, affine_body, ground, label_surface
from uipc.gui import SceneGUI
from uipc.unit import GPa, MPa

# MuJoCo model: 2 boxes, matching libuipc layout.
# Body 0 at (0, 1, -0.4): free joint
# Body 1 at (0, 1,  0.4): revolute around X axis
# Hinge pivot at the interface (0, 1, 0) between the two cubes.
# body1 frame = hinge location; geom offset places the mass center at (0, 1, 0.4).
MJCF_MODEL = """
<mujoco>
  <option gravity="0 -9.81 0" timestep="0.01"/>
  <worldbody>
    <body name="body0" pos="0 1 -0.4">
      <freejoint name="free"/>
      <geom type="box" size="0.2 0.2 0.2" mass="1.0" contype="0" conaffinity="0"/>
      <body name="body1" pos="0 0 0.4">
        <joint name="revolute" type="hinge" axis="1 0 0"/>
        <geom type="box" size="0.2 0.2 0.2" mass="1.0" pos="0 0 0.4" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

N_FREE = 6
N_REVOLUTE = 1
N_DOFS = N_FREE + N_REVOLUTE  # 7

Timer.enable_all()
Logger.set_level(Logger.Level.Warn)

this_output_path = AssetDir.output_path(__file__)
trimesh_path = AssetDir.trimesh_path()

# ------------------------------------------------------------------
# MuJoCo setup
# ------------------------------------------------------------------
mj_model = mujoco.MjModel.from_xml_string(MJCF_MODEL)
mj_data = mujoco.MjData(mj_model)

dt = mj_model.opt.timestep
assert mj_model.nv == N_DOFS, f"Expected {N_DOFS} DOFs, got {mj_model.nv}"

INIT_REV_OMEGA = -2.0  # rad/s on revolute joint
mj_data.qvel[6] = INIT_REV_OMEGA

mujoco.mj_forward(mj_model, mj_data)
M_init = np.zeros((N_DOFS, N_DOFS))
mujoco.mj_fullM(mj_model, M_init, mj_data.qM)
print(f"MuJoCo initial mass matrix ({N_DOFS}x{N_DOFS}):")
print(np.array2string(M_init, precision=4, suppress_small=True))

mj_state = {
    "M": M_init.copy(),
    "delta_theta_tilde": np.zeros(N_DOFS),
    "transforms": None,
}

# ------------------------------------------------------------------
# libuipc setup
# ------------------------------------------------------------------
engine = Engine("cuda", this_output_path)
world = World(engine)

config = Scene.default_config()
config["gravity"] = [[0.0], [0.0], [0.0]]
config["contact"]["enable"] = True
config["newton"]["velocity_tol"] = 0.05
config["newton"]["transrate_tol"] = 0.1
config["linear_system"]["tol_rate"] = 1e-5
config["contact"]["d_hat"] = 0.001
config["dt"] = dt
scene = Scene(config)

scene.contact_tabular().default_model(0.05, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()

abd = AffineBodyConstitution()

pre_transform = Transform.Identity()
pre_transform.scale(0.4)
io = SimplicialComplexIO(pre_transform)

# 2 body instances
body_object = scene.objects().create("bodies")
abd_mesh = io.read(f"{trimesh_path}/cube.obj")
abd_mesh.instances().resize(2)
label_surface(abd_mesh)
abd.apply_to(abd_mesh, 100.0 * MPa)
default_element.apply_to(abd_mesh)

trans_view = view(abd_mesh.transforms())

t0 = Transform.Identity()
t0.translate(Vector3.Values([0.0, 1.0, -0.4]))
trans_view[0] = t0.matrix()

t1 = Transform.Identity()
t1.translate(Vector3.Values([0.0, 1.0, 0.4]))
trans_view[1] = t1.matrix()

is_fixed = abd_mesh.instances().find(builtin.is_fixed)
is_fixed_view = view(is_fixed)
is_fixed_view[0] = 0
is_fixed_view[1] = 0

ref_dof_prev = abd_mesh.instances().create("ref_dof_prev", Vector12.Zero())
ref_dof_prev_view = view(ref_dof_prev)
ref_dof_prev_view[:] = affine_body.transform_to_q(view(abd_mesh.transforms()))

external_kinetic = abd_mesh.instances().find(builtin.external_kinetic)
view(external_kinetic)[:] = 1

geo_slot, rest_geo_slot = body_object.geometries().create(abd_mesh)

# Ground plane
ground_object = scene.objects().create("ground")
ground_object.geometries().create(ground())


def mujoco_geom_to_4x4(geom_id):
    """Build a 4x4 homogeneous transform from MuJoCo geom frame."""
    m = np.eye(4)
    m[:3, :3] = mj_data.geom_xmat[geom_id].reshape(3, 3)
    m[:3, 3] = mj_data.geom_xpos[geom_id]
    return m


def update_ref_dof_prev(info: Animation.UpdateInfo):
    if mj_state["transforms"] is None:
        return
    geo: SimplicialComplex = info.geo_slots()[0].geometry()
    rdp = geo.instances().find("ref_dof_prev")
    rdp_view = view(rdp)
    # Use single-matrix transform_to_q per body to avoid the batched path,
    # which does raw memory reinterpret (as_span_of) and misinterprets
    # numpy row-major data as Eigen column-major, transposing the rotation.
    for i in range(N_BODIES):
        rdp_view[i] = affine_body.transform_to_q(mj_state["transforms"][i])


scene.animator().insert(body_object, update_ref_dof_prev)

# ------------------------------------------------------------------
# Joints: FreeJoint + Revolute
# ------------------------------------------------------------------

# FreeJoint on instance 0
abfj = AffineBodyFreeJoint()
free_joint_mesh = abfj.create_geometry([geo_slot], np.array([0], dtype=np.int32))
fj_object = scene.objects().create("free_joint")
free_joint_slot, _ = fj_object.geometries().create(free_joint_mesh)

# Revolute joint: instance 0 -> instance 1 (axis along +X in libuipc)
# Pivot at (0, 1, 0) = interface between the two cubes, matching MuJoCo hinge.
abrj = AffineBodyRevoluteJoint()
joint_pivot = np.array([0.0, 1.0, 0.0], dtype=np.float32)
axis_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
rev_pos0 = np.array([joint_pivot], dtype=np.float32)
rev_pos1 = np.array([joint_pivot + axis_dir], dtype=np.float32)
revolute_mesh = abrj.create_geometry(
    rev_pos0, rev_pos1, [geo_slot], [0], [geo_slot], [1], [100.0]
)
# label_surface(revolute_mesh)
revolute_object = scene.objects().create("revolute_joint")
revolute_slot, _ = revolute_object.geometries().create(revolute_mesh)

# ------------------------------------------------------------------
# ExternalArticulationConstraint: 6 free + 1 revolute = 7 DOFs
# ------------------------------------------------------------------
eac = ExternalArticulationConstraint()

joint_geos = [free_joint_slot] * N_FREE + [revolute_slot]
indices = list(range(N_FREE)) + [0]
articulation = eac.create_geometry(joint_geos, indices)

mass_attr = articulation["joint_joint"].find("mass")
view(mass_attr)[:] = M_init.flatten()

articulation_object = scene.objects().create("articulation")
articulation_slot, _ = articulation_object.geometries().create(articulation)


def update_articulation(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()

    dtv = view(geo["joint"].find("delta_theta_tilde"))
    dtv[:] = mj_state["delta_theta_tilde"]

    mass_v = view(geo["joint_joint"].find("mass"))
    # 4 precision
    print("M:\n", np.array2string(mj_state["M"], precision=2))
    mass_v[:] = mj_state["M"].flatten()


scene.animator().insert(articulation_object, update_articulation)

# ------------------------------------------------------------------
# Init world + GUI
# ------------------------------------------------------------------
world.init(scene)
sgui = SceneGUI(scene, "split")

# Load ghost mesh template (same cube, same scale) for MuJoCo prediction overlay
ghost_sc = io.read(f"{trimesh_path}/cube.obj")
label_surface(ghost_sc)
ghost_verts_local = ghost_sc.positions().view().reshape(-1, 3).copy().astype(np.float64)
ghost_tris = ghost_sc.triangles().topo().view().reshape(-1, 3).copy()

ps.init()
ps.set_ground_plane_height(0.0)
sgui.register()
sgui.set_edge_width(1)

N_BODIES = 2
MJ_GEOM_IDS = [0, 1]
ghost_meshes: list[ps.SurfaceMesh] = []
for i in range(N_BODIES):
    R = mj_data.geom_xmat[MJ_GEOM_IDS[i]].reshape(3, 3)
    t = mj_data.geom_xpos[MJ_GEOM_IDS[i]]
    verts = (ghost_verts_local @ R.T) + t
    m = ps.register_surface_mesh(f"mj_ghost_{i}", verts, ghost_tris)
    m.set_color((0.2, 0.6, 1.0))
    m.set_transparency(0.4)
    ghost_meshes.append(m)

gui = {
    "run": False,
    "mujoco_only": False,
    "mj_frame": 0,
}


def rotvec_to_quat(rotvec):
    """Convert rotation vector to quaternion [w, x, y, z]."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / angle
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


def quat_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_rotvec(q):
    """Convert quaternion [w, x, y, z] to rotation vector."""
    w, x, y, z = q
    xyz = np.array([x, y, z])
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-10:
        return np.zeros(3)
    half_angle = np.arctan2(sin_half, w)
    return 2.0 * half_angle * xyz / sin_half


def quat_conjugate(q):
    """Conjugate (= inverse for unit quaternion) [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def mujoco_xform_to_4x4(xpos, xmat):
    """Build a 4x4 homogeneous transform from MuJoCo geom_xpos / geom_xmat."""
    m = np.eye(4)
    m[:3, :3] = xmat.reshape(3, 3)
    m[:3, 3] = xpos
    return m


def on_update():
    if imgui.Button("Run & Stop"):
        gui["run"] = not gui["run"]

    _, gui["mujoco_only"] = imgui.Checkbox("MuJoCo Only", gui["mujoco_only"])

    imgui.Separator()
    mode_str = "MuJoCo Only" if gui["mujoco_only"] else "MuJoCo + libuipc"
    imgui.Text(f"{mode_str}")
    imgui.Text("Body 0 (free) -> Body 1 (revolute)")
    imgui.Text(f"dt = {dt:.4f}s  |  Total DOFs = {N_DOFS}")
    imgui.Separator()

    imgui.Text("MuJoCo Mass Matrix M(q) diagonal:")
    M = mj_state["M"]
    imgui.Text(f"  free trans: [{M[0,0]:.3f}, {M[1,1]:.3f}, {M[2,2]:.3f}]")
    imgui.Text(f"  free rot:   [{M[3,3]:.4f}, {M[4,4]:.4f}, {M[5,5]:.4f}]")
    imgui.Text(f"  revolute:   {M[6,6]:.4f}")

    imgui.Separator()
    nq = mj_model.nq
    imgui.Text(f"MuJoCo qpos ({nq}): {np.array2string(mj_data.qpos, precision=3)}")
    imgui.Text(f"MuJoCo qvel ({N_DOFS}): {np.array2string(mj_data.qvel, precision=3)}")

    imgui.Separator()
    frame = gui["mj_frame"] if gui["mujoco_only"] else world.frame()
    imgui.Text(f"Frame: {frame}")
    imgui.Text(f"Time: {frame * dt:.3f}s")

    if not gui["run"]:
        return

    if gui["mujoco_only"]:
        # --- MuJoCo only: step MuJoCo, write transforms to libuipc scene ---
        mujoco.mj_step(mj_model, mj_data)
        gui["mj_frame"] += 1

        # Update ghost meshes
        for i in range(N_BODIES):
            R = mj_data.geom_xmat[MJ_GEOM_IDS[i]].reshape(3, 3)
            t = mj_data.geom_xpos[MJ_GEOM_IDS[i]]
            ghost_meshes[i].update_vertex_positions((ghost_verts_local @ R.T) + t)

        # Write MuJoCo body transforms into libuipc scene so Polyscope displays them
        geo: SimplicialComplex = geo_slot.geometry()
        t_view = view(geo.transforms())
        for i in range(N_BODIES):
            t_view[i] = mujoco_xform_to_4x4(
                mj_data.geom_xpos[MJ_GEOM_IDS[i]],
                mj_data.geom_xmat[MJ_GEOM_IDS[i]],
            )
        sgui.update()
    else:
        # --- Coupled mode: MuJoCo + libuipc ---
        mujoco.mj_forward(mj_model, mj_data)
        mj_state["transforms"] = np.array(
            [mujoco_geom_to_4x4(MJ_GEOM_IDS[i]) for i in range(N_BODIES)]
        )
        qpos_prev = mj_data.qpos.copy()
        mujoco.mj_step(mj_model, mj_data)

        # Update ghost meshes with MuJoCo's uncorrected prediction
        for i in range(N_BODIES):
            R = mj_data.geom_xmat[MJ_GEOM_IDS[i]].reshape(3, 3)
            t = mj_data.geom_xpos[MJ_GEOM_IDS[i]]
            ghost_meshes[i].update_vertex_positions((ghost_verts_local @ R.T) + t)

        M_dense = np.zeros((N_DOFS, N_DOFS))
        mujoco.mj_fullM(mj_model, M_dense, mj_data.qM)

        eigvals = np.linalg.eigvalsh(M_dense)
        if np.any(eigvals <= 0):
            print(f"[WARN] Frame {world.frame()}: M is NOT SPD!  eigenvalues = {eigvals}")
        else:
            print(f"[OK]   Frame {world.frame()}: M is SPD.  min_eigval = {eigvals[0]:.6e}")

        mj_state["M"] = M_dense

        dt_tilde = np.zeros(N_DOFS)
        dt_tilde[:3] = mj_data.qpos[:3] - qpos_prev[:3]
        q_diff = quat_multiply(mj_data.qpos[3:7], quat_conjugate(qpos_prev[3:7]))
        q_diff /= np.linalg.norm(q_diff)
        dt_tilde[3:6] = quat_to_rotvec(q_diff)
        dt_tilde[6] = mj_data.qpos[7] - qpos_prev[7]
        mj_state["delta_theta_tilde"] = dt_tilde

        # IPC step
        world.advance()
        world.retrieve()

        # Read IPC result and sync back to MuJoCo
        art_geo = articulation_slot.geometry()
        delta_theta = np.array(art_geo["joint"].find("delta_theta").view())

        mj_data.qpos[:3] = qpos_prev[:3] + delta_theta[:3]
        dq = rotvec_to_quat(delta_theta[3:6])
        mj_data.qpos[3:7] = quat_multiply(dq, qpos_prev[3:7])
        mj_data.qpos[3:7] /= np.linalg.norm(mj_data.qpos[3:7])
        mj_data.qpos[7] = qpos_prev[7] + delta_theta[6]
        # mj_data.qvel[:] = delta_theta / dt

        sgui.update()


ps.set_user_callback(on_update)
ps.show()

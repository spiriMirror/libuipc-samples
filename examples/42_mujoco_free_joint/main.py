"""
MuJoCo + libuipc FreeJoint coupling example.

Uses MuJoCo as a physics oracle to provide the 6x6 joint-space mass matrix
and predicted displacement for a free-joint box. libuipc handles IPC contact
resolution while MuJoCo supplies the inertial properties.

Coupling protocol (per frame):
  1. mj_step     -> integrate gravity into qvel, advance MuJoCo
  2. mj_fullM    -> extract dense 6x6 mass matrix (computed during step)
  3. delta_theta_tilde = qvel * dt  (now includes gravity)
  4. Write M and delta_theta_tilde to IPC articulation geometry
  5. world.advance() + world.retrieve()
  6. Read delta_theta from IPC
  7. Sync state back to MuJoCo qpos/qvel (IPC is authoritative)
"""

import mujoco
import numpy as np
import polyscope as ps
import uipc.builtin as builtin
from asset_dir import AssetDir
from polyscope import imgui
from uipc import AngleAxis, Animation, Engine, Logger, Scene, Timer, Transform, Vector3, Vector12, World, view
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyFreeJoint,
    ExternalArticulationConstraint,
)
from uipc.geometry import SimplicialComplex, SimplicialComplexIO, affine_body, ground, label_surface
from uipc.gui import SceneGUI
from uipc.unit import GPa, MPa

MJCF_MODEL = """
<mujoco>
  <option timestep="0.01" gravity="0 -9.8 0"/>
  <worldbody>
    <body name="box" pos="0 1 0">
      <freejoint name="free"/>
      <geom type="box" size="0.2 0.2 0.2" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

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
n_dofs = 6

INIT_ROT_Y = np.pi / 4  # 45 degrees around Y axis
INIT_OMEGA_Y = 2.0      # rad/s angular velocity around Y axis
half = INIT_ROT_Y / 2.0
mj_data.qpos[3:7] = [np.cos(half), 0.0, np.sin(half), 0.0]  # [w, x, y, z]
mj_data.qvel[4] = INIT_OMEGA_Y  # qvel layout: [vx, vy, vz, wx, wy, wz]

mujoco.mj_forward(mj_model, mj_data)
M_init = np.zeros((n_dofs, n_dofs))
mujoco.mj_fullM(mj_model, M_init, mj_data.qM)
print("MuJoCo initial mass matrix:")
print(M_init)

mj_state = {
    "M": M_init.copy(),
    "delta_theta_tilde": np.zeros(n_dofs),
}

# ------------------------------------------------------------------
# libuipc setup
# ------------------------------------------------------------------
engine = Engine("cuda", this_output_path)
world = World(engine)

config = Scene.default_config()
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = True
config["newton"]["velocity_tol"] = 0.1
config["newton"]["transrate_tol"] = 10
config["linear_system"]["tol_rate"] = 1e-4
config["contact"]["d_hat"] = 0.001
config["dt"] = dt
scene = Scene(config)

scene.contact_tabular().default_model(0.05, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()

abd = AffineBodyConstitution()

pre_transform = Transform.Identity()
pre_transform.scale(0.4)
io = SimplicialComplexIO(pre_transform)

body_object = scene.objects().create("body")
abd_mesh = io.read(f"{trimesh_path}/cube.obj")
label_surface(abd_mesh)
abd.apply_to(abd_mesh, 100.0 * MPa)
default_element.apply_to(abd_mesh)

t0 = Transform.Identity()
t0.translate(Vector3.Values([0.0, 1.0, 0.0]))
t0.rotate(AngleAxis(INIT_ROT_Y, Vector3.UnitY()))
trans_view = view(abd_mesh.transforms())
trans_view[0] = t0.matrix()

is_fixed = abd_mesh.instances().find(builtin.is_fixed)
view(is_fixed)[0] = 0

ref_dof_prev = abd_mesh.instances().create("ref_dof_prev", Vector12.Zero())
ref_dof_prev_view = view(ref_dof_prev)
ref_dof_prev_view[:] = affine_body.transform_to_q(view(abd_mesh.transforms()))

external_kinetic = abd_mesh.instances().find(builtin.external_kinetic)
view(external_kinetic)[0] = 1

geo_slot, rest_geo_slot = body_object.geometries().create(abd_mesh)

# Reference body: plain affine body falling under libuipc gravity (no MuJoCo coupling)
ref_object = scene.objects().create("reference_body")
ref_mesh = io.read(f"{trimesh_path}/cube.obj")
label_surface(ref_mesh)
abd.apply_to(ref_mesh, 100.0 * MPa)
default_element.apply_to(ref_mesh)

t_ref = Transform.Identity()
t_ref.translate(Vector3.Values([1.0, 1.0, 0.0]))
view(ref_mesh.transforms())[0] = t_ref.matrix()

view(ref_mesh.instances().find(builtin.is_fixed))[0] = 0

ref_object.geometries().create(ref_mesh)

# Ground plane for contact
ground_object = scene.objects().create("ground")
ground_object.geometries().create(ground())


def update_ref_dof_prev(info: Animation.UpdateInfo):
    geo: SimplicialComplex = info.geo_slots()[0].geometry()
    rdp = geo.instances().find("ref_dof_prev")
    rdp_view = view(rdp)
    t_view = view(geo.transforms())
    rdp_view[:] = affine_body.transform_to_q(t_view)


scene.animator().insert(body_object, update_ref_dof_prev)

# ------------------------------------------------------------------
# FreeJoint + ExternalArticulationConstraint
# ------------------------------------------------------------------
abfj = AffineBodyFreeJoint()
free_joint_mesh = abfj.create_geometry([geo_slot], np.array([0], dtype=np.int32))
fj_object = scene.objects().create("free_joint")
free_joint_slot, _ = fj_object.geometries().create(free_joint_mesh)

eac = ExternalArticulationConstraint()
joint_geos = [free_joint_slot] * n_dofs
indices = list(range(n_dofs))
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
    mass_v[:] = mj_state["M"].flatten()


scene.animator().insert(articulation_object, update_articulation)

# ------------------------------------------------------------------
# Init world + GUI
# ------------------------------------------------------------------
world.init(scene)
sgui = SceneGUI(scene, "split")

ps.init()
ps.set_ground_plane_height(-1.0)
sgui.register()
sgui.set_edge_width(1)

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


gui = {
    "run": False,
    "mujoco_only": True,
    "mj_frame": 0,
}


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
    imgui.Text("Left (z=0): MuJoCo-coupled via ExternalArticulation")
    imgui.Text("Right (x=1): native libuipc free body (reference)")
    imgui.Text(f"dt = {dt:.4f}s")
    imgui.Separator()

    imgui.Text("MuJoCo Mass Matrix (diagonal):")
    M = mj_state["M"]
    imgui.Text(f"  trans: [{M[0,0]:.4f}, {M[1,1]:.4f}, {M[2,2]:.4f}]")
    imgui.Text(f"  rot:   [{M[3,3]:.6f}, {M[4,4]:.6f}, {M[5,5]:.6f}]")

    imgui.Separator()
    imgui.Text(f"MuJoCo qpos: {np.array2string(mj_data.qpos, precision=3)}")
    imgui.Text(f"MuJoCo qvel: {np.array2string(mj_data.qvel, precision=3)}")

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

        # Write MuJoCo body transform into libuipc scene so Polyscope displays it
        geo: SimplicialComplex = geo_slot.geometry()
        t_view = view(geo.transforms())
        t_view[0] = mujoco_xform_to_4x4(
            mj_data.geom_xpos[0], mj_data.geom_xmat[0]
        )
        sgui.update()
    else:
        # --- Coupled mode: MuJoCo + libuipc ---
        qpos_prev = mj_data.qpos.copy()
        mujoco.mj_step(mj_model, mj_data)

        M_dense = np.zeros((n_dofs, n_dofs))
        mujoco.mj_fullM(mj_model, M_dense, mj_data.qM)
        mj_state["M"] = M_dense
        mj_state["delta_theta_tilde"] = mj_data.qvel.copy() * dt

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
        mj_data.qvel[:] = delta_theta / dt

        sgui.update()
        Timer.report()


ps.set_user_callback(on_update)
ps.show()

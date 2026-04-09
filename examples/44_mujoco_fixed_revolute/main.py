"""
MuJoCo + libuipc: Fixed body + revolute joint isolation test.

Chain topology:  [Body0 (FIXED)] --revolute-- [Body1]

Only 1 DOF (revolute). No free joint — isolates whether the convergence
issue at ±45 deg comes from the revolute joint or the free joint.

MuJoCo provides the 1x1 mass matrix and predicted displacement.
libuipc handles IPC contact resolution.
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
    AffineBodyRevoluteJoint,
    ExternalArticulationConstraint,
)
from uipc.geometry import SimplicialComplex, SimplicialComplexIO, affine_body, ground, label_surface
from uipc.gui import SceneGUI
from uipc.unit import GPa, MPa

# MuJoCo model: body0 is fixed (no freejoint), body1 has a hinge.
# Hinge pivot at the interface (0, 1, 0) between the two cubes.
MJCF_MODEL = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="body0" pos="0 1 -0.4">
      <geom type="box" size="0.2 0.2 0.2" mass="1.0"/>
      <body name="body1" pos="0 0 0.4">
        <joint name="revolute" type="hinge" axis="1 0 0"/>
        <geom type="box" size="0.2 0.2 0.2" mass="1.0" pos="0 0 0.4"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

N_DOFS = 1

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

INIT_REV_OMEGA = -2.0  # rad/s
mj_data.qvel[0] = INIT_REV_OMEGA

mujoco.mj_forward(mj_model, mj_data)
M_init = np.zeros((N_DOFS, N_DOFS))
mujoco.mj_fullM(mj_model, M_init, mj_data.qM)
print(f"MuJoCo initial mass matrix ({N_DOFS}x{N_DOFS}):")
print(np.array2string(M_init, precision=6, suppress_small=True))

mj_state = {
    "M": M_init.copy(),
    "delta_theta_tilde": np.zeros(N_DOFS),
}

# ------------------------------------------------------------------
# libuipc setup
# ------------------------------------------------------------------
engine = Engine("cuda", this_output_path)
world = World(engine)

config = Scene.default_config()
config["gravity"] = [[0.0], [0.0], [0.0]]
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

# 2 body instances in one mesh
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
is_fixed_view[0] = 1  # Body 0 is FIXED
is_fixed_view[1] = 0

external_kinetic = abd_mesh.instances().find(builtin.external_kinetic)
ek_view = view(external_kinetic)
ek_view[0] = 0  # fixed body: no external kinetic
ek_view[1] = 1  # revolute body: external kinetic from EAC

geo_slot, rest_geo_slot = body_object.geometries().create(abd_mesh)

# Ground plane
ground_object = scene.objects().create("ground")
ground_object.geometries().create(ground())

# ------------------------------------------------------------------
# Joint: Revolute only (no FreeJoint)
# ------------------------------------------------------------------
abrj = AffineBodyRevoluteJoint()
joint_pivot = np.array([0.0, 1.0, 0.0], dtype=np.float32)
axis_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
rev_pos0 = np.array([joint_pivot], dtype=np.float32)
rev_pos1 = np.array([joint_pivot + axis_dir], dtype=np.float32)
revolute_mesh = abrj.create_geometry(
    rev_pos0, rev_pos1, [geo_slot], [0], [geo_slot], [1], [100.0]
)
revolute_object = scene.objects().create("revolute_joint")
revolute_slot, _ = revolute_object.geometries().create(revolute_mesh)

# ------------------------------------------------------------------
# ExternalArticulationConstraint: 1 revolute DOF only
# ------------------------------------------------------------------
eac = ExternalArticulationConstraint()
articulation = eac.create_geometry([revolute_slot], [0])

mass_attr = articulation["joint_joint"].find("mass")
view(mass_attr)[:] = M_init.flatten()

articulation_object = scene.objects().create("articulation")
articulation_slot, _ = articulation_object.geometries().create(articulation)


def update_articulation(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    dtv = view(geo["joint"].find("delta_theta_tilde"))
    dtv[:] = mj_state["delta_theta_tilde"]
    mass_v = view(geo["joint_joint"].find("mass"))
    print(mj_state["M"])
    mass_v[:] = mj_state["M"].flatten()


scene.animator().insert(articulation_object, update_articulation)

# ------------------------------------------------------------------
# Init world + GUI
# ------------------------------------------------------------------
world.init(scene)
sgui = SceneGUI(scene, "split")

# Ghost mesh template
ghost_sc = io.read(f"{trimesh_path}/cube.obj")
label_surface(ghost_sc)
ghost_verts_local = ghost_sc.positions().view().reshape(-1, 3).copy().astype(np.float64)
ghost_tris = ghost_sc.triangles().topo().view().reshape(-1, 3).copy()

ps.init()
ps.set_ground_plane_height(0.0)
sgui.register()
sgui.set_edge_width(1)

# Only body1 ghost (body0 is fixed)
MJ_GEOM_ID_BODY1 = 1
R_init = mj_data.geom_xmat[MJ_GEOM_ID_BODY1].reshape(3, 3)
t_init = mj_data.geom_xpos[MJ_GEOM_ID_BODY1]
ghost_mesh = ps.register_surface_mesh(
    "mj_ghost_body1",
    (ghost_verts_local @ R_init.T) + t_init,
    ghost_tris,
)
ghost_mesh.set_color((0.2, 0.6, 1.0))
ghost_mesh.set_transparency(0.4)

gui = {"run": False}


def on_update():
    if imgui.Button("Run & Stop"):
        gui["run"] = not gui["run"]

    imgui.Separator()
    imgui.Text("Fixed Body + Revolute Joint Isolation Test")
    imgui.Text(f"dt = {dt:.4f}s  |  DOFs = {N_DOFS}")
    imgui.Separator()

    M = mj_state["M"]
    imgui.Text(f"MuJoCo M(q): {M[0,0]:.6f}")

    rev_angle_deg = np.degrees(mj_data.qpos[0])
    imgui.Text(f"Revolute angle: {rev_angle_deg:+.2f} deg")
    imgui.Text(f"Revolute vel:   {mj_data.qvel[0]:+.4f} rad/s")

    imgui.Separator()
    imgui.Text(f"Frame: {world.frame()}")
    imgui.Text(f"Time: {world.frame() * dt:.3f}s")

    if gui["run"]:
        qpos_prev = mj_data.qpos.copy()
        mujoco.mj_step(mj_model, mj_data)

        # Update ghost mesh with MuJoCo's uncorrected prediction
        R = mj_data.geom_xmat[MJ_GEOM_ID_BODY1].reshape(3, 3)
        t = mj_data.geom_xpos[MJ_GEOM_ID_BODY1]
        ghost_mesh.update_vertex_positions((ghost_verts_local @ R.T) + t)

        M_dense = np.zeros((N_DOFS, N_DOFS))
        mujoco.mj_fullM(mj_model, M_dense, mj_data.qM)

        eigvals = np.linalg.eigvalsh(M_dense)
        if np.any(eigvals <= 0):
            print(f"[WARN] Frame {world.frame()}: M is NOT SPD!  eigenvalues = {eigvals}")

        mj_state["M"] = M_dense
        mj_state["delta_theta_tilde"] = mj_data.qvel.copy() * dt

        # IPC step
        world.advance()
        world.retrieve()

        # Read IPC result
        art_geo = articulation_slot.geometry()
        delta_theta = np.array(art_geo["joint"].find("delta_theta").view())

        # Sync back to MuJoCo (only 1 DOF: revolute angle)
        mj_data.qpos[0] = qpos_prev[0] + delta_theta[0]
        mj_data.qvel[0] = delta_theta[0] / dt

        angle_deg = np.degrees(mj_data.qpos[0])
        print(f"Frame {world.frame():4d}: angle = {angle_deg:+8.2f} deg, "
              f"delta_theta = {delta_theta[0]:+.6f}")

        sgui.update()


ps.set_user_callback(on_update)
ps.show()

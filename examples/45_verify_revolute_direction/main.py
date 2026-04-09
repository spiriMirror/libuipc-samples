"""
Verify revolute joint sign convention (right-hand rule).

Setup:
  - Body 0 (left, fixed) at z=-0.8
  - Body 1 (right, free)  at z= 0.0
  - Revolute joint along +X axis between them (edge from [-0.5,0,-0.4] to [0.5,0,-0.4])

Expected: a positive delta_theta_tilde drives CCW rotation around +X
(right-hand rule: thumb along +X, fingers curl from +Y toward +Z).

Also sets up a FreeJoint body (Body 2) at z=0.8 to test FreeJoint RotY
with a positive delta_theta_tilde (should give CCW around +Y).
"""

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
from uipc.geometry import SimplicialComplex, SimplicialComplexIO, affine_body, label_surface
from uipc.gui import SceneGUI
from uipc.unit import GPa, MPa

Timer.enable_all()
Logger.set_level(Logger.Level.Info)

this_output_path = AssetDir.output_path(__file__)
trimesh_path = AssetDir.trimesh_path()

engine = Engine("cuda", this_output_path)
world = World(engine)

dt = 0.01
config = Scene.default_config()
config["gravity"] = [[0.0], [0.0], [0.0]]
config["contact"]["enable"] = False
config["newton"]["velocity_tol"] = 0.1
config["newton"]["transrate_tol"] = 10
config["linear_system"]["tol_rate"] = 1e-4
config["dt"] = dt
scene = Scene(config)

scene.contact_tabular().default_model(0.05, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()

abd = AffineBodyConstitution()

pre_transform = Transform.Identity()
pre_transform.scale(0.4)
io = SimplicialComplexIO(pre_transform)

# --- Body setup ---
# Body 0: fixed anchor at z=-0.8
# Body 1: free, revolute-connected to Body 0
# Body 2: free, FreeJoint (6 DOFs)
links = scene.objects().create("links")
abd_mesh = io.read(f"{trimesh_path}/cube.obj")
abd_mesh.instances().resize(3)
label_surface(abd_mesh)
abd.apply_to(abd_mesh, 100.0 * MPa)
default_element.apply_to(abd_mesh)

trans_view = view(abd_mesh.transforms())

t0 = Transform.Identity()
t0.translate(Vector3.UnitZ() * -0.8)
trans_view[0] = t0.matrix()

t1 = Transform.Identity()
trans_view[1] = t1.matrix()

t2 = Transform.Identity()
t2.translate(Vector3.UnitZ() * 0.8)
trans_view[2] = t2.matrix()

is_fixed = abd_mesh.instances().find(builtin.is_fixed)
is_fixed_view = view(is_fixed)
is_fixed_view[0] = 1
is_fixed_view[1] = 0
is_fixed_view[2] = 0

ref_dof_prev = abd_mesh.instances().create("ref_dof_prev", Vector12.Zero())
ref_dof_prev_view = view(ref_dof_prev)
ref_dof_prev_view[:] = affine_body.transform_to_q(view(abd_mesh.transforms()))

external_kinetic = abd_mesh.instances().find(builtin.external_kinetic)
view(external_kinetic)[:] = 1

geo_slot, rest_geo_slot = links.geometries().create(abd_mesh)


def update_ref_dof_prev(info: Animation.UpdateInfo):
    geo: SimplicialComplex = info.geo_slots()[0].geometry()
    rdp = geo.instances().find("ref_dof_prev")
    rdp_view = view(rdp)
    t_view = view(geo.transforms())
    rdp_view[:] = affine_body.transform_to_q(t_view)


scene.animator().insert(links, update_ref_dof_prev)

# --- Revolute joint: Body 0 -> Body 1, axis along +X ---
abrj = AffineBodyRevoluteJoint()
pos0s = np.array([[-0.5, 0.0, -0.4]], dtype=np.float32)
pos1s = np.array([[0.5, 0.0, -0.4]], dtype=np.float32)
revolute_mesh = abrj.create_geometry(
    pos0s, pos1s, [geo_slot], [0], [geo_slot], [1], [100.0]
)
revolute_object = scene.objects().create("revolute_joint")
revolute_slot, _ = revolute_object.geometries().create(revolute_mesh)

# --- FreeJoint on Body 2 ---
abfj = AffineBodyFreeJoint()
free_joint_mesh = abfj.create_geometry([geo_slot], np.array([2], dtype=np.int32))
fj_object = scene.objects().create("free_joint")
free_joint_slot, _ = fj_object.geometries().create(free_joint_mesh)

# --- ExternalArticulationConstraint: 1 revolute + 6 FreeJoint DOFs = 7 total ---
eac = ExternalArticulationConstraint()
n_free_dofs = 6
n_total = 1 + n_free_dofs

joint_geos = [revolute_slot] + [free_joint_slot] * n_free_dofs
indices = [0] + list(range(n_free_dofs))
articulation = eac.create_geometry(joint_geos, indices)

mass = articulation["joint_joint"].find("mass")
mass_view = view(mass)
mass_mat = np.eye(n_total, dtype=np.float32) * 64.0
mass_view[:] = mass_mat.flatten()

articulation_object = scene.objects().create("articulation")
articulation_object.geometries().create(articulation)

gui = {
    "run": False,
    "revolute_vel": 0.0,
    "fj_rot_y": 0.0,
}


def update_articulation(info: Animation.UpdateInfo):
    dt = info.dt()
    geo = info.geo_slots()[0].geometry()
    dtv = view(geo["joint"].find("delta_theta_tilde"))

    # DOF 0: revolute joint (Body 0 -> Body 1), axis along +X
    dtv[0] = gui["revolute_vel"] * dt

    # DOFs 1-6: FreeJoint on Body 2 (TransX,TransY,TransZ,RotX,RotY,RotZ)
    dtv[1] = 0.0
    dtv[2] = 0.0
    dtv[3] = 0.0
    dtv[4] = 0.0
    dtv[5] = gui["fj_rot_y"] * dt
    dtv[6] = 0.0


scene.animator().insert(articulation_object, update_articulation)

world.init(scene)
sgui = SceneGUI(scene, "split")

ps.init()
ps.set_ground_plane_height(-2.0)
sgui.register()
sgui.set_edge_width(1)


def on_update():
    if imgui.Button("Run & Stop"):
        gui["run"] = not gui["run"]

    imgui.Separator()
    imgui.Text("Revolute Joint Sign Convention Test")
    imgui.Text("Body 0 (z=-0.8): FIXED anchor")
    imgui.Text("Body 1 (z= 0.0): revolute joint, axis = +X")
    imgui.Text("Body 2 (z= 0.8): FreeJoint")
    imgui.Separator()

    imgui.Text("Revolute (Body 0->1), axis +X:")
    imgui.Text("  Positive = CCW (Y->Z when looking along +X)")
    _, gui["revolute_vel"] = imgui.SliderFloat(
        "Revolute (rad/s)", gui["revolute_vel"], -np.pi, np.pi
    )

    imgui.Separator()
    imgui.Text("FreeJoint RotY (Body 2):")
    imgui.Text("  Positive = CCW (Z->X when looking along +Y)")
    _, gui["fj_rot_y"] = imgui.SliderFloat(
        "FJ RotY (rad/s)", gui["fj_rot_y"], -np.pi, np.pi
    )

    imgui.Separator()
    imgui.Text(f"Frame: {world.frame()}")
    imgui.Text(f"Time: {world.frame() * dt:.2f}s")

    if gui["run"]:
        world.advance()
        world.retrieve()
        sgui.update()


ps.set_user_callback(on_update)
ps.show()

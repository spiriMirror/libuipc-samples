import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Scene, World, Engine, Transform, Vector3, Vector12, Animation, Logger, Timer
from uipc import builtin
from uipc.unit import MPa
from uipc.geometry import SimplicialComplexIO, SimplicialComplex, label_surface, linemesh
from uipc.geometry import affine_body
from uipc.constitution import AffineBodyConstitution, AffineBodyRevoluteJoint, AffineBodyPrismaticJoint, ExternalArticulationConstraint, NeoHookeanShell, DiscreteShellBending, ElasticModuli
from uipc.unit import GPa, kPa
from uipc.gui import SceneGUI
from asset_dir import AssetDir

Logger.set_level(Logger.Level.Info)
Timer.enable_all()

this_output_path = AssetDir.output_path(__file__)
trimesh_path = AssetDir.trimesh_path()

engine = Engine('cuda', this_output_path)
world = World(engine)

dt = 0.01
config = Scene.default_config()
config['gravity'] = [[0.0], [-9.8], [0.0]]
config['contact']['enable'] = True

config['newton']['velocity_tol'] = 0.1 # every low accuracy for interaction purpose
config['newton']['transrate_tol'] = 10 
config['linear_system']['tol_rate'] = 1e-4
config['contact']['d_hat'] = 0.001
config['collision_detection']['method'] = 'stackless_bvh'
config['dt'] = dt
print(config)
scene = Scene(config)

# Setup contact
scene.contact_tabular().default_model(0.05, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()

# Create constitutions
abd = AffineBodyConstitution()

# Load and setup cube mesh
pre_transform = Transform.Identity()
pre_transform.scale(0.4)
io = SimplicialComplexIO(pre_transform)

links = scene.objects().create('links')
abd_mesh = io.read(f'{trimesh_path}/cube.obj')
abd_mesh.instances().resize(3)
label_surface(abd_mesh)
abd.apply_to(abd_mesh, 100.0 * MPa)  # 100 MPa
default_element.apply_to(abd_mesh)

# Set initial transforms for 3 instances
trans_view = view(abd_mesh.transforms())
t0 = Transform.Identity()
t0.translate(Vector3.UnitZ() * -0.8)
trans_view[0] = t0.matrix()

t1 = Transform.Identity()
t1.translate(Vector3.UnitZ() * 0.0)
trans_view[1] = t1.matrix()

t2 = Transform.Identity()
t2.translate(Vector3.UnitZ() * 0.8)
trans_view[2] = t2.matrix()

# Set fixed instance
is_fixed = abd_mesh.instances().find(builtin.is_fixed)
is_fixed_view = view(is_fixed)
is_fixed_view[0] = 1  # instance 0 fixed
is_fixed_view[1] = 0  # instance 1 not fixed
is_fixed_view[2] = 0  # instance 2 not fixed

# Create ref_dof_prev attribute
# q vector: [translation(3), rotation_row0(3), rotation_row1(3), rotation_row2(3)]
ref_dof_prev = abd_mesh.instances().create('ref_dof_prev', Vector12.Zero())
ref_dof_prev_view = view(ref_dof_prev)
transform_view = view(abd_mesh.transforms())
ref_dof_prev_view[:] = affine_body.transform_to_q(transform_view)  # Shape: (3, 12, 1)


# Enable external kinetic for all instances
external_kinetic = abd_mesh.instances().find(builtin.external_kinetic)
external_kinetic_view = view(external_kinetic)
external_kinetic_view[:] = 0

geo_slot, rest_geo_slot = links.geometries().create(abd_mesh)

# Animator to update ref_dof_prev
def update_ref_dof_prev(info: Animation.UpdateInfo):
    geo: SimplicialComplex = info.geo_slots()[0].geometry()
    ref_dof_prev = geo.instances().find('ref_dof_prev')
    ref_dof_prev_view = view(ref_dof_prev)
    transform_view = view(geo.transforms())
    ref_dof_prev_view[:] = affine_body.transform_to_q(transform_view)

scene.animator().insert(links, update_ref_dof_prev)

# Create revolute joint
abrj = AffineBodyRevoluteJoint()

# Each edge defines a joint axis (2 points per edge)
Es = np.array([[0, 1]], dtype=np.int32)
Vs = np.array([[-0.5, 0.0, -0.4], [0.5, 0.0, -0.4]], dtype=np.float32)
joint_mesh = linemesh(Vs, Es)

# Use multi-instance API
l_geo_slots = [geo_slot]
l_instance_id = [0]
r_geo_slots = [geo_slot]
r_instance_id = [1]
strength_ratios = [100.0]

abrj.apply_to(joint_mesh, l_geo_slots, l_instance_id, r_geo_slots, r_instance_id, strength_ratios)

joints = scene.objects().create('joints')
revolute_joint_slots, rest_revolute_joint_slots = joints.geometries().create(joint_mesh)
revolute_slot = revolute_joint_slots

# Create prismatic joint
abpj = AffineBodyPrismaticJoint()
# Each edge defines a joint axis (2 points per edge)
Es = np.array([[0, 1]], dtype=np.int32)
Vs = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.4]], dtype=np.float32)
joint_mesh = linemesh(Vs, Es)

# Use multi-instance API
l_geo_slots = [geo_slot]
l_instance_id = [1]
r_geo_slots = [geo_slot]
r_instance_id = [2]
strength_ratios = [100.0]
abpj.apply_to(joint_mesh, l_geo_slots, l_instance_id, r_geo_slots, r_instance_id, strength_ratios)

joints = scene.objects().create('joints_prismatic')
prismatic_joint_slots, rest_prismatic_joint_slots = joints.geometries().create(joint_mesh)
prismatic_slot = prismatic_joint_slots

# Create external articulation constraint
eac = ExternalArticulationConstraint()
joint_geos = [revolute_slot, prismatic_slot]
indices = [0, 0]
articulation = eac.create_geometry(joint_geos, indices)

# Set mass matrix
mass = articulation['joint_joint'].find('mass')
print(articulation)
# joint mass
# [1e4, 5e3]
# [5e3, 1e4]

mass_view = view(mass)
mass_mat = np.eye(2) * 1e4
mass_mat[0, 1] = 5e3
mass_mat[1, 0] = 5e3
 
mass_view[:] = mass_mat.flatten()

print(mass_mat.flatten())

articulation_object = scene.objects().create('articulation_object')
articulation_object.geometries().create(articulation)

# GUI control variables
delta_theta_tilde_0 = np.pi / 6.0 # revolute joint angular velocity
delta_theta_tilde_1 = 0.0  # prismatic joint linear velocity
mass_00 = 1e4
mass_01 = 5e3
mass_11 = 1e4

# Animator to update delta_theta_tilde from GUI
def update_articulation(info: Animation.UpdateInfo):
    dt = info.dt()
    geo_slots = info.geo_slots()
    geo = geo_slots[0].geometry()
    
    delta_theta_tilde = geo['joint'].find('delta_theta_tilde')
    delta_theta_view = view(delta_theta_tilde)
    delta_theta_view[0] = delta_theta_tilde_0 * dt
    delta_theta_view[1] = delta_theta_tilde_1 * dt
    
    mass = geo['joint_joint'].find('mass')
    mass_view = view(mass)
    # symmetric matrix
    mass_view[:] = [mass_00, mass_01, mass_01, mass_11]

scene.animator().insert(articulation_object, update_articulation)

# Create cloth object
cloth = scene.objects().create('cloth')
t_cloth = Transform.Identity()
t_cloth.scale(2.0)
io_cloth = SimplicialComplexIO(t_cloth)
cloth_mesh = io_cloth.read(f'{trimesh_path}/grid20x20.obj')
label_surface(cloth_mesh)

# Apply cloth constitutions
nks = NeoHookeanShell()
dsb = DiscreteShellBending()
moduli = ElasticModuli.youngs_poisson(500 * kPa, 0.49)
nks.apply_to(cloth_mesh, moduli=moduli, mass_density=200, thickness=0.001)
dsb.apply_to(cloth_mesh, bending_stiffness=1.0)

# Position cloth above the articulated system
cloth_pos_view = view(cloth_mesh.positions())
cloth_pos_view[:, 1] += 1.0  # Move cloth up

# Apply contact element
default_element.apply_to(cloth_mesh)

cloth.geometries().create(cloth_mesh)


world.init(scene)
sgui = SceneGUI(scene, 'split')

ps.init()
ps.set_ground_plane_height(-1.0)
sgui.register()
sgui.set_edge_width(1)

run = False
def on_update():
    global run, delta_theta_tilde_0, delta_theta_tilde_1, mass_00, mass_01, mass_11
    
    if imgui.Button('Run & Stop'):
        run = not run
    
    imgui.Separator()
    imgui.Text('External Articulation Control')
    imgui.Text('Adjust delta_theta_tilde values:')
    
    # clear slider values every frame
    delta_theta_tilde_0 = 0.0
    delta_theta_tilde_1 = 0.0
    
    changed0, delta_theta_tilde_0 = imgui.SliderFloat(
        'Revolute Joint (rad/s)', 
        delta_theta_tilde_0, 
        -25 * np.pi * dt, 
        25 * np.pi * dt
    )
    
    changed1, delta_theta_tilde_1 = imgui.SliderFloat(
        'Prismatic Joint (m/s)', 
        delta_theta_tilde_1, 
        -25.0 * dt, 
        25.0 * dt
    )
    
    changed2, mass_00 = imgui.SliderFloat(
        'Mass 00', 
        mass_00, 
        1e4, 
        1e5
    )
        
    changed4, mass_11 = imgui.SliderFloat(
        'Mass 11', 
        mass_11, 
        1e4, 
        1e5
    )
    
    max_mass_01 = min(mass_00, mass_11)
    
    changed3, mass_01 = imgui.SliderFloat(
        'Mass 01', 
        mass_01, 
        1e3, 
        max_mass_01
    )
    
    
    
    
    imgui.Separator()
    imgui.Text(f'Frame: {world.frame()}')
    imgui.Text(f'Time: {world.frame() * dt:.2f}s')
    
    if run:
        world.advance()
        world.retrieve()
        sgui.update()
        Timer.report()

ps.set_user_callback(on_update)
ps.show()


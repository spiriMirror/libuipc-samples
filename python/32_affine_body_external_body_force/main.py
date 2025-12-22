import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Scene, World, Engine, Transform, Vector3, Matrix4x4, Animation, Logger
from uipc import builtin
from uipc.geometry import SimplicialComplexIO, label_surface, label_triangle_orient, flip_inward_triangles
from uipc.constitution import AffineBodyConstitution, AffineBodyExternalBodyForce
from uipc.gui import SceneGUI, SimplicialComplex
from asset_dir import AssetDir

Logger.set_level(Logger.Level.Warn)

tetmesh_dir = AssetDir.tetmesh_path()
this_output_path = AssetDir.output_path(__file__)

engine = Engine('cuda', this_output_path)
world = World(engine)

config = Scene.default_config()

# Disable built-in gravity since we're using external force
config['gravity'] = [[0.0], [0.0], [0.0]]
config['contact']['enable'] = False
config['dt'] = 0.005
scene = Scene(config)

# Create constitutions
abd = AffineBodyConstitution()
ext_force = AffineBodyExternalBodyForce()

# Setup contact
scene.contact_tabular().default_model(0.5, 1e9)
default_element = scene.contact_tabular().default_element()

# Load cube mesh
pre_trans = Transform.Identity()
pre_trans.scale(0.2)

io = SimplicialComplexIO(pre_trans)
cube = io.read(f'{tetmesh_dir}/cube.msh')

# Process surface
label_surface(cube)
label_triangle_orient(cube)
cube_processed = flip_inward_triangles(cube)

# Create object with single cube
cube_object = scene.objects().create('cube')

cube_processed.instances().resize(1)

# Apply constitutions
abd.apply_to(cube_processed, 1e8)  # stiffness

# Apply external force - initially zero, will be set by animator
initial_force = np.zeros(12)  # Vector12: [fx, fy, fz, f_a11, f_a12, f_a13, f_a21, f_a22, f_a23, f_a31, f_a32, f_a33]
ext_force.apply_to(cube_processed, initial_force)

default_element.apply_to(cube_processed)

# Set transform - position at y=0.5
trans_view = view(cube_processed.transforms())
t = Transform.Identity()
t.translate(Vector3.Values([0.0, 0.2, 0.0]))
trans_view[0] = t.matrix()

cube_object.geometries().create(cube_processed)

# Add animator for combined orbital and spinning motion
def animate_cube(info: Animation.UpdateInfo):
    geo:SimplicialComplex = info.geo_slots()[0].geometry()

    # Set is_constrained to enable external force
    is_constrained = geo.instances().find(builtin.is_constrained)
    is_constrained_view = view(is_constrained)
    is_constrained_view[0] = 1

    t = info.dt() * info.frame()
    force_magnitude = 10  # N
    theta = np.pi * t
    force_direction = np.array([-np.cos(theta), 0.0, np.cos(theta)])
    force_3d = force_direction * force_magnitude
    force = np.zeros(12)
    force[0:3] = force_3d
    
    force_attr = geo.instances().find('external_force')
    force_view = view(force_attr)
    force_view[:] = force.reshape(-1, 1)

scene.animator().insert(cube_object, animate_cube)

world.init(scene)
world.dump()

sgui = SceneGUI(scene, 'split')

ps.init()
ps.set_ground_plane_height(0)
sgui.register()
sgui.set_edge_width(1)

run = False
def on_update():
    global run
    if(imgui.Button('run & stop')):
        run = not run
    
    if(run):
        world.advance()
        world.retrieve()
        sgui.update()

ps.set_user_callback(on_update)
ps.show()


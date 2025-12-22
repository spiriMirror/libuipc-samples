import numpy as np
import polyscope as ps
from polyscope import imgui
import keyboard
from uipc import view
from uipc import Engine, World, Scene, Animation
from uipc import Vector3, Vector2, Transform, Logger, Quaternion, AngleAxis
from uipc import builtin
from uipc.geometry import GeometrySlot, SimplicialComplex, SimplicialComplexIO, pointcloud, linemesh, label_surface, label_triangle_orient, flip_inward_triangles, ground
from uipc.constitution import AffineBodyConstitution, Particle, HookeanSpring, SoftPositionConstraint
from uipc.gui import SceneGUI

from asset_dir import AssetDir

# initialization
ps.init()
Logger.set_level(Logger.Level.Warn)

class IO:
    @staticmethod
    def is_key_down(key_str:str):
        return keyboard.is_pressed(key_str.lower())
    
    @staticmethod
    def movement():
        w = 1.0 if IO.is_key_down('W') else 0.0
        s = -1.0 if IO.is_key_down('S') else 0.0
        d = 1.0 if IO.is_key_down('D') else 0.0
        a = -1.0 if IO.is_key_down('A') else 0.0
        e = 1.0 if IO.is_key_down('E') else 0.0
        q = -1.0 if IO.is_key_down('Q') else 0.0
        V = Vector3.Values([a+d, e+q, -(w+s)])
        # print(V)
        return V
# -----------------------------------------------------

tetmesh_path = AssetDir.tetmesh_path()
workspace = AssetDir.output_path(__file__)

engine = Engine('cuda', workspace)
world = World(engine)

config = Scene.default_config()
dt = 0.01
config['dt'] = dt
config['newton']['max_iter'] = 5
print(config)
scene = Scene()

abd = AffineBodyConstitution()
pt = Particle()
hs = HookeanSpring()
spc = SoftPositionConstraint()
scene.constitution_tabular().insert(abd)
scene.constitution_tabular().insert(pt)
scene.constitution_tabular().insert(hs)
scene.constitution_tabular().insert(spc)
scene.contact_tabular().default_model(0.1, 1e9)
default_element = scene.contact_tabular().default_element()

# load a cube
cube_obj = scene.objects().create('board')
t = Transform.Identity()
t.scale([3, 0.1, 3])
io = SimplicialComplexIO(t)
cube = io.read(f'{tetmesh_path}/cube.msh')
label_surface(cube)
label_triangle_orient(cube)
cube = flip_inward_triangles(cube)
abd.apply_to(cube, 1e8) # 100 MPa
default_element.apply_to(cube)
trans_view = view(cube.transforms())
t = Transform(trans_view[0])
t.translate([0, 1, 0])
trans_view[0] = t.matrix()
is_fixed = cube.instances().find(builtin.is_fixed)
is_fixed_view = view(is_fixed)
is_fixed_view[0] = 1

cube_obj.geometries().create(cube)

# create a particle
particle_obj = scene.objects().create('particle')
Vs = np.array([[0, 1.5, 0]])
particle = pointcloud(Vs)
label_surface(particle)
pt.apply_to(particle, thickness=0.1)
default_element.apply_to(particle)
spc.apply_to(particle)
particle_obj.geometries().create(particle)

# create a rope
rope_obj = scene.objects().create('rope')
Vs = np.array([[-0.4, 1.5, -1], [0.0, 1.5, -1], [0.4, 1.5, -1],[0.8, 1.5, -1]])
Es = np.array([[0, 1], [1, 2], [2, 3]])
rope = linemesh(Vs, Es)
label_surface(rope)
hs.apply_to(rope, thickness=0.1)
default_element.apply_to(rope)
rope_geo_slot, _ = rope_obj.geometries().create(rope)

# create a ground
ground_obj = scene.objects().create('ground')
ground_height = -0.5
g = ground(ground_height)
ground_obj.geometries().create(g)

# animate the particle
animator = scene.animator()

def animation(info:Animation.UpdateInfo):
    geo_slot:GeometrySlot = info.geo_slots()[0]
    particle:SimplicialComplex = geo_slot.geometry()
    is_constrained = particle.vertices().find(builtin.is_constrained)
    aim_pos = particle.vertices().find(builtin.aim_position)
    
    pos_view = particle.positions().view()
    aim_pos_view = view(aim_pos)
    
    V = IO.movement()
    
    if np.linalg.norm(V) > 0:
        view(is_constrained).fill(1)
        aim_pos_view[:] = pos_view[:] + V * dt
    else:
        view(is_constrained).fill(0)
    
   
animator.insert(particle_obj, animation)

world.init(scene)

# polyscope registration
run = False
sgui = SceneGUI(scene)
tri_surf, edge_surf, _ = sgui.register()
tri_surf.set_edge_width(1)
edge_surf.set_radius(0.1, False)
ps.set_ground_plane_height(ground_height)

Win = False
UsingTime = 0

def update():
    global run
    global Win
    global UsingTime
    if(imgui.Button('Run & Stop')):
        run = not run
    imgui.Text('W, A, S, D, E, Q to move the particle')
    imgui.Text('Push the rope to the ground to win!')
    if(Win):
        imgui.Text(f'You Win in {UsingTime:.2f}s !')
    else:
        UsingTime = world.frame() * dt
        imgui.Text(f'{UsingTime:.2f}s')
    
    if run:
        world.advance()
        world.retrieve()
        sgui.update()
        rope_geo:SimplicialComplex = rope_geo_slot.geometry()
        pos_view = rope_geo.positions().view()
        if(pos_view[0][1] < 0.0):
            Win = True

ps.set_user_callback(update)
ps.show()
import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, Timer, Animation
from uipc import Vector3, Vector2, Transform, Quaternion, AngleAxis
from uipc import builtin
from uipc.core import Engine, World, Scene, SceneIO
from uipc.geometry import GeometrySlot, SimplicialComplex, SimplicialComplexSlot, SimplicialComplexIO, label_surface
from uipc.constitution import AffineBodyConstitution, RotatingMotor
from uipc.unit import MPa
from uipc.gui import SceneGUI

from asset_dir import AssetDir

Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
this_folder = AssetDir.folder(__file__)

engine = Engine('cuda', workspace)
world = World(engine)

config = Scene.default_config()
config['dt'] = 0.005
config['contact']['d_hat'] = 0.02
config['contact']['friction']['enable'] = False
config['gravity'] = [[0.0], [-0.0], [0.0]]
print(config)

scene = Scene(config)

# begin setup the scene
t = Transform.Identity()
t.scale(0.05)
io = SimplicialComplexIO()
abd = AffineBodyConstitution()
rm = RotatingMotor()
scene.contact_tabular().default_model(0, 1e9)

screw_obj = scene.objects().create('screw')
screw_mesh = io.read(f'{AssetDir.trimesh_path()}/screw-and-nut/screw-big-2.obj')
label_surface(screw_mesh)
abd.apply_to(screw_mesh, 100 * MPa)
rm.apply_to(screw_mesh, 100, motor_axis=Vector3.UnitY(), motor_rot_vel=-np.pi)
screw_obj.geometries().create(screw_mesh)

def screw_animation(info:Animation.UpdateInfo):
    geo_slots = info.geo_slots()
    geo_slot: SimplicialComplexSlot = geo_slots[0]
    geo = geo_slot.geometry()
    is_constrained = geo.instances().find(builtin.is_constrained)
    view(is_constrained)[0] = 1
    RotatingMotor.animate(geo, info.dt())
    
scene.animator().insert(screw_obj, screw_animation)

nut_obj = scene.objects().create('nut')
nut_mesh = io.read(f'{AssetDir.trimesh_path()}/screw-and-nut/nut-big-2.obj')
label_surface(nut_mesh)
abd.apply_to(nut_mesh, 100 * MPa)
is_fixed = nut_mesh.instances().find(builtin.is_fixed)
view(is_fixed)[:] = 1
nut_obj.geometries().create(nut_mesh)

# end setup the scene

world.init(scene)

ps.init()
ps.set_ground_plane_height(-5)
ps.set_window_size(1600, 1280)
sgui = SceneGUI(scene, 'split')
sgui.register()
sgui.set_edge_width(1)

sio = SceneIO(scene)

run = False
def on_update():
    global run
    if(imgui.Button('run & stop')):
        run = not run
    if(imgui.Button('recover')):
        world.recover(1)
        world.retrieve()
        sgui.update()
        
    if(run):
        world.advance()
        world.retrieve()
        sgui.update()

ps.set_user_callback(on_update)
ps.show()
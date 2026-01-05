import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, Timer, Animation
from uipc import Vector3, Transform
from uipc import builtin
from uipc.core import Engine, World, Scene, SceneIO
from uipc.geometry import SimplicialComplexSlot, SimplicialComplexIO, label_surface
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
config['dt'] = 0.01
config['contact']['d_hat'] = 0.001
config['gravity'] = [[0.0], [-9.8], [0.0]]
config['line_search']['report_energy'] = True
config['contact']['friction']['enable'] = False
print(config)
scene = Scene(config)

# begin setup the scene

abd = AffineBodyConstitution()
rm = RotatingMotor()
scene.contact_tabular().default_model(0, 1e9)

t = Transform.Identity()
t.scale(0.5)
io = SimplicialComplexIO(t)

arm_obj = scene.objects().create('arm')
arm_mesh = io.read(f'{AssetDir.trimesh_path()}/pendulum/arm.obj')
label_surface(arm_mesh)
abd.apply_to(arm_mesh, 100 * MPa)
rm.apply_to(arm_mesh, 100, motor_axis=Vector3.UnitZ(), motor_rot_vel=-0.2*np.pi)
arm_obj.geometries().create(arm_mesh)

def arm_animation(info:Animation.UpdateInfo):
    geo_slots = info.geo_slots()
    geo_slot: SimplicialComplexSlot = geo_slots[0]
    geo = geo_slot.geometry()
    is_constrained = geo.instances().find(builtin.is_constrained)
    
    if(info.frame() <= 1):
        view(is_constrained)[0] = 1
        RotatingMotor.animate(geo, info.dt())
    else:
        view(is_constrained)[0] = 0
    
scene.animator().insert(arm_obj, arm_animation)

pin_obj = scene.objects().create('pin')
pin_mesh = io.read(f'{AssetDir.trimesh_path()}/pendulum/pin-short.obj')
label_surface(pin_mesh)
abd.apply_to(pin_mesh, 100 * MPa)
t = Transform.Identity()
t.translate(Vector3.UnitY() * -1)
view(pin_mesh.transforms())[:] = t.matrix()
is_fixed = pin_mesh.instances().find(builtin.is_fixed)
view(is_fixed)[:] = 1
pin_obj.geometries().create(pin_mesh)

# end setup the scene

world.init(scene)
world.retrieve()

ps.init()
ps.set_ground_plane_height(-5)
sgui = SceneGUI(scene)
tri_surf, lines, points = sgui.register()
tri_surf.set_edge_width(1)

sio = SceneIO(scene)

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
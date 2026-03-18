import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, Timer, Animation
from uipc import Vector3, Vector2, Transform, Quaternion, AngleAxis
import uipc.builtin as builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import GeometrySlot, SimplicialComplex, SimplicialComplexSlot, SimplicialComplexIO, ground, label_surface
from uipc.constitution import AffineBodyConstitution, RotatingMotor
from uipc.gui import SceneGUI

from asset_dir import AssetDir

Timer.enable_all()
Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
this_folder = AssetDir.folder(__file__)

engine = Engine('cuda', workspace)
world = World(engine)

config = Scene.default_config()
config['dt'] = 0.01
config['contact']['d_hat'] = 0.01
config['newton']['velocity_tol'] = 0.1
config['contact']['enable'] = True
config['contact']['friction']['enable'] = False
print(config)
scene = Scene(config)

# begin setup the scene
t = Transform.Identity()
t.rotate(AngleAxis(np.pi/2, Vector3.UnitX()))
io = SimplicialComplexIO(t)

# create constituiton
abd = AffineBodyConstitution()
# create constraint
rm = RotatingMotor()
scene.contact_tabular().default_model(0, 1e9)

gear_obj = scene.objects().create('gear')
gear_mesh = io.read(f'{AssetDir.trimesh_path()}/gear0/gear.obj')
label_surface(gear_mesh)
abd.apply_to(gear_mesh, 1e8) # 100 MPa
rm.apply_to(gear_mesh, 100, motor_axis=Vector3.UnitZ(), motor_rot_vel=np.pi)
gear_obj.geometries().create(gear_mesh)

def gear_animation(info:Animation.UpdateInfo):
    geo_slots = info.geo_slots()
    geo_slot: SimplicialComplexSlot = geo_slots[0]
    geo = geo_slot.geometry()
    is_constrained = geo.instances().find(builtin.is_constrained)
    view(is_constrained)[0] = 1
    RotatingMotor.animate(geo, info.dt())
    
scene.animator().insert(gear_obj, gear_animation)

pin_obj = scene.objects().create('pin')
pin_mesh = io.read(f'{AssetDir.trimesh_path()}/gear0//pin.obj')
label_surface(pin_mesh)
abd.apply_to(pin_mesh, 1e8) # 100 MPa
is_fixed = pin_mesh.instances().find(builtin.is_fixed)
view(is_fixed)[:] = 1
pin_obj.geometries().create(pin_mesh)

rail_obj = scene.objects().create('rail')
rail_mesh = io.read(f'{AssetDir.trimesh_path()}/gear0/rail.obj')
label_surface(rail_mesh)
abd.apply_to(rail_mesh, 1e8) # 100 MPa
t = Transform.Identity()
t.translate(Vector3.UnitY() * -1.3)
view(rail_mesh.transforms())[:] = t.matrix()
rail_obj.geometries().create(rail_mesh)

rail_guard_obj = scene.objects().create('rail_guard')
rail_guard_mesh = io.read(f'{AssetDir.trimesh_path()}/gear0/rail-guard.obj')
label_surface(rail_guard_mesh)
abd.apply_to(rail_guard_mesh, 1e8) # 100 MPa
view(rail_guard_mesh.transforms())[:] = t.matrix()
is_fixed = rail_guard_mesh.instances().find(builtin.is_fixed)
view(is_fixed)[:] = 1
rail_guard_obj.geometries().create(rail_guard_mesh)

ground_height = -1.5
ground_obj = scene.objects().create('ground')
ground_geo = ground(ground_height)
ground_obj.geometries().create(ground_geo)

# end setup the scene

world.init(scene)

ps.init()
ps.set_window_size(1600, 1280)

sgui = SceneGUI(scene, 'split')
sgui.register()
sgui.set_edge_width(1)

run = False
def on_update():
    global run
    if(imgui.Button('run & stop')):
        run = not run
    if(run):
        if(world.recover(world.frame() + 1)):
            world.retrieve()
            # ps.screenshot(f'{workspace}/screenshot/{world.frame()}.png')
        else:
            world.advance()
            world.retrieve()
            world.dump()
            Timer.report()

        sgui.update()

ps.set_user_callback(on_update)
ps.show()
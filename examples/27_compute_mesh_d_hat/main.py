import numpy as np
import polyscope as ps
from polyscope import imgui

import uipc
from uipc import Logger, Timer, Transform, Quaternion, Vector3, Vector2, view, builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import tetmesh, label_surface, compute_mesh_d_hat, ground
from uipc.geometry import SimplicialComplexIO
from uipc.constitution import AffineBodyConstitution
from uipc.gui import SceneGUI 
from uipc.unit import MPa

from asset_dir import AssetDir

Timer.enable_all()
Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
this_folder = AssetDir.folder(__file__)
trimesh_path = AssetDir.trimesh_path()
tetmesh_path = AssetDir.tetmesh_path()

engine = Engine('cuda', workspace)
world = World(engine)

default_d_hat = 1 # very large d_hat
config = Scene.default_config()
config['dt'] = 0.01
config['contact']['d_hat'] = default_d_hat
print(config)
scene = Scene(config)

abd = AffineBodyConstitution()

obj = scene.objects().create('cubes')
t = Transform.Identity()
t.scale(0.10)
sio = SimplicialComplexIO(t)
base_mesh = sio.read(f'{trimesh_path}/cube.obj')
abd.apply_to(base_mesh, 100.0 * MPa)
label_surface(base_mesh)
compute_mesh_d_hat(base_mesh, max_d_hat=default_d_hat)

upper = base_mesh.copy()
upper_d_hat_attr = upper.meta().find('d_hat')
print(f'upper d_hat: {upper_d_hat_attr.view()}')
t = Transform.Identity()
t.translate(Vector3.UnitY() * 0.2)
view(upper.transforms())[:] = t.matrix()
obj.geometries().create(upper)


lower = base_mesh.copy()
lower_d_hat_attr = lower.meta().find('d_hat')
view(lower_d_hat_attr)[:] = 0.01
print(f'lower d_hat: {lower_d_hat_attr.view()}')
obj.geometries().create(lower)

g = scene.objects().create('ground')
g_mesh = ground(-0.15)
g.geometries().create(g_mesh)

world.init(scene)

ps.init()
sgui = SceneGUI(scene, 'split')
sgui.register()
sgui.set_edge_width(1.0)

run = False
def on_update():
    global run
    global upper_d_hat_attr
    global lower_d_hat_attr
    
    if(imgui.Button('run & stop')):
        run = not run
    
    imgui.Text(f'Upper d_hat: {upper_d_hat_attr.view()} <- Very Large, Compute From Mesh Resolution')
    imgui.Text(f'Lower d_hat: {lower_d_hat_attr.view()} <- Manually Set to 0.01')

    if(run):
        world.advance()
        world.retrieve()
        sgui.update()

ps.set_user_callback(on_update)
ps.show()
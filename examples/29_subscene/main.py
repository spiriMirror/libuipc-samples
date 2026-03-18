import numpy as np
import polyscope as ps
from polyscope import imgui

import uipc
from uipc import Logger, Timer, Transform, Vector3, view
from uipc.core import Engine, World, Scene
from uipc.geometry import label_surface, ground
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

config = Scene.default_config()
config['dt'] = 0.01
config['contact']['d_hat'] = 0.01
print(config)
scene = Scene(config)

default_element = scene.subscene_tabular().default_element()
subscene_1 = scene.subscene_tabular().create('#1')
subscene_2 = scene.subscene_tabular().create('#2')

# we like the subscene_0 can contact with subscene_1 and subscene_2
# but the subscene_1 and subscene_2 cannot contact with each other
scene.subscene_tabular().insert(default_element, subscene_1, True)
scene.subscene_tabular().insert(default_element, subscene_2, True)
# note if not insert, default is no contact between 2 subscenes
scene.subscene_tabular().insert(subscene_1, subscene_2, False) # no contact

abd = AffineBodyConstitution()

obj = scene.objects().create('cubes')
t = Transform.Identity()
t.scale(0.10)
sio = SimplicialComplexIO(t)
base_mesh = sio.read(f'{trimesh_path}/cube.obj')
abd.apply_to(base_mesh, 100.0 * MPa)

label_surface(base_mesh)


left = base_mesh.copy()
t = Transform.Identity()
t.translate(Vector3.UnitX() * -0.02)
view(left.transforms())[:] = t.matrix()
subscene_1.apply_to(left)
obj.geometries().create(left)

right = base_mesh.copy()
t = Transform.Identity()
t.translate(Vector3.UnitX() * 0.02)
view(right.transforms())[:] = t.matrix()
subscene_2.apply_to(right)
obj.geometries().create(right)

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
    
    if(imgui.Button('run & stop')):
        run = not run

    if(run):
        world.advance()
        world.retrieve()
        sgui.update()

ps.set_user_callback(on_update)
ps.show()
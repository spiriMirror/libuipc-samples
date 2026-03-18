import numpy as np
import polyscope as ps
from polyscope import imgui
import json

from uipc import view
from uipc import Scene, World, Engine
from uipc.gui import SceneGUI
from uipc.geometry import SimplicialComplexIO, SimplicialComplex, label_region, apply_region, label_surface, ground, merge
from uipc.constitution import AffineBodyConstitution
from asset_dir import AssetDir
from uipc.unit import MPa, GPa

this_folder = AssetDir.folder(__file__)
output_path = AssetDir.output_path(__file__)

engine = Engine('cuda', output_path)
world = World(engine)

config = Scene.default_config()
dt = 0.01
config['dt'] = dt
config['gravity'] = [[0.0], [-9.8], [0.0]]
config['contact']['d_hat'] = 0.001
config['contact']['friction']['enable'] = True
scene = Scene(config)

scene.contact_tabular().default_model(0.01, 1.0 * GPa)

saga_obj = scene.objects().create('saga')
sio = SimplicialComplexIO()
abd = AffineBodyConstitution()
saga_mesh = sio.read(f'{AssetDir.trimesh_path()}/saga.obj')

if True:
    print(saga_mesh)
    label_region(saga_mesh)
    print(saga_mesh)
    j = saga_mesh.to_json()
    with open(f'{output_path}/saga.json', 'w') as f:
        json.dump(j, f, indent=4)
    saga_meshes: list[SimplicialComplex] = apply_region(saga_mesh)
else:
    saga_meshes = [saga_mesh]

for mesh in saga_meshes:
    label_surface(mesh)
    abd.apply_to(mesh, 100.0*MPa)
    saga_obj.geometries().create(mesh)

ground_height = -0.4
g = ground(ground_height)
ground_obj = scene.objects().create('ground')
ground_obj.geometries().create(g)

sgui = SceneGUI(scene, 'split')
world.init(scene)
world.dump()

ps.init()
sgui.register()
sgui.set_edge_width(1)

run = False
def on_update():
    global run
    if(imgui.Button('run' if not run else 'stop')):
        run = not run
    
    if(world.frame() >= 200):
        run = False
    
    if(imgui.Button('recover')):
        world.recover(0)
        world.retrieve()
        sgui.update()
        
    if(run):
        if(world.recover(world.frame() + 1)):
            world.retrieve()
        else:
            world.advance()
            world.retrieve()
            world.dump()
        sgui.update()

ps.set_user_callback(on_update)
ps.show()
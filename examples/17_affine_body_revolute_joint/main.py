import numpy as np
import polyscope as ps
from polyscope import imgui
import json

from uipc import view
from uipc import Scene, World, Engine, Transform, Vector3, Logger
from uipc.gui import SceneGUI
import uipc.builtin as builtin
from uipc.geometry import SimplicialComplexIO, SimplicialComplex, label_surface, ground, linemesh
from uipc.constitution import AffineBodyConstitution, AffineBodyRevoluteJoint
from asset_dir import AssetDir
from uipc.unit import MPa, GPa

Logger.set_level(Logger.Level.Warn)

this_folder = AssetDir.folder(__file__)
output_path = AssetDir.output_path(__file__)

engine = Engine('cuda', output_path)
world = World(engine)

config = Scene.default_config()
dt = 0.01
config['dt'] = dt
config['gravity'] = [[0.0], [-9.8], [0.0]]
config['contact']['d_hat'] = 0.01
config['contact']['friction']['enable'] = False
scene = Scene(config)

scene.contact_tabular().default_model(0.01, 1.0 * GPa)

t = Transform.Identity()
t.scale([0.9, 0.04, 0.6])

sio = SimplicialComplexIO(t)
abd = AffineBodyConstitution()
abd_mesh = sio.read(f'{AssetDir.trimesh_path()}/cube.obj')
abd.apply_to(abd_mesh, kappa=100.0 * MPa)
label_surface(abd_mesh)

left_link = scene.objects().create('left')
left_mesh = abd_mesh.copy()
t = Transform.Identity()
t.translate(Vector3.UnitX() * -1.2)
view(left_mesh.transforms())[:] = t.matrix()
left_geo_slot, left_rest_geo_slot = left_link.geometries().create(left_mesh)

mid_link = scene.objects().create('mid')
mid_mesh = abd_mesh.copy()
t = Transform.Identity()
view(mid_mesh.transforms())[:] = t.matrix()
view(mid_mesh.instances().find(builtin.is_fixed))[:] = 1
mid_geo_slot, mid_rest_geo_slot = mid_link.geometries().create(mid_mesh)

right_link = scene.objects().create('right')
right_mesh = abd_mesh.copy()
t = Transform.Identity()
t.translate(Vector3.UnitX() * 1.2)
view(right_mesh.transforms())[:] = t.matrix()
right_geo_slot, right_rest_geo_slot = right_link.geometries().create(right_mesh)

abrj = AffineBodyRevoluteJoint()
Es = np.array([[0,1],
               [2,3]], dtype=np.int32)  # Es
Vs = np.array([[-0.6, 0, -1], 
               [-0.6, 0, 1],
                [0.6, 0, -1], 
                [0.6, 0, 1]], dtype=np.float32)  # Vs
joint_mesh = linemesh(Vs, Es)

abrj.apply_to(joint_mesh, 
              [(left_geo_slot, mid_geo_slot),
               (mid_geo_slot, right_geo_slot)])
joints = scene.objects().create('joint')
joints.geometries().create(joint_mesh)

ground_height = -1.1
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
    
    if(run):
        world.advance()
        world.retrieve()
        sgui.update()

ps.set_user_callback(on_update)
ps.show()
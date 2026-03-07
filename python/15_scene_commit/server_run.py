import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, Timer, Animation
from uipc import Vector3, Transform, Quaternion, AngleAxis
import uipc.builtin as builtin
from uipc.core import Engine, World, Scene, SceneIO, SceneSnapshot
from uipc.geometry import GeometrySlot, SimplicialComplex, SimplicialComplexIO, label_surface, label_triangle_orient, flip_inward_triangles, ground
from uipc.constitution import AffineBodyConstitution, StableNeoHookean
from uipc.gui import SceneGUI
from uipc.unit import MPa, GPa
from asset_dir import AssetDir

Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
this_folder = AssetDir.folder(__file__)
output_path = AssetDir.output_path(this_folder)
print(f'output path: {output_path}')

engine = Engine('cuda', workspace)
world = World(engine)

config = Scene.default_config()
dt = 0.02
config['dt'] = dt
config['gravity'] = [[0.0], [-9.8], [0.0]]
config['contact']['friction']['enable'] = True
scene = Scene(config)

# friction ratio and contact resistance
scene.contact_tabular().default_model(0.5, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()

# create constitution and contact model
abd = AffineBodyConstitution()
snk = StableNeoHookean()

# load cube mesh
io = SimplicialComplexIO()
cube_mesh = io.read(f'{AssetDir.tetmesh_path()}/cube.msh')
# label the surface, enable the contact
label_surface(cube_mesh)
# label the triangle orientation to export the correct surface mesh
label_triangle_orient(cube_mesh)
cube_mesh = flip_inward_triangles(cube_mesh)

# ABD
abd_cube_obj = scene.objects().create('abd')
abd_mesh = cube_mesh.copy()
abd.apply_to(abd_mesh, 10.0 * MPa)
t = Transform.Identity()
t.translate(Vector3.UnitX() * 1.1)
view(abd_mesh.transforms())[:] = t.matrix()
# set the initial velocity to 1 m/s in z direction
vt = Transform.Identity()
vt.translate(Vector3.UnitZ())
vt.scale(0.0)
velocity = abd_mesh.instances().find(builtin.velocity)
view(velocity)[:] = vt.matrix()
abd_cube_obj.geometries().create(abd_mesh)

# FEM
fem_cube_obj = scene.objects().create('fem')
fem_mesh = cube_mesh.copy()
snk.apply_to(fem_mesh)
velocity = fem_mesh.vertices().find(builtin.velocity)
# set the initial velocity to 1 m/s in x direction
view(velocity)[:] = Vector3.UnitZ()
fem_cube_obj.geometries().create(fem_mesh)

ground_height = -2.0
ground_obj = scene.objects().create('ground')
g = ground(ground_height)
ground_obj.geometries().create(g)

scene_io = SceneIO(scene)
# --------------------------------------------------
# save the scene to bson (binary json)
scene_io.save(f'{output_path}/scene/scene0.bson')
# OR use:
j = scene_io.to_json()
# and pass on the json to anywhere you want
# --------------------------------------------------

world.init(scene)
# record the snapshot of current scene
# to create scene commit
ss = SceneSnapshot(scene)

while world.frame() < 1000:
    world.advance()
    world.retrieve()
    path = f'{output_path}/scene/scene{world.frame()}.bson'
    
    # -------------------------------------------------------
    # commit the scene update to a file
    scene_io.commit(ss, path)
    # OR use:
    j = scene_io.commit_to_json(ss)
    # and pass on the json to anywhere you want
    # -------------------------------------------------------
    
    # update the scene snapshot
    ss = SceneSnapshot(scene) 
    print(f'frame {world.frame()} saved to {path}')

print('finished!')
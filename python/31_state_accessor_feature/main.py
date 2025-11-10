import numpy as np
import polyscope as ps
from polyscope import imgui

import uipc
from uipc import Logger, Timer, Transform, Quaternion, Vector3, Vector2, view, builtin, Matrix4x4
from uipc.core import (Engine, World, Scene, 
    AffineBodyStateAccessorFeature, FiniteElementStateAccessorFeature,
    SanityCheckResult)
from uipc.geometry import tetmesh, label_surface, ground, label_triangle_orient, flip_inward_triangles
from uipc.geometry import SimplicialComplexIO
from uipc.constitution import AffineBodyConstitution, ElasticModuli, StableNeoHookean
from uipc.gui import SceneGUI 
from uipc.unit import MPa, GPa, kPa

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
print(config)
scene = Scene(config)

obj = scene.objects().create('cubes')
sio = SimplicialComplexIO()
base_mesh = sio.read(f'{tetmesh_path}/cube.msh')

label_surface(base_mesh)
label_triangle_orient(base_mesh)
base_mesh = flip_inward_triangles(base_mesh)

fem_mesh = base_mesh.copy()
abd_mesh = base_mesh.copy()

snk = StableNeoHookean()
parm = ElasticModuli.youngs_poisson(20.0 * kPa, 0.49)
snk.apply_to(fem_mesh, parm)

abd = AffineBodyConstitution()
abd.apply_to(abd_mesh, kappa=1.0 * MPa)
trans_view = view(abd_mesh.transforms())
t = Transform.Identity()
t.translate(Vector3.Values([0.0, 1.2, 0.0]))
trans_view[:] = t.matrix()

obj.geometries().create(fem_mesh)
obj.geometries().create(abd_mesh)

g = scene.objects().create('ground')
g_mesh = ground(-0.6)
g.geometries().create(g_mesh)

world.init(scene)
world.dump()

# 1) Get feature after world initialization
abd_state_accessor:AffineBodyStateAccessorFeature = world.features().find(AffineBodyStateAccessorFeature)
fem_state_accessor:FiniteElementStateAccessorFeature = world.features().find(FiniteElementStateAccessorFeature)

assert abd_state_accessor is not None and fem_state_accessor is not None, 'This version of uipc does not support state accessor feature.'

# 2) Create a state_geo to contain data
abd_state_geo = abd_state_accessor.create_geometry()
abd_state_geo.instances().create(builtin.transform, Matrix4x4.Zero()) # tell the backend we need transform information
fem_state_geo = fem_state_accessor.create_geometry()
fem_state_geo.vertices().create(builtin.position, Vector3.Zero()) # tell the backend we need position information

ps.init()
sgui = SceneGUI(scene, 'split')
sgui.register()
sgui.set_edge_width(1.0)

run = False
def on_update():
    global run
    
    if(imgui.Button('run & stop')):
        run = not run
    
    move = False
    
    if(imgui.Button('Move FEM Position Y')):
        fem_state_accessor.copy_to(fem_state_geo) # Get state data from backend
        view(fem_state_geo.positions())[:] += Vector3.Values([0.0, 0.1, 0.0])
        fem_state_accessor.copy_from(fem_state_geo) # Copy modified data back to backend
        move = True
    
    if(imgui.Button('Move ABD Transform Y')):
        abd_state_accessor.copy_to(abd_state_geo) # Get State data from backend
        view(abd_state_geo.transforms())[:, 0:3, 3] += np.array([0.0, 0.1, 0.0])
        abd_state_accessor.copy_from(abd_state_geo) # Copy modified data back to backend
        move = True
    
    if(move):
        world.retrieve() # Update scene data from backend to reflect the changes
        sgui.update() # Update GUI
        # If penetration happens after manual modification, we need to do a recovery step
        # to the last valid state
        if(world.sanity_checker().check() != SanityCheckResult.Success):
            world.recover(0)
    
    # common simulation step
    if(run):
        world.advance()
        world.retrieve()
        sgui.update()

ps.set_user_callback(on_update)
ps.show()
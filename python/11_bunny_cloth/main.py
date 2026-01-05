import numpy as np
import polyscope as ps
from polyscope import imgui

import uipc
from uipc import Logger, Timer, Transform, Quaternion, Vector3, Vector2, view, builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import tetmesh, label_surface, label_triangle_orient, flip_inward_triangles
from uipc.geometry import SimplicialComplexIO
from uipc.constitution import AffineBodyConstitution, NeoHookeanShell, DiscreteShellBending, ElasticModuli2D, StrainLimitingBaraffWitkinShell
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
config['dt'] = 0.01
config['contact']['d_hat'] = 0.001
print(config)
scene = Scene(config)

# begin setup the scene
cloth = scene.objects().create('cloth')
t = Transform.Identity()
t.scale(2.0)
io = SimplicialComplexIO(t)
cloth_mesh = io.read(f'{trimesh_path}/grid20x20.obj')
label_surface(cloth_mesh)
#nks = NeoHookeanShell()
slbws = StrainLimitingBaraffWitkinShell()
dsb = DiscreteShellBending()
moduli = ElasticModuli2D.youngs_poisson(5 * MPa, 0.4)
slbws.apply_to(cloth_mesh, moduli=moduli, mass_density=200, thickness=0.001)
dsb.apply_to(cloth_mesh, bending_stiffness=0.006)
view(cloth_mesh.positions())[:] += 1.0
cloth.geometries().create(cloth_mesh)

bunny = scene.objects().create('bunny')
t = Transform.Identity()
t.translate(Vector3.UnitX() + Vector3.UnitZ())
io = SimplicialComplexIO(t)
bunny_mesh = io.read(f'{tetmesh_path}/bunny0.msh')
label_surface(bunny_mesh)
label_triangle_orient(bunny_mesh)
bunny_mesh = flip_inward_triangles(bunny_mesh)
abd = AffineBodyConstitution()
abd.apply_to(bunny_mesh, 100 * MPa)
is_fixed = bunny_mesh.instances().find(builtin.is_fixed)
view(is_fixed)[:] = 1

bunny.geometries().create(bunny_mesh)
# end setup the scene

world.init(scene)

ps.init()
ps.set_ground_plane_height(-1.0)
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
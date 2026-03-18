import polyscope as ps
from polyscope import imgui

import uipc
from uipc import Logger, Timer, Transform, view
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

engine = Engine('cuda', workspace)
world = World(engine)

# -----------------------------------------------------------------------------
# advanced scene config
# -----------------------------------------------------------------------------
scene = Scene()
# setup config after scene is created
config = scene.config()
dt_attr = config.find('dt')
view(dt_attr)[:] = 0.03
friction_enable = config.find('contact/friction/enable')
view(friction_enable)[:] = 1

print(config)
# -----------------------------------------------------------------------------


abd = AffineBodyConstitution()

obj = scene.objects().create('cubes')
t = Transform.Identity()
t.scale(0.10)
sio = SimplicialComplexIO(t)
cube_mesh = sio.read(f'{trimesh_path}/cube.obj')
abd.apply_to(cube_mesh, 100.0 * MPa)
label_surface(cube_mesh)
obj.geometries().create(cube_mesh)

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
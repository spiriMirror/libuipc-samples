import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, Animation
from uipc import Vector3
from uipc import builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import GeometrySlot, SimplicialComplex, ground, tetmesh, label_surface, label_triangle_orient, flip_inward_triangles
from uipc.constitution import ElasticModuli, StableNeoHookean, SoftPositionConstraint
from uipc.gui import SceneGUI
from uipc.unit import MPa, GPa
from asset_dir import AssetDir

Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
engine = Engine('cuda', workspace)
world = World(engine)

config = Scene.default_config()
dt = 0.02
config['dt'] = dt
config['gravity'] = [[0.0], [-9.8], [0.0]]
config['contact']['friction']['enable'] = True
scene = Scene(config)

# friction ratio and contact resistance
scene.contact_tabular().default_model(0.1, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()

snh = StableNeoHookean()
spc = SoftPositionConstraint()
tet_object = scene.objects().create('tet_object')
Vs = np.array([[0,1,0],
               [0,0,1],
               [-np.sqrt(3)/2, 0, -0.5],
               [np.sqrt(3)/2, 0, -0.5]])
Ts = np.array([[0,1,2,3]])
tet = tetmesh(Vs, Ts)
label_surface(tet)
label_triangle_orient(tet)
tet = flip_inward_triangles(tet)
moduli = ElasticModuli.youngs_poisson(0.1 * MPa, 0.49)
snh.apply_to(tet, moduli)
spc.apply_to(tet, 100) # constraint strength ratio
tet_object.geometries().create(tet)

ground_object = scene.objects().create('ground')
g = ground(-0.5)
ground_object.geometries().create(g)

animator = scene.animator()

def animate_tet(info:Animation.UpdateInfo): # animation function
    geo_slots:list[GeometrySlot] = info.geo_slots()
    geo:SimplicialComplex = geo_slots[0].geometry()
    rest_geo_slots:list[GeometrySlot] = info.rest_geo_slots()
    rest_geo:SimplicialComplex = rest_geo_slots[0].geometry()

    is_constrained = geo.vertices().find(builtin.is_constrained)
    is_constrained_view = view(is_constrained)
    aim_position = geo.vertices().find(builtin.aim_position)
    aim_position_view = view(aim_position)
    rest_position_view = rest_geo.positions().view()

    is_constrained_view[0] = 1

    t = info.dt() * info.frame()
    theta = np.pi * t
    y = -np.sin(theta)

    aim_position_view[0] = rest_position_view[0] + Vector3.UnitY() * y

animator.insert(tet_object, animate_tet)

world.init(scene)
sgui = SceneGUI(scene)

ps.init()
ps.set_ground_plane_height(-0.5)
tri_surf, _, _ = sgui.register()
tri_surf.set_edge_width(1)

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
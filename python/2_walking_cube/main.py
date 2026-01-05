import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, Timer, Animation
from uipc import Vector3, Transform, Quaternion, AngleAxis
from uipc import builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import GeometrySlot, SimplicialComplex, SimplicialComplexIO, label_surface, label_triangle_orient, flip_inward_triangles, ground
from uipc.constitution import AffineBodyConstitution, RotatingMotor
from uipc.gui import SceneGUI
from uipc.unit import MPa, GPa
from asset_dir import AssetDir

Logger.set_level(Logger.Level.Info)

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
scene.contact_tabular().default_model(0.5, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()

# create constitution and contact model
abd = AffineBodyConstitution()
rm = RotatingMotor()

# load cube mesh
io = SimplicialComplexIO()
cube_mesh = io.read(f'{AssetDir.tetmesh_path()}/cube.msh')
# label the surface, enable the contact
label_surface(cube_mesh)
# label the triangle orientation to export the correct surface mesh
label_triangle_orient(cube_mesh)
cube_mesh = flip_inward_triangles(cube_mesh)

cube_object = scene.objects().create('cube_obj')

abd.apply_to(cube_mesh, 10.0 * MPa)
rm.apply_to(
    cube_mesh, 
    100.0, # constraint strength ratio
    Vector3.UnitX(), # rotation axis
    np.pi / 1.0 # rotation speed
)

# move the cube up by 2 units
trans_view = view(cube_mesh.transforms())
t = Transform.Identity()
t.translate(Vector3.UnitY() * 2)
trans_view[0] = t.matrix()

cube_object.geometries().create(cube_mesh)

ground_obj = scene.objects().create('ground')
g = ground()
ground_obj.geometries().create(g)

animator = scene.animator()

def animate_cube(info:Animation.UpdateInfo): # animation function
    # get all geometries attached to the object
    geo_slots:list[GeometrySlot] = info.geo_slots()
    geo:SimplicialComplex = geo_slots[0].geometry()

    # by setting is_constrained to 1, the cube will be controlled by the animation
    is_constrained = geo.instances().find(builtin.is_constrained)
    view(is_constrained)[0] = 1

    # using the RotatingMotor to animate the cube
    RotatingMotor.animate(geo, info.dt())

animator.insert(cube_object, animate_cube)

world.init(scene)

world.sanity_checker().report()

sgui = SceneGUI(scene)

ps.init()
ps.set_ground_plane_height(0)
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
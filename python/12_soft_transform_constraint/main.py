import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, Timer, Animation
from uipc import Vector3, Transform, Quaternion, AngleAxis
from uipc import builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import GeometrySlot, SimplicialComplex, SimplicialComplexIO, label_surface, label_triangle_orient, flip_inward_triangles, ground
from uipc.constitution import AffineBodyConstitution, SoftTransformConstraint
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
scene.contact_tabular().default_model(0.5, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()

# create constitution and contact model
abd = AffineBodyConstitution()
stc = SoftTransformConstraint()

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
stc.apply_to(cube_mesh, np.array([
    100.0, # strength ratio of translation constraint
    100.0 # strength ratio of rotation constraint
    # strength = strength_ratio * body_mass
]))

cube_object.geometries().create(cube_mesh)

ground_height = -2.0
ground_obj = scene.objects().create('ground')
g = ground(ground_height)
ground_obj.geometries().create(g)

animator = scene.animator()

def animate_cube(info:Animation.UpdateInfo): # animation function
    # get all geometries attached to the object
    geo_slots:list[GeometrySlot] = info.geo_slots()
    geo:SimplicialComplex = geo_slots[0].geometry()

    time = info.dt() * info.frame()

    # by setting is_constrained to 1, the cube will be controlled by the animation
    is_constrained = geo.instances().find(builtin.is_constrained)
    view(is_constrained)[0] = 1

    # get the aim_transform from the geometry
    aim_transform = geo.instances().find(builtin.aim_transform)
    
    # create the transform matrix
    t = Transform.Identity()
    # + An interesting periodic linear motion
    t.translate(0.5 * Vector3.UnitY() * np.sin(np.pi * time))
    # + A boring rotation
    rot = AngleAxis(time * np.pi, Vector3.UnitZ()) # rotate around z-axis
    t.rotate(rot)
    
    view(aim_transform)[:] = t.matrix()

animator.insert(cube_object, animate_cube)

world.init(scene)

ps.init()
sgui = SceneGUI(scene)
sgui.register()
sgui.set_edge_width(1)

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
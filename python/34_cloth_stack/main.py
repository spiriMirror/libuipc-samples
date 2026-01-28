import numpy as np
import polyscope as ps
from polyscope import imgui

import uipc
from uipc import Logger, Timer, Transform, Vector3, AngleAxis, view, builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import label_surface, ground
from uipc.geometry import SimplicialComplexIO
from uipc.constitution import (
    AffineBodyConstitution,
    DiscreteShellBending,
    ElasticModuli,
    StrainLimitingBaraffWitkinShell,
)
from uipc.gui import SceneGUI
from uipc.unit import MPa, kPa

from asset_dir import AssetDir

Timer.enable_all()
Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
trimesh_path = AssetDir.trimesh_path()

engine = Engine('cuda', workspace)
world = World(engine)

config = Scene.default_config()
config['dt'] = 0.02
config['contact']['d_hat'] = 0.001
config['contact']['enable'] = True
config['contact']['friction']['enable'] = False
config['contact']['constitution'] = 'ipc'
config['gravity'] = [[0.0], [0.0], [-9.8]]
config['newton']['velocity_tol'] = 0.1
config['newton']['transrate_tol'] = 10
config['linear_system']['tol_rate'] = 1e-3
scene = Scene(config)

scene.contact_tabular().default_model(0.0, 1.0e7)

abd = AffineBodyConstitution()
slbws = StrainLimitingBaraffWitkinShell()
dsb = DiscreteShellBending()
cloth_moduli = ElasticModuli.youngs_poisson(60 * kPa, 0.49)


def create_cloth(name: str, mesh_file: str, scale: float, pos, rotation, bending_stiffness: float):
    pre = Transform.Identity()
    pre.translate(pos)
    pre.rotate(rotation)
    pre.scale(scale)
    io = SimplicialComplexIO(pre)
    cloth_mesh = io.read(mesh_file)
    label_surface(cloth_mesh)
    slbws.apply_to(cloth_mesh, moduli=cloth_moduli, mass_density=200, thickness=0.001)
    dsb.apply_to(cloth_mesh, bending_stiffness=bending_stiffness)
    cloth_obj = scene.objects().create(name)
    cloth_obj.geometries().create(cloth_mesh)


# --------------------------------------------------------------------------
# Cloth stack (2 layers)
# --------------------------------------------------------------------------
create_cloth(
    name='cloth_large',
    mesh_file=f'{trimesh_path}/grid80x80.obj',
    scale=0.5,
    pos=(0.5, 0.0, 0.1),
    rotation=AngleAxis(np.pi / 2, Vector3.UnitX()),
    bending_stiffness=10.0,
)
create_cloth(
    name='cloth_small',
    mesh_file=f'{trimesh_path}/grid40x40.obj',
    scale=0.3,
    pos=(0.5, 0.0, 0.14),
    rotation=AngleAxis(np.pi / 2, Vector3.UnitX()),
    bending_stiffness=40.0,
)
create_cloth(
    name='cloth_small',
    mesh_file=f'{trimesh_path}/grid20x20.obj',
    scale=0.2,
    pos=(0.5, 0.0, 0.16),
    rotation=AngleAxis(np.pi / 2, Vector3.UnitX()),
    bending_stiffness=40.0,
)
create_cloth(
    name='cloth_small',
    mesh_file=f'{trimesh_path}/grid10x10.obj',
    scale=0.1,
    pos=(0.5, 0.0, 0.18),
    rotation=AngleAxis(np.pi / 2, Vector3.UnitX()),
    bending_stiffness=40.0,
)

# --------------------------------------------------------------------------
# Fixed cube grid (4x4)
# --------------------------------------------------------------------------
cube_obj = scene.objects().create('cubes')
cube_size = 0.05
cube_height = 0.02501
grid_spacing = 0.15

pre = Transform.Identity()
pre.scale(cube_size)
cube_io = SimplicialComplexIO(pre)
cube_mesh = cube_io.read(f'{trimesh_path}/cube.obj')
label_surface(cube_mesh)
cube_mesh.instances().resize(16)

abd.apply_to(cube_mesh, 100 * MPa)
trans_view = view(cube_mesh.transforms())
is_fixed = cube_mesh.instances().find(builtin.is_fixed)
is_fixed_view = view(is_fixed)

idx = 0
for i in range(4):
    for j in range(4):
        x = (i + 1.7) * grid_spacing
        y = (j - 1.5) * grid_spacing
        t = Transform.Identity()
        t.translate([x, y, cube_height])
        trans_view[idx] = t.matrix()
        # print volume of the cube
        cube_volume = cube_mesh.instances().find(builtin.volume).view()
        is_fixed_view[idx] = 1
        idx += 1

cube_obj.geometries().create(cube_mesh)

# Ground plane
ground_obj = scene.objects().create('ground')
g = ground(-0.001, Vector3.UnitZ())
ground_obj.geometries().create(g)

world.init(scene)

ps.init()
sgui = SceneGUI(scene, 'split')
sgui.register()
sgui.set_edge_width(1.0)

run = False


def on_update():
    global run
    if imgui.Button('run & stop'):
        run = not run

    if run:
        world.advance()
        world.retrieve()
        sgui.update()


ps.set_user_callback(on_update)
ps.show()
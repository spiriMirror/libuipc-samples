"""
Example: Create an affine body from rigid body quantities.

Instead of computing the ABD mass matrix from mesh geometry,
this example shows how to provide rigid body properties
(mass, center of mass, inertia tensor) and convert them
to the 12x12 ABD mass matrix using `from_rigid_body`.

Two bodies are created:
  - A cube mesh with mass from rigid body properties (free-falling)
  - A fixed ground plane
"""

import numpy as np
import polyscope as ps
from polyscope import imgui

import uipc
from uipc import Logger, Timer
from uipc.core import Engine, World, Scene
from uipc.geometry import (
    SimplicialComplexIO,
    tetmesh,
    pointcloud,
    label_surface,
    label_triangle_orient,
    flip_inward_triangles,
    ground,
)
from uipc.geometry.affine_body import from_rigid_body
from uipc.constitution import AffineBodyConstitution
from uipc.gui import SceneGUI
from uipc.unit import MPa, GPa
from asset_dir import AssetDir

Timer.enable_all()
Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.01
config["gravity"] = [[0.0], [-9.8], [0.0]]
scene = Scene(config)

abd = AffineBodyConstitution()

scene.contact_tabular().default_model(0.5, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()

# ---------------------------------------------------------------
# Body 1: A cube with mass properties from rigid body data
# ---------------------------------------------------------------
# Load a cube mesh (for collision geometry)
io = SimplicialComplexIO()
cube = io.read(f"{AssetDir.tetmesh_path()}/cube.msh")
label_surface(cube)
label_triangle_orient(cube)
cube = flip_inward_triangles(cube)
default_element.apply_to(cube)

# Rigid body properties for the cube:
#   mass = 1000 kg, center of mass = (0.5, 0.5, 0.5)
#   I_cm = (m * a^2 / 6) * I_3  (uniform cube, side a=1)
mass = 1000.0
center_of_mass = np.array([0.5, 0.5, 0.5])
I_cm = (mass / 6.0) * np.eye(3)
volume = 1.0

# Convert rigid body quantities to the 12x12 ABD mass matrix
M = from_rigid_body(mass, center_of_mass, I_cm)
print("ABD mass matrix from rigid body:")
print(f"  mass = {mass}, center_of_mass = {center_of_mass}")
print(f"  M shape = {M.shape}")

# Apply constitution with the explicit mass matrix and volume
abd.apply_to(cube, 100 * MPa, M, volume)

# Place the cube above the ground
pos_view = uipc.view(cube.positions())
pos_view += uipc.Vector3.UnitY() * 3.0

object_cube = scene.objects().create("rigid_cube")
object_cube.geometries().create(cube)

# ---------------------------------------------------------------
# Body 2: An empty affine body (dynamics only, no collision)
# ---------------------------------------------------------------
# This body has no mesh vertices -- it only participates in dynamics.
empty_mesh = pointcloud(np.zeros((0, 3)))
empty_mesh.instances().resize(1)

empty_mass = 500.0
empty_com = np.array([0.0, 0.0, 0.0])
empty_I_cm = (empty_mass / 6.0) * np.eye(3)
M_empty = from_rigid_body(empty_mass, empty_com, empty_I_cm)
abd.apply_to(empty_mesh, 100 * MPa, M_empty, 0.5)

trans_view = uipc.view(empty_mesh.transforms())
T = np.eye(4)
T[1, 3] = 5.0  # start at y=5
trans_view[0] = T

object_empty = scene.objects().create("empty_body")
object_empty.geometries().create(empty_mesh)

# ---------------------------------------------------------------
# Ground plane
# ---------------------------------------------------------------
g = ground()
default_element.apply_to(g)
scene.objects().create("ground").geometries().create(g)

# ---------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------
world.init(scene)
sgui = SceneGUI(scene)

ps.init()
tri_surf, line_surf, point_surf = sgui.register()
tri_surf.set_edge_width(1)

run = False


def on_update():
    global run
    if imgui.Button("run & stop"):
        run = not run

    if run:
        world.advance()
        world.retrieve()
        sgui.update()


ps.set_user_callback(on_update)
ps.show()

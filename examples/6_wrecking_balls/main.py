import json
import numpy as np
import polyscope as ps
from polyscope import imgui
from pathlib import Path

import uipc
from uipc import view
from uipc import Vector3, Transform, Logger, Quaternion, AngleAxis, Timer
from uipc.core import Engine, World, Scene
from uipc.geometry import SimplicialComplex, SimplicialComplexIO, ground, label_surface, label_triangle_orient, flip_inward_triangles
from uipc.constitution import AffineBodyConstitution
from uipc.unit import MPa, GPa
from uipc.gui import SceneGUI

from asset_dir import AssetDir


def process_surface(sc: SimplicialComplex):
    label_surface(sc)
    label_triangle_orient(sc)
    sc = flip_inward_triangles(sc)
    return sc


Timer.enable_all()
Logger.set_level(Logger.Level.Info)
workspace = AssetDir.output_path(__file__)
folder = AssetDir.folder(__file__)

engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.01
config["contact"]["d_hat"] = 0.01
config["newton"]["transrate_tol"] = 10
config["newton"]["velocity_tol"] = 1
print(config)

scene = Scene(config)
abd = AffineBodyConstitution()
scene.contact_tabular().default_model(0.02, 10 * GPa)
default_contact = scene.contact_tabular().default_element()

io = SimplicialComplexIO()

scene_json_path = Path(AssetDir.asset_path()) / "sim_data" / "wrecking_ball.json"
with open(scene_json_path, "r", encoding="utf-8") as f:
    wrecking_ball_scene = json.load(f)

tetmesh_dir = AssetDir.tetmesh_path()

cube = io.read(f"{tetmesh_dir}/cube.msh")
cube = process_surface(cube)
ball = io.read(f"{tetmesh_dir}/ball.msh")
ball = process_surface(ball)
link = io.read(f"{tetmesh_dir}/link.msh")
link = process_surface(link)

cube_obj = scene.objects().create("cubes")
ball_obj = scene.objects().create("balls")
link_obj = scene.objects().create("links")

abd.apply_to(cube, 100 * MPa)
default_contact.apply_to(cube)

abd.apply_to(ball, 100 * MPa)
default_contact.apply_to(ball)

abd.apply_to(link, 100 * MPa)
default_contact.apply_to(link)


def build_mesh(desc, obj: uipc.core.Object, mesh: SimplicialComplex):
    t = Transform.Identity()
    position = Vector3.Zero()
    if "position" in desc:
        position[0] = desc["position"][0]
        position[1] = desc["position"][1]
        position[2] = desc["position"][2]
        t.translate(position)

    if "rotation" in desc:
        rotation = Vector3.Zero()
        rotation[0] = desc["rotation"][0]
        rotation[1] = desc["rotation"][1]
        rotation[2] = desc["rotation"][2]
        rotation *= np.pi / 180
        q = (
            AngleAxis(rotation[2][0], Vector3.UnitZ())
            * AngleAxis(rotation[1][0], Vector3.UnitY())
            * AngleAxis(rotation[0][0], Vector3.UnitX())
        )
        t.rotate(q)

    is_fixed = 0
    if "is_dof_fixed" in desc:
        is_fixed = desc["is_dof_fixed"]

    this_mesh = mesh.copy()
    view(this_mesh.transforms())[0] = t.matrix()
    is_fixed_attr = this_mesh.instances().find("is_fixed")
    view(is_fixed_attr)[0] = is_fixed
    obj.geometries().create(this_mesh)


for obj in wrecking_ball_scene:
    if obj["mesh"] == "link.msh":
        build_mesh(obj, link_obj, link)
    elif obj["mesh"] == "ball.msh":
        build_mesh(obj, ball_obj, ball)
    elif obj["mesh"] == "cube.msh":
        build_mesh(obj, cube_obj, cube)

ground_obj = scene.objects().create("ground")
ground_obj.geometries().create(ground(-1.0))

sgui = SceneGUI(scene)
world.init(scene)

ps.init()
tri_surf, _, _ = sgui.register()
tri_surf.set_edge_width(1)

run = False


def on_update():
    global run
    if imgui.Button("run & stop"):
        run = not run

    if run:
        world.advance()
        world.retrieve()
        world.dump()
        Timer.report()

    sgui.update()


ps.set_user_callback(on_update)
ps.show()

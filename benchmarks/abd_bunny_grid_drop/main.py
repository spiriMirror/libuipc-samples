import polyscope as ps
from polyscope import imgui

from uipc import Logger, Timer, Transform, Vector3, view
from uipc.core import Engine, World, Scene
from uipc.geometry import SimplicialComplexIO, label_surface, label_triangle_orient, flip_inward_triangles
from uipc.constitution import AffineBodyConstitution
from uipc.gui import SceneGUI
from uipc.unit import MPa, GPa

from asset_dir import AssetDir


GRID_X = 10
GRID_Z = 10
GRID_SPACING = 1.2
DROP_HEIGHT = 8.0
SCALE = 0.1


def create_scene():
    config = Scene.default_config()
    config["dt"] = 0.01
    config["contact"]["d_hat"] = 0.01
    config["newton"]["transrate_tol"] = 10
    config["newton"]["velocity_tol"] = 1
    # Keep default collision_detection/method for GUI preview.
    scene = Scene(config)

    scene.contact_tabular().default_model(0.2, 10 * GPa)
    default_contact = scene.contact_tabular().default_element()
    abd = AffineBodyConstitution()

    transform = Transform.Identity()
    transform.scale(SCALE)
    io = SimplicialComplexIO(transform)
    bunny_path = f"{AssetDir.tetmesh_path()}/bunny0.msh"
    bunny = io.read(bunny_path)
    label_surface(bunny)
    label_triangle_orient(bunny)
    bunny = flip_inward_triangles(bunny)

    abd.apply_to(bunny, 100 * MPa)
    default_contact.apply_to(bunny)
    bunny.instances().resize(GRID_X * GRID_Z)

    transforms = view(bunny.transforms())
    for iz in range(GRID_Z):
        for ix in range(GRID_X):
            idx = iz * GRID_X + ix
            t = Transform.Identity()
            x = (ix - (GRID_X - 1) * 0.5) * GRID_SPACING
            z = (iz - (GRID_Z - 1) * 0.5) * GRID_SPACING
            p = Vector3.Zero()
            p[0] = x
            p[1] = DROP_HEIGHT
            p[2] = z
            t.translate(p)
            transforms[idx] = t.matrix()

    bunnies = scene.objects().create("bunny_grid")
    bunnies.geometries().create(bunny)
    return scene


def main():
    Timer.enable_all()
    Logger.set_level(Logger.Level.Info)
    workspace = AssetDir.output_path(__file__)

    engine = Engine("cuda", workspace)
    world = World(engine)
    scene = create_scene()
    world.init(scene)

    sgui = SceneGUI(scene, "split")
    ps.init()
    sgui.register()
    sgui.set_edge_width(1.0)

    run = False

    def on_update():
        nonlocal run
        if imgui.Button("Run & Stop"):
            run = not run
        imgui.Text(f"Frame: {world.frame()}")
        if run:
            world.advance()
            world.retrieve()
            sgui.update()
            Timer.report()
        else:
            sgui.update()

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    main()

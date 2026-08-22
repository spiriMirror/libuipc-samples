"""Example 87 -- a URDF robot hand with a rigid cube on the ground.

The hand's links are affine bodies pinned by soft transform constraints;
pose the hand with the GUI sliders, then press `run` to let contact resolve
the interaction with the cube.

Controls: joint/root sliders pose the hand manually (poses can be saved and
loaded as json); `run` starts/stops the simulation.
"""

import json
import pathlib

import numpy as np
import polyscope as ps
from polyscope import imgui

import uipc
from uipc import Logger, Transform, Vector3, builtin
from uipc.constitution import AffineBodyConstitution
from uipc.core import Engine, Scene, SceneIO, World
from uipc.geometry import SimplicialComplexIO, UrdfIO, label_surface
from uipc.gui import SceneGUI
from uipc.unit import GPa, MPa

import paths
from articulation import (ControlPanel, HandDriver, hold_current_pose,
                          make_link_constitutions)

Logger.set_level(Logger.Level.Warn)

DT = 0.01
GROUND_HEIGHT = -0.18
MAX_FRAME = 500


def make_scene_config() -> dict:
    config = Scene.default_config()
    config['dt'] = DT
    config['newton']['velocity_tol'] = 0.05
    config['contact']['d_hat'] = 0.001
    config['contact']['enable'] = True
    config['collision_detection']['method'] = 'info_stackless_bvh'
    config['sanity_check']['enable'] = False
    config['gravity'] = [[0.0], [-9.8], [0.0]]
    print(config)
    return config


def setup_contact(scene: Scene):
    tabular = scene.contact_tabular()
    arm = tabular.create('arm')
    tabular.default_model(0.8, 1.0 * GPa)
    ground = tabular.create('g_elem')
    cube = tabular.create('cube')

    tabular.insert(arm, arm, 0.0, 0.0, False)
    tabular.insert(cube, ground, 0.8, 1.0 * GPa, True)
    tabular.insert(ground, arm, 0.0, 0.0, False)
    tabular.insert(cube, cube, 0.0, 0.0, False)
    return arm, ground, cube


def load_hand(scene: Scene, contact) -> HandDriver:
    urdf_config = UrdfIO.default_config()
    print(urdf_config)
    urdf_config['load_visual_mesh'] = False
    arm = scene.objects().create('arm')
    controller = UrdfIO(urdf_config).read(
        arm, f'{paths.urdf_dir()}/robot_hand/robot_hand.urdf')

    make_link_constitutions(controller, contact)
    scene.animator().insert(arm, hold_current_pose)
    return controller


def load_cube(scene: Scene, contact):
    placement = Transform.Identity()
    placement.translate(Vector3.Values([0.0, -0.144, 0.15]))
    placement.scale(0.05)

    mesh = SimplicialComplexIO(placement).read(
        f'{paths.tetmesh_dir()}/cube.msh')
    label_surface(mesh)

    AffineBodyConstitution().apply_to(mesh, 100.0 * MPa, 1e3)
    contact.apply_to(mesh)
    scene.objects().create('cube').geometries().create(mesh)


def build_world():
    engine = Engine('cuda', paths.output_dir())
    world = World(engine)
    scene = Scene(make_scene_config())

    arm_contact, ground_contact, cube_contact = setup_contact(scene)

    controller = load_hand(scene, arm_contact)
    load_cube(scene, cube_contact)

    ground = scene.objects().create('ground')
    ground_mesh = uipc.geometry.ground(GROUND_HEIGHT)
    ground_contact.apply_to(ground_mesh)
    ground.geometries().create(ground_mesh)

    driver = HandDriver(controller, None, DT)
    driver.load_pose_file(str(paths.case_dir() / 'joint_poses' / 'pose_0.json'),
                          attr=builtin.transform)
    driver.snap_to_goal()

    world.init(scene)
    # NOTE: World only holds a weak reference to its Engine; the caller must
    # keep `engine` alive for as long as the world is used.
    return world, scene, driver, engine


class DemoApp:
    """Polyscope callback state: run flag, pose (un)dump helpers."""

    def __init__(self, world, scene, driver, scene_gui, scene_io):
        self.world = world
        self.driver = driver
        self.scene_gui = scene_gui
        self.scene_io = scene_io
        self.panel = ControlPanel(driver)
        self.running = False
        self.pose_file = str(paths.case_dir() / 'joint_poses' / 'pose_.json')
        self.pose_file_taken = False

    def frame(self):
        imgui.Text(f'frame: {self.world.frame()}')

        if imgui.Button('stop' if self.running else 'run'):
            self.running = not self.running

        if self.world.frame() >= MAX_FRAME:
            self.running = False
            imgui.Text('Simulation finished!')

        if self.running:
            self._simulation_frame()
        else:
            self._paused_frame()

    def _simulation_frame(self):
        self.panel.draw(builtin.aim_transform)
        self.world.advance()
        self.world.retrieve()
        self.scene_gui.update()
        self.scene_io.write_surface(
            f'{paths.output_dir()}/scene_surface{self.world.frame()}.obj')

    def _paused_frame(self):
        if self.panel.draw(builtin.transform):
            self.scene_gui.update()

        changed, self.pose_file = imgui.InputText(
            'manager dump file', self.pose_file)
        if imgui.Button('save manager'):
            self.pose_file_taken = pathlib.Path(self.pose_file).exists()
            if not self.pose_file_taken:
                with open(self.pose_file, 'w') as f:
                    json.dump(self.driver.dump_pose(), f, indent=4)
        if self.pose_file_taken:
            imgui.Text('file exists, wont save!')
        if imgui.Button('load manager'):
            path = pathlib.Path(self.pose_file)
            if path.exists():
                with open(path) as f:
                    self.driver.load_pose(json.load(f),
                                          attr=builtin.aim_transform)
                self.scene_gui.update()

        if imgui.Button('dump'):
            self.world.dump()
        if imgui.Button('recover'):
            self.world.recover()
            self.world.retrieve()
            self.scene_gui.update()


def main():
    world, scene, driver, engine = build_world()

    ps.init()
    scene_gui = SceneGUI(scene)
    ps.set_ground_plane_height(GROUND_HEIGHT)
    ps.set_ground_plane_height_mode('manual')

    surface, _, _ = scene_gui.register()
    surface.set_material('ceramic')

    app = DemoApp(world, scene, driver, scene_gui, SceneIO(scene))
    ps.set_user_callback(app.frame)
    ps.show()


if __name__ == '__main__':
    main()

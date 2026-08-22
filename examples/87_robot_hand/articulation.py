"""Drive an articulated URDF body through soft constraints.

Three layers, bottom up:

- ``RootMotionDriver``  -- rate-limited 6-DoF goal tracking for the base link.
- ``JointMotionDriver`` -- rate-limited goal tracking for every revolute joint.
- ``HandDriver``        -- bundles both drivers, applies them to the scene,
  and (de)serializes pose snapshots.

``ControlPanel`` renders the imgui widgets that steer a ``HandDriver``.
"""

import json
import pathlib

import ikpy.chain
import numpy as np
import polyscope.imgui as imgui

import uipc
from uipc import Animation, builtin
from uipc.constitution import AffineBodyConstitution, SoftTransformConstraint
from uipc.core import ContactElement
from uipc.geometry import UrdfController
from uipc.unit import MPa

CLOSE_TOL = 1e-6


def make_link_constitutions(controller: UrdfController, contact: ContactElement):
    """Attach ABD dynamics + soft transform constraints to every URDF link."""
    abd = AffineBodyConstitution()
    stc = SoftTransformConstraint()
    for slot in controller.links():
        geo = slot.geometry()
        label = geo.meta().find('urdf/name').view()[0]
        print(f'  link `{label}` <- affine body + soft transform')
        abd.apply_to(geo, 100 * MPa, mass_density=5e3)
        uipc.view(geo.instances().find(builtin.is_dynamic))[:] = 0
        contact.apply_to(geo)
        stc.apply_to(geo, np.array([10e3, 10e3]))
        uipc.view(geo.instances().find(builtin.is_constrained))[:] = 1
        aim = geo.instances().find(builtin.aim_transform)
        uipc.view(aim)[:] = geo.transforms().view()


def hold_current_pose(info: Animation.UpdateInfo):
    """Animator callback: constraints already pin the links, nothing to do."""
    pass


class IkSolver:
    """Thin wrapper around ikpy for optional target-based posing."""

    def __init__(self, urdf_file: str, base_elements: list[str] = ['world']):
        self._chain = ikpy.chain.Chain.from_urdf_file(
            urdf_file, base_elements=base_elements)
        self._names = [link.name for link in self._chain.links]
        self._chain.forward_kinematics(np.zeros(len(self._chain.links)))

    @property
    def link_names(self):
        return self._names

    def solve(self, position, orientation=None):
        angles = self._chain.inverse_kinematics(position, orientation)
        return dict(zip(self._names, angles))


class RootMotionDriver:
    """Rate-limited goal tracking for the articulated root."""

    def __init__(self, controller: UrdfController, dt: float):
        self._controller = controller
        self._dt = dt
        self.xyz = np.zeros(3)
        self.rpy = np.zeros(3)
        self.goal_xyz = np.zeros(3)
        self.goal_rpy = np.zeros(3)
        self.linear_cap = 0.05
        self.angular_cap = {axis: np.deg2rad(40.0) for axis in ('rx', 'ry', 'rz')}

    # -- caps ---------------------------------------------------------------
    def cap_linear(self, velocity: float):
        self.linear_cap = velocity

    def cap_angular(self, *args):
        """cap_angular(omega) for all axes, or cap_angular(axis, omega)."""
        if len(args) == 1:
            for axis in self.angular_cap:
                self.angular_cap[axis] = args[0]
        else:
            axis, omega = args
            if axis in self.angular_cap:
                self.angular_cap[axis] = omega

    # -- goal handling --------------------------------------------------------
    def teleport(self, xyz, rpy):
        self.xyz = np.asarray(xyz, dtype=float)
        self.rpy = np.asarray(rpy, dtype=float)
        self.goal_xyz = self.xyz
        self.goal_rpy = self.rpy
        self._controller.move_root(self.xyz, self.rpy)

    def snap_to_goal(self):
        self.xyz = self.goal_xyz
        self.rpy = self.goal_rpy
        self._controller.move_root(self.xyz, self.rpy)

    def set_goal(self, xyz, rpy):
        self.goal_xyz = np.asarray(xyz, dtype=float)
        self.goal_rpy = np.asarray(rpy, dtype=float)

    def step(self) -> bool:
        d_xyz = self.goal_xyz - self.xyz
        d_rpy = self.goal_rpy - self.rpy

        step_lin = self.linear_cap * self._dt
        norm = np.linalg.norm(d_xyz)
        if norm > step_lin:
            d_xyz = d_xyz / norm * step_lin

        step_ang = np.array(list(self.angular_cap.values())) * self._dt
        d_rpy = np.asarray(d_rpy)
        for i in range(3):
            if abs(d_rpy[i]) > step_ang[i]:
                d_rpy[i] = np.sign(d_rpy[i]) * step_ang[i]

        self.xyz = self.xyz + d_xyz
        self.rpy = self.rpy + d_rpy
        self._controller.move_root(self.xyz, self.rpy)
        return norm > CLOSE_TOL or np.linalg.norm(d_rpy) > CLOSE_TOL

    def reset(self):
        self.xyz = np.zeros(3)
        self.rpy = np.zeros(3)
        self._controller.move_root(self.xyz, self.rpy)

    def settled(self) -> bool:
        return (np.linalg.norm(self.xyz - self.goal_xyz) <= CLOSE_TOL
                and np.linalg.norm(self.rpy - self.goal_rpy) <= CLOSE_TOL)

    def dump(self) -> dict:
        return {
            'root_xyz': self.xyz.tolist(),
            'root_dst_xyz': self.goal_xyz.tolist(),
            'root_rpy': self.rpy.tolist(),
            'root_dst_rpy': self.goal_rpy.tolist(),
        }

    def load(self, data: dict):
        self.goal_xyz = np.asarray(data.get('root_dst_xyz', self.xyz))
        self.goal_rpy = np.asarray(data.get('root_dst_rpy', self.rpy))


class JointMotionDriver:
    """Rate-limited goal tracking for every revolute joint."""

    def __init__(self, controller: UrdfController, dt: float):
        self._controller = controller
        self._dt = dt
        joints = controller.revolute_joints()
        names = joints.geometry().instances().find('name').view()
        self.state = {name: [0.0, 0.0] for name in names}  # name -> [current, goal]
        self.angular_cap = {name: np.deg2rad(40.0) for name in names}

    def cap_angular(self, *args):
        """cap_angular(omega) for all joints, or cap_angular(name, omega)."""
        if len(args) == 1:
            for name in self.angular_cap:
                self.angular_cap[name] = args[0]
        else:
            name, omega = args
            if name in self.angular_cap:
                self.angular_cap[name] = omega

    def set_goal(self, name: str, angle: float):
        if name in self.state:
            self.state[name][1] = angle

    def teleport(self, name: str, angle: float):
        if name in self.state:
            self.state[name][0] = angle
            self.state[name][1] = angle
            self._controller.rotate_to(name, angle)

    def snap_to_goal(self):
        for name, (_, goal) in self.state.items():
            self.state[name][0] = goal
            self._controller.rotate_to(name, goal)

    def current(self, name: str) -> float:
        return self.state.get(name, [0.0])[0]

    def step(self) -> bool:
        changed = False
        for name, (cur, goal) in self.state.items():
            delta = goal - cur
            if cur != goal:
                cap = self.angular_cap[name] * self._dt
                if abs(delta) > cap:
                    delta = np.sign(delta) * cap
                cur += delta
                self.state[name][0] = cur
                changed = True
            self._controller.rotate_to(name, cur)
        return changed

    def reset(self):
        for name in self.state:
            self.state[name][0] = 0.0
            self._controller.rotate_to(name, 0.0)

    def settled(self) -> bool:
        return all(abs(cur - goal) <= CLOSE_TOL
                   for cur, goal in self.state.values())

    def dump(self) -> dict:
        return {name: cur for name, (cur, _) in self.state.items()}

    def load(self, data: dict):
        for name, angle in data.items():
            if name in self.state:
                self.state[name][1] = angle


class HandDriver:
    """Coordinates the root and joint drivers of one articulated hand."""

    def __init__(self, controller: UrdfController, ik: IkSolver | None, dt: float):
        self.controller = controller
        self.revolute_joints = controller.revolute_joints()
        self.root = RootMotionDriver(controller, dt)
        self.joints = JointMotionDriver(controller, dt)
        self.ik = ik
        self.dt = dt

    # -- high-level commands --------------------------------------------------
    def set_joint_goal(self, name: str, angle: float):
        self.joints.set_goal(name, angle)

    def set_root_goal(self, xyz, rpy):
        self.root.set_goal(xyz, rpy)

    def teleport_joint(self, name: str, angle: float):
        self.joints.teleport(name, angle)
        self.joints.step()

    def teleport_root(self, xyz, rpy):
        self.root.teleport(xyz, rpy)
        self.root.step()

    def ik_to(self, position, orientation=None):
        if self.ik is None:
            return
        for name, angle in self.ik.solve(position, orientation).items():
            self.joints.set_goal(name, angle)

    def step(self, attr: str = builtin.transform) -> bool:
        self.joints.step()
        self.root.step()
        self.controller.apply_to(attr=attr)
        self.controller.sync_visual_mesh()
        return True

    def settled(self) -> bool:
        return self.joints.settled() and self.root.settled()

    def snap_to_goal(self):
        self.joints.snap_to_goal()
        self.root.snap_to_goal()
        self.controller.apply_to(attr=builtin.transform)
        self.controller.sync_visual_mesh()

    # -- pose snapshots ---------------------------------------------------------
    def dump_pose(self) -> dict:
        return {
            'transform_controller': self.root.dump(),
            'angle_controller': self.joints.dump(),
        }

    def load_pose(self, data: dict, attr: str = builtin.transform):
        self.root.load(data['transform_controller'])
        self.joints.load(data['angle_controller'])
        self.joints.step()
        self.root.step()
        self.controller.apply_to(attr=attr)

    def load_pose_file(self, file: str, attr: str = builtin.transform):
        path = pathlib.Path(file)
        if not path.exists():
            print(f'pose file not found: {file}')
            return
        with open(path) as f:
            self.load_pose(json.load(f), attr=attr)


class KeyboardJog:
    """WASDQE keyboard jog for the IK target point."""

    def __init__(self):
        self.pos = np.zeros(3)
        self.step_size = 0.01

    @staticmethod
    def _down(name: str) -> bool:
        key = imgui.__dict__[f'ImGuiKey_{name}']
        return imgui.IsKeyDown(imgui.GetKeyIndex(key))

    def poll(self) -> bool:
        moved = False
        changed, size = imgui.InputFloat('step size', self.step_size)
        if changed:
            self.step_size = size
        for key, axis, sign in (('W', 2, 1), ('S', 2, -1), ('A', 0, -1),
                                ('D', 0, 1), ('Q', 1, -1), ('E', 1, 1)):
            if self._down(key):
                self.pos[axis] += sign * self.step_size
                moved = True
        return moved


class ControlPanel:
    """imgui widgets that steer a HandDriver."""

    def __init__(self, driver: HandDriver):
        self.driver = driver
        self.ik = driver.ik
        if self.ik is not None:
            self.ik_position = np.zeros(3)
            self.ik_orientation = np.zeros(3)
            self.jog = KeyboardJog()
        self.root_xyz = np.zeros(3)
        self.root_rpy = np.zeros(3)

    def _joint_sliders(self) -> bool:
        joints = self.driver.revolute_joints
        names = joints.geometry().instances().find('name').view()
        touched = False
        for i, name in enumerate(names):
            imgui.Text(name)
            imgui.SameLine()
            changed, angle = imgui.SliderAngle(
                f'joint {i}', self.driver.joints.current(name))
            if changed:
                self.driver.set_joint_goal(name, angle)
                touched = True
        return touched

    def _root_sliders(self) -> bool:
        root = self.driver.root
        cx, x = imgui.SliderFloat('root_x', root.xyz[0], -0.2, 0.2)
        cy, y = imgui.SliderFloat('root_y', root.xyz[1], -0.2, 0.2)
        cz, z = imgui.SliderFloat('root_z', root.xyz[2], -0.2, 0.2)
        crx, rx = imgui.SliderAngle('root_rx', root.rpy[0])
        cry_, ry = imgui.SliderAngle('root_ry', root.rpy[1])
        crz, rz = imgui.SliderAngle('root_rz', root.rpy[2])
        if any((cx, cy, cz, crx, cry_, crz)):
            self.root_xyz = np.array([x, y, z])
            self.root_rpy = np.array([rx, ry, rz])
            self.driver.set_root_goal(self.root_xyz, self.root_rpy)
            return True
        return False

    def _ik_widgets(self) -> bool:
        imgui.Separator()
        imgui.Text('IK')
        imgui.SameLine()
        changed_pos, new_pos = imgui.SliderFloat3(
            'dst_pos', self.ik_position, -1.0, 1.0)
        changed_ori = np.zeros(3)
        new_ori = np.zeros(3)
        for i, label in enumerate(('ori_x', 'ori_y', 'ori_z')):
            changed_ori[i], new_ori[i] = imgui.SliderAngle(
                label, self.ik_orientation[i])

        if self.jog.poll():
            jog_pos = self.jog.pos
            new_pos[2] = jog_pos[1]  # urdf_z <- space_y
            new_pos[1] = jog_pos[0]  # urdf_y <- space_x
            new_pos[0] = jog_pos[2]  # urdf_x <- space_z
            changed_pos = True

        if changed_pos or changed_ori.any():
            self.ik_position = new_pos
            self.ik_orientation = new_ori
            self.driver.ik_to(self.ik_position, self.ik_orientation)
            return True
        return False

    def draw(self, attr: str = builtin.transform) -> bool:
        touched = self._joint_sliders()
        imgui.Separator()
        touched |= self._root_sliders()
        if self.ik is not None:
            touched |= self._ik_widgets()
        imgui.Separator()
        self.driver.step(attr)
        return True

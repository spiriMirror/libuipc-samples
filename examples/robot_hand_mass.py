"""Initialize ABD mass properties for the shared robot-hand URDF."""

import xml.etree.ElementTree as ET

import numpy as np
from trimesh.transformations import euler_matrix
from uipc.constitution import AffineBodyConstitution
from uipc.geometry import affine_body
from uipc.unit import MPa


def apply_hand_abd(controller, urdf_path):
    robot = ET.parse(urdf_path).getroot()
    abd = AffineBodyConstitution()
    stiffness, density = 100 * MPa, 5e3
    links = {}
    for slot in controller.links():
        geo = slot.geometry()
        name = geo.meta().find("urdf/name").view()[0]
        links[name] = slot
        if name not in ("thumb_dip", "thumb_fingertip"):
            abd.apply_to(geo, stiffness, mass_density=density)
            continue

        # The collision surfaces of these links have non-manifold or open edges.
        inertial = robot.find(f"link[@name='{name}']/inertial")
        mass = float(inertial.find("mass").attrib["value"])
        origin = inertial.find("origin").attrib
        center = np.fromstring(origin["xyz"], sep=" ")
        rotation = euler_matrix(*np.fromstring(origin["rpy"], sep=" "))[:3, :3]
        tensor = inertial.find("inertia").attrib
        inertia = np.array([[tensor[k] for k in row] for row in (
            ("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz")
        )], dtype=float)
        matrix = affine_body.from_rigid_body(mass, center, rotation @ inertia @ rotation.T)
        abd.apply_to(geo, stiffness, mass=matrix, volume=mass / density)
    return links

import polyscope as ps

from uipc import Logger
from uipc.geometry import SimplicialComplexIO
from uipc.constitution import AffineBodyConstitution
from uipc.unit import MPa
from uipc.dev.mesh_doctor import MeshDoctor

from asset_dir import AssetDir


Logger.set_level(Logger.Level.Info)

# Setup paths
workspace = AssetDir.output_path(__file__)
trimesh_path = AssetDir.trimesh_path()

# Create MeshDoctor with GUI enabled
doctor = MeshDoctor(workspace, with_gui=True)

# Load the mesh to be checked
io = SimplicialComplexIO()
mesh = io.read(f"{trimesh_path}/bad_abd_mesh.obj")

# Create constitution to be checked
abd = AffineBodyConstitution()

# Check the mesh
is_valid = doctor.check_mesh(abd, mesh)

if is_valid:
    Logger.info("Mesh check passed!")
else:
    Logger.error("Mesh check failed! See visualization for details.")
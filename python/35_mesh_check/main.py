from uipc import Logger
from uipc.geometry import SimplicialComplexIO
from uipc.constitution import AffineBodyConstitution
from uipc.unit import MPa
import polyscope as ps

from asset_dir import AssetDir
from doctor import MeshDoctor

Logger.set_level(Logger.Level.Info)

# Setup paths
workspace = AssetDir.output_path(__file__)
trimesh_path = AssetDir.trimesh_path()

# Create MeshDoctor with GUI enabled
doctor = MeshDoctor(workspace, with_gui=True)

# Load the bad mesh
io = SimplicialComplexIO()
mesh = io.read(rf"D:\Assets\trashbin.obj")

# Create AffineBodyConstitution
abd = AffineBodyConstitution()

# Check the mesh
is_valid = doctor.check_mesh(abd, mesh)

if is_valid:
    Logger.info("Mesh check passed!")
else:
    Logger.error("Mesh check failed! See visualization for details.")
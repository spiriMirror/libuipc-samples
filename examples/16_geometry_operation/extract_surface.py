import numpy as np
import polyscope as ps
from polyscope import imgui

import uipc
from uipc import Logger, Timer, Transform, Quaternion, Vector3, Vector2, view, builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import tetmesh, label_surface, label_triangle_orient, flip_inward_triangles, extract_surface
from uipc.geometry import SimplicialComplexIO
from uipc.constitution import AffineBodyConstitution, NeoHookeanShell, DiscreteShellBending, ElasticModuli
from uipc.gui import SceneGUI 
from uipc.unit import MPa, GPa, kPa 
from uipc import Future
import time

from asset_dir import AssetDir

Timer.enable_all()
Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
this_folder = AssetDir.folder(__file__)
trimesh_path = AssetDir.trimesh_path()
tetmesh_path = AssetDir.tetmesh_path()


scene = Scene()

bunny = scene.objects().create('bunny')
t = Transform.Identity()
t.translate(Vector3.UnitX() + Vector3.UnitZ())
io = SimplicialComplexIO(t)
bunny_mesh = io.read(f'{tetmesh_path}/bunny0.msh')
label_surface(bunny_mesh)
label_triangle_orient(bunny_mesh)
bunny_mesh = flip_inward_triangles(bunny_mesh)
bunny.geometries().create(bunny_mesh)

# create a surface from tetmesh
my_vert_id = bunny_mesh.vertices().create('my_vert_id', 0)
view(my_vert_id)[:] = np.arange(bunny_mesh.vertices().size())
surf = extract_surface(bunny_mesh)
surf_my_vert_id = surf.vertices().find('my_vert_id')
assert surf_my_vert_id is not None


# NOTE:
# extract_surface will try to keep all the attributes on vertices/edges/faces
# The data on the extracted vertices/edges/faces are the same as the original tetmesh
# so does the user-defined attributes

surf_vert_id_view = surf_my_vert_id.view()
Text = 'Surf VID -> Tet VID\n'
for i in range(surf.vertices().size()):
    Text += f'{i} -> {surf_vert_id_view[i]}\n'
print(Text)

# move a bit for visualization
t = Transform.Identity()
t.translate(Vector3.UnitX())
t.apply_to(view(surf.positions()))

bunny.geometries().create(surf)

ps.init()
ps.set_ground_plane_height(-1.0)
sgui = SceneGUI(scene, 'split')
sgui.register()
sgui.set_edge_width(1.0)

def on_update():
    # show all indices
    imgui.Text(Text)
    pass

ps.set_user_callback(on_update)
ps.show()
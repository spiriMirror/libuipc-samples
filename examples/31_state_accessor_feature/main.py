import numpy as np
import polyscope as ps
from polyscope import imgui

import uipc
from uipc import Logger, Timer, Transform, Quaternion, Vector3, Vector2, view, builtin, Matrix4x4
from uipc.core import (Engine, World, Scene, 
    AffineBodyStateAccessorFeature, FiniteElementStateAccessorFeature,
    SanityCheckResult)
from uipc.geometry import tetmesh, label_surface, ground, label_triangle_orient, flip_inward_triangles
from uipc.geometry import SimplicialComplexIO
from uipc.constitution import AffineBodyConstitution, ElasticModuli, StableNeoHookean
from uipc.gui import SceneGUI 
from uipc.unit import MPa, GPa, kPa

from asset_dir import AssetDir

Timer.enable_all()
Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
this_folder = AssetDir.folder(__file__)
trimesh_path = AssetDir.trimesh_path()
tetmesh_path = AssetDir.tetmesh_path()

engine = Engine('cuda', workspace)
world = World(engine)

config = Scene.default_config()
print(config)
scene = Scene(config)

obj = scene.objects().create('cubes')
sio = SimplicialComplexIO()
base_mesh = sio.read(f'{tetmesh_path}/cube.msh')

label_surface(base_mesh)
label_triangle_orient(base_mesh)
base_mesh = flip_inward_triangles(base_mesh)

fem_mesh = base_mesh.copy()
abd_mesh = base_mesh.copy()

snk = StableNeoHookean()
parm = ElasticModuli.youngs_poisson(20.0 * kPa, 0.49)
snk.apply_to(fem_mesh, parm)

abd = AffineBodyConstitution()
abd.apply_to(abd_mesh, kappa=1.0 * MPa)
trans_view = view(abd_mesh.transforms())
t = Transform.Identity()
t.translate(Vector3.Values([0.0, 1.2, 0.0]))
trans_view[:] = t.matrix()

obj.geometries().create(fem_mesh)
obj.geometries().create(abd_mesh)

g = scene.objects().create('ground')
g_mesh = ground(-0.6)
g.geometries().create(g_mesh)

world.init(scene)
world.dump()

# 1) Get feature after world initialization
abd_state_accessor:AffineBodyStateAccessorFeature = world.features().find(AffineBodyStateAccessorFeature)
fem_state_accessor:FiniteElementStateAccessorFeature = world.features().find(FiniteElementStateAccessorFeature)

assert abd_state_accessor is not None and fem_state_accessor is not None, 'This version of uipc does not support state accessor feature.'

# 2) Create a state_geo to contain data
abd_state_geo = abd_state_accessor.create_geometry()
abd_transforms =abd_state_geo.instances().create(builtin.transform, Matrix4x4.Zero()) # tell the backend we need transform information
abd_velocities =abd_state_geo.instances().create(builtin.velocity, Matrix4x4.Zero()) # tell the backend we need velocity information
fem_state_geo = fem_state_accessor.create_geometry()
fem_positions = fem_state_geo.vertices().create(builtin.position, Vector3.Zero()) # tell the backend we need position information
fem_velocities = fem_state_geo.vertices().create(builtin.velocity, Vector3.Zero()) # tell the backend we need velocity information

ps.init()
sgui = SceneGUI(scene, 'split')
sgui.register()
sgui.set_edge_width(1.0)

run = False

class StateAccessorInspector:
    def __init__(
        self,
        abd_accessor: AffineBodyStateAccessorFeature,
        fem_accessor: FiniteElementStateAccessorFeature,
        abd_geo,
        fem_geo,
        abd_transforms_attr,
        abd_velocities_attr,
        fem_positions_attr,
        fem_velocities_attr,
    ):
        self.abd_accessor = abd_accessor
        self.fem_accessor = fem_accessor
        self.abd_geo = abd_geo
        self.fem_geo = fem_geo
        self.abd_transforms_attr = abd_transforms_attr
        self.abd_velocities_attr = abd_velocities_attr
        self.fem_positions_attr = fem_positions_attr
        self.fem_velocities_attr = fem_velocities_attr

    def _draw_vector3(self, label: str, vec):
        values = [
            float(np.asarray(vec[0]).item()),
            float(np.asarray(vec[1]).item()),
            float(np.asarray(vec[2]).item()),
        ]
        changed, new_vals = imgui.InputFloat3(label, values)
        if changed:
            vec[0], vec[1], vec[2] = new_vals
        return changed

    def _draw_matrix4x4(self, label: str, mat):
        changed = False
        imgui.Text(label)
        for r in range(4):
            row_vals = [
                float(np.asarray(mat[r, 0]).item()),
                float(np.asarray(mat[r, 1]).item()),
                float(np.asarray(mat[r, 2]).item()),
                float(np.asarray(mat[r, 3]).item()),
            ]
            changed_r, new_vals = imgui.InputFloat4(
                f"{label} row {r}",
                row_vals,
            )
            if changed_r:
                mat[r, 0], mat[r, 1], mat[r, 2], mat[r, 3] = new_vals
                changed = True
        return changed

    def draw(self):
        self.abd_accessor.copy_to(self.abd_geo)
        self.fem_accessor.copy_to(self.fem_geo)

        abd_transforms = view(self.abd_transforms_attr)
        abd_velocities = view(self.abd_velocities_attr)
        fem_positions = view(self.fem_positions_attr)
        fem_velocities = view(self.fem_velocities_attr)

        changed_abd = False
        changed_fem = False

        if imgui.CollapsingHeader('State Accessor Data'):
            imgui.Text(f'ABD instances: {abd_transforms.shape[0]}')
            if imgui.TreeNode('ABD transforms & velocities'):
                for i in range(abd_transforms.shape[0]):
                    if imgui.TreeNode(f'ABD instance {i}'):
                        changed_abd |= self._draw_matrix4x4('transform', abd_transforms[i])
                        changed_abd |= self._draw_matrix4x4('velocity', abd_velocities[i])
                        imgui.TreePop()
                imgui.TreePop()

            imgui.Separator()
            imgui.Text(f'FEM vertices: {fem_positions.shape[0]}')
            if imgui.TreeNode('FEM positions & velocities'):
                for i in range(fem_positions.shape[0]):
                    if imgui.TreeNode(f'FEM vertex {i}'):
                        changed_fem |= self._draw_vector3('position', fem_positions[i])
                        changed_fem |= self._draw_vector3('velocity', fem_velocities[i])
                        imgui.TreePop()
                imgui.TreePop()

        if changed_abd:
            self.abd_accessor.copy_from(self.abd_geo)
        if changed_fem:
            self.fem_accessor.copy_from(self.fem_geo)

        return changed_abd or changed_fem

state_inspector = StateAccessorInspector(
    abd_state_accessor,
    fem_state_accessor,
    abd_state_geo,
    fem_state_geo,
    abd_transforms,
    abd_velocities,
    fem_positions,
    fem_velocities,
)

def on_update():
    global run
    
    if(imgui.Button('run & stop')):
        run = not run
    
    moved = state_inspector.draw()

    if(moved):
        # Update scene data from backend to reflect the changes
        world.retrieve()
        sgui.update()
        # If penetration happens after manual modification, we need to do a recovery step
        # to the last valid state
        if(world.sanity_checker().check() != SanityCheckResult.Success):
            # We only dumped 0 frame, just recover it
            world.recover(0)
    
    # common simulation step
    if(run):
        world.advance()
        world.retrieve()
        sgui.update()

ps.set_user_callback(on_update)
ps.show()
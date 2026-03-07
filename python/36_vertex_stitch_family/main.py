import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, Animation
from uipc import Vector3
import uipc.builtin as builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import (
    GeometrySlot, SimplicialComplex,
    label_surface, pointcloud, linemesh, trimesh,
)
from uipc.constitution import (
    Particle, Empty, HookeanSpring, KirchhoffRodBending,
    SoftPositionConstraint,
    SoftVertexStitch, SoftVertexEdgeStitch, SoftVertexTriangleStitch,
    ElasticModuli, ElasticModuli2D,
)
from uipc.gui import SceneGUI
from uipc.unit import MPa
from asset_dir import AssetDir

Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
engine = Engine('cuda', workspace)
world = World(engine)

config = Scene.default_config()
dt = 0.01
config['dt'] = dt
config['gravity'] = [[0.0], [-9.8], [0.0]]
config['contact']['enable'] = False
scene = Scene(config)

# Constitutions
particle_const = Particle()
empty = Empty()
hs = HookeanSpring()
krb = KirchhoffRodBending()
spc = SoftPositionConstraint()

# Stitch constitutions
svs = SoftVertexStitch()
sves = SoftVertexEdgeStitch()
svts = SoftVertexTriangleStitch()

THICKNESS = 0.01
MIN_SEPARATE_DISTANCE = 0.02

def create_chain(name, start_x, start_z, n_verts=6, spacing=0.2):
    Vs = np.array([[start_x, 1.0 - i * spacing, start_z] for i in range(n_verts)])
    Es = np.array([[i, i + 1] for i in range(n_verts - 1)])
    chain = linemesh(Vs, Es)
    label_surface(chain)
    hs.apply_to(chain, 10.0 * MPa)
    krb.apply_to(chain, 1.0 * MPa)
    obj = scene.objects().create(name)
    geo_slot, rest_geo_slot = obj.geometries().create(chain)
    return obj, geo_slot, rest_geo_slot


# ===========================  1. VV Stitch  ================================
vv_particle_pos = np.array([[0.0, 1.0, 0.0]])
vv_particle_mesh = pointcloud(vv_particle_pos)
label_surface(vv_particle_mesh)
particle_const.apply_to(vv_particle_mesh)
spc.apply_to(vv_particle_mesh, 10000.0)

vv_anim_obj = scene.objects().create('vv_animated_particle')
vv_anim_slot, vv_anim_rest_slot = vv_anim_obj.geometries().create(vv_particle_mesh)

vv_chain_obj, vv_chain_slot, vv_chain_rest_slot = create_chain('vv_chain', 0.0, 0.0)

vv_pairs = np.array([[0, 0]], dtype=np.int32)
vv_stitch_geo = svs.create_geometry(
    (vv_anim_slot, vv_chain_slot), vv_pairs, kappa=1e6
)
vv_stitch_obj = scene.objects().create('vv_stitch')
vv_stitch_obj.geometries().create(vv_stitch_geo)


# ===========================  2. VE Stitch  ================================
VE_X = 1.5
ve_edge_pos = np.array([[VE_X, 1.0, -0.3], [VE_X, 1.0, 0.3]])
ve_edge_seg = np.array([[0, 1]])
ve_edge_mesh = linemesh(ve_edge_pos, ve_edge_seg)
label_surface(ve_edge_mesh)
empty.apply_to(ve_edge_mesh, thickness=THICKNESS)
spc.apply_to(ve_edge_mesh, 10000.0)

ve_anim_obj = scene.objects().create('ve_animated_edge')
ve_anim_slot, ve_anim_rest_slot = ve_anim_obj.geometries().create(ve_edge_mesh)

ve_chain_obj, ve_chain_slot, ve_chain_rest_slot = create_chain('ve_chain', VE_X, 0.0)

ve_pairs = np.array([[0, 0]], dtype=np.int32)
ve_stitch_geo = sves.create_geometry(
    (ve_chain_slot, ve_anim_slot),
    (ve_chain_rest_slot, ve_anim_rest_slot),
    ve_pairs,
    ElasticModuli2D.youngs_poisson(1.0 * MPa, 0.49),
    thickness=THICKNESS,
    min_separate_distance=MIN_SEPARATE_DISTANCE,
)
ve_stitch_obj = scene.objects().create('ve_stitch')
ve_stitch_obj.geometries().create(ve_stitch_geo)


# ===========================  3. VT Stitch  ================================
VT_X = 3.0
vt_tri_pos = np.array([[VT_X, 1.0, 0.0], [VT_X + 0.5, 1.0, 0.0], [VT_X, 1.0, 0.5]])
vt_tri_faces = np.array([[0, 1, 2]])
vt_tri_mesh = trimesh(vt_tri_pos, vt_tri_faces)
label_surface(vt_tri_mesh)
empty.apply_to(vt_tri_mesh, thickness=THICKNESS)
spc.apply_to(vt_tri_mesh, 10000.0)

vt_anim_obj = scene.objects().create('vt_animated_triangle')
vt_anim_slot, vt_anim_rest_slot = vt_anim_obj.geometries().create(vt_tri_mesh)

vt_chain_obj, vt_chain_slot, vt_chain_rest_slot = create_chain(
    'vt_chain', VT_X + 0.17, 0.17
)

vt_pairs = np.array([[0, 0]], dtype=np.int32)
vt_stitch_geo = svts.create_geometry(
    (vt_chain_slot, vt_anim_slot),
    (vt_chain_rest_slot, vt_anim_rest_slot),
    vt_pairs,
    ElasticModuli.youngs_poisson(120e3, 0.49),
    min_separate_distance=MIN_SEPARATE_DISTANCE,
)
vt_stitch_obj = scene.objects().create('vt_stitch')
vt_stitch_obj.geometries().create(vt_stitch_geo)


# ==========================  Animation  ====================================
animator = scene.animator()

AMPLITUDE = 0.5
PERIOD = 2.0

def animate(info: Animation.UpdateInfo):
    geo: SimplicialComplex = info.geo_slots()[0].geometry()
    rest_geo: SimplicialComplex = info.rest_geo_slots()[0].geometry()

    is_constrained = geo.vertices().find(builtin.is_constrained)
    aim_position = geo.vertices().find(builtin.aim_position)
    rest_pos = rest_geo.positions().view()

    t = info.dt() * info.frame()
    z_offset = AMPLITUDE * np.sin(2.0 * np.pi * t / PERIOD)

    ic_view = view(is_constrained)
    ap_view = view(aim_position)
    for i in range(len(rest_pos)):
        ic_view[i] = 1
        ap_view[i] = rest_pos[i] + Vector3.UnitZ() * z_offset


animator.insert(vv_anim_obj, animate)
animator.insert(ve_anim_obj, animate)
animator.insert(vt_anim_obj, animate)


# ==========================  Simulation  ===================================
world.init(scene)
sgui = SceneGUI(scene, 'split')

ps.init()
sgui.register()
sgui.set_edge_width(1)

run = False


def on_update():
    global run
    if imgui.Button('run & stop'):
        run = not run

    if run:
        world.advance()
        world.retrieve()
        sgui.update()


ps.set_user_callback(on_update)
ps.show()

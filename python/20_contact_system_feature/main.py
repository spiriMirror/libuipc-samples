import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, Timer, Animation
from uipc import Vector3, Transform, Quaternion, AngleAxis
import uipc.builtin as builtin
from uipc.core import Engine, World, Scene, ContactSystemFeature
from uipc.geometry import (GeometrySlot, SimplicialComplex, SimplicialComplexIO, Geometry,
                            label_surface, label_triangle_orient, flip_inward_triangles, ground)
from uipc.constitution import AffineBodyConstitution, StableNeoHookean
from uipc.gui import SceneGUI
from uipc.unit import MPa, GPa
from asset_dir import AssetDir

Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
engine = Engine('cuda', workspace)
world = World(engine)

config = Scene.default_config()
dt = 0.02
config['dt'] = dt
config['gravity'] = [[0.0], [-9.8], [0.0]]
scene = Scene(config)

# friction ratio and contact resistance
scene.contact_tabular().default_model(0.5, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()

# create constitution and contact model
abd = AffineBodyConstitution()
snk = StableNeoHookean()

# load cube mesh
io = SimplicialComplexIO()
cube_mesh = io.read(f'{AssetDir.tetmesh_path()}/cube.msh')
# label the surface, enable the contact
label_surface(cube_mesh)
# label the triangle orientation to export the correct surface mesh
label_triangle_orient(cube_mesh)
cube_mesh = flip_inward_triangles(cube_mesh)

geo_slot_list:list[GeometrySlot] = []

# ABD
abd_cube_obj = scene.objects().create('abd')
abd_mesh = cube_mesh.copy()
abd.apply_to(abd_mesh, 10.0 * MPa)
t = Transform.Identity()
t.translate(Vector3.UnitY() * 1.1)
view(abd_mesh.transforms())[:] = t.matrix()
abd_geo_slot, abd_rest_geo_slot = abd_cube_obj.geometries().create(abd_mesh)
geo_slot_list.append(abd_geo_slot)

# FEM
fem_cube_obj = scene.objects().create('fem')
fem_mesh = cube_mesh.copy()
snk.apply_to(fem_mesh, mass_density=1e3)
fem_geo_slot, fem_rest_geo_slot = fem_cube_obj.geometries().create(fem_mesh)
geo_slot_list.append(fem_geo_slot)

ground_height = -1.5
ground_obj = scene.objects().create('ground')
g = ground(ground_height)
g_geo_slot, g_rest_geo_slot = ground_obj.geometries().create(g)
geo_slot_list.append(g_geo_slot)

world.init(scene)
csf:ContactSystemFeature = world.features().find(ContactSystemFeature)

sgui = SceneGUI(scene, 'split')

ps.init()
sgui.register()
sgui.set_edge_width(1)

class ContactInfo:
    def __init__(self, name, csf:ContactSystemFeature):
        self.name = name
        self.csf :ContactSystemFeature= csf
        # Normal Contact
        self.NE = Geometry()  # energy
        self.NG = Geometry()  # gradient
        self.NH = Geometry()  # hessian
        # Frictional Contact
        self.FE = Geometry()  # energy
        self.FG = Geometry()  # gradient
        self.FH = Geometry()  # hessian
    
    def retrieve(self):
        # Normal Contact
        self.csf.contact_energy(self.name + '+N', self.NE)
        self.csf.contact_gradient(self.name + '+N', self.NG)
        self.csf.contact_hessian(self.name + '+N', self.NH)
        # Frictional Contact
        self.csf.contact_energy(self.name + '+F', self.FE)
        self.csf.contact_gradient(self.name + '+F', self.FG)
        self.csf.contact_hessian(self.name + '+F', self.FH)
    
    def display_energy(self, t:str, geo:Geometry):
        topo = geo.instances().find('topo')
        if topo is None:
            return
        topo_view = topo.view()
        imgui.Text(f'[{self.name}+{t}] Contact Topo: {topo_view.reshape(-1, topo_view.shape[1])}')
        energy = geo.instances().find('energy')
        imgui.Text(f'[{self.name}+{t}] Contact Energy: {energy.view()}')
    
    def display_gradient(self, t:str, geo:Geometry, show_detail:bool):
        i = geo.instances().find('i')
        if i is None:
            return
        if not show_detail:
            imgui.Text(f'[{self.name}+{t}] Contact Gradient Count: {geo.instances().size()}, skipping detail display.')
            return
        imgui.Text(f'[{self.name}+{t}] Contact Gradient I: {i.view()}')
        grad = geo.instances().find('grad')
        imgui.Text(f'[{self.name}+{t}] Contact Gradient: {grad.view()}')
    
    def display_hessian(self, t:str, geo:Geometry, show_detail:bool):
        i = geo.instances().find('i')
        if i is None:
            return
        if not show_detail:
            imgui.Text(f'[{self.name}+{t}] Contact Hessian Count: {geo.instances().size()}, skipping detail display.')
            return
        imgui.Text(f'[{self.name}+{t}] Contact Hessian I: {i.view()}')
        j = geo.instances().find('j')
        if j is None:
            return
        imgui.Text(f'[{self.name}+{t}] Contact Hessian J: {j.view()}')
        hess = geo.instances().find('hess')
        imgui.Text(f'[{self.name}+{t}] Contact Hessian: {hess.view()}')
    
    def display(self):
        '''
        Display the contact information for this contact primitive.
        Gradient and Hessian are not displayed by default (Too many data).
        '''
        self.display_energy('N', self.NE)
        self.display_gradient('N', self.NG, show_detail=False)
        self.display_hessian('N', self.NH, show_detail=False)
        self.display_energy('F', self.FE)
        self.display_gradient('F', self.FG, show_detail=False)
        self.display_hessian('F', self.FH, show_detail=False)
        imgui.Separator()

# contact primitives geometry
# point-halfplane
PH = ContactInfo('PH', csf)
# point-point
PP = ContactInfo('PP', csf)
# point-edge
PE = ContactInfo('PE', csf)
# point-triangle
PT = ContactInfo('PT', csf)
# edge-edge
EE = ContactInfo('EE', csf)

run = False
def on_update():
    global run
    global geo_slot_list
    global PH

    if(imgui.Button('run & stop')):
        run = not run
    
    for geo_slot in geo_slot_list:
        geo = geo_slot.geometry()
        gvo = geo.meta().find(builtin.global_vertex_offset)
        if gvo is not None:
            imgui.Text(f'[{geo_slot.id()}] Global Vertex Offset: {gvo.view()}')
        else:
            imgui.Text(f'[{geo_slot.id()}] This version dont support global vertex offset!')

    if(run):
        world.advance()
        world.retrieve()
        sgui.update()
    
    # contact primitives
    types = csf.contact_primitive_types()
    imgui.Text(f'Contact Primitive Types: {types}')
    imgui.Separator()
    
    if(run):
        world.advance()
        world.retrieve()
        sgui.update()
        
        PH.retrieve()
        PP.retrieve()
        PE.retrieve()
        PT.retrieve()
        EE.retrieve()
    
    PH.display()
    PP.display()
    PE.display()
    PT.display()
    EE.display()

ps.set_user_callback(on_update)
ps.show()

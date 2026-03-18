import json

import numpy as np
import polyscope as ps
import trimesh as libtrimesh
from asset_dir import AssetDir
from polyscope import imgui
from uipc import Logger, Timer, Vector3, builtin, view
from uipc.constitution import (
    DiscreteShellBending,
    ElasticModuli2D,
    Empty,
    NeoHookeanShell,
    SoftPositionConstraint,
    SoftVertexStitch,
)
from uipc.core import Engine, Scene, SceneIO, World
from uipc.geometry import (
    SimplicialComplexIO,
    label_surface,
)
from uipc.gui import SceneGUI


Timer.disable_all()
Logger.set_level(Logger.Warn)
output_dir = AssetDir.output_path(__file__)
engine = Engine("cuda", output_dir)
world = World(engine)

curr_folder = AssetDir.folder(__file__)

config = Scene.default_config()
config["dt"] = 1.0 / 60
config["contact"]["d_hat"] = 0.002
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["newton"]["velocity_tol"] = 0.05
config["newton"]["max_iter"] = 1024
config["extras"]["debug"]["dump_surface"] = False
config["linear_system"]["tol_rate"] = 1e-3
print(config)
scene = Scene(config)

empty = Empty()
snh = NeoHookeanShell()
dsb = DiscreteShellBending()
spc = SoftPositionConstraint()
io = SimplicialComplexIO()
svs = SoftVertexStitch()

default_elem = scene.contact_tabular().default_element()

scene.contact_tabular().default_model(0.01, 1e9)
t_shirt_front_elem = scene.contact_tabular().create("t_shirt_front")
t_shirt_back_elem = scene.contact_tabular().create("t_shirt_back")

moduli = ElasticModuli2D.youngs_poisson(1e5, 0.49)
t_shirt_obj = scene.objects().create("t_shirt")
t_shirt_front = io.read(str(curr_folder / "output_panel_top_front.obj"))
t_shirt_back = io.read(str(curr_folder / "output_panel_top_back.obj"))
label_surface(t_shirt_front)
label_surface(t_shirt_back)
snh.apply_to(t_shirt_front, moduli=moduli, thickness=0.0002, mass_density=100.0)
snh.apply_to(t_shirt_back, moduli=moduli, thickness=0.0002, mass_density=100.0)

dsb.apply_to(t_shirt_front, bending_stiffness=10)
dsb.apply_to(t_shirt_back, bending_stiffness=10)
t_shirt_front_elem.apply_to(t_shirt_front)
t_shirt_back_elem.apply_to(t_shirt_back)

PANEL_FILES = {
    "top_front": curr_folder / "output_panel_top_front.obj",
    "top_back": curr_folder / "output_panel_top_back.obj",
}
JSON_FILE_PATH = curr_folder / "output_stitch_data_local_indices.json"

all_points1 = []
all_points2 = []
indices1 = []
indices2 = []

try:
    print("--- Loading mesh files ---")
    loaded_meshes = {}
    for panel_name, file_path in PANEL_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Error: OBJ file '{file_path}' not found.")

        print(f"Loading: {file_path}")
        mesh = libtrimesh.load_mesh(file_path)
        if isinstance(mesh, libtrimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        loaded_meshes[panel_name] = mesh

    if not JSON_FILE_PATH.exists():
        raise FileNotFoundError(f"Error: JSON file '{JSON_FILE_PATH}' not found.")
    print(f"\n--- Loading stitch data from: {JSON_FILE_PATH} ---")
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        all_stitch_data = json.load(f)

    print(
        "\n--- Searching for all stitch connections between 'top_front' and 'top_back' ---"
    )
    total_connections_found = 0
    p1_print_name, p2_print_name = "", ""

    for stitch_info in all_stitch_data:
        p1_name = stitch_info.get("panel_1")
        p2_name = stitch_info.get("panel_2")

        # Check if this entry connects the two panels we loaded (in any order)
        if {p1_name, p2_name} == set(PANEL_FILES.keys()):
            total_connections_found += 1
            p1_print_name, p2_print_name = p1_name, p2_name
            print(
                f"  -> Connection found ({total_connections_found}): '{p1_name}' (edge {stitch_info['edge_1_index']}) <--> '{p2_name}' (edge {stitch_info['edge_2_index']})"
            )

            mesh1_vertices = loaded_meshes[p1_name].vertices
            mesh2_vertices = loaded_meshes[p2_name].vertices

            for stitch_pair in stitch_info.get("stitch_pairs_by_index", []):
                idx1 = stitch_pair["vertex_index_panel_1"]
                idx2 = stitch_pair["vertex_index_panel_2"]

                indices1.append(idx1)
                indices2.append(idx2)
                all_points1.append(mesh1_vertices[idx1])
                all_points2.append(mesh2_vertices[idx2])

    if total_connections_found == 0:
        print(
            f"\nError: No stitch connection found between 'top_front' and 'top_back' in '{JSON_FILE_PATH}'."
        )

    if total_connections_found > 0:
        points1_np = np.array(all_points1)
        points2_np = np.array(all_points2)

        print("\nData processing complete!")
        print(
            f"Found a total of {len(points1_np)} stitch point pairs across {total_connections_found} connections."
        )

        print(
            f"\nShape of the first array (from {p1_print_name}): {points1_np.shape}"
        )
        print(
            f"Shape of the second array (from {p2_print_name}): {points2_np.shape}"
        )

except (FileNotFoundError, Exception) as e:
    print(f"\nA fatal error occurred during file processing: {e}")

assert len(indices1) == len(indices2), "Index lengths do not match!"

rest_t_shirt_front = t_shirt_front.copy()
rest_t_shirt_back = t_shirt_back.copy()

stitch_Vs = np.array([[i, j] for i, j in zip(indices1, indices2)], dtype=np.int32)
print(stitch_Vs)

# ----------------------------------------------------------------------------
# Disable stitch contact
stitch_front = scene.contact_tabular().create("stitch_front")
stitch_back = scene.contact_tabular().create("stitch_back")
scene.contact_tabular().insert(stitch_front, stitch_back, 0, 1e9, True)
# Add stitch constraints
stitch_obj = scene.objects().create("stitch")
svs = SoftVertexStitch()
front_geo_slot, _ = t_shirt_obj.geometries().create(
    t_shirt_front, rest_t_shirt_front
)
back_geo_slot, _ = t_shirt_obj.geometries().create(t_shirt_back, rest_t_shirt_back)
stitch_geo = svs.create_geometry(
    # geometry pair to stitched
    (front_geo_slot, back_geo_slot), 
    # vertex pairs to stitch
    stitch_Vs, 
    # contact elements for stitching vertex pairs
    (stitch_front, stitch_back), 
    1000.0
)
stitch_obj.geometries().create(stitch_geo)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# make body no contact with itself
body_elem = scene.contact_tabular().create("body")
scene.contact_tabular().insert(body_elem, body_elem, 0, 0, False)
scene.contact_tabular().insert(default_elem, body_elem, 0, 0, False)

io = SimplicialComplexIO()
body = io.read(str(curr_folder / "body.obj"))
label_surface(body)
empty.apply_to(body, thickness=0.0)
spc.apply_to(body, 1000)
body_elem.apply_to(body)
is_constrained = body.vertices().find(builtin.is_constrained)
view(is_constrained)[:] = 1
is_dynamic = body.vertices().find(builtin.is_dynamic)
view(is_dynamic)[:] = 0
body_gravity = body.vertices().create(builtin.gravity, Vector3.Zero())
body_obj = scene.objects().create("body")
slot, rest_slot = body_obj.geometries().create(body)
# -----------------------------------------------------------------------------

world.init(scene)
sio = SceneIO(scene)

ps.init()
sgui = SceneGUI(scene, 'split')
sgui.register()
sgui.set_edge_width(1.0)

def write_to_disk(frame):
    io = SimplicialComplexIO()
    io.write(
        f"{output_dir}/cloth_front_surface{frame}.obj",
        front_geo_slot.geometry(),
    )
    io.write(
        f"{output_dir}/cloth_back_surface{frame}.obj", 
        back_geo_slot.geometry()
    )
    sio.write_surface(f"{output_dir}/scene_surface{frame}.obj")

run = False
save_frames = False
def on_update():
    global run, save_frames
    if(imgui.Button('run & stop')):
        run = not run
    
    imgui.SameLine()
    if(imgui.Button('save frame')):
        write_to_disk(world.frame())
        print(f"Frame {world.frame()} saved.")
    
    imgui.SameLine()
    changed , value = imgui.Checkbox('auto save', save_frames)
    if changed:
        save_frames = value
        
    if(run):
        world.advance()
        world.retrieve()
        sgui.update()
        if save_frames:
            write_to_disk(world.frame())
            print(f"Frame {world.frame()} done.")

ps.set_user_callback(on_update)
ps.show()

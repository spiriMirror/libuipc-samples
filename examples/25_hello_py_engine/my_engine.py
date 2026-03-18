from uipc import Engine, World, Scene
from uipc.core import PyIEngine
from uipc.backend import WorldVisitor, SceneVisitor
from uipc.geometry import SimplicialComplex
from uipc import view
import warp as wp

@wp.kernel
def increment_positions(positions:wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    positions[i] += wp.vec3d(wp.float64(1.0), wp.float64(1.0), wp.float64(1.0))  # Increment each position by (0, 0.1, 0)

class MyEngine(PyIEngine):
    def __init__(self):
        wp.init()
        super().__init__()
        self.frame = 0

    def do_init(self):
        wv = WorldVisitor(self.world())
        sv = wv.scene()
        sv.info()
        print(sv.contact_tabular().contact_models())
        print(f"Initializing the world -> {sv.info()}")
    
    def do_advance(self):
        self.frame += 1
        # a fake simulation
        if self.frame == 1:
            wv = WorldVisitor(self.world())
            sv = wv.scene()
            geo_slots = sv.geometries()
            
            for geo_slot in geo_slots:
                if(geo_slot.geometry().type() == "SimplicialComplex"):
                    sc:SimplicialComplex = geo_slot.geometry()
                    pos = view(sc.positions())
                    pos = pos.reshape((-1, 3))
                    
                    # create wp array for pos
                    wp_positions = wp.from_numpy(pos)
                    # call the kernel to increment positions
                    wp.launch(increment_positions, dim=wp_positions.shape[0], inputs=[wp_positions])
                    # copy back the modified positions
                    pos[:] = wp_positions.numpy()
                    
                    print(f"Advancing geometry {geo_slot.id()} to position {view(sc.positions())[:]}")
    
    def do_retrieve(self):
        pass
    
    def do_sync(self):
        wp.synchronize()
    
    def get_frame(self):
        return self.frame
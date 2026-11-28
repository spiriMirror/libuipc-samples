# 52 — Robot hand grasps and places an apple

This example turns the manually posed hand from sample 45 into a reproducible,
four-finger pick-and-place simulation. It loads the same URDF collision meshes
and drives every link with `SoftTransformConstraint`; the apple is a free ABD
body with no animation, fixed flag or attachment. Gravity, IPC normal contact
and Coulomb friction are responsible for holding, carrying and final support.

The 500 output frames run at 30 FPS with four solver steps per output frame
(`dt = 1/120 s`). The sequence holds for 50 frames, approaches and closes all
four fingers, lifts the apple, transfers it 0.45 m, lowers it onto the table,
opens and retreats. Smoothstep interpolation gives each scripted stage zero
endpoint velocity. It changes physical servo targets, not solved trajectories.

```bash
cd examples/52_robot_hand_grasp_apple
python main.py
python main.py --headless 500
```

The GUI advances four solver steps per displayed output frame. The headless mode
writes `validation.json` and `trajectory.npz` under
`output/examples/52_robot_hand_grasp_apple/`. Add `--write-surfaces` to export
OBJ surfaces at representative action frames. Short headless runs use a frame
count suffix such as `validation_0050.json`, so they never replace the complete
500-frame evidence.

The checked full run requires the apple to be lifted by more than 0.15 m,
transported by more than 0.35 m, supported at the destination, settled after
release, and within 1.5 mm of every fingertip at output frames 250 and 325.
Distances use a conservative bidirectional vertex-to-triangle surface query.
The JSON records the actual distances and trajectory observables, including
failure details before a failed full validation raises.

## Coordinate and solver notes

`UrdfIO` uses libuipc/Polyscope's Y-up convention. `BASIS` in `main.py` maps
the Blender grasp plan from Z-up coordinates. With the thumb links' URDF
inertias, the closed target for joint 6 is increased by 0.12 rad to keep
the second fingertip in contact during transport.

The robot follows sample 45's quasi-static ABD design: rigidity 100 MPa,
density 5000 kg/m³ and translation/rotation strength ratios 10000. The
collision meshes for `thumb_dip` and `thumb_fingertip` are assemblies of
closed components that overlap. Their URDF mass, center of mass and inertia
define their dynamics, with an effective volume of mass divided by density. Internal hand
contact is disabled, while hand/apple/table contact remains enabled with hand
friction 0.8. The apple uses rigidity 100 MPa and density 820 kg/m³. Its closed
tetrahedral mesh is generated from a symmetric icosphere surface and
center-to-face cells; this keeps its volume centroid aligned with the
geometric/grasp center. The existing `ball.msh` is deliberately not reused
because its volume centroid is about 5.6 mm off its bounding-box center after
normalization, causing a free apple to roll before the hand arrives.

This strongly driven/light-object system selects standard IPC with contact
distance `1e-3 m`, Newton velocity tolerance `1e-2 m/s`, PCG tolerance `1e-6`,
and 32 line-search trials. Semi-implicit termination retains the library default.
These settings do not add damping or constrain the apple. The root library
defaults are unchanged.

With the repaired collision meshes and URDF thumb inertias, two Windows/RTX
5090 runs each completed all 1,996 solver steps, lifting the apple 0.221 m
and transporting it 0.451 m. The largest fingertip-to-apple distances were
under 0.018 mm at frame 250 and 0.033 mm at frame 325 in both runs.

The scene uses sample 45's sanity-check settings and enables strict solver
failure reporting. This is a scripted position-servo example,
not a calibrated torque-controlled robot or actuator-force validation.

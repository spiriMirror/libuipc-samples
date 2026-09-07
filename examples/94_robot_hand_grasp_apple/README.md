# 94 — Robot hand grasps and places an apple

This example turns the manually posed hand from sample 87 into a reproducible,
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
cd examples/94_robot_hand_grasp_apple
python main.py
python main.py --headless 500
```

The GUI advances four solver steps per displayed output frame. The headless mode
writes `validation.json` and `trajectory.npz` under
`output/examples/94_robot_hand_grasp_apple/`. Add `--write-surfaces` to export
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

`UrdfIO` uses libuipc/Polyscope's Y-up convention. `grasp_plan.json` records the
four-pad joint solution from the verified Blender integration, whose local
coordinates are Z-up; `BASIS` in `main.py` performs the explicit vector mapping.
Joint angles themselves are unchanged.

The robot follows sample 87's quasi-static ABD design: rigidity 100 MPa,
density 5000 kg/m³ and translation/rotation strength ratios 10000. Internal hand
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

The validated Windows/RTX 5090 full run completed all 1,996 solver steps. It
lifted the apple 0.2180 m, transported it 0.4495 m, and ended with RMS center
speed about `5.83e-11 m/s`. The four fingertip distance ranges were
0.95–55.6 micrometers at frame 250 and 0.97–40.0 micrometers at frame 325.

The sample-87 URDF assets contain known isolated zero-volume/open collision
fragments, so scene sanity checking is disabled as in sample 87; strict solver
failure reporting remains enabled. This is a scripted position-servo example,
not a calibrated torque-controlled robot or actuator-force validation.

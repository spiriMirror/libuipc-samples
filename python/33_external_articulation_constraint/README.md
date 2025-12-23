# External Articulation Constraint Example

This example demonstrates how to use `ExternalArticulationConstraint` to control the motion of articulated bodies by prescribing joint DOFs (degrees of freedom).

## Overview

The example creates a simple articulated system with:
- 3 cube instances connected by joints
- A revolute joint between instance 0 and instance 1
- A prismatic joint between instance 1 and instance 2
- External articulation constraint to control joint motion

## Features

- **Interactive GUI**: Use Polyscope/ImGui sliders to adjust `delta_theta_tilde` values in real-time
- **Revolute Joint Control**: Adjust angular velocity of the revolute joint
- **Prismatic Joint Control**: Adjust linear velocity of the prismatic joint
- **Real-time Simulation**: See the effects of your adjustments immediately

## Usage

Run the example:
```bash
python main.py
```

### GUI Controls

- **Run & Stop**: Toggle simulation on/off
- **Revolute Joint (rad/s)**: Slider to control angular velocity (-π to π rad/s)
- **Prismatic Joint (m/s)**: Slider to control linear velocity (-1.0 to 1.0 m/s)

The `delta_theta_tilde` values are updated each frame based on the slider values multiplied by the time step.

## Technical Details

- The example uses `ref_dof_prev` attribute to synchronize motion between external systems and IPC
- Mass matrix is set to a 2x2 matrix for the two joints
- External kinetic is enabled for all affine body instances
- Instance 0 is fixed, instances 1 and 2 are free to move


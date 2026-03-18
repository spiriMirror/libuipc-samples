# External Articulation Constraint Example

This example demonstrates how to use `ExternalArticulationConstraint` to control articulated bodies by prescribing joint DOFs (degrees of freedom) in real-time.

## Overview

The example creates an articulated system with:
- 3 cube instances connected by joints
- A revolute joint between instance 0 and instance 1
- A prismatic joint between instance 1 and instance 2
- External articulation constraint to control joint motion via `delta_theta_tilde`
- Mass matrix is set to a 2x2 matrix for the two joints, and can be adjusted in runtime

## Core Snippets

Animate `ref_dof_prev` attribute to synchronize the motion between the external system and the IPC system. (In this example, we take the transform of IPC system just for demonstration purpose)

```python
def update_ref_dof_prev(info: Animation.UpdateInfo):
    geo: SimplicialComplex = info.geo_slots()[0].geometry()
    ref_dof_prev = geo.instances().find('ref_dof_prev')
    ref_dof_prev_view = view(ref_dof_prev)
    # external_dof_prev is the DOF at last frame of the external system
    ref_dof_prev_view[:] = external_dof_prev
```

Animate `delta_theta_tilde` and `mass` to couple the motion between the external system and the IPC system.

```python
def update_articulation(info: Animation.UpdateInfo):
    dt = info.dt()
    geo_slots = info.geo_slots()
    geo = geo_slots[0].geometry()
    
    delta_theta_tilde = geo['joint'].find('delta_theta_tilde')
    delta_theta_view = view(delta_theta_tilde)
    delta_theta_view[0] = delta_theta_tilde_0 * dt
    delta_theta_view[1] = delta_theta_tilde_1 * dt
    
    mass = geo['joint_joint'].find('mass')
    mass_view = view(mass)

    # external_mass_matrix is the joint mass matrix at last frame of the external system
    mass_view[:] = external_mass_matrix.flatten() 
```
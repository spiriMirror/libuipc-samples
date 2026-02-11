# State Accessor Feature

This example demonstrates how to use the `StateAccessorFeature` to directly inspect and edit the state data of finite element entities and affine bodies in the simulation via an ImGui panel in Polyscope.

## Summary

The `StateAccessorFeature` allows users to access and modify the internal state of simulation objects, such as positions, velocities, and transforms, at runtime. This feature is useful for tasks like manual adjustments, debugging, or implementing custom behaviors, e.g. domain randomization.

## Usage

1. **Initialize the State Accessor**: After initializing the simulation world, retrieve the `AffineBodyStateAccessorFeature` and `FiniteElementStateAccessorFeature` from the world's features.
2. **Create State Geometries**: Use the accessor to create geometries that hold the state data (e.g., positions or transformations).
3. **Inspect/Edit State Data**: Use the ImGui panel to view all ABD transforms/velocities and FEM positions/velocities; edits are pushed back with `copy_from()` when fields change.
4. **Update the Simulation**: Retrieve the updated state and refresh the GUI to reflect the changes; run a sanity check and recover if needed.
5. **View/Set Values**: Use the GUI to view and edit state values directly (ABD transforms/velocities, FEM positions/velocities).

Refer to the `main.py` script for a detailed implementation.

Note that, if the manual modifications may lead to invalid states (e.g., penetrations), user should perform sanity checks and recover the last valid state if necessary.

![GUI](./image.png)

For more insights and updates, please visit the discussion thread: https://github.com/spiriMirror/libuipc/discussions/232
# Compute Mesh d_hat Example

This example demonstrates how to compute the mesh-wise $\hat{d}$ value for a mesh in a simulation environment.

```python
compute_mesh_d_hat(mesh, max_d_hat=default_d_hat)
```

This `compute_mesh_d_hat` function will compute the mesh-wise $\hat{d}$ value for the provided mesh. The $\hat{d}$ is computed based on the mesh resolution (normally the edge length of the mesh).
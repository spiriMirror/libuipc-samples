# Anisotropic cloth: independent directional stretch

Three procedural cloth squares receive identical biaxial edge traction:

| Sheet | a direction (X) | b direction (Y) |
| --- | --- | --- |
| `shared` | Soft, old API | Soft, old API |
| `a_stiff` | Stiff, new API | Soft |
| `b_stiff` | Soft | Stiff, new API |

The left and bottom edges are pinned; the right and top edges are pulled with
10 N/m. Gravity and contact are disabled to isolate membrane response. Each
sheet has the same mesh, mass density, thickness, shear and bending parameters.
No mesh download is needed. The stiff direction should elongate less.

## Run

Use a CUDA-capable machine and **pyuipc built from libuipc with the new
`stretch_a_moduli` / `stretch_b_moduli` overload**. The script checks this API
before creating the scene; older packages such as v0.0.29 do not support it.
From this directory:

```shell
python main.py
python main.py --headless 120 --validate
```

The GUI shows the three sheets side by side and reports mean directional edge
strains. Click **Run** to start. The headless command checks finite positions,
fixed pins and reduced strain along each stiff direction, then writes
`summary.json` and `cloth_positions.npz` beneath
`output/examples/53_anisotropic_cloth/`. This is a dynamic comparison at a
finite frame count, not an equilibrium measurement.

Headless runs also emit `BENCHMARK_RESULT` with per-frame wall times, solver
statistics and final strains. The parent libuipc benchmark registry selects
`--resolution 8 --headless 30` as its small workload (243 total vertices and
384 triangles). This short workload measures execution; use the 120-frame
`--validate` command above to check the directional material response.
Synchronized Timer diagnostics are disabled unless `UIPC_BENCHMARK_TIMERS=1`.
Compare timings on the same machine and configuration; no GPU model or
absolute frame-time target is prescribed.

Useful controls include `--resolution 24`, `--soft-young 2000`,
`--stiff-young 200000`, `--soft-poisson 0.25`, `--stiff-poisson 0.3`,
`--traction 10` and `--output <directory>`. Validation requires at least 80
frames and a stiff/soft coefficient ratio of at least 10.

## Material API and interpretation

```python
shell.apply_to(
    mesh,
    stretch_a_moduli=ElasticModuli2D.youngs_poisson(2e5, 0.3),
    stretch_b_moduli=ElasticModuli2D.youngs_poisson(2e3, 0.25),
    shear_moduli=ElasticModuli2D.youngs_poisson(10, 0.25),
    thickness=0.001,
    mass_density=200,
    strain_rate=100,
)
```

All three moduli are required keyword arguments in the new Python overload.
The old single-modulus and stretch/shear overloads remain supported and share
one stretch coefficient between directions. Calling an old overload again
restores shared stretch on that mesh.

The existing rest-frame axes are used. These meshes lie in XY with +Z normals,
so `a` is X and `b` is Y. This example does not add UV or arbitrary fiber-frame
support. For other rest orientations, do not assume that a/b are world X/Y.

With the one-sided thickness radius `r`, coefficients are
`Ka = Ea*(2*r)/(1-nua^2)` and `Kb = Eb*(2*r)/(1-nub^2)`.
The independent shear coefficient is `Eg/(2*(1+nug))`, without another thickness
factor. The triangle energy is rest area times
`Ka*phi(stretch_a) + Kb*phi(stretch_b) + Kg*shear^2`, where
`phi(s)=(s-1)^2 + strain_rate*max(s-1,0)^3`.

These E/nu inputs retain the old coefficient convention. The two nu values
adjust the two stiffnesses independently; they do **not** specify measured
transverse contraction ratios. There is no normal-strain cross term in this
Baraff-Witkin model. This is directional stiffness control, not a calibrated
orthotropic Hooke material.

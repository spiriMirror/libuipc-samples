# Bunny Cloth

This is a simple cloth simulation with libuipc. 

![image](image.png)

In this example, we use `NeoHookeanShell` and `DiscreteShellBending` to simulate the cloth.

Note that in order to keep the consistency of Elastic Moduli, we use a unified interface called `ElasticModuli`, and specify the Young's modulus and Poisson's ratio to calculate the Lamé parameters.

```python
nks = NeoHookeanShell()
dsb = DiscreteShellBending()
moduli = ElasticModuli.youngs_poisson(10 * kPa, 0.499)
nks.apply_to(cloth_mesh, moduli=moduli, mass_density=200, thickness=0.001)
dsb.apply_to(cloth_mesh, bending_stiffness = 10.0)
```
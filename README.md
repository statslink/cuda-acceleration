# CUDA Acceleration for CFD (2D Euler Solver)

This repository implements a finite-volume solver for the two-dimensional compressible Euler equations, with both CPU (C++) and GPU (CUDA) implementations. The focus is on accelerating classical CFD problems using modern GPU architectures.

## Overview

We solve the system of conservation laws:

∂u/∂t + ∂f(u)/∂x + ∂g(u)/∂y = 0

where:

u = (ρ, ρv_x, ρv_y, E)^T

This represents conservation of mass, momentum, and energy. The system is closed using the ideal gas equation of state:

p = (γ − 1)ρϵ

## Numerical Method

The solver is based on:

- Finite Volume Method (FVM)
- SLIC (Slope Limited Centered) scheme
- Dimensional splitting (x and y directions)

Core steps of the algorithm:

1. Slope-limited reconstruction
2. Half-time step evolution
3. Flux computation (FORCE scheme)
4. Finite-volume update

The update step is given by:

u^{n+1}_{i,j} = u^n_{i,j}
− (Δt/Δx)(f_{i+1/2,j} − f_{i−1/2,j})
− (Δt/Δy)(g_{i,j+1/2} − g_{i,j−1/2})

## Test Problems

### Quadrant Problem
A 2D domain split into four regions with different initial conditions. This tests shock interactions and discontinuities.

### Shock–Bubble Interaction
A shock wave interacts with a helium bubble, producing complex flow structures and validating shock-capturing behaviour.

## CPU vs GPU Implementation

Both implementations follow the same numerical structure, with differences in execution:

- CPU: nested loops over grid cells
- GPU: CUDA kernels operating over flattened memory

GPU optimisation techniques include:

- Coalesced memory access (i + j·nx indexing)
- Kernel decomposition (separate kernels per stage)
- Shared memory for reductions

## Performance

Observed speed-ups:

- Main simulation loop: ~10–18×
- Flux computations: up to ~100×

CPU and GPU solutions are numerically consistent, with small L2 differences at final time steps. :contentReference[oaicite:0]{index=0}


## Hardware

- CPU: Intel i7-13620H
- GPU: NVIDIA RTX 4050 (6GB)
- Compilers: gcc, nvcc
- Visualisation: Python (matplotlib)

## Results

- GPU implementation achieves significant acceleration across all components
- Visual outputs from CPU and GPU are identical
- Numerical differences remain small and acceptable for the given grid sizes

## Key Takeaways

- Memory layout is critical for GPU performance
- SLIC scheme is well-suited for parallelisation
- CFD workloads benefit strongly from data-parallel execution

## Future Work

- Multi-GPU support (MPI + CUDA)
- Higher-order numerical schemes
- Adaptive mesh refinement
- Extension to more complex flow problems

## References

- Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*
- Anderson, *Fundamentals of Aerodynamics*
- Liska & Wendroff (quadrant problem)
- Bagabir & Drikakis (shock-bubble interaction)

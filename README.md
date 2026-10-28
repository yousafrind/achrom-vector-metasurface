# achrom-vector-metasurface

Inverse-designed, three-wavelength (450/532/635 nm) metasurface that turns an arbitrary input
polarization into an input-state-coupled Poincare (vector) beam at a common focal plane.
Implementation paper: maths serves design and verification.

## Layout
- `achrom/rcwa.py`   eig-free differentiable RCWA in JAX (expm + scattering-matrix doubling)
- `tests/test_rcwa.py`  validation suite: `python tests/test_rcwa.py`
- `bench/jax_bench.py`  hardware check (already run on the Ryzen AI Max)
- `bench/bench_rcwa.py` solver throughput at 121-529 Fourier orders (run on the Ryzen, push the JSON)

## Ryzen environment (from the benchmark)
JAX 0.11.1 + `rocm[libraries]==7.13.0` from https://repo.amd.com/rocm/whl/gfx1151/ (see
github.com/YauhenBichel/strix-halo-jax).  Run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
Plain `pip install jax jax-rocm7-plugin` silently falls back to CPU; the benchmarks refuse to run on CPU.

## Status (package 1)
Solver validated: uniform film vs Fresnel (1e-9), subwavelength grating vs exact Rytov index,
inverse-rule vs Laurent limit, energy conservation (1e-12), symmetry, gradients vs finite
differences, convergence in orders.  Doubling is essential: without it energy error is 1e139.
Known limits: normal incidence only, one patterned layer, Li inverse rule is exact only for
rectilinear shapes (staircase approximation for freeform maps).

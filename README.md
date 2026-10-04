# ocp++

C++17 framework for numerical optimal control problems (OCPs), following an
allocation-free, header-only approach.

> **Provenance:** this project is (currently) written by Qwen3.8 under the
> tight guidance and supervision of a human.

## Goal

Provide a stable, solver-agnostic mathematical problem interface so that many
different solver algorithms can consume the same problem definition. The
architecture strictly separates:

1. **the problem** — dynamics, cost, constraints and their derivatives,
   implemented by the user as a concrete class (no virtual dispatch, no
   exceptions in the numerical core), and
2. **the solver** — a template over the concrete problem type (QP solvers,
   SQP solvers, ...).

The design goal is a research-grade interface with compile-time structural
dimensions, fixed-size Eigen linear algebra, and a runtime (or compile-time)
horizon. The HPIPM QP solver and the acados-style SQP solver are provided as
the first (reference) solver implementations.

## What is here

- `include/ocp/problem.hpp` — the problem interface: `Problem<Dims, Scalar>`
  (CRTP), `Trajectory`, `Solution`, box / linear / terminal constraint specs.
- `include/ocp/integrators/` — continuous-time support: ODE models and a
  continuous-problem interface wrapped by explicit, implicit and multistep
  Runge-Kutta integrators to produce a discrete-time problem.
- `include/ocp/solvers/hpipm/` — the HPIPM QP solver, the first (reference)
  QP solver implementation.
- `include/ocp/solvers/acados/` — the acados-style SQP outer loop (line
  search, regularization, merit function), the first (reference) SQP solver
  implementation.
- `tools/acados2ocp_pp.py` — generator that ports an acados OCP (from its
  CasADi-generated `acados_ocp_nlp.json` + C model files) into an ocp++
  problem class.
- `tests/` — unit tests for the QP and SQP solvers and the integrators, plus
  comparisons against the acados reference solver.

## Examples

- `examples/double_integrator/` — the minimal example: a small hand-written
  discrete-time problem that exercises the full problem + SQP interface.
- `examples/pendulum_on_cart/` and `examples/unicycle/` — nontrivial
  continuous-time problems (CasADi-generated implicit-DAE dynamics) ported
  from acados with the JSON generator, validated against the acados reference
  trajectory (`compare.sh`).

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/double_integrator   # interface sanity test; exit 0 = pass
```

Only dependency: Eigen3 (header-only). C++17 required.

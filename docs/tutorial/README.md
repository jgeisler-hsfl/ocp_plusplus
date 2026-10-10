# ocp++ Tutorial

A hands-on introduction to the ocp++ framework, organized in increasing order
of difficulty. Every chapter builds on the previous ones; each compiles and
runs on its own so you can check your understanding as you go.

## Prerequisites

- A C++17 compiler (GCC or Clang).
- [CMake](https://cmake.org) >= 3.16.
- [Eigen3](https://eigen.tuxfamily.org) (header-only). CMake finds it in the
  standard locations; if it is installed elsewhere, pass
  `-DEIGEN3_INCLUDE_DIR=/path/to/eigen` at configure time.
- Everything else (the multidual header used by the automatic-differentiation
  integrator paths) is vendored in the repository.

## How to work through it

Build all examples once from the repository root:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Chapters 2, 6 and 7 each have a companion example program under
`examples/tutorial/` (targets `tutorial_minimal`, `tutorial_continuous`,
`tutorial_advanced`). The remaining chapters reference the existing examples
(`examples/double_integrator/`, `examples/mass_spring/`, ...) and tests, which
serve as complete, runnable reference material for every feature.

## Chapters

| # | File | Topic |
|---|------|-------|
| 1 | [01-getting-started.md](01-getting-started.md) | What an OCP is, what ocp++ provides, first contact |
| 2 | [02-first-problem.md](02-first-problem.md) | Writing your first discrete-time problem |
| 3 | [03-solving-the-ocp.md](03-solving-the-ocp.md) | The SQP solver: options, solution, statistics, multipliers |
| 4 | [04-constraints.md](04-constraints.md) | Box, linear, nonlinear and soft constraints |
| 5 | [05-second-order-models.md](05-second-order-models.md) | Exact Hessians vs Gauss-Newton, Levenberg-Marquardt |
| 6 | [06-continuous-time-and-integrators.md](06-continuous-time-and-integrators.md) | ODE/DAE models, Runge-Kutta integrators, automatic differentiation |
| 7 | [07-advanced-topics.md](07-advanced-topics.md) | Compile-time horizons, free initial state, QP internals, code generation |

## Further reading

- `ARCHITECTURE.md` — the design principles behind the interface.
- `include/ocp/problem.hpp` — the central interface contract (the comment
  block at the top is kept in sync with the code and is authoritative).
- `../acados/docs/algorithm/` (if you have the acados repository next to
  this one) — the algorithm documentation the reference solvers are ported
  from.
- `tests/` — unit tests for every solver and integrator feature; a good
  source of minimal usage snippets for each option.

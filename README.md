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
  (CRTP), `Trajectory`, `Solution`, box / linear / terminal constraint
  specs, plus an optional duck-typed `params()` hook that lets a problem
  share a user-defined parameter object (scalars and Eigen matrices)
  between its model functions — by-name discovery and setting via the
  inspect protocol.
- `include/ocp/integrators/` — continuous-time support: ODE models and a
  continuous-problem interface wrapped by explicit, implicit and multistep
  Runge-Kutta integrators to produce a discrete-time problem.
- `include/ocp/scaling.hpp` — `ScaledProblem<Base>`, a composition wrapper
  that presents any problem in diagonally scaled state/input coordinates
  (`x_s = sx * x`, `u_s = su * u`), turning a badly scaled problem into a
  well-conditioned one; the solver sees only the scaled interface, and
  `unscale()` recovers the primal + KKT solution in the original units.
- `include/ocp/solvers/hpipm/` — the HPIPM QP solvers: a staged QP solver and
  a dense QP solver (used by the SQP full-condensed mode), the first
  (reference) QP solver implementations. The staged solver also supports
  QP-level forward/adjoint sensitivities (`sens_forward` / `sens_adj`,
  HPIPM seed module, `seed.hpp`).
- `include/ocp/solvers/acados/` — the acados-style SQP outer loop (line
  search, regularization, merit function), the first (reference) SQP solver
  implementation. Its QP subproblem is solved staged (default), as a single
  full-condensed dense QP, or as a block-condensed staged QP (stages grouped
  into blocks), selected by `AcadosOptions.qp_mode`. A second driver,
  `RtiSolver`, implements acados' standard real-time-iteration SQP (one
  QP assembly + solve per call, full step, no line search). The driver
  also exposes first-order NLP-level sensitivities (`sens_x0` w.r.t. the
  fixed initial state, `sens_param` w.r.t. a model parameter vector via
  `Dims::np`), available in all three QP modes.
- `tools/acados2ocp_pp.py` — generator that ports an acados OCP (from its
  CasADi-generated `acados_ocp_nlp.json` + C model files) into an ocp++
  problem class.
- `tests/` — unit tests for the QP and SQP solvers and the integrators, plus
  comparisons against the acados reference solver.
- `mex/` — optional MATLAB MEX wrapper (one MEX binary per problem/solver
  configuration, persistent session state, by-name option and solution
  access); see [mex/README.md](mex/README.md).

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

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — design principles behind the
  interface.
- [docs/tutorial/](docs/tutorial/README.md) — a hands-on tutorial, from
  writing your first problem to advanced features (constraints, Hessian
  models, continuous-time integrators, compile-time horizons, code
  generation), with runnable companion examples under `examples/tutorial/`.

## License

This project is licensed under the GNU General Public License version 3
(see [LICENSE](LICENSE)). It includes third-party components (acados,
HPIPM, cppduals) that retain their original licenses; see
[NOTICE](NOTICE) for the full provenance and copyright statements.

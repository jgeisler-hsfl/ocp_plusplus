# Problem-level unified autodiff: AutoDiffProblem mix-in

TODO item: "The autodiff concept needs to be extended to the whole
problem formulation including costs and nonlinear constraints in a
unified way, ideally leaving the choice for user supplied vs AD
derivatives to the user on a function per function basis."

## Problem

The AD concept so far covered only the dynamics level (ODE `f` on the
explicit integrator, DAE `F` on the implicit integrator via
`AutoDiffDae`). The `Problem<Dims>` cost and nonlinear-constraint
derivative entries — stage/terminal cost grad + Hessian, stage/terminal
(non)linear (in)equality Jacobians and multiplier-contracted Hessians —
had to be written by hand (or generated), independently of the integrator
level. The goal was one CRTP mix-in that provides all of those entries
from the six scalar-templated value functions alone, while letting the
user replace any individual entry with a hand-written one.

## Design

New header `include/ocp/autodiff_problem.hpp` with

    template <class Derived, class Dims, bool WithConstrHess = true>
    class AutoDiffProblem

a CRTP mix-in that does **not** inherit `Problem<Dims>` (no diamond; the
concrete class inherits the dynamics adapter, e.g. `ContinuousProblem`,
*and* `AutoDiffProblem`).

### Value functions (user-written, once)

Six scalar-templated value functions, mirroring the `Problem<Dims>`
value entries but accepting `D` for `scalar_t`:

- `stage_cost_value(k, x, u)` -> `D`
- `stage_inequality_value(k, x, u)` -> `Matrix<D, ng, 1>`
- `stage_equality_value(k, x, u)` -> `Matrix<D, ne, 1>`
- `terminal_cost_value(x)` -> `D`
- `terminal_inequality_value(x)` -> `Matrix<D, ng_t, 1>`
- `terminal_equality_value(x)` -> `Matrix<D, ne_t, 1>`

Because the solver calls value functions with Eigen *expressions* (trial
states `x + step`, `CwiseBinaryOp`), the convention is to write them
expression-templated:

    template <class XExpr, class UExpr>
    auto stage_cost_value(int k, const XExpr& x, const UExpr& u) const
        -> typename XExpr::Scalar
    { using D = typename XExpr::Scalar; ... }

(Non-template double-only functions would also work with concrete
`Matrix<double, ...>` parameters via Eigen's converting constructor, but
they then cannot drive the AD passes — see below.)

### AD derivative entries (mix-in-provided)

The mix-in provides the 12 concrete `Problem<Dims>` derivative
signatures, each computed by a single forward pass on the vendored
multidual scalar (`thirdparty/cppduals`, Eigen glue in
`integrators/duals_eigen3.hpp`):

- `*_value_grad` (stage + terminal cost): one pass with
  `Dual1 = dual<scalar_t, nIn>`; value carried for free.
- `*_value_grad_hess` (stage + terminal cost): one pass with
  `Dual3 = dual<Dual1, nIn>`; input `i` carries the inner basis `e_i`
  (gradient) and the outer basis `e_i` (full Hessian basis), so
  `grad(j) = c.rpart().dpart(j)`, `hess(a,b) = c.dpart(a).dpart(b)`.
  Cheaper than `nIn` nested-dual HVP passes.
- `*_value_jac` (4 constraint groups): one `Dual1` pass; both Jacobian
  blocks extracted per row.
- `*_value_jac_hess` (4 constraint groups, only when
  `WithConstrHess = true`): one `Dual3` pass; the multiplier `lam` is
  contracted only at extraction:
  `hess = sum_i lam_i * D^2(phi_i)`, `[x; u]` (stage) / `[x]` (terminal)
  layout.

`nIn = nx + nu` for stage entries (z = [x; u]), `nIn = nx` for terminal
entries (z = [x]).

### `WithConstrHess` switch

Mirrors `AutoDiffDae::WithHess`:

- `WithConstrHess = true` (default): the primary template plus a partial
  specialization adds the four constraint `*_value_jac_hess` entries;
  pair with `Dims::has_constr_hess = true` (exact-Newton constraint
  Hessian model).
- `WithConstrHess = false`: the primary template only; the constraint
  Hessian entries are absent (they `static_assert` if called), pair with
  `Dims::has_constr_hess = false` (Gauss-Newton model). Cost entries and
  constraint Jacobians are unaffected.

`provides_constr_hess` is the corresponding trait constant.

### Per-function user-vs-AD choice

Name hiding + using-declarations, no runtime dispatch:

- An entry hand-written in the concrete class hides both the mix-in's AD
  version and the `Problem<Dims>` stub; the solver (templated on the
  concrete type) dispatches to it.
- Entries left to AD must be pulled into scope:

      using ocp::AutoDiffProblem<MyProb, Dims>::stage_cost_value_grad;

  Without it, the `Problem` stub and the mix-in entry of the same name
  are both visible and the call is ambiguous (same depth, two unrelated
  bases).

### Dual-evaluability detection

Six SFINAE traits (`stage_cost_value_evaluable_at<Derived, Dims, D>`,
etc.) probe whether the value function is instantiable at dual scalar `D`
*and* returns the D-typed value (guards against Eigen's converting
constructor accepting `Matrix<D>` where `Matrix<double>` is expected).
Each AD entry carries a `static_assert` on the matching trait with a
message naming the entry and the remedy (hand-write it, drop the
using-declaration).

## Key decisions / pitfalls hit

- **No re-declared `Problem` aliases in the mix-in.** The first draft
  re-declared `state_t`, `control_t`, ... in the mix-in; since the
  concrete class inherits *both* `Problem<Dims>` and the mix-in, every
  such name became ambiguous (two unrelated bases, same depth). The
  mix-in now only declares `using P = Problem<Dims>;` and refers to
  `typename P::...` everywhere.
- **Public seeders.** The `true` specialization reuses the primary's
  `seedS1/seedS3/seedT1/seedT3` via `Base::`; a qualified call to a
  private static of a dependent base is ill-formed, so the seeders are
  public (documented as implementation detail).
- **Expression-templated value functions.** The solver passes Eigen
  expressions, not concrete `Matrix` objects; templated value functions
  with concrete `Matrix<D, nx, 1>` parameters fail template deduction.
  The documented convention is `template <class XExpr, class UExpr>` +
  `-> ... typename XExpr::Scalar ...`.
- **`typename` on all `P::...` member types** in the mix-in (dependent
  type `P`).

## Changes

- `include/ocp/autodiff_problem.hpp` (new): the mix-in, SFINAE traits,
  file-level contract comment.
- `tests/integrators/problem_autodiff_4l.cpp` (new, phase-4l block):
  - `FullValues`: six polynomial value functions (expression-templated);
    analytic references + central finite differences.
  - `ProbFull`: all 12 entries from AD (`WithConstrHess = true`),
    checked against analytic refs and FD.
  - `ProbMix`: two hand-written marker entries + AD for the rest —
    proves per-function override precedence.
  - `ProbGn`: `WithConstrHess = false`; trait + cost-AD-unchanged checks.
  - `SqpAd` / `SqpHand`: continuous double-integrator OCP (state box,
    stage ineq, terminal (in)equality) solved end-to-end by
    `AcadosSolver`; AD vs hand-written derivatives must agree
    (status, NLP residuals, cost, trajectories).
- `tests/integrators/integrators_main.cpp`: `run_problem_autodiff_4l_tests`.
- `CMakeLists.txt`: `problem_autodiff_4l.cpp` into `integrators_unit`.

## Tests

`integrators_unit` phase-4l lines:

    4l full  ProbFull    OK  (12 AD entries, analytic + FD)
    4l mix   ProbMix     OK  (hand-written beats AD, rest AD)
    4l gn    ProbGn      OK  (no constr hess, cost/jac AD)
    4l sqp   SqpAd/SqpHand OK  (cost=11.832906, maxdx=9.6e-12, maxdu=5.0e-11)

Full regression: 22 test binaries + 3 compare.sh green.

The negative cases (missing / double-only value function driving an AD
entry; `WithConstrHess = false` calling a constraint Hessian entry) are
compile-time `static_assert`s by design and are not exercised at
runtime.

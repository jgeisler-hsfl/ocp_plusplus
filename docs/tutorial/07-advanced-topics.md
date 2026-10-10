# Chapter 7 — Advanced topics

This chapter covers the remaining features: compile-time horizons, free
initial states (moving-horizon estimation), warm starting, the QP
internals, globalization, and the acados JSON generator. The companion
example is `examples/tutorial/advanced.cpp` (target `tutorial_advanced`).

## 7.1 Compile-time horizons

By default the horizon `N` is a runtime quantity: `Solution<P> sol(N)`
heap-allocates its trajectories, and a solver `resize`s its workspaces
once per solve. A non-negative template parameter fixes the horizon at
compile time:

```cpp
constexpr int N = 10;
ocp::Solution<AdvFixed, N> sol;        // no constructor argument!
ocp::AcadosSolver<AdvFixed, N> solver; // solvers are templated over NH too
```

With a fixed horizon:

- the trajectories become fixed-extent `std::array` storage — stack
  allocated, zero-initialized, **no heap allocation** and no `resize`
  (a `resize(N)` call becomes a consistency check);
- `sol.N` is initialized from `NH`;
- the QP solver (`HpipmQpSolver<P, NH>`), its data model (`Qp`, `QpSol`,
  `QpRes`) and the SQP driver all carry the same `NH` parameter, so the
  per-stage workspace quantities use the same mechanism.

Use it when the horizon is known at compile time and you want zero heap
allocation and fully static code. The companion example demonstrates it in
`fixed_horizon_and_warm_start()`:

```cpp
ocp::Solution<AdvFixed, N> sol;
// sol.N == N, sol.x.size() == N + 1, lambda_dyn[0] is zero-initialized
```

## 7.2 Free initial state (MHE)

`Dims::fixed_initial_state = false` turns the initial state `x_0` into a
**decision variable** — the setting for moving-horizon estimation (MHE)
and batch problems that estimate the initial state:

- `initial_state()` is *not part of the interface* (implementing it on the
  problem is pointless; the solver never calls it);
- the stage-0 state box now bounds `x_0` like any other stage (a natural
  way to encode a measurement prior);
- the stationarity residual omits the `x_0` component when it is fixed;
  with a free `x_0` it is a regular variable.

The companion example solves the same double integrator with a free `x_0`
and a narrow stage-0 prior box `[0.8, 1.2] x [0.3, 0.7]`:

```
MHE   solve: status=1 iters=2 cost=2.557808537  x0=[0.8000; 0.3000]
```

The estimate lands on the corner of the prior box: the cost wants `x_0`
near the origin, so it pushes against the lower bounds — the stage-0 box
is active and does its job. (A real MHE adds an arrival cost and a
measurement model; the mechanics are the same.)

When `fixed_initial_state` is true, the stage-0 state box is ignored: the
solver pins `x_0` to `initial_state()` with internal pin rows instead.

## 7.3 Warm starting

`ocp::Solution` is both the input and the output of a solve, so warm
starting is passing the previous solution back in. The receding-horizon
(MPC) pattern shifts the trajectory *and the multipliers* one stage up and
re-solves the same solver object:

```cpp
ocp::Solution<AdvFixed, N> warm(sol);          // copy = warm start
for (int k = 0; k < N - 1; ++k)
{
    warm.x[k] = sol.x[k + 1];
    warm.u[k] = sol.u[k + 1];
    warm.lambda_dyn[k] = sol.lambda_dyn[k + 1];
    warm.lambda_box_state[k] = sol.lambda_box_state[k + 1];
}
warm.x[0] = problem.initial_state();   // new initial state (measurement)
warm.u[N - 1].setZero();
solver.solve(problem, warm);
```

Notes:

- The solver-internal slacks are warm-started across repeated solves on
  the same solver object automatically (when the horizon matches).
- The QP warm start is controlled separately: `opts.qp_warm_start` (0 =
  cold, 2 = primal+dual, 3 = hot) and `opts.warm_start_first_qp` (also
  warm-start the very first QP). By default the SQP driver warms the QP
  duals from the previous iteration and keeps the primal step cold.
- On the small convex example the warm solve takes the same two
  iterations as the cold one — the QP step is already exact on this
  problem, so both need one step plus the confirming iteration. The warm
  start pays off on harder problems (nonlinear dynamics, active
  constraints, tighter tolerances) where the first QP is far from the
  solution.

## 7.4 Statistics

Every solve records one `ocp::SqpIteration` row per iteration:

```cpp
struct SqpIteration
{
    double res_stat, res_eq, res_ineq, res_comp;  // NLP residuals at start
    int qp_status, qp_iter;                       // inner QP result
    double step_norm, alpha;                      // step norm, accepted step
    int funnel_iter_type; double funnel_width;    // funnel globalization
};
```

`solver.statistics()` (an `ocp::AcadosStatistics`) exposes the rows and
the final status, and can export CSV:

```cpp
solver.statistics().write_csv("tutorial_advanced_stat.csv");
```

which writes a header plus one row per iteration:

```
iter,res_stat,res_eq,res_ineq,res_comp,qp_status,qp_iter,step_norm,alpha
0,1.3062322170e+00,1.0931058930e+00,0.0,1.0e-16,1,3,4.8890175030e+00,1.0e+00
1,3.3750779949e-14,2.8421709430e-14,0.0,1.0e-16,1,3,4.8890175030e+00,1.0e+00
```

## 7.5 QP internals (for the curious)

The SQP assembles a *step-formulation* QP at the current iterate: the
variables are `z_k = (du_k; dx_k; s_k)` (control step, state step, slacks
— note the `(u; x)` order, the solver permutes the `[x; u]` problem
layout), the rows are the linearized constraints, and the right-hand sides
carry the residuals and bound offsets of the current iterate.

The staged QP data model lives in `include/ocp/solvers/hpipm/qp.hpp`
(`QpDim<P>`, `Qp<P, NH>`, `QpSol<P, NH>`, `QpRes<P, NH>`). The driver
exposes the last assembled QP and its solution for inspection:

```cpp
const ocp::Qp<AdvFixed, N>& qp = solver.last_qp();
const ocp::QpSol<AdvFixed, N>& step = solver.last_qp_sol();
```

The QP solver itself (`HpipmQpSolver`) is a port of the HPIPM OCP-QP
interior-point method (relative formulation, Mehrotra
predictor-corrector). Its options (`ocp::HpipmOptions`, the BALANCE preset
plus acados overrides), statistics (21 columns per IPM iteration) and
residual tolerances (`solver.qp()`-level `set_residual_tolerances`) are
rarely touched — the SQP driver drives them via the options of chapter 3:

- **QP scaling:** `opts.scale_qp_objective` / `opts.scale_qp_constraints`
  (Gershgorin objective scaling, inf-norm constraint scaling) improve
  badly scaled problems.
- **Per-iteration QP tolerances:** `opts.nlp_qp_tol_strategy` (0 = fixed
  HPIPM defaults, 1 = from the NLP residuals, 2 = from the scaling
  factors) trades inner QP effort against SQP iterations.
- **Timeout:** `opts.timeout_max_time` (wall-clock budget) and
  `opts.timeout_heuristic` (per-iteration estimate; exit status
  `kTimeout`).

## 7.6 Globalization

The SQP globalizer decides how full a step to accept. The default is merit
backtracking (`ocp::MeritBacktracking`), configured via `opts.glob`:

| Option | Default | Meaning |
|--------|---------|---------|
| `alpha_min` | 0.05 | minimum accepted step size |
| `alpha_reduction` | 0.7 | backtracking shrink factor |
| `full_step_dual` | false | full-step vs relaxed dual update |
| `use_soc` | false | second-order-correction re-solve when a step is rejected (Nocedal & Wright 18.51) |
| `eps_sufficient_descent` | 1e-4 | sufficient-descent margin |

The solver template takes the globalizer (and regularizer and QP solver)
as template parameters, so they can be swapped:

```cpp
ocp::AcadosSolver<P, NH, QpSolver, Regularizer, Globalizer>
```

The statistics expose which mechanism accepted each step
(`funnel_iter_type`: `-1` for merit backtracking; a funnel globalizer
records `f`/`h`/`b`/`p` accept types and its width).

## 7.7 The acados JSON generator

Hand-writing a problem is ideal for small models, but for models
developed in the acados toolbox (CasADi-generated dynamics, cost and
constraint code) the generator
`tools/acados2ocp_pp.py` ports an existing acados OCP into an ocp++
problem class. It reads the standard `acados_ocp_nlp.json` artifact and
emits:

```
<out>/<Name>.hpp          Dims + model + integrator alias + Problem class
<out>/<Name>_capi.hpp     extern-C wrappers + CasADi workspaces
<out>/<name>_main.cpp     solve + CSV driver
```

Usage:

```bash
python3 tools/acados2ocp_pp.py \
    --json examples/masses_chain/acados_codegen/acados_ocp_nlp.json \
    --name MassesChainGen --out examples/masses_chain/generated \
    --csrc examples/masses_chain/acados_codegen/masses_chain_model/masses_chain_impl_dae_fun.c
```

The CasADi-generated C value functions are *not* regenerated — they are
linked into the target and wrapped by the generated capi header (their
argument layout is read from the ABI comment at the top of each file).
Jacobians and Hessians of nonlinear cost/constraint terms are central
finite differences of the value functions, consistent with the
dynamics-Jacobian treatment.

Supported (validated on the `masses_chain`, `pendulum_on_cart`,
`unicycle`, `p5probe` examples):

- cost types `LINEAR_LS` / `NONLINEAR_LS` / `EXTERNAL` (stage-0, path,
  terminal),
- integrator `IRK` (Gauss-Legendre / Radau IIA, 2–4 stages) or `ERK`
  (K1–K4), any number of sub-steps,
- box constraints (stage state incl. a stage-0 box, stage control,
  terminal state), free or fixed initial state,
- linear two-sided stage/terminal constraints,
- nonlinear (in)equality stage/terminal constraints,
- soft constraints (nonlinear rows with per-side weights; box/linear rows
  require equal weights per side),
- time-varying functions.

Not supported: algebraic states (`nz > 0`), more than one parameter
(`np > 1`), sparse CasADi arguments, stage-0-specific constraint rows, and
a few rare acados features (the tool raises a descriptive error).

The generated problems are exactly the examples of chapters 5–6 in
generated form: `examples/pendulum_on_cart/generated/PendulumGen.hpp` is
an `IRK` Gauss-Legendre-4 problem with a Gauss-Newton Hessian
(`has_dynamics_hess = false`), and `examples/unicycle/`,
`examples/masses_chain/`, `examples/p5probe_erk/`, `examples/p5probe_irk/`
cover other combinations. The `compare.sh` scripts in the example
directories build the acados reference and the ocp++ port and report the
cross-solver differences.

## 7.8 Where to go from here

- `tests/` contains a unit test per feature (options, warm start,
  funnel, SOC, timeout, scaling, integrators, DAE autodiff, ...) — the
  fastest reference for the exact semantics of an option.
- `ARCHITECTURE.md` documents the design principles.
- Not yet part of the interface (anticipated extensions): parameter
  sensitivities and algebraic (eliminated) variables; an interactive
  "discovery" mechanism for options and statistics is on the roadmap.

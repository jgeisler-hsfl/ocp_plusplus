# Chapter 7 — Advanced topics

This chapter covers the remaining features: compile-time horizons, free
initial states (moving-horizon estimation), warm starting, the QP
internals, globalization, the acados JSON generator, real-time iteration
and NLP-level scaling. The companion example is
`examples/tutorial/advanced.cpp` (target `tutorial_advanced`).

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
(MPC) pattern shifts the trajectory *and the multipliers* one stage up
and re-solves the same solver object. `ocp::Solution::shift` does this
in place:

```cpp
// after the previous MPC step's solve:
//   x[k] <- x[k+1] (k = 0..N-1),  u[k] <- u[k+1] (k = 0..N-2)
//   stage multipliers shift one stage up and are zeroed at k = N-1,
//   terminal multipliers stay,  x[0] := new measurement
sol.shift(problem.initial_state());     // ShiftFill::kRollover (default)
solver.solve(problem, sol);
```

`shift(x0_new, fill)` takes the new initial state as an argument and an
optional `ocp::ShiftFill` policy for the new terminal stage: `kRollover`
(default) keeps the old terminal state and last control as the warm
guess, `kZero` zeroes them. `cost_value` and `status` are reset.

When the same solver object solves consecutive MPC steps, route the
shift through the driver so the solver-internal slack variables advance
one stage together with the solution:

```cpp
solver.shift(sol, problem.initial_state());  // solution + internal slacks
solver.solve(problem, sol);
```

The driver-level `shift` applies the same `Solution::shift` mapping and
additionally advances `SqpSlacks` (only effective when the last solve
ran at the same horizon). If a shifted warm start turns out unhelpful,
`solver.reset_warm_start()` zeroes the slacks and the persistent QP
duals for a one-shot cold start.

For reference, the manual version of the shift (equivalent to
`kRollover` on a fresh copy):

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
- The QP warm start is controlled separately: the mode lives in the QP
  solver's options (`ocp::HpipmOptions::warm_start`, 0 = cold, 2 =
  primal+dual, 3 = hot), reachable by name as `qp.options.warm_start`,
  plus `opts.warm_start_first_qp` (also warm-start the very first QP).
  The SQP driver always keeps the primal step cold; with mode 2 or 3 it
  reuses the duals from the previous QP, with mode 0 each QP starts cold.
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
    double cond_time;                             // condense+expand wall clock (s)
};
```

`solver.statistics()` (an `ocp::AcadosStatistics`) exposes the rows and
the final status, and can export CSV:

```cpp
solver.statistics().write_csv("tutorial_advanced_stat.csv");
```

which writes a header plus one row per iteration:

```
iter,res_stat,res_eq,res_ineq,res_comp,qp_status,qp_iter,step_norm,alpha,funnel_iter_type,funnel_width,cond_time
0,1.3062322170e+00,1.0931058930e+00,0.0,1.0e-16,1,3,4.8890175030e+00,1.0e+00,-1,0.000000,0.000000
1,3.3750779949e-14,2.8421709430e-14,0.0,1.0e-16,1,3,4.8890175030e+00,1.0e+00,-1,0.000000,0.000000
```

`funnel_iter_type` is `-1` when the active globalizer is not a funnel
(`-1`, `1`=f, `2`=h, `3`=b, `4`=p for funnel accept types); `funnel_width`
is the funnel width used in that iteration. `cond_time` is the wall-clock
seconds spent in the condense + expand round-trip for that iteration's QP
(non-zero only in `QpMode::kFullCondensed` / `kPartialCondensed` mode).

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

- **QP scaling:** the live scaling options live on the solver's QP scaler
  (`QpScaler::Opts`, reachable by name as `scaling.options.scale_objective`
  / `scaling.options.scale_constraints`; Gershgorin objective scaling,
  inf-norm constraint scaling) and improve badly scaled problems.
- **Per-iteration QP tolerances:** `opts.nlp_qp_tol_strategy` (0 = fixed
  HPIPM defaults, 1 = from the NLP residuals, 2 = from the scaling
  factors) trades inner QP effort against SQP iterations.
- **Timeout:** `opts.timeout_max_time` (wall-clock budget) and
  `opts.timeout_heuristic` (per-iteration estimate; exit status
  `kTimeout`).

#### QP solve modes (condensing)

Beyond the staged default, `opts.qp_mode` (chapter 3) selects how the staged
QP is solved:

- **`QpMode::kStaged`** (default) — the staged HPIPM IPM over
  `(u_k; x_k; s_k)` per stage, mirroring acados `SQP_RTI_HPIPM`.
- **`QpMode::kFullCondensed`** — the N stages are condensed into a single
  dense QP over `v = (x_0 … x_N, u_0 … u_{N-1})` (the dynamics are
  eliminated), solved with the dense HPIPM IPM, then expanded back to the
  staged layout. Mirrors acados `FULL_CONDENSING_HPIPM`. Well suited to short
  horizons; the dense matrix cost grows with `N`, so keep it modest.
- **`QpMode::kPartialCondensed`** — the N stages are grouped into
  `qp_cond_N` blocks of `NB = N / qp_cond_N` stages each; each block is
  condensed to a mini QP (internal states eliminated) and the resulting
  block-stage QP is solved with the staged HPIPM IPM. A middle ground that
  shrinks the number of stages while staying sparse. Mirrors acados
  `PARTIAL_CONDENSING_HPIPM`; when `qp_cond_N` is 0 the mode falls back to
  `kStaged`.

All three modes consume the same staged assembly and produce identical
`ocp::Solution` quantities. `cond_time` in the statistics (section 7.4)
reports the condense + expand round-trip for the condensed modes.

#### QP-level sensitivities

The HPIPM QP solver also exposes a sensitivity layer: given a converged
solution and a `QpSeed` of right-hand-side perturbations
(`grad` / `b` / `d` / `m` per stage, `include/ocp/solvers/hpipm/seed.hpp`),
`HpipmQpSolver::sens_forward` / `sens_adj` return the linearized response of
the full QP solution (`u`, `x`, `π`, `λ`, `t`) to that seed. The first call
lazily re-factors the KKT system at the final iterate; later seeds reuse the
cached factors and only recompute the constraint-force term, so repeated
seeds cost one `O(N)` pass. Hessian perturbations are expressed as the
equivalent gradient seed `grad = E z*`; the returned response carries the
KKT's internal negative sign (dz/dH = −K⁻¹ (E z*)). This mirrors HPIPM's
`OCP_QP_IPM_SENS_FRW` / `OCP_QP_IPM_SENS_ADJ`.

#### NLP-level sensitivities (`sens_x0`, `sens_param`)

One level above the QP seed layer, the SQP driver exposes first-order
sensitivities of the *converged OCP solution*:

- `AcadosSolver::sens_x0(problem, sol, dx0, dsol)` — the response
  `d(x*, u*)/dx0 * dx0` to a perturbation `dx0` of the fixed initial
  state (requires `Dims::fixed_initial_state`);
- `AcadosSolver::sens_param(problem, sol, dp, dsol)` — the response
  `d(x*, u*)/dp * dp` to a perturbation `dp` of the model's parameter
  vector (requires `Dims::np > 0`).

Each call re-solves the QP KKT system at the converged iterate: the left
hand side keeps the converged Hessian / Jacobian, only the right-hand side
carries the seed derived from the perturbation (for `sens_param`, seeds on
constraint sides that are inactive at the solution are gated to zero — an
inactive constraint does not move to first order). The response is
returned in `QpSol` layout (per stage `ux = (u; x; s)`, terminal
`ux_term = (x; s)`); for `sens_x0` the pinned initial state moves exactly
by `dx0`. Both entries work in all three QP modes ("QP solve modes" above)
and each costs one KKT solve on top of the base solve.

The result is a *first-order* (local) response: it is the exact
linearization of the solution map around the converged iterate, and it is
valid as long as the perturbation is small enough that the active set and
the QP linearization remain the right local model. The shipped tests
(`tests/sqp/sens_x0_5f.cpp`, `tests/sqp/sens_param_5i.cpp`,
`tests/sqp/sens_param_5j.cpp`) verify this against central finite
differences of the full SQP solve in all three QP modes.

A typical use is extracting a local linear state feedback law. One
`sens_x0` call per basis vector of `x_0` yields one column of the gain
`K = du_0*/dx0`:

```cpp
// P: concrete problem with Dims::fixed_initial_state;
// sol: converged Solution at x0* = p.initial_state()
const int nx = P::nx;
Eigen::MatrixXd K = Eigen::MatrixXd::Zero(P::nu, nx);
for (int i = 0; i < nx; ++i)
{
    P::state_t e_i;
    e_i.setZero();
    e_i(i) = 1.0;
    ocp::QpSol<P, Eigen::Dynamic> dsol(sol.N);
    const ocp::Status st = solver.sens_x0(p, sol, e_i, dsol);
    if (st != ocp::Status::kSolved)
    {
        break;  // or handle the failure
    }
    K.col(i) = dsol.ux_first.head(P::nu).cast<double>();
}

// local feedback law around the solved iterate:
// u_0(x_0) ~= u_0* + K (x_0 - x_0*)
const Eigen::VectorXd u0 =
    sol.u[0].cast<double>() + K * (x0 - p.initial_state());
```

For problems with linear dynamics and quadratic cost (e.g. the double
integrator) the optimal policy is affine in `x_0`, so this law is exact up
to the solver tolerances; for general nonlinear problems it is a
first-order approximation around `x_0*`.

#### Classical (non-square-root) Riccati factorization

`HpipmOptions::square_root_alg` (default `true`) selects the square-root
Riccati backward factorization. Set it to `false` for the classical
(non-square-root) variant (HPIPM `square_root_alg = 0`), which
Cholesky-factors only the leading `(u;s)` block and builds the state Hessian
as an explicit Schur complement — useful for cross-checking the two
factorizations.

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

The alternative globalizer is `ocp::Funnel<P, NH>` (funnel acceptance,
after acados `ocp_nlp_globalization_funnel.c`). Instead of accepting a
step only on a merit decrease, it keeps a "funnel" of acceptable L1
infeasibility: steps are accepted while the trial iterate stays inside
the funnel, and the funnel is contracted whenever the infeasibility
drops sufficiently. Each accepted step is labeled `f`/`h`/`b`/`p`
(`funnel_iter_type` 1–4, section 7.4):

- `f` — inside the funnel with a predicted objective decrease that
  passes the switching condition, plus an Armijo test on the objective;
- `h` — the L1 infeasibility falls below
  `sufficient_decrease_factor` times the funnel width: accept and
  shrink the funnel;
- `b` — infeasibility improves but the objective does not: accept on
  the (penalty-weighted) merit and enter penalty mode;
- `p` — in penalty mode: accept on a merit decrease; leave penalty mode
  and shrink the funnel once the infeasibility drops again.

Swap it in as the globalizer template argument and (optionally) tune
the funnel-specific parameters via `solver.globalizer()` — they live on
the globalizer instance itself; the shared line-search options still
come from `opts.glob` (copied into the globalizer at the start of each
`solve()`):

```cpp
ocp::AcadosSolver<P, NH, ocp::HpipmQpSolver<P, NH>,
                  ocp::GlmRegularizer<P, NH>, ocp::Funnel<P, NH>>
    solver(opts);
solver.globalizer().kappa = 0.9;   // funnel shrink blend, before solve()
```

The funnel-specific parameters (public members of `ocp::Funnel`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `initialization_increase_factor` | 15.0 | initial width = `max(initialization_upper_bound, factor * L1_0)` |
| `initialization_upper_bound` | 1.0 | floor of the initial funnel width |
| `sufficient_decrease_factor` | 0.9 | `c_sd`: accept as `h` and shrink when `L1_trial <= c_sd * width` |
| `kappa` | 0.9 | shrink rule `width <- (1-kappa) * L1_trial + kappa * width` |
| `fraction_switching_condition` | 1e-3 | gate for an `f` accept: the predicted objective decrease must be at least this fraction of the current L1 infeasibility |
| `initial_penalty_parameter` | 1.0 | initial penalty weight `rho` of the merit `rho * F + L1` |
| `penalty_contraction` | 0.5 | contraction factor for `rho` in penalty mode |
| `penalty_eta` | 1e-6 | threshold that triggers the `rho` update |
| `use_merit_fun_only` | false | disable the funnel gate; acceptance reduces to the merit/objective tests |

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
- time-varying functions,
- model parameters (`np >= 1`): the generator emits a `{name}Params` struct
  holding the whole parameter vector as a fixed-size `p` leaf (initial values
  from the JSON `parameter_values`), shared by the dynamics model and the
  cost/constraint code and exposed through the duck-typed `params()` hook —
  so it is settable by name via the inspect protocol and the MEX `param_*`
  commands.

Not supported: algebraic states (`nz > 0`), sparse CasADi arguments,
stage-0-specific constraint rows, and a few rare acados features (the tool
raises a descriptive error).

The generated problems are exactly the examples of chapters 5–6 in
generated form: `examples/pendulum_on_cart/generated/PendulumGen.hpp` is
an `IRK` Gauss-Legendre-4 problem with a Gauss-Newton Hessian
(`has_dynamics_hess = false`), and `examples/unicycle/`,
`examples/masses_chain/`, `examples/p5probe_erk/`, `examples/p5probe_irk/`
cover other combinations. The `compare.sh` scripts in the example
directories build the acados reference and the ocp++ port and report the
cross-solver differences.

## 7.8 Real-time iteration (RTI) solver

For embedded or tightly-timed settings where a *fixed, bounded* computation
time matters more than full convergence, `ocp::RtiSolver<P>`
(`include/ocp/solvers/acados/rti.hpp`) implements the standard
real-time-iteration SQP: one Newton step per `solve()` call — no line
search, no convergence test, no iteration loop.

A single `solve()` call performs two phases:

- `prepare` — assembles the QP **LHS** (Hessians + Jacobians, with a fixed
  Levenberg-Marquardt shift) at the current iterate;
- `feedback` — assembles the **RHS** (gradients + offsets) from the current
  state, solves the QP once (in any `QpMode`, section 7.5), and applies the
  full primal step (`alpha = 1`) with a full-step dual update.

```cpp
ocp::RtiOptions rti;                       // all defaults: staged QP, mu = 0
ocp::RtiSolver<MinimalProblem> rti_solver(rti);
// ... build/resize sol as in chapter 3 ...
ocp::Status st = rti_solver.solve(problem, sol);   // one QP, full step
```

`RtiOptions` carries its own copies of the relevant driver options:
`compute_hess_dyn` / `compute_hess_constr`, `levenberg_marquardt`,
`qp_mode` + `qp_cond_N`, `qp_warm_start` (HPIPM warm-start mode, as in
`HpipmOptions::warm_start`) / `warm_start_first_qp`, and the QP tolerance
strategy. `rti_solver.statistics()` returns an
`ocp::RtiStatistics` with the pre-/post-step NLP residuals of the last call
(populated when `rti_log_residuals` is set) and a running `n_calls` count.

The shared LHS/RHS stage assembly used by both `AcadosSolver` and
`RtiSolver` lives in `solvers/acados/assemble.hpp`; the cost and
slack-penalty helpers are shared via `solvers/acados/cost.hpp`.

## 7.9 NLP-level scaling (`ScaledProblem`)

When your states/inputs span very different magnitudes (e.g. position in
metres vs. an angle in radians vs. a torque in newton-metres), the solver
converges faster when all variables are O(1). `ocp::ScaledProblem<Base>`
(`include/ocp/scaling.hpp`) presents any problem in diagonally scaled
coordinates — no changes to your problem class are needed:

```cpp
DoubleIntegrator raw;                         // your problem in natural units
ocp::ScaledProblem<DoubleIntegrator> scaled(raw);
scaled.sx_ << 1.0, 0.1;                        // scale the states
scaled.su_ << 0.5;                             // scale the controls
```

The wrapper holds `raw` by reference (keep it alive) and implements the
full `ocp::Problem` interface: every model call is evaluated in
unscaled coordinates and the values/Jacobians/Hessians/bounds are mapped
to scaled units (values stay invariant, gradients scale with `D^-T`,
Hessians with `D^-T H D^-1`, box bounds and their soft-penalty rows scale
with the variable, nonlinear soft penalties stay unchanged). Feed it to
any solver exactly like the raw problem:

```cpp
ocp::Solution<ocp::ScaledProblem<DoubleIntegrator>> sol(N);
// warm-start sol in scaled units ...
ocp::AcadosSolver<ocp::ScaledProblem<DoubleIntegrator>> solver(opts);
auto st = solver.solve(scaled, sol);          // st: kSolved, sol in scaled units

auto sol_raw = scaled.unscale(sol);           // back in natural units
// sol_raw.x[k], sol_raw.u[k], all KKT multipliers, cost_value, status
```

`unscale()` recovers the complete solution in the original units: the
primal is divided by the scaling entries, the dynamics and box multipliers
are multiplied back by them (they are co-variable), and the
(in)equality/linear multipliers are copied unchanged. The scaling vectors
are public, user-managed members (all entries must be strictly positive);
changing them takes effect on the next model call.

> **Tip.** The solver's own *QP-level* scaling (section 7.5) operates on
> the assembled QP and is a different, complementary mechanism; NLP-level
> scaling rescales the model itself before the QP is ever assembled.

The companion example demonstrates this in `nlp_scaling()` (Part 5):
it wraps the chapter's double integrator, solves the scaled problem,
recovers the solution in natural units with `unscale()`, and compares
against a direct (unscaled) solve — the two agree to solver tolerance.

## 7.10 Where to go from here

- `tests/` contains a unit test per feature (options, warm start,
  funnel, SOC, timeout, scaling, integrators, DAE autodiff, ...) — the
  fastest reference for the exact semantics of an option.
- `docs/ARCHITECTURE.md` documents the design principles.
- By-name option/statistics discovery exists via the inspect protocol
  (chapter 3.3, `include/ocp/utils/inspect.hpp`) and is exposed to MATLAB
  through the MEX `options` / `opt_get` / `opt_dump` / `stats` commands
  (chapter 8).
- Not yet part of the interface (anticipated extensions): parameter
  sensitivities and algebraic (eliminated) variables.

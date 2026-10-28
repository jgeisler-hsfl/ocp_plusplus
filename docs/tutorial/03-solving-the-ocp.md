# Chapter 3 — Solving the OCP

This chapter covers the solver half of `examples/tutorial/minimal.cpp`:
attaching the SQP solver, configuring it, and interpreting the result.

## 3.1 The SQP solver

`ocp::AcadosSolver<P>` (in `include/ocp/solvers/acados/acados.hpp`) is a
sequential quadratic programming (SQP) driver: at each iteration it
linearizes the problem at the current iterate, assembles a quadratic
program (QP), solves the QP with the built-in HPIPM-style interior-point
solver (`ocp::HpipmQpSolver`), takes a (line-searched) step, and repeats
until the KKT residuals are small.

The full template is

```cpp
ocp::AcadosSolver<P, NH, QpSolver, Regularizer, Globalizer>
```

with defaults for everything except the problem type `P`: `NH` is the
horizon (runtime by default, chapter 7), `QpSolver` the QP backend,
`Regularizer` the Hessian regularization and `Globalizer` the line search.
You normally use the defaults.

## 3.2 The solution container

`ocp::Solution<P>` holds everything a solver consumes and produces. The
same object is used as the initial iterate (*warm start*) and as the
result:

```cpp
const int N = 10;  // horizon (runtime quantity)
ocp::Solution<MinimalProblem> sol(N);   // constructor allocates all trajectories

// initialize the iterate
sol.x[0] = problem.initial_state();
for (int k = 0; k < N; ++k)
{
    sol.u[k].setZero();
    sol.x[k + 1] = problem.dynamics_next_state(k, sol.x[k], sol.u[k]);
}
```

Its members:

| Member | Size | Meaning |
|--------|------|---------|
| `N` | — | horizon (the definitive one; the trajectories have its extents) |
| `x` | `N + 1` | state trajectory |
| `u` | `N` | control trajectory |
| `lambda_dyn` | `N` | dynamics (costate) multipliers |
| `lambda_ineq_stage`, `lambda_eq_stage`, `lambda_lin_stage` | `N` | stage constraint multipliers (per group) |
| `lambda_ineq_term`, `lambda_eq_term`, `lambda_lin_term` | 1 | terminal constraint multipliers |
| `lambda_box_state` | `N + 1` | state-box multipliers, layout `[lower; upper]` (one pair per active box row) |
| `lambda_box_control` | `N` | control-box multipliers, same layout |
| `cost_value` | — | objective at `(x, u)` (includes the solver-internal slack penalty of soft constraints) |
| `status` | — | exit status, see section 3.5 |

The multipliers are zero until a solver fills them; they are warm-started
when you reuse the same `Solution` for the next solve.

## 3.3 Options

Solver options live in `ocp::AcadosOptions`, a plain struct with defaults;
you set what you need and pass it to the constructor:

```cpp
ocp::AcadosOptions opts;
opts.print_level = 1;  // print one row per SQP iteration
ocp::AcadosSolver<MinimalProblem> solver(opts);
```

The most commonly used options:

| Option | Default | Meaning |
|--------|---------|---------|
| `max_iter` | 20 | SQP iteration limit |
| `tol_stat`, `tol_eq`, `tol_ineq`, `tol_comp` | 1e-8 | convergence tolerances on the four KKT residual norms (section 3.6) |
| `print_level` | 0 | 0 = silent, 1 = per-iteration rows on stdout, 2 = also dump the per-QP HPIPM stat table |
| `qp_mode` | `kStaged` | how the staged QP is solved: staged, fully condensed, or block-condensed (chapter 7.5) |
| `levenberg_marquardt` | 0.0 | Hessian damping `mu` (chapter 5) |

The remaining options, all with sensible defaults (grouped by topic):

- **Residuals / termination:** `tol_min_step_norm` (exit when steps become
  tiny), `tol_unbounded` (exit when the cost drops below this), `tau_min`
  (complementarity floor in `res_comp`), `eval_residual_at_max_iter`.
- **Hessian gates:** `compute_hess_dyn`, `compute_hess_constr` (chapter 5).
- **Adaptive Levenberg-Marquardt:** `with_adaptive_lm`, `adaptive_lm_mu0`,
  `adaptive_lm_lam`, `adaptive_lm_mu_min`, `adaptive_lm_obj_scalar`
  (chapter 5).
- **QP warm starting:** the QP solver's `warm_start` option (0 = cold,
  2 = primal+dual, 3 = hot; by name: `qp.options.warm_start`) and
  `warm_start_first_qp` (chapter 7).
- **QP condensing:** `qp_cond_N` — number of condensation blocks used by
  `kPartialCondensed` (0 = fall back to `kStaged`); see chapter 7.5.
- **Timeout:** `timeout_max_time` (wall-clock budget in seconds, 0 = none),
  `timeout_heuristic` (per-iteration duration estimate: 0 = ZERO, 1 =
  LAST, 2 = MAX, 3 = AVERAGE).
- **QP scaling:** the live scaling options on the solver's QP scaler,
  `scaling.options.scale_objective`, `scaling.options.scale_constraints`
  and their bounds (chapter 7).
- **NLP-level scaling:** if your states/controls span very different
  magnitudes, wrap the problem in `ocp::ScaledProblem` to rescale the
  model itself before any QP is assembled (chapter 7.9).
- **Per-iteration QP tolerances:** `nlp_qp_tol_strategy` (0 = fixed, 1 =
  from the NLP residuals, 2 = from the scaling factors) and its parameters
  (chapter 7).
- **Line search (globalization):** nested in `opts.glob`: `alpha_min`,
  `alpha_reduction`, `full_step_dual`, `use_soc` (second-order
  correction), `eps_sufficient_descent` (chapter 7).

## 3.4 Solving

```cpp
const ocp::Status st = solver.solve(problem, sol);
```

`solve()` advances `sol` in place: on a successful exit it contains the
converged trajectory (and multipliers), plus `sol.status` and
`sol.cost_value`. Nothing else in the numerical core allocates: the solver
sizes its internal workspaces once for the horizon of `sol`.

Per-iteration records are available afterwards via
`solver.statistics()` (`ocp::AcadosStatistics`), and can be exported as
CSV:

```cpp
solver.statistics().write_csv("sqp_stat.csv");
```

## 3.5 Exit status

`ocp::Status` values you may see:

| Status | Meaning | What to do |
|--------|---------|------------|
| `kSolved` (1) | all four residual norms below their tolerances | done |
| `kMaxIterations` (2) | iteration limit reached | increase `max_iter`, loosen tolerances, or warm start |
| `kInfeasible` (3) | (in)feasibility detected | check the constraints (e.g. conflicting bounds) |
| `kQpFailure` (4) | inner QP solve failed | inspect `solver.statistics()` rows; consider regularization (chapter 5) |
| `kMinStep` (5) | step length fell below `tol_min_step_norm` | typically a stall: check scaling, Hessian model |
| `kUnbounded` (6) | cost below `tol_unbounded` | problem likely unbounded (missing constraints) |
| `kNanDetected` (7) | NaN in the residuals | check the model functions for invalid evaluations |
| `kAborted` (8) | internal solver error | — |
| `kTimeout` (9) | wall-clock budget exceeded | raise `timeout_max_time` |
| `kNotInitialized` (10) | `solve()` called before `init()` | call `init()` first |

For log output, `ocp::to_string(sol.status)` returns the enumerator
spelling (e.g. `"kSolved"`).

## 3.6 Convergence: the four residual norms

The solver measures convergence with four infinity norms
(`ocp::NlpResiduals`, also available standalone via
`ocp::compute_nlp_residuals(problem, sol)`):

| Norm | Measures |
|------|----------|
| `res_stat` | stationarity: max inf-norm of the Lagrangian gradient w.r.t. `(u_k, x_k)` (the `x_0` part is omitted when `x_0` is fixed) |
| `res_eq` | dynamics gap: `max_k ||f_k(x_k, u_k) - x_{k+1}||_inf` |
| `res_ineq` | max positive violation over all constraint rows (0 if feasible) |
| `res_comp` | complementarity: how well `lambda * violation = 0` holds on the active sides |

Converged means all four are below their tolerances simultaneously.

## 3.7 Run and interpret

```bash
./build/tutorial_minimal
```

The solver output (with `print_level = 1`) is one row per iteration:

```
  # it     res_stat       res_eq     res_ineq     res_comp   qp_stat   qp_iter    step_norm      alpha
     0   3.0000e+01   0.0000e+00   0.0000e+00   0.0000e+00         1         3     1.09e+01   1.00e+00
     1   2.9088e-14   2.2204e-16   0.0000e+00   0.0000e+00         1         3     1.09e+01   1.00e+00

status=1 (kSolved=1) cost=4.215141256191 SQP iters=2
```

Reading it:

- `# it` — SQP iteration; `res_*` — the four norms at the iteration start;
  `qp_stat` / `qp_iter` — exit status and iteration count of the inner QP
  solve; `step_norm` — max inf-norm of the primal step (including slacks);
  `alpha` — accepted line-search step length.
- Iteration 0 starts far from stationary (`res_stat = 30`); the first QP
  step already reaches the optimum of this convex problem (`alpha = 1`,
  large step). Iteration 1 confirms convergence (`res_stat ~ 1e-14`).
- `status=1` is `kSolved`, `cost=4.2151...` is the objective at the
  solution.

The printed trajectory shows the expected behavior of an unconstrained
double integrator with a terminal cost pulling `(q, v)` toward the origin:
brake hard first (`a = -10.9`), then accelerate to carry the position
closer to zero by the horizon end — the classic LQR-like trade-off between
the velocity cost and the terminal position cost.

## 3.8 Multipliers

After a solve, the KKT multipliers in `sol` tell you *why* the optimum
looks the way it does:

- `sol.lambda_dyn[k]` — the costate at stage `k` (the sensitivity of the
  cost to the state).
- `sol.lambda_box_state[k]` / `sol.lambda_box_control[k]` — one entry per
  active box row and side, layout `[lower; upper]`. A large upper entry
  means the corresponding upper bound is active and pushing.
- `sol.lambda_ineq_stage[k](j) >= 0` — multiplier of inequality row `j`
  (`g <= 0`, nonneg by convention); `lambda_eq_*` and `lambda_lin_*` are
  signed net multipliers of the two-sided rows.

They are also the warm start for the next solve (e.g. in a receding-horizon
scheme, chapter 7).

## 3.9 Exporting and plotting the solution

`ocp::Solution::to_csv` writes the primal trajectories to a CSV file:

```cpp
sol.to_csv("solution.csv");  // returns bool; false if the file cannot be opened
```

The file has the header `k` followed by one column per state and control
component — the column names come from the problem's `Dims` tag
(`state_names` / `control_names`, chapter 2; defaults `x_i` / `u_i`) — and
one row per stage `k = 0..N`, with values in scientific notation. The
terminal row leaves the control columns blank (there is no `u_N`). The
objective value and the KKT multipliers are not included in the file.

To plot the trajectories, the repository ships `tools/plot_solution.py`, a
small script built on [plotly](https://plotly.com/python/) that draws one
subplot per variable (shared x-axis over the stage index `k`):

```bash
pip install plotly        # user-installed; not needed for the C++ build
python3 tools/plot_solution.py solution.csv             # -> solution.html
python3 tools/plot_solution.py solution.csv -o out.png  # static PNG
```

The HTML output is self-contained and can be opened in any browser; the PNG
export additionally requires kaleido (`pip install plotly kaleido`). Blank
terminal-row cells are drawn as gaps in the control traces. The
`examples/double_integrator` program demonstrates the feature: it checks
the component names and writes `double_integrator_solution.csv`.

## What's next

Chapter 4 adds constraints to the problem — box, linear, nonlinear and
soft.

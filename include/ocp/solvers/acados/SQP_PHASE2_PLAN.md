# Phase 2 plan: SQP driver (NLP level)

Companion to `SQP_PLAN.md` (§3, §7–§9, §11 Phase 2). This file is the
working reference for the phase: the refined sub-step plan, the
conventions this phase nails down, and the running checkpoints (append a
checkpoint entry per sub-step, same protocol as `hpipm/SQP_PHASE1_WORKLOG.md`).

**Protocol for every sub-step:**
1. Re-read the acados sources listed for the sub-step (fresh; note line
   numbers in the checkpoint).
2. Implement header-only in `include/ocp/solvers/acados/`.
3. Build (`cmake --build build`), run all tests, warning-free under
   `-Wall -Wextra -Werror`.
4. Append a checkpoint entry below (what was done, test results, any
   deviation from this file — deviations win; update the entry here).
5. One commit per sub-step (`solver:` / `test:` scope).

## 0. State and file layout

- Phase 1 done: `solvers/hpipm/qp.hpp` (QpDim / Qp / QpSol / QpRes,
  stage-type layouts), `solvers/hpipm/hpipm.hpp` (`HpipmQpSolver<P, NH>`,
  relative-formulation IPM, verified against the dense full-KKT oracle).
- All new code of this phase lives in **`include/ocp/solvers/acados/`**
  (project decision 2026-10-01): the NLP-level, acados-derived components.
  The QP-level components stay in `solvers/hpipm/`.
  - `acados/regularize.hpp` — `GlmRegularizer<P>`, `NoRegularizer<P>` (2a)
  - `acados/globalize.hpp`  — `GlobOptions`, `SqpSlacks<P, NH>`,
    NLP↔QP dual mapping, `apply_sqp_step`, `MeritBacktracking<P, NH>` (2b, 2d)
  - `acados/sqp.hpp`        — `NlpResiduals` / `compute_nlp_residuals` (2c),
    `SqpOptions` / `SqpIteration` / `SqpStatistics` (2e),
    `SqpSolver<P, QpSolver, Regularizer, Globalizer, NH>` (2f, 2g)
- Header-only, C++17, `ocp::` namespace, include order per AGENTS.md.
  Include chain: `sqp.hpp` → `globalize.hpp` → `regularize.hpp` →
  `hpipm/hpipm.hpp` → `hpipm/qp.hpp` → `problem.hpp`. No cycle.

## 1. Conventions nailed down by this phase

SQP_PLAN §11: "This phase nails down the sign/factor conventions from §2."
The concrete decisions (each re-verified against source in 2i):

### 1.1 Step-formulation QP (HPIPM-faithful)

The staged QP variables are **steps** `z_k = (δu_k; δx_k; s_k)` /
`z_N = (δx_N; s_N)`; `b_k = f_k(x̄_k, ū_k) − x̄_{k+1}` (dynamics residual at
the current iterate, as already defined in `qp.hpp`). At the QP solution:
`δx_{k+1} = BA_k z_k + b_k`. NLP update: `x_new = x̄ + α·δx`.
(The Phase-1 synthetic tests pin `z_j = x_0` with `d = ±x_0` because their
synthetic iterate is the zero point; that is a data choice of the oracle
test, not a different formulation.)

### 1.2 Pin rows (fixed x_0)

Row value `v = δx_0j`; bounds `lo = hi = x_0j`; offsets
`d_lo = x_0j − x̄_0j`, `d_hi = −(x_0j − x̄_0j)`. The NLP driver always keeps
`x̄_0 = initial_state()`, so in practice `d = 0`. Assembly computes the
offset generically (and `assert`s `|x̄_0 − x_0|` is tiny). The `qp.hpp`
header note "pin rows: d_lo = x_0" is corrected in 2f to the offset form.

### 1.3 NLP Lagrangian and dual mapping

NLP Lagrangian (sign convention used by `compute_nlp_residuals` and the
dual update):

```
L = Σ_k L_k(x_k, u_k) + L_N(x_N)
  + Σ_k π_kᵀ (x_{k+1} − f_k(x_k, u_k))
  + Σ_k λ_g,kᵀ g_k(x_k, u_k)          (g ≤ 0, λ ≥ 0)
  + Σ_k μ_kᵀ e_k(x_k, u_k)            (signed net multiplier)
  + box/linear terms:  λ_lo(lo − w) + λ_hi(w − hi)   (λ ≥ 0 per side)
```

Mapping between `Solution` multipliers and the QP's per-side multipliers
(`QpSol.lam_*` sides; the QP `pi` is the *negative* of the NLP dynamics
multiplier, because `res_g` carries `+BAᵀπ_qp − pin(π_qp, prev)`):

| NLP (`Solution`)                    | QP side                     |
|-------------------------------------|-----------------------------|
| `lambda_dyn[k]` (π)                 | `pi_qp[k] = −π`            |
| `lambda_ineq_stage[k][r]` (λ ≥ 0)   | `lam[side_hi(r)]`          |
| `lambda_eq_stage[k][r]` (signed μ)  | `μ = lam[side_hi] − lam[side_lo]` |
| `lambda_lin_stage[k][r]` (signed)   | `net = lam[side_hi] − lam[side_lo]` |
| `lambda_box_state[k]` [lower; upper]| lower ↔ `lam[side_lo]`, upper ↔ `lam[side_hi]` |
| `lambda_box_control[k]`             | same, control box rows      |
| terminal variants                   | terminal stage rows, same rules |

Verification gates: (a) finite-difference Lagrangian gradient at a
converged solution (2h), (b) NLP↔QP residual consistency: warm the QP
duals from the NLP multipliers, the QP's initial `res_g` must equal the
NLP `res_stat` at the same point (2c/2f cross-check).

### 1.4 `res_comp` adds `tau_min`

acados `ocp_nlp_res_compute` (:3792–3844): the comment says
`− tau_min·ones` but the code **adds** `tau_min·ones` — matched as-is
(SQP_PLAN §2 resolved-quirks). Equality rows (our `eq` group) and pin
rows are zeroed in `res_comp` (acados `idxe` masking); their violation is
counted by `res_ineq` instead.

### 1.5 Dynamics Hessian term

With the §1.3 Lagrangian, the dynamics contribution to the stage Hessian
is **minus** the HVP: `H_k += −M_dyn` where
`M_dyn·v = dynamics_hess_prod(k, x̄, ū, w = lambda_dyn[k], v_x, v_u)`.
(Equivalent to acados's `+HVP` because acados's residual sign
`f − x_{k+1}` makes its π the negative of ours.) Constraint HVPs enter
with **+**: `H_k += M_ineq (w = λ_ineq_stage[k])`,
`H_k += M_eq (w = λ_eq_stage[k] net)`, terminal state-only analogues.
HVP matrices are built from `nx + nu` (terminal: `nx`) unit-vector calls.

### 1.6 Slack penalty

`0.5·w·s²` per soft row's slacks; HPIPM `Z = w`, `grad` slack part = 0
(SQP_PLAN §12.3, resolved in phase 0). `Solution.cost_value` **includes**
the slack penalty term (the problem cost functions exclude slacks).

## 2. Sub-steps

### 2a — `regularize.hpp`

Sources: `acados/ocp_nlp/ocp_nlp_reg_glm.c:66–283`,
`acados/ocp_nlp/ocp_nlp_reg_noreg.c`, `acados/utils/math.c:1145–1170`
(Gershgorin estimate), `acados/docs/algorithm/04-regularization-and-qpscaling.md` Part A.

Duck-typed 2-method contract (mirrors `ocp_nlp_reg_config`, minus the
RTI-only `regularize_lhs` / `regularize_rhs` split):

```cpp
void regularize(Qp<P, NH>& qp) const;
void correct_dual_sol(Qp<P, NH>&, QpSol<P, NH>&) const;
```

- `GlmRegularizer<P>`: `double epsilon = 1e-6;` Per stage type
  (first, each path, term): take the **(u;x) block** of `hess`
  (first/path: leading `(nu+nx)×(nu+nx)`; term: the whole matrix).
  Gershgorin min-eigen estimate `tmp = min_i (a_ii − Σ_{j≠i} |a_ij|)`.
  If `tmp < epsilon`: shift the block by `α·I` with
  `α = |tmp| + epsilon` if `tmp < 0` else `epsilon`. Slack diagonal and
  slack off-diagonals are **not** touched.
- `correct_dual_sol`: no-op for both GLM and NoReg (doc 04: a pure
  Hessian shift leaves the duals valid).
- `NoRegularizer<P>`: both no-ops.

Test: `tests/sqp/regularize_2a.cpp` — indefinite (u;x) block is shifted
by `|tmp|+ε`; a PSD block with min-eig > ε is untouched; slack diagonal
unchanged; `NoRegularizer` is a no-op.

### 2b — `globalize.hpp`: options, slacks, step application

Sources: `acados/ocp_nlp/ocp_nlp_globalization_common.h:39–148`,
`acados/ocp_nlp/ocp_nlp_common.c:3355–3414` (`ocp_nlp_update_variables_sqp`).

- `struct GlobOptions { double alpha_min = 0.05, alpha_reduction = 0.7,
  eps_sufficient_descent = 1e-4; bool full_step_dual = false; };`
  (SOC and sufficient-descent options are phase 3, per SQP_PLAN §8.)
- `SqpSlacks<P, NH>` — solver-internal slack storage (first / path /
  term vectors, `resize(int)`, `setZero()`). Lives here (not in
  `Solution` — slacks are solver-internal per AGENTS.md) and is shared
  with the merit strategy (deviation from the SQP_PLAN §8 contract,
  which lacks a slack argument; refinement recorded here).
- `namespace detail` — the §1.3 dual mapping as
  `map_qp_duals_to_solution<P, NH>(const QpSol<P, NH>&, Solution<P>&)`.
- `template <class P, int NH> void apply_sqp_step(const Solution<P>& start,
  const QpSol<P, NH>& step, double alpha, bool full_step_dual,
  Solution<P>& dest)`:
  - primal: `dest.u[k] = start.u[k] + α·(u part of step.ux_k)`;
    `dest.x[k+1] = start.x[k+1] + α·(x part of step.ux_k)`;
    `x_0` untouched when `fixed_initial_state` (pinned; the step's x_0
    part is 0 anyway).
  - duals: `full_step_dual` → `dest.lambda_* = mapped(step)` (absolute);
    else relaxed `dest.lambda_* = (1−α)·start.lambda_* + α·mapped(step)`,
    with the §1.3 mapping (incl. the dynamics sign flip and the
    lo/hi → [lower; upper] box split).
  - Slacks are updated by the driver: `s ← s + α·δs` (analogous to
    acados's algebraic `z` update; kept out of `Solution`).

Test: `tests/sqp/apply_step_2b.cpp` — hand-computed tiny problem: primal
interpolation, relaxed vs full dual update, every §1.3 mapping row
incl. the `lambda_dyn = −pi` sign flip.

### 2c — NLP residuals

Sources: `acados/ocp_nlp/ocp_nlp_common.c:3743–3846`
(`ocp_nlp_res_compute`), `acados/ocp_nlp/ocp_nlp_sqp.c:351–430`
(`check_termination` tolerance usage).

```cpp
struct NlpResiduals { S res_stat, res_eq, res_ineq, res_comp; };
template <class P, int NH>
NlpResiduals compute_nlp_residuals(const P& problem, const Solution<P>& sol);
```
(free function in `sqp.hpp`; the driver's member wraps it)

- **res_stat**: per stage `k`, the `(nu+nx)` vector
  `∂L/∂u_k = ∇_u L_k − B_kᵀ π_k + G_uᵀλ_g + E_uᵀμ + B_linᵀλ_net + (−λ_boxu_lo + λ_boxu_hi)`
  and `∂L/∂x_k = ∇_x L_k − A_kᵀ π_k + (k>0 ? π_{k−1} : 0) + G_xᵀλ_g + E_xᵀμ
  + A_linᵀλ_net + (−λ_boxx_lo + λ_boxx_hi)` (§1.3 signs); terminal:
  `∇_x L_N + π_{N−1}` + terminal constraint adjoints. Inf-norm per stage,
  max over stages. Jacobians come from the `*_jacobian` interface calls.
- **res_eq**: `max_k ‖f_k(x_k, u_k) − x_{k+1}‖∞` (dynamics gap only —
  nonlinear-equality violations are counted in `res_ineq`, acados
  parity: equalities are two-sided ineq rows there).
- **res_ineq**: max over all stages/terminal and all rows of the
  *positive* violations: ineq `max(0, g)`; eq `|e|` (both sides);
  box/lin `max(0, v − hi)`, `max(0, lo − v)`. 0 when feasible.
- **res_comp**: per stage, over every active side:
  hi side `λ_hi·max(0, v − hi)`, lo side `λ_lo·max(0, lo − v)` with the
  **current iterate** multipliers from `Solution` (not the merit
  weights); `eq` rows and pin rows zeroed (§1.4); then
  `+= tau_min` (acados quirk, §1.4); inf-norm per stage, max over
  stages. `tau_min` is a `SqpOptions` field (default: verify against
  `ocp_nlp_opts` at implementation time; expect 1e-16).

Test: `tests/sqp/residuals_2c.cpp` — hand-computed values on a small
deliberately-infeasible iterate: all four norms checked exactly, incl.
the `tau_min` add and the eq-row zeroing.

### 2d — `MeritBacktracking`

Sources: `acados/ocp_nlp/ocp_nlp_globalization_merit_backtracking.c:1–938`
(esp. :148–288 merit gradient [phase 3], :290–408 line search,
:606–755 merit + weights), `acados/docs/algorithm/03-globalization.md`.

```cpp
template <class P, int NH = Eigen::Dynamic>
struct MeritBacktracking
{
    GlobOptions opts;
    // status (mutable, per-solve): pi_w (N state vectors) + per-group
    // weights w_ineq (ng), w_eq (ne), w_lin (nl), w_box_state (2*nbx),
    // w_box_control (2*nbu), terminal w_ineq_t (ng_t), w_eq_t (ne_t),
    // w_lin_t (nl_t), w_box_term (2*nbx_t)

    // driver calls this once at iter 0 (before any QP solve):
    void initialize(const P& problem, const Solution<P, NH>& sol);

    // refined contract (SQP_PLAN §8 + SqpSlacks argument, see 2b):
    Status find_acceptable_iterate(const P& problem, Solution<P, NH>& cur,
                                   const QpSol<P, NH>& step,
                                   const SqpSlacks<P, NH>& slacks,
                                   Solution<P, NH>& scratch,
                                   double& alpha) const;
};
```

- **Merit** (trial iterate `(x, u, s)`):
  `merit = Σ_k L_k(x_k, u_k) + L_N(x_N)          (problem cost fns)
         + 0.5·Σ_k w_soft·s²                     (solver-side slack penalty, §1.6)
         + Σ_k |π_w,k|·‖f_k(x_k, u_k) − x_{k+1}‖₁   (weighted L1 dynamics gap)
         + Σ_rows |λ_w|·violation                 (only positive violations:
                                                    ineq max(0,g); eq |e|;
                                                    box/lin max(0,v−hi), max(0,lo−v))`
- **Weights** (`:705–755`): iter 0: `π_w ← |−qp.pi|`,
  `λ_w ← mapped qp lam` (ineq/box sides as-is ≥ 0; eq/lin `|net|`).
  Every later iter (at the start of the line search):
  `w ← elementwise_max(|qp dual|, 0.5·(|w_old| + |qp dual|))`.
- **Line search** (`ocp_nlp_line_search`, `:290–408`, match the loop
  bound exactly): `merit0 = merit(cur)`; `α = 1`;
  `while (α·reduction > alpha_min)`: trial `x/u = cur + α·step_xu`
  (in `scratch`), trial `s = slacks + α·δs`; if `merit1 < merit0` and
  finite → accept, `alpha = α`, return `kSolved`; else `α *= reduction`.
  Exhausted: `alpha = α` (post-shrink value, acados quirk — the driver
  still applies it, `:894`), return `kMinStep` (`kNanDetected` if the
  last merit was NaN/Inf). No SOC pre-pass, no sufficient-descent
  (phase 3).
- State is `mutable` (the contract is `const`-callable, per plan §8);
  `initialize` is called once per `solve()` at iteration 0.

### 2e — options, statistics, printing

```cpp
struct SqpOptions {
    int max_iter = 20;
    double tol_stat = 1e-8, tol_eq = 1e-8, tol_ineq = 1e-8, tol_comp = 1e-8;
    double tol_min_step_norm = 1e-12;
    double tol_unbounded = -1e10;
    bool compute_hess = true;        // gates the HVP terms (§1.5)
    double levenberg_marquardt = 0.0;
    bool with_adaptive_lm = false;   // phase 3 (option kept, inert in v1)
    int print_level = 0;             // 0 silent, 1 per-iteration rows
    int qp_warm_start = 0;           // v1: HPIPM cold-starts each QP (inert)
    bool warm_start_first_qp = false;// v1: inert
    bool eval_residual_at_max_iter = false;
    double timeout_max_time = 0.0;   // phase 3
    bool scale_qp_objective = false, scale_qp_constraints = false;  // phase 3
    double tau_min = 1e-16;          // res_comp shift (§1.4); verify default
    GlobOptions glob;
};

struct SqpIteration { double res_stat, res_eq, res_ineq, res_comp;
                      int qp_status, qp_iter;
                      double step_norm, alpha; };

struct SqpStatistics {
    int iter = 0;
    Status status = Status::kUnset;
    std::vector<SqpIteration> rows;
    void record(const SqpIteration&);
    void write_csv(const char* path) const;  // header + one row per iter
};
```

CSV columns: `iter,res_stat,res_eq,res_ineq,res_comp,qp_status,qp_iter,
step_norm,alpha` (mirrors the acados `stat` matrix, doc 01 §4, with the
QP columns of row `i` belonging to the QP solved in iteration `i`).
`print_level > 0`: header line + one formatted row per iteration on
stdout (acados `print_iteration` analog, `ocp_nlp_sqp.c:449`).

### 2f — QP assembly (`SqpSolver::assemble_qp` + `add_lm_term`)

Sources: `acados/ocp_nlp/ocp_nlp_common.c:3079–3203`
(`approximate_qp_matrices` / `approximate_qp_vectors_sqp`),
`:3034–3059` (LM term), `acados/ocp_nlp/ocp_nlp_constraints_bgp.c`
(slack sign in the Jacobian, verified in phase 0),
`acados/docs/algorithm/02-qp-approximation.md`.

`void assemble_qp(const P& problem, Solution<P, NH>& sol)`:

Per stage (first k=0 / path k=1..N−1 / term k=N):

1. **hess** — start from the permuted cost Hessian
   (`[H_uu H_ux; H_xu H_xx]`, from `stage_cost_hessian` /
   `terminal_cost_hessian` in `[x;u]` layout);
   if `opts.compute_hess && P::has_dynamics_hess_prod`:
   `hess -= M_dyn` (§1.5, `nx+nu` unit-vector HVP calls with
   `w = sol.lambda_dyn[k]`);
   if `opts.compute_hess && P::has_constr_hess_prod`:
   `hess += M_ineq (w = lambda_ineq_stage[k])`,
   `hess += M_eq (w = lambda_eq_stage[k])`; terminal: state-only
   analogues with the terminal multipliers.
   Slack diagonal: soft rows → their penalty weight `w`
   (`stage_inequality_constr_soft_penalty` / spec `soft_penalty`);
   unused slack slots → `1.0`; all slack off-diagonals 0.
2. **grad** — permuted cost gradient `[g_u; g_x]` (terminal `[g_x]`);
   slack part 0.
3. **BA / b** (k ≤ N−1 only) — `BA = [B | A]` from
   `dynamics_jacobian`; `b = dynamics_next_state − sol.x[k+1]`.
4. **DC** (nrow × nvar, natural orientation, row layout of `QpDim`):
   pin `e_{idx_x0[r]}`; box `e_{idxb[r]}`; ineq/eq/lin rows
   `[d(·)/du | d(·)/dx]` from the `*_jacobian` calls — **full Jacobian
   incl. the x_0 part at the first stage** (the x_0 step variable is a
   real, pin-row-constrained decision variable, acados parity); soft
   rows get the slack column: `+1` (lo side), `−1` (hi side).
5. **d / d_mask** (offset form, §1.1–1.2): general rule for a row with
   current value `w_cur` and bounds `(lo, hi)`: `d_lo = lo − w_cur`,
   `d_hi = −hi + w_cur`, `d_mask = 1` iff the bound is finite
   (`std::isfinite`); ineq rows have `w_cur = g(x̄, ū)`, `hi = 0`, no lo
   side; eq rows `lo = hi = 0`; pin rows per §1.2; slack sides
   `d = 0`, `d_mask = 1`. `m` stays 0 (v1).
6. **First-stage degeneracy check**: if `fixed_initial_state` and a
   first-stage ineq/lin row is violated at the iterate and its Jacobian
   w.r.t. `(u_0; s_0)` is identically zero (it depends only on the
   pinned x_0) → the problem is infeasible; `solve()` returns
   `kInfeasible` (record in checkpoint; double_integrator stage-0 rows
   are exactly this shape but feasible, so they pass).

`add_lm_term(Qp<P, NH>& qp, double mu)`: if
`compute_hess && levenberg_marquardt > 0`, add `mu·I` to the (u;x)
block of every stage Hessian (first/path: leading `(nu+nx)` diagonal;
term: full). (acados scales by the cost module's `scaling` factor; our
interface has no per-stage cost scaling — the examples use 1; recorded
as a deviation in the checkpoint.) Adaptive-μ schedule: phase 3.

### 2g — driver `SqpSolver::solve`

Source: `acados/ocp_nlp/ocp_nlp_sqp.c:474–809` (main loop),
`:351–430` (termination), `acados/docs/algorithm/01-sqp-main-loop.md`.

```cpp
template <class P, class QpSolver = HpipmQpSolver<P, NH>,
          class Regularizer = GlmRegularizer<P>,
          class Globalizer = MeritBacktracking<P, NH>,
          int NH = Eigen::Dynamic>
class SqpSolver
{
public:
    explicit SqpSolver(SqpOptions opts = {});
    Status solve(const P& problem, Solution<P, NH>& sol);
    const SqpOptions& options() const;
    const SqpStatistics& statistics() const;
    const Qp<P, NH>& last_qp() const;
    const QpSol<P, NH>& last_qp_sol() const;
private:
    SqpOptions opts_; QpSolver qp_; Regularizer reg_; Globalizer glob_;
    SqpStatistics stat_;
    Qp<P, NH> qp_in_; QpSol<P, NH> qp_out_;
    Solution<P, NH> trial_; SqpSlacks<P, NH> slacks_;
    double alpha_ = 0.0, step_norm_ = 0.0, cost_value_ = 0.0;

    void assemble_qp(const P&, Solution<P, NH>&);
    void add_lm_term(Qp<P, NH>&, double) const;
    NlpResiduals compute_nlp_residuals(const P&, const Solution<P, NH>&) const;
    double compute_cost(const P&, const Solution<P, NH>&) const; // incl. 0.5 w s²
    Status check_termination(int iter, const NlpResiduals&, double step_norm,
                             double cost) const;
};
```

`solve()` (mirrors `ocp_nlp_sqp.c:541–798`, `iter = 0..max_iter`):

1. resize `qp_in_ / qp_out_ / trial_ / slacks_` to `sol.N`;
   `slacks_` starts from the previous solve's values when the horizon
   matches (warm-started across repeated solves on the same object —
   the acados `nlp_out` slack analog), else zeroed.
2. loop `iter = 0..max_iter`:
   a. `assemble_qp(problem, sol)`;
   b. `res = compute_nlp_residuals(problem, sol)`;
   c. `iter == 0`: `glob_.initialize(problem, sol)`;
   d. `reg_.regularize(qp_in_)` — **before** the termination check, as
      in acados (comment `:599–600`);
   e. `check_termination` (order, doc 01 §2.F): NaN → `kNanDetected`;
      `iter ≥ max_iter && !eval_residual_at_max_iter` →
      `kMaxIterations`; all four norms ≤ tols → `kSolved`;
      `iter > 0 && step_norm < tol_min_step_norm` → `kMinStep`;
      `cost ≤ tol_unbounded` → `kUnbounded`; `iter ≥ max_iter` →
      `kMaxIterations`; (timeout: phase 3). On hit: record stat, set
      `sol.status`, return.
   f. `qp_status = qp_.solve(qp_in_, qp_out_)`; if not in
      `{kSolved, kMaxIterations}` → record, return `kQpFailure`
      (acados: MAXITER from the QP is *acceptable*; refine the plan's
      "kQpFailure or kMaxIterations" wording — follow acados).
   g. `step_norm_` = max over stages of the inf-norm of the primal step
      (acados `ocp_qp_out_compute_primal_nrm_inf`; verify whether slacks
      are included at implementation time);
   h. `gstatus = glob_.find_acceptable_iterate(problem, sol, qp_out_,
      slacks_, trial_, alpha_)`; on non-`kSolved` the globalizer has
      already advanced `sol` by `apply_sqp_step` with the final
      `alpha_` (acados `:894` behavior); record stat, return `gstatus`.
   i. commit: `apply_sqp_step(sol, qp_out_, alpha_,
      opts_.glob.full_step_dual, sol)`; `slacks_ += alpha_·δs`;
      `cost_value_ = compute_cost(problem, sol)`;
      `stat_.record(...)`; print if `print_level > 0`.
3. after the loop: `sol.status = kMaxIterations`, record, return.

`sol.cost_value` (final) includes the slack penalty (§1.6).
`last_qp()` / `last_qp_sol()` expose the final QP (diagnostics, future
RTI). Warm-start options are inert in v1 (HPIPM cold-starts each QP) —
documented in the header.

### 2h — end-to-end tests + CMake

`tests/sqp_unit` target (new, bundles 2a–2c files:
`tests/sqp/regularize_2a.cpp`, `tests/sqp/apply_step_2b.cpp`,
`tests/sqp/residuals_2c.cpp`).

`tests/sqp_double_integrator/sqp_double_integrator.cpp`
(target `sqp_double_integrator`), N = 10:
- warm start: `x[0] = initial_state()`, forward-simulate with `u = 0`
  (adjust if infeasible start makes the merit line search stall),
  zero multipliers;
- solve with default `SqpSolver<DoubleIntegrator>` (GLM + merit);
- assert `kSolved`;
- **independent** residual check inside the test (recomputed from the
  problem interface, not from the solver): dynamics gap, constraint
  violations, complementarity with the solution multipliers, and
  stationarity via a **finite-difference Lagrangian gradient** at the
  solution (this is the §1.3 sign-convention gate);
- `x[0] == initial_state()`; terminal equality `q_N + v_N ≈ 1`;
  box bounds hold (soft rows: `v ≤ v_max + slack`);
- write `sqp_double_integrator.csv` (via `SqpStatistics::write_csv`)
  and print the stat rows (`print_level = 1` in the test).

`tests/sqp_mass_spring/sqp_mass_spring.cpp` (target `sqp_mass_spring`),
N = 20: same structure (no terminal equality; boxes only), CSV
`sqp_mass_spring.csv`.

CMake: three new targets, `include` + `examples` include dirs,
`-Wall -Wextra -Werror`.

### 2i — acados reference diff + sign-off

- Best effort: build the acados `mass_spring_example`
  (`acados/examples/c/no_interface_examples/mass_spring_example.c` —
  hand-written C model, no CasADi needed in the example itself; the
  acados core needs Eigen + its bundled deps). Extract the reference
  solution (x, u, cost) and compare: cost within 1e-4 relative,
  `(x, u)` trajectories within 1e-4 component-wise (acados SQP
  tolerances are looser than ours by default; adjust if needed and
  record). If the acados core cannot be built in this environment,
  record the blocker, fall back to the independent KKT/FD verification
  of 2h, and flag the reference diff as deferred (do not block the
  phase).
- Sign off every §1 convention: re-check §1.3 against
  `ocp_nlp_common.c` (res_compute + update_variables) and the
  `ocp_nlp_constraints_bgp.c` slack signs; confirm the `tau_min`
  default (§1.4) and the `primal_nrm_inf` slack treatment (§2g.5).
- Update `SQP_PLAN.md` §12 open items with the resolutions; append all
  checkpoints to this file.

## 3. Test matrix

| target | covers |
|---|---|
| `qp_dim`, `qp_unit` (existing) | phase 0/1 regression |
| `sqp_unit` | 2a GLM/NoReg, 2b step application + dual mapping, 2c residual formulas |
| `sqp_double_integrator` | 2e–2g end-to-end: assembly, HVP-free (linear) QP, merit, driver, residuals, CSV |
| `sqp_mass_spring` | end-to-end, larger N, box-only constraints |
| (2i) acados reference | cost / (x,u) parity |

Exit criteria for the phase: all targets build warning-free and pass;
`double_integrator` (N=10) and `mass_spring` (N=20) reach `kSolved` with
all four NLP residual norms below the option tolerances; both CSVs
written; §1 conventions signed off in the checkpoints.

## 4. Checkpoints

### 2a — regularize.hpp (GlmRegularizer / NoRegularizer)

- Re-read `ocp_nlp_reg_glm.c:69,216-258` (epsilon default 1e-6; per-stage
  Gershgorin estimate; `alpha = |tmp|+eps` if `tmp<0` else `eps`;
  `correct_dual_sol` no-op) and `math.c:1145-1170`
  (`compute_gershgorin_min_eig_estimate`: `min_i (a_ii - Σ_{j≠i}|a_ij|)`).
- Implemented `include/ocp/solvers/acados/regularize.hpp`:
  - `gershgorin_min_eig(M)` — free function mirroring math.c:1145.
  - `GlmRegularizer<P, NH>` / `NoRegularizer<P, NH>` — duck-typed 2-method
    contract. `regularize` shifts only the leading (u;x) block of each
    stage Hessian (first/path: (nu+nx)×(nu+nx); term: nx×nx); slack
    diagonal and off-diagonals untouched. `correct_dual_sol` no-op (both).
- **Deviation (wins):** the plan wrote `GlmRegularizer<P>` /
  `NoRegularizer<P>`, but the contract takes `Qp<P, NH>&` /
  `QpSol<P, NH>&`, so both are templated `<P, int NH = Eigen::Dynamic>`,
  mirroring `HpipmQpSolver<P, NH>`.
- Test: `tests/sqp/regularize_2a.cpp` + `tests/sqp/sqp_unit.cpp` main;
  CMake target `sqp_unit`. Cases: indefinite first-stage block shifted by
  exactly |tmp|+ε (DI N=2); PSD block (min-eig 4.9 > ε) bit-exact
  untouched on all stages; zero blocks (tmp=0) shifted by ε only;
  terminal indefinite block (MS, tmp=−2 → all 8 diags +2+ε); 0≤tmp<ε →
  ε-only shift; slack diagonal/off-diagonal sentinels untouched;
  NoRegularizer no-op; `correct_dual_sol` no-op.
- Build: warning-free under `-Wall -Wextra -Werror`. `./build/sqp_unit`
  passes; `double_integrator`, `mass_spring`, `qp_dim`, `qp_unit` all
  still pass.

# Plan: Reimplement the acados SQP + HPIPM QP solver (TODO step 4)

Status: **approved** (2026-09-28). This file is the working reference for the
implementation; `TODO.md` item 4 is the roadmap entry it advances.

## 1. Scope

Port `ocp_nlp_sqp` and the HPIPM primal-dual IPM OCP-QP solver it calls, onto
`ocp::Problem`.

**Out of scope:** RTI SQP, DDP, Anderson acceleration, sensitivities,
partial condensing, algebraic variables (`z`), user constraint masking,
OpenMP, other QP backends.

Reference: `acados/docs/algorithm/00–06` (source-verified; line references
there). Every acados C function maps to exactly one C++ method (see §10).

## 2. Resolved decisions

| # | Decision |
|---|---|
| Q1 | Extend `ocp::Status` with `kQpFailure`, `kMinStep`, `kUnbounded`, `kNanDetected`, `kTimeout` |
| Q2 | QP-internal variable order **(u; x; slacks)** (HPIPM-faithful); `[x;u]`↔`[u;x]` permutation only at the NLP↔QP boundary |
| Q3 | QP Hessian v1 = **user cost Hessian + dynamics HVP·π + nonlinear-constraint HVP·λ** → extend `problem.hpp` with HVPs (full acados standard-SQP parity) |
| Q4 | Pluggable components (QP solver, regularizer, globalization) are **template parameters** of `SqpSolver` |
| Q6 | Soft **box** rows (`idxs_rev` coupling) deferred to phase 3 (neither example uses them) |
| Q10 | Tests in a **new `tests/` directory** (separate CMake targets, reusing example problem headers) |

Standing defaults (not separately discussed):

- **Homogeneous `QpStage`** over stages 0..N; absent rows (e.g. terminal
  control box) masked via `d_mask` (HPIPM's infinite-bound mechanism) →
  single Riccati code path.
- HPIPM v1 is **minimal**: `lq_fact=0` (Cholesky only), `itref_*_max=0`,
  `split_step=0`, `abs_form=0`, `var_init_scheme=1`, `warm_start` driven by
  SQP options; LQ fallback / refinement / split-step / abs-form join in
  phase 3.
- **Internal solver storage is runtime-extent** (sized at `solve()`); only
  user-facing `Solution<P, NH>` uses fixed-extent mode.
- **Header-only** (no new `.cpp`), per project convention.
- `SqpSolver` exposes const `last_qp()` / `last_qp_sol()` accessors
  (diagnostics, future RTI).
- Acados quirks are **verified against source at implementation time and
  matched as-is**: `res_comp` sign (doc 02 ambiguity: code adds `tau_min`),
  slack-penalty factor (HPIPM's slack objective is `sᵀs + 0.5 sᵀZs`, our
  contract says `0.5·w·s²` ⇒ slack Hessian diagonal `w`, `Z = w − 2`), exact
  `d`/`d_mask`/`lam`/`t`/`idx*` layouts (doc: `d` holds bound *offsets*, not
  distances; `lam` = `[λ_lower(rows); λ_upper(rows)]`).

## 3. Problem interface extension (`problem.hpp`)

New contract methods (base stubs with `static_assert`; keep the "Interface
contract" comment block and the "Extensions anticipated" note in sync):

```cpp
/// Hessian-vector product of the dynamics map f_k: (x,u) -> x_{k+1}  (k = 0..N-1).
/// Bilinear Hessian contraction (∇²f_k)·(w, v) with w in state space (dual to the
/// output; the SQP passes the dynamics multiplier π_k) and v = (v_x, v_u):
///   hv_x = Σ_i w_i (∂²f_i/∂x² v_x + ∂²f_i/∂u∂x v_u)
///   hv_u = Σ_i w_i (∂²f_i/∂x∂u v_x + ∂²f_i/∂u² v_u)
void dynamics_hess_prod(int k, const state_t& x, const control_t& u,
                        const state_t& w, const state_t& v_x, const control_t& v_u,
                        state_t& hv_x, control_t& hv_u) const;
```

```cpp
/// Hessian-vector product of the stage nonlinear inequality g_k(x,u) ≤ 0 (k = 0..N-1):
/// (∇²g_k)·(λ, v) with λ the per-row net multiplier (ineq order), v = (v_x, v_u).
/// Only required when ng > 0.
void stage_inequality_constr_hess_prod(int k, const state_t& x, const control_t& u,
                                       const ineq_t& lam, const state_t& v_x,
                                       const control_t& v_u,
                                       state_t& hv_x, control_t& hv_u) const;
// analogous: stage_equality_constr_hess_prod (ne > 0),
//            terminal_inequality_constr_hess_prod (ng_t > 0),
//            terminal_equality_constr_hess_prod (ne_t > 0)  (state-only vectors)
```

Notes:

- Linear constraints and boxes have zero Hessian → no HVP methods (matches
  acados: only the nonlinear part has `*_fun_jac_hess`).
- For equalities, the net multiplier for `∇²e` is
  `λ_u(e≤tol) − λ_u(−e≤tol)` (assembly detail, not interface detail).
- Verified against source: acados applies the dynamics HVP to the **stage's
  own** dynamics multiplier (`pi_ptr → nlp_out->pi+i`,
  `ocp_nlp_common.c:2812`), and the standard SQP's constraint Hessian comes
  from `bgh`'s `nl_constr_h_fun_jac_hess` with net multiplier
  `λ_upper − λ_lower` (the pure-linear `bgp` module refuses `compute_hess`).
- Both examples are linear ⇒ their HVPs are `setZero()`.

## 4. Module layout

```
include/ocp/problem.hpp           extended: Status codes + HVP contract
include/ocp/qp.hpp                QpDim<P>, Qp<P>, QpSol<P>  (pure data)
include/ocp/solver/hpipm.hpp      HpipmOptions, HpipmStatistics, HpipmQpSolver<P>
include/ocp/solver/regularize.hpp GlmRegularizer<P>, NoRegularizer<P>
include/ocp/solver/globalize.hpp  GlobOptions, MeritBacktracking<P>, Funnel<P>, apply_sqp_step()
include/ocp/solver/sqp.hpp        SqpOptions, SqpStatistics, SqpSolver<P, QpSolver, Regularizer, Globalizer>
tests/qp_unit/…                   phase 1: synthetic staged QPs
tests/sqp_double_integrator/…     phase 2: end-to-end
tests/sqp_mass_spring/…           phase 2/4: end-to-end + reference diff
```

`qp.hpp` → `problem.hpp`; solver headers → `qp.hpp`; `sqp.hpp` → all.
No cycle.

## 5. QP data model (`include/ocp/qp.hpp`)

```cpp
template <class P>
struct QpDim
{
    static constexpr int NX = P::nx, NU = P::nu;
    static constexpr int NGEN  = P::ng + 2 * P::ne + P::nl;          // general rows, stage
    static constexpr int NIN   = P::nbx + P::nbu + NGEN;             // two-sided rows, stage k
    static constexpr int NIN_T = P::nbx_t + P::ng_t + 2 * P::ne_t + P::nl_t;
    static constexpr int NROW  = (NIN > NIN_T) ? NIN : NIN_T;        // homogeneous
    static constexpr int NS    = P::nbx_soft + P::nbu_soft + P::ng_soft + P::ne_soft + P::nl_soft;
    static constexpr int NS_T  = P::nbx_t_soft + P::ng_t_soft + P::ne_t_soft + P::nl_t_soft;
    static constexpr int NSLACK = (NS > NS_T) ? NS : NS_T;
    static constexpr int NV    = NU + NX + 2 * NSLACK;               // (u; x; slacks)
    static constexpr int NCT   = 2 * (NROW + NSLACK);                // lam/t: (lower; upper) per row
    // compile-time stand-ins for HPIPM's runtime index sets:
    static constexpr std::array<int, /*nb*/> idxb     = /*...*/;      // box var index in (u;x)
    static constexpr std::array<int, /*2*NSLACK*/> idxs_rev = /*...*/; // slack pair <-> row (phase 3 use)
    static constexpr std::array<int, /*nbx if fixed x0*/> idxe = /*...*/;
};
```

Row order (convention, documented):
`[state box; control box; ineq; eq (2 per row); lin]`.

```cpp
template <class P>
struct QpStage
{
    using D = QpDim<P>;  using S = typename P::scalar_t;
    Eigen::Matrix<S, D::NV, D::NV>  hess{};  // "RSQrq", over (u;x;slacks); slack diag = w (soft rows)
    Eigen::Matrix<S, D::NV, 1>      grad{};  // "rqz"; slack part = w ∘ s_k (current slacks)
    Eigen::Matrix<S, D::NX, D::NU + D::NX> BAbt{};  // [B | A], stages 0..N-1
    Eigen::Matrix<S, D::NX, 1>      b{};           // dynamics rhs, stages 0..N-1
    Eigen::Matrix<S, D::NROW, D::NV> DCt{};        // [D | C] general rows, incl. ±1 slack cols
    Eigen::Matrix<S, 2 * D::NROW, 1> d{};          // (lower; upper) bound offsets
    Eigen::Matrix<S, 2 * D::NROW, 1> d_mask{};     // 0 = infinite/absent side
};

template <class P>
struct Qp
{
    Trajectory<QpStage<P>, Eigen::Dynamic> stage;  // N + 1 (sized at solve())
};

template <class P>
struct QpSol
{
    using D = QpDim<P>;  using S = typename P::scalar_t;
    // mixed convention exactly as HPIPM: ux = primal STEP; pi/lam/t = ABSOLUTE iterates
    Trajectory<Eigen::Matrix<S, D::NV,  1>, Eigen::Dynamic> ux;   // N + 1  (du; dx; ds)
    Trajectory<Eigen::Matrix<S, D::NX,  1>, Eigen::Dynamic> pi;   // N      dynamics multipliers
    Trajectory<Eigen::Matrix<S, D::NCT, 1>, Eigen::Dynamic> lam;  // N + 1
    Trajectory<Eigen::Matrix<S, D::NCT, 1>, Eigen::Dynamic> t;    // N + 1  slacks
};
```

## 6. HPIPM solver (`solver/hpipm.hpp`)

```cpp
struct HpipmOptions;   // d_ocp_qp_ipm_arg fields; defaults = BALANCE preset + acados overrides
                       // (mu0=1, alpha_min=1e-8, iter_max=stat_max=50, res_g 1e-6, res_b/d/m 1e-8,
                       //  var_init_scheme=1, lq_fact=0 in v1, itref=0, split_step=0, abs_form=0)

struct HpipmIteration { /* 21 quantities, doc 05 §5.3 */ };
struct HpipmStatistics { /* per-IPM-iteration records, stat_max-sized */ };

template <class P>
class HpipmQpSolver
{
public:
    explicit HpipmQpSolver(HpipmOptions opts = HpipmOptions{});
    Status solve(const Qp<P>& in, QpSol<P>& out);      // out: step + absolute duals
    const HpipmOptions& options() const;
    const HpipmStatistics& statistics() const;
private:
    HpipmOptions opts_;  HpipmStatistics stat_;
    struct Workspace { /* L_k per stage, Pb cache,
        res_g/res_b/res_d/res_m, dv/dpi/dlam/dt, gamma, t_inv, mu, sigma, ... */ } ws_;

    // 1:1 translations of the HPIPM macros (doc 05 §3-4):
    void init_point(const Qp<P>&, QpSol<P>&);                       // OCP_QP_INIT_VAR
    void compute_residuals(const Qp<P>&, const QpSol<P>&, QpRes<P>&); // OCP_QP_RES_COMPUTE
    void fact_solve_kkt(const Qp<P>&, QpSol<P>&, QpRes<P>&);        // OCP_QP_FACT_SOLVE_KKT_STEP
    void solve_kkt(const Qp<P>&, QpSol<P>&, QpRes<P>&);             // OCP_QP_SOLVE_KKT_STEP
    void solve_kkt_unconstr(const Qp<P>&, QpSol<P>&);               // inequality-free fast path
    double compute_alpha(...);                                       // COMPUTE_ALPHA_QP
    double compute_mu_aff(...);                                      // COMPUTE_MU_AFF_QP
    void apply_centering(...);                                       // COMPUTE_CENTERING[_CORRECTION]_QP
    void update_vars(...);                                           // UPDATE_VAR_QP
};
```

Mechanics (doc 05): backward pass
`L_k = chol(H_k + reg·I + D_kᵀΓ_kD_k + BAbt_k L_{k+1,xx} L_{k+1,xx}ᵀ BAbt_kᵀ)`
(square-root Schur; Eigen `LLT` per stage; singularity → `reg_prim` schedule
1e-4/÷3/×100/×8); forward substitution (`L_{k,uu}ᵀ` triangular solve,
`dux_{k+1,x} = BAbt_kᵀ dux_k + res_b_k`); closed-form dual/slack
`dt = D v̂ − res_d`, `dλ = −t⁻¹(res_m′ + λ⊙dt − λ⊙res_d)`; Mehrotra
`σ = (mu_aff/mu)³` with conditional pure-centering; exit on
`iter < iter_max && α > alpha_min && residuals ≤ tol && dual_gap ≤ dual_gap_max`.

## 7. Regularization (`solver/regularize.hpp`)

Duck-typed 2-method contract (mirrors `ocp_nlp_reg_config` minus RTI-only
`lhs`/`rhs`):

```cpp
void regularize(Qp<P>&) const;
void correct_dual_sol(Qp<P>&, QpSol<P>&) const;
```

- `GlmRegularizer<P>`: `epsilon_ = 1e-6`; per-stage Gershgorin
  min-eigenvalue estimate, shift the (u;x) Hessian block by `α·I`
  (α = |tmp|+ε if tmp<0 else ε); `correct_dual_sol` no-op.
- `NoRegularizer<P>`: both no-ops.

## 8. Globalization (`solver/globalize.hpp`)

```cpp
struct GlobOptions { double alpha_min = 0.05, alpha_reduction = 0.7,
                     eps_sufficient_descent = 1e-4; bool full_step_dual = false; };

// ocp_nlp_update_variables_sqp:
template <class P>
void apply_sqp_step(const Solution<P>& start, const QpSol<P>& step, double alpha,
                    bool full_step_dual, Solution<P>& dest);

// Contract: Status find_acceptable_iterate(const P&, Solution<P>& cur,
//   const QpSol<P>&, const Qp<P>&, Solution<P>& scratch, double& alpha) const;

struct MeritBacktracking { /* state: merit weights pi_w (N), lam_w (N) */ };  // phase 2
struct Funnel            { /* state: funnel_width, penalty_parameter,
                           l1_infeasibility, iter_type */ };                  // phase 3
```

- Merit (phase 2): plain merit-decrease backtracking;
  `merit = cost + |π_w|∘|dyn gap| + |λ_w|∘(violations)`; weights init
  `w ← |qp dual|`, update `w ← max(|qp dual|, 0.5(w + qp dual))`; optional
  Armijo sufficient-descent + SOC pre-pass in phase 3.
- **Slack penalty is solver-side**: the problem's cost functions exclude
  slacks (solver-internal), so the NLP objective, merit, and directional
  derivative add `0.5·w·s²` explicitly; `Solution.cost_value` includes it.
- **Internal slack trajectory** lives in `SqpSolver` (warm-started across
  repeated solves on the same solver object — the acados `nlp_out` slack
  analogue).

## 9. SQP driver (`solver/sqp.hpp`)

```cpp
struct SqpOptions { int max_iter = 20; double tol_stat = 1e-8, tol_eq = 1e-8,
    tol_ineq = 1e-8, tol_comp = 1e-8; double tol_min_step_norm = 1e-12,
    tol_unbounded = -1e10;
    bool compute_hess = true;                 // gates dyn/constr HVP terms (acados parity)
    double levenberg_marquardt = 0.0; bool with_adaptive_lm = false; /* phase 3 */
    int print_level = 0; int qp_warm_start = 0; bool warm_start_first_qp = false;
    bool eval_residual_at_max_iter = false; double timeout_max_time = 0.0; /* phase 3 */
    bool scale_qp_objective = false, scale_qp_constraints = false; };  /* phase 3 */

struct SqpIteration { double res_stat, res_eq, res_ineq, res_comp;
                      int qp_status, qp_iter; double step_norm, alpha; };
struct SqpStatistics { void record(const SqpIteration&);
                       void write_csv(const char*) const; ... };

template <class P, class QpSolver = HpipmQpSolver<P>,
          class Regularizer = GlmRegularizer<P>, class Globalizer = MeritBacktracking<P>>
class SqpSolver
{
public:
    explicit SqpSolver(SqpOptions opts = SqpOptions{});
    Status solve(const P& problem, Solution<P>& sol);   // sol: warm start in, solution out
    const SqpOptions& options() const;
    const SqpStatistics& statistics() const;
    const Qp<P>& last_qp() const;  const QpSol<P>& last_qp_sol() const;
private:
    SqpOptions opts_;  QpSolver qp_solver_;  Regularizer reg_;  Globalizer glob_;
    SqpStatistics stat_;  Qp<P> qp_in_;  QpSol<P> qp_out_;
    Solution<P> trial_;                       // globalization scratch
    Trajectory<slack_t, Dynamic> slacks_;     // solver-internal, warm-started
    double alpha_ = 0.0, cost_value_ = 0.0;

    void assemble_qp(const P&, Solution<P>&);     // matrices + vectors + LM shift
    void add_lm_term(Qp<P>&, double mu) const;
    NlpResiduals compute_nlp_residuals(const P&, const Solution<P>&) const;
    Status check_termination(int iter, const NlpResiduals&, double step_norm) const;
};
```

`solve()` loop (mirrors `ocp_nlp_sqp.c:541–798`, `iter = 0..max_iter`
inclusive):

1. `assemble_qp` (per stage: `A_k,B_k` from `dynamics_jacobian`;
   `hess_k = permute(stage_cost_hessian)` + LM·I; if `compute_hess`:
   += dynamics HVP matrix (built from `nx+nu` HVP calls with
   `w = sol.lambda_dyn[k]`, `v` = unit vectors) += constraint HVP terms with
   net multipliers from the current iterate; `b_k = dynamics_next_state −
   x_{k+1}`; `grad` = permuted cost gradient + `w∘s_k`; `d`/`d_mask` from
   the constraint specs; fixed `x_0` pinned via `idxe` rows)
2. `compute_nlp_residuals` (4 norms, doc 02 §8; first-order adjoints
   `−[A;B]ᵀπ` + π carry, `Cᵀλ_net`; signs verified in phase 2 against the
   source) → stats row
3. globalization `initialize` (iter 0)
4. `reg_.regularize(qp_in_)` — **before** the termination check, as in acados
5. `check_termination` (NaN → max_iter → tolerances → min step → unbounded →
   timeout) → return
6. `qp_solver_.solve(qp_in_, qp_out_)`; failure → `kQpFailure` (or
   `kMaxIterations` if the QP hit `iter_max`)
7. predicted reductions if the strategy needs them (`gᵀd`, QP objective,
   L1 infeasibility)
8. `glob_.find_acceptable_iterate(...)` → `alpha_` → commit to `sol` via
   `apply_sqp_step`
9. record `alpha_`; loop

## 10. C → C++ mapping (traceability)

| acados C | ocp++ C++ |
|---|---|
| `ocp_nlp_sqp` main loop | `SqpSolver::solve` |
| `check_termination` | `SqpSolver::check_termination` |
| `ocp_nlp_approximate_qp_matrices` / `_vectors_sqp` | `SqpSolver::assemble_qp` |
| `ocp_nlp_add_levenberg_marquardt_term` (+ adaptive mu) | `SqpSolver::add_lm_term` (adaptive: phase 3) |
| `ocp_nlp_res_compute` | `SqpSolver::compute_nlp_residuals` |
| `ocp_nlp_qpscaling_scale_qp` / `rescale_solution` | `QpScaler` (phase 3) |
| `ocp_nlp_reg_glm.c` | `GlmRegularizer` |
| `ocp_nlp_update_variables_sqp` | `apply_sqp_step` |
| `ocp_nlp_globalization_merit_backtracking.c` | `MeritBacktracking` |
| `ocp_nlp_globalization_funnel.c` | `Funnel` (phase 3) |
| `ocp_nlp_perform_second_order_correction` | phase 3 |
| `ocp_qp_hpipm` glue | `HpipmQpSolver::solve` |
| `OCP_QP_INIT_VAR` / `OCP_QP_RES_COMPUTE` / `OCP_QP_FACT_SOLVE_KKT_STEP` / `OCP_QP_SOLVE_KKT_STEP` / `COMPUTE_ALPHA_QP` / `COMPUTE_MU_AFF_QP` / `COMPUTE_LAM_T_QP` / `UPDATE_VAR_QP` | private methods of `HpipmQpSolver` (names in §6) |
| `FACT_LQ_SOLVE_KKT_STEP`, iterative refinement | phase 3 |
| `stat` matrices | `SqpStatistics` / `HpipmStatistics` |

## 11. Phases & verification

**Phase 0 — scaffolding + interface extension.** `problem.hpp` (Status
codes, HVP contract + stubs, doc notes); examples get zero HVPs; `qp.hpp`
(row/`d`/`lam`/`idx*` layouts verified against `hpipm_d_ocp_qp_dim.h`,
`x_ocp_qp_res.c`); `tests/` CMake targets. *Verify:* all targets build
warning-free; existing example tests still pass.

**Phase 1 — HPIPM QP solver.** Everything in §6 (minimal option scope).
*Tests:* synthetic staged QPs (known analytic solution; with/without
inequalities; with slacks) cross-checked against an **independent dense
full-KKT Eigen solve**; exit residuals below tolerances.

**Phase 2 — SQP end-to-end.** Assembly, NLP residuals, GLM/NoReg,
`apply_sqp_step`, merit backtracking, options/statistics/printing, driver.
*Tests:* `double_integrator` (N=10) and `mass_spring` (N=20): `kSolved`, all
four NLP residual norms below tolerances, cost and (x,u) within tolerance of
the acados reference (built from `acados/examples/.../generic_dyn_disc`);
both write CSV. This phase nails down the sign/factor conventions from §2.

**Phase 3 — robustness & parity.** LQ fallback + iterative refinement;
`split_step`, `warm_start ≥ 2`, `abs_form`; QP scaling (objective +
constraints) with adaptive QP tolerances; funnel; adaptive LM; SOC pre-pass;
timeout; soft box rows (`idxs_rev`).

**Phase 4 — validation (TODO step 5).** CSV reference-diff utility; more
acados examples.

## 12. Open items (verify at implementation time, not design decisions)

1. Exact `d`/`d_mask`/`lam`/`t`/`idxb`/`idxs_rev`/`idxe` semantics vs HPIPM
   headers (phase 0).
2. `res_comp` sign — match the source's actual behavior (adds `tau_min`).
3. Slack Hessian factor — expected `H_slack_diag = w`, `Z = w − 2` (HPIPM's
   `sᵀs + 0.5 sᵀZs` formulation vs our `0.5 w s²` contract); confirm `Z`'s
   role in factorization (`COND_SLACKS_FACT`).
4. `dual_gap_max` default value (not in the doc's preset table).

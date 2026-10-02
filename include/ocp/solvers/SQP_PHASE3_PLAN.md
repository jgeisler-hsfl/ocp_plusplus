# Phase 3 plan: robustness & parity

Companion to `SQP_PLAN.md` (§11 Phase 3) and the phase-1/2 records
(`hpipm/SQP_PHASE1_WORKLOG.md`, `acados/SQP_PHASE2_PLAN.md`). This file is the
working reference for the phase: the refined sub-step plan, the source-verified
conventions this phase nails down, and the running checkpoints (append a
checkpoint entry per sub-step, same protocol as the phase-1/2 files).

**Protocol for every sub-step:**
1. Re-read the acados/HPIPM sources listed for the sub-step (fresh; note line
   numbers in the checkpoint).
2. Implement header-only. QP-level changes go in
   `include/ocp/solvers/hpipm/hpipm.hpp`; NLP-level changes go in
   `include/ocp/solvers/acados/{sqp,globalize,regularize}.hpp`.
3. Build (`cmake --build build`), run all tests, warning-free under
   `-Wall -Wextra -Werror`.
4. Append a checkpoint entry below (what was done, test results, any deviation
   from this file — deviations win; update the entry here).
5. One commit per sub-step (`solver:` / `test:` scope).

## 0. State and file layout

- Phase 2 done: full SQP driver (`acados/sqp.hpp`: `SqpSolver`, assembly,
  residuals, options/stats, merit backtracking), HPIPM QP solver in the minimal
  v1 scope (`hpipm/hpipm.hpp`: Cholesky-only, no refinement, `split_step=0`,
  `abs_form=0`, cold start). End-to-end `double_integrator` / `mass_spring` /
  `acados_ref` pass to 1e-7.
- **All `HpipmOptions` fields for this phase already exist**
  (`hpipm.hpp:81-87`): `lq_fact`, `itref_pred_max`, `itref_corr_max`,
  `split_step`, `abs_form`, `warm_start` — currently inert (hard-coded to the
  v1 defaults). `SqpOptions` fields for the NLP items also already exist
  (`sqp.hpp:569-576`): `with_adaptive_lm`, `timeout_max_time`,
  `scale_qp_objective`, `scale_qp_constraints`, `qp_warm_start`,
  `warm_start_first_qp` — currently inert. This phase makes them live.
- `Status::kTimeout` already exists (`problem.hpp:846`). No new status needed.
- Soft box rows are **dormant** in both shipped examples (DI: soft ineq +
  terminal eq; MS: no soft rows) — enabling them is pure correctness/parity,
  no existing test regresses.

Sub-steps (each = one commit). Ordering rationale: soft-box DC first (foundational
correctness the QP work depends on), then the HPIPM IPM features (3b–3e), then
the NLP driver features (3f–3j).

```
3a  soft box rows (idxs_rev DC slack columns)     [NLP assembly, small]
3b  HPIPM split_step                              [QP, medium]
3c  HPIPM warm_start >= 2 (+ driver wiring)        [QP + NLP, medium]
3d  HPIPM LQ fallback + iterative refinement       [QP, large]
3e  HPIPM abs_form (absolute formulation)          [QP, large]
3f  timeout                                       [NLP, small]
3g  adaptive Levenberg-Marquardt                  [NLP, small]
3h  QP scaling + adaptive QP tolerances           [NLP, medium]
3i  funnel globalization                          [NLP, medium]
3j  second-order-correction pre-pass              [NLP, medium]
```

## 1. Conventions nailed down this phase

All facts below were re-read against the checked-out acados/HPIPM tree
(`acados/acados/ocp_nlp/...`, `acados/external/hpipm/...`) this session; exact
line numbers are cited per sub-step. The standing v1 conventions from phase 2
(step-formulation QP, `[lo;hi;slack]` side layout, per-side `lam`/`t`,
`res_comp` adds `tau_min`, slack Hessian diagonal `= w`, `Z = w`) are unchanged.

- **Soft box rows (3a).** A soft box row carries a `+1` (lo side) / `−1` (hi
  side) slack column in its `DC` row, exactly like ineq/eq/lin rows. HPIPM
  Schur-complements the slack out (`1/(Z + reg + Γ_row + Γ_slack)`); our
  explicit-slack model keeps it in `M_k` — mathematically equivalent, so the
  only missing piece was the `DC` column. The residual side
  (`stage_residuals`, `hpipm.hpp`) already couples box-row slacks via
  `idxs_lo_*/idxs_hi_*`, so **no solver change is needed**; only the assembly
  (`fill_dc_*` in `sqp.hpp`) must add the columns.
- **Warm start (3c).** In acados the primal `ux` is zeroed unconditionally
  before every QP solve, so only `warm_start ∈ {2,3}` (dual warm start) is
  effective; `warm_start=1` (primal) behaves like `0`. `qp_warm_start` maps
  1:1 to HPIPM `warm_start`. First-QP override: if `!warm_start_first_qp`, the
  first QP uses `warm_start=0` regardless of `qp_warm_start`.
- **abs_form (3e).** Orthogonal formulation: the KKT unknown is the *new
  iterate* (absolute) rather than the step; RHS is the original `rqz`/`b`/`d`
  (not the residuals); the complementarity RHS is transformed to
  `-(res_m + 2·qp.m)`; after each KKT solve the current iterate is subtracted
  to recover a step; the exit test is the scalar `|mu − tau_min|` instead of
  the four residuals + dual gap. No iterative refinement in the abs path.
- **LQ (3d).** `lq_fact ∈ {0,1,2}`: 0 = always Cholesky; 1 = Cholesky first,
  fall back to LQ when the linearized KKT residual exceeds `1e-5` (hard-coded
  threshold) or is NaN; 2 = always LQ. LQ factors the wide square-root matrix
  `W_k = [Lh_k | D_box √Γ_b | C √Γ_g | L_{k+1,xx}ᵀ BAbt_kᵀ]`; `L̃_k L̃_kᵀ = M_k`.
  The Hessian factor `Lh_k` is cached per stage (`use_hess_fact`). Iterative
  refinement solves the KKT system for the linear residual, adds it to the
  step, and stops when `lin_res < min(res_X_max, 1e-3 · res_X)` (componentwise,
  all four).
- **Funnel (3i).** Uses a *different* merit from the Leineweber weighted-L1:
  `ρ·F + L1_infeasibility` with a raw (no slack-penalty) `F` and an L1
  infeasibility (full |·| for dynamics, positive-part for constraints). The
  penalty `ρ` only ever *decreases* (contraction); there is no
  multiplier-increase branch. The accept test is a 4-way tree (f/h/b/p).
- **SOC (3j).** Belongs to the merit path, **not** the funnel. It re-linearizes
  by rewriting the QP RHS `b`/`d` from the *nonlinear* values at the full step
  (`d_i = c_i(x+p) − ∇c(x)ᵀ p`), re-solves the QP (Jacobian/Hessian unchanged),
  and corrects the duals. Triggered when the full step fails to lower both the
  merit and the L∞ violation.

## 2. Sub-steps

### 3a — soft box rows (`idxs_rev` DC slack columns)

Sources: `acados/external/hpipm/ocp_qp/x_ocp_qp.c:1575-1678` (`SET_IDXS_REV`),
`acados/external/hpipm/ocp_qp/x_ocp_qp_res.c:487-495` (residual idxs_rev loop),
`acados/ocp_nlp/ocp_nlp_constraints_bgp.c` (slack sign in the box fun).
Confirmed: box vs general rows differ *only* in how `v` is read (`z[idxb]` vs
`DC·z`) and how the reduced weight is applied (diagonal vs dense SYRK); the
`idxs_rev` residual loop and the Schur form are identical.

- In `fill_dc_first` / `fill_dc_path` / `fill_dc_term` (`sqp.hpp`), for every
  **soft** box row add the slack columns alongside the `idxb` unit vector:
  `st.DC(r, D::idxs_lo_<type>[r]) = +1` (lo side, if soft) and
  `st.DC(r, D::idxs_hi_<type>[r]) = −1` (hi side, if soft). State box
  (first/path), control box (first/path), terminal state box. The `idxs_*`
  arrays are already `−1` for hard rows, so guard on `>= 0`.
- No solver (`hpipm.hpp`) change: `stage_residuals` + `usx_build_side_dc`
  already consume the slack columns generically.
- Confirm the `fill_slack_diag` (Hessian slack diagonal = `w`) already covers
  soft box slacks (it does — `sqp.hpp:2244-2276` iterates all soft groups incl.
  box).
- **Test:** (i) assembly test (`tests/sqp/assemble_2f.cpp`): a soft state-box
  and soft control-box row now carry `+1`/`−1` slack columns; a hard box row
  carries only the `idxb` unit vector. (ii) QP-level oracle
  (`tests/hpipm/suite_1i.cpp` or a new case): a small QP with one soft state-box
  and one soft control-box row; verify the weighted-slack KKT
  (`λ_lo = w·s_lo`, `λ_hi = w·s_hi`, `s ≥ 0`) and cross-check `z`/`π`/obj
  against the dense full-KKT oracle.

### 3b — HPIPM `split_step` (separate primal/dual step lengths)

Sources: `acados/external/hpipm/core/x_core_qp_ipm_aux.c:193-468`
(`COMPUTE_ALPHA_QP`, `split_step==1` branch), `:472-582` (`UPDATE_VAR_QP`).
Preset note: `split_step=1` in SPEED/SPEED_ABS, `0` in BALANCE/ROBUST.

- `compute_alpha` (currently the `split_step=0` single-α path, `hpipm.hpp`):
  add the `split_step==1` path — two alphas, `alpha_prim` (drives `v`, `t`) and
  `alpha_dual` (drives `π`, `λ`):
  - **Pass 1** (per active side): `lam1 = lam + a_d·dlam`, `t1 = t + a_p·dt`.
    `lam1 < 0 → a_d = −lam/dlam`; `t1 < 0 → a_p = −t/dt`. If
    `lam1·t1 − m_safe·m < −1e-12`: if `dlam < 0 ∧ dt ≥ 0 → a_d =
    (m_safe·m − lam·t)/(dlam·t1)`; if `dlam ≥ 0 ∧ dt < 0 → a_p =
    (m_safe·m − t·lam1)/(dt·lam1)`; both-negative case deferred to Pass 2.
  - **Pass 2** (quadratic, for sides with `dlam < 0 ∧ dt < 0` still violating):
    `c = lam·t − m_safe·m`, `a = a_d·dlam·a_p·dt`,
    `b = a_d·dlam·t + lam·a_p·dt`; if `c > 0` and `disc = b²−4ac ≥ 0` then
    `r = (−b − √disc)/(2a)` and `a_d *= r`, `a_p *= r`; else `a_p = a_d = 0`.
  - **Pass 3**: re-apply Pass-1 feasibility clipping to any side the Pass-2
    scaling may have violated.
  - Store both in the workspace (`ws_.alpha_prim`, `ws_.alpha_dual`); the
    `split_step=0` path keeps `alpha = alpha_prim = alpha_dual`.
- `update_vars`: damp each alpha independently
  (`α' = α·(0.99(1−α) + 0.9999999·α)` when `α < 1`), then
  `ux += a_p·δz`, `π += a_d·δπ`, `lam += a_d·δlam`, `t += a_p·δt` (with the
  `t_lam_min` floors).
- `solve` exit test: replace the single `ws_.alpha` with
  `alpha_prim > alpha_min && alpha_dual > alpha_min`; map to `kMinStep` when
  **either** falls below `alpha_min`.
- **Test:** hand-computed `compute_alpha` cases where the primal and dual bind
  at different ratios (assert distinct `alpha_prim`/`alpha_dual`); an
  `update_vars` case with the two alphas; a short (≤5-iter) solve on a small QP
  with `split_step=1` asserting `lam`/`t` stay ≥ floors and residuals decrease.

### 3c — HPIPM `warm_start ≥ 2` (+ driver wiring)

Sources: `acados/external/hpipm/ocp_qp/x_ocp_qp_ipm.c:1632-1732`
(`OCP_QP_IPM_INIT_VAR` warm-start branches), `acados/ocp_nlp/ocp_qp_hpipm.c:327-336`
(ux zeroing), `acados/ocp_nlp/ocp_nlp_sqp.c:660-691` (first-QP override +
1:1 `qp_warm_start → warm_start`), `docs/algorithm/README.md:90-100`.

- `init_point` (`hpipm.hpp`): generalize the cold-start zeroing into a
  `warm_start` switch:
  - `≥ 3` (hot): keep all of `ux, π, lam, t`; clip `lam ≥ lam0_min`,
    `t ≥ t0_min` (`1e-9`); `return` (skip the heuristic).
  - `== 2` (primal+dual): keep `π, lam, t`; clip to `thr0 = 1e-1`; zero `ux`;
    run the existing heuristic **only** to repair any `t < thr0` / infeasible
    box variable (the current `init_stage` logic on the preserved `lam`/`t`).
  - `≤ 1` (cold): the current behavior (zero `ux`/`π`, heuristic `lam`/`t`).
    (`warm_start=1` ≡ `0` in acados because the driver zeroes `ux` regardless.)
- **Driver wiring** (`sqp.hpp`): `qp_out_` is already a persistent member.
  Before each `qp_.solve(qp_in_, qp_out_)`: zero `qp_out_.ux_*` (primal always
  cold), set `qp_solver_`'s `warm_start` to `opts_.qp_warm_start` (except
  `warm_start = 0` when `iter == 0 && !opts_.warm_start_first_qp`). This makes
  the absolute `π`/`lam`/`t` from iteration `k−1` the IPM initial iterate for
  iteration `k`.
- **Test:** solve a QP twice; the second solve warm-started from the first
  solution reaches the same `z`/`π` in fewer IPM iterations
  (`HpipmStatistics::iter` smaller), same `kSolved`. Also the driver-level:
  a `SqpSolver` with `qp_warm_start=2` vs `0` on `double_integrator` — both
  `kSolved`, warm version uses fewer total IPM iterations.

### 3d — HPIPM LQ fallback + iterative refinement

Sources: `acados/external/hpipm/ocp_qp/x_ocp_qp_kkt.c:1903-2301`
(`FACT_LQ_SOLVE_KKT_STEP`: `W_k` build, `GELQF`, sign normalization
`2044-2046`, `use_hess_fact` cache `2009-2033`/`2104-2129`), `x_ocp_qp_kkt.c`
`COND_SLACKS_FACT`; `acados/external/hpipm/ocp_qp/x_ocp_qp_res.c:535-685`
(`OCP_QP_RES_COMPUTE_LIN`); `x_ocp_qp_ipm.c:2373-2505` (predictor itref),
`:2609-2732` (corrector itref), `:2274-2369` (`lq_fact` mode dispatch, the
`1e-5` switch at `2324-2333`).

- **LQ solve** (`fact_solve_kkt_lq`, new private method): per stage, build the
  wide matrix `W_k = [Lh_k | D_box √Γ_b | C √Γ_g | L_{k+1,xx}ᵀ BAbt_kᵀ]`
  (terminal drops the dynamics block), `Lh_k = chol(H_k + reg·I)` cached via a
  per-stage `use_hess_fact` flag, then LQ-factor `W_k` (Eigen `HouseholderQr`,
  take `L̃` from the R-diagonal-sign-normalized factor so `L̃ L̃ᵀ = M_k`);
  sign-normalize (flip column `ii` if the `ii`-th diagonal is negative). The
  forward/backward recursions and the closed-form `δt`/`δλ` are **shared** with
  the Cholesky path (only the factor source differs). A singular `Lh_k`
  (Cholesky of the Hessian block fails) triggers the `reg_prim` growth ladder
  (phase-1 note: 1e-4 / ÷3 / ×100 / ×8 per the plan §2).
- **`lq_fact` dispatch** in the delta step: `0` → Cholesky; `1` → Cholesky,
  compute the linear residual (`OCP_QP_RES_COMPUTE_LIN`), and if
  `max(lin_res) > 1e-5` or any NaN → re-factor with LQ and set
  `ws_.last_lq_fact = 1` (per-iteration, not persistent); `2` → LQ always.
- **Iterative refinement** (`refine_step`, new private method): compute
  `lin_res = {res_g, res_b, res_d, res_m}` of the *linearized* KKT system the
  step is supposed to satisfy (formulas from `OCP_QP_RES_COMPUTE_LIN`,
  `x_ocp_qp_res.c:598-681`); loop up to `itref_pred_max`/`itref_corr_max` times:
  break when `lin_res[i] < min(res_X_max, 1e-3 · res_X_max_current)` for all
  `i` (componentwise, the four components); else solve the KKT system for
  `lin_res` (reuse factors, `use_Pb = 0`), add the correction to the step,
  recompute; **undo** (subtract the correction) and break if the new max
  `lin_res` ≥ the old. Record `itref_pred`/`itref_corr` and the final
  `lin_res_*` into the stat row (columns 13-19).
- **Stat**: fill `HpipmIteration::{lq_fact, itref_pred, itref_corr,
  lin_res_stat/eq/ineq/comp}` (currently hard-zeroed in `solve`,
  `hpipm.hpp:472-479`).
- **Test:** an ill-conditioned / near-singular stage Hessian that makes the
  Cholesky-path linear residual exceed `1e-5` → assert LQ is selected
  (`stat.lq_fact = 1`) and the QP still solves to tolerance; a case with
  `itref_corr_max ≥ 1` → assert `itref_corr` is recorded and the linear
  residual shrinks; cross-check `z`/`π`/obj against the dense oracle.

### 3e — HPIPM `abs_form` (absolute formulation)

Sources: `acados/external/hpipm/ocp_qp/x_ocp_qp_ipm.c:2777-3051`
(`OCP_QP_IPM_SOLVE` abs branch, setup `2978-3003`, loop `3022-3051`),
`:2053-2228` (`OCP_QP_IPM_ABS_STEP`: m-RHS transform `2075`, post-solve
subtract `2096-2099`), `hpipm_d_ocp_qp_ipm.h` (field semantics), preset: only
`SPEED_ABS` uses `abs_form=1` (+ `split_step=1`, `comp_* = 0`).

- New `solve_abs(...)` private path (or a branch in `solve`), mirroring the
  delta path with these differences:
  - **RHS**: the KKT solve uses the *original* QP vectors (`rqz = grad`, `b`,
    `d`) instead of the residuals (`res_g`/`res_b`/`res_d`).
  - **m-RHS transform**: before each step, `res_m ← −(res_m + 2·qp.m)`
    (`AXPBY(nc, −1.0, res_m, −2.0, qp.m)`); `qp.m` is 0 in v1 (no
    `m_relax`), so this is `res_m ← −res_m`.
  - **Post-solve subtract**: after each KKT solve, `step.ux −= iter.ux`,
    `step.pi −= iter.pi`, `step.lam −= iter.lam`, `step.t −= iter.t`, so the
    subsequent `compute_alpha`/`update_vars` operate on a step.
  - **Exit test**: `|mu − tau_min| > res_m_max` (scalar `mu` = masked L1 mean
    of `lam∘t − m`), plus the `alpha_prim`/`alpha_dual > alpha_min` guard;
    **no** per-iteration `OCP_QP_RES_COMPUTE` (only the inline `mu`).
  - **No iterative refinement** in the abs path.
- `abs_form` is orthogonal to `split_step` (share `compute_alpha`/
  `update_vars`).
- **Test:** solve a small QP with `abs_form=1` and `abs_form=0`; assert the
  converged `z`/`π` agree to ~1e-9 (same optimum, different trajectory); assert
  the exit residual `|mu − tau_min|` is within `res_m_max`.

### 3f — timeout

Sources: `acados/ocp_nlp/ocp_nlp_sqp.c:351-443` (`check_termination` timeout
branch `434-440`), `:519-520` (estimate init), `:538-539` + `607-644` (per-iter
estimate update), `:113-117` (defaults: `timeout_heuristic = ZERO`,
`timeout_max_time = 0`), `utils/types.h:111-118` (heuristic enum).

- `SqpOptions`: add `int timeout_heuristic = 0;` (0 = `ZERO`, 1 = `LAST`,
  2 = `MAX`, 3 = `AVERAGE`) alongside the existing `timeout_max_time`.
- `SqpSolver::solve`: capture a `steady_clock` at entry; before each
  `check_termination`, if `timeout_max_time > 0` and `iter > 0`, update
  `timeout_estimated_per_iter_` per the heuristic (`LAST`: last duration;
  `MAX`: running max; `AVERAGE`: `0.5·new + 0.5·old`; `ZERO`: stays 0) and set
  the elapsed = wall-clock so far.
- `check_termination`: add the **final** branch (after unbounded): if
  `timeout_max_time > 0` and
  `timeout_max_time ≤ elapsed + timeout_estimated_per_iter_` → `kTimeout`.
- **Test:** a duck-typed slow problem (or a forced estimate) with a tiny
  `timeout_max_time` → `kTimeout`; `timeout_max_time = 0` (default) → inert.

### 3g — adaptive Levenberg-Marquardt

Sources: `acados/ocp_nlp/ocp_nlp_common.c:3011-3032`
(`adaptive_levenberg_marquardt_update_mu`), `:3034-3059`
(`ocp_nlp_add_levenberg_marquardt_term`, effective damping =
`obj_scalar · cost_value · mu`), `:1212,1265-1268` (defaults: `lm = 0`,
`lam = 5.0`, `mu_min = 1e-16`, `obj_scalar = 2.0`; `mu0` set by the interface
to `1e-3`), `ocp_nlp_common.h:327-332` (fields).

- `SqpOptions`: add `adaptive_lm_mu0 = 1e-3`, `adaptive_lm_lam = 5.0`,
  `adaptive_lm_mu_min = 1e-16`, `adaptive_lm_obj_scalar = 2.0` (the existing
  `with_adaptive_lm` gates it).
- `SqpSolver`: state `lm_mu_`, `lm_mu_bar_` (reset in `solve`). Each iteration,
  *before* `add_lm_term`: if `with_adaptive_lm`, update
  `iter == 0 → mu = mu_bar = mu0`; else if `alpha_ == 1.0`
  (full step): `mu = max(mu_min, mu_bar/lam)`, `mu_bar = mu_old`; else
  (truncated): `mu = min(lam·mu, 1.0)`. Then set the effective LM coefficient
  `mu_eff = obj_scalar · raw_cost · mu` and pass it to `add_lm_term`
  (replacing the static `opts_.levenberg_marquardt` when adaptive is on).
  **`raw_cost` = stage + terminal cost only, no slack penalty** (acados'
  `cost_value` excludes the QP slack term) — add a `raw_cost(problem, sol)`
  helper (reuse the two sum loops in `compute_cost` minus `slack_penalty`).
- `add_lm_term` itself is unchanged (already adds `mu·I` to the `(u;x)` block).
- **Test:** a sequence of full steps → `mu` decays by `1/lam` each (down to
  `mu_min`); a truncated step → `mu` grows by `·lam` (capped at 1); the
  effective damping on the Hessian equals `obj_scalar·raw_cost·mu`.

### 3h — QP scaling + adaptive QP tolerances

Sources: `acados/ocp_nlp/ocp_nlp_qpscaling.c` (`compute_obj_scaling_factor`
`483-540`, `scale_qp` `~124-160`, `rescale_solution` `668-686`,
`scale_objective` `379-392`, `out_scale_duals` `396-407`,
`scale_constraints` `545-618`, `min_constraint_scaling` `282-297`,
`coeff_norm` `314-324`), `ocp_nlp_common.c:3625,3631,4522-4574,4596-4615`
(precompute, regularizer pointer, tolerance strategy, solve-order),
`ocp_nlp_sqp.c:594-597,602,685` (call order), `utils/types.h:102-141`
(strategy + scaling enums), `utils/math.c:1118-1143`
(`compute_gershgorin_max_abs_eig_estimate`).

- **New `acados/qpscaling.hpp`** (or a private section in `sqp.hpp`): a
  `QpScaler<P, NH>` holding `obj_factor`, per-stage/per-row
  `constr_scaling`, a `scaled_qp_in`/`scaled_qp_out` pair, and `precompute()`
  (alias the shared fields, allocate the separate ones per which scalings are
  on).
- **Objective scaling** (`scale_objective`):
  `max_abs_eig = max over stages of (gershgorin_max_abs_eig(H_k) , ‖Z_k‖∞)`;
  `‖grad‖∞` over `(u;x;s)`; if `max_abs_eig < ub_max_abs_eig`
  (`1e5`) → `obj_factor = 1`, `max_upscale = ub_max_abs_eig/max_abs_eig`;
  else `obj_factor = ub_max_abs_eig/max_abs_eig`, `max_upscale = obj_factor`;
  if `obj_factor·‖grad‖∞ ≤ lb_norm_inf_grad_obj` (`1e-4`) →
  `obj_factor = max(obj_factor, min(max_upscale, lb/‖grad‖∞))` (flag
  `bounds_not_satisfied`). Apply `obj_factor` to each stage's `hess`, `grad`,
  and slack Hessian diagonal.
- **Constraint scaling** (`scale_constraints`, **general rows only** —
  `g_ineq`/`g_eq`/`g_lin`; box rows untouched): per row `j`,
  `s_j = 1/max(1, max(bound_max_j, ‖DC row j‖∞))` where
  `bound_max_j = max(|mask_lo·d_lo|, |mask_hi·d_hi|)`; scale `DC` row `j` by
  `s_j`, the row's `d_lo`/`d_hi` (where active) by `s_j`, and the row's slack
  columns: `grad` slack by `s_j`, Hessian slack diagonal by `s_j²`. Track
  `min_constraint_scaling = min_j s_j`.
- **Call order** (match acados `ocp_nlp_sqp.c`): `assemble_qp` →
  **`scale_qp`** → `reg_.regularize` (regularizer acts on the *scaled* Hessian,
  matching `ocp_nlp_common.c:3631`) → `check_termination` →
  **adaptive tolerances** (below) → `qp_.solve(scaled_qp_in, scaled_qp_out)` →
  **`rescale_solution`** (`lam`/`π` scaled back by `1/obj_factor` if objective
  scaled; general-row `lam` by `s_j`, slack `ux` by `1/s_j`; primal `(u;x)` and
  box multipliers untouched) → `reg_.correct_dual_sol` (scaled-space, then
  rescaled) → duals into `sol` via the existing §1.3 mapping.
- **Adaptive QP tolerances** (`ocp_nlp_common.c:4522-4574`), set the HPIPM
  `res_g_max/res_b_max/res_d_max/res_m_max` per-iteration before the solve:
  - `ADAPTIVE_CURRENT_RES_JOINT` (reduction factor `r = 1e-1`):
    `tmp_X = min(r·res_X_nlp, 1e-2)` for each of the four;
    `joint = max(tmp_X)`; `tol_X = max(safety(0.1)·tol_X, joint)`.
  - `ADAPTIVE_QPSCALING`: `stat_factor = min(obj_factor,
    min_constraint_scaling)`; `tmp_stat = safety·tol_stat·stat_factor`,
    `tmp_eq = safety·tol_eq·1.0`, `tmp_ineq = safety·tol_ineq·min_constr`,
    `tmp_comp = safety·tol_comp·min_constr`; floor each at
    `tol_min_stat(1e-9)/eq(1e-10)/ineq(1e-10)/comp(1e-11)`.
  - `FIXED_QP_TOL` (default): use the HPIPM fixed tolerances (no change).
- `SqpOptions`: replace the boolean `scale_qp_objective`/`scale_qp_constraints`
  with enums (`kNoObjectiveScaling`/`kObjectiveGershgorin`,
  `kNoConstraintScaling`/`kInfNorm`) or keep booleans + add
  `ub_max_abs_eig`, `lb_norm_inf_grad_obj`, `nlp_qp_tol_strategy`,
  `nlp_qp_tol_reduction_factor`, `nlp_qp_tol_safety_factor`,
  `nlp_qp_tol_min_{stat,eq,ineq,comp}`. (Decide at implementation; document.)
- **Test:** a `SqpSolver` with both scalings on vs off on a poorly-scaled
  QP (e.g. one constraint row 1e4× the rest) → both reach the same `z`/`π`/
  cost to ~1e-6; assert the scaled-space QP residuals (the 4 `res_*_max`
  reported from the scaled solve) are within the adapted tolerances; a
  unit test of `compute_obj_scaling_factor` / `scale_constraints` against
  hand-computed `obj_factor`/`s_j`.

### 3i — funnel globalization

Sources: `acados/ocp_nlp/ocp_nlp_globalization_funnel.c` (merit `444,504-506`,
`is_trial_iterate_acceptable_to_funnel` `319-413`, `update_funnel_penalty_parameter`
`240-263`, `initialize_funnel_width` `229-233`, `initialize_memory`
`588-605`, `backtracking_line_search` `516-535`),
`acados/ocp_nlp/ocp_nlp_globalization_funnel.h:56-96` (opts + memory),
`ocp_nlp_globalization_common.h:54-71,82-90` (config vtable + common opts),
`ocp_nlp_common.c:2626-2644` (`gradient_directional_derivative`),
`:2714-2749` (`get_l1_infeasibility`), `ocp_nlp_sqp.c:517,573-576,753-759,778`
(driver pre-computes + call).

- **New `Funnel<P, NH>`** in `globalize.hpp` (same `find_acceptable_iterate`
  contract as `MeritBacktracking`, duck-typed by the driver):
  - **Merit**: `ρ·F + L1` where `F` = raw stage+terminal cost (no slack
    penalty) and `L1` = `L1_infeasibility` (full `|·|` on the dynamics gap
    `f − x_{k+1}`; positive-part on constraint residuals with slacks absorbed,
    i.e. a soft row contributes `max(0, v − s_hi)` etc.; pin rows excluded).
  - **Predicted reductions** (computed inside the globalizer from `problem`,
    `sol`, `step`, `slacks` — avoids widening the driver contract, deviation
    from acados' pre-feed): `predicted_optimality_reduction = −Σ step·grad`
    (the QP `rqz·d`); `predicted_infeasibility_reduction = L1_infeasibility`
    at the current iterate (the documented acados quirk — a level, not a
    reduction).
  - **Penalty update** (`update_funnel_penalty_parameter`, before the loop):
    clamp tiny-negative `predicted_obj_red` in `(-1e-4, 0)` to 0; if
    `ρ·predicted_obj_red + predicted_infeas_red < eta(1e-6)·predicted_infeas_red`
    → `ρ = max(0, min(contraction(0.5)·ρ, (1−eta)·predicted_infeas_red /
    (−predicted_obj_red + 1e-9)))`; else unchanged. (Never increases.)
  - **Accept tree** (`f`/`h`/`b`/`p`), with
    `eps_sufficient_descent = 1e-4`, `alpha_min = 0.05`,
    `alpha_reduction = 0.7`:
    - inside funnel (`L1_trial ≤ funnel_width`) and not in penalty mode:
      - switching `α·P_obj ≥ c_sw(1e-3)·P_infeas` **and** Armijo on the
        objective → accept `'f'`.
      - else if `L1_trial ≤ c_sd(0.9)·funnel_width` → accept `'h'` + shrink
        funnel.
      - else if `L1_trial < L1_cur` **and** Armijo on the merit → accept `'b'`,
        enter penalty mode.
    - in penalty mode: Armijo on the merit → accept `'p'`; if also
      `L1_trial ≤ c_sd·funnel_width` → shrink funnel, leave penalty mode.
    - outside funnel, not merit-only: reject.
  - **Funnel shrink** (`decrease_funnel`):
    `w ← (1−κ(0.9))·L1_trial + κ·w`.
  - **Init** (`initialize`): `L1 ← L1_infeasibility(cur)`,
    `funnel_width ← max(upper_bound(1.0), increase_factor(15)·L1_init)`,
    `ρ ← objective_multiplier` (driver value, default 1.0), `α = 1`,
    `funnel_iter_type = '-'`, `funnel_penalty_mode = false`.
- **Driver**: `Funnel` needs the raw cost and the QP `grad` (for
  `predicted_optimality_reduction`). Pass the assembled `qp_in_` into
  `find_acceptable_iterate` (or a `needs_qp_objective_value`-style flag so the
  driver pre-computes `predicted_*`). On `kMinStep` the funnel must **not**
  advance `sol` (unlike merit backtracking, which commits the tiny step) —
  match `funnel.c:525-529`.
- **Test:** funnel on `double_integrator` (N=10) → `kSolved`, all four NLP
  residuals below tolerance, `funnel_width` monotonically non-increasing and
  ending near the final `L1`; assert the accept-type sequence is logged
  (add a debug/`iter_type` field to `SqpIteration` or a Funnel stat).

### 3j — second-order-correction pre-pass

Sources: `acados/ocp_nlp/ocp_nlp_common.c:4290-4442`
(`ocp_nlp_perform_second_order_correction`: `b` rewrite `4327-4334`, general-
row `d` rewrite `4339-372`, slack `d` correction `4359-4372`, re-solve `4393-
4394`, dual correction `4398-4399`),
`acados/ocp_nlp/ocp_nlp_globalization_merit_backtracking.c:493-564`
(`ocp_nlp_line_search_merit_check_full_step`), `:566-602`
(`ocp_nlp_soc_line_search`), `:871-897` (SOC gate in
`find_acceptable_iterate`), `ocp_nlp_globalization_common.h:84`
(`use_SOC`, default 0).

- `GlobOptions`: add `bool use_soc = false;` (and optionally
  `line_search_use_sufficient_descent = false`).
- **New `SqpSolver::perform_second_order_correction`** (private, needs
  `qp_in_`/`qp_out_`/`qp_`/`reg_`): at the full step,
  - dynamics: `b_k ← −BA_kᵀ·δz_k − [f_k(x+δx, u+δu) − x_{k+1} − δx_{k+1}]`
    (re-evaluate the **nonlinear** `dynamics_next_state` at the trial iterate;
    the sign of the nonlinear term is confirmed against the source at
    implementation time — acados' own comment flags uncertainty);
  - general rows (ineq/eq/lin, **not** box): `d_lo ← g(x+δ) + DC·δz`,
    `d_hi ← g(x+δ) − DC·δz`; soft rows additionally subtract the slack step
    from the side's `d`;
  - re-solve the QP **in place** (`qp_.solve(qp_in_, qp_out_)`) with the
    original Jacobian/Hessian, then `reg_.correct_dual_sol`; on QP failure
    return `kQpFailure`.
- **Merit restructure** (`MeritBacktracking`): before the backtracking loop,
  if `use_soc`: evaluate merit + L∞ violation at the full step
  (`α = 1`); if `merit1 < merit0` **and** `violation_step < violation_current`
  → accept full step (`α = 1`, skip line search); else trigger SOC
  (`SqpSolver::perform_second_order_correction`) and then run the line search
  on the corrected step. Match the acados weight handling: (re)initialize /
  back up the merit weights at the full-step check, and re-update them from
  the (SOC-corrected) QP duals before the line search.
- **Driver**: expose `perform_second_order_correction` to the globalizer
  (the globalizer needs to re-solve the QP — pass the `SqpSolver` or the
  needed QP data + a re-solve callable). This widens the
  `find_acceptable_iterate` contract only when `use_soc` is on; document.
- **Test:** a crafted iterate whose full step lowers neither the merit nor the
  L∞ violation → assert SOC fires (QP re-solved, `qp_out` changed) and the
  corrected step is accepted; with `use_soc = false` the current behavior is
  unchanged (regression: all phase-2 tests still pass).

## 3. Test matrix

| target | covers |
|---|---|
| `qp_unit` (+ `suite_1i` new cases) | 3a soft-box QP oracle; 3b split-step alpha/update; 3c warm-start; 3d LQ + refinement; 3e abs_form |
| `sqp_unit` | 3a soft-box DC assembly; 3g adaptive LM; 3h scaling factor / rescale unit tests; 3j SOC trigger |
| `sqp_double_integrator` / `sqp_mass_spring` | end-to-end with 3b/3c/3f/3g/3h on (default options keep v1 behavior); 3i funnel variant |
| `sqp_acados_ref` | regression: all features off (v1) still matches the acados reference; a second run with the phase-3 features on matches to the same 1e-4 |

Exit criteria for the phase: all targets build warning-free and pass; the
`double_integrator` (N=10) and `mass_spring` (N=20) end-to-end tests reach
`kSolved` with all four NLP residual norms below the option tolerances **both
with and without** the phase-3 features enabled; the new HPIPM features
(3b/3c/3d/3e) are each verified against an independent dense/full-KKT oracle;
the §1 conventions are signed off in the checkpoints; `SQP_PLAN.md` §11
Phase 3 marked done and any §12 open items updated.

## 4. Checkpoints

### 3a — soft box rows (`idxs_rev` DC slack columns)

- Re-read `x_ocp_qp.c:1575-1678` (`SET_IDXS_REV`) and
  `x_ocp_qp_res.c:487-495`; confirmed box rows differ from general rows only
  in how `v` is read (`z[idxb]` vs `DC·z`) and how the reduced weight is
  applied (diagonal vs dense SYRK) — the `idxs_rev` residual loop and the Schur
  form are identical, so no `hpipm.hpp` change is needed.
- Implemented in `include/ocp/solvers/acados/sqp.hpp`: in
  `fill_dc_first` / `fill_dc_path` / `fill_dc_term`, every soft box row now
  also writes `st.DC(r, D::idxs_lo_<type>[r]) = +1` (lo side) and
  `st.DC(r, D::idxs_hi_<type>[r]) = -1` (hi side), alongside the existing
  `idxb` unit vector. Covers state box (first/path), control box (first/path)
  and terminal state box. Each write is guarded by `idxs_* >= 0` so hard rows
  (idxs `= -1`) are untouched.
- Confirmed `fill_slack_diag` / `fill_slack_diag_term` already set the box
  slack Hessian diagonals to `w` for all soft groups (no change).
- No solver change: `stage_residuals` + `usx_build_side_dc` +
  `compute_gamma` consume the slack columns generically.
- Test:
  - Assembly (`tests/sqp/assemble_2f.cpp`, new `SoftBoxProbe`: nx=2, nu=2,
    fixed x_0, N=2; soft state-box row 0 (w=2), soft control-box row 0 (w=5),
    soft terminal state-box row 0 (w=3); row 1 in each group hard).
    Verifies soft rows carry `+1`/`-1` slack columns in DC, slack Hessian
    diagonals = w, and hard rows carry `idxs_* < 0` with zero slack-column
    entries.
  - QP-level oracle (`tests/hpipm/suite_1i.cpp`, new case 3): N=1, x_0 pinned,
    soft control box (w=4) + soft terminal state box (w=25), the non-violated
    sides masked (`d_mask = 0`, `d = -1e3`). Verifies weighted-slack KKT
    (`lam = w*s`, `s >= 0`) and cross-checks `z` / `pi` / obj / active
    multipliers against the dense full-KKT oracle (hi- and lo-violated
    sub-cases; expected `u = x_1 = 41/31`, `s = 10/31`, `lam_bu = 40/31`,
    `lam_bx_term = 250/31`, `obj = -391/31`).
- Deviation: the masked (non-violated) box side keeps a tiny positive
  multiplier, so its slack is `lam/w ~ 1e-5` rather than exactly 0; the oracle
  comparison zeroes that entry and separately asserts it is `< 1e-3`. All
  other components match the oracle to `~1e-11`.
- Build: warning-free under `-Wall -Wextra -Werror`. All targets green:
  `qp_unit`, `sqp_unit`, `double_integrator`, `mass_spring`,
  `sqp_double_integrator`, `sqp_mass_spring`, `sqp_acados_ref`.

### 3b — HPIPM `split_step` (separate primal/dual step lengths)

- Re-read `x_core_qp_ipm_aux.c:193-468` (`COMPUTE_ALPHA_QP`, the
  `split_step==1` branch) and `:472-582` (`UPDATE_VAR_QP`). Confirmed the
  three-pass structure: Pass 1 per-side feasibility clipping into
  `alpha_prim` / `alpha_dual`; Pass 2 quadratic scaling for the still-violating
  `dlam < 0 ∧ dt < 0` sides; Pass 3 re-applies the complementarity check only
  (the C source does **not** re-run the Pass-1 feasibility clip in Pass 3 —
  the §2 sub-step note was imprecise; the source is authoritative).
- `compute_alpha` now returns `void` and stores both alphas in the workspace
  (`ws_.alpha_prim`, `ws_.alpha_dual`); public accessors `alpha_prim()` /
  `alpha_dual()` added. The `split_step==1` path implements the two alphas with
  a defensive `disc >= 0` guard in Pass 2 (the C source has no guard; `disc < 0`
  is mathematically impossible in practice, the guard is harmless). The
  `split_step==0` path sets `alpha_prim = alpha_dual = alpha` (unchanged values).
- `compute_mu_aff` uses `alpha_dual` for `dlam` and `alpha_prim` for `dt`
  (matching `COMPUTE_MU_AFF_QP`, `x_core_qp_ipm_aux.c:636`).
- `update_vars`: independent damping `α' = α·(0.99(1−α) + 0.9999999·α)` when
  `min(α_p, α_d) < 1`; `ux` / `t` scaled by `alpha_prim`, `pi` / `lam` by
  `alpha_dual` (with the `t` / `lam` floors).
- `solve()` init sets both alphas to 1; the exit test maps to `kMinStep` when
  **either** `alpha_prim` or `alpha_dual` falls at or below `alpha_min`
  (`x_ocp_qp_ipm.c:3119-3128`).
- Tests (`tests/hpipm/split_step_3b.cpp`): hand-computed `compute_alpha` cases
  (no-binding, dual-binding only, primal-binding only, both-binding, and the
  both-negative quadratic Pass-2 case); an `update_vars` case asserting
  `ux`/`t` use `alpha_prim` and `pi`/`lam` use `alpha_dual`; a 5-iteration
  predictor (affine) loop with `split_step=1` on DI/MS (N=1,2 dynamic + MS
  NH=2 fixed-extent) asserting both alphas stay > `alpha_min`, `lam`/`t` stay
  above their floors, and all values stay finite.
- Deviations: (i) Pass 3 re-applies only the complementarity adjustment, not
  the Pass-1 feasibility clipping, per the C source. (ii) The predictor-loop
  test uses a well-conditioned fill (contractive dynamics `A = 0.5·I`, small
  `DC`/`grad`/`d`) rather than the adversarial random fill of `alpha_1f`: the
  split-step (asymmetric `α_p ≠ α_d`) predictor trajectory is more aggressive
  and the adversarial fill drives the iterate to a singular stage Cholesky
  within 5 iterations. (iii) The test does **not** assert residual decrease:
  a predictor-only affine loop with the split dual step does not guarantee
  monotone `res_g` / `res_b` decrease (line-search backtracking plus faster
  dual movement can increase them); the guaranteed invariants (Cholesky
  success, alpha floors, `lam`/`t` floors, finiteness) are asserted instead.
- Also updated `tests/hpipm/alpha_1f.cpp` and
  `tests/hpipm/centering_1g.cpp` for the `compute_alpha` void-return +
  `alpha_prim()` / `alpha_dual()` accessors (behavior unchanged; the
  `split_step=0` values are identical).
- Build: warning-free under `-Wall -Wextra -Werror`. All targets green:
  `qp_unit` (incl. the new `split_step (3b)` suite), `sqp_unit`,
  `double_integrator`, `mass_spring`, `sqp_double_integrator`,
  `sqp_mass_spring`, `sqp_acados_ref`.

### 3c — HPIPM `warm_start ≥ 2` (+ driver wiring)

- Re-read `x_ocp_qp_ipm.c:1631-1794` (`OCP_QP_INIT_VAR` warm-start branches)
  and the driver `ocp_qp_hpipm.c:327-336` (ux zeroing). Confirmed:
  `warm_start >= 3` keeps the whole iterate (ux, pi, lam, t) and clips
  lam/t to `lam0_min`/`t0_min`; `== 2` keeps pi/lam/t, clips to `thr0 = 1e-1`
  and does **not** zero ux (the driver zeroes ux unconditionally); `<= 1`
  zeroes ux/pi and inits lam/t (cold).
- `init_point` (`hpipm.hpp`) now branches on `opts_.warm_start` using three
  small mutators (`zero_primal`, `zero_dual`, `clip_lam_t`): `>= 3` clips to
  the warm floors and returns; `== 2` clips to `thr0` and returns; `<= 1`
  zeros primal+dual and runs the existing heuristic. New public setter
  `set_warm_start(int)` for the driver.
- **Driver wiring** (`sqp.hpp`): before each `qp_.solve(qp_in_, qp_out_)` the
  driver zeroes `qp_out_.ux_*` (primal always cold) and sets the QP solver's
  warm-start to `qp_warm_start` — except `warm_start = 0` on the very first
  QP when `warm_start_first_qp` is false. The absolute pi/lam/t from
  iteration k-1 become the IPM initial iterate for iteration k.
- Tests (`tests/sqp/warm_start_3c.cpp`):
  - `init_point` branch unit tests (ws=3 clip, ws=2 clip, ws=0 cold) on a
    small assembled DI QP.
  - QP-level warm start: a synthetic well-conditioned QP (PD Hessian
    `I + R'R`, contractive `A = 0.5*I`, wide non-binding bounds, the pin
    equality, `d_mask = 1`) solved cold (13 IPM iters) then hot re-solved
    (`warm_start=3`, 1 iter) and warm-dual re-solved (`warm_start=2`); asserts
    `hot iter < cold iter`, same `z`/`pi` to `1e-3`, `kSolved`.
  - Driver-level: `SqpSolver` with `qp_warm_start = 2` vs `0` on DI N=10 —
    both `kSolved`, same cost / terminal state, `total_ipm <= cold`.
- Deviations: (i) `init_point` for `== 2` does **not** zero ux (C-faithful;
  the driver does it) — the standalone QP test therefore seeds ux explicitly
  for the warm-dual case. (ii) The plan's "== 2 heuristic repair" is omitted:
  the C source only clips + returns (running `init_stage` would clobber the
  preserved multipliers via `lam = mu0/t`). (iii) `warm_start <= 1` is treated
  uniformly as cold (C keeps ux for `== 1`; the driver zeroes it, so they are
  equivalent once driven). (iv) The DI QP (N=10, ~40 vars) is small and
  well-conditioned, so the dual-warm start (primal always cold in the driver)
  shows no measurable IPM-iteration reduction; the driver test asserts
  correctness + no-regression (`<=`), while the QP-level synthetic test
  (hot start keeps the primal) demonstrates the 13 → 1 iteration reduction.
  The QP-level test uses a synthetic QP because the real DI QP is only PD
  after the driver adds the LM/regularization terms, which a standalone QP
  test does not replicate.
- Build: warning-free under `-Wall -Wextra -Werror`. All targets green:
  `qp_unit`, `sqp_unit` (incl. the new `warm_start (3c)` suite),
  `double_integrator`, `mass_spring`, `sqp_double_integrator`,
  `sqp_mass_spring`, `sqp_acados_ref`.

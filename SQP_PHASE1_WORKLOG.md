# Phase 1 worklog: HPIPM QP solver (`include/ocp/solvers/hpipm/hpipm.hpp`)

Companion to `SQP_PLAN.md` (§5, §6, §11 Phase 1). This file is the running
record: verified facts, the derived Newton system, sub-step plan, and
checkpoints. **Protocol for every sub-step:**

1. Re-read the HPIPM functions listed for the sub-step (fresh; note line
   numbers in the checkpoint).
2. Implement in `include/ocp/solvers/hpipm/hpipm.hpp` (header-only).
3. Build (`cmake --build build`), run tests, warning-free under
   `-Wall -Wextra -Werror`.
4. Append a checkpoint entry below: what was done, test results, any
   deviation from this file (deviations win; update the fact here).
5. One commit per sub-step (`solver:` scope).

## 0. State

- Phase 0 done (commits `2e311bd`..`479fb90`): `include/ocp/solvers/hpipm/qp.hpp` (QpDim,
  QpStageFirst/Path/Term, Qp, QpSol, QpRes), `problem.hpp` Status enum
  (kSolved, kMaxIterations, kInfeasible, kQpFailure, kMinStep, kUnbounded,
  kNanDetected, kAborted, kTimeout) + HVP contract, `tests/qp_dim`.
- Phase 1: sub-steps 1a (scaffolding) and 1b (residuals) done —
  `include/ocp/solvers/hpipm/hpipm.hpp` (`HpipmOptions`, `HpipmIteration`,
  `HpipmStatistics`, `HpipmQpSolver<P, NH>` shell + `compute_residuals`).
  Tests live in `tests/hpipm/` (`qp_dim.cpp`, `qp_unit.cpp`,
  `residuals_1b.cpp`). Sub-steps 1c..1i pending.

## 1. Our model (re-verified from `qp.hpp`)

Per stage type, variable vector `z`:
- first/path: `z_k = (u_k; x_k; s_k)` (`s` = slack vars, one per soft side)
- term: `z_N = (x_N; s_N)` (no control; acados sets `nu[N] = 0`)

Row groups `[pin; bx; bu; ineq; eq; lin]`; side layout
`[lo sides; hi sides; slack sides]`. `d`: lo = `lo`, hi = `-hi`, slack = 0.
`d_mask`: 1 active, 0 absent. `DC` (nrow × nvar, natural orientation)
**includes the slack columns** (+1 lo side / −1 hi side); box/pin rows carry
unit-vector rows at the constrained variable. Hence for every row
`v = DC[r,:]·z` (box/pin included).

Side sign `σ_i`: `+1` for lo sides and slack sides, `−1` for hi sides.

### Residuals (verified against `x_ocp_qp_res.c`; matches `qp.hpp` header)

- `res_b_k = BA_k z_k + b_k − x_{k+1}` (k = 0..N−1)
- `res_d_i = d_i + t_i − σ_i v_{row(i)}` (lo: `d+t−v`; hi: `d+t+v`; slack: `d+t−s`)
- `res_g_k = H_k z_k + g_k + DC_kᵀ(σ∘λ_k) + BA_kᵀπ_k − pin(π_{k−1})`
  (the `−π_{k−1}` hits the x-part of z_k only; π_{−1} = 0)
- `res_m_i = λ_i t_i − m_i`
- `res_mu = Σ|res_m|·d_mask / Σd_mask`; `res_*_max` = inf-norms across stages
- `obj` = primal objective at the iterate; `dual_gap` — **VERIFY formula in
  `x_ocp_qp_res.c` during 1b** (primal − dual objective; exact terms TBD)
- OPEN: HPIPM's slack-side `res_m` may use `m_safe` instead of `m`
  (BALANCE: `m_safe = 0.5`). **VERIFY in `x_ocp_qp_res.c` during 1b.**

### Presets (BALANCE + acados overrides, `ocp_qp_hpipm.c:101-113`)

- res_g_max = 1e-6; res_b_max = res_d_max = res_m_max = 1e-8
- iter_max = stat_max = 50; alpha_min = 1e-8; mu0 = 1; reg_prim = 1e-12
  (verify reg default in `hpipm_d_ocp_qp_ipm.h`)
- dual_gap_max = 1e15 (HPIPM default, `x_ocp_qp_ipm.c:83`; acados does NOT
  override → effectively inactive in v1)
- lam_min = t_min = 1e-16; tau_min = 1e-16; lam0_min = t0_min = 1e-9;
  m_safe = 0.5; t_lam_min = 2; t0_init = 2; cond_pred_corr = 1;
  split_step = 0; square_root_alg = 1; update_fact_exit = 1;
  var_init_scheme = 1; warm_start = 0
- v1 skips: lq_fact = 0, itref = 0, use_Pb cache

### Exit test (`x_ocp_qp_ipm.c` ~:3119)

Converged (→ kSolved) when all hold:
`α > alpha_min`, `‖res_g‖∞ ≤ res_g_max`, `‖res_b‖∞ ≤ res_b_max`,
`‖res_d‖∞ ≤ res_d_max`, `‖res_m − tau_min·d_mask‖∞ ≤ res_m_max`,
`dual_gap ≤ dual_gap_max`. Loop bound `kk < iter_max` (else kMaxIterations);
`α ≤ alpha_min` → kMinStep; Cholesky failure after reg schedule → kQpFailure;
NaN → kNanDetected.

## 2. Newton system (derived this session; toy-verified)

Per iteration, per stage k (relative form; δ-variables are steps):

- `γ_i = (res_m_i − λ_i·res_d_i)/t_i`, `Γ_i = λ_i/t_i` (masked sides: 0)
  — HPIPM `COMPUTE_GAMMA` (`x_core_qp_ipm_aux.c`)
- Stage matrix: `M_k = H_k + reg·I + DC_kᵀ·diag(Γ_k)·DC_k`
  (FULL nvar_k × nvar_k — includes slacks; slack sides enter through their
  DC row = e_s and coupling columns automatically)
- RHS: `r̃_k = −res_g_k − DC_kᵀ(σ∘γ_k)`
  (= HPIPM's `−l̃` with `l̃ = res_g + DCᵀ(γ_l − γ_u) + γ_slack·e`)
- KKT linear system per stage:
  `M_k δz_k + BA_kᵀ δπ_k − δπ_{k−1} = r̃_k`  (π_{−1} = 0; BA-term absent at k = N)
  `δx_{k+1} = BA_k δz_k + res_b_k`
- Closed forms after δz:
  `δt_i = σ_i DC[r(i)] δz − res_d_i`
  `δλ_i = γ_i − Γ_i·σ_i DC[r(i)] δz`
  δπ backward (uses M_k only, no factor):
  `δπ_{N−1} = (M_N δz_N)_x − r̃_{N,x}`
  `δπ_{k−1} = (M_k δz_k)_x + A_kᵀ δπ_k − r̃_{k,x}`, k = N−1..1
  (consistency check: `(M_k δz_k)_u + B_kᵀ δπ_k ≈ r̃_{k,u}`)

### Two-pass solve (to re-derive + oracle-check in 1e)

Standard OCP Cholesky Riccati (HPIPM: backward `OCP_QP_FACT_SOLVE`,
forward in `solve_kkt`; see `acados/docs/algorithm/05-hpipm-qp-solver.md`
and `x_ocp_qp_ipm.c` ~:1400-1560):

- Backward: `L_N = chol(M_N)`; for k = N−1..0:
  `M̃_k = M_k + BA_k·(L_{k+1}L_{k+1}ᵀ)_{xx-block}·BA_kᵀ`, `L_k = chol(M̃_k)`
- Forward: `L_0 δz_0 = r̃_0`; `L_{k+1} δz_{k+1} = r̃_{k+1} + BA_k·L_{k+1,xx}ᵀ δz_k`

**The exact forward formula and the x-block indexing of L with slacks
present are the open math items** — re-derive in 1e, validate against the
dense full-KKT oracle (tol 1e-10). The δπ closed form above needs no factor.

### Decision D1 (recorded)

HPIPM Schur-eliminates slacks (`Zs_inv`, `Γ_eff = Γ − ΓZs⁻¹Γ`; its L is
(nu+nx+1)×(nu+nx), `x_ocp_qp_ipm.c:1024`). Our `DC` already contains the
slack columns, so we keep slacks **explicit** in `M_k` (full dense Cholesky
over (u;x;s)). Mathematically equivalent; simpler; oracle-verified in 1e.
No `idxs_rev` coupling code in v1 (plan Q6).

## 3. Mehrotra / line search (details re-verified in owning sub-step)

- 1f: `COMPUTE_ALPHA` (`x_core_qp_ipm_aux.c`), single α (split_step = 0):
  feasibility ratios over res_d (t vs res_d), complementarity ratios
  (m_zero/tau_min handling — re-read exact `m_zero` semantics), λ≥0/t≥0
  guards; `α ≤ alpha_min` → kMinStep.
- 1f: `UPDATE_VAR`: `z += α_p δz`, `π += α_d δπ`, `λ += α_d δλ`,
  `t += α_p δt` (here α_p = α_d = α); if α < 1: `α ← α(0.99(1−α) + 0.9999999α)`;
  clip t,λ to ≥ t_lam_min (2) — **verify exact clip rule**.
- 1g: `COMPUTE_MU_AFF`: `mu_aff = (1/nc)·Σ|−m_i + (λ_i + δλ^aff_i)(t_i + δt^aff_i)|`
  (affine/predictor step, m → 0 target; slack sides per m_safe convention).
- 1g: `COMPUTE_CENTERING`: `σ = (mu_aff/mu)³`; corrector target
  `res_m_bkp + δt⊙δλ − σ·mu`; pure centering fallback when cond_pred_corr
  and mu_aff worsened — **re-read exact condition + backup protocol**
  (the `res_m ← res_m − tau_min·d_mask` predictor shift and its backup).
- 1c: `OCP_QP_INIT_VAR` (var_init_scheme = 1, t0_init = 2, warm_start = 0):
  re-read exact λ/t initialization.

## 4. Sub-steps (each = one commit + checkpoint entry)

- **1a scaffolding**: `hpipm.hpp` with `HpipmOptions` (defaults from §1),
  `HpipmIteration`, `HpipmStatistics` (21 columns per plan §6), workspace
  struct (per-stage storage as `Trajectory<T, traj_extent<NH, ...>>`,
  allocated in `solve`), `HpipmQpSolver<P, NH>` class shell (`solve`,
  `options()`, `statistics()`). CMake: add `tests/qp_unit` target (trivial
  main that includes hpipm.hpp). Verify: compiles warning-free.
- **1b residuals**: `compute_residuals(const Qp&, const QpSol&) → QpRes`
  (§1 formulas; resolve dual_gap + m_safe questions). Test: hand-computed
  residuals for a small synthetic staged QP (N = 1 and N = 2, with/without
  slacks), zero-solution and random-solution cases.
- **1c init point**: interior (t > 0, λ > 0, d_mask respected),
  `res_mu ≈ mu0`. Test: init + `compute_residuals` on the synthetic QPs.
- **1e KKT solve**: `fact_solve_kkt` (two-pass, §2; reg_prim schedule:
  1e-12, ×10 per retry, 100 tries → kQpFailure) + `solve_kkt` (reuse factor
  for the corrector step) + δt/δλ/δπ closed forms. Test: KKT linear
  residual ~1e-12; δz matches dense full-KKT Eigen oracle (assemble the
  full block KKT in the test, FullPivLU); δπ consistency check; forward
  formula recorded here (§2). [1d merged: oracle helper lives in tests.]
- **1f alpha + update**: `compute_alpha`, `update_vars`. Test: manual
  5-iteration loop on a small QP; λ/t stay ≥ 0; residuals decrease.
- **1g centering**: `compute_mu_aff`, `apply_centering` (record exact
  predictor/corrector targets + backup protocol in §3 after re-read).
  Test: mu_aff < mu on a toy; corrector m-target equals hand formula.
- **1h main loop**: `solve()` (iterate 1b→1c-once→[γ→1g→1e→1f→update];
  exit test §1; statistics; fast path: no active inequality sides at all →
  single unconstrained KKT solve, iter = 0, kSolved). Test: end-to-end
  convergence on the synthetic QPs (all §1 tolerances).
- **1i test suite**: `tests/hpipm/` — (1) staged LQR-like QP with known
  analytic solution (hard constraints only, double-integrator-like data),
  (2) soft rows + slacks (verify optimum satisfies weighted-slack KKT),
  (3) fixed x_0 pin rows, (4) terminal ineq/eq/lin + terminal box,
  (5) unbounded/infeasible edge case → status codes. All cross-checked vs
  the dense oracle + KKT residuals below tolerances.

## 5. Checkpoints

### 1a — scaffolding (2026-09-30)

Sources re-read:
- `hpipm_d_ocp_qp_ipm.h:59-96` — `d_ocp_qp_ipm_arg` fields (all mirrored in
  `HpipmOptions`); `:100-149` — `d_ocp_qp_ipm_ws` fields (Gamma/gamma, Pb,
  L/Ls/P, stat, iter, stat_m, use_Pb, status).
- `x_ocp_qp_ipm.c:69-221` — `OCP_QP_IPM_ARG_SET_DEFAULT`: SPEED_ABS
  `:75-109`, SPEED `:110-144`, BALANCE `:145-179`, ROBUST `:180-214`.
  BALANCE values confirmed: mu0 1e1, alpha_min 1e-12, res_g 1e-6, res_b/d/m
  1e-8, dual_gap_max 1e15 (`:153`), iter_max = stat_max = 30, reg_prim 1e-15
  (`:160`), pred_corr = cond_pred_corr = 1, itref_corr_max 2, lq_fact 1,
  lam_min = t_min = tau_min 1e-16, lam0_min = t0_min 1e-9, warm_start 0,
  square_root_alg 1, split_step 0, m_safe 0.5, var_init_scheme 0, t_lam_min
  2, t0_init 2, update_fact_exit 1.
- `x_ocp_qp_ipm.c:804-805, :986` — `stat_m = 21`; stat allocated
  `21 x (1 + stat_max)` (row 0 = pre-iteration; rows written under guard
  `kk+1 < stat_max`; row `stat_max` allocated but never written — matched).
- `x_ocp_qp_ipm.c:3096-3209` — main loop: pre-iteration residuals into stat
  row 0 (`:3099-3107`); loop condition `kk < iter_max & alpha_prim >
  alpha_min & alpha_dual > alpha_min & (res_g > tol | res_b > tol | res_d >
  tol | res_m_tau > res_m_max | dual_gap > dual_gap_max)` (`:3119-3128`);
  status mapping (`:3182-3209`): `kk == iter_max` → MAX_ITER,
  `alpha <= alpha_min` → MIN_STEP, `isnan(mu)` → NAN_SOL, else SUCCESS.
- doc 05 §5.3 (`:436-444`) — the 21 stat columns in print order:
  alpha_prim_aff, alpha_dual_aff, mu_aff, sigma, alpha_prim, alpha_dual, mu,
  res_stat, res_eq, res_ineq, res_comp, dual_gap, obj, lq_fact, itref_pred,
  itref_corr, lin_res_stat, lin_res_eq, lin_res_ineq, lin_res_comp,
  singular (npd_reg_hess) → field names of `HpipmIteration`.
- acados overrides `ocp_qp_hpipm.c:101-113` — the file is code-generated and
  not present in this checkout; verified via doc 05 §6.2 (`:502-509`):
  iter_max = stat_max = 50, alpha_min 1e-8, mu0 1e0, var_init_scheme 1
  (residual tolerances unchanged).

Implemented:
- `include/ocp/solvers/hpipm/hpipm.hpp`: `HpipmOptions` (defaults = BALANCE +
  acados overrides + v1 minimal flags per plan §2), `HpipmIteration`
  (21 fields, doc 05 §5.3 order), `HpipmStatistics` (`stat_max + 1` rows,
  HPIPM stat-matrix semantics), `HpipmQpSolver<P>` shell: per-stage-type
  aliases, `Workspace` (M/L factors, Pb cache, `QpRes`, gamma/Gamma, t_inv,
  rhs, dv/dpi/dlam/dt, IPM scalars; `resize(N)` allocates path vectors),
  `solve()` stub (returns `kAborted` until 1h), `options()` /
  `statistics()` accessors. Planned private methods listed in comments.
- CMake: `tests/qp_unit` target (`-Wall -Wextra -Werror`).
- `tests/qp_unit/main.cpp`: option defaults, 21-field iteration record,
  statistics layout, accessors, `solve()` stub status; instantiated for
  DoubleIntegrator and MassSpring.

Tests: `double_integrator`, `mass_spring`, `qp_dim`, `qp_unit` all build
warning-free and pass.

Refactor (user request 2026-09-30): `include/ocp/solver/` renamed to
`solvers/` (more solver families planned); `hpipm.hpp` and the
HPIPM-specific `qp.hpp` now live together in `solvers/hpipm/`
(`include/ocp/solvers/hpipm/{qp,hpipm}.hpp`); include paths in the tests
and all plan/worklog/AGENTS references updated.

Refinement (same sub-step, user request 2026-09-30): path-wise workspace
quantities now use the `Trajectory` mechanism instead of plain
`std::vector` — `HpipmQpSolver<P, NH>` is templated over the horizon like
`Qp` / `QpSol` / `QpRes`; per-stage storage is
`Trajectory<T, traj_extent<NH, offset>()>` (runtime-extent by default,
fixed-extent `std::array` for compile-time `NH`; `Workspace::resize`
allocates only in dynamic mode). `solve(const Qp<P, NH>&, QpSol<P, NH>&)`
resizes the workspace on entry. Noted in AGENTS.md (Conventions) and
ARCHITECTURE.md (sec. 5); plan sec. 2 "Internal solver storage" bullet
revised accordingly. `tests/qp_unit` also instantiates the solver with
`NH = 5` (fixed-extent path compiles + runs).

Deviation from this file: §1 listed `reg_prim = 1e-12 (verify ...)` —
verified default is **1e-15** (`x_ocp_qp_ipm.c:160`, BALANCE; acados does
not override it). Also, the §4/1e note "reg_prim schedule: 1e-12, x10 per
retry, 100 tries" is NOT an HPIPM OCP-level behavior: HPIPM's fact status is
handled inside the delta-step macro and no reg-growth retry loop exists at
this level (the acados SQP driver re-invokes the QP solver). To resolve in
1e: mirror the HPIPM behavior (single factorization attempt per iteration,
Cholesky failure → report failure status; reg growth only if we choose a
more robust policy).

### 1b — residuals (2026-09-30)

Sources re-read:
- `x_ocp_qp_res.c:345-531` — `OCP_QP_RES_COMPUTE`: SYMV_L `beta = 2.0`
  objective trick (`:431-438`, then `AXPY -1` on grad → `res_g = H z + g`);
  `−π_{k−1}` on the x-part (`:440`); `tmp_nbgM = λ_hi − λ_lo` (`:444`); box
  sparse add + extract (`:452-456`); general rows: one GEMV for both
  `res_g += DCtᵀ(λ_hi − λ_lo)` and `v = DCt·z` (`:458`);
  `res_d = d + t − v (lo) / d + t + v (hi)` (`:460-466`); slack block:
  `GEMV_DIAG` (`:469`), `−λ_s` (`:485`), `idxs_rev` loop:
  `res_g(s_lo) −= λ_lo`, `res_g(s_hi) −= λ_hi`,
  `res_d(lo) −= s_lo`, `res_d(hi) −= s_hi` (`:487-493`),
  `res_d(slack) = d + t − s` (`:495`); `dual_gap −= dᵀ(λ∘mask)` (`:497`);
  dynamics: `res_b = BA z + b − x_next`, `res_g += BAᵀπ_k`,
  `dual_gap −= bᵀπ_k` (`:505-510`); `res_m = (λ∘t − m)∘d_mask`
  (`:513-516`); inf-norms + `res_mu` in a separate pass
  (`OCP_QP_RES_COMPUTE_INF_NORM`, `:689`).
- `x_core_qp_ipm_aux.c:59-133, 137-162, 164-189` — `COMPUTE_GGAMMA_QP` /
  `COMPUTE_GAMMA_QP`: `γ = (res_m − λ res_d)/t`, `Γ = λ/t` (with
  `t_lam_min`/`lam_min` clamping); `COMPUTE_LAM_T_QP`:
  `δλ = −t⁻¹(res_m + λ·Δv_σ − λ·res_d)`, then `dt −= res_d` (the applied
  t-step).
- `x_ocp_qp_kkt.c:1866-1886` — signed constraint-value change `Δv_σ`
  before `COMPUTE_LAM_T_QP`: `+DC·δz` on the lo half, sign-flipped copy on
  the hi half, slack expansion after.

Implemented:
- `hpipm.hpp`: public `compute_residuals(const Qp<P, NH>&, const
  QpSol<P, NH>&, QpRes<P, NH>&)` + private `stage_residuals` (per stage)
  + private static `inf_norm` helper. Per-stage: `res_g = H z + g`,
  multiplier coupling `res_g.head(nux) += DC_noxᵀ(λ_hi − λ_lo)`
  (masked), per-side slack-column coupling via `D::idxs_lo_*/idxs_hi_*`
  (`−λ_lo`, `−λ_hi`) and `−λ_s` per slack side, `res_b = BA z + b − x_next`,
  `res_g += BAᵀπ_k` (u;x part) and `−pin(π_{k−1})` (x part);
  `res_d` per side (`d + t − v_lo` / `d + t + v_hi`, `d + t − s`), masked;
  `res_m = d_mask ∘ (λ ∘ t − m)`; `obj = Σ 0.5 zᵀHz + gᵀz`;
  `dual_gap = Σ zᵀ(Hz + g) − dᵀ(λ∘d_mask) − Σ_{k<N} bᵀπ_k`;
  `res_mu = Σ|res_m|·mask / Σmask`; `res_*_max` inf-norms over the fully
  coupled stage residuals (separate pass, as in HPIPM).
- Test reorg (user request): all HP solver tests moved into a single flat
  `tests/hpipm/` directory (mirroring `include/ocp/solvers/hpipm/`);
  `main.cpp` files renamed to their target name.
- `tests/hpipm/residuals_1b.cpp`: zero-iterate case with hand-computed
  expectations (N = 1) + random-data/random-iterate cases against an
  independent element-looped reference (`ref_residuals`); DI N = 1, 2
  (soft ineq → slacks) and MS N = 1, 2 + MS NH = 2 (no slacks,
  fixed-extent).

Tests: `qp_unit` (all 1b checks, tol 1e-10), `qp_dim`, `double_integrator`,
`mass_spring` build warning-free and pass.

Deviations / corrections to this file:
- §1 `res_g` formula `DC_kᵀ(σ∘λ_k)` assumed HPIPM's full per-side DC (lo
  rows sign-flipped). In our compact storage (one hi-oriented DC row per
  constraint) the (u;x) coupling is `DC_noxᵀ(λ_hi − λ_lo)` and the
  slack-column coupling is explicit: `res_g(s_lo) −= λ_lo`,
  `res_g(s_hi) −= λ_hi`, `res_g(s_j) −= λ_s`. The stored DC slack columns
  (+1 lo / −1 hi) are NOT used in the residual; they enter only the KKT
  matrix `M_k` (1e). `nux = nvar − nslack`; the coupling touches
  `res_g.head(nux)` only.
- §1 `dual_gap` resolved (was OPEN): `Σ_k z_kᵀ(H z_k + g_k)
  − Σ_k d_kᵀ(λ_k ∘ d_mask_k) − Σ_{k<N} b_kᵀπ_k`
  (`x_ocp_qp_res.c:439, :497, :508`).
- §1 `m_safe` OPEN resolved: `res_m` uses `m` (not `m_safe`)
  (`x_ocp_qp_res.c:513-516`); `m_safe` matters only in centering (1g).
- §2 closed form `δλ_i = γ_i − Γ_i·σ_i DC[r(i)] δz` is wrong (sign of the
  γ term, and Δv/δt conflation). Verified:
  `δλ_i = −γ_i − Γ_i·Δv_{σ,i}` where `Δv_{σ,i} = σ_i·(side value)_i change`
  (`σ_lo = +1`, `σ_hi = −1`), equivalently
  `δλ_i = −γ_i − Γ_i·δt_i − Γ_i·res_d_i` with the applied t-step
  `δt_i = Δv_{σ,i} − res_d_i` (`x_core_qp_ipm_aux.c:181-182`,
  `x_ocp_qp_kkt.c:1866-1884`).
- Plan §6 listed `compute_residuals` as private; made it public for
  testability (also needed by 1c/1h).

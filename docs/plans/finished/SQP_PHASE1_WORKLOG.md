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
- Phase 1: sub-steps 1a (scaffolding), 1b (residuals), 1c (init point),
  1e (KKT solve), 1f (alpha + update), 1g (centering), 1h (main loop)
  done — `include/ocp/solvers/hpipm/hpipm.hpp`
  (`HpipmOptions`, `HpipmIteration`, `HpipmStatistics`,
  `HpipmQpSolver<P, NH>` shell + `compute_residuals` + `init_point` +
  `fact_solve_kkt` / `solve_kkt` / `solve_kkt_unconstr` +
  `mask_step` / `compute_alpha` / `update_vars` +
  `compute_mu_aff` / `apply_centering` + `solve` (IPM main loop +
  unconstrained fast path)). Tests live in
  `tests/hpipm/` (`qp_dim.cpp`, `qp_unit.cpp`, `residuals_1b.cpp`,
  `init_1c.cpp`, `kkt_1e.cpp`, `alpha_1f.cpp`, `centering_1g.cpp`,
  `solve_1h.cpp`, `suite_1i.cpp`).

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

### 1c — init point (2026-09-30)

Sources re-read:
- `x_ocp_qp_ipm.c:1632-2049` — `OCP_QP_INIT_VAR`: hot-start (warm_start
  >= 3: clip lam/t to lam0_min/t0_min) and warm-start-2 (clip to thr0 =
  1e-1) branches are out of v1 scope (warm_start = 0). Cold start zeros
  `ux` (`:1697-1708`) and `pi` (`:1725-1732`). `t0_init = 0/1` set
  lam = t = sqrt(mu0) / (mu0, 1.0) uniformly (`:1734-1760`); v1 uses
  `t0_init = 2` (heuristic, `:1761-2042`).
- `x_ocp_qp_ipm.c:1905-2040` — the heuristic branch used by acados
  (var_init_scheme = 1): per stage, in order —
  1. slacks: `t_s = -d_s + s` (`AXPY :1936`), clip `t_s < thr0` →
     `t_s = thr0`, `s = d_s + thr0` (`:1937-1949`);
  2. box rows: `t_lb = x[idxb] + s_lo(soft) - d_lb`,
     `t_ub = -x[idxb] + s_hi(soft) - d_ub` (`:1954-1967`); repair
     (`:1970-1997`): both violated → `x = 0.5*(d_lb - d_ub)`,
     both `t = thr0`; only lo → `t_lb = thr0`, `x = d_lb + thr0`; only
     hi → `t_ub = thr0`, `x = -d_ub - thr0`;
  3. general rows: `t_lg = C x + s_lo(soft) - d_lg`,
     `t_ug = -C x + s_hi(soft) - d_ub`... (exact: `t_ug = -v + s_hi - d_ug`,
     `:2003-2030`), clipped to `max(thr0, t)` (no variable repair);
  4. `lam = mu0 / t` on every side (`:2035-2036`).

Implemented:
- `hpipm.hpp`: public `init_point(const Qp<P, NH>&, QpSol<P, NH>&)` (zeros
  the cold-start primal + pi, then per stage-type calls private
  `init_stage` with the box-type (row, varidx) list: pin rows via
  `idx_x0` (first stage only), bx/bu via `idxb_*`; terminal only bx via
  `idxb_term`). `init_stage` implements the 4-step heuristic above with
  thr0 = 1e-1 (`x_ocp_qp_ipm.c:1655`).
- `tests/hpipm/init_1c.cpp` (linked into `qp_unit`): feasible [-1, 1]
  synthetic data (hess = I, DC = 0, d = -1 on lo / hi sides, d = 0 on
  ineq hi and slack sides, d_mask with every 5th side = 0); checks
  t > 0 and lam > 0 finite on all sides, t >= thr0, finite decision
  variables, and `res_mu == mu0` (1e-12) after `compute_residuals`.
  Cases: DI N = 1, 2 (soft ineq → slacks), MS N = 1, 2 + MS NH = 2
  (no slacks, fixed-extent).

Tests: `qp_unit` (all 1a+1b+1c checks), `qp_dim`, `double_integrator`,
`mass_spring` build warning-free and pass.

Deviations / corrections to this file:
- Plan §6 listed `init_point` as private; made it public for testability
  (the 1c test calls it directly, same precedent as 1b's
  `compute_residuals`).
- Sec. 3 said "re-read exact λ/t initialization" for 1c — done: the exact
  heuristic is recorded above; note the both-violated repair sets
  `x = 0.5*(d_lo - d_hi)` (midpoint of the bound interval, since
  d_lo = lo, d_hi = -hi), matching `x_ocp_qp_ipm.c:1978`.

### 1e — KKT solve (two-pass Riccati Cholesky) (2026-10-01)

Sources re-read:
- `x_ocp_qp_kkt.c:316` — `OCP_QP_FACT_SOLVE_KKT_UNCONSTR`: unconstrained
  fast path (no inequality rows); same two-pass with Gamma = gamma = 0.
- `x_ocp_qp_kkt.c:1117` — `OCP_QP_FACT_SOLVE_KKT_STEP`: gamma/Gamma
  (`:1162`), terminal factor `L_N = chol(M_N)` with `DIARE(nu+nx, preg)`
  (`:1227`) — regularization added to the (u;x) diagonal only, slacks
  unregularized — then slack conditioning / Gamma-diagonal update /
  `SYRK_POTRF` (`:1230-1253`); middle stages build `AL = D_k L_{k+1,xx-rows}`
  and fold `AL·ALᵀ` into `M_k` before `SYRK_POTRF_LN_MN` (`:1277-1315`);
  singular diagonal → `preg` retry schedule (`:1173-1210`): first retry
  `preg += 1e-4` (or `/3` if already boosted), later `×100` (or `×8`).
- `x_ocp_qp_kkt.c:2306` — `OCP_QP_SOLVE_KKT_STEP`: forward pass only,
  reusing the stored factors (Mehrotra corrector / refinement).
- `x_core_qp_ipm_aux.c:38` — `COMPUTE_GAMMA_GAMMA_QP`:
  `Gamma_i = lam_i/t_i` (floored at `t_lam_min`/`lam_min` when
  `t_lam_min` flag), `gamma_i = (res_m_i − lam_i·res_d_i)/t_i`, masked
  sides → 0.
- `x_core_qp_ipm_aux.c:164` — `COMPUTE_LAM_T_QP`:
  `dlam_i = −(res_m_i + lam_i·(Δv_i − res_d_i))/t_i` with applied
  `δt_i = Δv_i − res_d_i` — exactly our closed forms.

Implemented (`hpipm.hpp`):
- `fact_solve_kkt` (`:600`): `ws_.resize(N)`, `compute_gamma`, then three
  passes — (1) backward factorization `k = N..0`: `M̃_N = M_N`,
  `M̃_k = M_k + D_kᵀ·P_{k+1}·D_k`, `L_k = chol(M̃_k)` (Eigen::LLT over the
  full (u;s;x) block; failure → `kQpFailure`; HPIPM's preg retry schedule
  is deferred to 1f), `P_k = L_k(xx)·L_k(xx)ᵀ`; (2) backward reduction
  `k = N..0`: `r̃_k = r̃⁰_k − D_kᵀ(P_{k+1}·res_b_k + q_{k+1})` and
  `q_k = −r̃_k,x + L_k(x,w)·z`, `z = L_k(w,w)⁻¹·r̃_k,w`; (3) forward
  `k = 0..N`: `δz_k = M̃_k⁻¹(r̃_k + e_x·δπ_{k−1})` (two triangular solves
  with `L_k`), `δπ_k = P_{k+1}(D_k·δz_k + res_b_k) + q_{k+1}`; then
  closed forms `δt = C·δz − res_d`,
  `δλ = d_mask∘(−(res_m + λ∘δt)/t)` per stage.
- `solve_kkt` (`:635`): forward pass only, reusing stored `L_k`/`P_k`/`q_k`
  (HPIPM `OCP_QP_SOLVE_KKT_STEP`); caller resets gamma/Gamma for the
  corrector (1g).
- `solve_kkt_unconstr` (`:656`): `fact_solve_kkt` with Gamma = gamma = 0
  (no inequality sides; δt = δλ = 0).
- Stage helpers `factor_one_stage` / `reduce_one_stage` /
  `forward_one_stage` take the stage dims as non-type template params
  (`NU, NX, NSK` first, before deduced types) so all intermediates are
  fixed-size Eigen; the (u;s;x) permutation is done by
  `usx_permute_mat` / `usx_permute_vec` / `usx_build_side_dc`.

Tests: `tests/hpipm/kkt_1e.cpp` (linked into `qp_unit`): per case,
assembles the full coupled Newton system
`[M_k + D_kᵀP_{k+1}D_k | D_kᵀ; −D_k | M_{k+1} + ...] [δz; δπ] = r̃` in
dynamic-size Eigen (test-only) and solves with `FullPivLU` as an
independent oracle; compares solver `δz`/`δπ` (rel tol 1e-9), the
`δt`/`δλ` closed forms, and verifies `solve_kkt` (factor reuse) agrees
with a fresh `fact_solve_kkt`. Cases: DI N = 1, 2 (soft ineq → slacks),
MS N = 1, 2 + MS NH = 2 (no slacks, fixed-extent).

Tests: `qp_unit` (all 1a+1b+1c+1e checks), `qp_dim`,
`double_integrator`, `mass_spring` build warning-free and pass.

Deviations / corrections to this file:
- §2 two-pass formula corrected: `P_k` is the (x,x) block of `L_k` times
  its transpose — `L_k(xx)·L_k(xx)ᵀ` — which equals the Schur complement
  `M̃_xx − M̃_xw·M̃_ww⁻¹·M̃_wx` of the (w,x) split. The tentative §2 formula
  used the full `(L_{k+1}L_{k+1}ᵀ)_{xx}` block (= `M̃_{k+1,xx}`, includes
  `L_xw·L_xwᵀ`), which is NOT equivalent and gives wrong `δz`. The
  forward pass solves the full `M̃_k` stage system (two triangular
  solves) rather than §2's `L_{k+1,xx}ᵀ`-coupled update; both describe
  the same solution, oracle-verified.
- §2 `M_k = H + reg·I + ...`: reg is added to the (u;x) diagonal only,
  never to slack variables — matches HPIPM's `DIARE(nu+nx, preg)`
  (`x_ocp_qp_kkt.c:1227`).
- §2 open item ("x-block indexing of L with slacks present") resolved:
  with (u;s;x) ordering the x block is last, so `L` is block-lower-
  triangular `[[L_w,w  0],[L_x,w  L_x,x]]` and the indexing above is
  exact; oracle-verified.
- Singular-factor handling: v1 returns `kQpFailure` on a failed Cholesky;
  HPIPM's preg retry ladder (`:1173-1210`) is not wired yet (1f).

### 1f — alpha + update (2026-10-01)

Sources re-read:
- `x_core_qp_ipm_aux.c:193-468` — `COMPUTE_ALPHA_QP`: split_step = 0 branch
  (`:375-449`); single alpha from 1.0; per side, feasibility `alpha =
  -lam/dlam` (dlam < 0) and `alpha = -t/dt` (dt < 0); complementarity when
  `m != 0`: `m1 = m_safe * m` (clamped `m_safe` in [0,1]), if
  `lam1*t1 - m1 < -1e-12` then quadratic `a = dlam*dt`,
  `b = dlam*t + lam*dt`, `d = b^2 - 4ac` with `c = lam*t - m1` (`c > 0` ->
  `alpha = (-b - sqrt(d))/(2a)`; else `alpha = 0`). `m_zero` is passed in
  from the OCP layer.
- `x_ocp_qp_ipm.c:2966-2973` — `m_zero` computed as the inf-norm of the raw
  (unmasked) `m` being exactly 0; v1 always has m = 0, so `m_zero = 1`.
- `x_core_qp_ipm_aux.c:472-582` — `UPDATE_VAR_QP`: alpha damping
  `alpha_p = alpha_p * ((1-alpha_p)*0.99 + alpha_p*0.9999999)` when
  `alpha < 1` (`:504-510`, applied to alpha_prim AND alpha_dual, with
  split_step = 0 both equal); `v += alpha_p dv`, `pi += alpha_d dpi`,
  `lam += alpha_d dlam`, `t += alpha_p dt`; `t_lam_min = 2` adds the floors
  `lam >= lam_min`, `t >= t_min` (`:545-561`).
- `x_ocp_qp_ipm.c:2263-2286` — DELTA_STEP prelude: `BACKUP_RES_M` +
  `COMPUTE_TAU_MIN_QP` (`x_core_qp_ipm_aux.c:756-778`: `res_m =
  res_m_bkp - tau_min`, all entries) then the KKT step; step masking
  `d_mask ∘ sol_step->t`, `d_mask ∘ sol_step->lam` after the factorization
  (`:2284-2285`), BEFORE `COMPUTE_ALPHA` and `UPDATE_VAR`.
- `x_ocp_qp_ipm.c:3085-3163` — main loop: residuals -> (exit test uses
  `res_m_tau = ||res_m - tau_min*d_mask||_inf` computed at `:3115` and
  `:3160`) -> delta step. The tau shift is applied to the workspace res_m
  once per iteration (via `COMPUTE_TAU_MIN_QP`) and is NOT a separate user
  step; our test mirrors it by shifting the local `res.res_m_*` before
  `fact_solve_kkt`.

Implemented (`hpipm.hpp`):
- `mask_step(const Qp&, QpSol& step)` (public, `:716`): `step.lam_*` /
  `step.t_*` multiplied in place by `d_mask` (the `ux` step is NOT masked,
  matching HPIPM's commented-out `ux` masking at `:2283`).
- `compute_alpha(const Qp&, const QpSol& iter, const QpSol& step)`
  (public, `:740`): single alpha (split_step = 0), processes stages in
  order (first, path k = 1..N-1, term); `m_zero` from the raw m inf-norm
  (per-stage max), `m_safe` clamped to [0,1]; uses the (already masked)
  step's `lam`/`t` as dlam/dt; stores result in `ws_.alpha`; returns it.
  Feasibility + complementarity exactly as in the HPIPM split_step = 0
  branch.
- `update_vars(QpSol& iter, const QpSol& step)` (public, `:822`): reads
  `ws_.alpha` (set by `compute_alpha`); damping for `alpha < 1`;
  `ux += alpha_p step.ux`, `pi += alpha_d step.pi`, per-stage
  `lam += alpha_d step.lam` / `t += alpha_p step.t` with the `t_lam_min = 2`
  floors (clipped to `lam_min` / `t_min`) via the private helper
  `update_lam_t` (`:877`).

Tests: `tests/hpipm/alpha_1f.cpp` (linked into `qp_unit`):
- `compute_alpha` hand cases (d_mask = 1): no binding -> 1; dual binding
  (lam = 1, dlam = -2) -> 0.5; primal binding (t = 1, dt = -2) -> 0.5;
  complementarity (m = 1, lam = t = 1, dlam = dt = -0.5) -> 2 - sqrt(2).
  DI and MS, N = 1.
- `mask_step`: even indices kept, odd zeroed, on a 2-stage DI.
- `update_vars`: damped (alpha = 0.5) update of ux / pi / lam / t matches
  `alpha_p = 0.5*(0.99*0.5 + 0.9999999*0.5)`; lam/t floors clip a
  sub-`lam_min` lam to `lam_min` at alpha = 1.
- 5-iteration predictor-only (affine) loop for DI N = 1, 2, MS N = 1, 2,
  MS NH = 2: init -> [residuals -> res_m -= tau_min -> fact_solve_kkt ->
  mask_step -> compute_alpha -> update_vars] x 5; checks
  alpha > alpha_min, lam/t >= floors, finiteness, and that res_g_max and
  res_b_max decrease over the loop.

Tests: `qp_unit` (all 1a+1b+1c+1e+1f checks), `qp_dim`, `double_integrator`,
`mass_spring` build warning-free and pass.

Deviations / corrections to this file:
- §3 1f said "complementarity ratios (m_zero/tau_min handling)" — the
  m_zero gate is a pure `m == 0` inf-norm check from the OCP layer
  (`x_ocp_qp_ipm.c:2966-2973`), unrelated to tau_min; tau_min only enters
  the res_m shift (`COMPUTE_TAU_MIN_QP`). v1 always has m = 0.
- §4 1f listed the preg retry ladder as deferred to 1f; it remains deferred
  to 1h (the main loop), not 1f — the single-shot `fact_solve_kkt` from 1e
  is sufficient for the predictor steps tested here.
- `mask_step` / `compute_alpha` / `update_vars` are public (plan §6 listed
  them private), same precedent as 1b/1c/1e: needed by the 1f test and by
  the 1h main loop.
- HPIPM masks `sol_step->lam` / `sol_step->t` AFTER the KKT solve and
  reuses that masked step for both `COMPUTE_ALPHA` and `UPDATE_VAR`; we do
  the same via `mask_step` (called once per iteration between
  `fact_solve_kkt` and `compute_alpha`).

### 1g — Mehrotra centering (compute_mu_aff + apply_centering) (2026-10-01)

Sources re-read:
- `x_core_qp_ipm_aux.c:636-668` — `COMPUTE_MU_AFF_QP`:
  `mu_aff = (1/nc_mask) * Σ_{d_mask=1} | -m + (lam + a_d·dlam)·(t + a_p·dt) |`
  over the active sides; `a_p = a_d = ws->alpha` (split_step = 0); the
  complementarity RHS is `qp->m` (NOT the workspace `res_m`); result stored
  in `ws->mu_aff`.
- `x_core_qp_ipm_aux.c:672-690` — `BACKUP_RES_M`: copies `ws->res->res_m_*`
  into `ws->res_m_bkp` (unshifted complementarity residual) each iteration.
- `x_core_qp_ipm_aux.c:695-724` — `COMPUTE_CENTERING_CORRECTION_QP`:
  `sigma = (mu_aff/mu)^3`; `sigma_mu = max(sigma·mu, tau_min)` (floor only,
  HPIPM does NOT scale tau_min by the (mu/mu_ref)^3 factor here);
  `res_m = d_mask ∘ (res_m_bkp + dt ∘ dlam - sigma_mu)`.
- `x_core_qp_ipm_aux.c:729-751` — `COMPUTE_CENTERING_QP` (pure centering,
  used when the corrector target would worsen `mu_aff`):
  `res_m = d_mask ∘ (res_m_bkp - sigma_mu)` (no `dt∘dlam` term).
- `x_core_qp_ipm_aux.c:756-778` — `COMPUTE_TAU_MIN_QP`: the predictor shift
  `res_m -= tau_min` (all entries; the OCP layer applies it on the
  unmasked `res_m`).
- `x_ocp_qp_ipm.c:2060-2228` — `OCP_QP_IPM_DELTA_STEP`: the predictor step
  sequence `backup_res_m -> compute_tau_min -> FACT_SOLVE_KKT ->
  mask step -> COMPUTE_ALPHA` (→ the 1f `compute_alpha`), then
  `COMPUTE_MU_AFF`, then the corrector (or pure centering) target is built
  into `res` and `OCP_QP_SOLVE_KKT_STEP` reuses the stored factors.
- `x_ocp_qp_kkt.c:2306-2346` — `OCP_QP_SOLVE_KKT_STEP`: a leading
  `COMPUTE_GAMMA_QP` (`:2346`) recomputes `Gamma`/`gamma` from the
  (centering-shifted) `res_m` before the forward pass reuses the factors.

Implemented (`hpipm.hpp`):
- `compute_mu_aff(const Qp<P, NH>&, const QpSol<P, NH>& iter, const
  QpSol<P, NH>& step)` (public, after `update_vars`): the HPIPM
  `COMPUTE_MU_AFF_QP` masked mean; `a_p = a_d = ws_.alpha`; stores
  `ws_.mu_aff`; returns it. Must be called after `compute_alpha` (reads
  `ws_.alpha`).
- `apply_centering(const Qp<P, NH>&, const QpRes<P, NH>& res_bkp, const
  QpSol<P, NH>& step, bool correction, QpRes<P, NH>& res_out)` (public):
  sets `sigma = (mu_aff/mu)^3`, `sigma_mu = max(sigma·mu, tau_min)`;
  corrector: `res_out.res_m = d_mask ∘ (res_m_bkp + dt∘dlam - sigma_mu)`;
  pure centering: `d_mask ∘ (res_m_bkp - sigma_mu)`. Reads `mu` from
  `res_bkp.res_mu` and `ws_.mu_aff` (from `compute_mu_aff`); stores the
  unfloored `sigma` in `ws_.sigma`. Writes only the `res_m_*` fields of
  `res_out` (the caller fills the other residuals first).
- `solve_kkt` now calls `compute_gamma(in, iter, res)` at entry (was:
  "caller must have set `ws_.gamma_*`/`ws_.Gamma_*`"), matching HPIPM's
  leading `COMPUTE_GAMMA_QP` in `OCP_QP_SOLVE_KKT_STEP`
  (`x_ocp_qp_kkt.c:2346`); the trailing "remaining HPIPM flow" comment now
  lists only `OCP_QP_IPM_DELTA_STEP` + main IPM loop → 1h.
- **Bug fix (`qp.hpp`)**: `QpStageFirst` / `QpStagePath` / `QpStageTerm`
  got a default constructor calling `setZero()`. The per-member `{}`
  initializers do NOT zero fixed-size Eigen matrices (`Eigen::Matrix` has a
  user-provided default ctor, so it is not an aggregate and `{}` only runs
  that ctor, leaving the data uninitialized). `first` / `term` are direct
  members of `Qp`, so they never received the `setZero()` that the `path`
  stages get via `TrajectoryStorage`'s ctor. Symptom: for `MS NH=2 N=2` the
  uninitialized `grad` corrupted `res_g`, giving a near-singular KKT step
  (`dlam` ~ 1e285, `alpha` ~ 5.7e-286 < `alpha_min`); the identical data
  under `Dynamic` extent happened to land on memory that did not break the
  check. All three structs now self-zero on construction. (Note: the same
  latent pattern exists for the direct `first`/`term` Eigen members of
  `QpSol` / `QpRes`, but every such member is written before it is read on
  all current paths, so left unchanged.)

Tests: `tests/hpipm/centering_1g.cpp` (linked into `qp_unit`):
- `compute_mu_aff` hand formula on uniform data (DI N=1): `m = 0` → 0.85,
  `m = 0.3` → 0.55, masked sides (active-only) → 0.85, all matching the
  independent `hand_mu_aff` reference to 1e-12.
- `apply_centering` hand checks (uniform data, DI N=1): corrector target
  `res_m = d_mask ∘ (res_m_bkp + dt∘dlam - sigma_mu)`, pure-centering
  target `res_m = d_mask ∘ (res_m_bkp - sigma_mu)`, the `sigma_mu`
  `tau_min` floor, and masked sides → 0; each matching the hand formula
  to 1e-12.
- Pipeline (one affine step, DI N = 1, 2; MS N = 1, 2 + MS NH = 2):
  `init_point -> compute_residuals -> res_m -= tau_min -> fact_solve_kkt ->
  mask_step -> compute_alpha -> compute_mu_aff`; checks `alpha > alpha_min`
  and `mu_aff < mu` (the affine step must reduce the barrier parameter).

Tests: `qp_unit` (all 1a+1b+1c+1e+1f+1g checks), `qp_dim`,
`double_integrator`, `mass_spring` build warning-free and pass.

Deviations / corrections to this file:
- §3 1g said "slack sides per m_safe convention" for `compute_mu_aff` —
  resolved: HPIPM's `COMPUTE_MU_AFF_QP` uses the raw `qp->m` on every
  active side (no `m_safe`), so v1 (m = 0) uses `m` directly.
- §4 1f deferred the corrector/predictor backup protocol to 1g; the exact
  corrector vs. pure-centering switch (the `cond_pred_corr` guard on a
  worsened `mu_aff`) is an OCP-layer decision and belongs to the 1h main
  loop — `apply_centering` just exposes both targets behind `correction`.

### 1h — main IPM loop (solve) (2026-10-01)

Sources re-read:
- `x_ocp_qp_ipm.c:2060-2228` — `OCP_QP_IPM_DELTA_STEP`: predictor step
  `backup_res_m -> compute_tau_min -> FACT_SOLVE_KKT_STEP -> mask step ->
  COMPUTE_ALPHA -> COMPUTE_MU_AFF`, then corrector (or pure centering) target
  into `res` + `OCP_QP_SOLVE_KKT_STEP` reusing the stored factors, then a
  second `COMPUTE_ALPHA` + `UPDATE_VAR`.
- `x_ocp_qp_ipm.c:3085-3209` — `OCP_QP_IPM_SOLVE` main loop: pre-iteration
  residuals into stat row 0; exit test `alpha > alpha_min & res_g <= tol &
  res_b <= tol & res_d <= tol & res_m_tau <= res_m_max & dual_gap <=
  dual_gap_max`; status mapping `:3182-3209`. Unconstrained fast path
  (`:2908-2950`): a single KKT solve, iter = 0.

Implemented (`hpipm.hpp`):
- `solve(const Qp<P, NH>&, QpSol<P, NH>&)` (`:350`): `ws_.resize(N)` +
  `stat_.init`; fast path when no side is active (`has_active_side`); else
  `init_point` + `mask_abs_lam` + pre-iteration `compute_residuals` into
  stat row 0, then the IPM loop. Per iteration: exit test → BACKUP_RES_M
  (`ws_.res_bkp`) + `shift_res_m(tau_min)` → `fact_solve_kkt` (predictor) →
  `mask_step` → `compute_alpha` (stat cols 0,1) → if `pred_corr`:
  `compute_mu_aff` (col 2) + corrector `apply_centering(correction=true)` +
  `solve_kkt` (factor reuse) + `mask_step` + `compute_alpha` (cols 4,5),
  and a `cond_pred_corr` pure-centering fallback when the corrector's
  `mu_aff` more than doubles the predictor's; then `update_vars` (1f),
  fresh `compute_residuals`, stat row kk+1 (cols 3, 6-12 + v1-zero cols
  13-20), `stat_.iter = kk + 1`. Status mapping: `iter_max` →
  kMaxIterations, `alpha <= alpha_min` → kMinStep, NaN `res_mu` →
  kNanDetected, else kSolved.
- Helpers: `has_active_side` (any `d_mask > 0.5`), `res_m_tau`
  (`||res_m − tau_min∘d_mask||_inf`), `mask_abs_lam` (post-init `lam *=
  d_mask`), `shift_res_m`, `fill_stat_residuals` (cols 7-12), `stat_row`
  (row kk+1, null when `kk+1 >= stat_max`), `solve_unconstr` (fast path:
  zero iterate, `solve_kkt_unconstr`, stat row 0, iter = 0).
- `Workspace` gained a `QpRes<P, NH> res_bkp` member (BACKUP_RES_M).

Tests: `tests/hpipm/solve_1h.cpp` (linked into `qp_unit`):
- `fill_qp` builds a feasibility-friendly synthetic QP per case (PD Hessian
  `I + RᵀR` on (u;x) + unit on slacks, small random DC, origin-interior
  box/linear bounds lo = −3 / hi = 3, eq at v = 0, ineq hi = 1, m = 0, all
  sides active, contractive dynamics 0.5·I on x + small u-part).
- Convergence cases (kSolved + every KKT residual within the option
  tolerances, incl. the tau-shifted complementarity; stat rows populated):
  DI N = 1, 2; MS N = 1, 2; MS NH = 2 (N = 2).
- Masked-sides case (every 5th side `d_mask = 0`, DI/MS N = 2): exercises
  `mask_step`/`mask_abs_lam`; checks kSolved, residuals within tolerance,
  and absent sides stay interior (`t >= t_min`, `lam >= 0`).
- Unconstrained fast path (all `d_mask = 0`, DI N = 1; MS N = 2 + MS NH = 2):
  iter = 0, kSolved, res_g/res_b at machine precision.

Tests: `qp_unit` (all 1a+1b+1c+1e+1f+1g+1h checks), `qp_dim`,
`double_integrator`, `mass_spring` build warning-free and pass.

Deviations / corrections to this file:
- §4 1h "fast path: no active inequality sides" — the actual HPIPM
  fast-path condition is `nc == 0` (no constraint rows at all), reached in
  our solver by `has_active_side` (no `d_mask > 0.5`). Since an all-zero
  `d_mask` means no active side, the two coincide; `solve_unconstr` is
  taken whenever every side is masked, matching HPIPM's unconstr branch.
- The 1a shell test previously asserted `solve()` returns `kAborted` (the
  stub); with `solve()` implemented that call would read uninitialized
  Eigen data on the empty QP, so the shell test now checks only the
   scaffolding accessors / stat layout and defers `solve()` coverage to the
   1h test.

### 1i — test suite (2026-10-01)

Sources re-read:
- `qp.hpp:22-34, 91-95` — the staged-QP contract: dynamics
  `x_{k+1} = BA_k z_k + b_k`; per-row sides `lo: v − d ≤ t` (d holds `lo`),
  `hi: −v − d ≤ t` (d holds `−hi`), so feasibility is `lo ≤ v ≤ −d_hi`; pin
  rows `d_lo = x0, d_hi = −x0`; soft rows carry a `+1` (lo) / `−1` (hi)
  slack column in DC; equality rows are two-sided with `lo == hi`.
- `double_integrator.hpp` / `mass_spring.hpp` — the two concrete Dims used
  by the suite (DI: nx 2, nu 1, soft ineq, terminal eq, fixed x0; MS:
  nx 8, nu 3, no soft, fixed x0).

Implemented: `tests/hpipm/suite_1i.cpp` (linked into `qp_unit`):
- `Stg<P, NH>` per-stage accessors (H/grad/BA/b/DC/d/d_mask, nvar/zoff,
  layout) over a `Qp`.
- `dense_oracle`: an independent dense KKT solve of the staged QP —
  stationarity (`H z + g + BAᵀπ − pin(π_prev)`), dynamics
  (`BA z + b − x_{k+1}`), pin rows (`x_0 = x0`), plus one multiplier per
  explicitly-listed active side/slack; `FullPivLU` with a KKT-residual
  guard. The reference the solver's primal/dual is cross-checked against.
- Case 1 (LQR cross-check, DI N = 2, 3; MS N = 2): `fill_lqr` builds a PD
  synthetic QP (PD `I + RᵀR` Hessian on (u;x) + unit on slacks,
  contractive 0.5·I dynamics, wide ±1e3 box/ineq/lin **and eq** sides so
  every non-pin side is interior); solver vs oracle z_k / π_k / objective;
  x_0 pin exact; non-pin hi sides interior.
- Case 2 (soft ineq, DI N = 1, 2): `fill_soft` makes one ineq the only
  active row (Jacobian `jg`, slack weight `w`, hi = 0); verifies the
  weighted-slack KKT `λ_hi == w·s`, `s > 0`, slack-side multiplier ~ 0,
  KKT residuals; for N = 1 cross-checks z_0 / z_1 / π_0 / λ_hi against the
  oracle with the ineq listed active.
- Case 4 (terminal, TermConstr N = 1, 2): `fill_terminal` with an active
  terminal equality and wide terminal ineq/lin/box; verifies the terminal
  equality `x_N(0) − x_N(1) = 0` and cross-checks all z_k / π_k / objective
  against the oracle with the terminal eq active.
- Case 5 (edge cases, DI): infeasible QP (contradictory control box
  lo > hi) → a failure status (observed `kMinStep`, status 5),
  non-PSD Hessian → `kQpFailure` (status 4),
  tiny `iter_max = 1` → `kMaxIterations` (status 2, `iter == 1`).

Design note (found while making cases 1/2 oracles agree with the solver):
the dense oracle lists only the *active* sides. A row whose hi side is set
tight (`d_hi = 0`, i.e. `v ≤ 0`) is active whenever `v > 0` at the optimum,
so it must be either (a) kept wide in the fill (cases 1 and 2 — every
non-target side is interior) or (b) listed as an active row in the oracle
(case 4 — the terminal equality). The first `fill_lqr` / `fill_soft`
versions left the synthetic equality rows tight (`d_hi = 0`) while their
oracles listed no active row, so the solver (solving the QP with that
equality active) disagreed with the oracle (ignoring it); the fix was to
make the synthetic equality rows wide in those two fills.

Tests: `qp_unit` now runs all 1a+1b+1c+1e+1f+1g+1h+1i checks; `suite_1i`
reports "All suite (1i) checks passed." Full `qp_unit`, `qp_dim`,
`double_integrator`, `mass_spring` build warning-free under
`-Wall -Wextra -Werror` and pass (exit 0).

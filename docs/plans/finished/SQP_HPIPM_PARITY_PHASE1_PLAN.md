# Plan: Phase 1 — full condensing (dense QP)

Status: **draft** (2026-10-10). Execution subplan for Phase 1 of
`SQP_HPIPM_PARITY_PLAN.md`: add acados' `FULL_CONDENSING_HPIPM` capability —
condense the staged SQP-QP to a dense QP on (x_0, u_0..u_{N-1}), solve it with
a new dense-QP IPM, expand the solution back to the staged layout.

Acados reference (re-read at implementation time):
`acados/ocp_qp/ocp_qp_full_condensing.c`,
`acados/external/hpipm/cond/x_cond.c` + `x_cond_aux.c`
(`COND_QP_COMPUTE_DIM`, `COND_BABT/BAT/B`, `COND_RSQRQ`/`COND_RQ`,
`COND_DCTD`/`COND_D`, `COND_EXPAND_SOL`),
`acados/external/hpipm/dense_qp/x_dense_qp_ipm.c`.

## 1. Data model decisions (resolved)

**Variable order.** Dense variable vector `v = [x_0; u_0; ...; u_{N-1}; s_0; ...; s_N]`
(ocp++ natural order; HPIPM uses `[u_{N-1};...;u_0;x_0]` — irrelevant for the
expanded solution, our order keeps the flow-map assembly straightforward).

**Slacks are regular dense variables.** ocp++'s staged model already treats
slacks as plain variables (in `hess`, `DC`, `grad`), one per soft *side*.
The dense QP keeps that: the slack block of `H` is diagonal (the soft-penalty
weights), and each slack carries a non-negativity side (`s >= 0`, lo side
only, hi side masked).

**Per-side slack coupling (C_lo / C_hi).** The staged OCP couples each side
to its *own* slack only (HPIPM `x_ocp_qp_res.c`: `res_d[lo] = d + t − v − s_lo`,
`res_d[hi] = d + t + v − s_hi`). A single shared general-row Jacobian with
both the +1 s_lo and −1 s_hi columns would cross-couple the two sides
(`s_lo` appearing in the hi constraint and vice versa) and define a
*different* problem. The dense model therefore stores two general-row
Jacobians: `C_lo` (carries only the +1 s_lo column of each soft row) and
`C_hi` (carries only the −1 s_hi column); both share the same (x_0; u)
flow-map coefficients. The signed side matrix is `A = [box; +C_lo; −C_hi]`.
Consequence: `ns_dense = 0` in the HPIPM sense — no `idxs`/`idxs_rev`
slack-pair machinery, no slack Schur elimination in the dense KKT.

**Dense dimensions** (compile-time in `DenseQpDim<P>`):
```
nv  = nx + N*nu + nslack_first + (N-1)*nslack_path + nslack_term
```
(nslacks from `QpDim`; N = horizon. In `NH == Dynamic` mode the dense model
is heap-backed: `Eigen::MatrixXd`/`VectorXd` — sanctioned exception to the
fixed-size-Eigen rule, since nv is runtime-sized; `NH >= 1` gets fixed-size.)

```
nb = nbox_first + (N-1)*nbox_path + nslack_total
     // nbox_first: pin rows (fixed x_0) + HARD state box (free x_0) +
     //             hard control box
     // nbox_path:  hard control box, k = 1..N-1; plus the slack
     //             non-negativity rows
ng = (ng + ne + nl + nbx_soft + nbu_soft)    // stage-0: soft state / control
                                               // box + ineq/eq/lin
   + (N-1)*(nbx + ng + ne + nl + nbu_soft)   // k = 1..N-1: state box (becomes
                                               // general) + soft control box
                                               // + ineq/eq/lin
   + (nbx_t + ng_t + ne_t + nl_t)            // terminal: everything is general
ne_dense = 0                                 // dynamics eliminated by substitution
```
Soft box rows (a `*_soft_idx` entry) become *general* rows: their staged DC
row carries the +1 / −1 slack columns, which a dense box row (one `idxb`
variable) cannot express.  Only hard box rows keep the box-row representation.
Side layout: `[lo sides (nb+ng); hi sides (nb+ng)]`, i.e. `d/d_mask/m/lam/t`
have `2*(nb+ng)` entries. Slack non-negativity rows: lo side active (d = 0),
hi side masked (d_mask = 0). Pin rows (fixed x_0) condense to ordinary box
rows on x_0 (lo == hi) — `x_0` stays a dense variable in both fixed and
free cases (parity-plan decision Q4).

**`DenseQp<P, NH>`** fields: `H` (nv×nv symmetric, lower storage), `g` (nv),
`C_lo` / `C_hi` (each ng × nv, general rows; lo row carries the +1 s_lo slack
column, hi row the -1 s_hi column — see "Per-side slack coupling" above),
`idxb` (nb ints),
`d, d_mask, m` (2*(nb+ng)). **`DenseQpSol`**: `v` (nv), `lam`, `t`
(2*(nb+ng) each); no `pi`. **`DenseQpRes`**: `res_g` (nv), `res_d`, `res_m`
(2*(nb+ng)), inf-norm maxima, `res_mu`, `obj`, `dual_gap`.

## 2. Condense (staged → dense)

Workspace (rebuildable per call, heap in dynamic mode):
- Flow maps `Gamma_k` (k = 1..N): rows = `[x_0; u_0; ...; u_{k-1}]`,
  columns = components of x_k, with `x_k = Gamma_k · [x_0; u_{0..k-1}] + gamma_k`
  (step formulation: x_k here is the *step* variable).
   Recursion: `Gamma_1 = [I_x0 | B_0]`, `gamma_1 = b_0`;
   `Gamma_{k+1} = [A_k · Gamma_k; B_k]`, `gamma_{k+1} = A_k gamma_k + b_k`
   (`A_k`/`B_k` from `BA_k = [B_k | A_k]`, k = 0..N-1; `b = A gamma + b` is
   exactly HPIPM's `COND_BABT`).
   (The implementation stores the transpose `G_k = Gamma_k'` — size
   `nvh x nx`, `G_k(v, d) = dx_k[d]/dv`; the u-row updates are then
   `G_k.rows · A_k'`.)

   **Implementation note:** the code stores the transpose
   `G[k] = Gamma_k'` (size `nvh x nx`, `G[k](v, d) = dx_k[d]/dv`), so the
   recurrence reads `G[k+1].rows = G[k].rows · A_k'` (x0 and prior-u blocks)
   and `G[k+1](u_k block) = B_k'`.  ⚠ The prior `A · G[k].rows` order was a
   bug: it is only valid when `A` is symmetric / `nact == nx`; for
   `k·nu != nx` it is a dimensionally *invalid* product (assert compiled out
   under `-DNDEBUG`, giving OOB reads).  Corrected to `G[k].rows · A'`.

1. **H and g** (direct flow-map substitution, not HPIPM's L-recursion):
   substituting `x_k = G[k]' v_hard + gamma_k` into each staged objective
   gives, per stage (hard part only; `E_k` embeds `u_k` and `x_k`):
   ```
   H += sum_k E_k' H_k,ux E_k      // = sum E_k' H_k E_k incl. slack cols
   g += sum_k E_k' (H_k,ux c_k + g_k,ux),  c_k = [0; gamma_k]
   ```
   concretely, per stage k (0..N-1), `nact = nx + k·nu`, `Gk_a = G[k](0:nact)`:
   ```
   H(uu)      += H_k(uu)
   H(u,x)     += H_k(ux) Gk_a'   (and its transpose)
   H(top-left nact) += Gk_a · H_k(xx) · Gk_a'
   g(u block) += H_k(ux) gamma_k + g_k(u)
   g(head nact) += Gk_a · (H_k(xx) gamma_k + g_k(x))
   ```
   plus the terminal `H += G[N] H_N(xx) G[N]'`, `g += G[N](H_N(xx) gamma_N + g_N(x))`.
   Slacks: `H(s_k, s_k) = diag(stage-k hess slack diagonal)`;
   `g(slack) = stage-k grad slack part` (zero for standard soft constraints).
   (HPIPM's L-recursion / square-root variant is algebraically the same sum;
   deferred as a micro-opt. The unit test cross-checks against the explicit
   `E_k' H_k E_k` / `E_k'(H_k c_k + g_k)` substitution.)
  2. **C_lo / C_hi (general rows).** Stage 0: soft box rows + ineq/eq/lin
     rows — copy the (x_0; u_0) part of `DC` (hard box rows go to box
     instead). Stage k ≥ 1: for a row with staged Jacobian `[C_x | C_u]`:
     `C = [C_x · Gamma_k | C_u]` (x part replaced by its flow-map
     expression). Terminal stage: `C_x = I_xN` part of `DC`, same rule with
     `Gamma_N`. Slack columns: `C_lo` carries the row's +1 s_lo column only,
     `C_hi` the row's −1 s_hi column only (opposite column zeroed) — per-side
     slack coupling as in the staged OCP (see "Per-side slack coupling").
  3. **idxb (box rows).** Only the *hard* box rows keep the box-row
     representation: stage-0 pin rows + hard state box (free x_0) + hard
     control box, plus the path hard control box (k = 1..N-1): variable
     indices into the x_0 / u_k blocks. Slack rows: index of the slack
     variable. (Soft box rows become general rows — see "Dense dimensions".)
4. **d / d_mask / m.** Direct-variable box rows: copied verbatim. General
   rows (incl. converted state-box rows) get the flow-map constant shift,
   per ocp++ side convention (lo: `d' = d − C_x γ_k`; hi: `d' = d + C_x γ_k`,
   k ≥ 1; k = 0 has no shift). `d_mask` / `m` copied verbatim.
   (Equals HPIPM's `COND_D` incl. the `+γ` upper-side shift,
   `x_cond_aux.c:975-977, 1084-1086`.)
5. **API:** `condense_full(const Qp<P,NH>&, DenseQp<P,NH>&)` (one-shot;
   HPIPM splits LHS/RHS for SQP warm-start caching — see open item 3).

## 3. Expand (dense → staged)

- **Primal.** `u_k = v[u_k]`, `x_0 = v[x_0]`, slacks from `v`. States by
  forward simulation (the QP dynamics, step semantics — same as the staged
  solver's forward pass): `x_{k+1} = BA_k (u_k; x_k) + b_k`, k = 0..N-1.
- **Dynamics multipliers** (HPIPM `COND_EXPAND_SOL` dual part, re-derived in
  ocp++'s convention from `hpipm.hpp:567-577`, where
  `res_g_k = ... + BA_k' π_k − pin(π_{k−1})`, `res_b = BA z + b − x_{k+1}`):
  stationarity gives `π_{k−1} = (res_g_k^{no-π})_x + A_k' π_k`, with the
  terminal seed `π_{N−1} = (res_g_N^{no-π})_x`. Implementation: call the
  existing `stage_residuals` per stage with `pi_prev = pi_cur = nullptr`
  (and `res_b = nullptr`), then the one-line backward sweep above.
- **lam / t.** Map dense sides back to staged sides: stage-0 sides 1:1;
  path/terminal control-box sides 1:1; converted state-box rows read from the
  dense *general* side region; general rows 1:1; slack sides 1:1. (HPIPM:
  `x_cond_aux.c:2367-2468`.)

## 4. Dense QP solver (`HpipmDenseQpSolver<P, NH>`, in `hpipm.hpp`)

Options: reuse `HpipmOptions` (`kkt_fact_alg`, `square_root_alg`,
`lq_fact`, `remove_lin_dep_eq` are N/A — no equalities, no dynamics; ignore
with a comment). Residuals:
```
res_g = H v + g − A' (d_mask ⊙ λ)
res_d = d + t − A v      (A: signed side matrix, lo rows +, hi rows −)
res_m = d_mask ⊙ (λ ⊙ t) − m
```
(ocp++ sign convention, matching `stage_residuals`; the dense KKT uses the
same γ/Γ definitions as `compute_gamma`, hpipm.hpp:2705-2743.)

KKT (ne = 0, Schur complement only):
```
W = H + reg_prim I + A' diag(Γ) A
[ W  −A' ] [dv]   [ −res_g − A' γ ]
[ A   0  ] [dt] = [  res_d       ]
dt = A dv then dλ = −(1/t)(res_m + (d_mask⊙λ)⊙dt − (d_mask⊙λ)⊙res_d), dt −= res_d
```
Factorization: `chol(W)` (reg growth on singularity, HPIPM schedule
`x_dense_qp_ipm.c:1237-1309`). Loop structure identical to the staged
solver's delta loop: res → factor+solve → alpha (reuse `compute_alpha`
pattern) → Mehrotra P-C + conditional pure centering (KKT re-solve, reuse)
→ update → exit test. `abs_form`: support later (delta first).

## 5. Driver hook

`AcadosOptions`: `enum class QpMode { kStaged, kFullCondensed }` (default
`kStaged` — zero regression). In `AcadosSolver::solve`, after QP scaling +
regularization (same order as acados: scale → condense → solve → expand,
`ocp_nlp_sqp.c:594-597`): dispatch to
`condense_full → HpipmDenseQpSolver::solve → expand_full_sol`, then the
existing `rescale_solution` path. Condense/expand wall time recorded in
`SqpIteration` (new `cond_time` column).

## 6. Execution blocks

1. **1a — data model.** `dense_qp.hpp` (`DenseQpDim<P>`, `DenseQp`,
   `DenseQpSol`, `DenseQpRes`). Test: `tests/hpipm/dense_qp_dim.cpp` —
   dimension counts vs hand-computed values for the mass_spring/double
   integrator dims, both fixed and free x_0.
2. **1b — condense. ✅** `condense.hpp::condense_full` (+ `expand_full_sol`).
   H/g assembled by **direct flow-map substitution** (per-stage `E_k' H_k E_k`
   / `E_k'(H_k c_k + g_k)` blocks), not the L-recursion. Test:
   `tests/hpipm/condense_full.cpp` part 1 — assembled dense QP (H, g, C_lo/C_hi,
   idxb, d, d_mask, m) matches an *independent* in-test reference (explicit
   flow maps + the same `E_k' H_k E_k` / `E_k'(H_k c_k + g_k)` substitution,
   d shifts) to ≤ 1e-12. Coverage: `TinyFree` / `TinyFixed` (nx=2, nu=2,
   N=1 & 4, dynamic + fixed NH=3, soft box/ineq rows, both pin modes),
   `DoubleIntegrator` N=5, `MassSpring` NH=4. ASan-clean (guards the flow-map
   block products that are dimensionally invalid under the old `A·G` order).
3. **1c — dense solver. ✅** `HpipmDenseQpSolver` (delta form, Cholesky + reg
   growth, P-C + conditional centering) in `hpipm.hpp`, plus the unconstrained
   fast path (all d_mask = 0 → single `(H + reg I_(x0;u)) v = -g` factor,
   stat iter = 0) and the d_mask-aware cold-start init (t floored on masked
   sides before lam = mu0/t). Test: `tests/hpipm/dense_unit.cpp` —
   (i) `compute_residuals` vs an independent Eigen evaluation;
   (ii) `kkt_step` at a random iterate matches the closed-form Newton
   relations (W dv = rhs, dt = A dv - res_d, dlam = d_mask .(...) ./ t; the
   full KKT block [ W -A'; A 0 ] is singular for nside > nv and cannot be
   used as reference); (iii) full IPM solve of a random dense QP (m = 0) to
   the option tolerances, incl. hot warm-start convergence;
   (iv) unconstrained fast path vs the analytic `(H + reg I)⁻¹(-g)`.
   ASan-clean; full suite green.
4. **1d — expand + integration. ✅** `expand_full_sol` (primal: u / x_0 / s
   from v, states by forward simulation; π by the terminal-seeded backward
   sweep; lam / t by the inverse of the dense side layout).
   Test: `tests/hpipm/condense_full.cpp` part 2 — condensed solve +
   expansion vs the staged `HpipmQpSolver` on the same QP:
   - primal (u, x, s) agrees to ≤ 1e-5 (unique: PD Hessian);
   - staged `compute_residuals` at the expanded solution: res_g ≤ 1e-5,
     res_b ≤ 1e-9, res_d ≤ 1e-7, res_m ≤ 1e-5 (this is the π-formula
     check); π agrees to ≤ 1e-4;
   - lam / t are compared after `normalize_lam` / `normalize_t` (test
     helpers): the dual is not fully unique — on masked sides (d_mask = 0)
     lam / t are free, and on degenerate rows (lo == hi, both sides active,
     e.g. pin rows) KKT fixes only (λ_hi − λ_lo), not the split; the
     normalizers zero the free entries and collapse a degenerate pair to
     its difference. Unmasked, non-degenerate lam agrees to ≤ 1e-3, t to
     ≤ 1e-5.
   Implementation notes found while bringing this block to green:
   - soft box rows (stage-0 bx / bu, path bu) must condense to *general*
     rows — a dense box row (single idxb) cannot carry the +1 s_lo / −1 s_hi
     slack columns (root cause of the original t / λ / res_d failures);
     see "Dense dimensions";
   - the dense solver now masks `out.lam` by `d_mask` right after
     `init_point` (parity with the staged `mask_abs_lam`; HPIPM masks
     `qp_sol->lam` after init), so masked-side multipliers stay 0 instead
     of keeping their `mu0 / t` init value.
   Coverage: `TinyFree` / `TinyFixed` N = 3 (dynamic + fixed NH=3),
   soft state-box + soft ineq + masked lo sides, both pin modes.
5. **1e — driver. ✅** `QpMode` enum (`kStaged` / `kFullCondensed`) in
   `acados.hpp` + `AcadosOptions.qp_mode` (default `kStaged`, all existing
   behavior unchanged). `AcadosSolver::solve_qp` dispatches the main-loop QP
   solve and the second-order-correction re-solve; the full-condensed path
   runs `condense_full` (on the scaled, regularized staged QP) →
   `HpipmDenseQpSolver::solve` → `expand_full_sol` into the staged output.
   Dual warm start (qp_warm_start = 2/3) persists in the dense workspace
   (`dense_qp_out_` lam / t; primal v stays cold, same convention as the
   staged path). `SqpIteration.cond_time` records the condense + expand
   wall clock (0 in staged mode); not added to the acados-parity CSV.
   The dense solver gained `set_residual_tolerances` (maps res_b_max onto
   res_d_max; the dense QP has no separate box residual).
   Test: `tests/sqp/qpmode_1e.cpp` (bundled in `sqp_unit`) —
   DoubleIntegrator, dynamic N = 10 and fixed NH = 10 (N < 10 is
   infeasible for the terminal equality, see sqp_double_integrator.cpp):
   kStaged vs kFullCondensed both kSolved, x / u agree ≤ 1e-6, cost ≤ 1e-8;
   kFullCondensed + qp_warm_start = 2 converges to the same solution;
   cond_time > 0 only in condensed mode. Regression: full suite green.
   Also fixed a latent compile error in `condense_full` for stages with a
   compile-time 0-row `DC` (Eigen row-block `.segment` static assert; the
   driver dispatch instantiates `condense_full` for every problem type) —
   the general-row emitters now use coefficient access.
 6. **1f — E2E. ✅** `run_variant` in `tests/sqp_acados_ref` takes a
    `QpMode` (default `kStaged`); `main` runs boxonly and termeq in both
    modes against the FULL_CONDENSING_HPIPM references. The condensed runs
    match the references at cost rel ≤ 1.7e-13 and (x,u) inf-norm ≤ 2.2e-6
    (same as staged; thresholds < 1e-4). `tests/sqp/e2e_1f.cpp` (bundled in
    `sqp_unit`, which now links `MC_MODEL_C` for the MassesChain DAE
    residual): DoubleIntegrator N = 10 and MassesChain N = 40 (NLP tol 1e-6)
    solved from identical warm starts in both modes agree ≤ 1e-8 —
    DI dx 1.6e-14 / du 2.2e-15 / rel_cost 2.6e-14;
    MC dx 3.6e-11 / du 3.6e-10 / rel_cost 9.4e-15.
7. **1g — docs. ✅** ARCHITECTURE.md gains section 8 ("Solver-internal
    formulation"): the QP formulation (staged vs full-condensed dense) is a
    solver-internal choice behind `AcadosOptions.qp_mode`, not part of the
    problem interface. README.md solver bullets updated (staged + dense HPIPM
    QP solvers; SQP QP subproblem mode selectable via `qp_mode`).
    `qp_mode` registered in DUAL_ACCESS_PLAN.md's `AcadosOptions` inventory
    (33 flat scalars + `QpMode qp_mode` enum; the plan's "Enums" open
    decision already covers registration via the underlying int — the
    registry itself is not yet implemented, so this is the pending-list
    route).

Gate for every block: `cmake --build build` warning-free
(`-Wall -Wextra -Werror`), full test suite green.

## 7. Open items (verify during implementation, not design)

1. **LQ fallback / itref for the dense KKT.** Phase 1 ships Cholesky + reg
   growth only; add `lq_fact` / iterative refinement if the staged-solver
   comparisons ever need it (the staged solver's LQ exists for exactly the
   ill-conditioned-Hessian cases).
2. **`abs_form`** for the dense solver (HPIPM dense default presets offer a
   SPEED_ABS mode). Delta form first; abs if benchmarks ask.
3. **SQP warm-start caching** (HPIPM `cond_hess` / LHS-once-per-Hessian /
   RHS-per-iteration split, `ocp_qp_full_condensing.c` warm-start path).
   Only if profiling shows condense cost matters (N ≥ ~50 horizons).
4. **x_0 fixed: substitution alternative.** Currently the pin rows condense
   to a lo==hi box row on x_0 (keeps x_0 in v). A follow-up may eliminate
   x_0 from v entirely (nv −= nx) when fixed, matching HPIPM's
   `reduce_eq_dof` path.
5. **`update_fact_exit`** (exit factorization for sensitivities) — deferred
   with Phase 4 (sensitivities).

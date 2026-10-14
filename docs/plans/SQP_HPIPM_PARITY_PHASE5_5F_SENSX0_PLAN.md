# Plan: hard-equality KKT treatment + OCP-level initial-state sensitivity (Phase 5, item 5f)

Status: **completed** (2026-10-14). `AcadosSolver::sens_x0` (item 5f of
`SQP_HPIPM_PARITY_PLAN.md` §8) is unblocked and verified against central
finite differences (`tests/sqp/sens_x0_5f.cpp`, DI N=10, rel. err < 1e-6).

**Final approach (supersedes §3, the saddle design).** The saddle WIP
(d_mask = 2, saddle two-pass extension, LDLT for *every* factorization, LQ
disabled for eq problems) was retired. The shipped design is:

- **Main solve unchanged** — the staged QP keeps the baseline barrier
  two-pass, with equality rows barrier-ized as ordinary two-sided rows
  (`d_mask = 1`, `assemble.hpp`). No `d_mask = 2`, no saddle block.
- **Sens path only** — `HpipmQpSolver::sens_solve` gets a *separate*
  hard-equality block-KKT (`fact_solve_kkt_hardeq`, full staged KKT with the
  equality rows as first-class rows and free multipliers, Eigen `LDLT`, LHS
  cached in `ws_` across sens calls). It is taken **only** when
  `eqs_hard(in)` is true: every row in `D::eq_rows_*` has `d(sl) + d(sh) ≈ 0`
  (lo == hi, a genuine equality). When the eq-group rows are ordinary
  inactive boxes (the 4d synthetic QP, where the barrier two-pass is correct
  and well-conditioned), the two-pass is used instead.
- **No `seed.b[0] = df_dx·dx0`** in `AcadosSolver::sens_x0`: the stage-0
  dynamics matrix (`BA`) already carries the `x_0` column, so the pin-row
  `d` perturbation alone propagates `dx0`; seeding `b[0]` would double-count.
- **LQ re-enabled for all problems** (the obsolete `!D::has_eq` gate in
  `delta_factor_solve` was removed): the main solve treats eq rows as
  barrier rows, which the LQ wide-matrix factor handles.

`QpDim` keeps the `n_eq_*` / `eq_rows_*` / `has_eq` data model (§3.1);
`assemble.hpp` does **not** set `d_mask = 2` (equalities stay barrier rows in
the main solve). The dense/block/RTI paths (§3.5–3.6) are **out of scope**
and unchanged; 5f is staged-only. The dead saddle WIP (`*_saddle` functions,
`ws_.Ceq_*/Sinv_*/dpi_eq_*`) is removed in the close-out.

## 1. Goal (item 5f)

First-order sensitivity of the OCP solution w.r.t. the fixed initial state,
as a linear feedback law:

- API: `AcadosSolver::sens_x0(problem, sol, dx0, dsol)` — reassembles the
  staged QP at the converged SQP iterate, seeds the QP sensitivity
  (`seed.b[0] = df_dx(x0,u0)·dx0`, pin-row `d` offsets `+dx0/−dx0`), and runs
  `HpipmQpSolver::sens_forward` (HPIPM seed module, Phase 4).
- Test: `tests/sqp/sens_x0_5f.cpp` (DI, N=10, x0 ± h=1e-6, central FD of the
  full SQP re-solve; `du/dx0`, `dx/dx0` rel. err < 1e-6; pinned
  `dx_0 == e_i` exact). N < 7 is infeasible (terminal eq unreachable with
  |u| ≤ 1 from x0 = [1; 0.5]).

**WIP already on disk (uncommitted, branch `irk_hess` @ `9440d15`):**
`acados.hpp` `sens_x0` (+63 lines), `tests/sqp/sens_x0_5f.cpp` (new),
`tests/sqp/sqp_unit.cpp` + `CMakeLists.txt` wiring. Build is clean; base solve
and the inner QP solve pass; `sens_forward` returns `kQpFailure`.

## 2. Root cause (verified by instrumentation)

Equality rows in the staged QP (`g_eq`, and `g_pin` when
`fixed_initial_state`) are barrier-ized as two-sided rows (`d_mask = 1` both
sides, slacks `t_lo, t_hi`). At the converged QP, active equality rows have
`t → 0` with finite `λ`, so `Γ = λ/t → ∞`. Measured on DI N=10 at
`μ ≈ 3e-10`: terminal equality `q_N+v_N=1` (row `[1,1]`, 2-var terminal
stage) gives `Γ ≈ 3.2e17`, and the stage matrix
`M = H + D'ΓD + reg·I` becomes numerically rank-deficient (the O(1) Hessian
eigenvalue in the `[1,−1]` direction is below `ulp(Γ) ≈ 30`). Cholesky in
`factor_one_stage` (terminal stage, `k = N`) fails inside the
`sens_solve` exit refactor (`fact_solve_kkt_reg` with
`reg = ws_.preg_last = 1e-15`). A zero-seed `sens_forward` fails identically
→ the factorization is broken, not the seed.

The main SQP/QP solve converges fine: its last Newton factorization happens
one iterate before the final `t_eq`, and the reg ladder can still repair;
only the *exit* refactor at the fully converged solution is singular.

**HPIPM parity finding.** HPIPM's OCP QP has no equality dimension
(`hpipm_d_ocp_qp_dim`: `nx,nu,nb,ng,ns`); acados folds general equalities
into two-sided general rows (`lc == uc`, `ocp_qp_common_frontend.c`). Its
`OCP_QP_FACT_KKT_STEP` (`x_ocp_qp_kkt.c`) applies `Γ = lam/t` to *all*
active sides with **no clamp by default** (`t_lam_min = 2`;
`d_compute_Gamma_qp` clamps only when `t_lam_min == 1`) and returns
`void` — so the same terminal-stage failure is a **latent, silent defect in
HPIPM itself**, never exposed because acados' examples have no active
general equalities on this path and `sens_frw` is rarely called there.
HPIPM's *dense* QP **does** have a first-class `ne` (hard equalities, free
multipliers, Schur/null-space KKT, `remove_lin_dep_eq`) — the hard-equality
treatment exists in HPIPM, just not in the OCP-QP path.

## 3. Design decision (approved 2026-10-13: "option 2, the parity solution,
in all paths")

Structurally hard rows (lo == hi by construction: `g_eq` rows and `g_pin`
rows) become **hard KKT equalities with a free multiplier** — no barrier, no
`Γ`, exactly like the dynamics rows. All other rows (box, ineq, lin, and
*soft* eq rows with slacks) keep the barrier treatment.

Why not just cap `Γ` / grow `reg`: at the converged iterate the O(1)
Hessian information is already lost in rounding against the 1e17 barrier
term — no factorization of the reduced matrix (Cholesky or LQ) can recover
it. The equality must be in the KKT system as an exact row.

**Scope:** staged solver (`HpipmQpSolver`) first (unblocks 5f), then
full-condensing (`HpipmDenseQpSolver`), then block condensing (flag
carry-through; reuses the staged solver), RTI (driver only — no change).
Soft-equality rows and runtime `lo==hi` lin/box rows stay barrier-ized
(documented limitation; HPIPM `nbxe`-style fixed-box parity is future work).

### 3.1 Data model: `d_mask` gains the value 2

- `d_mask ∈ {0, 1, 2}` per side: 0 = absent, 1 = barrier side,
  **2 = hard-equality side**.
- Set structurally in `assemble.hpp` `fill_d_mask_lhs_{first,path,term}`:
  rows in group `g_eq` or `g_pin` get `d_mask = 2` on both sides (replacing
  the current `isfinite(lo/hi) ? 1 : 0`); all other rows unchanged.
- `QpDim` gains `n_eq_first = (fixed_initial_state ? nx : 0) + ne`,
  `n_eq_path = ne`, `n_eq_term = ne_t` (pin rows are contiguous group 0, eq
  rows contiguous group 4 — `C_eq` is gathered from `DC` rows; both groups
  exist compile-time, so the gather is a fixed index list).
- **Layouts unchanged** (`nrow`, `nside`, `nvar`, `QpSol`, `QpRes` keep
  their sizes): an eq row still occupies its two sides.

### 3.2 Storage convention (keeps `QpSol` / `globalize.hpp` untouched)

- `QpSol`: **hi-side lam slot = free multiplier `π`, lo-side lam slot = 0**;
  `t_lo = t_hi ≡ 0` on eq sides.
- `globalize.hpp:166` extracts `λ_eq = lam[hi] − lam[lo]` → returns `π`
  unchanged. **Sign to be verified numerically** (NLP KKT residual at the
  converged solution must be ~0, not `2|π|·|J|`); if wrong, swap slots.
- `res_g` already uses `DC'(λ_hi − λ_lo)` (hpipm.hpp:687) → picks up
  `C_eq'·π` automatically (sign check as above).
- Canonical equality row = **lo side** (`+DC` row, rhs = `d_lo`).

### 3.3 Per-stage KKT (staged solver)

Stage variables `(w = (u;s), x)` plus `π_k ∈ R^{n_eq_stage}`:

```
M_k   = H_k + reg·I_(u;x) + D_in' Γ_in D_in + D_k' P_{k+1} D_k   (Γ only mask==1)
K̃_k  = [ M_k    C_eq_k' ]  (nvar + n_eq) × (nvar + n_eq)
       [ C_eq_k  0       ]
```

- **Factorization:** `Eigen::LDLT` (indefinite saddle; `LLT` invalid).
  `n_eq == 0` (compile-time) → the *current* Cholesky path is taken
  untouched → **bit-identical behaviour for eq-free problems** (zero
  regression risk there).
- **`P_k`** (Riccati Hessian w.r.t. `x_k` on the equality manifold):
  `P_k = (K̃_k^{-1}[x,x])^{-1}` via `K̃ X = E_x` (dense `nx×nx` block
  invert; `nx` is small). Terminal stage `k = N`: `P` unused (`p_next = 0`).
- **Two-pass extension** (reduce / forward / closed form):
  - reduce: `rhs` extended with the equality residual rows
    (`res_d_lo` of the eq sides); the x-part offset `q` generalizes to the
    saddle factor (carry the extra `n_eq` rows of `L^{-1} rhs` alongside
    the existing `q = −rx + L(x,w)·z` form).
  - forward: solve `K̃ [dw; dx; dπ] = [rtilde_w; rtilde_x + dpi_prev;
    res_eq]` with the stored `L`/`D` factors; split into `dw, dx, dπ`.
    (Today `x` is eliminated *before* the stage solve via `P`; with eq rows
    it stays in the stage solve and is recovered from the saddle solve.)
  - `dpi_k = P_{k+1}(D_k·dz_k + res_b_k) + q_{k+1}` — unchanged formula,
    `dz_k = (dw; dx)` part only.
  - closed form for **mask==1 sides** unchanged (`dt = −res_d − D_in dw`,
    `dlam = Γ·dt + γ`); **mask==2 sides**: `dt = 0`, `dlam_hi = dπ`
    (from the saddle solve), `dlam_lo = 0`.
- **`compute_gamma`** (staged, hpipm.hpp:3018): `d_mask == 2` sides get
  `Γ = 0`, `γ = 0` (today the `dmask > 0.5` branch would form the blowing-up
  `Γ = λ/t`).
- **Residuals** (`compute_residuals`, hpipm.hpp:698): `res_m` on mask==2
  sides ≡ 0 (no complementarity); `res_mu` count and `res_m_max` exclude
  mask==2 sides (today `d_mask > 0.5`); `res_d` on eq sides **kept**
  (equality feasibility, must reach `res_d_max`).
- **Initialization / warm start** (`init_point`): eq sides `t = 0`, no
  `t`-clip; `lam_hi` warm-started from the previous `π` in modes 2/3, zero
  otherwise. Step update: eq-side `t` step ≡ 0 (`mask_step` keeps
  `lam_hi = dπ`, zeroes `t`).
- **`sens_solve`** (hpipm.hpp:2089) inherits everything: the exit refactor
  factors the same `K̃` (LHS), the seeded RHS flows through `res_d` (pin-row
  `d` seeds and `b[0]` seeds unchanged in meaning).

### 3.4 Variant coverage (staged)

- `square_root_alg = 1` (default): saddle `LDLT` as above.
- `square_root_alg = 0` (classical, 5a): same `K̃` build; the
  leading-`(w;x)`-block Schur form generalizes (Mww = `(w)` block of `K̃`
  after eliminating `π`... — implementation detail: factor `K̃` once with
  LDLT and derive both `Lw` and `P` from it; cross-check both variants
  agree on an eq problem, extending `tests/hpipm/sqrt_alg_5a.cpp`).
- LQ path (3d): when `n_eq > 0`, use the LDLT saddle factor unconditionally
  (the wide-matrix `W W' = Mtilde` trick does not extend to a saddle block).
  Documented deviation; the `lq_fact` option then only matters for eq-free
  stages.

### 3.5 Full-condensing (dense QP)

- `DenseQp` `d_mask` carries 2 (the condenser `condense.hpp` must copy the
  flag through its side → dense-row mapping; verify at implementation time —
  dense rows fold staged sides into `C`/`C_lo`/`C_hi` with slack columns).
- `HpipmDenseQpSolver`: split `A` into `A_in` (mask==1) / `A_eq` (mask==2);
  `W = H + reg·I_(x0;u) + A_in' Γ A_in`; coupled KKT
  `[ W  A_eq'; A_eq 0 ] [dv; dπ] = [rhs; res_d_eq]`, LDLT; `closed_form_step`
  on mask==1 sides only; `mask_step`: mask==2 → `t = 0`, `lam_hi = dπ`.
  (Mirrors HPIPM dense-QP `ne` treatment incl. its Schur/null-space choice;
  `remove_lin_dep_eq` stays 0.)
- Dense sensitivities remain deferred (Phase 4 scope); 5f is staged-only.

### 3.6 Block condensing / RTI

- `block_condense.hpp`: eq rows flow-map into block general rows — carry the
  `d_mask = 2` flag into the block QP's `d_mask`; the block QP is solved by
  `HpipmQpSolver` (templated over `BlockQpDim`), so it inherits 3.3. Verify
  `BlockQpDim` exposes the eq-row count for the gather (its row groups are
  re-derived from the block mapping — check at implementation time).
- `rti.hpp`: driver only (same assembly + solver) → no change; run the RTI
  tests to confirm.

### 3.7 NLP level

No change: NLP equalities are already hard; the SQP linearization fills the
QP `g_eq`/`g_pin` rows; `globalize.hpp` dual extraction works via 3.2.

## 4. Execution phases

- **A. Scope pass** — done (this doc).
- **B. Staged path** (`qp.hpp`, `assemble.hpp`, `hpipm.hpp`):
  1. `QpDim`: `n_eq_*` + eq/pin row-index gathers; `d_mask` doc (values 0/1/2).
  2. `fill_d_mask_lhs_*`: `g_eq`/`g_pin` → 2.
  3. `compute_gamma`: mask==2 → 0.
  4. `factor_one_stage{,_classic,_lq}`: saddle `K̃` + LDLT; `n_eq == 0` →
     current path; `P_k` extraction.
  5. `reduce_one_stage{,_classic}` / `forward_one_stage{,_classic}` /
     `closed_form_one_stage`: saddle-aware (3.3).
  6. `init_point`, `compute_residuals`, step/mask updates (3.3).
  7. Build + new `tests/hpipm` unit test: staged QP with an active equality
     (synthetic + DI N=10 QP at the converged iterate): solve converges,
     `res_d_eq < tol`, `sens_forward` FD cross-check (QP-level, as 4d).
- **C. 5f green**: `./build/sqp_unit` (sens_x0_5f checks), then full
  regression (`cmake --build build -j4`, all test binaries + examples).
- **D. Dense path**: `condense.hpp` flag carry-through, dense KKT saddle
  (3.5); DI full-condensed SQP + `tests/hpipm/dense_unit` extended with an
  eq row.
- **E. Block path**: `block_condense.hpp` flag mapping; DI NB=2/4 regression
  (terminal eq lives in the last block).
- **F. RTI**: run `tests/sqp/rti_*` (no code change expected).
- **G. A/B** vs baseline worktree: SQP solutions (x, u, λ_eq, cost, iters)
  on all examples; expect solution agreement < 1e-8, IPM iter counts may
  differ (different factorization, same limit point).
- **H. Docs + commit**: `qp.hpp` header contract (`d_mask` 0/1/2),
  `ARCHITECTURE.md` §8, main parity plan §8 5f entry. Commits (two logical
  changes): (1) `solver:` hard-equality KKT treatment (qp.hpp, assemble.hpp,
  hpipm.hpp, condense.hpp, block_condense.hpp + solver tests); (2)
  `solver:`/`test:` sens_x0 API + 5f FD test (acados.hpp, sens_x0_5f.cpp,
  sqp_unit.cpp, CMakeLists.txt) — (2) lands only after (1).

## 5. A/B baseline

Worktree `/home/jgeisler/repos/ocp_plusplus-hard_eq_base` @ `9440d15`
(detached HEAD, clean) — pre-hard-equality state for solution comparison
(phase G). Build there with the same Release flags.

## 6. Open questions (verify at implementation time, not design decisions)

1. **Sign of `π`** vs the `globalize.hpp` extraction (`lam_hi − lam_lo`):
   verify with the NLP KKT residual at the converged solution (must be ~0);
   flip the slot convention if wrong.
2. **`P_k` from `K̃^{-1}[x,x]`** vs the classical-variant Schur form:
   cross-check `square_root_alg` 0/1 agree on an eq problem.
3. **LQ path with eq rows**: LDLT-only (3.4) — confirm `lq_itref` tests.
4. **Condenser row mapping** (`condense.hpp` / `block_condense.hpp`):
   exactly where `d_mask` values land in the dense/block side vectors;
   verify with existing condense tests + a new eq-row case.
5. **Convergence checks**: QP-IPM exit uses `res_d_max` (now includes eq
   residuals) and `res_m_max` (excludes them) — confirm the QP still exits
   `kSolved` on DI / mass-spring.
6. **Soft-eq rows** (`ne_soft > 0`): stay barrier-ized; no example has them
   — document.
7. `reg` ladder interaction: if `K̃`'s leading block is not PD at the
   converged iterate (flat cost along the eq manifold), the ladder must
   still fire (`factor` returns false → reg growth) — keep the
   `kQpFailure` contract.

## 7. Verification gate

- Warning-free build under `-Wall -Wextra -Werror`.
- Full existing test suite (all `tests/*` + example binaries) before/after;
  no silently weakened tests.
- `tests/sqp/sens_x0_5f.cpp` green (du/dx0, dx/dx0 < 1e-6 vs central FD;
  `dx_0 == e_i` exact).
- A/B: SQP solutions on all examples agree with the baseline worktree to
  < 1e-8 (x, u, cost, λ_eq).

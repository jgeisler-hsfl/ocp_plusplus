# Plan: dense-QP sensitivities + hybrid block KKT (Phase 5, item 5g)

Status: **completed** (2026-10-14). `HpipmDenseQpSolver::sens_forward` /
`sens_adj` are implemented and verified (QP-level FD in
`tests/hpipm/dense_seed_5g.cpp`; OCP-level FD + dense-vs-staged cross-check
in `tests/sqp/sens_x0_5f.cpp`, `qp_mode = kFullCondensed`, rel. err < 1e-6).

**Final approach (supersedes §3.2–3.3, the eq-rows-only design).** The
original plan moved only the data-equality rows (lo == hi) into the KKT and
kept every other active side in the barrier term `A'ΓA`. That LHS is
ill-conditioned at the converged IPM iterate: *every* pinned side —
including active inequality sides — carries the barrier weight
Γ = λ/t ~ 1e9..1e18, so `cond(H + A'ΓA) ~ 1e15` and the H-dominated part of
the sensitivity step is lost (staged 5f avoids the pathology because its
Riccati two-pass never forms the dense W; the dense path must). The shipped
design classifies all present sides into two groups and builds a hybrid
block KKT:

- `split_sides` (static, hpipm.hpp): a present side (d_mask > 0.5) is
  **hard** iff its slack has collapsed (`t < 1e-9`) or its barrier weight
  is large (`Γ = λ/t > 1e6`); all other present sides are **moderate**.
  A t-threshold alone misses active sides sitting just above the floor;
  a weight-only test misclassifies soft-constraint slacks (small λ and t,
  Γ ~ O(1..1e2)) — making those exact rows over-constrains the step.
  A lo == hi row has BOTH sides hard; they enter as a row pair, consistent
  under the coherent (opposite-sign) seed the driver builds.
- Hybrid block KKT (`fact_solve_kkt_active`, Eigen LDLT, cached in `ws_`
  keyed by (N, hard set, moderate set, reg)):
  ```
  W_mod = H + reg*I_(x0;u) + A_mod' Gamma_mod A_mod   (moderate only)
  K     = [ W_mod   A_hard' ]        rhs = [ -seed_g - A_mod' gamma_mod ]
          [ A_hard    0     ]                [      seed_d(hard)       ]
  ```
  Moderate Γ is bounded by the split threshold, so `W_mod` stays
  well-conditioned (cond ~ 1e2..1e4). `sens.v` comes from the KKT; the
  moderate (and masked) sides get the closed-form barrier `dt`/`dlam`
  (`closed_form_step`); the hard sides carry the free KKT multipliers with
  `t = 0` (as in the staged eq-row treatment). The seed_m channel stays a
  first-order no-op (fixed side split); the adjoint pre-scales m by t and
  post-scales `sens.t` by 1/t (HPIPM `SENS_ADJ` parity).
- Driver `AcadosSolver::sens_x0` (`kFullCondensed`): condense, solve the
  dense QP, seed the pin box rows coherently (`seed_d(lo) += dx0`,
  `seed_d(hi) -= dx0`; no dynamics seed — the condensing flow map already
  carries the x0 column), `sens_forward`, `expand_full_sol` back to
  staged layout. Verified double-count-free against staged 5f (the
  dense-vs-staged cross-check agrees to ~2e-7 on u, ~2e-8 on x at DI N=10).

## 1. Goal

- `HpipmDenseQpSolver::sens_forward` / `sens_adj` (HPIPM
  `DENSE_QP_IPM_SENS_FRW` / `SENS_ADJ`, `x_dense_qp_ipm.c:2817/2883`):
  re-solve the dense KKT at the last converged iterate with a seed
  (`DenseQpSeed`) as the forcing; `new_sol ≈ sol + sens`.
- Well-conditioned: at the converged IPM iterate *every* pinned side
  (equality rows AND active inequality sides) carries barrier weight
  Γ = λ/t ~ 1e9..1e18, so the plain barrier LHS `H + A'ΓA` has
  cond ~ 1e15 and the H-dominated part of the sensitivity step is lost.
  The sens solve instead uses a **hybrid block KKT (Eigen `LDLT`,
  indefinite)**: pinned sides enter as exact constraint rows (free
  multipliers); all other present sides keep a *bounded* barrier term
  (below the same pathology as staged 5f).
- `AcadosSolver::sens_x0` works for `qp_mode = kFullCondensed`
  (currently `kStaged` only, acados.hpp:1100).

Out of scope here: the **partial-condensed** (block) path — the block QP is
solved by `HpipmQpSolver` templated on `BlockQpDim`, so it inherits the
staged 5f sens path once `BlockQpDim` exposes the eq rows and
`block_condense.hpp` maps eq rows into block rows (5h).

## 2. Background / why this shape

- **Main dense solve stays unchanged.** `HpipmDenseQpSolver::solve` keeps
  the barrier two-pass (Mehrotra) with all rows as inequality sides
  (`d_mask ∈ {0,1}`); equalities are lo == hi rows, decision Q4. This is
  the shipped 5f policy for staged and now for dense: *main solve =
  barrier, sens solve = hard-equality block KKT when applicable*.
- **HPIPM reference.** `DENSE_QP_IPM_SENS_FRW` builds a `tmp_qp` that
  aliases the LHS of the last-solved QP (`Hv`, `A`, `Ct`, `d_mask`, `Z`,
  `idxs_rev`) but substitutes `seed_g`/`seed_d`/`seed_m` for
  `gz`/`d`/`m`, then calls `SOLVE_KKT_STEP_DENSE_QP`. Because the stored
  iterate satisfies the original KKT to tolerance, the RHS is, to leading
  order, the seed itself. `SENS_ADJ` additionally scales `seed_m` by `t`
  before the solve and `sens.t` by `1/t` after (dual scaling, same as
  staged).
- **No `seed_b`.** HPIPM's dense seed has a `seed_b` slot, but in our
  condensed model the dynamics are eliminated by substitution
  (decision: the condensed problem has NO equality rows / no dynamics
  offset), so `DenseQpSeed` carries only `g`, `d`, `m`.

## 3. Design

### 3.1 `DenseQpSeed<P, NH>` (new, in `dense_qp.hpp` or `seed.hpp`)

```
struct DenseQpSeed<P, NH>
{
    V_t    seed_g{};   // (nv)  linear-cost seed   (HPIPM seed_g)
    Side_t seed_d{};   // (nside) bound-offset seed, side layout [lo; hi]
    Side_t seed_m{};   // (nside) complementarity-RHS seed (0 in v1)
    int N, nv, nside;  resize(n_stages) like DenseQp
};
```

Same NH dynamic/fixed mechanism as `DenseQp`/`DenseQpSol`
(`detail::DenseVec`/`DenseMat`). All-zero default (an unset seed
contributes nothing).

### 3.2 Hard-equality detection on the dense QP

```
bool dense_eqs_hard(const DenseQp<P, NH>& in)   // + row-index gather
```

A dense row `r` (box row `0..nb-1` or general row `0..ng-1`) is a hard
equality iff both sides are active and the bounds coincide:
`d_mask(lo)>0 && d_mask(hi)>0 && |d(lo) + d(hi)| < 1e-8`
(lo sides store `lo`, hi sides store `−hi`). This catches the pin box
rows (x_0 fixed) and the condensed `g_eq` general rows, exactly mirroring
the staged `eqs_hard` gate (hpipm.hpp) but self-locating the rows from
`d`/`d_mask` (the dense QP has no row-group metadata of its own).
Returns the equality-row index list (used for the `A_eq` gather) and a
bool (any present). One-sided rows (`d_mask` on one side only) are true
inequalities and never qualify.

### 3.3 Block KKT for the dense sens solve

Let `A ∈ R^{nside × nv}` be the signed side matrix
(`build_side_matrix`, hpipm.hpp:5021). Partition:

- `A_eq  = A(lo-rows of the equality rows, :)`   (one row per equality)
- `A_in  = A(active non-equality sides, :)`      (`d_mask == 1`, not eq)

At the converged iterate (λ, t):

```
Gamma_in(i) = lam(i)/t(i)      (masked; 0 on eq sides)
W = H + reg·I_(x0;u) + A_in' diag(Gamma_in) A_in
K = [ W   A_eq' ]
    [ A_eq  0   ]              (nv + n_eq) square, symmetric, indefinite
```

Factor with Eigen `LDLT` (cached in the workspace, keyed by
`(N, reg)` exactly like the staged `ws_.kkt_*` cache; invalidated in
`solve()`). If `dense_eqs_hard` finds no equality rows, the sens solve
reuses the stored barrier factor (`ws_.llt` of `W` with full `Γ`, the
staged `has_factors_` path) — no block assembly.

**RHS (forward).** Mirror the staged `fact_solve_kkt_hardeq` RHS
construction with the dense quantities (seed vectors `sg, sd, sm`):

```
rhs_v  = sg − H v − A_in' lam_in − A_eq' pi_eq
        (≈ sg at the converged iterate, since res_g ≈ 0)
rhs_eq = −res_d_eq − sd_eq        (eq-side feasibility + seed)
```

where `pi_eq` is the implicit free multiplier of the equality rows at the
iterate (recovered once from the stationarity condition,
`pi_eq = −(A_eq H + ...) `— *verify the exact recovery against the
linearized KKT at implementation time; the leading-order RHS is
`[sg; −sd_eq]` plus the Γ-in/γ-in correction terms exactly as in staged
5f*). Solve `K [dv; dπ] = [rhs_v; rhs_eq]`, then the closed-form
inequality-side update on `A_in` sides only:

```
dt_in   = A_in dv − res_d_in
dlam_in = d_mask . (−res_m_in − lam_in ⊙ dt_in) ./ t_in
```

Equality sides: `t = 0`, `dt = 0`, the multiplier step is `dπ` (lo side
convention, matching the staged `lam_hi − lam_lo` extraction used by
`globalize.hpp`).

**Adjoint.** Same solve with `sm` scaled by `t` first
(`sm ← sm ⊙ t`), and `sens.t` rescaled by `1/t` afterwards (HPIPM
`SENS_ADJ` lines 2924–2963; identical to the staged adjoint).

### 3.4 API on `HpipmDenseQpSolver`

```
Status sens_forward(const DenseQp<P,NH>& in, const DenseQpSol<P,NH>& sol,
                    const DenseQpSeed<P,NH>& seed, DenseQpSol<P,NH>& sens);
Status sens_adj(...);
```

Precondition: a `solve()` has just converged on `in` (factors / iterate
available in `ws_`); same policy as the staged `sens_solve`
(`has_factors_` → reuse, else one factor pass). `DenseQpSol` gains no new
fields (v, lam, t, obj suffice; `obj` unused by sens).

Workspace additions (mirroring the staged 5f cache):
`kkt_K` (dynamic (nv+n_eq)²), `kkt_ldlt`, `kkt_valid`, `kkt_N`, `kkt_reg`,
plus the equality-row index list and `A_eq`/`A_in` gather scratch.

### 3.5 `AcadosSolver::sens_x0` for `kFullCondensed`

acados.hpp:1094 — remove the `kStaged`-only guard, branch on mode:

1. `assemble_qp` (staged QP, as today).
2. `condense_full(qp_in_)` → `DenseQp` (existing, condense.hpp:64).
3. `qp_dense_.solve(dense, dsol_dense)` (warm-started from the SQP
   driver's last dense solution).
4. `DenseQpSeed`: seed the **pin box rows** — for `x_0` variable `r`:
   `seed_d(lo) += dx0(r)`, `seed_d(hi) −= dx0(r)` (dense box-row order:
   pin rows first in the stage-0 box block, condense.hpp:255–258). No
   dynamics seed (no `seed_b`; the condensing already maps `x_0`
   through the flow — *verify no double count against the staged 5f
   result, §5*).
5. `qp_dense_.sens_forward(dense, dsol_dense, seed, dsens_dense)`.
6. `expand_full_sol(qp_in_, dense, dsens_dense, dsol)` (condense.hpp:481)
   → staged-layout `QpSol` (states by forward `BA` simulation, lam/t via
   side mapping, dynamics multipliers recovered as in the normal path).

`kPartialCondensed` keeps returning `kAborted` until 5h.

## 4. Execution steps

1. **Seed model** — `DenseQpSeed` (3.1) in `dense_qp.hpp` (or `seed.hpp`,
   next to `QpSeed`; prefer `seed.hpp` for symmetry).
2. **Gate + gather** — `dense_eqs_hard` (3.2) as a static helper in
   `HpipmDenseQpSolver` (or free template in `dense_qp.hpp`).
3. **Block KKT** — `fact_solve_kkt_hardeq_dense` (3.3) + workspace
   additions; `solve()` invalidates the cache (as staged line 478).
4. **Sens API** — `sens_forward` / `sens_adj` (3.4), gate:
   `dense_eqs_hard(in)` → block KKT; else reuse stored factor
   (`compute_gamma` + `solve_kkt` with seed RHS).
5. **Driver** — `sens_x0` kFullCondensed branch (3.5).
6. **Tests** (§5).
7. **Docs + commit** — `dense_qp.hpp`/`seed.hpp` header contracts,
   ARCHITECTURE.md §8 (sensitivities now staged + dense), main parity
   plan line 44 ("dense deferred" → "dense done (5g)"). Commits:
   (1) `solver:` dense sens + block KKT (dense_qp.hpp, hpipm.hpp,
   acados.hpp); (2) `test:` FD tests + wiring; (3) `docs:`.

## 5. Tests

- **QP-level FD (new `tests/hpipm/dense_seed_5g.cpp`)**: synthetic dense
  QP (nv ≈ 6–8) with one lo == hi box row and one lo == hi general row
  (hard equalities) plus a soft row; central-FD re-solve of the full
  dense IPM vs `sens_forward` for seeds in `g`, `d` (box + general),
  rel. err < 1e-6; adjoint duality identity
  `⟨seed_a, dsol_b⟩ = ⟨seed_b, dsol_a⟩` on two seeds. Also the eq-free
  variant (no lo==hi rows) to cover the stored-factor path.
- **OCP-level (extend `tests/sqp/sens_x0_5f.cpp`)**: DI N=10, central FD
  of the full SQP re-solve with `qp_mode = kFullCondensed` (x0 ± 1e-6),
  rel. err < 1e-6, pinned dx_0 exact; **cross-check the dense `sens_x0`
  against the staged `sens_x0`** (same QP, two modes) — agreement
  ~1e-8 (both must agree with each other, not just with FD).
- **Regression**: full `qp_unit`, `sqp_unit`, all example binaries
  unchanged (main dense solve untouched); ASan repro clean.

## 6. Open questions (verify at implementation time)

1. **RHS exact form** — the leading-order dense sens RHS is
   `[seed_g; −seed_d_eq]`; the staged 5f carries additional
   `−A'γ`-type correction terms (staged `r̃ = −res_g − c'γ`). Derive the
   dense analog once and validate against the FD test before freezing.
2. **`pi_eq` recovery** — the equality multipliers at the barrier
   iterate: `λ_lo`/`λ_hi` of the lo==hi sides; the free multiplier is
   `λ_hi − λ_lo` (the `globalize.hpp` convention). Confirm the sign
   against the NLP KKT residual at the converged SQP solution.
3. **Tolerance** — staged `eqs_hard` uses `|d(lo)+d(hi)| < 1e-8`; reuse
   the same threshold (document in `dense_eqs_hard`).
4. **Double-count check** — staged 5f proved the pin-row `d` seed alone
   propagates dx0 (no `seed.b[0]`). Verify the dense condensing does the
   same (the flow map already contains the x0 column) via the
   staged-vs-dense cross-check in §5.

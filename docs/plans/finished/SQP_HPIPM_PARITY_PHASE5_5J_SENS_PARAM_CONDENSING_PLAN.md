# Plan: dense / block seed condensing for `sens_param` (Phase 5, item 5j)

Status: **DONE (2026-10-15)** — `AcadosSolver::sens_param` works in all three
QP modes (kStaged 5i, kFullCondensed, kPartialCondensed), FD-verified for
DI N=10, p_0…p_5. Deferred from
`SQP_HPIPM_PARITY_PHASE5_5I_SENS_PARAM_PLAN.md` §6 ("Out of scope (5j)").

## Outcome

- `sens_param` dispatch: staged seed (`fill_param_seed`, 5i) → `build_seed_qp`
  → condensed seed solve (`sens_param_full_condensed` /
  `sens_param_block_condensed`) → staged-layout expansion via
  `expand_full_sol(qseed, …, include_b = true)` /
  `expand_block_step(qseed, …, include_b = true)`. The `include_b` flag was
  added to `expand_block_step` (default `false`, so the 5h x0-sensitivity
  call site is unchanged): the block primal forward simulation carries the
  seed `b` (dynamics p-derivative) exactly like the dense path.
- Tests: `tests/sqp/sens_param_5j.cpp` (dense + block, cross-check vs staged,
  central FD); `sens_param_5i.cpp` updated (condensed modes now kSolved).
- Root cause of an optimization-level-dependent 5j failure (block p_2
  garbage at -O2/-O3, clean at -O0/-O1): `BoxSensP::dlo_dp/dhi_dp` in
  `problem.hpp` still used `{}` NSDMIs; on Eigen 3.3.9 a user-provided
  `Matrix()` default ctor means value-init does NOT zero the coefficients.
  Uninitialized denormal garbage then flowed through
  `fill_param_seed` (`bsx_sp.dlo_dp * dp`) into the seed `d` and was
  amplified by the barrier KKT. Fixed with `::Zero()` NSDMIs (same
  convention as the sibling spec/sens structs). Also fixed the same
  pattern in `DoubleIntegrator::initial_state_sens_p` (`x0_dp_t::Zero()`).
- Dense sens KKT hard-row dedup: `fact_solve_kkt_active` previously kept
  both lo and hi sides for lo==hi rows (A_hi = -A_lo), giving
  linearly-dependent KKT rows and a rank-deficient factor (DI N=10:
  rank 34 of 37). Now the lo-side partner of each hard lo==hi pair is
  dropped (kept hi side only, matching the staged
  `fact_solve_kkt_hardeq` convention): the KKT is full rank
  (34 of 34), the dropped lo side gets `lam = t = 0` (the closed-form
  update divides by t = 0 there), and its Gamma/gamma are zeroed so the
  barrier part of W is unaffected. Primal sensitivities are unchanged
  (the removed rows were consistent duplicates); verified identical
  5j FD results and full-rank KKT.

## 1. Goal

- `AcadosSolver::sens_param` works in `kFullCondensed` and
  `kPartialCondensed` in addition to `kStaged` (removing the 5i `kAborted`
  guard), returning the same QP-space response `dsol` (staged layout via
  `expand_full_sol` / `expand_block_step`).
- The 5i staged behaviour, active-set gating, and FD-verified results are
  unchanged (staged path is not touched).

## 2. Formulation — condense the sensitivity QP

The staged seed solve (5i, `hpipm.hpp::sens_solve`) builds a QP
`in_seed` with **LHS from the converged QP** (hess / BA / DC / d_mask) and
**RHS from the seed** (grad / b / d / m), then solves its KKT with the seed
vectors as forcing. `in_seed` is a perfectly ordinary staged QP, so the
existing condensing machinery maps it to the condensed-QP sensitivity
directly:

```
dense seed:    condense_full (in_seed ) -> (g, d)   == seed_g, seed_d
block seed:    condense_block(in_seed) -> (grad,b,d) == block seed
```

This is correct because condensing is a linear, structure-preserving map:
`sens(condense(QP)) == condense(sens(QP))`. The flow-map elimination
(`G[k]` / `gamma[k]` in `condense_full`, block flow maps in
`condense_block`) folds the seed `b` (dynamics p-derivative) into the
condensed `g` / `d` exactly as the 5i plan §6 described ("the `b` seed
flows through the same Riccati/flow-map elimination as the data; the dense
QP has no `b` field, so the dynamics p-derivative is folded into the dense
`seed_g`"). No new flow-map code is needed.

**Sanity checks against the existing seed paths:**

- x0 seed (5f/5g/5h): seed has only `d_first` pin rows, `b = 0`,
  `grad = 0` ⇒ `gamma = 0`, no d-shift; pin rows map directly (stage 0 /
  block 0). `condense_full/in_seed` reproduces exactly the 5g construction
  (`seed_d(r) += dx0`, `seed_d(nb+ng+r) -= dx0`) and the 5h block pin-row
  seed. ✓
- 5i staged seed: condensed seed on a fully identity-condensed horizon
  (NB = 1) must equal the staged seed field-for-field (`BlockQpDim<P,1>`
  == `QpDim<P>`). ✓ (used as a test)

**Active-set gating** is applied when building the *staged* seed
(`fill_param_seed`, 5i) and carries through condensing unchanged
(linear map of an already-gated vector).

**What is NOT condensed:** the seed `m` (zero in v1 — p-independent slack
penalties; `condense_*` keeps it zero). The LHS is *not* re-derived: the
converged dense/block QP (`condense_full(qp_in_)` /
`condense_block(qp_in_)`) is solved on a throwaway solver to factor its
LHS and supply the converged iterate; only the *seeded* QP's condensed RHS
is used as the forcing. (The condensed LHS of `in_seed` is identical to the
converged QP's — same converged LHS data.)

## 3. Implementation (`include/ocp/solvers/acados/acados.hpp`)

Restructure `sens_param` (~line 1311):

```
Status sens_param(problem, sol, dp, dsol):
  N = sol.N; resize(N); assemble_qp(problem, sol)
  throwaway staged solve of qp_in_ -> qsol          (5i; needed for gating)
  staged seed: fill_param_seed(problem, sol, dp, qsol, seed)   (5i)
  if kStaged:  return sens_qp.sens_forward(qp_in_, qsol, seed, dsol)   (5i, unchanged)
  qseed = build_seed_qp(qp_in_, seed)                (new helper, below)
  if kFullCondensed:    return sens_param_full_condensed(qp_in_, qseed, dsol)
  else (kPartialCondensed):
     NB-mismatch check (mirror sens_x0_partial_condensed, ~line 1195-1201):
     NB==0 / N2>=N / N%N2!=0 / N/N2!=NB -> staged fallback (sens_forward)
     return sens_param_block_condensed(qp_in_, qseed, N, N2, dsol)
```

- **`build_seed_qp(q, seed) -> Qp<P,NH>`** (private helper): mirrors the
  `in_seed` construction in `hpipm.hpp::sens_solve` (hpipm.hpp:2689-2716) —
  LHS fields (hess / BA / DC / d_mask) from `q`, RHS fields (grad / b / d /
  m) from `seed`, per first / path / term.
- **`sens_param_full_condensed(q, qseed, dsol)`** (mirrors
  `sens_x0_full_condensed`, ~line 1253):
  `condense_full(q, dq)` → throwaway `HpipmDenseQpSolver` solve →
  `condense_full(qseed, dqseed)` → `DenseQpSeed` from `dqseed` (g / d / m)
  → `sens_forward(dq, dsol_dense, seed, dsens)` →
  `expand_full_sol(q, dq, dsens, dsol, include_b=false)`.
- **`sens_param_block_condensed(q, qseed, N, N2, dsol)`** (mirrors
  `sens_x0_partial_condensed`, ~line 1191):
  `condense_block<P,NH,NB>(q, bq)` → throwaway
  `HpipmQpSolver<P,Dynamic,BlockQpDim<P,NB>>` solve →
  `condense_block<P,NH,NB>(qseed, bqseed)` → `QpSeed<P,Dynamic,BlockQpDim>`
  from `bqseed` RHS (grad / b / d / m, first / path / term) →
  `sens_forward(bq, bsol, bseed, bsens)` →
  `expand_block_step<P,NH,Dynamic,NB>(q, bsens, dsol)`.
- Update the `sens_param` doc comment (drop "kStaged only (… is 5j)").
- No changes to `hpipm.hpp`, `condense.hpp`, `block_condense.hpp`,
  `problem.hpp`, or the DoubleIntegrator example.

Cost note (acceptable; sensitivities are not inner-loop): each condensed
mode does one staged QP solve (gating) + one condensed-QP solve + one
condensed seed solve, plus two condense passes.

## 4. Tests

- **New `tests/sqp/sens_param_5j.cpp`** (wired into `sqp_unit`):
  - `test_sens_param_dense`: DI N=10, `qp_mode = kFullCondensed`. Per
    parameter: `sens_param` kSolved; `dx_0 == 0` (p-independent pinned
    state, < 1e-12); cross-check vs. a kStaged `sens_param` on the *same*
    base solution (< 1e-6 on (du,dx)); central FD of the full SQP re-solve
    (dense-mode solvers at p ± h; h = 1e-5 / tol = 1e-5 for Ts,
    h = 1e-4·max(1,|p|) / tol = 1e-6 otherwise — same calibration as 5i).
  - `test_sens_param_block`: DI N=10, NB = 2
    (`AcadosSolver<..., HpipmQpSolver<DiSens,Dynamic>,..., 2>`),
    `qp_mode = kPartialCondensed`, `qp_cond_N = N/2`; identical checks
    (cross-check vs. staged, FD in block mode).
- **Update `tests/sqp/sens_param_5i.cpp`**:
  `test_sens_param_non_staged_aborted` (line ~174) asserted `kAborted` for
  the condensing modes — 5j changes that behaviour, so the test is
  rewritten to assert `kSolved` (the kFullCondensed check also needs a
  kSolved + zero-dp-style sanity, not a full FD — the FD lives in 5j).
  Also the header comment (line 9) and the 5i plan's "kStaged only"
  references stay as historical record (not edited).

## 5. Out of scope

- Adjoint parameter sensitivity (`sens_adj` with the p-seed) — mechanism
  exists (Phase 4); not requested.
- p-dependent slack penalties (`seed_m` nonzero) — zero in v1.
- Warm-starting / caching the condensed seed solve across repeated
  `sens_param` calls (optimization, not correctness).
- Non-uniform block sizes (parity plan §9 item 4).

## 6. Execution phases

- **A. Scope pass** — this doc (flow-map / seed mapping verified against
  `condense.hpp` / `block_condense.hpp` / `hpipm.hpp` code).
- **B. Solver** (`acados.hpp`): `build_seed_qp` + `sens_param_full_condensed`
  + `sens_param_block_condensed`; remove the kStaged-only guard; doc update.
- **C. Tests**: `tests/sqp/sens_param_5j.cpp` (+ CMake + `sqp_unit` wiring);
  update `sens_param_5i.cpp` non-staged check.
- **D. Regression**: warning-free full build; all test binaries (staged 5i
  results must be unchanged; all np=0 examples bit-identical).
- **E. Docs + commits**: this plan (status/outcome), parity plan §2 row 44
  + §9.3, `ARCHITECTURE.md` (5j note on `sens_param`), commits
  (1) `solver:` + (2) `test:` + (3) `docs:`.

## 7. Verification gate

- [x] Warning-free build under `-Wall -Wextra -Werror`.
- [x] Staged 5i FD results unchanged (regression).
- [x] Dense + block: cross-check vs. staged < 1e-6; FD per parameter within
      the 5i calibration (Ts 1e-5 / others 1e-6); `dx_0 == 0`.
- [x] Full existing test suite passes.
- [x] Docs updated (this plan, parity plan §2/§9, ARCHITECTURE.md).

## 8. Deviations from §3

- `block_condense.hpp` changed: `expand_block_step` gained an `include_b`
  parameter (see Outcome) — the "no changes to block_condense.hpp" note in
  §3 does not hold; the x0 path (5h) is untouched (default `include_b =
  false`).
- `problem.hpp` + `double_integrator.hpp` changed: the BoxSensP /
  `initial_state_sens_p` zero-initialization fix (see Outcome).

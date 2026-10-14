# Plan: block-QP sensitivities via the staged 5f path (Phase 5, item 5h)

Status: **completed** (2026-10-14). `AcadosSolver::sens_x0` now supports
`qp_mode = kPartialCondensed`. Verified in `tests/sqp/sens_x0_5f.cpp`
(`test_sens_x0_fd_block`; DI N=10, NB = 2, 5 blocks): central finite
differences of the full block-mode SQP re-solve (x0 ± 1e-6) agree with the
sens_x0 response to < 1e-6, the pinned `dx_0` is exact, and the block
sens_x0 agrees with the staged sens_x0 to ~2e-7.

## 1. Goal

Lift the Phase 4 deferral ("dense / partial-condensed deferred") for the
**partial-condensing (block)** QP path: first-order sensitivity of the OCP
solution w.r.t. the fixed initial state (`AcadosSolver::sens_x0`) with
`qp_mode = kPartialCondensed`.

## 2. Key observation

The block QP is solved by the *staged* solver class templated over a
different dimension struct: `HpipmQpSolver<P, NH, BlockQpDim<P, NB>>`.
`BlockQpDim` (qp.hpp) already exposes the hard-equality machinery —
`n_eq_first/path/term`, `has_eq`, and the compile-time `eq_rows_*` arrays
(pin rows first in block 0, then the non-soft `g_eq` rows of each block).
The staged 5f sens path (`eqs_hard` gate → `fact_solve_kkt_hardeq` →
closed-form barrier sides) is fully Dim-generic. The block path therefore
inherits 5f almost for free; the actual work is small:

1. **Lift the guard.** `HpipmQpSolver::sens_solve` carried
   `static_assert(std::is_same_v<Dim, QpDim<P>>)` from 5f (added before
   `BlockQpDim` gained the eq rows). Removed. `sens_forward` / `sens_adj`
   / `sens_solve` now take the seed as a type parameter so the block seed
   (`QpSeed<P, NH, BlockQpDim<P, NB>>`, sized by the block-row layout) is
   accepted. Everything downstream (`has_active_side`, `eqs_hard`,
   `fact_solve_kkt_hardeq`, `compute_gamma`, the KKT factor) is already
   Dim-generic.
2. **Driver branch** (`acados.hpp`): `sens_x0` dispatches on the QP mode;
   the former inline staged body moved to `sens_x0_staged`.
   `sens_x0_partial_condensed` mirrors the `solve_qp_partial_condensed`
   fallback policy (NB == 0, N2 >= N, N % N2 != 0, or N / N2 != NB →
   staged path) and otherwise: `condense_block` → fresh
   `HpipmQpSolver<..., BlockQpDim<P, NB>>` solve (default options, cold,
   same policy as the staged sens path) → seed the pin rows of block 0
   coherently (`d_first(side_lo(r)) += dx0(r)`, `side_hi(r) -= dx0(r)`,
   r = 0..nx-1 of `BlockQpDim::lay_first`) → `sens_forward` →
   `expand_block_step` back to the staged layout.
3. **Step expansion** (`block_condense.hpp`, `expand_block_step`): maps
   the block sens *step* (dv, dt, dlam, dpi) to the staged `QpSol` layout.
   - Primal: du/ds sliced directly from the block step vectors; dx_{k0}
     from the block; internal state increments forward-simulated
     **homogeneously**, `dx_{k+1} = BA [du_k; dx_k]` (no `+ b`: the seeded
     KKT enforces the step dynamics without the affine offset).
   - Duals: `dlam`/`dt` transfer row-group by row-group via
     `copy_block_duals` (extracted from `expand_block_sol`, shared by
     both; the flow map is a feasible-set bijection, so multipliers and
     slacks transfer verbatim).
   - Dynamics multipliers: `bstep.pi[ii]` is the staged multiplier of the
     block's last stage (the composite-row multiplier equals the last
     stage's chain multiplier); the internal `dpi_{k0}..dpi_{k0+NB-2}`
     come from a **linearized** stationarity backward sweep,
     `dpi_{k-1} = (H_k dz_k)_x + DC_k,x'(dlam_hi - dlam_lo) + A_k' dpi_k`
     (no grad term; the reg-primal correction is O(reg) ~ 1e-15 and is
     omitted).
4. **Latent 5g pi fix** (`condense.hpp`): `expand_full_sol`'s pi sweep
   used the absolute KKT residual `(H z + g + DC'λ)_x`; in step mode
   (`include_b = false`, the 5g dense sens call) the inputs are the step,
   so the sweep is now the linearized stationarity
   `(H dz + DC' dλ)_x` (no grad term). Absolute-solution calls
   (`include_b = true`, main solve path) are unchanged.

## 3. Verification

- `tests/sqp/sens_x0_5f.cpp` `test_sens_x0_fd_block`: DI N=10,
  `AcadosSolver<..., NB = 2>`, `qp_cond_N = N/2` (5 blocks). Per unit
  direction e_i: `sens_x0` returns kSolved; pinned `dx_0 == e_i` to 1e-12;
  block vs staged sens_x0 agree to < 1e-6 on (du, dx); central FD
  (x0 ± 1e-6) of the full block-mode SQP re-solve matches the sensitivity
  to < 1e-6. Observed: max |du diff| 2.7e-7, max |dx diff| 3.4e-7.
- Regression: full build warning-free under `-Wall -Wextra -Werror`;
  `qp_unit` and `sqp_unit` pass (main-solve behavior of the block path
  unchanged — `expand_block_sol` still uses the absolute-value pi sweep);
  ASan clean; all example binaries pass.

## 4. Out of scope

- NLP-level parameter sensitivity (needs a parameter vector in the problem
  interface) — separate roadmap item.
- `sens_adj` for the block path is available (same Dim-generic code) but
  only the forward path is exercised by `sens_x0` and the tests.

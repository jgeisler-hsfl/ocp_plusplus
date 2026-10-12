# Plan: full feature parity with the acados SQP + HPIPM stack

Status: **draft** (2026-10-10). Working reference for closing the gap between
the ocp++ `AcadosSolver` + `HpipmQpSolver` stack and acados' standard SQP +
HPIPM OCP-QP stack. Scope agreed: **NLP/SQP level — the full SQP+HPIPM stack**
(no other QP backends, no DDP/IRKSQP, no DAE `z` variables).

## 1. Reference

- acados repo (read-only, `ACADOS_SOURCE_DIR`, default `../acados`):
  - `acados/ocp_qp/ocp_qp_xcond_solver.c` — dispatch between condensing modes
    and the inner QP solver.
  - `acados/ocp_qp/ocp_qp_partial_condensing.c` — block grouping (`N2`,
    `block_size`), condense/expand glue.
  - `acados/ocp_qp/ocp_qp_full_condensing.c` — dense-QP condense/expand glue.
  - `external/hpipm/cond/x_part_cond.c` — `PART_COND_QP_COMPUTE_BLOCK_SIZE`,
    `PART_COND_QP_COMPUTE_DIM`, `PART_COND_QP_COND_{,LHS,RHS,SEED,SOL,EXPAND_SOL}`.
  - `external/hpipm/cond/x_cond.c` — `COND_QP_COMPUTE_DIM`, `COND_QP_COND_{,LHS,RHS}`.
  - `external/hpipm/dense_qp/x_dense_qp_ipm.c` — dense-QP IPM.
  - `external/hpipm/ocp_qp/x_ocp_qp_seed.c` — QP sensitivity seeds.
  - `acados/ocp_nlp/ocp_nlp_sqp_rti.c` — RTI SQP.
- Local algorithm docs: `acados/docs/algorithm/05-hpipm-qp-solver.md`,
  `06-data-flow.md`.
- This repo: `SQP_PLAN.md` (phases 1–5, finished), `SQP_PHASE{1,2,3,4}_*.md`
  (finished).

## 2. Current state (parity matrix)

| Feature (acados name) | ocp++ status |
|---|---|
| OCP-QP IPM, relative + abs formulation, Mehrotra P-C, conditional pure centering | done (3e) |
| KKT two-pass Riccati, Cholesky + LQ fallback | done (3d) |
| Iterative refinement | done, known guard flaw (SQP_PLAN §12.5) |
| `split_step`, warm start 0/2/3, `m_safe`, `t_lam_min`, `t0_init` | done |
| `square_root_alg` (Riccati variant 0 = classical) | **declared in `HpipmOptions` but never read — classical path missing** |
| GLM / no-regularization | done |
| Merit backtracking + funnel + full-step dual + sufficient descent | done (2d, 3i) |
| Second-order correction pre-pass | done (3j) |
| Adaptive LM, timeout heuristic, QP scaling + adaptive QP tolerances | done (3f, 3g, 3h) |
| GN / EXACT Hessian split (`compute_hess_dyn/constr` + `has_*_hess`) | done |
| **PARTIAL_CONDENSING** (block grouping, `N2`/`block_size`) | done (Phase 2; uniform `NB`, non-uniform `block_size` deferred) |
| **FULL_CONDENSING** (dense QP + dense IPM) | done (Phase 1) |
| **RTI SQP** (standard `sqp_rti`: LHS+RHS per call, full step, no line search) | done (Phase 3, `RtiSolver` in `solvers/acados/rti.hpp`; `_reversible` / `_merit` variants out of scope) |
| **QP sensitivities** (forward/adjoint, HPIPM seed module) | done (Phase 4, staged forward + adjoint; dense / partial-condensed deferred) |
| `print_level = 2` (detailed per-iteration printout) | partial (0/1 only) |
| condensing/expansion time in statistics | done (`SqpIteration::cond_time`) |

Note: the staged path (`kStaged`) solves the *uncondensed* staged QP
directly. Acados' default `PARTIAL_CONDENSING_HPIPM` with
`qp_solver_cond_N = N` (the default) performs an *identity* block
condensing before the same OCP-QP IPM, so the staged behaviour is
numerically the identity-condensing case; both the non-trivial `N2 < N`
grouping (Phase 2) and the dense path (Phase 1) are implemented.

## 3. Resolved decisions

| # | Decision |
|---|---|
| Q1 | Solver selection lives in `AcadosOptions` as `enum class QpMode { kStaged, kPartialCondensed, kFullCondensed }` (default `kStaged` = current behaviour, zero regression) plus `int qp_cond_N = 0` (0 ⇒ N, mirrors acados `qp_solver_cond_N`). |
| Q2 | `HpipmQpSolver` is refactored to be templated on a **dim-traits** type instead of hardcoding `QpDim<P>`: `template <class P, int NH, class Dim = QpDim<P>>`. All solver bodies must already go through `D::...`; any residual direct `P::` use of structural dimensions is routed through `Dim`. This lets the same IPM solve both the staged `Qp` and the condensed `BlockQp`. |
| Q3 | Partial condensing uses a **uniform compile-time block size `NB`** (template param), requiring `N % NB == 0` (asserted). HPIPM's variable `block_size` (first blocks `bs0+1`) is a later generalization (§8). |
| Q4 | Pinned `x_0`: keep the pin rows (lo == hi equality rows) inside the condensed QP instead of an acados-style `reduce_eq_dof` pre-pass. The condense maps treat them like any two-sided equality row; numerics are identical and it avoids a second reduction pass. `reduce_eq_dof` for DAE-style equality DoFs is out of scope (no `z` variables). |
| Q5 | Sensitivities: QP level only (forward/adjoint on the staged QP via the
      HPIPM seed/linearized-KKT mechanism). NLP-level parameter sensitivities
      are deferred until the problem interface exposes a parameter vector. |
| Q6 | RTI is a separate driver `RtiSolver` in `solvers/acados/rti.hpp` sharing
      QP assembly + slacks with `AcadosSolver` (extract a shared assembly
      helper; keep `AcadosSolver`'s public API untouched). |
| Q7 | Full-condensing dense-QP storage: fixed-size Eigen when `NH` is fixed;
      runtime-sized (`std::vector` + `Eigen::Map`) for `NH = Dynamic`. This is
      the one sanctioned exception to "no dynamic-size Eigen for structural
      quantities", because `nv = nx + N·nu` is runtime-sized in dynamic mode. |
| Q8 | New scalar options are registered in the `ValueRegistry` per
      `DUAL_ACCESS_PLAN.md` (or, if that lands later, added to the pending
      registration list). |

## 4. Phase 1 — full condensing (dense QP)

Files: `solvers/hpipm/dense_qp.hpp` (new), `solvers/hpipm/condense.hpp` (new),
`hpipm.hpp` (extend with `HpipmDenseQpSolver`), `acados.hpp` (dispatch).

1. **Data model** `DenseQp<P, NH>` / `DenseQpSol` / `DenseQpRes`:
   `nv = nx + N·nu`, dense `H` (nv×nv, symmetric), `rqz` (nv), constraint
   matrix `C` (nrow_total × nv), bounds `lo/hi`, `m` (complementarity target,
   soft rows only), `d_mask`-style active-side masking, Z (soft weights).
   Dimensions exactly per `COND_QP_COMPUTE_DIM` (x_part/x_cond: `nv` counts
   x_0 + all u; stage-k box rows for k ≥ 1 become general rows; terminal
   carries no control).
2. **Dense QP IPM** (`HpipmDenseQpSolver`): port
   `d_dense_qp_ipm_solve` — KKT factorization with `kkt_fact_alg`
   (0 null-space, 1 Schur-complement; start with Schur + Cholesky, LQ as
   sub-step), `reg_dual`, `remove_lin_dep_eq` (start with 0), `abs_form`,
   `split_step`, warm start, Mehrotra P-C + conditional pure centering,
   iterative refinement. Options struct mirrors `d_dense_qp_ipm_arg`
   (defaults = HPIPM BALANCE + acados overrides, as done for `HpipmOptions`).
3. **Condense/expand** (`condense.hpp`):
   - `condense_lhs(qp_in, dense_qp)`: forward substitution of the linearized
     dynamics `x_{k+1} = A_k x_k + B_k u_k + b_k` into the flow map;
     `H = Σ J_kᵀ H_k J_k`, `C = [C_0 | 0…] + Σ J_kᵀ C_k J_k`, index maps for
     box/general rows and the `idxs`/`idxs_rev` slack coupling.
   - `condense_rhs(qp_in, dense_qp)`: `rqz = Jᵀ r`, `b`-correction,
     bound offsets, `m`, `Z`.
   - `expand_sol(dense_qp, dense_sol, qp_out)`: x by forward dynamics,
     `π_{k}` from stationarity (backward), map `lam/t` per side.
   - Reference: `x_cond.c` `COND_LHS` / `COND_RHS` / `COND_EXPAND_SOL` —
     re-read at implementation time; deviations recorded in the worklog.
4. **Driver hook**: `AcadosSolver::solve` dispatches on
   `opts_.qp_mode` *after* `scaler_.scale_qp` (acados scales the QP before
   the xcond frontend, `ocp_nlp_sqp.c:594-597`): condense → solve →
   expand, then the existing `rescale_solution` path. Condense/expand time
   recorded in `SqpIteration` (new column `cond_time`, mirroring
   acados `time_qp_xcond`).
5. **Tests**:
   - `tests/hpipm/dense_unit.cpp`: dense IPM vs an independent dense full-KKT
     Eigen solve on a synthetic QP (with/without box, with/without soft rows).
   - `tests/hpipm/condense_full.cpp`: N=5, nx=2, nu=1 staged QP with soft rows
     and pinned x_0 — condensed-solve + expansion must match the staged
     `HpipmQpSolver` result to ≤ 1e-10 (primal, pi, lam, t); expansion
     residuals (dynamics / stationarity) ≤ 1e-12.
   - Extend `tests/sqp_acados_ref` (mass_spring, N=15): the embedded
     references were generated with acados `FULL_CONDENSING_HPIPM` — rerun
     the comparison with `qp_mode = kFullCondensed`; cost rel. err < 1e-4,
     (x,u) inf-norm < 1e-4 (existing thresholds).

Gate: warning-free build, all existing tests unchanged, new tests green.

## 5. Phase 2 — partial condensing (block grouping)

Files: `solvers/hpipm/block_qp.hpp` (new), `condense.hpp`, `hpipm.hpp`,
`acados.hpp`.

1. **Traits refactor** (decision Q2): introduce `template <class P,
   int NH, class Dim = QpDim<P>> class HpipmQpSolver`; the staged `Qp` keeps
   its data model untouched. Verify the solver body only uses `Dim::`
   members (grep for direct `P::nx/nu/nb...` uses; move any into the traits).
   Regression: all `tests/hpipm/*` bit-identical results before/after.
2. **`BlockQpDim<P, NB>` + `BlockQp<P, NH, NB>`**: N2 = N/NB stages
   (`assert(N % NB == 0)`); block stage i (original stages `i·NB..i·NB+NB-1`)
   has variables `(u_{i·NB..i·NB+NB-1}; x_{i·NB}; s_block)` — mirroring
   `PART_COND_QP_COMPUTE_DIM` (internal state box rows → general rows;
   `nu2 = NB·nu`, `ng2 += ng + nbx` of internal stages). Terminal block
   absorbs the terminal stage (no control), as in `x_part_cond.c:116-134`.
3. **Condense/expand**: per-block `COND_BABT` (A/B accumulation), `COND_RSQ`
   (Hessian Schur complement over internal states), `COND_DCT` (constraint
   Jacobian + slack index remap); RHS: `COND_B`, `COND_RQ`, `COND_D`.
   Expansion: internal states via forward dynamics inside each block, `π`
   from per-stage stationarity (`PART_COND_QP_EXPAND_SOL`).
4. **Driver hook**: `kPartialCondensed` + `qp_cond_N` (0 ⇒ N ⇒ identity fast
   path: skip condense/expand entirely and call the staged solver as today).
5. **Tests**:
   - `tests/hpipm/condense_part.cpp`: N=4, NB=2 and N=8, NB=4; condensed
     solve + expansion vs direct staged solve, ≤ 1e-10 all fields.
   - Dim-mapping unit checks against `PART_COND_QP_COMPUTE_DIM` values for a
     problem with pin rows, soft box rows, and ineq/eq/lin groups.
   - End-to-end: double_integrator + masses_chain with NB=2, NB=4: same
     solution as staged within 1e-8; SQP iteration counts comparable.

Gate: warning-free build, full regression, new tests green.

## 6. Phase 3 — RTI SQP

**Done** — see `SQP_HPIPM_PARITY_PHASE3_PLAN.md` (standard `sqp_rti`
only; `_reversible` / `_merit` variants out of scope).

Files: `solvers/acados/rti.hpp` (new), `acados.hpp` (shared assembly
extraction).

1. Extract QP assembly + slack + cost machinery shared by `AcadosSolver`
   and the new `RtiSolver` (free-function template or a small base); no
   public API change to `AcadosSolver`.
2. `RtiSolver`: per acados `ocp_nlp_sqp_rti.c` (re-read at implementation
   time): assemble QP (Hessian treatment per `as_rti_iter` /
   `sqp_rti_reversible` semantics), solve once (any `QpMode` from Phase
   1/2), take the full primal step (alpha = 1), optional dual update
   (`compute_dual_sol`), residual logging (`rti_log_residuals`,
   `rti_log_only_available_residuals`).
3. `AcadosOptions`-style `RtiOptions` struct; `RtiSolver` exposes the same
   `solve(problem, sol)` contract.
4. **Tests**: `tests/sqp/rti_*.cpp` — DI with constant Hessian: one RTI call
   ≈ one SQP QP solve, residual reporting matches; mass_spring RTI vs acados
   reference (generate a `sqp_rti` reference with the standalone C driver,
   extend `tests/sqp_acados_ref`).

Gate: warning-free build, full regression, RTI tests green.

## 7. Phase 4 — QP sensitivities

Files: `solvers/hpipm/seed.hpp` (new), `hpipm.hpp`.

1. `QpSeed<P, NH>` data model (port `hpipm_d_ocp_qp_seed.h`: seeds on `rqz`,
   `b`, `d`, `m`, `H` (sparse-by-stage), `BA`, `DC`, `Z`).
2. Forward: linearized KKT solve for `d(sol)/d(seed)` — reuses the
   factorization left on exit (`update_fact_exit`); adjoint: dual seed pass.
   Port from `x_ocp_qp_seed.c` (re-read at implementation time).
3. API: `HpipmQpSolver::sens_forward(qp, sol, seed, dsol)` /
   `sens_adj(...)`; status codes as usual.
4. **Tests**: FD cross-check — perturb `b_k` / `H_k` / `g_k` by ±h, re-solve,
   central differences vs `sens_forward`; rel. err < 1e-6. Adjoint checked
   against `sens_forward` with transposed seed (duality identity
   `⟨seed_x, dsol_y⟩ = ⟨seed_y, dsol_x⟩`).

Out of scope here: NLP-level parameter sensitivities (needs a parameter
vector in the problem interface, §8 item 3).

Gate: warning-free build, full regression, FD tests green.

## 8. Phase 5 — remaining options and polish

- **`square_root_alg`**: implement Riccati variant 0 (classical, non
  square-root) in `backward_factor_stage` / `forward_stage` — the option is
  currently declared in `HpipmOptions` but never read; add a unit test
  comparing variant 0 vs 1 on a synthetic QP (they must agree to solver
  tolerance).
- **itref guard fix** (SQP_PLAN §12.5): `refine_step` shrink-guard should
  monitor the *actual* KKT residual, not the linearized one; re-run
  `lq_itref_3d` + mass_spring with itref on.
- **`print_level = 2`**: detailed per-iteration dump (QP residuals, mu,
  alpha, lq/itref columns) mirroring acados' verbose mode.
- **Statistics**: `cond_time` column (Phases 1/2) already adds the
  condensing/expansion timing; extend `write_csv` + `HpipmStatistics`
  accordingly.
- **Docs**: README/ARCHITECTURE update (condensing modes, RTI,
  sensitivities); fix the "partial condensing (HPIPM)" comments in the
  example `example_ocp.py` files to describe the current reality (HPIPM
  two-pass Riccati on the uncondensed QP); register new options per
  `DUAL_ACCESS_PLAN.md`.

## 9. Open items (verify at implementation time, not design decisions)

1. acados ordering of QP scaling vs condensing — assumed
   scale-then-condense (`ocp_nlp_sqp.c:594-597`); verify.
2. RTI Hessian semantics per variant (`sqp_rti` vs `_reversible` vs
   `_merit`) — verify in `ocp_nlp_sqp_rti.c` / `ocp_nlp_sqp.c`.
3. NLP-level (parameter) sensitivities: blocked on a parameter-vector
   concept in `problem.hpp`.
4. Non-uniform block sizes (HPIPM variable `block_size`) once uniform NB is
   in.
5. `initialize_next_xcond_qp_from_qp_out` (warm-start the condensed QP from
   the previous iteration's expanded solution) — benchmark the benefit
   first; cheap to add later.
6. Dense-IPM options beyond the core set (`scale`, `comp_res_pred`,
   `remove_lin_dep_eq = 1`) — add on demand.

## 10. Verification (all phases)

- `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`
  warning-free under `-Wall -Wextra -Werror`.
- Full existing test suite (all `tests/*` targets + example binaries) before
  and after each phase; no silently weakened tests.
- `tests/sqp_acados_ref` serves as the standing acados cross-check (its
  embedded references were produced with `FULL_CONDENSING_HPIPM`, so Phase 1
  is verified directly against them).

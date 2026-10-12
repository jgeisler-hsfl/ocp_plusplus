# SQP + HPIPM parity — Phase 3: RTI SQP (standard real-time iteration)

Status: **active** (blocks 3a–3b complete; 3c–3e pending). Builds on Phase 1 (full
condensing) and Phase 2 (partial condensing), both complete. Adds acados'
`SQP_RTI` solver in its **standard** form (`as_rti_iter = 0`,
`as_rti_level = STANDARD_RTI`): the QP is assembled and solved **once** per
call — LHS (Hessian + Jacobians) then RHS (gradients + offsets) — and the
**full primal step (alpha = 1)** is applied without a line search. Mirrors
the standard-RTI path of acados `ocp_nlp_sqp_rti.c`
(`preparation_step` + `feedback_step`).

Reference files (read at implementation time; do not modify):
- `../acados/acados/ocp_nlp/ocp_nlp_sqp_rti.c`
  - `ocp_nlp_sqp_rti` — dispatch on `rti_phase`
  - `ocp_nlp_sqp_rti_preparation_step` — LHS assembly + fixed LM (mu = 1.0)
    + (two-phase only) `regularize_lhs` + `condense_lhs`
  - `ocp_nlp_sqp_rti_feedback_step` — RHS assembly + `regularize_rhs` /
    full `regularize` + `ocp_nlp_solve_qp_and_correct_dual` + step
  - `ocp_nlp_sqp_rti_eval_kkt_residual`, `rti_store_residuals_in_stats`,
    `prepare_full_residual_computation`
- `../acados/acados/ocp_nlp/ocp_nlp_common.c`
  - `ocp_nlp_approximate_qp_matrices` (LHS) /
    `ocp_nlp_approximate_qp_vectors_sqp` (RHS)
  - `ocp_nlp_solve_qp_and_correct_dual` — adaptive QP tolerances + solve +
    `regularize->correct_dual_sol` + QP-scaling rescale
  - `ocp_nlp_add_levenberg_marquardt_term` (RTI preparation passes
    alpha = 1.0, iter = 0; with `with_adaptive_levenberg_marquardt = 0`
    the shift is the fixed `nlp_opts->levenberg_marquardt` (default 0.0)
    on the (u;x) Hessian — the alpha/iter args only matter for the
    adaptive scheme)
- Local algorithm docs: `../acados/docs/algorithm/01-sqp-main-loop.md`,
  `02-qp-approximation.md`, `06-data-flow.md`.

## Scope

**In scope (standard RTI, combined `PREPARATION_AND_FEEDBACK`):**
- `RtiSolver::solve(problem, sol)` = prepare (LHS) + feedback (RHS) + one
  QP solve (any `QpMode` from Phase 1/2) + full primal step (alpha = 1) +
  optional residual logging. No iteration loop, no termination check, no
  line search.

**Out of scope (later generalizations):**
- AS-RTI advanced-step iterations (`as_rti_iter > 0`, levels A–C,
  `preparation_advanced_step`): re-linearization between preparation and
  feedback.
- The two-phase API with LHS pre-factorization (acados `PREPARATION` +
  `FEEDBACK` split; would need factorization caching / a `condense_lhs`
  pre-pass in the staged solver).
- NLP-level parameter sensitivities (Phase 4 territory).

## Key design (fixed up-front)

`RtiSolver` is a second SQP-level driver next to `AcadosSolver`, in
`solvers/acados/rti.hpp`: same `solve(problem, sol)` contract, same
`Solution` output, single non-iterating QP solve.

- **Shared assembly (master plan item 1).** The per-stage assembly
  (`assemble_first` / `assemble_path` / `assemble_term`, today private
  members of `AcadosSolver` writing into `qp_in_`) is extracted into
  free-function templates in a new `solvers/acados/assemble.hpp`, split
  per stage into **LHS** and **RHS** fills, mirroring acados'
  `approximate_qp_matrices` / `approximate_qp_vectors_sqp`:
  - LHS: `hess`, `BA`, `DC`, `lo` / `hi`, slack weights (Z part of
    `fill_slack_diag`), `d_mask`
  - RHS: `grad` (cost gradient + slack linear part), `b` (dynamics
    offset), bound offsets `d_lo` / `d_hi`, complementarity targets `m`
  Each fill takes `(problem, sol, slacks, k, AssembleFlags, target stage)`;
  `AssembleFlags` carries the Hessian gates (`compute_hess_dyn` /
  `compute_hess_constr`, read from `opts_` today). `AcadosSolver::
  assemble_qp` becomes "lhs then rhs" with **zero behavior change** (gate:
  full regression byte-identical). The first-stage infeasibility check
  (degenerate fixed-x0, `kInfeasible`) stays with the first-stage
  assembly. The SOC `d`-rewrite (`soc_rewrite_d_stage`) is
  `AcadosSolver`-specific and stays there (it post-modifies the assembled
  QP, it is not assembly).
- **Hessian / LM.** Like acados RTI preparation: a **fixed** LM shift
  (mu = 1.0) on the (u;x) Hessian blocks when Hessian computation is
  enabled — i.e. `add_lm_term(qp_in_, 1.0)` after LHS assembly (reuse the
  existing `AcadosSolver::add_lm_term` logic; no adaptive LM, no
  iteration-dependent mu).
- **Regularization.** Our regularizers (GLM / no-reg) touch the LHS only
  (`correct_dual_sol` is a no-op for GLM), so the full
  `regularize(qp_in_)` is called in the feedback phase **after** RHS
  assembly — exactly the acados combined-phase ordering. `correct_dual_sol`
  runs after the QP solve (acados `solve_qp_and_correct_dual`).
- **Step.** Full primal step, alpha = 1, via the existing free function
  `detail::apply_sqp_step(sol, step, 1.0, full_step_dual, sol)` (in-place
  is safe: Eigen temporaries) plus the full slack shift
  (`detail::shift_slacks`). At alpha = 1 the dual update is the absolute
  QP multipliers either way, so acados' `compute_dual_sol` (declared but
  never read in the reference C code) needs no option of its own.
- **QP solve.** Same three-mode dispatch as `AcadosSolver::solve_qp`
  (staged / full-condensed / partial-condensed incl. the NB-mismatch
  fallback): `RtiSolver` owns its own `Qp`/`QpSol` workspaces, the
  pluggable `QpSolver`, the dense and block workspaces (`detail::
  BlockQpWorkspaces`), and `solve_qp` becomes shared plumbing — extract
  the dispatch (incl. cold-primal zeroing and `cond_time` accumulation)
  into a small shared helper or keep a parallel copy in `RtiSolver`;
  decide at implementation time, prefer extraction.
- **Scaling / tolerances.** QP scaling: acados RTI never calls
  `qpscaling_scale_qp` (only standard SQP does) → `RtiSolver` does not
  scale. Adaptive QP-tolerance strategy: it is part of acados
  `solve_qp_and_correct_dual` and is a no-op with the default strategy 0;
  port it (needs the pre-solve NLP residuals, which RTI computes anyway
  for logging).
- **Warm start.** Duals persist across `solve()` calls on the same object
  in the QP workspaces (primal cold each call), selected by
  `RtiOptions.qp_warm_start` (HPIPM 0/2/3). acados'
  `warm_start_first_qp_from_nlp` (seed QP duals from NLP multipliers) is
  **not** ported in v1: after the first call the persistent QP duals are
  the same steady-state behavior for MPC use.
- **Status.** RTI has no convergence test: `solve` returns `kSolved` after
  the step; a QP status other than `kSolved` / `kMaxIterations` returns
  `kQpFailure` (acados `ACADOS_QP_FAILURE`); `kNanDetected` propagates as
  such.

## Options

`RtiOptions` (own struct, not `AcadosOptions` — RTI reads a different
subset):
- `QpMode qp_mode` + `int qp_cond_N` (Phase 1/2 modes; 0 = staged)
- `bool compute_hess_dyn = true`, `bool compute_hess_constr = true`
- `double levenberg_marquardt = 1.0` (fixed shift; 0 disables)
- regularizer epsilon via the pluggable `Regularizer` (GLM default)
- `int qp_warm_start = 0` (HPIPM 0 cold / 2 primal+dual / 3 hot)
- `double tau_min = 1e-16` (residual comp floor, as in `AcadosOptions`)
- `bool rti_log_residuals = false`,
  `bool rti_log_only_available_residuals = false`
- QP-tolerance strategy: `nlp_qp_tol_strategy` + the `nlp_qp_tol_*` set
  (default 0 = fixed HPIPM tolerances)
- `int print_level = 0`

## Blocks

1. **3a — LHS/RHS assembly split (done).** Extract
   `assemble_{first,path,term}` into `solvers/acados/assemble.hpp` as
   free-function templates `assemble_stage_{first,path,term}_{lhs,rhs}`,
   with `AssembleFlags{compute_hess_dyn, compute_hess_constr}`;
   `AcadosSolver::assemble_qp` calls lhs + rhs per stage. Gate: warning-
   free build; a new `tests/sqp/assemble_rti_3a.cpp` (in `sqp_unit`)
   assembles one QP via the new path and checks field-by-field equality
   (all stage fields, incl. `d_mask`, `m`) against a QP assembled by the
   old code path kept in the test for one problem (DI + a soft-row
   problem); full regression byte-identical (modulo timing prints).
2. **3b — `RtiOptions` + `RtiSolver` core (done).** New `solvers/acados/rti.hpp`:
   `RtiOptions`, `RtiStatistics` (per-call row: res_stat/eq/ineq/comp,
   qp_status, qp_iter, step_norm, alpha, cond_time), and
   `RtiSolver<P, NH = Dynamic, QpSolver = HpipmQpSolver<P, NH>,
   Regularizer = GlmRegularizer<P, NH>, NB = 0>` with
   `prepare(problem, sol)` (resize, LHS assembly, fixed-LM shift),
   `feedback(problem, sol)` (RHS assembly, pre-step NLP residuals,
   tolerance update, regularize, QP solve in the active `QpMode`,
   `correct_dual_sol`, full-step application incl. slacks, status), and
   `solve(problem, sol)` = prepare + feedback. No iteration, no
   globalizer.
3. **3c — Residual logging + printing.** `rti_log_residuals` stores the
   pre-step NLP residuals (the "available" ones); unless
   `rti_log_only_available_residuals`, a full residual recompute at the
   post-step iterate is appended to the row (acados
   `prepare_full_residual_computation` + second `res_compute`). Note:
   acados' pre-step residuals are the cheap QP-data-based norms; ours are
   the full `compute_nlp_residuals` at the pre-step iterate — a cost
   deviation, recorded in the worklog. `print_level = 1` prints the row.
4. **3d — Tests.** `tests/sqp/rti_3d.cpp` (in `sqp_unit`):
   - DoubleIntegrator (constant Hessian): one RTI `solve` from a warm
     start must reproduce one `AcadosSolver` SQP iteration (assemble +
     LM + regularize + QP solve + full step) in x/u to ~1e-10; a second
     RTI call from the (near-optimal) result takes a ~0 step and stays
     put; residuals reported by `rti_log_residuals` match
     `compute_nlp_residuals`.
   - MassesChain / mass_spring (soft rows, terminal equality): single RTI
     call from the forward-simulation start lands within 1e-6 of the
     `AcadosSolver` solution (QP is its own linearization: one Newton
     step converges) in x/u inf-norm and cost.
   - QP-mode cross-check: same RTI result for `kStaged`,
     `kFullCondensed`, and (N % 2 == 0) `kPartialCondensed` NB = 2 within
     1e-8.
5. **3e — acados reference + docs.** Extend `tests/sqp_acados_ref` with an
   `SQP_RTI` reference (standalone acados C driver, `nlp_solver_type =
   SQP_RTI`, mass_spring QP) compared at the existing 1e-4 thresholds; if
   the acados C build is unavailable in the dev environment, gate 3e on
   it and keep 3d as the parity evidence. Docs: ARCHITECTURE.md (new
   driver + LHS/RHS split), README.md, `qp_mode` / `RtiOptions` noted in
   the DUAL_ACCESS inventory, master-plan parity matrix row "RTI SQP" →
   done, pointer from master plan §6 to this document.

Gate for every block: `cmake --build build` warning-free
(`-Wall -Wextra -Werror`), full test suite green (byte-identical output
for the pre-existing targets, modulo wall-clock prints).

## Open items (verify during implementation, not design)

1. In-place full-step application (`apply_sqp_step` with `start == dest`):
   **resolved (3b)** — `RtiSolver::feedback` applies the full step in
   place; the `rti_3b` test verifies the result against a manual
   re-application from a saved start iterate (bit-exact).
2. Dual warm start across `RtiSolver::solve()` calls with
   `qp_warm_start = 3` (hot) — check the staged solver's hot path accepts
   a cold-primal / warm-dual mix as the QP solver expects.
3. Whether the shared QP-solve dispatch (3b) can be extracted without
   duplicating the `cond_time` / `qp_iter_` bookkeeping — **resolved
   (3b)**: took the parallel copy in `RtiSolver` (`solve_qp` /
   `solve_qp_full_condensed` / `solve_qp_partial_condensed`); extracting
   would have required refactoring the tested phase 1/2 driver. The cost
   / slack-penalty functions were extracted to `cost.hpp` instead, since
   both drivers need identical cost computation.

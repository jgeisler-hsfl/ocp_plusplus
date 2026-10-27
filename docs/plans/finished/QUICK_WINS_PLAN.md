# Quick Wins Plan

Status: implemented — awaiting commit permission

Three "quick win" items from the regrouped `docs/plans/TODO.md`
(2026-10-27 triage), all small and mostly independent. Work happens in the
`quick-wins` branch / worktree.

## B1 — Plan + AGENTS.md link

- [x] Write this plan; link it in AGENTS.md item 4.

## B2 — `ocp::Status` -> string (TODO group D)

Add a diagnostics helper next to the enum; no behavior change to the
solver. (Naming decision: implemented as `ocp::to_string(Status)`
returning `const char*` — matches the standard spelling, allocation-free.)

- [x] `include/ocp/problem.hpp`: after `enum class Status` add
      `constexpr const char* to_string(Status s)` — `switch` over all 11
      enumerators (each maps to its own spelling), out-of-range values
      map to "unknown".
- [x] New test `tests/problem/status_unit.cpp` (plain-main `check()`
      style, like `tests/inspect/inspect_unit.cpp`): every enumerator
      maps to its own spelling; a sentinel value (99, -1) maps to
      "unknown".
- [x] `CMakeLists.txt`: `add_executable(status_unit tests/problem/
      status_unit.cpp)` with `include ${EIGEN3_INCLUDE_DIR}` (problem.hpp
      needs Eigen) + `OCP_WARN_FLAGS`, placed next to `inspect_unit`.
- [x] `docs/tutorial/03-solving-the-ocp.md` §3.5: one sentence pointing
      at `ocp::to_string` for log output.
- [x] Tick the TODO item in the committing change.

## B3 — Funnel globalization in the tutorial (TODO group D)

07 §7.6 currently documents the default merit backtracking but never
shows how to actually use `ocp::Funnel`. Two blockers to fix first:

- [x] `include/ocp/solvers/acados/acados.hpp`: add a public accessor
      next to `last_qp_sol()`:
      ```cpp
      Globalizer& globalizer() { return glob_; }
      ```
      Rationale: `glob_` is private and nothing today lets the user set
      the funnel-specific parameters (only the shared `GlobOptions` via
      `opts_.glob` is reachable, copied at acados.hpp:2143).
- [x] `tests/sqp/funnel_3i.cpp`: new `test_funnel_params_settable()` —
      build a `FunnelSolver`, mutate `solver.globalizer().kappa` and
      `initial_penalty_parameter`, read them back, then solve
      `double_integrator` N=10 to `kSolved` (proves the user-tuned
      parameters survive a solve). Registered in `run_funnel_3i_tests()`.
- [x] `docs/tutorial/07-advanced-topics.md` §7.6: after the
      template-parameter paragraph add the funnel subsection:
      - f/h/b/p accept-type semantics (verified against
        `Funnel::find_acceptable_iterate` in `globalize.hpp`);
      - concrete instantiation
        `AcadosSolver<P, NH, HpipmQpSolver<P, NH>, GlmRegularizer<P, NH>,
        Funnel<P, NH>>` (pattern from `tests/sqp/funnel_3i.cpp`);
      - how to configure: shared options via `opts.glob`, funnel-specific
        via `solver.globalizer()` (note the per-solve `opts.glob` copy);
      - table of the 9 funnel parameters with defaults + meanings.
- [x] Tick the TODO item in the committing change.

## B4 — AutoDiffDae usage example in 06 (TODO group D)

06 explained `AutoDiffDae` in passing (§6.3, §6.5) but never showed how
to write one; usage only existed in `tests/integrators/dae_autodiff_4j.cpp`.

- [x] `docs/tutorial/06-continuous-time-and-integrators.md`: new
      `## 6.6 DAE derivatives via AutoDiffDae` after §6.5; renumbered the
      old 6.6 -> 6.7 and 6.7 -> 6.8 (no other cross-references to 6.6/6.7
      exist; verified by grep). Content:
      - CRTP snippet in the chapter's Duffing model (scalar-templated
        `F = xdot - f`), wiring `ImplicitRkIntegrator<..., Dae, 4,
        GaussLegendre4Tag>` + `ContinuousProblem` + the existing
        `DuffingProblem<Base>` mixin;
      - the `WithHess` template switch: default (full Newton HVPs) vs
        `false` (Gauss-Newton; pair with `Dims::has_dynamics_hess =
        false`);
      - cost note: `F_jac` = one dual pass over `z = [x; xdot; u]`,
        `F_hess_prod` = nested-dual pass; prefer hand-written
        derivatives for cheap closed forms, AutoDiffDae for long or
        numerically-composed residuals;
      - pointer to `tests/integrators/dae_autodiff_4j.cpp`.
- [x] Verify the snippet: scratch-compiled it (self-contained TU in
      `/tmp/opencode`, same includes as `examples/tutorial/continuous.cpp`
      plus a third solve variant) — `kSolved`, cost 7.655647491,
      identical to the OdeToDae IRK-GL4 reference; the snippet as written
      in the tutorial compiles clean.
- [x] Tick the TODO item in the committing change.

## B5 — Finishing

- [x] `docs/plans/TODO.md`: tick B2/B3/B4 items (done in this branch,
      committed with each change).
- [x] Full build `cmake -B build && cmake --build build` (new
      `status_unit` target; -Werror stays green, zero warnings).
- [x] Run: `status_unit` (pass), `sqp_unit` (pass, incl. new
      funnel-params test), `integrators_unit` (pass),
      `tutorial_continuous` (pass, output unchanged),
      `double_integrator` (pass).
- [x] AGENTS.md doc-sync check: `to_string` (covered by tutorial 03
      §3.5) and `globalizer()` (covered by tutorial 07 §7.6); no
      ARCHITECTURE.md / README change needed (both files document design
      principles, not individual API members; the new accessor mirrors
      the existing `options()` / `last_qp()` pattern).
- [ ] On completion + commit permission: move this plan to
      `docs/plans/finished/`, unlink from AGENTS.md item 4.

## Commits (one logical change each, each ticks its TODO item)

1. `problem: add to_string(Status) for diagnostics`
   (problem.hpp, tests/problem/status_unit.cpp, CMakeLists.txt,
   tutorial 03, TODO.md)
2. `solver: expose globalizer accessor, document funnel in tutorial`
   (acados.hpp, tests/sqp/funnel_3i.cpp, tutorial 07, TODO.md)
3. `docs: add AutoDiffDae usage example to tutorial chapter 6`
   (tutorial 06, TODO.md)

## Findings

- The project CMake flags carry `-Wno-maybe-uninitialized
  -Wno-array-bounds` (pre-existing Eigen swap warnings in
  `hpipm.hpp` under `-O2`). Any scratch compile outside CMake needs the
  same flags or it hits `-Werror`.
- `ContinuousProblem` copies the model by value (`Ode ode_;`), so the
  tutorial snippet may pass a named or temporary DAE instance.
- Helper is named `ocp::to_string(Status)` (not `status_name`), matching
  the standard spelling; returns `const char*` (allocation-free).

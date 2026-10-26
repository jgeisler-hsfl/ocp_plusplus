# AGENTS.md
## Core Operational Protocol (Mandatory)
1. **Ask for clarification** when requirements are ambiguous and
  different interpretations would lead to materially different implementations.
2. **Task Breakdown:** Before executing any significant chunk of work, break the task down into distinct execution blocks.
3. **Intermediate Summaries:** After finishing a major sub-task, output a short summary note (e.g., `Checkpoint: [Sub-task] completed. Result: [Key outcome]`). This preserves state and ensures easy recovery if interrupted.
4. **Loop Prevention:** If you repeat reasoning without progress or hit a dead end, STOP immediately. State `[STUCK]`, briefly explain the roadblock, and adjust your strategy before continuing.
5. **When updating tasks or resuming after a long/interrupted/compacted session:** Always call `todoread` first to verify the current state of the panel.

## Project
C++17 framework for numerical optimal control problems (OCP). Strict separation of:
1. the mathematical problem definition (user-implemented, `include/ocp/problem.hpp`), and
2. solver algorithms (templates over the concrete problem type).

The acados repo is expected as a sibling directory (`../acados/` relative to
this repo). It is a read-only reference used to guide the SQP/HPIPM
reimplementation. Do not modify it; do not copy its structure or naming.
Its algorithm documentation lives in `../acados/docs/algorithm/`.
Scripts reference it via the `ACADOS_SOURCE_DIR` environment variable
(defaults to `${ROOT}/../acados`).

## Build & test
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/double_integrator   # interface sanity test; exit 0 = pass
```
Eigen3 is header-only (found at `/usr/local/include/eigen3` or `/usr/local/include`).
No other dependencies. C++17 required.

- Run the tests relevant to the changes.
- Do not suppress compiler warnings merely to obtain a successful build.
- Add tests for new functionality.
- Update tests when changing behavior.
- Do not silently weaken tests to make a change pass.
- Report tests that could not be run and explain why.

## Layout
- `include/ocp/problem.hpp` — the problem interface (central contract): `Problem<Dims, Scalar>`,
  `Trajectory<T, NSlots>`, `Solution<P, NH>`, `BoxSpec` / `LinearSpec` / `TerminalLinearSpec`.
- `include/ocp/solvers/` — solver implementations, one subdirectory per solver
  family. `solvers/hpipm/` holds the HPIPM QP solver (`hpipm.hpp`) and its
  staged QP data model (`qp.hpp`: `QpDim<P>`, `Qp<P,NH>`, `QpSol<P,NH>`,
  `QpRes<P,NH>`).
- `include/ocp/utils/` — cross-cutting support headers shared by problem,
  solvers and the MEX glue. `utils/inspect.hpp` holds the inspect protocol
  (named option/statistics access, `ocp::inspect` walkers).
- `examples/double_integrator/` — minimal concrete problem + finite-difference sanity test.
- `docs/plans/` — working plan docs; `docs/plans/finished/` holds completed
  plans/worklogs (SQP phases 1–4, GN-Hessian, model API, generator,
  phase-1 worklog).
- `../acados/` — sibling repo, reference only (see `ACADOS_SOURCE_DIR`).

## Conventions
- Structural dimensions (`nx, nu, ng, ne, nl, ng_t, ne_t, nl_t`) and the const
  box index sets (`state_box_idx`, ...) are compile-time, supplied via a `Dims`
  tag struct; the active box row counts (`nbx, nbu, nbx_t`) are *derived* in
  `Problem` from the index-set sizes (plain CRTP reading `Derived::nx` fails
  because the base instantiates while the derived class is still incomplete).
- All vectors/matrices are fixed-size Eigen (`Eigen::Matrix<scalar_t, N, M>`); no
  dynamic-size Eigen for structural quantities.
- Horizon `N` is owned by `ocp::Solution` (`sol.N`, the definitive one);
  stage index `k` is `0..N-1`, terminal stage is `k = N`. By default `N` is
  a runtime quantity (set via `Solution(int)` / `resize`, heap trajectories);
  `Solution<P, NH>` (non-negative `NH`) fixes the horizon at compile time and
  switches the trajectories to fixed-extent `std::array` storage. The same
  `NH` pattern applies to `ocp::Qp<P, NH>`, `ocp::QpSol<P, NH>`,
  `ocp::QpRes<P, NH>` in `qp.hpp` (default `NH = Eigen::Dynamic`).
- Per-stage (path-wise) quantities inside solvers use the same
  `Trajectory<T, NSlots>` mechanism: runtime-extent by default, fixed-extent
  `std::array` when `NH` is a compile-time horizon. Solvers are therefore
  templated over the horizon too (e.g. `HpipmQpSolver<P, NH>`), and their
  workspace `resize(N)` allocates only in dynamic mode.
- Extents: `x` has N+1 stages, `u` / dynamics / stage constraints have N,
  terminal quantities have 1.
- Box constraints: only the active rows (const index sets in `Dims`) are
  bounded; `lo`/`hi`/`soft_penalty` vectors have one entry per active row.
- Soft constraint rows are a compile-time decision: per group, a const
  `*_soft_idx` array in `Dims` selects the soft rows; the soft counts
  (`ng_soft`, `nbx_soft`, ...) are derived from the array sizes. Nonlinear
  (in)equality constraints report their slack penalties via
  `stage_{inequality,equality}_constr_soft_penalty(int k)` /
  `terminal_{inequality,equality}_constr_soft_penalty()`; box/linear carry
  them in the spec's `soft_penalty`. Slacks are solver-internal.
- `Dims::fixed_initial_state` selects the initial-state treatment: when true,
  x_0 is a fixed given via `initial_state()` (not a decision variable, not a
  lo==hi stage-0 box); when false (e.g. MHE), x_0 is a decision variable and
  `initial_state()` is not part of the interface.
- Combined (x,u) layout: state first (`[x; u]`).
- Stage/terminal quantities have separate methods and separate dimensions.
- Problem parameters are user-managed inside the concrete class; they do not
  appear in interface signatures.
- The problem interface is non-virtual (static dispatch); every interface stub
  in the base carries a `static_assert(false, ...)` with the exact required
  signature so a missing override fails at the call site.
- Solver-internal state (slacks, barrier/regularization params, multiplier
  orderings) stays inside each solver; only shared solution quantities
  (x, u, KKT multipliers, cost, status) go into `ocp::Solution`.
- Option leaves may declare closed-interval bounds as trailing `vis()`
  arguments in `inspect()` (number or `std::nullopt`;
  `inspect::kMinPositive` = strict-positivity floor). The named writers
  (`inspect::set_field` / `set_double`) enforce them on the double-coerced
  value before any int/bool cast and return a `SetResult`
  (`kOk` / `kUnknownPath` / `kOutOfRange`, with bounds in `lo`/`hi`).
  Bounds are a host-side guardrail: the read path and direct member
  writes bypass them.
- Further information can be found in the ARCHITECTURE.md file.
- Current development plans live in docs/plans. Finished developments are documented 
  in docs/plans/finished.

## Style
- C++17, Allman braces, 4-space indent, no tabs, ≤ 100 chars per line.
- Naming: types `PascalCase`, functions/variables `snake_case`, data members
  with trailing underscore (`N_`), enum values `kPascalCase` (`kSolved`),
  template parameters `PascalCase`.
- Inputs as `const&`, outputs by value or non-const reference; model functions
  are `const`; `noexcept` only on trivial members.
- No exceptions in the numerical core (errors via `Status` / return codes);
  no `new`/`delete`, no global/static state.
- Include order: project headers, then C++ standard, then third-party (Eigen
  last). One `#pragma once` per header.
- Builds must stay warning-free under `-Wall -Wextra -Werror`.

## Comments
- `///` Doxygen on public/interface members: what, dimensions, layout
  conventions (`[x; u]` order, stage extents). Keep in sync with the
  interface contract in `problem.hpp`.
- `//` for non-obvious local decisions only ("why", not "what").
- No comments that restate the code; section banners only in files > 150 lines.

## Documentation 
- Document new dependencies and any additional setup requirements.
- Update documentation when behavior or public interfaces change.
- Explain relevant design decisions when they are not obvious from the code.

## Portability and Safety
- Do not include absolute, machine-specific filesystem paths in documentation,
  source code, or configuration files.
- Do not document usernames, home directories, or local installation paths.
- Never commit credentials, access tokens, passwords, or private keys.
- Do not copy secrets into documentation, source code, or configuration.

## Commits
- Imperative subject, ≤ 72 chars, optional scope prefix matching the layout:
  `problem:`, `solver:`, `example:`, `docs:`, `build:`.
- Body (when present) explains *why*, not *what*; reference the TODO item when
  a commit advances one.
- One logical change per commit; no drive-by reformatting.

## Notes for changes
- Keep `problem.hpp` self-contained (Eigen only) and keep the "Interface
  contract" comment block at the top in sync with the stubs.
- `TODO.md` is the project roadmap; do not edit it without asking.

# AGENTS.md

## Project
C++17 framework for numerical optimal control problems (OCP). Strict separation of:
1. the mathematical problem definition (user-implemented, `include/ocp/problem.hpp`), and
2. solver algorithms (templates over the concrete problem type).

`acados/` is a read-only reference (symlink to the acados repo) used to guide the
SQP/HPIPM reimplementation. Do not modify it; do not copy its structure or naming.
Its algorithm documentation lives in `acados/docs/algorithm/`.

## Build & test
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/double_integrator   # interface sanity test; exit 0 = pass
```
Eigen3 is header-only (found at `/usr/local/include/eigen3` or `/usr/local/include`).
No other dependencies. C++17 required.

## Layout
- `include/ocp/problem.hpp` — the problem interface (central contract): `Problem<Dims, Scalar>`,
  `Trajectory<T, NSlots>`, `Solution<P, NH>`, `BoxSpec` / `LinearSpec` / `TerminalLinearSpec`.
- `examples/double_integrator/` — minimal concrete problem + finite-difference sanity test.
- `acados/` — reference only.

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
  switches the trajectories to fixed-extent `std::array` storage.
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

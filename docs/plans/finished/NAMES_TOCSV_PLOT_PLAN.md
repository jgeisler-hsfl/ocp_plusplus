# Variable names, Solution::to_csv, and plot script

TODO item C: "does ocp::Solution have a to_csv? if not, add it and add a
simple mecanism to plot it png or preferably html (probably best use
python and plotly, leave it to the user to install)"
(docs/plans/TODO.md).

## Motivation

Solution trajectories are currently exported only by hand-written CSV
loops in the examples (e.g. `mass_spring/main.cpp`), with anonymous
`x0`, `u1` column headers. Adding user-supplied state/input names to the
problem definition, a `Solution::to_csv` member, and a small plotly-based
plot script makes results self-describing and easy to inspect.

## Settled design decisions (agreed with the user)

- **Names in `Dims`** as `static constexpr std::array<const char*, nx>`
  (`state_names`) and `std::array<const char*, nu>` (`control_names`),
  both *optional*. Detected via SFINAE (same pattern as the optional
  `np`); default names are `"x_0", "x_1", ...` / `"u_0", "u_1", ...`.
  Exposed as `static std::string Problem::state_name(int i)` /
  `control_name(int i)` (static: names are a compile-time property of
  `Dims`).
- **`Solution::to_csv(const char* path) const`** returns `bool` (no
  exceptions in the numerical core). Columns: `k`, one per state
  component (named), one per control component (named). **No cost
  column.** Rows k = 0..N; the terminal row (k = N) has blank control
  fields (no u_N). Values as `%.12e`, matching the existing example CSVs.
- **Plot:** `tools/plot_solution.py` using python + plotly
  (user installs: `pip install plotly`, plus `kaleido` only for PNG).
  **One subplot per variable**, shared x-axis (stage k), HTML output
  preferred; `.png` output extension as an option.

## Blocks

### 1. Optional names in the problem definition

- [x] `include/ocp/problem.hpp`:
  - add `#include <cstdio>` and `#include <string>`;
  - `detail::has_state_names<D>` / `has_control_names<D>` SFINAE traits
    (mirror of `has_np`), plus size helpers for the static asserts;
  - `Problem`: `static constexpr bool has_state_names /
    has_control_names`, `static_assert` that a supplied array has exactly
    nx / nu entries, and the static accessors `state_name(i)` /
    `control_name(i)`;
  - update the Dims documentation comment block at the top of the file
    (Dims flags list) and the `Problem` class doxygen.
- [x] `examples/double_integrator/double_integrator.hpp`: add
  `state_names = {"q", "v"}`, `control_names = {"a"}` to
  `DoubleIntegratorDims` (demonstrates the feature).

### 2. Solution::to_csv

- [x] `include/ocp/problem.hpp`, inside `Solution`: `to_csv` member as
  specified above, with a `///` doxygen comment (columns, terminal-row
  convention, `bool` return).

### 3. Plot script

- [x] `tools/plot_solution.py`: reads the `to_csv` CSV (stdlib `csv`),
  builds a plotly figure with one subplot per column (the header names),
  shared x-axis over stage k; writes HTML by default, PNG for a `.png`
  output extension (kaleido; friendly pip hint otherwise).
  Empty terminal-row control cells -> NaN (gap in the trace).

### 4. Tests

- [x] `tests/problem/solution_unit.cpp`:
  - a minimal inline problem (nx = 2, nu = 1, no name arrays) ->
    defaults `"x_0", "x_1", "u_0"`;
  - `DoubleIntegrator` -> `"q", "v", "a"`;
  - `to_csv` round-trip: header, row count (N + 1), a few values, blank
    control field on the terminal row;
  - `to_csv` on an unopenable path returns `false`.
- [x] `CMakeLists.txt`: new `solution_unit` target (include dirs:
  `include`, `examples`, Eigen), next to `status_unit`.

### 5. Example demo

- [x] `examples/double_integrator/main.cpp`: short block checking the
  names and writing `double_integrator_solution.csv` via `to_csv`
  (same spirit as the `mass_spring` CSV dump).

### 6. Docs

- [x] `README.md`: mention the optional names and `tools/plot_solution.py`
      (incl. the pip install note).
- [x] `docs/ARCHITECTURE.md`: new section 9 on naming + `to_csv`.
- [x] `docs/tutorial/02-first-problem.md`: Dims table row, code example
      (commented-out) and note on the optional `state_names` /
      `control_names`.
- [x] `docs/tutorial/03-solving-the-ocp.md`: new section 3.9 on exporting
      (`to_csv`) and plotting (`tools/plot_solution.py`); chapter 3 topic
      line in `docs/tutorial/README.md` updated.
- [x] `docs/plans/TODO.md`: item C checked off with a pointer.

## Execution order

1. Block 1 (problem.hpp names + example Dims)
2. Block 2 (to_csv)
3. Block 4 (test + CMake) -> build + run `solution_unit`,
   `double_integrator`, `status_unit`
4. Block 5 (example demo)
5. Block 3 (plot script, smoke test on a generated CSV)
6. Block 6 (docs sweep)
7. Full build + all runnable tests; report anything not run.

## Findings / notes

- The size-check helpers had to be *struct* SFINAE traits (with a
  `::value` member), not function-template partial specializations
  (GCC rejects the latter).
- Test off-by-one fixed during development: data row k lives at
  `lines[k + 1]` because `lines[0]` is the header.
- Environment here has no `pip`, so plotly/kaleido could not be installed
  for a full end-to-end render. The script was smoke-tested with
  `python3 -m py_compile`, the graceful "plotly is not installed" exit,
  and its stdlib-only CSV reader (`read_solution_csv`) run on the
  `double_integrator_solution.csv` produced by the example binary.
  The plotly figure-building / write_html path is untested here; report
  as such.

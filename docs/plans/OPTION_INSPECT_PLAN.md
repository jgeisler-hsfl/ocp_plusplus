# Option & Statistics Dual Access via the `inspect` Protocol

**Status: completed** (2026-10-26, branch `matlab_mex`; all phases P1–P5
green, MEX smoke test passing in MATLAB).

Supersedes the remaining open scope of
`docs/plans/finished/DUAL_ACCESS_PLAN.md`
(the `ValueRegistry` approach for options, and its planned statistics
registry, are replaced by the `inspect` reflection protocol described here;
`MATLAB_MEX_PLAN.md` (finished) stays as the historical MEX record).
`DUAL_ACCESS_PLAN.md` now lives in `docs/plans/finished/` with a pointer
note, and TODO.md item 11 ("dual access approach") is marked done
(user-approved 2026-10-26).

## Motivation

The MEX work (branch `matlab_mex`) exposed two structural flaws:

1. `MexSession` is coupled to `AcadosSolver` (it owns an `AcadosOptions`
   copy and pushes it via `apply_options`), so it is not solver-agnostic.
2. Name-based option access via `ValueRegistry` has three problems:
   - the registry metadata is a `std::vector` (heap allocation at options
     construction and on every copy) — not acceptable for a real-time
     build of the solver;
   - options live in a *copy* owned by the session/solver push pipeline;
   - HPIPM options are not settable at all, and the QP-scaler knobs are
     duplicated in `AcadosOptions` and "handed through" via
     `make_scaler_opts`.

Additionally: statistics fields want *named columns* on the
iteration-row table (no allocation, names may vary per solver, table
access only — no per-value name access).

The `inspect` protocol replaces the registry: every option-holding struct
and every solver component exposes its own scalar leaves and children via
a `static constexpr inspect(Visitor)` that enumerates
`(name, &Self::member)` pairs. A generic dot-path walker then gives
get/set/discovery over the whole tree from a single root (the solver
object itself). Each class is responsible only for its own children; the
mixin `InspectableLeaf` covers the "may or may not expose values" case
(e.g. integrators today, parameters later).

## Confirmed decisions (2026-10-26)

- Name surface: `options.*` (driver), `qp.options.*` (HPIPM),
  `scaling.options.*` (QpScaler) — the double "options" segment is
  intentional (owner.name semantics).
- The 5 duplicated fields are deleted from `AcadosOptions`:
  `qp_warm_start`, `scale_qp_objective`, `scale_qp_constraints`,
  `ub_max_abs_eig`, `lb_norm_inf_grad_obj` (only the 3 SQP test files
  reference them; no examples/generated code).
- New plan doc supersedes `DUAL_ACCESS_PLAN.md`; TODO.md item 11 is
  updated on completion (user-approved).

## 1. Core mechanism: new `include/ocp/inspect.hpp`

Replaces `include/ocp/named_values.hpp` (deleted in P4). Namespace
`ocp::inspect`. All traversal is allocation-free (string literals,
stack); the only allocating APIs are the host-side discovery helpers
(`leaf_names`, `stat_column_names` return `std::vector<std::string>`).

- **`InspectableLeaf`** — mixin with an empty
  `template <class V> static constexpr void inspect(V&&) {}`; inherited by
  classes whose value exposure is optional/undecided.
- **`has_inspect<T>`** — C++17 SFINAE trait (the project is C++17; a
  `requires` expression would be C++20):
  ```cpp
  namespace detail {
  struct probe { template <class N, class M> static void call(N, M) {} };
  }
  template <class T, class = void>
  struct has_inspect : std::false_type {};
  template <class T>
  struct has_inspect<T, std::void_t<decltype(T::inspect(std::declval<detail::probe&>()))>>
      : std::true_type {};
  ```
- **Path splitting** — `split_first(std::string_view, char)` (first dot;
  returns `pair{head, rest}` with empty rest if no dot).
- **Typed access** (the core pattern):
  ```cpp
  template <class T, class Value>
  bool set_field(T& obj, std::string_view path, Value&& value);
  // dotted path; leaf reached -> obj.*mem_ptr = value; child with
  // has_inspect -> recurse; first matching name per level wins;
  // false on unknown path or on a path that ends at a non-leaf child.

  template <class T, class Value>
  bool get_field(const T& obj, std::string_view path, Value& out);
  ```
- **Double-coerced access** (MATLAB is double-only; host/MEX surface):
  ```cpp
  template <class T> bool set_double(T& obj, std::string_view path, double v);
  template <class T> bool get_double(const T& obj, std::string_view path, double& out);
  ```
  Coercion preserved from today's registry semantics: leaf type `double`
  direct; `int` via `static_cast<int>(v)` on write; `bool` via
  `v != 0.0` on write; int/bool read as double.
- **Discovery**:
  ```cpp
  template <class T> std::vector<std::string> leaf_names(const T& obj);
  // all leaf paths, dotted, depth-first, declaration order.

  template <class T> void dump(const T& obj, std::FILE* f = stdout);
  // flat "name = value" lines (int and double formatting matching the
  // current registry dump output).
  ```
- **Statistics table helpers** (generic over any stats type exposing
  `int iter`, `Status status`, `std::vector<Row> rows` where `Row` has
  `inspect` with scalar leaves only):
  ```cpp
  template <class Stats>
  std::vector<std::string> stat_column_names(const Stats& st);
  // leaf names of the row type (source: the string literals in
  // Row::inspect — no allocation, varies per solver).

  template <class Stats>
  bool stat_column(const Stats& st, std::string_view name,
                   std::vector<double>& out);
  // one named column, int/bool rows coerced to double; false on unknown.

  template <class Stats>
  bool stat_table(const Stats& st, std::vector<double>& out,
                  int& rows, int& cols);
  // row-major (rows = st.iter, cols = number of row-leaf fields);
  // same matrix the hard-coded MexSession::stat_rows produced.
  ```

**Access-control note (lock in with a unit test):** the visitor forms
member pointers inside the owning class (`&AcadosSolver::opts_`), which is
legal even for private members; C++ does not re-check access at the point
of `obj.*mem_ptr` application in the free-function walker. Exposed members
therefore stay private.

**Leaf-only semantics:** a path is valid only if its final segment names a
scalar member (double/int/bool). A path ending at a child struct
(has `inspect`) returns false — same as the registry's rejection of
`kRegistry` entries as values.

## 2. Option-owner structs: `ValueRegistry reg` → `static constexpr inspect`

| Struct | File | Change |
|---|---|---|
| `AcadosOptions` | `include/ocp/solvers/acados/acados.hpp` | `inspect` enumerating the remaining 28 scalars + `vis("glob", &AcadosOptions::glob)`; drop the registry ctor block; **delete** `qp_warm_start`, `scale_qp_objective`, `scale_qp_constraints`, `ub_max_abs_eig`, `lb_norm_inf_grad_obj` |
| `GlobOptions` | `include/ocp/solvers/acados/globalize.hpp` | `inspect` over its 5 scalars; drop `reg` |
| `HpipmOptions` | `include/ocp/solvers/hpipm/hpipm.hpp` | `inspect` over its 27 scalars; drop `reg` + ctor adds |
| `QpScaler::Opts` | `include/ocp/solvers/acados/qpscaling.hpp` | new `inspect` over its 4 scalars (`scale_objective`, `scale_constraints`, `ub_max_abs_eig`, `lb_norm_inf_grad_obj`) |

Every option is reachable both as a struct field (`opts.glob.alpha_min`)
and by name from a root (`options.glob.alpha_min`) — the dual-access goal,
no copies, no metadata allocation.

## 3. Component exposure: each class exposes its own children

| Class | File | `inspect` body |
|---|---|---|
| `HpipmQpSolver` | `hpipm.hpp` | `vis("options", &HpipmQpSolver::opts_);` |
| `QpScaler` | `qpscaling.hpp` | `vis("options", &QpScaler::opts_);` |
| `AcadosSolver` | `acados.hpp` | `vis("options", &AcadosSolver::opts_); vis("qp", &AcadosSolver::qp_); vis("scaling", &AcadosSolver::scaler_);` |

Resulting MEX by-name surface: `options.max_iter`, `options.glob.alpha_min`,
`qp.options.mu0`, `qp.options.warm_start`, `scaling.options.scale_objective`, …
— every option of every component, live, from one root.

## 4. Live options, no copies (driver changes in `acados.hpp`)

- **Delete** `AcadosSolver::apply_options` and
  `AcadosSolver::make_scaler_opts`.
- Scaler construction: `scaler_(make_scaler_opts(opts))` → `scaler_()`
  (default `Opts`, scaling off — identical effective defaults); the
  driver never pushes scaling options again; the scaler's `scale_qp` reads
  its own live `opts_` as today. `QpScaler::set_opts` stays for standalone
  use.
- Warm start: the per-iteration read `int ws = opts_.qp_warm_start;`
  becomes `int ws = qp_.options().warm_start;` (const accessor on the QP
  solver). The NLP-level policy `warm_start_first_qp` stays in
  `AcadosOptions`.
- `update_qp_tolerances` continues to write the QP solver's live
  tolerances when `nlp_qp_tol_strategy != 0` (acados behavior,
  documented); with strategy 0 (default) user-set
  `qp.options.res_*_max` are honored as-is.
- The per-solve `GlobOptions` resync into `MeritBacktracking` is
  unchanged.
- Net effect: a name-based set writes the exact object the solver reads on
  the next iteration — the "copy, then push" pipeline is gone.

## 5. Named statistics columns (no registry on statistics)

| Type | Change |
|---|---|
| `SqpIteration` (acados.hpp) | add `inspect` over its 10 scalar fields (`res_stat`, `res_eq`, `res_ineq`, `res_comp`, `qp_status`, `qp_iter`, `step_norm`, `alpha`, `funnel_iter_type`, `funnel_width`) |
| `HpipmIteration` (hpipm.hpp) | add `inspect` over its fields |
| `AcadosStatistics` / `HpipmStatistics` | **no structural change** (`iter`, `status`, `rows`); names, column and table access come entirely from the generic `inspect.hpp` helpers |

This replaces the statistics registry that `DUAL_ACCESS_PLAN.md` §3
planned: names attach without allocation (string literals), vary per
solver (they live in each row struct), and enable table/column access
only — no per-value name access.

## 6. `MexSession` (`include/ocp/mex/mex_api.hpp`) — solver-agnostic, copy-free

- **Drop**: the `AcadosOptions opts_` member, the `apply_options` push in
  `option_set`, the `#include "../solvers/acados/acados.hpp"`, the default
  template argument `Solver = AcadosSolver<Problem, NH>` (configs pass
  `Solver` explicitly already), and the `options()` accessor.
- Options (signatures of the public methods stay the same):
  - `option_names()` → `inspect::leaf_names(solver_)`
  - `option_get(name, out)` → `inspect::get_double(solver_, name, out)`
  - `option_set(name, value)` → `inspect::set_double(solver_, name, value)`
  - `option_dump(f)` → `inspect::dump(solver_, f)`
- Statistics:
  - `stat_rows(out, rows, cols)` → generic `inspect::stat_table(solver_.statistics(), ...)`
  - new `stat_column_names()` → `inspect::stat_column_names(...)`
  - new `stat_column(name, out)` → `inspect::stat_column(...)`
  - `stat_iter()` / `stat_status()` unchanged (duck-typed `.iter` / `.status`).
- Document the **solver contract** in the header: a valid `Solver` type
  provides (1) `inspect` exposing its option children (dotted leaf paths
  are the option names), (2)
  `Status solve(const Problem&, Solution<Problem, NH>&)`, (3)
  `const Stats& statistics() const` where `Stats` has `int iter`,
  `Status status`, `std::vector<Row> rows` and `Row` has a scalar-leaf
  `inspect`. Any future NLP driver meeting this contract drops into the
  session unchanged (flaw 1 fixed).
- Extension point (not implemented here): when a `Problem` later gains
  `inspect` (parameters, integrator options), the session enumerates the
  union of solver- and problem-side leaves and dispatches `option_set`
  problem-side after solver-side. `InspectableLeaf` is the placeholder for
  components that expose nothing yet.

## 7. MEX glue + MATLAB side

- `mex/ocp_mex_impl.hpp`:
  - `stats` command now returns `{iter, status, names (cell of column
    names), rows (row-major matrix)}`;
  - new command `stat_col <name>` → one named column;
  - usage/help text updated to the new option names (`options.*`,
    `qp.options.*`, `scaling.options.*`) and the new stats surface.
- `mex/test_mex.m`: update exercised option names; add a
  `qp.options.*` / `scaling.options.*` set/get round-trip; use the new
  `stats.names` + `stat_col`.
- `mex/README.md`: document the new name surface and stats commands.
- `mex/configs/*.hpp`: unchanged.

## 8. Tests

- **New** `tests/inspect/inspect_unit.cpp` (replaces
  `tests/registry/named_values.cpp`):
  - flat + dotted get/set, first-match-wins, unknown path → false, path
    ending at a child → false;
  - double-API coercion: int/bool write + read round-trips;
  - `leaf_names` content and depth-first order; `dump` output format;
  - `InspectableLeaf` no-op; `has_inspect` true/false;
  - private-member traversal through the walker (locks the access-control
    note);
  - `stat_column_names` / `stat_column` / `stat_table` on a small fake
    stats struct (including an int column coercion); unknown column → false.
- `tests/sqp/options_2e.cpp`: by-name checks move to
  `inspect::get_double(o, "max_iter", ...)` / `set_double`; the checks for
  the 5 deleted fields move to `HpipmOptions` (`warm_start` default) and
  `QpScaler::Opts` defaults.
- `tests/sqp/warm_start_3c.cpp`: `co.qp_warm_start = 0/2` → after
  construction, `inspect::set_double(solver, "qp.options.warm_start", 2)`
  (the solver object is non-const in the test).
- `tests/sqp/qpscaling_3h.cpp`: `off.scale_qp_objective` etc. →
  `QpScaler::Opts` fields directly (`off.scale_objective`, …).
- `tests/mex/mex_api_unit.cpp`: new option names (`options.max_iter`,
  …); add set/get round-trips for `qp.options.*` and `scaling.options.*`;
  stats column names + `stat_column` extraction; live-set behavior (a set
  takes effect on the next solve with no push step).
- `CMakeLists.txt`: test target `named_values` (or wherever registered)
  renamed to `inspect`, pointing at `tests/inspect/inspect_unit.cpp`.

## 9. Documentation

- `ARCHITECTURE.md` — rewrite the "Separation of parameters" paragraph
  around the `inspect` protocol (replaces the `ValueRegistry` paragraph;
  the MEX is the first by-name consumer).
- `mex/README.md`, `mex/test_mex.m` — per section 7.
- On completion (P5): move `docs/plans/DUAL_ACCESS_PLAN.md` to
  `docs/plans/finished/` with a one-line pointer to this plan; mark
  TODO.md item 11 done (approved).

## 10. Out of scope (extension points only)

- Problem-side option exposure (parameters, `IrkOptions` for
  `ImplicitRkIntegrator`) — the `inspect` protocol + `InspectableLeaf`
  make this a later local change; the session's union dispatch is the
  only session-side piece.
- A `stats_qp`-style MEX command exposing QP-level
  (`HpipmStatistics`) rows — the generic helpers already support it.
- C++20 upgrade (the pattern's `requires` clause) — C++17 SFINAE is used.

## Execution phases

Each phase ends with a green build (`cmake --build build`, warning-free
under `-Wall -Wextra -Werror`) and the relevant tests run.

- **P1 — Core + tests.** Add `include/ocp/inspect.hpp`
  (`InspectableLeaf`, `has_inspect`, `split_first`, `set_field` /
  `get_field`, `set_double` / `get_double`, `leaf_names`, `dump`,
  `stat_column_names` / `stat_column` / `stat_table`) and
  `tests/inspect/inspect_unit.cpp`; register the test in CMake. Coexists
  with `named_values.hpp`; no other change.
  *Checkpoint: inspect unit test green.*
- **P2 — Option structs + driver.** Convert
  `AcadosOptions` / `GlobOptions` / `HpipmOptions` / `QpScaler::Opts`
  from `ValueRegistry` to `inspect`; delete the 5 duplicated
  `AcadosOptions` fields; driver: drop `apply_options` +
  `make_scaler_opts`, default-construct the scaler, read warm start from
  `qp_.options().warm_start`; update `options_2e.cpp`,
  `warm_start_3c.cpp`, `qpscaling_3h.cpp` and any other SQP test touching
  the renamed/removed fields.
  *Checkpoint: full SQP test suite green.*
- **P3 — Component + statistics exposure.** `inspect` on
  `HpipmQpSolver`, `QpScaler`, `AcadosSolver`, `SqpIteration`,
  `HpipmIteration`.
  *Checkpoint: build + SQP tests green; a smoke check that
  `inspect::set_double(acados_solver, "qp.options.mu0", 2.0)` round-trips.*
- **P4 — MexSession + MEX.** Rewrite `MexSession` per section 6 (generic
  forwarder, solver contract documented); delete
  `include/ocp/named_values.hpp` and `tests/registry/named_values.cpp`;
  update `mex/ocp_mex_impl.hpp` (stats names + `stat_col`),
  `tests/mex/mex_api_unit.cpp`, CMake.
  *Checkpoint: `mex_api_unit` green.*
- **P5 — MEX build, MATLAB, docs.** Build both MEX configs
  (`-DOCP_BUILD_MEX=ON`, MATLAB on PATH), run `mex/test_mex.m`; update
  `ARCHITECTURE.md`, `mex/README.md`; move `DUAL_ACCESS_PLAN.md` to
  `finished/` with a pointer; mark TODO.md item 11; run the full test
  suite.
  *Checkpoint: MEX plan complete.*

## Verification gates

- `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`
  warning-free (`-Wall -Wextra -Werror`); no suppressed warnings.
- All C++ tests green, including the new `inspect` unit test, the three
  updated SQP tests, and `mex_api_unit`.
- `./build/double_integrator` exit 0 (interface sanity).
- MEX: both configs build; `mex/test_mex.m` passes in MATLAB, including
  the new `stats.names`, `stat_col`, and `qp.options.*` /
  `scaling.options.*` round-trips.
- No heap allocation in option get/set/discovery *of the registry*:
  traversal is literal + stack based; only `leaf_names` /
  `stat_column_names` (host-side discovery) allocate.

# MEX API Improvements Plan (questions.md follow-up)

Status: done

Implements the twelve improvement items collected in `questions.md`
(review of `include/ocp/mex/mex_api.hpp` and `mex/ocp_mex_impl.hpp`).
Branch: `matlab_mex`.

## Decisions (user-confirmed)

| Item | Decision |
|------|----------|
| Q7 stats shape | `[stats, status] = ocp_mex('stats')`; `stats` is a `1 x iter` struct array with one field per column name; `status` (int `ocp::Status`) is the second output and **optional** (`stats = ...('stats')` gives the array only). `stat_col` unchanged. |
| Q8 solve | **No MEX error on non-convergence.** `solve` keeps returning the status, but outputs become optional: `cost = ...('solve')` or `[status, cost] = ...('solve')`. |
| Q9 struct options | `opt_set` overloaded by argument type: `opt_set name value` (existing) or `opt_set optStruct`. Struct nesting mirrors the dotted paths (`s.options.max_iter`); unknown fields → MATLAB warning, skipped; partial structs fine. |
| Q8 init/set/opt_set | Keep the logical success output but make it **optional** (no forced assignment); failure paths keep raising MEX errors. |

## Items (mapping to questions.md)

| # | Item | Phase |
|---|------|-------|
| 3 | `arg_matrix` must check the array is double (`mxIsNumeric` alone is not enough) | P2 |
| 4 | `arg_scalar`: use `mxGetScalar` | P2 |
| 5 | `make_matrix`: check `size == rows*cols` | P2 (subsumed by zero-copy + session-side extent validation; see note) |
| 1+2 | Drop `std::vector<double>` as intermediate matrix; pass the MATLAB `mxGetPr` pointer through | P2 |
| 7 | `stats` returns a struct array; status separate | P3 |
| 8 | operation-status outputs optional / direct errors | P4 |
| 9 | set options via (partial) struct, warn on unknown fields | P5 |
| 10 | `option_dump(stdout)` vs `mexPrintf` | P6 |
| 11 | move `mex/configs/` into `examples/` | P8 |
| 12 | new `Status::kNotInitialized`, returned by `MexSession::solve` | P1 |
| 6 | "the caller must not destroy it afterwards" comment | P9 |

## Phases

### P1 — `Status::kNotInitialized`

- `include/ocp/problem.hpp`: add `kNotInitialized` **after** `kTimeout`
  (appending preserves the numeric values of all existing enumerators;
  `test_mex.m`'s `status == 1` for `kSolved` stays valid). Comment:
  "solve called before init()".
- `include/ocp/mex/mex_api.hpp`: `MexSession::solve` returns
  `Status::kNotInitialized` instead of `kAborted` when `sol_.N < 1`;
  update the doc comment.
- `docs/tutorial/03-solving-the-ocp.md`: add row
  `| kNotInitialized (10) | solve called before init() | call 'init' / resize first |`
  to the exit-status table.
- `tests/mex/mex_api_unit.cpp`: update the "solve before init" check
  from `kAborted` to `kNotInitialized`.

### P2 — zero-copy field access + strict argument checks

Rationale: `arg_matrix` currently copies the mxArray into a
`std::vector<double>` and `make_matrix` copies a vector back into a fresh
mxArray; both copies are redundant because MATLAB arrays *are*
column-major contiguous doubles, which is exactly the layout the session
wants.

`include/ocp/mex/mex_api.hpp`:
- New `bool field_extent(std::string_view field, int& rows, int& cols)
  const` — allocation-free extent query; false for unknown fields and,
  for all stage/terminal fields, when `sol_.N < 1` (session not
  initialized). Scalars `cost_value` / `status` / `N` are always 1 x 1.
  This also closes a latent edge: `field_get("x")` before `init()`
  would index `traj[0]` on an empty trajectory today.
- `field_get` becomes `bool field_get(std::string_view field,
  double* out) const` — fills a caller-allocated buffer of exactly
  `rows*cols` (from `field_extent`), same column-major layout (column k
  = stage k).
- `field_set` becomes `bool field_set(std::string_view field,
  const double* in, int rows, int cols)` — validation unchanged (rows,
  cols, `in` has `rows*cols` entries implied by the extent check).
- Statistics: `int stat_iters() const` (alias of `stat_iter`), plus
  `void stat_fill(double* out, int rows, int cols) const` filling the
  row-major table (rows = iter, cols = `stat_column_names().size()`);
  `stat_rows` (vector form) removed.
- Keep `<vector>` include: `option_names` / `stat_column_names` still
  return `std::vector<std::string>` (host-side discovery, allowed).

`mex/ocp_mex_impl.hpp`:
- `arg_scalar`: check `mxIsDouble` (excludes logicals/singles), then use
  `mxGetScalar` (asserts 1x1, raises on violation) instead of manual
  element-count + `mxGetPr`.
- `arg_matrix`: add `mxIsDouble` to the existing
  `!mxIsComplex` + ndim-2 checks (fixes Q3: `mxIsNumeric` also passes
  logicals, singles, ints).
- `do_get` / `do_solution`: query `field_extent`, create
  `mxCreateDoubleMatrix(rows, cols)`, fill through `mxGetPr(m)` via
  `field_get`. Zero-extent fields yield 0 x n or n x 0 arrays as before.
- `do_stats`: `mxCreateDoubleMatrix` + `stat_fill` (see P3 for the new
  shape).
- `make_matrix`: delete (no remaining caller). Q5's size check is
  subsumed: buffers are allocated exactly to the queried extent and the
  session re-validates rows/cols on the write path.

`tests/mex/mex_api_unit.cpp`: convert all `field_get` / `field_set` /
`stat_rows` call sites to the pointer-based API (buffers can stay
`std::vector<double>` in the test; pass `.data()`). Add checks:
`field_extent` before `init()` is false for `"x"`, true for `"N"`;
`field_get` with a too-small buffer is not callable (compile-time shape
makes this a caller discipline — instead assert extents match).

### P3 — `stats` as struct array (fixes a latent layout bug)

**Bug found during planning:** the current `do_stats` builds the row
table row-major (inspect `stat_table`) and `make_matrix` memcpy's it
into a *column-major* mxArray — the `st.rows` matrix MATLAB receives has
transposed element placement. Existing tests only check
`size(st.rows, 2) == 10` (always true by construction) and never verify
`st.rows` values, so this went unnoticed. The new struct-array shape is
filled element-by-element and is correct by construction.

`mex/ocp_mex_impl.hpp` `do_stats`:
- `rows = session.stat_iter()`, `cols = session.stat_column_names().size()`
  (host-side name discovery stays allocation-based, as before).
- Fill a flat buffer with `session.stat_fill(buf, rows, cols)`
  (row-major: `buf[r * cols + c]` = row r, column c).
- Build `mxCreateStructMatrix(1, rows, cols, names)` where `names` is
  the column names; then per row r, column c:
  `mxSetFieldNumber(s, r, c, mxCreateDoubleScalar(buf[r*cols + c]))`.
- `plhs[0]` = struct array; `plhs[1]` = `mxCreateDoubleScalar(stat_status())`
  only when `nlhs >= 2` (second output optional, per Q7 decision).
- The old struct fields `iter` / `status` / `names` / `rows` are gone;
  `iter` is `numel(stats)`, `names` is `fieldnames(stats)`.
- `stat_col` command unchanged.

`mex/test_mex.m` stats section:
- `[st, ststatus] = ...('stats')`; `assert(ststatus == status)`;
  `assert(isstruct(st) && numel(st) == ...iter)`.
- `assert(isfield(st, 'res_stat') && isfield(st, 'funnel_width'))`.
- Value check replacing the old row-size check:
  `assert(all(st.alpha >= 0 & st.alpha <= 1))` and cross-check one row
  against `stat_col('alpha')`: `assert(isequal(st.alpha, col))`
  (both are column-ordered per iteration).
- Single-output form: `st2 = ...('stats')` works.

### P4 — command output semantics (Q8)

`mex/ocp_mex_impl.hpp`:
- `init`, `set`, `opt_set`: remove the `if (nlhs < 1) die(...)` guards;
  set `plhs[0]` (logical success) only when `nlhs >= 1`. Failure paths
  keep raising MEX errors (unknown option, wrong extents, read-only
  field, bad N).
- `solve`: no error on non-`kSolved` (user decision). Outputs:
  `nlhs == 1` → `plhs[0] = cost`; `nlhs >= 2` →
  `plhs[0] = status (double), plhs[1] = cost`; `nlhs == 0` allowed.
  Update the usage text accordingly.
- `usage()`: update the `init` / `set` / `opt_set` / `solve` / `stats`
  lines to describe optional outputs.

`mex/test_mex.m`: existing two-output `[status, cost]` and asserted
logical outputs stay valid; add:
- `cost2 = ...('solve')` (single-output form) and
  `assert(abs(cost2 - cost) < 1e-12)`.
- bare calls `ocp_mex_...('opt_set', 'options.max_iter', 40)` and
  `('set', 'u', u)` without output.
- `stats` single-output form.

### P5 — `opt_set` with a struct (Q9)

No session change needed: the glue walks the MATLAB struct and calls
the existing `session.option_set(path, value)` per leaf.

`mex/ocp_mex_impl.hpp`:
- `opt_set` dispatch: second argument char → existing name/value path;
  second argument struct → `set_option_struct(prhs[2], "", ...)`.
- `set_option_struct(mxArray* s, const std::string& prefix)`:
  iterate `mxGetNumberOfFields` / `mxGetFieldNameByNumber` /
  `mxGetFieldByNumber`:
  - double scalar (non-complex, 1 element) → `path = prefix + name`;
    if `session.option_set(path, value)` fails → `mexWarnWithId` with
    id `ocp_mex:UnknownOption` and skip;
  - struct → recurse with `path + "."`;
  - anything else (cell, char, non-scalar numeric) →
    `mexWarnWithId` (`ocp_mex:BadOptionValue`) and skip.
- `usage()` + README: document both `opt_set` forms with an example.

`mex/test_mex.m`:
- Build `opt.options.max_iter = 40; opt.options.glob.alpha_min = 0.1;
  opt.qp.options.warm_start = 2; opt.bogus = 5;`, call
  `ocp_mex_...('opt_set', opt)`; assert the three round-trips via
  `opt_get`; clear/restore afterwards.
- Warning check: `lastwarn('clear')` before, then
  `[~, wid] = lastwarn; assert(strcmp(wid, 'ocp_mex:UnknownOption'))`
  after the call with the `bogus` field.

### P6 — `opt_dump` via `mexPrintf` (Q10)

`include/ocp/inspect.hpp`:
- Add host-side `std::string dump_string(const T& obj)` (allocation
  allowed: host/discovery tier, same tier as `leaf_names`). Implement
  as a mirror of `dump_rec` appending formatted lines to a
  `std::string` (`<sstream>` not needed — direct append with
  `std::to_string`-style formatting; doubles formatted with
  `%.12e` via a small `char[64]` buffer to keep the output identical to
  `dump`). `dump(T, FILE*)` unchanged.

`include/ocp/mex/mex_api.hpp`: add
`std::string option_dump_str() const { return inspect::dump_string(solver_); }`.

`mex/ocp_mex_impl.hpp`: `opt_dump` becomes
`mexPrintf("%s", session.option_dump_str().c_str());` — `mexPrintf` is
the documented MEX console channel; raw `stdout` from a MEX is
unreliable (buffered, may never reach the MATLAB console, and may mix
with MATLAB's own stderr handling). A dev-time probe in
`matlab -batch` will record whether plain `stdout` shows up at all, but
the switch to `mexPrintf` is done regardless.

`tests/mex/mex_api_unit.cpp`: check `option_dump_str()` contains
`options.max_iter = 2.000000000000e+01` (or the corresponding default)
and `scaling.options.scale_objective = false`.
`mex/test_mex.m`: `opt_dump` call stays (must not error).

### P7 — no separate phase

(Reserved: P7 in the old numbering was the struct-array work, now P3.)

### P8 — move `mex/configs/` into `examples/` (Q11)

- `git mv mex/configs/double_integrator.hpp
  examples/double_integrator/double_integrator_mex_config.hpp`
- `git mv mex/configs/mass_spring.hpp
  examples/mass_spring/mass_spring_mex_config.hpp`
- Config headers: include the problem via the same-directory form
  (`#include "double_integrator.hpp"`), update the path comment.
- `CMakeLists.txt`:
  `OCP_MEX_CONFIG="${CMAKE_SOURCE_DIR}/examples/${cfg}/${cfg}_mex_config.hpp"`.
- Remove the now-empty `mex/configs/` directory.
- Update path references: `mex/ocp_mex_main.cpp` header comment,
  `mex/README.md` ("Adding a configuration"),
  `docs/plans/MATLAB_MEX_PLAN.md` is historical (leave as-is, it is a
  finished plan).

### P9 — ownership comment (Q6)

The comment "mxSetField / mxSetCell take ownership ... the caller must
not destroy it afterwards" reads as if arrays survive across MEX calls.
Rewrite it above the `struct_set_*` helpers to state:

- each helper transfers ownership of the passed array to the owning
  struct/cell (MATLAB does not deep-copy);
- the top-level array returned from `mexFunction` is owned by MATLAB
  after the call returns;
- every MEX call builds its own fresh arrays — nothing crosses call
  boundaries; the only process-persistent state is the C++ `Session`.

### P10 — verification

- Full build incl. MEX targets (`OCP_BUILD_MEX=ON`), warning-free under
  `-Wall -Wextra -Werror` (MEX targets excluded from the flag set, as
  before).
- Run all 14 test binaries, especially `mex_api_unit` and
  `inspect_unit`.
- `matlab -batch "addpath('mex'); test_mex"` (R2025a) — must print
  "All MEX smoke checks passed." including the new struct-array,
  single-output, struct-opt_set and warning checks.
- Manual `opt_dump` console probe (P6 note).

## Commit boundaries (as landed)

1. `problem: add Status::kNotInitialized for pre-init solves`
   (problem.hpp + tutorial table; consumed by mex session in commit 2).
2. `mex: zero-copy field access with strict argument checks`
   (P1 session side, P2; mex_api.hpp + ocp_mex_impl.hpp +
   mex_api_unit.cpp, plus inspect.hpp and the inspect/hpipm/sqp
   unit tests covering the zero-copy session API).
3. `mex: struct-array stats, struct opt_set, optional outputs`
   (P3, P5, P6, P9, and P4 — the optional-outputs usage/dispatch
   hunks interleave with P3/P5, so they landed here; includes the
   stats layout fix and the test_mex.m v2 update).
4. `mex: move configs into examples; document v2 surface`
   (P8; CMake config paths + ocp_mex_main.cpp + README + this plan).
Docs (`mex/README.md`, tutorial row) travel with the commits that
change the behavior they document.

## Open / out of scope

- `TODO.md` is not touched (roadmap edits require user approval).
- IRK integrator options remain out of scope (covered by the same
  inspect mechanism automatically once the integrator exposes one).
- No change to the inspect protocol itself; all work is session/glue
  level plus the one new enumerator.

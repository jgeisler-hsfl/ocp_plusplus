# Chapter 8 — MATLAB (MEX) interface

This chapter covers the MATLAB MEX wrapper: how the MEX binaries are
provisioned on the C++ side (sections 8.1–8.4), and the command surface
you use from MATLAB (sections 8.5–8.8). The wrapper exposes one
(problem, solver) configuration as process-persistent global state,
which makes it convenient for interactive option tuning, scripting, and
embedding a solve into larger MATLAB toolboxes.

Unlike the other chapters, this one has no companion program under
`examples/tutorial/` — the reference material is the wrapper itself
(`mex/`) and its smoke test (`mex/test_mex.m`).

## 8.1 Layering

The wrapper is layered so that almost everything is testable without
MATLAB:

| File | Role |
|------|------|
| `include/ocp/mex/mex_api.hpp` | `ocp::mex::MexSession<Problem, Solver, NH>` — the host-independent core. Owns one problem instance, one solver and the working solution; exposes option discovery/get/set by name, model-parameter discovery/get/set (duck-typed `params()`), solution-field access, `solve()` and the per-iteration statistics. Pure C++ (no `mex.h`). |
| `tests/mex/mex_api_unit.cpp` | Unit test for the session core (target `mex_api_unit`); builds and runs without MATLAB. |
| `mex/ocp_mex_impl.hpp` | Thin `mex.h` adapter: translates MATLAB arrays/strings into session calls and back, and implements the command table (section 8.5). |
| `mex/ocp_mex_main.cpp` | The single compiled translation unit of *every* MEX binary; `#include OCP_MEX_CONFIG` selects the configuration. |
| `examples/<name>/<name>_mex_config.hpp` | The per-configuration "special include" that names the problem/solver types (section 8.2). |
| `mex/test_mex.m` | End-to-end smoke test (section 8.8). |

Two state and type conventions follow from this design:

- **One session per loaded binary.** In `mexFunction`, a function-local
  static owns a single `MexSession`; it persists for the lifetime of the
  MATLAB process. That is the one documented exception to the
  no-static-state rule — an I/O boundary, not the numerical core. `init`
  resizes the working solution (existing stages keep their warm-start
  values), so re-solves at the same horizon stay warm automatically.
- **Everything crosses as double.** MATLAB is double-only, so scalars
  cross the host boundary as double; int and bool options are coerced on
  the way in and out (bools read back as `1.0` / `0.0`).

## 8.2 The configuration header

Each MEX binary is the *same* wrapper compiled against a small
configuration header. The one for the double integrator
(`examples/double_integrator/double_integrator_mex_config.hpp`)
shows the whole contract:

```cpp
namespace ocp_mex
{
    using Problem = DoubleIntegrator;              // chapter 2
    static constexpr int kHorizon = Eigen::Dynamic; // runtime horizon
    using Solver = ocp::AcadosSolver<Problem, kHorizon>;
    static constexpr const char* kName = "double_integrator";
}
```

- **`Problem`** — the concrete problem type from chapter 2.
- **`kHorizon`** — `Eigen::Dynamic` keeps the horizon a runtime quantity
  set by `init` (default); a non-negative integer fixes it at compile
  time (chapter 7.1).
- **`Solver`** — the NLP driver. The session requires the driver to
  expose
  1. a `static constexpr inspect(Visitor)` listing its option children —
     the dotted leaf paths *are* the option names (chapter 3.3);
  2. `Status solve(const Problem&, Solution<Problem, NH>&)`;
  3. `const Stats& statistics() const` with `int iter`, `Status status`,
     `std::vector<Row> rows` and a scalar-leaf `inspect` on `Row`.

  `ocp::AcadosSolver<P, NH>` satisfies the contract; the RTI driver
  (`ocp::RtiSolver`) does not yet expose its options/statistics in this
  form.
- **`kName`** — the configuration name: the binary is
  `ocp_mex_<kName>.*`, and `kName` is printed in error messages and
  returned by `info` (`i.solver`).

## 8.3 Provisioning a new MEX binary

Two steps, no wrapper changes:

1. Add `examples/<name>/<name>_mex_config.hpp` (copy the pattern from
   8.2 and point `Problem`/`Solver` at your types).
2. Add `<name>` to `OCP_MEX_CONFIGS` in the root `CMakeLists.txt`.

Each configuration compiles the same `ocp_mex_main.cpp` with a different
`OCP_MEX_CONFIG` definition. CMake then builds a target
`ocp_mex_<name>` per configuration, producing
`build/ocp_mex_<name>.<ext>` (`.mexa64` on Linux x86-64).

## 8.4 Building the MEX targets

MATLAB is a heavy dependency, so the MEX targets are opt-in:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DOCP_BUILD_MEX=ON
cmake --build build --target ocp_mex_double_integrator ocp_mex_mass_spring
```

- CMake locates MATLAB via `find_package(Matlab)`. If it misses your
  install, pass the root explicitly: `-DMatlab_ROOT=<matlab-root>`
  (the directory containing `bin`/`extern`), or set the `MATLAB` /
  `MATLAB_ROOT` environment variable.
- With `OCP_BUILD_MEX=ON` but no MATLAB found, the MEX targets are
  skipped with a status message; everything else (including
  `mex_api_unit`) still builds and runs.
- The MEX targets intentionally do *not* carry the project's `-Werror`
  flag set: third-party MATLAB headers are not warning-clean under it.

The pure-C++ session test is always available:

```bash
cmake --build build --target mex_api_unit && ./build/mex_api_unit
```

## 8.5 Command surface

All calls take a char-string command as the first argument:
`ocp_mex_<name>(cmd, ...)`. Outputs are optional in the sense that they
are produced only when requested — a bare call still executes the
command. All numerics are doubles.

| Command | Arguments | Returns |
|---------|-----------|---------|
| `info` | – | struct: `nx nu ng ne nl ng_t ne_t nl_t nbx nbu nbx_t fixed_initial_state N solver` |
| `init` | `N` (positive integer) | logical success (optional); resizes the session and sets `x_0` from `initial_state()` when the initial state is fixed |
| `options` | – | cell array of all discoverable option names |
| `opt_get` | `name` | option value as double (int/bool coerced) |
| `opt_set` | `name`, `value` | logical success (optional); writes the live solver, effective on the next `solve` |
| `opt_set` | `optStruct` | as above; field nesting mirrors the dotted names, partial structs are fine |
| `opt_dump` | – | prints a flat `name = value` snapshot to the console |
| `param_names` | – | cell array of all model parameter names from the problem's duck-typed `params()` object (empty cell when the problem declares no parameters) |
| `param_extent` | `name` | leaf extent as `[rows, cols]` (scalars: `[1 1]`) |
| `param_get` | `name` | the leaf as a `rows x cols` double matrix (scalars: `1 x 1`) |
| `param_set` | `name`, `M` | logical success (optional); `M` must have the leaf's extent; writes the live problem parameters, effective on the next `solve` |
| `param_dump` | – | prints a flat `name = value` parameter snapshot to the console |
| `set` | `field`, `M` | logical success (optional); warm start; only `x` and `u` are writable |
| `get` | `field` | matrix; column `k+1` = stage `k` (fields below) |
| `solution` | – | struct with every `get` field |
| `solve` | – | `[status, cost]` or `cost` (outputs optional); `status` = int value of `ocp::Status` |
| `stats` | – | `[stats, status]` (`status` optional); `stats` is a `1 x iter` struct array with one field per column name |
| `stat_col` | `name` | one named statistics column as an `(n x 1)` vector |

Option names are the dotted inspect paths of the live solver
(chapter 3.3), e.g. `options.max_iter`, `options.glob.alpha_min`,
`qp.options.warm_start`, `scaling.options.scale_objective`. The struct
form of `opt_set` mirrors that nesting: `opt.options.max_iter`,
`opt.qp.options.warm_start`, ...

Argument types are checked strictly: scalars must be real scalar
doubles (logicals, ints and singles are rejected); matrices must be
2-D real doubles.

Errors and warnings:

- Unknown command, unknown option name, out-of-range `opt_set` (name
  form), unknown parameter name, out-of-range `param_set`, or a `set` with
  wrong extents / read-only field → MATLAB error
  via the MEX error channel (the C++ core itself stays
  exception-free).
- In the *struct* form, bad leaves do not abort the struct: unknown
  field → warning `ocp_mex:UnknownOption`, out-of-range value → warning
  `ocp_mex:OptionOutOfRange` (the write is skipped), non-scalar
  non-struct value → warning `ocp_mex:BadOptionValue`.
- `solve` before `init` is **not** an error: it returns
  `status == 10` (`kNotInitialized`). `get` / `set` on stage or
  terminal fields before `init` *is* an error.

`status` is the int value of `ocp::Status` (`1` = `kSolved`,
`2` = `kMaxIterations`, ...; full table in section 3.5).

## 8.6 Data layout

Matrices are column-major with column `k+1` = stage `k`. Extents:

| Field | Size | Notes |
|-------|------|-------|
| `x` | `nx x (N+1)` | writable (warm start) |
| `u` | `nu x N` | writable (warm start) |
| `lambda_dyn` | `nx x N` | |
| `lambda_ineq_stage`, `lambda_eq_stage`, `lambda_lin_stage` | `ng/ne/nl x N` | one row per constraint row |
| `lambda_ineq_term`, `lambda_eq_term`, `lambda_lin_term` | `*_t x 1` | |
| `lambda_box_state` | `2*nbx x (N+1)` | active box rows only; layout `[lower; upper]` |
| `lambda_box_control` | `2*nbu x N` | same layout |
| `cost_value`, `status`, `N` | `1 x 1` | |

Zero-extent fields (e.g. `lambda_eq_stage` for the double integrator,
where `ne == 0`) come back as empty arrays.

## 8.7 Example session (double integrator)

```matlab
addpath('build');
N = 10;
assert(ocp_mex_double_integrator('init', N));

% live option tweaking (dotted names reach into the nested groups)
assert(ocp_mex_double_integrator('opt_set', 'options.max_iter', 40));
assert(ocp_mex_double_integrator('opt_set', 'options.glob.alpha_min', 0.1));

% live model-parameter tweaking (the problem's duck-typed params() object)
pn = ocp_mex_double_integrator('param_names');
assert(ocp_mex_double_integrator('param_set', 'w_stage', [2.0; 0.2; 0.05]));
assert(isequal(ocp_mex_double_integrator('param_get', 'w_stage'), [2.0; 0.2; 0.05]));
assert(ocp_mex_double_integrator('param_set', 'w_stage', [1.0; 0.1; 0.01]));  % restore

% or set several at once from a (partial) struct
opt.options.max_iter = 20;
opt.qp.options.warm_start = 2;
ocp_mex_double_integrator('opt_set', opt);

% warm-start the control (optional; the solver also runs from zero init)
u0 = 0.01 * ones(1, N);
assert(ocp_mex_double_integrator('set', 'u', u0));

[status, cost] = ocp_mex_double_integrator('solve');
assert(status == 1);   % kSolved
x = ocp_mex_double_integrator('get', 'x');
assert(size(x, 1) == 2 && size(x, 2) == N + 1);

% per-iteration statistics: one struct element per iteration, one field
% per column name (status is the optional second output)
[st, ststatus] = ocp_mex_double_integrator('stats');
alpha = ocp_mex_double_integrator('stat_col', 'alpha');
assert(ststatus == status && size(alpha, 1) == numel(st));
assert(isequal(st.alpha, alpha));
```

## 8.8 The smoke test

`mex/test_mex.m` exercises the full command table end to end on the
double integrator: `info` / `init`, option discovery and get/set
(including the struct form and the error/warning paths), parameter
discovery and get/set (scalar and matrix leaves, including the
wrong-extent and out-of-range error paths), warm starts, `solve`, field
read-back, `solution`, `stats` / `stat_col`, `opt_dump`,
`param_dump`, and the rejection of unknown commands:

```bash
matlab -batch "cd <repo-root>; addpath('mex'); test_mex"
```

It prints `All MEX smoke checks passed.` on success.

## Where to go from here

- `mex/README.md` — the full command table and build notes.
- `docs/plans/MATLAB_MEX_PLAN.md` and `docs/plans/
  MEX_IMPROVEMENTS_PLAN.md` — design rationale and the v2 command
  surface.
- `tests/mex/mex_api_unit.cpp` — the C++ unit test of the session core.

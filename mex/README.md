# MATLAB MEX wrapper

A generic MEX "template" exposing one (problem, solver) configuration to
MATLAB as process-persistent global state. The C++ core (`MexSession`,
`include/ocp/mex/mex_api.hpp`) is unit-tested without MATLAB via
`mex_api_unit`; the MEX glue (`ocp_mex_impl.hpp`) is a thin `mex.h`
adapter. Design and rationale: `docs/plans/MATLAB_MEX_PLAN.md`.

## Build

The MEX targets are opt-in (MATLAB is a heavy dependency):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DOCP_BUILD_MEX=ON
cmake --build build --target ocp_mex_double_integrator ocp_mex_mass_spring
```

Produces `build/ocp_mex_double_integrator.mexa64` and
`build/ocp_mex_mass_spring.mexa64`. The `.mexa64` suffix (Linux x86-64) is
set by CMake; on other platforms adjust `SUFFIX` in the root `CMakeLists.txt`
to the value reported by `mexext`.

- If CMake cannot locate your MATLAB install, pass the root explicitly:
  `-DMatlab_ROOT=<matlab-root>` (the directory containing `bin`/`extern`),
  or set the `MATLAB`/`MATLAB_ROOT` environment variable.
- With `OCP_BUILD_MEX=ON` but no MATLAB found, the MEX targets are skipped
  with a status message; `mex_api_unit` still builds and runs.
- The MEX targets intentionally do **not** use the project's `-Werror`
  flag set: third-party MATLAB headers are not warning-clean under it.

The pure-C++ test is always available:

```bash
cmake --build build --target mex_api_unit && ./build/mex_api_unit
```

## Adding a configuration

1. Add `examples/<name>/<name>_mex_config.hpp` naming `Problem`,
   `kHorizon`, `Solver` (see
   `examples/double_integrator/double_integrator_mex_config.hpp`).
2. Add `<name>` to `OCP_MEX_CONFIGS` in the root `CMakeLists.txt`.

No wrapper changes; each configuration compiles the same
`ocp_mex_main.cpp` with a different `OCP_MEX_CONFIG` define.

## Commands

All calls take a char-string command as the first argument.

| Command | Arguments | Returns |
| --- | --- | --- |
| `info` | – | struct: `nx nu ng ne nl ng_t ne_t nl_t nbx nbu nbx_t fixed_initial_state N solver` |
| `init` | `N` | logical success (optional output); resizes the session, sets `x_0` from `initial_state` when the initial state is fixed |
| `options` | – | cell of all option names (`options.*`, `qp.options.*`, `scaling.options.*`) |
| `opt_get` | `name` | double (int/bool options coerced; bools are 1.0/0.0) |
| `opt_set` | `name`, `value` **or** `optStruct` | logical success (optional output); the value is written to the live solver, effective on the next `solve`. Struct form: field nesting mirrors the dotted names (`opt.options.max_iter`, `opt.qp.options.warm_start`); unknown fields warn (`ocp_mex:UnknownOption`) and are skipped, so partial structs are fine |
| `opt_dump` | – | prints a flat `name = value` snapshot to the console |
| `set` | `field`, `M` | logical success (optional output); warm start; only `x` (`nx x (N+1)`) and `u` (`nu x N`) are writable |
| `get` | `field` | matrix; column `k+1` = stage `k` (see field list below) |
| `solution` | – | struct with every `get` field |
| `solve` | – | `[status, cost]` or `cost` (outputs optional); `status` = int value of `ocp::Status` (1 = `kSolved`); no error on non-convergence |
| `stats` | – | `[stats, status]` (`status` optional); `stats` is a `1 x iter` struct array, one field per column name, element `r` = iteration `r` |
| `stat_col` | `name` | named statistics column as an `(n x 1)` vector; error on unknown name |

`get`/`solution` fields (names = `Solution` member names, verbatim):
`x`, `u`, `lambda_dyn`, `lambda_ineq_stage`, `lambda_eq_stage`,
`lambda_lin_stage`, `lambda_ineq_term`, `lambda_eq_term`,
`lambda_lin_term`, `lambda_box_state` (`2*nbx`, active rows `[lower; upper]`),
`lambda_box_control` (`2*nbu`), `cost_value`, `status`, `N`.
Zero-extent fields (e.g. `lambda_eq_stage` for the double integrator, where
`ne == 0`) come back as empty arrays.

Errors are raised through the MEX error channel (MATLAB error with a
descriptive message); the C++ core itself stays exception-free.

## Example session

```matlab
addpath('build');
N = 10;
assert(ocp_mex_double_integrator('init', N));

% live option tweaking (dotted names reach the globalization group)
assert(ocp_mex_double_integrator('opt_set', 'options.max_iter', 40));
assert(ocp_mex_double_integrator('opt_set', 'options.glob.alpha_min', 0.1));

% or set several at once from a (partial) struct
opt.options.max_iter = 20;
opt.qp.options.warm_start = 2;
ocp_mex_double_integrator('opt_set', opt);

% warm start (optional; the solver also runs from the zero init)
u0 = 0.01 * ones(1, N);
assert(ocp_mex_double_integrator('set', 'u', u0));

[status, cost] = ocp_mex_double_integrator('solve');
assert(status == 1);      % kSolved
x = ocp_mex_double_integrator('get', 'x');
assert(size(x, 1) == 2 && size(x, 2) == N + 1);

% per-iteration statistics: one struct element per iteration, one field
% per column name (status is the optional second output)
[st, ststatus] = ocp_mex_double_integrator('stats');
alpha = ocp_mex_double_integrator('stat_col', 'alpha');
assert(ststatus == status && size(alpha, 1) == numel(st));
assert(isequal(st.alpha, alpha));
```

The full smoke test (`test_mex.m`) runs the whole table end to end:

```bash
matlab -batch "cd <repo-root>; addpath('mex'); test_mex"
```

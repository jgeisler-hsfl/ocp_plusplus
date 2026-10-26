# MATLAB MEX Wrapper Plan (dual access: options & solution)

Adds a MATLAB-facing entry point for the ocp++ framework: a generic MEX
"template" that, per build, is instantiated for one concrete
problem + solver configuration. The MEX holds the main objects (problem,
options, solver, solution) as process-persistent ("global") C++ state and
exposes

1. the solver's **options** by name (discovery + get/set, including the
   nested `glob.*` group) through the `ValueRegistry` from
   `docs/plans/DUAL_ACCESS_PLAN.md`,
2. the **solution** object (trajectories, multipliers, cost, status),
3. a full `solve` call.

The (problem, solver, configuration) choice is made at **compile time** by a
"special include" — a config header selected via a compiler define — so one
MEX binary serves exactly one configuration while the wrapper itself stays
generic and untouched.

Advances the interactive-environment aim of TODO.md item 11. Per the current
scope decision, this plan covers **options and solution only**; the
statistics structs are *not* registry-ized here (the plan only adds a raw
read of the already-existing `AcadosStatistics` fields for interactive
debugging).

## Goal

From MATLAB, with one loaded MEX per problem configuration:

```matlab
ocp_mex_di('init', 10)                       % horizon N = 10
ocp_mex_di('options')                        % -> cell of all option names (incl. glob.*)
ocp_mex_di('opt_get', 'max_iter')            % -> 20
ocp_mex_di('opt_set', 'max_iter', 40)        % live: applied to the running solver
ocp_mex_di('opt_set', 'glob.alpha_min', 0.1) % nested, dotted
[status, cost] = ocp_mex_di('solve')
x = ocp_mex_di('get', 'x')                   % nx x (N+1)
```

Direct member access in C++ and the by-name access are two views of the same
storage (dual access), exactly as established for the options structs in
`DUAL_ACCESS_PLAN.md`.

## Current state (inventory)

| Object | Location | Relevance |
| --- | --- | --- |
| `AcadosSolver<P, NH, QpSolver, Regularizer, Globalizer>` | `solvers/acados/acados.hpp` | the SQP driver; `solve(problem, sol)`, `options()`, `statistics()`, `resize(N)`, `last_qp()`, `last_qp_sol()` |
| `AcadosOptions` | `solvers/acados/acados.hpp:722` | 33 flat scalars + nested `glob`; carries `ValueRegistry reg` (item 10) |
| `GlobOptions` | `solvers/acados/globalize.hpp:39` | 5 scalars, own registry, linked as `"glob"` |
| `HpipmQpSolver<P, NH>` | `solvers/hpipm/hpipm.hpp` | inner QP solver; holds its own `HpipmOptions opts_` (separate from `AcadosOptions`), plus `set_warm_start()` / `set_residual_tolerances()` |
| `QpScaler<P, NH>` | `solvers/acados/qpscaling.hpp` | owns a private `Opts opts_` (scaling flags + bounds), built once from `AcadosOptions` in the `AcadosSolver` ctor |
| `Solution<P, NH>` | `problem.hpp:977` | `N`, `x` (N+1), `u` (N), `lambda_dyn`/`lambda_ineq_stage`/`lambda_eq_stage`/`lambda_lin_stage` (N), `lambda_ineq_term`/`lambda_eq_term`/`lambda_lin_term` (1), `lambda_box_state` (N+1, `2*nbx`), `lambda_box_control` (N, `2*nbu`), `cost_value`, `status` |
| `ValueRegistry` | `named_values.hpp` | offset-based by-name get/set/has/names/dump; copy-safe |

Usage constraints the design must respect (verified in code):

- **Options are copied into the solver at construction** (`AcadosSolver(opts)`
  → `opts_(opts)`, `scaler_(make_scaler_opts(opts))`, acados.hpp:925-928;
  `HpipmQpSolver(opts)` → `opts_(opts)`, hpipm.hpp:396). Editing an
  `AcadosOptions` *outside* the solver after construction has no effect.
  Exception: the globalization group is re-synced at the start of every
  `solve()` via `glob_.opts = opts_.glob;` (acados.hpp:1156), and the QP
  sub-solver receives `set_warm_start` / `set_residual_tolerances` each
  iteration (acados.hpp:1286, 1446). So "live" options require an explicit
  push into the solver (Design §3).
- **`N` is a runtime quantity** (`NH = Eigen::Dynamic` default):
  `AcadosSolver::resize(N)` (acados.hpp:1033) sizes all workspaces; `solve`
  derives `N` from `sol.N`. The MEX therefore owns a `Solution` and resizes
  the solver per solve.
- **The QP solver's options are a separate surface** from `AcadosOptions`
  (inner `HpipmOptions` defaults; the driver only forwards
  `qp_warm_start` and, for `nlp_qp_tol_strategy != 0`, the tolerance
  strategy fields). v1 exposes `AcadosOptions` (incl. `glob.*`) by name;
  the inner `HpipmOptions` stays driver-internal.
- **MATLAB is double-only**: all MEX scalars are `double`; `int`/`bool`
  options need coercion in both directions.
- The problem type is a **compile-time template parameter** (static
  dispatch, fixed-size Eigen), so one MEX binary = one problem. "Different
  problems" = different MEX builds (see §2).

## Design

### 1. Configuration selection (the "special include")

One generic wrapper, one config header per problem/solver configuration,
selected by a compiler define:

- **Config headers** `mex/configs/<name>.hpp` — the special include. Each one
  includes the concrete problem header and names the types:

  ```cpp
  // mex/configs/double_integrator.hpp
  #pragma once
  #include "double_integrator/double_integrator.hpp"   // via target include dir "examples"
  namespace ocp_mex
  {
  using Problem = DoubleIntegrator;
  static constexpr int kHorizon = Eigen::Dynamic;
  using Solver = ocp::AcadosSolver<Problem, kHorizon>;
  }  // namespace ocp_mex
  ```

- **The MEX source** `mex/ocp_mex_main.cpp` — the only compiled translation
  unit, identical for every configuration:

  ```cpp
  #include OCP_MEX_CONFIG   // stringized path, from -DOCP_MEX_CONFIG=...
  #include "ocp_mex_impl.hpp"
  ```

  `ocp_mex_impl.hpp` (next to it) defines `mexFunction` as a thin adapter
  over `ocp::mex::MexSession<ocp_mex::Problem, ocp_mex::kHorizon,
  ocp_mex::Solver>` (below). It includes `mex.h`.

- **Build switch** (root `CMakeLists.txt`, option `OCP_BUILD_MEX`, default
  OFF): for each configuration a `MODULE`/shared target
  `ocp_mex_<name>` compiling `mex/ocp_mex_main.cpp` with
  `OCP_MEX_CONFIG=\"${CMAKE_SOURCE_DIR}/mex/configs/<name>.hpp\"`, include
  dirs `include`, `examples`, Eigen, and `${MATLAB_INCLUDE_DIR}`, linking
  `${MATLAB_LIBRARIES}` (MXLIB), output name `ocp_mex_<name>` + the MATLAB
  platform suffix (`.mexa64` on Linux x86-64). v1 configurations:
  `double_integrator`, `mass_spring` (the plain-C++ examples; the
  CasADi-backed examples work the same way plus their `.c` sources).
  When `find_package(Matlab)` fails, the MEX targets are skipped with a
  message; the pure-C++ `MexSession` (step 2) and its unit test remain
  buildable and are tested regardless.

Adding a new problem/solver configuration = one config header + one CMake
loop entry; the wrapper code is unchanged.

### 2. Two layers: testable core + thin MEX glue

- **`include/ocp/mex/mex_api.hpp`** — pure C++ (no `mex.h`), the substance:

  ```cpp
  namespace ocp::mex
  {
  template <class Problem, int NH = Eigen::Dynamic,
            class Solver = AcadosSolver<Problem, NH>>
  class MexSession
  {
  public:
      Status init(int N);                 // resize sol_, set x[0] from problem if fixed_initial_state
      std::vector<std::string> option_names() const;   // opts_.reg.names(), incl. glob.*
      bool option_get(std::string_view name, double& out) const;
      bool option_set(std::string_view name, double value);  // + solver_.apply_options(opts_)
      bool field_get(std::string_view field, std::vector<double>& out,
                     int& rows, int& cols) const;  // solution fields, see §5
      bool field_set(std::string_view field, const std::vector<double>& in,
                     int rows, int cols);          // warm start x / u
      Status solve(double& cost);                  // solver_.solve(problem_, sol_); cost = sol_.cost_value
      // statistics (raw, no registry — statistics dual access is a follow-up)
      int stat_iter() const; int stat_status() const;
      bool stat_rows(std::vector<double>& out, int& rows, int& cols) const;
      struct Info { ... nx, nu, ng, ne, nl, *_t, nbx, nbu, nbx_t, fixed_initial_state, N ... };
      Info info() const;
  private:
      Problem problem_;
      AcadosOptions opts_;
      Solver solver_;
      Solution<Problem, NH> sol_;
  };
  }
  ```

  Errors are `bool`/`Status` (project rule: no exceptions in the numerical
  core); the glue layer translates failures into MATLAB errors.

- **`mex/ocp_mex_impl.hpp`** — owns the persistent state and dispatches:

  ```cpp
  namespace {
  // Process-persistent ("global") state of this MEX binary.
  ocp::mex::MexSession<ocp_mex::Problem, ocp_mex::kHorizon, ocp_mex::Solver> session;
  }
  void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
  { /* command string + args -> MexSession calls -> mxArray results */ }
  ```

  A single function-local `static` is the idiomatic MEX persistence
  mechanism (the MEX is loaded once into the MATLAB process; the object
  outlives every `mexFunction` call). This is a deliberate, documented
  exception to the "no global/static state" rule, which governs the
  numerical core; all numerics stay in the (tested) solver classes, the
  static only *references* the same objects a C++ user would hold.

### 3. Making options live (small additive core change)

Because of the copy-at-construction behavior (§ inventory),
`MexSession::option_set` must push the new values into the solver. Add:

```cpp
// qpscaling.hpp
void set_opts(const Opts& o) { opts_ = o; }   // scaling factors re-derived in scale_qp()
```

```cpp
// acados.hpp, AcadosSolver
/// Replace the driver options between solves (e.g. from an interactive
/// environment). Must not be called while a solve() is in flight.
/// glob.* is re-synced at the next solve() (existing behavior); the QP
/// sub-solver picks up qp_warm_start / the tolerance strategy there as well.
void apply_options(const AcadosOptions& o)
{
    opts_ = o;
    scaler_.set_opts(make_scaler_opts(o));
}
```

Optionally (for the future QP-only MEX path, non-goal for v1):
`HpipmQpSolver::apply_options(const HpipmOptions& o)` = `opts_ = o;` plus
`stat_.init(o.stat_max)` when `stat_max` changed.

### 4. Command surface (v1)

| Command | Args | Returns |
| --- | --- | --- |
| `info` | — | struct: `nx nu ng ne nl`, `ng_t ne_t nl_t`, `nbx nbu nbx_t`, `fixed_initial_state`, `N`, `solver` (string) |
| `init` | `N` (int) | `true`/error; resizes `sol_`/solver, sets `x[0] = problem_.initial_state()` when `fixed_initial_state` |
| `options` | — | cell of all discoverable names (`option_names()`, incl. `glob.*`) |
| `opt_get` | `name` (char) | `double` (int/bool coerced) |
| `opt_set` | `name` (char), `value` (double) | logical; dotted names work; type-safe (int/bool coerced from double) |
| `opt_dump` | — | prints `reg.dump(stdout)` (interactive snapshot) |
| `set` | `field` ∈ {`x`,`u`}, matrix | warm start: `x` = `nx × (N+1)`, `u` = `nu × N` |
| `get` | `field` (char) | see §5 |
| `solution` | — | struct with all `get` fields + `N`, `cost_value`, `status` |
| `solve` | — | `[status, cost]` (status = `static_cast<int>(ocp::Status)`, also written to `sol_.status`) |
| `stats` | — | struct: `iter` (int), `status` (int), `rows` (matrix, 10 NLP columns × `iter`) |

Unknown command / bad dimensions / unknown option or field name →
`mexErrMsgTxt` with a descriptive message (the MEX error channel; the C++
core itself stays exception-free).

Conventions: column-major stage order (MATLAB column `k+1` = stage `k`),
active box rows only (`2*nbx`, `[lower; upper]`), doubles throughout.

### 5. Solution field access (flat naming = C++ member names)

`get`/`set` fields, mirroring the `Solution` member names exactly (no rename
layer, consistent with DUAL_ACCESS_PLAN.md's naming decision):

| field | extent |
| --- | --- |
| `x` | `nx × (N+1)` |
| `u` | `nu × N` |
| `lambda_dyn` | `nx × N` |
| `lambda_ineq_stage` / `lambda_eq_stage` / `lambda_lin_stage` | `ng × N` / `ne × N` / `nl × N` (0×N → empty) |
| `lambda_ineq_term` / `lambda_eq_term` / `lambda_lin_term` | `ng_t × 1` / `ne_t × 1` / `nl_t × 1` |
| `lambda_box_state` | `2*nbx × (N+1)` (active rows, `[lower; upper]`) |
| `lambda_box_control` | `2*nbu × N` |
| `cost_value`, `status`, `N` | scalar (double / int / int) |

Flattening is trivial per-stage Eigen → `std::vector<double>`; extents are
compile-time known, so the glue can pre-size the output `mxArray`s.

### 6. Files

New:

- `include/ocp/mex/mex_api.hpp` — `MexSession<...>` (pure C++, testable).
- `mex/ocp_mex_main.cpp` — TU reading `OCP_MEX_CONFIG`, includes the impl.
- `mex/ocp_mex_impl.hpp` — `mexFunction` + static `MexSession` + arg
  conversion.
- `mex/configs/double_integrator.hpp`, `mex/configs/mass_spring.hpp`.
- `mex/test_mex.m` — MATLAB smoke script (loads one MEX, runs the command
  table end-to-end on the double integrator).
- `mex/README.md` — build instructions (incl. the MATLAB-flag caveat),
  command reference, example session.
- `tests/mex/mex_api_unit.cpp` — C++ unit test of `MexSession` (no MATLAB),
  registered as executable `mex_api_unit` in the root `CMakeLists.txt`.

Modified:

- `include/ocp/named_values.hpp` — add double-coerced conveniences
  `bool get_as_double(std::string_view, double&) const` and
  `bool set_from_double(std::string_view, double) const`, dispatching on the
  stored `ValueType` (int/bool ↔ double). This is what a double-only host
  environment needs; the typed `get`/`set` remain for C++.
- `tests/registry/named_values.cpp` — extend with `get_as_double` /
  `set_from_double` round-trips (int, bool, double, dotted name, type-safe
  failure cases).
- `include/ocp/solvers/acados/qpscaling.hpp` — `QpScaler::set_opts`.
- `include/ocp/solvers/acados/acados.hpp` — `AcadosSolver::apply_options`.
- Root `CMakeLists.txt` — `OCP_BUILD_MEX` option + MEX targets;
  `mex_api_unit` target (always on).
- `README.md` — short pointer to `mex/README.md`.
- `ARCHITECTURE.md` — one line under "Separation of parameters" noting the
  MEX wrapper as the first consumer of the by-name option surface.

### 7. Execution steps

1. **Core conveniences + live-options hook.** Add
   `get_as_double`/`set_from_double` to `ValueRegistry` (+ extend
   `tests/registry/named_values.cpp`); add `QpScaler::set_opts` and
   `AcadosSolver::apply_options`. Gate: `cmake --build build` warning-free,
   `named_values_unit` + `sqp_unit` + `qp_unit` pass. Checkpoint: core hooks
   done.
2. **`MexSession` + C++ test.** Implement `include/ocp/mex/mex_api.hpp`;
   write `tests/mex/mex_api_unit.cpp` covering: `info` extents vs
   `QpDim`, `option_names()` completeness (33 flat + 5 `glob.*` leaves),
   `option_get`/`option_set` round-trips incl. dotted `glob.*` and
   int/bool coercion, live effect (set a tolerance, re-solve, observe
   behavior change), warm-start `field_set` of `x`/`u` with wrong extents →
   `false`, `solve` → `kSolved` on the double integrator, `field_get`
   read-back matches `Solution` members, `stat_rows` row count ==
   `stat.iter`. Register `mex_api_unit` in CMake. Gate: warning-free build,
   `mex_api_unit` passes. Checkpoint: testable core done.
3. **MEX glue + configs + build wiring.** Implement `mex/ocp_mex_impl.hpp` +
   `ocp_mex_main.cpp`, the two config headers, and the `OCP_BUILD_MEX` CMake
   block (find_package(Matlab), per-configuration targets with
   `OCP_MEX_CONFIG`). Gate: C++ suite still green; if MATLAB is available in
   the build environment, the MEX targets build; otherwise report that they
   were skipped and why. Checkpoint: MEX builds.
4. **MATLAB smoke test + docs.** Add `mex/test_mex.m` (init → options →
   opt_set → solve → get/solution → stats on the double integrator, with a
   few value checks against known quantities) and `mex/README.md`; pointer
   in `README.md`; one line in `ARCHITECTURE.md`. Run the smoke test when
   MATLAB is available; report if it could not be run. Checkpoint: MEX
   usable from MATLAB.
5. **Full regression.** Rebuild everything under `-Wall -Wextra -Werror`;
   run every test executable and example binary (double_integrator,
   mass_spring, masses_chain, sqp_masses_chain, masseschaingen_gen,
   sqp_masses_chain_gen_equiv, pendulum_on_cart, unicycle, sqp_pendulum,
   sqp_unicycle, sqp_p5probe_erk, sqp_p5probe_irk, qp_dim, qp_unit,
   sqp_unit, sqp_double_integrator, sqp_mass_spring, sqp_acados_ref,
   integrators_unit, sqp_continuous, named_values_unit, mex_api_unit,
   tutorial_minimal, tutorial_continuous, tutorial_advanced). Confirm zero
   behavior change in the existing C++ path (only additive API + new
   wrapper; `apply_options` is opt-in). Checkpoint: MEX plan complete.

## Non-goals (deferred)

- **Statistics dual access** (registry + `column_names()`/`column()` on
  `AcadosStatistics` / `HpipmStatistics`): explicitly out of scope per the
  current options-only direction. The MEX `stats` command still exports the
  existing raw fields (`iter`, `status`, `rows`).
- **The solver-level root registry** (linking options, statistics, QP
  sub-solver, and globalizer registries into one hierarchy — TODO.md item
  11). With it, `opt_get("qp.iter_max")` / `opt_set("glob.opts...", ...)`
  style paths and the inner `HpipmOptions` surface become available without
  touching the wrapper.
- **Problem-side parameter registration** (user-managed fields like
  `DoubleIntegrator::Ts_`) — also item 11 territory.
- **QP-only MEX** (driving `HpipmQpSolver` on a hand-assembled `Qp`),
  multi-problem binaries, fixed-horizon (`NH != Dynamic`) MEX variants,
  and MATLAB-side object/callback abstractions.

## Open decisions (resolved in this plan, recorded for review)

- **One MEX binary per configuration**, selected by
  `-DOCP_MEX_CONFIG=\".../mex/configs/<name>.hpp\"` (the "special include"):
  fits the static-dispatch architecture (the problem type is a compile-time
  template parameter) and keeps each binary small; the config header is the
  single place to change for a new problem/solver combination.
- **Persistence = one function-local `static` `MexSession`** in the MEX TU
  (alternatives: `mexMakeArrayPersistent`/MATLAB-global bookkeeping — more
  code, no benefit for a single-TU MEX). Documented as the sanctioned
  exception to the no-static-state rule (I/O boundary, not numerical core).
- **Live options via `apply_options`** (small additive solver method +
  `QpScaler::set_opts`) rather than reconstructing the solver on every
  option change (which would re-allocate workspaces and drop QP
  warm-start/factorization state). `apply_options` must not be called while
  a `solve` is in flight (documented; trivially true for a synchronous MEX
  command model).
- **MATLAB is double-only** → `ValueRegistry` gains
  `get_as_double`/`set_from_double`; int options round-trip as
  20.0-style values, booleans as 1.0/0.0 (documented in the MEX README).
  The typed `get`/`set` remain the C++ surface.
- **Field names = C++ member names verbatim** (`x`, `lambda_box_state`,
  `cost_value`, ...), consistent with DUAL_ACCESS_PLAN.md's naming decision;
  no rename layer.
- **Two-layer split** (`mex_api.hpp` pure-C++ / `ocp_mex_impl.hpp` MEX glue)
  so the substance is unit-testable in C++ without MATLAB; the glue is a
  thin, mostly mechanical adapter, smoke-tested from MATLAB when available.
- **v1 scope:** SQP driver only, `double_integrator` + `mass_spring` configs
  (the plain-C++ examples; adding a CasADi-backed example = config header +
  CMake entry + its `.c` sources, no wrapper change).

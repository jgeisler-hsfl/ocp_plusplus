# Plan: Tutorial chapter for the MATLAB MEX wrapper

Add tutorial chapter 8 covering (a) how the MEX binaries are provisioned
on the C++ side and (b) the MATLAB command surface.

## Sources of truth

- `include/ocp/mex/mex_api.hpp` — `ocp::mex::MexSession` (pure C++ core)
- `mex/ocp_mex_impl.hpp` — thin `mex.h` glue, command table in header comment
- `mex/ocp_mex_main.cpp` — single compiled TU per binary
- `mex/README.md` — build instructions + command table
- `mex/test_mex.m` — end-to-end smoke test
- `examples/<name>/<name>_mex_config.hpp` — config header pattern
- root `CMakeLists.txt` (lines ~351–384) — `OCP_BUILD_MEX`, `OCP_MEX_CONFIGS`
- `tests/mex/mex_api_unit.cpp` — pure-C++ unit test (no MATLAB needed)

## Execution blocks

- [x] B1: write `docs/tutorial/08-mex-matlab.md`
  - 8.1 What the wrapper is (layering: session core / glue / single-TU
    binary / config header; one persistent session per loaded binary)
  - 8.2 The config header contract (`ocp_mex::{Problem, kHorizon, Solver,
    kName}`) + solver contract (inspect / solve / statistics)
  - 8.3 Provisioning a new MEX binary (config header + `OCP_MEX_CONFIGS`)
  - 8.4 Building (`-DOCP_BUILD_MEX=ON`, `find_package(Matlab)`,
    MATLAB-less fallback, `mex_api_unit` always available)
  - 8.5 The MATLAB command surface (full command table: info, init,
    options, opt_get, opt_set name/value + struct, opt_dump, set, get,
    solution, solve, stats, stat_col; optional outputs; strict argument
    types; error/warning ids `ocp_mex:{UnknownOption,
    OptionOutOfRange, BadOptionValue}`)
  - 8.6 Data layout conventions (column-major, column k+1 = stage k,
    field extents, box multipliers `[lower; upper]`, doubles only,
    zero-extent fields empty; status int values cf. ch. 3.5)
  - 8.7 Example session (double integrator, MATLAB snippet)
  - 8.8 The smoke test (`test_mex.m` + how to run it)
- [x] B2: sync `docs/tutorial/README.md` (chapter table + note that ch. 8
  needs no companion example; MEX targets + `mex/test_mex.m` instead)
- [x] B3: fix stale pointer in `docs/tutorial/07-advanced-topics.md` 7.9
  ("interactive discovery mechanism ... on the roadmap" -> done, see
  ch. 8)
- [ ] B4: report back; commit only on explicit permission

## Findings

- Chapter 8 content cross-checked against `mex_api.hpp`,
  `ocp_mex_impl.hpp`, `mex/README.md`, `test_mex.m`, both config headers,
  and the CMake MEX section (lines ~351–384 of root `CMakeLists.txt`).
- `ocp::RtiSolver` does not satisfy the MexSession solver contract
  (no `inspect`, `RtiStatistics` lacks `iter`/`status`/`rows`), so the
  chapter states only `AcadosSolver` qualifies for MEX use.
- Root `README.md` already points to `mex/README.md`; no change needed
  there.

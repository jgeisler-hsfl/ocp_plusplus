# Multi-parameter support in `acados2ocp_pp.py`

Advance TODO.md §C: *"add support for more than one parameter in
acados2ocp_pp.py"*.

## Goal

Remove the `np > 1` limitation in `tools/acados2ocp_pp.py`. The generator
accepts any `dims.np >= 0` from the acados codegen JSON and emits a
standardized `{name}Params` struct (a fixed-size Eigen vector leaf holding
the whole parameter vector) that is

* shared by the dynamics model (ODE/DAE) and the cost / constraint code,
* exposed through the duck-typed `params()` hook (`include/ocp/problem.hpp`),
* enumerable / settable by name via the inspect protocol (and therefore via
  the MEX `param_*` commands).

## Key facts from investigation

* **Blocker**: `load_spec()` fails on `np > 1` (`acados2ocp_pp.py` ~line 307).
* **`np == 1` today**: a bare `double p_` member in *both* the model struct
  (`emit_header`, ~line 1279) and the problem class (~line 1294) — two
  independent copies. The capi `eval()` signatures take `double p` by value
  (~lines 952, 1009, 1089) and `_arg_entry` passes `(const double*)&p`
  (~line 794). `pval = cnum(spec["parameter_values"][0])` (~line 1123) reads
  only index 0.
* **The `params()` interface** (`problem.hpp`): optional duck-typed hook
  `Params& params()` / `const Params& params() const`. `Params` is a plain
  struct exposing the inspect protocol (`include/ocp/utils/inspect.hpp`),
  supporting scalar *and* fixed-size Eigen-matrix leaves
  (`get_matrix_double` / `set_matrix_double` / `matrix_extent`). The
  `EmptyParams` base stub makes leaf-free problems degrade gracefully.
  `ScaledProblem` forwards `params()` via
  `auto params() -> decltype(base_.params())` (scaling.hpp:1115). The MEX
  glue already reads/writes `problem.params()` generically
  (mex/mex_api.hpp:146-240). Reference pattern: `DiParams` in
  `examples/double_integrator/double_integrator.hpp`.
* **JSON**: `acados_ocp_nlp.json` carries only `dims.np` (count) and
  `parameter_values` (flat initial values). **No parameter names**, so the
  natural generated leaf is a single vector, not named scalars.
* **casadi ABI**: the `p` slot is already an `np`-element argument
  (e.g. `i4` for np=1, `i4[2]` for np=2); `plan_args` already maps the `p`
  slot to size `np_` when `np_ > 0`. Only the C++ glue hard-codes a scalar.
* **`has_params`** (`problem.hpp:1626`) is used only in tests/docs; the
  integrators / ODE-DAE duck-types do **not** inspect `params()`, so adding
  a `params()` accessor to the generated model struct is collision-free.

## Design decisions (confirmed with user)

1. **Unify `np == 1` into the struct path.** One code path for all `np >= 1`
   (a 1-element Eigen vector for `np == 1`). This changes masses_chain's
   generated code (scalar `p_` → 1-vector) and requires regenerating +
   rebuilding it as a regression check.
2. **Single shared object.** The model owns `{name}Params params_`; the
   problem forwards `params()` to it. A small `ContinuousProblem::ode()`
   accessor is added so the problem can reach the stored ODE. This makes
   MEX-set parameters affect dynamics *and* cost/constraints (no stale-copy
   wart).

## Work blocks

All blocks completed. Findings:

* Block 6: masses_chain regeneration changed only the intended lines
  (capi comment, `const double* p` signatures, `double p_` removed,
  `nullptr` p at np=0); the full 27-target suite passes, including
  `sqp_masses_chain` (embedded acados reference) and
  `sqp_masses_chain_gen_equiv`.
* Block 6: the np >= 2 end-to-end fixture is a hand-written casadi-ABI
  explicit ODE (`tests/params/np2/acados_codegen/np2_model/np2_expl_ode_fun.c`,
  `f(q, v, p) = [v; p0*u + p1*q]`) plus a minimal JSON
  (`dims.np = 2`, ERK K4, LINEAR_LS path + terminal cost, no
  constraints); the generator consumed it without changes. The
  `params_np2` test covers the 2-element `p` leaf, inspect round-trip
  (incl. rejection of a wrong element count), the shared-object
  forwarding (`p.params() == p.ode().params()`, also through
  `ScaledProblem`), and that the live `p` feeds both the ODE value and
  its FD Jacobian. The `load_spec` length validation was verified to
  reject a mismatched `parameter_values`.
* The generated np = 0 output intentionally changes (not byte-identical):
  the unused `double p_` member is dropped, dyn evals pass `nullptr`,
  new capi signatures — verified by the full suite.

### 1. Framework accessor [done]
`include/ocp/integrators/continuous_problem.hpp`: add public accessors
```cpp
Ode& ode() { return ode_; }
const Ode& ode() const { return ode_; }
```
Doxygen: "the ODE/DAE model stored by this adapter". No other framework
change.

### 2. Generator: parse [done]
`load_spec()` in `tools/acados2ocp_pp.py`:
* delete the `np_ > 1: fail(...)` guard;
* when `np_ > 0`, validate `len(parameter_values) == np_` (else `fail`);
* keep `spec["parameter_values"]` as the (possibly multi-element) list.
Update the module docstring: feature set `nz == 0, np <= 1` → `np >= 0`;
drop `np > 1 (multiple parameters)` from the "Not supported" list.

### 3. Generator: emit the `Params` struct [done]
New helper `emit_params_struct(name, np, values)` producing, in the header
`namespace ocp` block before the model struct:
```cpp
struct {name}Params
{
    Eigen::Matrix<double, {np}, 1> p;   // np model parameters
    {name}Params() {{ p << v0, v1, ...; }}
    template <class Visitor>
    static constexpr void inspect(Visitor&& vis) {{ vis("p", &{name}Params::p); }}
};
```

### 4. Generator: capi `eval()` signature → pointer [done]
* Unify all three `eval()` emitters to take `const double* p`:
  `_emit_multi_out_struct`, `_emit_ode_jac_struct`, and the generic
  single-output struct in `emit_capi`.
* `_arg_entry("p", sz)`: return `"p"` when `sz > 0` else `"nullptr"`
  (replaces `"(const double*)&p" if sz == 1`).
* `_void_body`: void `p` when no `sz > 0` p slot (replace `sz == 1`).
* `_emit_ode_jac_struct`: VDE arg `tail` uses `", p"` (was `", &p"`) when
  `np_ > 0`.
* Update the capi header comment (drop "np <= 1").

### 5. Generator: model + problem wiring [done]
* **Model struct**: replace `double p_ = {pval};` with
  `{name}Params params_;`; add `params()` / `params() const` accessors; all
  model `eval`/VDE calls pass `params_.p.data()`.
* **Problem class**: delete its own `double p_` member; add public
  `params()` / `params() const` forwarding to `Base::ode().params()`;
  cost / constraint helpers read the p pointer from there (a `pptr()`
  helper or inline `params().p.data()`).
* Update the generated Dims comment (the "user-managed scalar" note) to
  name the `{name}Params` object. `Dims::np` (NLP *sensitivity* vector)
  stays 0 — the model parameter vector is a separate, user-managed object.

### 6. Verification & tests [done]
* Regenerate + rebuild **masses_chain** (`np == 1`); diff `ocp_pp.csv` vs
  `ref_acados.csv` (values unchanged → must match).
* Create a test OCP with `np >= 2` (e.g. a small masses/dynamics problem
  with two parameters), run acados codegen, generate, build, and compare its
  CSV against the acados reference.
* Unit test (extend `tests/params/`): the generated / a small hand-written
  problem's `params()` exposes `p` as a matrix leaf; round-trip via
  `inspect::set_matrix_double` / `get_matrix_double`; assert
  `has_params<...>` is true.
* Full `cmake --build` stays warning-free under `-Wall -Wextra -Werror`.

### 7. Docs [done]
* `tools/acados2ocp_pp.py` docstring (feature set / not-supported list).
* `docs/ARCHITECTURE.md` parameter-handling section: the generator now emits
  a shared `params()` struct for `np >= 1`.
* `docs/plans/TODO.md`: mark the §C multi-parameter item `[x]`.

## Risks / notes

* Biggest regression surface is masses_chain (generated code changes from a
  scalar `p_` to a 1-vector). Covered by work block 6.
* MEX glue, `ScaledProblem`, and the `has_params` trait need **no** changes.
* No `new`/`delete`, no global/static state introduced; the generated
  structs are plain value types.

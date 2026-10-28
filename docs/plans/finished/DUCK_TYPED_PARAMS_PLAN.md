# Duck-typed parameter interface for `Problem`

## Goal

Give `ocp::Problem` a duck-typed **parameter** interface so a single parameter
object can be shared by:

- the dynamics (including an ODE/DAE user class that sits behind an
  integrator such as `ExplicitRkIntegrator` / `ImplicitRkIntegrator`),
- the cost (stage and terminal),
- the constraints (stage and terminal).

Requirements:

1. **Duck-typed** — no base class to inherit. The user defines a plain
   struct; the framework detects and consumes it, exactly like the existing
   duck-typed ODE/DAE model contract (`ode_model.hpp`, `dae_model.hpp`).
2. **Optional / empty stub** — a problem that does not declare parameters
   still compiles and behaves correctly: `params()` yields an empty object and
   all by-name access degrades to "unknown".
3. **Holds scalars and Eigen matrices** — a parameter value is either a
   scalar (`double` / `int` / `bool`) or a fixed-size `Eigen::Matrix`.
4. **Exposable by name** — parameters are enumerable and settable through the
   inspect protocol and the MEX wrapper (mirrors the option surface), because
   the host (MATLAB / interactive tooling) is double-only and wants to read
   and mutate parameters at runtime.

## Distinction from the existing `np` sensitivity vector

`Dims::np` + the `*_sens_p` entries describe **NLP-level parameter-vector
sensitivities** (derivatives of the model w.r.t. a declared vector `p`). This
new interface is a separate concern: **live, user-managed model parameters**
that the model code reads. The two can coexist; a concrete problem may read
its parameters from the shared object *and* still expose `*_sens_p`
derivatives w.r.t. `np`. Do not conflate them in docs or naming.

## Design

### 1. Parameter object: user-defined plain struct (duck-typed)

The user writes an ordinary struct with members that are either scalars or
fixed-size Eigen matrices, plus a `static constexpr inspect(Visitor)` that
lists the members (reusing the inspect leaf protocol):

```cpp
struct DiParams
{
    double Ts      = 0.1;
    double v_max   = 2.0;
    double wq      = 1.0;
    double wf      = 10.0;
    Eigen::Matrix<double, 2, 1> target;   // e.g. a desired final state

    template <class Visitor>
    static constexpr void inspect(Visitor&& vis)
    {
        vis("Ts",     &DiParams::Ts);
        vis("v_max",  &DiParams::v_max, 0.0, std::nullopt);
        vis("wq",     &DiParams::wq,    0.0, std::nullopt);
        vis("wf",     &DiParams::wf,    0.0, std::nullopt);
        vis("target", &DiParams::target);  // matrix leaf
    }
};
```

Ownership is the user's choice (see §4). The framework never dictates where the
object lives; it only needs `problem.params()` to return it.

### 2. `Problem` hook + empty stub

Add to `Problem` (kept self-contained, Eigen-only):

```cpp
/// Default parameter object for problems without parameters.
/// Exposes no leaves (no-op inspect) so by-name access degrades to
/// "unknown" rather than failing to compile.
struct EmptyParams
{
    template <class Visitor>
    static constexpr void inspect(Visitor&&) {}
};

/// Duck-typed parameter access. The concrete problem may override with
/// `Params& params()` / `const Params& params() const` returning a
/// user-defined parameter struct (see DUCK_TYPED_PARAMS_PLAN.md). When not
/// overridden this returns an empty object and carries no leaves.
EmptyParams params()        { return EmptyParams{}; }
const EmptyParams& params() const { /* see §2 note */ }
```

Notes:
- Return **by reference to owned/persistent storage** in the user override so
  that MEX / tooling writes take effect on the live object the ODE/DAE also
  reads. The base stub may return a temporary; it has no leaves, so a write
  is always a no-op `kUnknownPath` and never silently corrupts state.
- Add the entry to the "Interface contract" comment block at the top of
  `problem.hpp`, explicitly marked **optional** (a problem without parameters
  leaves the stub in place).
- Provide a detection trait in `detail`, e.g.
  `has_params<P>` (pointer-to-member identity check: true iff `P` declares its
  own `params()` distinct from `Problem<Dims,Scalar>::params()`). This lets
  tooling / examples branch on "does this problem declare parameters" without
  depending on the concrete return type.

### 3. Extend the inspect protocol for matrix leaves

`include/ocp/utils/inspect.hpp` currently recognises only scalar leaves
(double/int/bool) and inspect-bearing children. A fixed-size Eigen member
falls through both branches and is silently skipped. Add a third leaf kind:

- `detail::is_fixed_matrix<T>` — true for `Eigen::Matrix<S, R, C, ...>` with
  `R > 0 && C > 0` (covers vectors as 1xN / Nx1). This is purely additive;
  existing scalar-only trees (options, statistics rows) are unaffected.
- New walkers / accessors (mirror the scalar ones):
  - `matrix_extent(const T&, path, int& rows, int& cols)`
  - `get_matrix(const T&, path, Mat&)` / `set_matrix(T&, path, const Mat&)`
  - double-flattened host variants (column-major), for MEX / interactive use:
    `get_matrix_double(const T&, path, double* out, int& rows, int& cols)` and
    `set_matrix_double(T&, path, const double* in, int n)` (n must equal
    rows*cols, else reject).
- Update `collect_names` / `dump_rec` / `dump_str_rec` to also emit matrix
  leaves. Chosen dump format for a matrix leaf: `name = [v0 v1 ...]`
  (column-major, space-separated) on a single line, so the flat `name = value`
  convention is preserved. `fill_row` (statistics) stays scalar-only; a matrix
  leaf in a stat row is a contract violation and is skipped (documented).

Keep the existing `walk` / `get_field` / `set_field` / `get_double` /
`set_double` unchanged; the matrix path is a parallel entry point so no
existing caller's lambda arity changes.

### 4. Ownership / sharing pattern (documented)

Because `ContinuousProblem` stores the ODE **by value** and the integrator
holds a `const Ode&`, the ODE/DAE user class cannot take a back-reference to
the problem. The supported sharing pattern is a **user-owned parameter object
passed by reference to every consumer**:

```cpp
DiParams p;                                  // single shared instance
DiOde ode{p};                               // ODE/DAE reads p
DiProblem prob(ode, /*h=*/0.1);             // problem cost/constraints read p
// DiProblem also exposes:  DiParams& params() { return p; }
```

The same `p` is therefore visible to dynamics (through the ODE), cost, and
constraints. For a plain discrete `Problem` (no integrator) the user may
instead own the params as a problem member and return it from `params()`.
Document both shapes in the plan and the tutorial.

### 5. Wrapper forwarding

`ScaledProblem<Base>` must forward the parameter hook so a scaled problem
exposes the same parameters (scaling is coordinate-only; parameters are
invariant). Add:

```cpp
auto params()        { return base_.params(); }
auto params() const   { return base_.params() const; }
```

(Concretely, forward by reference matching the base's return type; verify the
deduced type is the same `Params&` / `const Params&` so MEX writes hit the
underlying object.)

### 6. MEX surface

`include/ocp/mex/mex_api.hpp` (`MexSession`): add, mirroring the option
commands and operating on the **live problem**:

- `param_names()`      -> `inspect::leaf_names<ParamsT>()` (ParamsT =
  decay of `problem_.params()`); empty for the stub.
- `param_extent(name, &rows, &cols)` -> 1x1 for a scalar leaf, RxC for a
  matrix leaf; false when unknown.
- `param_get(name, double* out)` -> scalar into out[0]; matrix flattened
  column-major.
- `param_set(name, const double* in, int n)` -> scalar (n=1) via
  `set_double`, matrix via `set_matrix_double`; returns `inspect::SetResult`.
- `param_dump(f)` / `param_dump_str()` -> `inspect::dump_string(problem_.params())`.

`mex/ocp_mex_impl.hpp`: add the matching commands (`param_names`,
`param_get`, `param_set`, `param_dump`), update `usage()`. Update
`mex/README.md` and `mex/test_mex.m` with the new commands. Because the
surface is generic over `Problem`, no per-example config header change is
required (each example's `*_mex_config.hpp` already binds the problem type).

## Execution blocks

- [x] **Block 1 — inspect matrix leaves.**
  Add `detail::is_fixed_matrix`, `matrix_extent`, `get_matrix`,
  `set_matrix`, `get_matrix_double`, `set_matrix_double`; extend
  `collect_names` / `dump_rec` / `dump_str_rec`. Leave scalar path untouched.
  Tests in `tests/inspect/` (extend the existing inspect test binary).
  *Done (2026-10-28).* Implementation notes:
  - `is_fixed_matrix` is a **struct trait** (like `has_inspect`), so every
    `if constexpr` site must write `is_fixed_matrix<M>::value` — using the
    struct type bare in a boolean context is a parse error.
  - `get_matrix` / `set_matrix` copy **element-wise** (not `out = m`): a
    plain Eigen assignment would trip a compile-time static assert inside the
    template for statically mismatched extents, and the element-wise form
    also handles mixed scalar types (e.g. float leaf into a double buffer).
  - Dump format (decided, resolving the open question): one line per matrix
    leaf, `name = v0 v1 ...` column-major flattened, `%.12e` like the
    scalars. `fill_row` stays scalar-only; a matrix leaf in a stat row is a
    contract violation and is silently skipped (documented in its comment).
  - `inspect_unit` gained the Eigen include dir; the new checks live in
    `test_matrix_leaves()` (extent queries, name discovery, dump format,
    typed + double-flattened round trips, extent/count mismatch rejection,
    float coercion, nested (dotted) matrix leaves, dynamic-extent invisibility).

- [x] **Block 2 — `Problem` hook + stub.**
  Add `EmptyParams` + `params()` / `params() const` to `Problem`, the
  `detail::has_params` trait, and the "optional" entry in the interface
  contract block. Keep `problem.hpp` self-contained.
  *Done (2026-10-28).* Implementation notes:
  - `EmptyParams` is a leaf-free struct with a no-op static `inspect`, so it
    satisfies the duck-typed inspect protocol and all by-name access degrades
    to "unknown" (same pattern as `ocp::inspect`'s empty fallback).
  - The stubs are backed by a private `EmptyParams params_{}` member; **both**
    overloads return a reference to that persistent member (resolves the open
    question: by-reference in the stub as well, so the const path needs no
    temporary and the MEX `param_get` deduces `const EmptyParams&` cleanly).
  - `detail::has_params<P>` is `!std::is_same_v<decay_t<decltype(
    declval<P&>().params())>, EmptyParams>`: every problem derives a
    `params()` from the base, so detection reduces to comparing the decayed
    return type against the stub type.
  - Verified: full build clean (`-Wall -Wextra -Werror`), all 32 test
    binaries pass; scratch TU confirmed trait detection, stub degradation
    (empty `leaf_names`, `kUnknownPath` on get/set) and user-override
    scalar + matrix round trips on const and non-const paths.

- [x] **Block 3 — `ScaledProblem` forwarding.** Forward `params()`.
  *Done (2026-10-28).* Both overloads forward to the wrapped problem's
  `params()`; the const overload names the base's const return type via
  trailing `decltype(std::declval<const Base&>().params())` (the reference
  member `base_` is unaffected by `this` constness, so the body call
  resolves to the base's non-const overload and binds to the const return
  type). Added `#include <utility>` to `scaling.hpp`. All 32 test binaries
  pass.

- [x] **Block 4 — MEX.** `MexSession::param_*` methods + glue commands +
  `usage()` + README + `test_mex.m`.
  *Note: executed after Block 5's example, because the `param_*` unit
  tests (mex_api_unit) need a problem that declares parameters.*
  *Done (2026-10-28).* Implementation notes:
  - `MexSession::param_names / param_extent / param_get / param_set /
    param_dump(_str)` mirror the `option_*` surface; reads go through the
    inspect protocol on `problem_.params()`, scalars report 1 x 1 extent.
    `param_set` is non-const (live write into the session's problem object);
    a wrong element count or unknown name yields kUnknownPath, a scalar
    bounds violation kOutOfRange with the leaf untouched.
  - Glue commands `param_names | param_extent | param_get | param_set |
    param_dump` in `mex/ocp_mex_impl.hpp` (usage table updated); the glue
    pre-checks the extent, so `param_set` fails with a clear message on a
    wrong-size MATLAB matrix and raises `ocp_mex:ParamOutOfRange` on a
    bounds violation.
  - Verified: `mex_api_unit` (new `test_params`: discovery, extents,
    scalar/matrix get, live set round-trips, kOutOfRange on Ts = 0,
    kUnknownPath on unknown name / wrong element count, dump contents)
    passes; full MATLAB smoke run of `test_mex.m` (extended with a
    param section: names, extents, get/set round-trips, wrong-extent and
    unknown-name errors, ParamOutOfRange id + untouched leaf, param_dump)
    passes on the double_integrator config; both MEX binaries compile.

- [x] **Block 5 — example + shared-params test.** Extend the double
  integrator (or a new minimal example) to define a `DiParams` struct with at
  least one scalar and one Eigen-matrix parameter, shared by dynamics, cost
  and constraints; add a focused test that (a) reads a param via
  `params()`, (b) mutates it by name through `inspect::set_*` and confirms a
  model call reflects the change, (c) confirms the empty-stub problem
  reports zero parameter names and `kUnknownPath` on access.
  *Done (2026-10-28).* Implementation notes:
  - Migrated `DoubleIntegrator` to a `DiParams` object: scalars `Ts`
    (bounded strictly positive via `kMinPositive`), `v_max`, `wf` plus the
    matrix leaf `w_stage` (3x1 = [wq; wv; wa]). The six old public members
    were removed; model bodies read `params_.<leaf>`; 19 consumer files
    updated mechanically (`p.Ts_` -> `p.params().Ts`, ...), verified by the
    full 32-binary suite plus the new test.
  - New focused test `tests/params/params_unit.cpp` (CMake target
    `params_unit`): reading by member/name (scalar + matrix, extents,
    name discovery), mutation by name reflected in dynamics/inequality/
    cost calls, bound rejection (Ts=0 -> kOutOfRange, leaf untouched),
    wrong element count rejection, unknown names, empty-stub degradation
    (zero names, kUnknownPath, const path) and the `has_params` trait
    (including through `ScaledProblem`).

- [x] **Block 6 — docs + roadmap.** Update `docs/ARCHITECTURE.md`
  (parameter handling), the tutorial (a subsection under ch. 2 "first
  problem" for the discrete pattern and ch. 6 "continuous" for the
  ODE/DAE sharing pattern, plus ch. 7/8 for MEX param commands), and
  `docs/plans/TODO.md` (close the "share parameters between ODE/DAE and cost
  and constraints" item). Verify `README.md`.
  *Done (2026-10-28).* `ARCHITECTURE.md` gained the "Shared model
  parameters (duck-typed `params()`)" principle bullet plus the
  `ScaledProblem` forwarding note; tutorial ch. 2 gained §2.4 (discrete
  pattern, old 2.4/2.5 renumbered), ch. 6 gained §6.9 (ODE/DAE sharing via
  a user-owned object held by reference), ch. 8 gained the `param_*` table
  rows + example session + smoke-test description, ch. 4's `v_max_`
  reference was updated; `TODO.md` parameter-sharing item closed;
  `README.md` problem-interface bullet mentions the hook.

## Open questions / risks

- ~~**`params() const` storage for the stub.**~~ Resolved in Block 2: both
  stub overloads return a reference to a persistent `EmptyParams params_`
  member (no by-value path, no temporary).
- **Dump format for matrices** — resolved in Block 1: single line per
  matrix leaf, `name = v0 v1 ...` (column-major flattened, `%.12e`);
  `param_dump` uses the same format, confirmed against `test_mex.m`.
- **Scalar type coverage** — confirmed in Block 4: matrix leaves are
  always double-backed at the MEX surface (the glue flattens via
  `static_cast<double>`); the C++ API (`get_matrix` / `set_matrix`) also
  handles mixed scalar types element-wise.

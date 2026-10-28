# Per-stage integrator step size

Closes TODO item B "make it possible to set the integrator step size per
stage from the problem" (`docs/plans/TODO.md`).

## Goal

Allow the OCP-interval step size `h_k` (and the stage start time `t_k`) to
vary per stage, supplied by the problem (user side) instead of being fixed
by the `ContinuousProblem` constructor. The constant-`h` case stays the
default and **all existing problems keep compiling unchanged**.

## Design decisions (settled in the design discussion)

1. **Layering: the step size is a discretization quantity.** It belongs to
   the problem layer, not the ODE/DAE model contract
   (`ode_model.hpp` / `dae_model.hpp` stay unchanged). The integrators do
   not ask the model for the grid.
2. **Integrators become pure functions of `(x, u, t_k, h)`.** `h` moves
   from the constructor to every public entry point. No member state
   depends on `h` (`h_`, `h_ss_` removed; `h_ss = h / NumSteps` becomes a
   per-call local). The forward-cache keys gain `h`.
3. **`ContinuousProblem` gets a CRTP hook:**
   - `struct StepSpec { double h; double t; };` — per-stage interval width
     and start time.
   - New 4th template argument `Derived`, defaulting to the sentinel
     `ocp::ConstantStepSize` (namespace `ocp`, defined in
     `continuous_problem.hpp`).
   - `StepSpec get_step_size(int k) const` in the base:
     `ConstantStepSize` → `{h_, k * h_}`; otherwise
     `static_cast<const Derived*>(this)->get_step_size(k)` (a named
     `Derived` must define its own hook — guarded, see below).
   - All four `dynamics_*` call sites do `auto spec = get_step_size(k);`
     and pass `(spec.t, spec.h)` to the integrator.
4. **Relation to the `params()` SSOT**
   (`docs/plans/finished/DUCK_TYPED_PARAMS_PLAN.md`): the grid / schedule
   data belongs in the user's params struct as an inspect leaf (fixed-size
   Eigen; MEX `param_set`-able at runtime); the problem's `get_step_size()`
   reads `params()` (thin accessor, the "model bodies read
   `params_.leaf`" pattern — here the dynamics discretization reads
   `params_.h_grid`). The framework is agnostic about where the data
   lives; the tutorial shows the params-based shape.

## API

### Integrators (`rk_explicit.hpp`, `rk_implicit.hpp`)

```
ExplicitRkIntegrator(ode)                        // h dropped
value(x, u, t_k, h, x_next)
value_jac(x, u, t_k, h, x_next, df_dx, df_du)
jacobian(x, u, t_k, h, df_dx, df_du)
value_jac_hess_prod(x, u, t_k, h, ...)
hess_prod(x, u, t_k, h, w, v_x, v_u, hv_x, hv_u)

ImplicitRkIntegrator(dae, newton_max = 3, newton_tol = 0.0)   // h dropped
// same h addition to all entry points
```

Note: `ContinuousProblem`'s constructor **keeps** `h` (now the constant
fallback), so `ContinuousProblem` call sites and the generator are
unaffected; only *direct* integrator constructions change.

### ContinuousProblem

```cpp
template <class Dims, class Ode, class Integ,
          class Derived = ConstantStepSize, int NH = Eigen::Dynamic>
class ContinuousProblem : public Problem<Dims>
{
    explicit ContinuousProblem(const Ode& ode, double h, Args&&... args);
    StepSpec get_step_size(int k) const;   // hook, see above
    // dynamics_next_state / dynamics_value_jac / dynamics_value_jac_hess
    // use get_step_size(k)
};
```

Opt-in shape (user code):

```cpp
struct MyProb : ocp::ContinuousProblem<MyDims, MyOde, MyInteg, MyProb>
{
    MyParams params_;
    MyParams& params() { return params_; }
    const MyParams& params() const { return params_; }
    ocp::StepSpec get_step_size(int k) const
    {
        return ocp::StepSpec{params_.h_grid[k], params_.t_grid[k]};
    }
};
```

## Guard against a CRTP misconfiguration

Naming `Derived` without defining its own `get_step_size` would make the
base hook call itself (inherited member) and recurse at runtime. A
class-scope `static_assert` cannot inspect `Derived` (incomplete at base
instantiation — the same reason plain CRTP reading `Derived::nx` fails,
see AGENTS.md conventions), so the check lives **inside**
`get_step_size` (member functions instantiate on first use, after
`Derived` is complete): member-pointer identity of
`&Derived::get_step_size` vs. the base's; if identical, `Derived` does not
hide the hook → clear `static_assert` failure. A hook with a mismatching
signature fails the one-argument call at compile time (acceptable, error
points at the hook call).

## Execution blocks

- [x] **Block 1 — integrators: per-call `h`.**
  - `rk_explicit.hpp`: constructor drops `h`; `h` added to
    `value` / `value_jac` / `jacobian` / `value_jac_hess_prod` /
    `hess_prod` and threaded into `ad_forward`, `forward_jac`,
    `thread_jvp`, `backward_sweep`; fwd cache gains `ws_.last_h`
    (`fwd_hit` compares it). Header banner: "interval of size `h`" →
    `h` is a per-call argument, the caller owns the time grid.
  - `rk_implicit.hpp`: constructor drops `h` (new order:
    `dae, newton_max, newton_tol`); `h` added to all entry points and
    threaded into `newton_single`, `warm_start`, `solve_newton`,
    `substep_jac`, `multi_step_sweep`, `forward_traj`,
    `thread_jvp_traj`, `backward_traj`, `eval_stage_states`,
    `eval_jacobians`, `eval_residual`, `eval_block_jacobian`,
    `x_next_from_cached_K`, `hess_prod_substep`; fwd + traj cache keys
    gain `h`.
  - Done. Note: first full build surfaced two Block-1 oversights in
    `rk_implicit.hpp` — `x_next_from_cached_K` referenced the removed
    `h_ss_` member (now uses `h`, valid since it is NumSteps==1-only) and
    `backward_traj` dropped `h` when calling `hess_prod_substep` (now
    passes it; `traj_hit` guarantees the cached trajectory used the same
    `h`). Both fixed.
- [x] **Block 2 — `ContinuousProblem`: `StepSpec` + CRTP hook.**
  Add `StepSpec` + `ConstantStepSize` (namespace `ocp`); `Derived`
  template argument (4th position, before `NH`); `get_step_size` with
  fallback + guard `static_assert`; the 4 dynamics call sites; header
  docs (opt-in pattern, params-SSOT pointer, "name your class" rule).
  Done.
- [x] **Block 3 — existing call sites (mechanical).**
  Direct integrator calls gain the `h` argument:
  `tests/integrators/rk_explicit_4b.cpp`, `rk_implicit_4d.cpp` (8 direct
  constructions + entry calls), `rk_multistep_4g.cpp`,
  `time_varying_4h.cpp`, `cstr_4i.cpp` (+ its 6 direct integrator
  constructions `k2(ode, 0.1)` → `k2(ode)` etc.),
  `dae_autodiff_4j.cpp`;
  `tests/sqp_pendulum/*.cpp`, `tests/sqp_unicycle/*.cpp`,
  `tests/sqp_masses_chain/*.cpp` (`integ.value(x, u, k*DT, xn)` → add
  `DT`). Problem-level sites (aliases / inheritances / direct
  `ContinuousProblem` instantiations, generated examples, tutorial,
  `sqp_continuous_4e.cpp`) are **unchanged** — verified by compilation.
  Done: full build warning-free, all 35 test binaries pass.
- [x] **Block 4 — new test** `tests/integrators/step_grid_4m.cpp`:
  - Time-varying hand-written ODE (with `jacobian` + `hess_prod`) through
    `ExplicitRkIntegrator` K4 `NumSteps = 2`, and through
    `ImplicitRkIntegrator` Radau IIA 2 via `OdeToDae`.
  - A problem whose grid comes from a `params()` struct (SSOT demo,
    inspect leaves), non-uniform `h_k`, `t0 != 0` (prefix-sum `t_k`).
  - Checks: (a) `dynamics_next_state` == reference that chains standalone
    integrator calls with the same `(t_k, h_k)`; (b)
    `dynamics_value_jac` vs. finite differences of the per-stage composed
    map; (c) `dynamics_value_jac_hess` vs. FD (ERK hand-written path);
    (d) IRK value/jac vs. the same references; (e) fallback regression:
    problem without the hook == constant-`h` reference; (f) constant
    hook override (`{h0, k*h0}`) == fallback; (g) mutate the grid leaf by
    name via `inspect::set_*` and confirm the dynamics reflect the live
    params object.
  - CMake: register alongside the existing integrator test targets
    (follow the `CMakeLists.txt` integrators section pattern).
  - Done: registered in `integrators_unit` (CMake) and
    `integrators_main.cpp`; all 4m checks pass (ERK value 7.3e-8, jac
    1.4e-10, hess 2.6e-6 vs FD/RK4 refs; IRK value 4.9e-5, jac 6.2e-11;
    fallback/const-hook/live-mutation regression checks). Note: the
    reference loops in the test must advance `t` across sub-steps
    (time-varying ODE); the value tolerances carry ~10x headroom over
    the integrator truncation error.
- [x] **Block 5 — docs.**
  - Tutorial ch. 06: new section "per-stage step size" (opt-in CRTP +
    params-SSOT example + MEX note).
  - `docs/ARCHITECTURE.md`: discretization grid is problem-level data
    (params SSOT); integrators are pure functions of `(t_k, h)`.
  - `README.md`: only if the integrator bullet claims a fixed `h`.
  - `docs/plans/TODO.md`: close the item, point here.
  - Done: tutorial ch. 06 gained §6.10 "Per-stage step size" (opt-in CRTP
    + params-SSOT example, hook rules, MEX note) and §6.1/§6.3 were updated
    (new `Derived` template argument, per-call `h`, cache key `(x, u,
    t_k, h)`); `ARCHITECTURE.md` gained a design-principle bullet; TODO.md
    item B closed. README unchanged (its integrator bullet never claimed a
    fixed `h`).
- [x] **Block 6 — full build + full suite.**
  `-Wall -Wextra -Werror` clean; all test binaries pass.
  - Done: full Release build warning-free; all 35 test binaries pass.

## Non-goals / unchanged

- `problem.hpp` discrete contract (no time/`h` concept leaks to solvers;
  the QP sees dynamics Jacobians with `h` already baked in).
- ODE/DAE model contracts and the `OdeToDae` / `AutoDiffDae` wrappers.
- `tools/acados2ocp_pp.py` and generated examples (acados has a single
  global `Ts`; emitted problems keep constant `h` via the fallback).
- MEX (the grid is exposed by name through the existing `param_*`
  commands whenever the user stores it in `params()`).
- Solvers / QP layer.

# AD HVP opt-out flag for ExplicitRkIntegrator

TODO item: "can we deactivate hess_prod in the AD path of
ExplicitRkIntegrator?"

## Problem

For a dual-evaluable ODE (scalar-templated `f`), the explicit integrator
takes the nested-dual (`Dual2 = dual<Dual1, 1>`) HVP path. The `Dual2`
type instantiates a second level of multidual codegen through every ODE
RHS call in the RK stage recursion. When the dynamics Hessian is not
needed (Gauss-Newton model), that codegen cost is pure compile-time
overhead with no runtime use.

## Semantics

A dual-evaluable ODE may define

    static constexpr bool ad_hess_prod = false;

to opt out of the nested-dual HVP pass at compile time. The flag is
detected by the new `ocp::ode_ad_hess_prod_enabled<Ode>` trait
(`ode_model.hpp`; SFINAE probe, defaults to `true` when the member is
absent).

Effects when the flag is `false`:

- `ExplicitRkIntegrator::supports_value_jac_hess_prod` is `false`, so
  `ContinuousProblem::dynamics_value_jac_hess` takes the legacy
  (value_jac + per-column hess_prod) branch — which then fails the
  `hess_prod` static_assert if ever instantiated.
- `hess_prod` / `value_jac_hess_prod` carry a static_assert that names
  the flag and the remedy.
- The `Dual1` paths (`value`, `value_jac`, `jacobian`) are unaffected.

The flag must be paired with `Dims::has_dynamics_hess = false`: with
`has_dynamics_hess = true` the solver instantiates
`dynamics_value_jac_hess` unconditionally (the `if constexpr (P::
has_dynamics_hess)` gate in acados.hpp is compile-time; the runtime
`compute_hess_dyn` switch sits inside it) and the opt-out ODE hits the
static_assert. With `has_dynamics_hess = false` that method is never
instantiated, and the solver runs the Gauss-Newton dynamics Hessian.

No fallback to a hand-written HVP path is provided: a dual-evaluable ODE
without `ad_hess_prod` support simply has no exact dynamics Hessian in
this framework; the intended model is GN.

## Changes

- `ode_model.hpp`: `ode_ad_hess_prod_probe` SFINAE probe + public
  `ode_ad_hess_prod_enabled<Ode>` trait; contract paragraph in the header
  comment.
- `rk_explicit.hpp`: the `Dual2` gates in
  `supports_value_jac_hess_prod`, `value_jac_hess_prod`, and
  `hess_prod` now also require `ode_ad_hess_prod_enabled<Ode>`; the
  terminal-else static_assert in `hess_prod` names the flag.
- `tests/integrators/dae_autodiff_4j.cpp`: `OdeXsqUAdNoHess` (same
  templated `f` + `ad_hess_prod = false`), `GnDims` (`has_dynamics_hess =
  false` variant of `DoubleIntegratorDims`), trait/capability
  static_asserts, and `check_ad_hess_opt_out()` — Dual1 `value_jac`
  FD-checked on the opted-out ODE, plus a `ContinuousProblem<GnDims,
  OdeXsqUAdNoHess, Erk>` value + Jacobian cross-check against the
  enabled ODE at the same point.

## Tests

`integrators_unit` (phase 4j block): new lines
`4j integ vjac  K4-AD-NoHess` and `4j ad opt-out  K4-NoHess`. Full
regression: 20 test binaries + 3 compare.sh green.

The negative case (flag off + `has_dynamics_hess = true`) is a
compile-time error by design and is not exercised at runtime.

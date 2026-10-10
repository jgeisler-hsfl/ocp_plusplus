# Chapter 6 — Continuous time and integrators

Until now the dynamics were hand-written discrete maps. ocp++ also
supports **continuous-time models**: you define an ODE (or DAE) once and
pick a Runge-Kutta integrator; an adapter turns the combination into an
ordinary discrete-time problem, so everything from chapters 2–5 (cost,
constraints, solver, Hessian models) works unchanged. The companion
example is `examples/tutorial/continuous.cpp` (target
`tutorial_continuous`).

## 6.1 The adapter pattern

`ocp::ContinuousProblem<Dims, Ode, Integ>` (in
`include/ocp/integrators/continuous_problem.hpp`) derives from
`ocp::Problem<Dims>` and provides the dynamics entries by delegating to
the integrator. You only implement what is * yours*: cost, constraints,
initial state. The class template parameters:

- `Dims` — the same tag struct as in chapter 2,
- `Ode` (or `Dae`) — your continuous-time model,
- `Integ` — the integrator (section 6.3),
- `NH` — optional compile-time horizon (chapter 7).

The stage map is `t_k = k * h`, where `h` is the integrator step size
passed to the constructor; a time-varying ODE receives `t_k` in every
evaluation, a time-invariant one simply ignores it.

## 6.2 The ODE contract

The ODE model is *duck-typed*: a plain struct, no base class (like the
`Dims` tag for the discrete problem). The contract
(`include/ocp/integrators/ode_model.hpp`):

```cpp
// right-hand side:  xdot = f(x, u, t)
state_t f(const state_t& x, const control_t& u, double t) const;

// Jacobian blocks df/dx, df/du at (x, u, t)
void jacobian(const state_t& x, const control_t& u, double t,
              dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const;

// bilinear Hessian-vector product of the ODE map (nonlinear ODE only):
//   hv = sum_i w_i * D^2 f_i(x, u, t) * (v_x, v_u)
void hess_prod(const state_t& x, const control_t& u, double t,
               const state_t& w, const state_t& v_x, const control_t& v_u,
               state_t& hv_x, control_t& hv_u) const;
```

All methods are `const`; parameters are user-managed members. `hess_prod`
is only required when the ODE is nonlinear *and* the derivative source
needs it (section 6.5).

## 6.3 The integrators

Schemes are selected by tag types (`include/ocp/integrators/butcher.hpp`),
all transcribed from the acados reference:

| Tag | Scheme | Properties |
|-----|--------|------------|
| `K1Tag` | explicit Euler | 1 stage |
| `K2Tag` | explicit RK2 (Heun) | 2 stages |
| `K3Tag` | explicit RK3 | 3 stages |
| `K4Tag` | classic explicit RK4 | 4 stages |
| `RadauIia2Tag` / `3` / `4` | Radau IIA collocation | implicit, L-stable |
| `GaussLegendre2Tag` / `3` / `4` | Gauss-Legendre collocation | implicit; GL4 is the acados IRK default, no L-stability |

**Explicit integrator** (`ocp::ExplicitRkIntegrator<Dims, Ode, NS, Tag,
NumSteps = 1>`): `NS` is the number of RK stages, `Tag` the scheme.
`NumSteps` subdivides each OCP interval of size `h` into `NumSteps`
sub-steps of size `h/NumSteps` (multi-step mode; the default 1 is the
plain single-step scheme) — a cheap way to improve accuracy without
changing the OCP grid.

**Implicit integrator** (`ocp::ImplicitRkIntegrator<Dims, Dae, NS, Tag,
NumSteps = 1>`): consumes the DAE residual

```cpp
state_t F(const state_t& x, const state_t& xdot,
          const control_t& u, double t) const;      // F = 0

void F_jac(const state_t& x, const state_t& xdot, const control_t& u,
           double t, dyn_df_dx_t& dFdx, dyn_df_dx_t& dFxdot,
           dyn_df_du_t& dFdu) const;

// optional (nonlinear DAE): bilinear HVP of F
void F_hess_prod(...);
```

directly — the form the generated CasADi code provides. An ODE written as
a residual `F = xdot - f` is a special case; the Newton solve is exact for
any (regular) `dF/dxdot`, so genuine implicit DAEs work unchanged. The
adapter `ocp::OdeToDae<Ode, Dims>` wraps a hand-written ODE into this
contract (it also maps `hess_prod` to `F_hess_prod` when present), and
`ocp::AutoDiffDae<Derived, Dims>` (a CRTP base) generates `F_jac` and
`F_hess_prod` from a single scalar-templated residual via forward-mode AD
on the vendored multidual header.

Both integrators are allocation-free (fixed-size scratch, methods
`const`) and cache their forward state keyed on `(x, u, t_k)`, so the
per-column Hessian evaluations of `ContinuousProblem` reuse the forward
pass.

## 6.4 The companion example

`examples/tutorial/continuous.cpp` defines a mass on a nonlinear (Duffing)
spring,

```
    q_dot = v
    v_dot = -q - 0.5*q^3 + F
```

with a quadratic cost and a control box, and solves the *same* problem two
ways:

```cpp
// variant 1: explicit RK4, automatic differentiation path
using Erk4 = ocp::ExplicitRkIntegrator<DuffingDims, DuffingOde, 4,
                                       ocp::K4Tag>;
using ErkBase = ocp::ContinuousProblem<DuffingDims, DuffingOde, Erk4>;
DuffingProblem<ErkBase> erk(ode, 0.05);

// variant 2: implicit RK, Gauss-Legendre 4-stage collocation
using Dae = ocp::OdeToDae<DuffingOde, DuffingDims>;
using IrkGl4 = ocp::ImplicitRkIntegrator<DuffingDims, Dae, 4,
                                         ocp::GaussLegendre4Tag>;
DuffingProblem<IrkBase> irk(dae, 0.05);
```

The problem class is a thin mixin over the base, so the cost, box and
initial state are written once. Compile and run:

```bash
cmake --build build --target tutorial_continuous
./build/tutorial_continuous
```

Expected output:

```
ERK4     status=1 cost=7.655648614 iters=4  q_N=0.356392 v_N=-0.626972
IRK-GL4  status=1 cost=7.655647491 iters=8  q_N=0.356392 v_N=-0.626972
```

Both variants converge (`kSolved`) and agree on the solution to about
1e-6 in cost and trajectory — only the discretization of the dynamics
differs. Nothing downstream of the problem interface changed between the
two runs.

## 6.5 Automatic differentiation

The explicit integrator detects at compile time whether the ODE's `f` is
* dual-evaluable* (a template in the scalar type):

```cpp
template <class D>
Eigen::Matrix<D, 2, 1> f(const Eigen::Matrix<D, 2, 1>& x,
                         const Eigen::Matrix<D, 1, 1>& u, double t) const;
```

If so, it evaluates the RK stage recursion itself in dual scalars:

- **value + Jacobian**: one forward pass in `dual<double, nx + nu>` — the
  value and all first-order partials come out of the same pass;
- **Hessian**: one forward pass in a nested dual type (full gradient basis
  inside, HVP direction outside); the multiplier `w` is contracted only at
  extraction.

Rules to know:

- **AD wins.** A dual-evaluable `f` means the integrator never calls your
  hand-written `jacobian` / `hess_prod` — even if you provide them. The
  hand-written paths are used only when `f` is defined at `double` only.
  (The implicit integrator has no AD path; it always uses the DAE model's
  `F_jac` / `F_hess_prod`, e.g. from `OdeToDae` or `AutoDiffDae`.)
- **Opt-out.** A dual-evaluable ODE may declare
  `static constexpr bool ad_hess_prod = false;` to skip the nested-dual
  HVP codegen entirely (useful with a Gauss-Newton model); pair it with
  `Dims::has_dynamics_hess = false` (chapter 5).

In the example, `DuffingOde::f` is dual-evaluable: ERK4 takes the AD path,
while the IRK variant consumes the hand-written `jacobian` / `hess_prod`
through `OdeToDae` — both feed the same `dynamics_value_jac_hess` contract
of chapter 5.

## 6.6 Choosing a scheme

- **Smooth dynamics, moderate accuracy:** explicit RK (`K2Tag`..`K4Tag`)
  with AD — the cheapest and fastest derivatives.
- **Stiff dynamics, DAEs, or generated CasADi residual code:** implicit
  collocation (`RadauIia*Tag` for L-stability, `GaussLegendre4Tag` for the
  highest order; the acados IRK default).
- **Accuracy without changing the OCP grid:** increase `NumSteps`
  (sub-stepping) or the stage count.
- **Convergence tolerances** should match what the discretization and
  Hessian model can deliver: the example sets 1e-6 (as the generated
  examples do); with the default 1e-8 the IRK run converges linearly and
  stalls just above `tol_stat`, exiting with `kMinStep` — the iterate is
  still the solution (cost agrees with ERK4 to 1e-6), but the exit status
  tells you the residual floor was hit.

## 6.7 Time-varying problems

Every ODE/DAE method receives the physical evaluation time `t` (the
integrator computes the per-stage time from the Butcher abscissae), so a
time-varying ODE needs no extra machinery — just use `t`. A
time-invariant ODE ignores it. See `tests/integrators/time_varying_4h.cpp`
for a complete example.

## What's next

Chapter 7 covers the remaining advanced features: compile-time horizons,
free initial states, QP internals, globalization and the acados JSON
generator.

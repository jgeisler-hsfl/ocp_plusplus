# Chapter 2 — Your first problem

This chapter walks through `examples/tutorial/minimal.cpp` (target
`tutorial_minimal`), a complete, minimal discrete-time OCP with no
constraints. Chapter 3 covers the solver part of the same program.

The example problem is a double integrator:

```
    x = [q; v]   (position, velocity),    u = [a]   (acceleration)

    dynamics:    q_{k+1} = q_k + Ts * v_k
                 v_{k+1} = v_k + Ts * a_k

    stage cost:  0.5 * (wq*q^2 + wv*v^2 + wa*a^2)
    terminal:    wf * (q_N^2 + v_N^2)
```

## 2.1 Step 1: the `Dims` tag struct

Every problem starts with a small tag struct carrying the *structural*
dimensions. They are compile-time constants: the framework derives all
fixed-size Eigen types from them, so a solve allocates nothing.

```cpp
struct MinimalDims
{
    static constexpr int nx = 2;   // [q; v]
    static constexpr int nu = 1;   // [a]

    static constexpr int ng   = 0;  // stage nonlinear inequalities g <= 0
    static constexpr int ne   = 0;  // stage nonlinear equalities   e == 0
    static constexpr int nl   = 0;  // stage linear two-sided rows
    static constexpr int ng_t = 0;  // terminal nonlinear inequalities
    static constexpr int ne_t = 0;  // terminal nonlinear equalities
    static constexpr int nl_t = 0;  // terminal linear two-sided rows

    static constexpr bool fixed_initial_state = true;
    static constexpr bool has_dynamics_hess   = false;
    static constexpr bool has_constr_hess     = false;

    // active box rows (empty here; chapter 4)
    static constexpr std::array<int, 0> state_box_idx   = {};
    static constexpr std::array<int, 0> control_box_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_idx = {};

    // soft constraint rows (all empty here; chapter 4)
    static constexpr std::array<int, 0> ineq_soft_idx   = {};
    // ... (seven more `*_soft_idx` sets, all empty here)
};
```

What each field means:

| Field | Meaning |
|-------|---------|
| `nx`, `nu` | state / control vector sizes (fixed for all stages) |
| `ng`, `ne`, `nl` | stage nonlinear inequality `g <= 0`, nonlinear equality `e == 0`, and linear two-sided constraint row counts |
| `ng_t`, `ne_t`, `nl_t` | the same three groups at the terminal stage (state only) |
| `fixed_initial_state` | `true`: `x_0` is a fixed given via `initial_state()`; `false`: `x_0` is a decision variable (e.g. moving-horizon estimation, chapter 7) |
| `has_dynamics_hess` | `true`: you implement `dynamics_value_jac_hess` and the solver assembles the exact Lagrangian Hessian; `false`: Gauss-Newton QP Hessian (chapter 5) |
| `has_constr_hess` | `true`: the constraint `*_value_jac_hess` methods are implemented; `false`: constraint Hessian terms omitted (valid exactly when the constraint maps are linear, chapter 5) |
| `state_box_idx`, `control_box_idx`, `terminal_state_box_idx` | which state / control components are box-bounded; the array *size* is the number of active box rows (chapter 4) |
| `*_soft_idx` (nine sets) | which constraint rows of each group are *soft* (may be violated by a solver slack; chapter 4) |

Note that **all fields must be present** even if a feature is unused (zero
dimensions, empty index sets): the base class reads them all at compile
time.

## 2.2 Step 2: the concrete problem class

The concrete problem inherits `ocp::Problem<MinimalDims>` and implements
the model functions. The base class provides one stub per interface
function carrying the exact expected signature: if you forget a function,
misspell it, or get the signature wrong, compilation fails at the call site
with a descriptive message.

```cpp
class MinimalProblem : public ocp::Problem<MinimalDims>
{
public:
    // user-managed problem parameters (not part of the interface)
    double Ts_ = 0.1;
    double wq_ = 1.0;
    // ...

    state_t initial_state() const;
    state_t dynamics_next_state(int k, const state_t& x,
                                const control_t& u) const;
    void dynamics_value_jac(int k, const state_t& x, const control_t& u,
                            state_t& x_next, dyn_df_dx_t& df_dx,
                            dyn_df_du_t& df_du) const;
    // ...
};
```

Key rules:

- **Model functions are `const` and pure.** They may read user-managed
  parameter members (like `Ts_` above) but take no parameter argument:
  parameters are deliberately not part of the interface — you keep them in
  your class and change them between solves.
- **Stage index `k`.** Dynamics, stage cost and stage constraints receive
  `k = 0..N-1`; the terminal stage `k = N` is served by the dedicated
  `terminal_*` functions, which depend on the state only.
- **Fused evaluation entries.** Where a solver needs value and derivatives,
  the interface has one fused call (`dynamics_value_jac`,
  `stage_cost_value_grad`, `stage_cost_value_grad_hess`, ...) so everything
  is computed in a single pass.

### Which functions must you implement?

The exact set depends on the `Dims` tag:

| Function | Required when |
|----------|---------------|
| `initial_state()` | `fixed_initial_state = true` (absent from the interface otherwise) |
| `dynamics_next_state`, `dynamics_value_jac` | always (`nx > 0`) |
| `dynamics_value_jac_hess` | `has_dynamics_hess = true` |
| `stage_cost_value`, `_grad`, `_grad_hess` | always |
| `terminal_cost_value`, `_grad`, `_grad_hess` | always |
| `stage_inequality_value_jac(_hess)` | `ng > 0` |
| `stage_equality_value_jac(_hess)` | `ne > 0` |
| `stage_linear_constr` | `nl > 0` |
| `stage_state_box_constr` | any box row set non-empty |
| `stage_control_box_constr` | `nbu > 0` (non-empty `control_box_idx`) |
| `stage_*_constr_soft_penalty` | the group's `*_soft_idx` non-empty |
| `terminal_*` | correspondingly, when the `*_t` dimension is nonzero |

A zero dimension means the solver never calls the corresponding function,
and you can simply omit it (the stub stays unimplemented on purpose) —
this is what the minimal example does for all constraint groups.

### Vector and matrix types

The base class re-exposes convenient type aliases derived from the `Dims`:

```cpp
using state_t   = Eigen::Matrix<double, nx, 1>;      // 2 x 1
using control_t = Eigen::Matrix<double, nu, 1>;      // 1 x 1
using dyn_df_dx_t = Eigen::Matrix<double, nx, nx>;   // df/dx block
using dyn_df_du_t = Eigen::Matrix<double, nx, nu>;   // df/du block
using stage_grad_t = Eigen::Matrix<double, nx + nu, 1>;   // [g_x; g_u]
using stage_hess_t = Eigen::Matrix<double, nx + nu, nx + nu>;
using term_grad_t  = Eigen::Matrix<double, nx, 1>;
using term_hess_t  = Eigen::Matrix<double, nx, nx>;
```

Combined (x, u) quantities always list the **state first**.

## 2.3 The minimal example's model functions

`examples/tutorial/minimal.cpp` implements the nine functions required by
the `MinimalDims` above. The dynamics, with their Jacobian blocks:

```cpp
state_t dynamics_next_state(int, const state_t& x,
                            const control_t& u) const
{
    state_t xn;
    xn(0) = x(0) + Ts_ * x(1);
    xn(1) = x(1) + Ts_ * u(0);
    return xn;
}

void dynamics_value_jac(int, const state_t& x, const control_t& u,
                        state_t& x_next,
                        dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
{
    x_next = dynamics_next_state(0, x, u);
    df_dx(0, 0) = 1.0;  df_dx(0, 1) = Ts_;
    df_dx(1, 1) = 1.0;
    df_du(1, 0) = Ts_;
}
```

The stage cost Hessian is the constant diagonal of the weights — an
*exact* Hessian here. Note that your `stage_cost_value_grad_hess` provides
the **cost** Hessian only; the solver adds the dynamics and constraint
Hessian terms itself (chapter 5). Since `has_dynamics_hess = false`, the
solver also never calls `dynamics_value_jac_hess`, and the QP Hessian is
assembled from your cost Hessian plus the linearized dynamics — for this
linear problem both Hessian models coincide.

## 2.4 Step 3: verify your derivatives

Before trusting a solver, check the hand-written derivatives against
central finite differences — the same pattern the bundled examples use:

```cpp
// gradient: perturb one variable at a time, compare with the fused call
const double fd = (problem.stage_cost_value(k, xp, up) -
                   problem.stage_cost_value(k, xm, um)) / (2.0 * h);
check_close("stage grad", g(i), fd, 1e-4);

// dynamics Jacobian: perturb x (or u), compare each block entry
const double fd = (fp(i) - fm(i)) / (2.0 * h);
check_close("dyn df/dx", A(i, j), fd, 1e-4);
```

## 2.5 Compile and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tutorial_minimal
./build/tutorial_minimal
```

Expected output (derivative part):

```
== derivative checks (chapter 2) ==
ok   stage grad: 0.7
ok   stage grad: -0.02
ok   stage grad: 0.003
ok   dyn df/dx: 1
ok   dyn df/dx: 0.1
...
```

All rows should read `ok`; the program exits with status 1 if any check
failed.

## What's next

Chapter 3 attaches the SQP solver to this problem and explains the
solution, options, statistics and multipliers.

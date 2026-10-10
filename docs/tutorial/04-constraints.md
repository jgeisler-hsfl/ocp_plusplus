# Chapter 4 — Constraints

This chapter covers every constraint type ocp++ supports: box, linear,
nonlinear (in)equality, at the stage and the terminal level, and soft
constraints. The worked example is the `double_integrator` problem
(`examples/double_integrator/double_integrator.hpp`), which uses almost all
of them at once.

Recall the general form: nonlinear inequalities are written `g(x, u) <= 0`,
nonlinear equalities `e(x, u) == 0`, and box/linear constraints are
two-sided `lo <= ... <= hi`. Which constraints exist is fixed at compile
time by the `Dims` tag; the *values* (bounds, matrices) are returned per
stage by the problem at solve time.

## 4.1 Overview: groups, dimensions, index sets

| Group | `Dims` field | Interface function(s) | Stages |
|-------|--------------|------------------------|--------|
| state box | `state_box_idx` (size = `nbx`) | `stage_state_box_constr(k)` | `0..N-1` |
| control box | `control_box_idx` (size = `nbu`) | `stage_control_box_constr(k)` | `0..N-1` |
| terminal state box | `terminal_state_box_idx` (size = `nbx_t`) | `terminal_state_box_constr()` | `N` |
| linear stage | `nl` | `stage_linear_constr(k)` | `0..N-1` |
| linear terminal | `nl_t` | `terminal_linear_constr()` | `N` |
| nonlinear ineq stage | `ng` | `stage_inequality_value_jac(_hess)` | `0..N-1` |
| nonlinear eq stage | `ne` | `stage_equality_value_jac(_hess)` | `0..N-1` |
| nonlinear ineq terminal | `ng_t` | `terminal_inequality_value_jac(_hess)` | `N` |
| nonlinear eq terminal | `ne_t` | `terminal_equality_value_jac(_hess)` | `N` |

Every group has a *per-group soft-row index set* (`ineq_soft_idx`,
`lin_soft_idx`, `state_box_soft_idx`, ..., `terminal_state_box_soft_idx`)
selecting which rows are soft — see section 4.6.

A zero dimension means the group does not exist: the solver never calls
the corresponding function, and you omit it from your class.

## 4.2 Box constraints and the active-row concept

Box constraints bound selected *components* of the state and control. The
index sets in `Dims` select which:

```cpp
static constexpr std::array<int, 2> state_box_idx   = {0, 1};  // q and v
static constexpr std::array<int, 1> control_box_idx = {0};     // a
static constexpr std::array<int, 2> terminal_state_box_idx = {0, 1};
```

The array size **is** the number of active box rows (`nbx`, `nbu`,
`nbx_t`): bounds are stored per active row only, in the order of the index
set. The stage functions return a `BoxSpec`:

```cpp
state_box_t stage_state_box_constr(int) const
{
    state_box_t spec;
    spec.lo << -10.0, -10.0;   // bounds of state_box_idx[0], state_box_idx[1]
    spec.hi <<  10.0,  10.0;
    spec.penalty = {};         // soft-constraint weights (section 4.6)
    return spec;
}
```

`BoxSpec` carries `lo`, `hi` (one entry per active row) and a
`SoftPenalty` per row. The spec is stage-invariant here, but the problem is
free to return different bounds per stage.

**Fixed initial state.** When `Dims::fixed_initial_state` is true, `x_0`
is not a decision variable: the solver pins it to `initial_state()` with
internal *pin rows* instead of box rows, so no stage-0 state box is needed
to hold `x_0` (a stage-0 state box would have no effect). When
`fixed_initial_state` is false (chapter 7), the stage-0 state box bounds
`x_0` like any other stage.

## 4.3 Linear constraints

Linear two-sided stage constraints are `lo <= A x + B u <= hi` at each
stage; terminal ones are `lo <= A x <= hi` (no control). An additive
constant, if any, is absorbed into the bounds.

```cpp
stage_linear_t stage_linear_constr(int) const
{
    stage_linear_t spec;
    spec.A(0, 0) = 0.5;  spec.A(0, 1) = 1.0;   // 0.5*q + v
    spec.B.setZero();
    spec.bounds.lo(0) = -3.0;
    spec.bounds.hi(0) =  3.0;
    return spec;
}
```

`LinearSpec<NRow, Nx, Nu>` carries `A`, `B` and a `BoxSpec` of bounds;
`TerminalLinearSpec<NRow, Nx>` the same without `B`. Linear rows are
handled exactly (no linearization needed), and their multipliers are
*signed*: positive means the upper bound is active, negative the lower.

## 4.4 Nonlinear (in)equality constraints

Nonlinear inequalities are `g(x, u) <= 0` (one-sided), equalities
`e(x, u) == 0`. You provide fused value + Jacobian calls (and, when
`has_constr_hess = true`, a multiplier-contracted Hessian variant, chapter
5):

```cpp
// value + Jacobian in one pass; g <= 0
void stage_inequality_value_jac(int k, const state_t& x, const control_t& u,
                                ineq_t& g, ineq_dg_dx_t& g_dx,
                                ineq_dg_du_t& g_du) const;

// value + Jacobian in one pass; e == 0
void stage_equality_value_jac(int k, const state_t& x, const control_t& u,
                              eq_t& e, eq_de_dx_t& e_dx,
                              eq_de_du_t& e_du) const;
```

The gradient blocks use the `[g_x; g_u]` order, matching the gradient
layout. The terminal variants take the state only and have no `du` blocks.

A subtlety worth knowing: the QP treats equalities as *two-sided* rows
(both sides may carry slacks if the row is soft); inequality rows are
one-sided (`hi = 0`). The solver reports a *signed* multiplier for
equality and linear rows and a nonnegative multiplier for inequality rows.

## 4.5 The worked example: the double integrator

`examples/double_integrator/double_integrator.hpp` combines almost
everything:

```
    box:         -10 <= q <= 10,  -10 <= v <= 10   (all stages incl. terminal)
                 -1  <= a <= 1                     (stages 0..N-1)
    inequality:  v - v_max <= 0                    (stages 0..N-1, soft row)
    linear:      -3 <= 0.5*q + v <= 3              (stages 0..N-1)
    terminal:    q_N + v_N = 1                     (nonlinear equality, ne_t = 1)
    initial:     x_0 = [1; 0.5]
```

Its `Dims` sets every group: `ng = 1, nl = 1, ne_t = 1`,
`state_box_idx = {0, 1}`, `control_box_idx = {0}`,
`terminal_state_box_idx = {0, 1}`; and leaves out what does not exist
(`ne = ng_t = nl_t = 0`): the corresponding interface functions are simply
absent from the class.

Compile and run the full SQP on this problem (the test embeds a verified
warm start and independently checks the KKT conditions):

```bash
cmake --build build --target sqp_double_integrator
./build/sqp_double_integrator
```

It prints the SQP iterations (`print_level = 1`), writes
`sqp_double_integrator.csv`, and verifies at the solution: dynamics gap,
constraint violations, complementarity against the returned multipliers,
and stationarity via a finite-difference Lagrangian gradient. Exit 0 means
all checks passed.

Reading the example's problem definition side by side with the
`Dims`-to-function table of section 4.1 is the fastest way to see how a
full problem maps onto the interface.

## 4.6 Soft constraints

A *soft* row may be violated at the price of a slack variable. Which rows
are soft is a **compile-time decision** made by the per-group index sets
in `Dims`:

```cpp
static constexpr std::array<int, 1> ineq_soft_idx   = {0};  // ineq row 0 is soft
static constexpr std::array<int, 0> eq_soft_idx     = {};   // everything else hard
static constexpr std::array<int, 0> lin_soft_idx    = {};
// ... likewise terminal_* and state/control_box_soft_idx
```

The set size is the number of soft rows in that group; an empty set means
the whole group is hard.

Each soft row `i` gets a penalty weight returned by the problem:

- **Nonlinear groups** report it via a method:
  `stage_inequality_constr_soft_penalty(k)` (returning a `SoftPenalty`
  with one `{quad, lin}` pair per soft row, in `*_soft_idx` order).
- **Box and linear groups** carry it in the returned spec's `penalty`
  field (one weight per row).

The solver adds to the objective, per soft *side*,

```
    0.5 * quad * s^2  +  lin * s        (s >= 0)
```

with `quad` the Hessian (second-order) weight and `lin` the gradient
(first-order) weight. The standard soft constraint is `quad = w`,
`lin = 0` — there is a constructor shortcut
`SoftPenalty(w)` for exactly that. This mirrors acados' per-slack weights
`Z` (quadratic) and `z` (linear).

Sides per group:

| Group | Sides | Slack semantics |
|-------|-------|-----------------|
| nonlinear inequality | upper only | relaxes `g <= 0` to `g <= s` |
| nonlinear equality | lower and upper | row may lie in `[-s_lo, s_hi]` |
| box | lower and upper | bound relaxed outward on each side |
| linear | lower and upper | same as box |

The slack variables themselves are solver-internal: they are *not* part of
`ocp::Solution` (only the multipliers and the penalized objective are).
`sol.cost_value` includes the slack penalty, so it is the objective the
solver actually minimizes.

In the double-integrator example the velocity cap `v - v_max <= 0` is the
only soft row (`ineq_soft_idx = {0}`, weight `quad = 100`): violating it
costs `0.5 * 100 * s^2`, so the solver prefers to slow down rather than
violate — but a violent maneuver that cannot respect the cap within the
horizon pays the slack instead of failing.

To see a soft row in action, raise `v_max_` above what the maneuver
requires and watch the velocity hit `v_max_` exactly (multiplier positive,
slack zero); then lower it below the attainable range and watch the slack
take over.

## 4.7 Soft constraints and feasibility

Two practical notes:

- Soft constraints are the standard cure for *infeasible* warm starts and
  slightly inconsistent constraint sets: making a few rows soft lets the
  SQP start from an infeasible iterate without a phase-1 procedure.
- The generator (chapter 7) maps acados soft constraints one-to-one for
  nonlinear rows; box/linear rows require `Zl == Zu` there because ocp++
  carries one weight per row.

## What's next

Chapter 5 looks at the second-order information the solver consumes: exact
Hessians, Gauss-Newton models and Levenberg-Marquardt damping.

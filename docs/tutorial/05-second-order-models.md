# Chapter 5 — Second-order models

The SQP subproblem needs a Hessian of the Lagrangian. This chapter
explains the two Hessian models ocp++ supports, the compile-time and
runtime switches that select between them, the Hessian-vector-product
contract you implement for nonlinear dynamics and constraints, and the
Levenberg-Marquardt damping options.

## 5.1 Two models: exact vs Gauss-Newton

The QP Hessian the solver assembles at each stage is

```
    H = H_cost  +  H_dynamics  +  H_constraints
```

where `H_cost` is *your* `stage_cost_value_grad_hess` Hessian, and the
dynamics and constraint terms are multiplier-contracted second derivatives
of the corresponding maps. Two models are supported:

| Model | `Dims` flags | What the solver does | acados equivalent |
|-------|--------------|----------------------|-------------------|
| **Exact Lagrangian** | `has_dynamics_hess = true` | adds the multiplier-weighted dynamics Hessian (via your `dynamics_value_jac_hess`) | `hessian_approx = EXACT` |
| **Gauss-Newton** | `has_dynamics_hess = false` | dynamics enter the QP only through the linearized dynamics matrix (`BA`); the second-order term is dropped | `hessian_approx = GAUSS_NEWTON` |

The two coincide exactly when the dynamics map is linear (zero dynamics
Hessian). Constraint Hessians are controlled independently by
`has_constr_hess`: `false` omits the constraint Hessian terms, which is
exact whenever the constraint maps are linear (boxes, linear rows), and a
common choice for nonlinear rows as well.

**When to choose which:**

- *Linear or mildly nonlinear dynamics, tracking-type costs:*
  Gauss-Newton (`has_dynamics_hess = false`). Cheaper, robust, and for
  least-squares costs it is the natural model. All generated examples use
  it.
- *Strongly nonlinear dynamics, exact curvature matters, fast local
  convergence wanted:* Exact (`has_dynamics_hess = true`) with your own
  Hessian-vector products.

## 5.2 What *you* provide: the cost Hessian

Your `stage_cost_value_grad_hess` / `terminal_cost_value_grad_hess`
provide the **cost** Hessian only — the solver adds the dynamics and
constraint parts itself. The interface does not distinguish what kind of
Hessian you return:

```cpp
// exact Hessian of 0.5*(wq*q^2 + wv*v^2 + wa*a^2): the constant diagonal
H(0, 0) = wq_;
H(1, 1) = wv_;
H(2, 2) = wa_;
```

For a nonlinear least-squares cost `L = 0.5 ||r(x, u)||^2` you may return
the Gauss-Newton approximation `J_r^T J_r` instead of the exact Hessian
`J_r^T J_r + sum_i r_i D^2 r_i` — that is precisely what the framework's
Gauss-Newton model does for the dynamics part, and a perfectly valid
choice for the cost part.

The Hessian uses the block layout `[H_xx H_xu; H_ux H_uu]` (state first).

## 5.3 The Hessian-vector-product contract

When `has_dynamics_hess = true`, you implement the fused entry

```cpp
void dynamics_value_jac_hess(int k, const state_t& x, const control_t& u,
                             const state_t& lam,
                             state_t& x_next, dyn_df_dx_t& df_dx,
                             dyn_df_du_t& df_du,
                             dyn_hess_t& hess) const;
```

with the multiplier-contracted semantics

```
    hess  =  sum_i lam_i * D^2 f_i(x, u)
```

where `lam` is the dynamics (costate) multiplier the solver passes (at
stage `k` it is `sol.lambda_dyn[k]`), and the `[x; u]` layout applies.
The solver permutes the result into its internal `(u; x)` ordering and
adds it to the QP Hessian with the sign the Lagrangian requires — you
always return `sum_i lam_i D^2 f_i`, nothing else.

The same contract applies to nonlinear constraints
(`stage_inequality_value_jac_hess`, `stage_equality_value_jac_hess` and
the terminal variants): the contracted Hessian of the group's map with the
group's multiplier, `[x; u]` layout (terminal: state only).

Returning the Hessian *matrix* (not just HVPs) is the contract at the
problem level; chapter 6 shows how the continuous-time machinery builds it
from Hessian-vector products internally.

## 5.4 Runtime gates

The two `Dims` flags are compile-time *capabilities*. Two independent
runtime options drop the corresponding terms without recompiling
(mirroring acados' `exact_hess_dyn` / `exact_hess_constr`):

```cpp
ocp::AcadosOptions opts;
opts.compute_hess_dyn    = false;  // drop the dynamics Hessian term (GN)
opts.compute_hess_constr = false;  // drop the constraint Hessian terms
```

A term is used only if both the capability flag and the runtime gate are
true. This is handy for experiments: solve once exact, once
Gauss-Newton, and compare iterations and cost.

## 5.5 Levenberg-Marquardt damping

Nonconvex problems can produce indefinite QP Hessians, which slows the
line search or stalls convergence. The solver can add a damping term
`mu * I` to the `(u; x)` Hessian block of every stage:

**Fixed damping** — set a constant:

```cpp
opts.levenberg_marquardt = 1e-2;
```

**Adaptive damping** — a schedule ported from acados: `mu` starts at
`adaptive_lm_mu0`, grows by `adaptive_lm_lam` whenever a step is truncated
(`alpha < 1`) and shrinks by `1/adaptive_lm_lam` after full steps, and the
applied damping is `adaptive_lm_obj_scalar * cost * mu`:

```cpp
opts.with_adaptive_lm        = true;
opts.adaptive_lm_mu0         = 1e-3;   // initial mu
opts.adaptive_lm_lam         = 5.0;    // growth / shrink factor
opts.adaptive_lm_mu_min      = 1e-16;  // floor for the full-step decay
opts.adaptive_lm_obj_scalar  = 2.0;    // mu_eff = scalar * raw_cost * mu
```

Use damping on nonconvex problems (e.g. bang-bang or chaotic dynamics) or
as a fallback when the solver stalls (`kMinStep`); the damping is a no-op
when both `compute_hess_dyn` and `compute_hess_constr` are false or
`mu <= 0`.

## 5.6 Summary of the switches

| Switch | Level | Effect |
|--------|-------|--------|
| `Dims::has_dynamics_hess` | compile time | `true`: `dynamics_value_jac_hess` implemented and used (EXACT); `false`: Gauss-Newton |
| `Dims::has_constr_hess` | compile time | `true`: constraint `*_value_jac_hess` implemented and used; `false`: constraint Hessians omitted |
| `opts.compute_hess_dyn` | runtime | drops the dynamics Hessian term (Gauss-Newton for the dynamics) |
| `opts.compute_hess_constr` | runtime | drops the constraint Hessian terms |
| `opts.levenberg_marquardt` | runtime | constant `mu * I` damping |
| `opts.with_adaptive_lm` (+ params) | runtime | adaptive `mu` schedule |

## What's next

Chapter 6 moves from hand-written discrete dynamics to continuous-time
models: ODE and DAE contracts, the Runge-Kutta integrators and their
automatic differentiation.

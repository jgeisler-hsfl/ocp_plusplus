# Chapter 1 — Getting started

## 1.1 What is an optimal control problem?

A discrete-time optimal control problem (OCP) asks for a state and control
trajectory that minimizes a cost while satisfying the dynamics and
constraints:

```
    min_{x, u}   sum_{k=0}^{N-1} L_k(x_k, u_k)  +  L_N(x_N)
    subject to   x_{k+1} = f_k(x_k, u_k),       k = 0..N-1
                 box, linear and general (in)equality constraints
```

Here `N` is the *horizon*, `x_k` the states (`k = 0..N`, so `N + 1` of them)
and `u_k` the controls (`k = 0..N-1`, one per dynamics step). `L_k` is the
stage cost, `L_N` the terminal cost, and `f_k` the discrete-time dynamics
map.

## 1.2 What ocp++ provides

ocp++ is a C++17, header-only framework with a strict separation of two
concerns:

1. **The problem** — dynamics, cost, constraints and their derivatives. This
   is *your* code: a concrete C++ class implementing a fixed template
   interface (`ocp::Problem<Dims>`, see chapter 2).
2. **The solver** — a template over the concrete problem type. The reference
   implementation is the acados-style SQP driver (`ocp::AcadosSolver`)
   around an HPIPM-style interior-point QP solver
   (`ocp::HpipmQpSolver`).

Design principles you will meet throughout:

- **Static polymorphism.** No virtual dispatch, no exceptions in the
  numerical core. A missing model function fails at compile time with the
  exact expected signature.
- **Compile-time structure.** All structural dimensions (`nx`, `nu`,
  constraint row counts, which rows are bounded or soft) are compile-time
  constants, so all vectors and matrices are fixed-size Eigen objects and
  nothing is allocated during a solve.
- **Runtime (or compile-time) horizon.** The same problem type can be
  solved with any horizon length; a compile-time horizon is also possible
  (chapter 7).
- **Allocation-free solves.** Workspaces are sized once per solver object;
  `solve()` allocates nothing (in dynamic-horizon mode it (re)sizes once).

## 1.3 Repository layout

```
include/ocp/problem.hpp        the problem interface (central contract)
include/ocp/solvers/           solver implementations
    hpipm/                     HPIPM-style QP solver + staged QP data model
    acados/                    SQP driver (line search, regularization, merit)
include/ocp/integrators/       continuous-time support (ODE/DAE + RK integrators)
examples/                      hand-written and generated example problems
tests/                         unit tests (solvers, integrators, comparisons)
tools/acados2ocp_pp.py         generator: acados OCP JSON -> ocp++ problem class
docs/                          architecture, plans, this tutorial
```

## 1.4 Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/double_integrator
```

The `double_integrator` example is an interface sanity test: it checks the
compile-time dimensions, forward-simulates the dynamics, verifies gradients
and Jacobians against central finite differences, and evaluates all
constraint groups. It prints one `ok`/`FAIL` line per check and exits with
status 0 on success.

Throughout the tutorial, `Status::kSolved` (printed as `0`) means the solver
converged to an (approximate) KKT point.

## 1.5 Notation used in this tutorial

- **Stage index `k`:** `0..N-1` for dynamics, stage cost and stage
  constraints; the terminal stage `k = N` is served by dedicated
  `terminal_*` functions (there is no `u_N`).
- **Extents:** state trajectory `N + 1` stages, control trajectory `N`,
  stage constraints `N`, terminal constraints `1`.
- **Combined (x, u) layout:** whenever a gradient or Hessian mixes states
  and controls, the state comes first: gradients are `[g_x; g_u]`, Hessians
  the block layout `[H_xx H_xu; H_ux H_uu]`.
- **Sign conventions:** nonlinear inequality constraints are written
  `g(x, u) <= 0`, equalities `e(x, u) == 0`; linear and box constraints are
  two-sided `lo <= ... <= hi`.

## What's next

Chapter 2 walks you through writing your first problem from scratch.

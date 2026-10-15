# Architecture

## 1. Purpose and Scope

This project is a C++17 framework for developing and evaluating numerical algorithms for optimal control problems (OCPs).

The primary goal is **not** to implement a single OCP solver, but to provide a stable mathematical problem interface that can be used by many different optimization algorithms and research implementations.

The architecture therefore separates:

1. the mathematical problem definition and
2. the solver algorithm (different solver algorithms shall consume the same problem definition).

---

# 2. Design Principles

The architecture follows these principles:

* **Static polymorphism:** the numerical core shall primarily use C++17 templates rather than virtual dispatch. The exact C++ mechanism may use traits, compile-time detection, CRTP, C++17 SFINAE, or a combination thereof. The final mechanism should be chosen for clarity and compile-time guarantees rather than abstraction for its own sake.
* **Compile-time structure:** intrinsic problem dimensions must be available at compile time to allow for static allocation of all problem und solution variables and optimized matrix operations.
* **Compact problem definition:** the optimization problem, constisting of the model dynamics, cost function (stage and terminal), constraints (stage and terminal) and their derivatives, and the dimensions of all relevant vectors is encapsuled in a class (possibly using sub-classes or structure, e.g. for dimensions, dynamics, cost and constraints) that adheres to a defined "template-interface". Concrete implementations of a specific Problem are strictly user implemented.
* **Discrete dynamics first:** at first only discrete dynamic systems are considered. It is up to the user to implement discretisation schemes and derivatives.
* **Efficient linear algebra** using the eigen3 library. All vectors and matrices are fixed-size eigen3 obejcts.
* **Runtime flexibility:** quantities such as the horizon may be runtime-variable.
* **Representation independence:** mathematical objects such as trajectories should not be defined by their underlying storage format.
* **Minimal abstraction:** abstractions should represent meaningful mathematical or architectural concepts rather than anticipated future requirements.
* **Separation of parameters:** parameters of the model and problem are managed by the user supplied class. Parameters and options of the solver algorithm are managed individually by each solver implementation, sharing only a generaly high-level interface that allows to define arbitry option variables and access them as member variables of a solver-specific options class or by a string name via a map.
* **Separation of internal variables:** each solver implementation stores its own internal states and intermediate variables (like e.g. multipliers, barrier parameters, regularization parameters, or solver-internal slack variables). Only solution variables common to all solvers (especially the trajectories of states and inputs) are accessible via a shared interface.

---

# 3. Scalar Type

Numerical types should preferably be template parameters where practical.

Conceptually:

```cpp
template<class Scalar>
class Problem;
```

This allows the framework to support different scalar types without making `double` an intrinsic requirement of the mathematical model.

---

# 4. Static and Runtime Dimensions and Data Types

The architecture distinguishes between **structural dimensions** and **instance dimensions**.

Structural dimensions such as `nx`, `nu`, and constraint dimensions normally describe the mathematical type and therefore belong at compile time.

The horizon is different. The same mathematical problem type may naturally be used with different horizon lengths:

```text
N = 50
N = 100
N = 200
```

Therefore the architecture shall support a runtime horizon.

At the same time, compile-time horizons should remain possible for applications where they are beneficial.

The architecture must therefore accommodate both:

```text
fixed structural dimensions + fixed horizon
```

and

```text
fixed structural dimensions + runtime horizon
```

without making either implementation mandatory at this stage.

---

# 5. Horizon Semantics and Trajectory Representation

The temporal extent of a quantity must be explicit.

For a standard discrete-time OCP,

$$
x_0,\ldots,x_N
$$

and

$$
u_0,\ldots,u_{N-1}
$$

have different extents.

Typical quantities therefore have:

```text
state trajectory       N + 1
control trajectory     N
dynamics residual      N
path constraints       N
terminal constraints   1
```

The framework shall not rely on an implicit convention that every horizon-dependent object has the same length.

The framework shall provide a semantic representation for horizon-dependent quantities.

The mathematical interface should conceptually expose objects such as:

```cpp
Trajectory<State, N>
Trajectory<Control, N>
```

rather than defining trajectories directly as Eigen matrices.

A trajectory represents a sequence of stage-wise quantities. Its physical storage is an implementation detail.

Horizon-dependent quantities internal to a solver (per-stage workspace data such as factors, steps and multipliers) shall use the same trajectory representation, and shall support both a runtime horizon and a compile-time fixed horizon selected by a template parameter (as in `HpipmQpSolver<P, NH>`).

Stage-wise access should be natural, for example:

```cpp
x[k]
u[k]
constraint[k]
slack[k]
```

or through an equivalent interface.

---

# 6. Objective

The architecture must allow alternative formulations such as:

* exact Hessians,
* Gauss-Newton approximations,
* other valid approximations,
* or absence of second-order information for algorithms that do not require it.

The derivative source is not part of the mathematical interface.

---

# 7. Constraints

The problem interface shall represent constraints independently of their numerical treatment.

Supported categories shall include, where applicable:

* Box constraints
* Linear in states and inputs constraints
* General equality constraints
* General inequality constraints
* Individually defined slacks (soft-constraints) on all types of constraints

The interface must remain sufficiently general to accommodate additional formulations without requiring changes to the solver architecture.

# 8. Solver-internal formulation

A solver may solve its subproblems in a formulation that differs from the
stage-wise layout of the problem definition; such a choice is solver-internal
(see "Separation of internal variables") and is selected by a solver-specific
option, not by the problem interface. The reference SQP solver exposes three QP
solve modes via `AcadosOptions.qp_mode` (`QpMode::kStaged`, the default,
`QpMode::kFullCondensed`, and `QpMode::kPartialCondensed`): the staged HPIPM
QP over `(x, u, slacks)` with dynamics multipliers; a single dense QP over
the decision vector `v = [x_0 .. x_N, u_0 .. u_{N-1}]` obtained by
eliminating the dynamics via full condensation; or — partial condensing —
the N stages grouped into `AcadosOptions.qp_cond_N` blocks of NB stages
each (NB a compile-time parameter of the solver), each block condensed to a
mini QP whose internal states are eliminated, and the resulting
block-stage QP solved with the staged HPIPM IPM. All modes consume the same
staged QP assembly and produce identical `ocp::Solution` quantities.

A second, non-iterative driver, `RtiSolver` (`solvers/acados/rti.hpp`),
implements acados' standard real-time-iteration SQP (`ocp_nlp_sqp_rti`,
`as_rti_level = STANDARD_RTI`): one `solve()` call assembles the LHS
(Hessian + Jacobians, fixed Levenberg–Marquardt shift) and the RHS
(gradients + offsets), solves the QP once in any `QpMode`, and applies the
full primal step (alpha = 1) with a full-step dual update — no line search,
no convergence test. The shared LHS/RHS stage assembly lives in
`solvers/acados/assemble.hpp` (used by both `AcadosSolver` and
`RtiSolver`); cost and slack-penalty helpers are shared via `cost.hpp`.

The staged HPIPM QP solver also exposes a QP-level sensitivity layer
(`HpipmQpSolver::sens_forward` / `sens_adj`, seeds in
`solvers/hpipm/seed.hpp`): given a converged solution and a `QpSeed` of
right-hand-side perturbations (grad / b / d / m per stage), it returns the
linearized response of the full QP solution (u, x, pi, lam, t) to that seed,
mirroring HPIPM's `OCP_QP_IPM_SENS_FRW` / `SENS_ADJ` (seed module,
`x_ocp_qp_seed.c`). The first call lazily re-factors the KKT system at the
final iterate (HPIPM `update_fact_exit`); subsequent seeds reuse the cached
P-chain / L factors and only recompute the constraint-force term (gamma) and
run the forward/backward sweeps, so repeated seeds cost one O(N) pass.
Hessian perturbations are expressed as the equivalent grad seed
`grad = E z*`; the returned response carries the KKT's internal negative
sign, i.e. dz/dH = -K^{-1} (E z*).

The dense (full-condensing) QP solver has an equivalent sensitivity layer
(`HpipmDenseQpSolver::sens_forward` / `sens_adj`, seeds in
`DenseQpSeed`, `solvers/hpipm/seed.hpp`), mirroring HPIPM's
`DENSE_QP_IPM_SENS_FRW` / `SENS_ADJ`. Because the dense barrier LHS
`W = H + A'ΓA` is ill-conditioned at the converged iterate (Γ = λ/t ~ 1e9..1e18
on every pinned side, cond ~ 1e15), the sens solve uses a **hybrid block
KKT**: `split_sides` classifies each present side as *hard* (pinned:
`t < 1e-9` or `Γ > 1e6`) or *moderate* (finite Γ). The hard sides become
exact rows of the KKT (Eigen LDLT, cached in the workspace keyed by the
side split); a `lo == hi` row (pinned box / hard-equality) has both sides
hard with `A_hi = -A_lo`, so its lo side is deduplicated away (kept hi
side only, as in the staged hard-eq KKT) to keep the KKT full rank. The
moderate sides stay in the barrier term
`W_mod = H + reg·I + A_mod'Γ_mod A_mod`, keeping W well-conditioned.
`sens.v` is the KKT primal; moderate sides get the closed-form `dt`/`dlam`;
hard sides carry the free KKT multipliers with `t = 0`. The adjoint
pre-scales the m-seed by t and post-scales `sens.t` by 1/t (HPIPM parity).
The block (partial-condensing) QP solver has an equivalent sensitivity
layer: `HpipmQpSolver<..., BlockQpDim<P, NB>>::sens_forward` / `sens_adj`
(5h) reuse the staged 5f hard-equality KKT path verbatim — `BlockQpDim`
exposes the hard-equality rows (`eq_rows_*`), so the staged
`fact_solve_kkt_hardeq` is Dim-generic and the seed type is generic too
(`QpSeed<P, NH, Dim>`; the block seed is sized by `BlockQpDim`). The block
step is mapped back to the staged layout by `expand_block_step`
(`solvers/hpipm/block_condense.hpp`): controls/slacks and block-start
states transfer directly, internal state increments are forward-simulated
by default **homogeneously** (`dx_{k+1} = BA [du_k; dx_k]` — the x_0
sensitivity path, whose seed carries no dynamics offset), or with the
affine seed offset (`include_b = true`: `dx_{k+1} = BA [du_k; dx_k] + b_k`,
used by `sens_param` 5j, where the seed `b` is the dynamics p-derivative);
`dlam`/`dt` transfer row-group by row-group
(`copy_block_duals`), and the internal dynamics multipliers come from a
*linearized* stationarity backward sweep (no grad term; the reg-primal
correction is O(reg) ~ 1e-15 and is omitted). The same linearized sweep
is used by `expand_full_sol` when called in step mode (`include_b =
false`).

`AcadosSolver::sens_x0` supports all three QP modes: `kStaged` (5f),
`kFullCondensed` (5g: condense → solve → seed the pin box rows coherently
→ `sens_forward` → `expand_full_sol`), and `kPartialCondensed` (5h:
condense to the block QP → solve → seed the pin rows of block 0 →
`sens_forward` → `expand_block_step`; falls back to the staged path when
the block size does not match the solver's compile-time `NB`).

`AcadosSolver::sens_param` (5i/5j; requires `Dims::np > 0`) computes the
first-order sensitivity of the converged OCP solution to a parameter
vector `p`. It reassembles the staged QP at the converged iterate,
re-solves it on a throwaway solver (caching the exit KKT factors), builds
the seed from the problem's `*_sens_p` entries (p-derivatives of the QP
RHS plus the `(dJ/dp)' lambda` stationarity couplings, contracted with
the direction `dp`), and runs one seeded KKT re-solve. Because the QP is
in step form and the converged step is zero, no p-derivative of the LHS
(Hessian / BA / DC) is needed. Active-set gating: the d-seed is written
only on constraint sides that are active at the converged solution
(barrier-side slack below 1e-3); for an inactive constraint the
first-order response of its value / bound is exactly zero, and seeding it
would otherwise leak an O(mu_exit) artifact through the
nearly-singular barrier row. All three QP modes work (5j): the staged
seed is condensed with the same machinery as the base QP
(`build_seed_qp` → `condense_full` / `condense_block`), the condensed
seed solve runs on the LHS of the converged condensed QP, and the result
is expanded back to the staged layout with
`expand_full_sol(qseed, …, include_b = true)` /
`expand_block_step(qseed, …, include_b = true)` — the internal-state
forward simulation carries the seed `b` (dynamics p-derivative). The
x0-sensitivity path (5h) keeps the default `include_b = false` (its seed
has no dynamics offset).

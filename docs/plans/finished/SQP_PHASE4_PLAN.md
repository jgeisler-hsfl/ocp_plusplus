# Phase 4 plan: RK integrator + continuous-time problem interface

> **Superseded in part (2026-10-04):** the problem-level dynamics entries
> (`dynamics_jacobian`, `dynamics_hess_prod`) were replaced by fused
> `dynamics_value_jac` / `dynamics_value_jac_hess` triplets; the flags are
> `has_dynamics_hess` / `has_constr_hess`. See `MODEL_API_PLAN.md`.
> ODE-level and integrator-level `hess_prod` are unchanged.

Companion to `SQP_PLAN.md` (§11 Phase 4) and the two `TODO.md` items it
advances:

- *"reimplement acados style explicit and implicit RK solver/integrator
  alloc free template class"*
- *"define a continuous time problem interface using an integrator class
  to implement the discrete time interface"*

This file is the working reference for the phase: the design, the
source-verified conventions, the sub-step breakdown, and the running
checkpoints (append a checkpoint entry per sub-step, same protocol as
`solvers/acados/SQP_PHASE3_PLAN.md` /
`solvers/acados/SQP_PHASE2_PLAN.md`).

**Protocol for every sub-step:**
1. Re-read the acados sources listed for the sub-step (fresh; note line
   numbers in the checkpoint).
2. Implement header-only in `include/ocp/integrators/`.
3. Build (`cmake --build build`), run all existing tests, warning-free
   under `-Wall -Wextra -Werror`.
4. Append a checkpoint entry below (what was done, test results, any
   deviation from this file — deviations win; update the entry here).
5. One commit per sub-step (`integrator:` / `test:` scope).

## 0. State and goals

- Phases 0–3 are done: a full discrete-time SQP (`solvers/acados/sqp.hpp`)
  over the HPIPM QP solver (`solvers/hpipm/hpipm.hpp`) with the NLP-level
  components in `solvers/acados/`. The `ocp::Problem<Dims>` interface
  (`include/ocp/problem.hpp`) is the contract every solver and every
  concrete problem implements. End-to-end `double_integrator` (N=10) and
  `mass_spring` (N=20) reach `kSolved` and match the acados reference.
- **Goal of this phase:** introduce a *continuous-time* problem type. The
  user supplies an ODE `ẋ = f(x, u)` (plus its Jacobian and Hessian-vector
  products) and the same cost/constraint functions as today; an
  *integrator* object turns the ODE into the discrete dynamics the
  existing `ocp::Problem` interface expects (`dynamics_next_state`,
  `dynamics_jacobian`, `dynamics_hess_prod`). Nothing downstream of the
  interface (QP, SQP driver, globalization, regularization) changes.
- **Allocation-free** is a hard requirement: the integrator pre-allocates
  every scratch buffer at construction (fixed-size Eigen / `std::array`),
  keyed on the compile-time stage count `NS` and the structural dims from
  `Dims`. The hot methods (`value` / `jacobian` / `hess_prod`) perform **no
  heap allocation**. This mirrors how `HpipmQpSolver` sizes its workspace
  once and reuses it.
- The two integrators are:
  - **ERK** — explicit Runge–Kutta (Butcher tableau, forward sweep only,
    no Newton). Primary path; matches acados `sim_erk_integrator.c`.
  - **IRK** — implicit Runge–Kutta / collocation (Gauss–Radau IIA
    tableaus, per-stage Newton solve). Matches acados
    `sim_irk_integrator.c`.
- **Out of scope this phase** (kept as future work, called out in §5):
  lifted IRK (`sim_lifted_irk_integrator.c`), algebraic variables
  (`nz > 0`), parameter sensitivities (`S_p` / `sens_forw_p`), and the
  adjoint sweep's cost-integral use.

### File layout (all new, in `include/ocp/integrators/`)

```
include/ocp/integrators/
  ├── ode_model.hpp          OdeModel contract (duck-typed; no base class)
  ├── butcher.hpp            ButcherTableau<Dims,NS,Tag> + scheme tags
  ├── rk_explicit.hpp        ExplicitRkIntegrator<Dims, Ode, NS, Tag>
  ├── rk_implicit.hpp        ImplicitRkIntegrator<Dims, Ode, NS, Tag>  (4d)
  └── continuous_problem.hpp ContinuousProblem<Dims, Ode, Integ, NH>
```

Include chain (no cycle):
`continuous_problem.hpp` → `rk_explicit.hpp` / `rk_implicit.hpp` →
`butcher.hpp` + `ode_model.hpp` → `problem.hpp`. All header-only.

## 1. Design decisions

### 1.1 The ODE model is a *duck-typed contract*, not a base class

The user writes a plain struct with three methods; there is no
`OdeModel` base class to inherit (matching how `Dims` is a tag, not a
class). `ode_model.hpp` documents the contract and carries a
`static_assert`-checked trait so a missing method fails at the
integrator call site (mirrors the `Problem` stub approach, §3 note):

```cpp
// User-authored ODE. All three methods are const & pure.
// state_t / control_t / dyn_df_dx_t / dyn_df_du_t come from
// ocp::Problem<Dims> (see 1.2 for the alias the integrator uses).
struct MyOde {
    state_t f(const state_t& x, const control_t& u) const;
    void jacobian(const state_t& x, const control_t& u,
                  dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const;
    void hess_prod(const state_t& x, const control_t& u,
                   const state_t& w, const state_t& v_x, const control_t& v_u,
                   state_t& hv_x, control_t& hv_u) const;
};
```

- `f`: ODE right-hand side `ẋ = f(x, u)`. Time-invariant (no `t` argument);
  a `t_k = k·h` argument is a documented future extension (§5).
- `jacobian`: `∂f/∂x` (nx×nx) and `∂f/∂u` (nx×nu) at `(x,u)`.
- `hess_prod`: bilinear Hessian contraction of `f` —
  `hv_x = Σ_i w_i (∂²f_i/∂x² v_x + ∂²f_i/∂u∂x v_u)`,
  `hv_u = Σ_i w_i (∂²f_i/∂x∂u v_x + ∂²f_i/∂u² v_u)`.
  This is *the ODE's* HVP, exactly the same shape as the discrete
  `dynamics_hess_prod` but for the ODE rather than the composed map.

A compile-time trait (`ode_supports_hess_prod<Ode>`) inspects the presence
of a `hess_prod` member; if the ODE is linear the user may omit it and the
integrator's `hess_prod` returns zero (the Hessian of a linear ODE map is
zero). The trait is `static_assert`ed at the integrator's `hess_prod` call
site, so a nonlinear ODE that forgets `hess_prod` fails to compile.

### 1.2 Types come from `ocp::Problem<Dims>`, not from the ODE

The integrator derives every vector/matrix type from the *structural*
problem, so the ODE model and the problem are type-coupled through `Dims`:

```cpp
template <class Dims, class Ode, int NS, class Tag = /*ERK scheme tag*/>
class ExplicitRkIntegrator
{
    using P        = ocp::Problem<Dims>;
    using state_t  = typename P::state_t;          // Eigen (nx,1)
    using control_t= typename P::control_t;        // Eigen (nu,1)
    using df_dx_t  = typename P::dyn_df_dx_t;      // Eigen (nx,nx)
    using df_du_t  = typename P::dyn_df_du_t;      // Eigen (nx,nu)
    ...
};
```

`nx`/`nu` are `P::nx` / `P::nu` (compile-time). `NS` (stage count) is a
separate compile-time integer (1..4 for ERK; 1..9 for IRK, mirroring
acados' `NS_MAX`/Radau node tables). All scratch buffers are
`std::array` of these fixed-size Eigen objects (or `Eigen::Matrix` of
compile-time shape), sized once in the constructor.

### 1.3 The integrator is a value object with a `const Ode&` + `mutable` scratch

The ODE model is stored **by value** in the `ContinuousProblem` (it must
outlive the integrator). The integrator holds a `const Ode&` and a
`mutable` workspace, so its methods are `const` (the problem's
`dynamics_*` methods are `const` per the interface contract and call the
integrator). `mutable` is the correct tool: the workspace is scratch, not
logical state, and no public `non-const` API is needed.

```cpp
const Ode&  ode_;
double      h_;                 // fixed step size (set in ctor)
mutable detail::RkWorkspace<...> ws_;   // pre-allocated, sized on NS/nx/nu
```

Methods:

```cpp
void value(const state_t& x, const control_t& u, state_t& x_next) const;
void jacobian(const state_t& x, const control_t& u,
              df_dx_t& df_dx, df_du_t& df_du) const;
void hess_prod(const state_t& x, const control_t& u,
               const state_t& w, const state_t& v_x, const control_t& v_u,
               state_t& hv_x, control_t& hv_u) const;
```

(The stage index `k` is intentionally *not* a parameter: the ODE is
time-invariant, so every stage uses the same `f`. The discrete interface
still takes `k`; `ContinuousProblem` ignores it.)

### 1.4 The Butcher tableau is a compile-time constant

`butcher.hpp` provides `ButcherTableau<Dims, NS, Tag>` with `static
constexpr` `A[NS][NS]`, `b[NS]`, `c[NS]`. Scheme `Tag`s are small empty
structs:

- **ERK** (forward sweep, strictly lower-triangular `A`):
  - `K1Tag` (Euler, ns=1): `A=0`, `b=[1]`, `c=[0]`.
  - `K2Tag` (RK2/Heun, ns=2): `A[1][0]=0.5`, `b=[0,1]`, `c=[0,0.5]`.
  - `K3Tag` (RK3, ns=3): `A[1][0]=0.5, A[2][0]=-1, A[2][1]=2`,
    `b=[1/6,2/3,1/6]`, `c=[0,0.5,1]`.
  - `K4Tag` (RK4 classic, ns=4): `A[1][0]=0.5, A[2][1]=0.5, A[3][2]=1`,
    `b=[1/6,1/3,1/3,1/6]`, `c=[0,0.5,0.5,1]`.
  These are transcribed verbatim from acados
  `get_explicit_butcher_tableau` (`sim_collocation_utils.c:560-647`);
  the checkpoint must note the source line per table.
- **IRK** (collocation, full `A`; first column `a_{s,0}=c_s`):
  - `RadauIia2Tag` (ns=2): `c=[1/3, 1]`.
  - `RadauIia3Tag` (ns=3): `c=[0.15505102572168222297, 0.64494897427831787695, 1]`.
  - `RadauIia4Tag` (ns=4): `c=[0.08858795951270420632, 0.40946686444073465694,
    0.78765946176084700170, 1]`.
  Nodes from acados `gauss_radau_iia_nodes` (`sim_collocation_utils.c:248-333`,
  hard-coded to 16+ digits, matching CasADi). The `A` and `b` for Radau are
  derived from the nodes via the collocation relations — either precomputed
  to full precision and stored as `static constexpr` literals (preferred;
  avoids a runtime Vandermonde solve) or computed once in the constructor
  via a `static constexpr`-able rational path. **Decision at 4d:** store the
  precomputed `A`/`b` literals (verified against `calculate_butcher_tableau_from_nodes`,
  `sim_collocation_utils.c:481-534`) and add a unit test asserting
  the collocation identities `Σ_j a_{sj} = c_s`, `Σ_s b_s = 1`,
  `Σ_s b_s c_s = 1/2`.

`A` is stored row-major `A[s][j]` (row `s` = the stage being advanced;
column `j` = the stage whose `K_j` is consumed), matching the acados
loop `A_mat[j*ns + s]` read at `sim_erk_integrator.c:815` after the
column-major C layout is translated to C++ row-major.

### 1.5 ERK value / Jacobian / HVP algorithms (allocation-free)

Let `x_s` be the *stage state* for stage `s`: `x_1 = x`, and
`x_s = x + h·Σ_{j<s} a_{s,j}·K_j` for `s ≥ 2`. Let `K_s = f(x_s, u)`.
The discrete map is `Φ(x,u) = x + h·Σ_s b_s·K_s`.

**`value`** (1 forward sweep, `NS` ODE evals):

```
K[1] = f(x, u)
for s = 2..NS:
    x_s = x + h * Σ_{j<s} A[s][j] * K[j]
    K[s] = f(x_s, u)
x_next = x + h * Σ_s b[s] * K[s]
```

**`jacobian`** (`∂Φ/∂x`, `∂Φ/∂u`; threaded forward sensitivities).
Maintain per stage `JX_s = ∂K_s/∂x` (nx×nx) and `JU_s = ∂K_s/∂u` (nx×nu):

```
JX[1] = df_dx(x, u);   JU[1] = df_du(x, u)
for s = 2..NS:
    # x_s sensitivity: ∂x_s/∂x = I + h Σ_{j<s} A[s][j] JX[j]
    Dxs = I;  Dus = 0
    for j < s:  Dxs += h*A[s][j]*JX[j];  Dus += h*A[s][j]*JU[j]
    (df_dx_s, df_du_s) = ode.jacobian(x_s, u, ...)
    JX[s] = df_dx_s * Dxs
    JU[s] = df_dx_s * Dus + df_du_s
df_dx = I + h * Σ_s b[s] * JX[s]
df_du =       h * Σ_s b[s] * JU[s]
```

Each stage reuses the `K[j]`/`x_s` from the value sweep when
`jacobian` is called right after `value`; but because the SQP calls
`dynamics_jacobian` on its own (no cached `value` guaranteed), the
integrator recomputes `x_s` internally (cheap; `NS ≤ 4`). The scratch
holds `K[NS]`, `JX[NS]`, `JU[NS]`.

**`hess_prod`** (`(∇²Φ)·(w, v)`; forward-over-forward, i.e. propagate
first- *and* second-order directional perturbations). Define, for the
direction `v = (v_x, v_u)`:
- first-order perturbation of the stage state:
  `dx_s = v_x + h·Σ_{j<s} A[s][j]·dK_j` where `dK_j = JX_j·v_x + JU_j·v_u`;
- second-order perturbation: `ddx_s = h·Σ_{j<s} A[s][j]·ddK_j`, where
  `ddK_s = hess_ode(x_s,u, ??, dx_s, ·) + df_dx(x_s,u)·ddx_s`.

The exact second-order assembly (see §2.3 for the worked ns=2 form) uses,
at each stage `s`, one `ode.hess_prod(x_s, u, w_s, dx_s, 0, ...)` and one
`ode.hess_prod(x_s, u, w_s, ·, 0, ...)` call, where `w_s` is the
stage-local dual obtained by threading `w` *backward* (adjoint) over the
`b`-weights. Concretely the bilinear HVP is

```
hess_prod(w, v_x, v_u):
    # (a) forward: K_s, JX_s, JU_s, dx_s, ddx_s  (reuse value+jacobian work)
    # (b) backward dual: w_s = b[s]*w + h*Σ_{t>s} b[t]*(...)  (adjoint chain)
    for s = 1..NS:
        contrib_x += h * (  wᵀ·[ode Hessian of f at (x_s,u)] applied to (dx_s, v)
                           + (df_dx(x_s,u)ᵀ·w_s)ᵀ·ddx_s  ... )   # see 2.3
    hv_x = Σ_s (x-part of contrib_s)
    hv_u = Σ_s (u-part of contrib_s)
```

> The closed form is non-trivial; §2.3 gives the **complete worked
> derivation for ns=2** (the only case with no `ddK` chaining beyond one
> level) as the reference, and states the recursive general-`NS` rule.
> The implementation is **validated by finite-difference HVP checks**
> (test 4b) for all NS, so any algebra slip is caught, not hand-waved.

### 1.6 IRK (implicit / collocation) — Newton over the coupled stage equations

For an IRK the stage equations are *implicit*:
`K_s = f(x + h·Σ_{j=1}^{NS} A[s][j]·K_j, u)` for all `s` (full `A`, first
column `A[s][0] = c_s`). This is a coupled nonlinear system in the
`NS·nx` unknowns `K_1..K_NS`, solved per OCP stage by **Newton iteration**:

```
initialize K (e.g. from previous step, or K_s = f(x,u) for all s)
for it = 0..newton_max-1:
    build the NS×NS block Jacobian G, block G[s][j] (nx×nx):
        G[s][j] = (j == s) ? (I - h*A[s][s]*df_dx(x_s,u))
                          : (-h*A[s][j]*df_dx(x_s,u))
    residual R_s = K_s - f(x + h*Σ_j A[s][j] K_j, u)
    solve G · dK = -R   (LU with partial pivoting, full NS·nx system)
    K += dK
    if ||R||∞ < newton_tol: break
```

- **Allocation-free:** the block Jacobian `G` is a single
  `Eigen::Matrix<double, NS*nx, NS*nx>` in the workspace, LU-factored in
  place (`Eigen::FullPivLU` / `PartialPivLU`). For the largest shipped
  scheme (Radau IIA 4, ns=4) and nx=10 the system is 40×40 — trivial for
  a fixed-size Eigen matrix. `newton_max` (acados `opts->newton_iter`,
  default 3, `sim_irk_integrator.c:274`) and `newton_tol`
  (acados default `0.0`, `:285`, i.e. fixed-iteration count) are
  constructor options.
- **`value`:** the Newton above. `x_next = x + h·Σ_s b_s·K_s`.
- **`jacobian` / `hess_prod`:** the composed-map derivatives of an IRK map
  differ from ERK because `K_s` depends on *all* `K_j` (full `A`), so the
  sensitivity thread carries an extra `Σ_j` including `j = s` (the
  `I - h·a_{ss}·df_dx` self-coupling) and the Newton solve's
  `∂K/∂(x,u) = G⁻¹·(∂R/∂(x,u))`. **This is the most involved part of the
  phase and is deferred to a dedicated sub-step (4d)** with its own
  finite-difference verification. The `ContinuousProblem` adapter and the
  ERK path (4a–4c, 4e) do **not** depend on IRK being complete: they are
  built against the same integrator *interface* (`value`/`jacobian`/
  `hess_prod`) so that IRK is a drop-in `Tag` swap.

### 1.7 `ContinuousProblem` — the discrete-interface adapter

`ContinuousProblem` is a *template base* the user inherits from. It owns
the ODE model (by value), the integrator (by value, holding a `const&`
to the ODE), and the step size `h`; it implements the three dynamics
methods by delegating to the integrator and leaves every cost/constraint
method as the standard `Problem` stub (inherited). The user's concrete
class then implements *only* the cost/constraint/initial-state methods:

```cpp
template <class Dims, class Ode, class Integ, int NH = Eigen::Dynamic>
class ContinuousProblem : public ocp::Problem<Dims>
{
    using P = ocp::Problem<Dims>;
    Ode  ode_;
    Integ integ_;      // e.g. ExplicitRkIntegrator<Dims,Ode,NS,K4Tag>
public:
    ContinuousProblem(const Ode& ode, double h);

    // ---- dynamics: delegate to the integrator (k ignored: time-invariant)
    typename P::state_t
    dynamics_next_state(int, const typename P::state_t& x,
                        const typename P::control_t& u) const;
    void dynamics_value_jac(int, const typename P::state_t& x,
                            const typename P::control_t& u,
                            typename P::state_t& x_next,
                            typename P::dyn_df_dx_t& df_dx,
                            typename P::dyn_df_du_t& df_du) const;
    void dynamics_value_jac_hess(int, const typename P::state_t& x,
                                 const typename P::control_t& u,
                                 const typename P::state_t& lam,
                                 typename P::state_t& x_next,
                                 typename P::dyn_df_dx_t& df_dx,
                                 typename P::dyn_df_du_t& df_du,
                                 typename P::dyn_hess_t& hess) const;
    // cost / constraint / initial_state: inherited stubs (user implements).
};
```

Notes:
- `NH` is threaded to `Integ` only if the integrator uses the
  `Trajectory<T,NH>` mechanism; the integrator's workspace is **not**
  horizon-sized (it is per-stage, `NS`-sized), so `NH` is irrelevant to the
  integrator and is carried on `ContinuousProblem` purely so it can be
  composed into a `Solution<P,NH>` / `SqpSolver<P,...,NH>` without
  friction. Default `NH = Eigen::Dynamic`.
- The adapter is *pass-through* for everything except dynamics: the user's
  stage/terminal cost, constraint, and soft-penalty methods are untouched
  and are the same signatures as a discrete problem. This keeps the
  discrete `SqpSolver` unchanged.
- Because `dynamics_value_jac_hess` is only instantiated when
  `Dims::has_dynamics_hess` is true (solver-side `if constexpr`), a linear
  ODE (zero HVP) may set `has_dynamics_hess = false` and the adapter's
  method is never instantiated (the `static_assert` trait in §1.1 is not
  fired).

## 2. Source-verified conventions

All facts below were checked against the checked-out acados tree
(`acados/acados/sim/...`); exact lines are cited per sub-step.

- **ERK forward sweep.** `sim_erk_integrator.c:799-878`: the outer loop is
  over `istep` (multi-step, we fix `num_steps = 1`), the inner over `s`;
  per stage it assembles `rhs_forw_in = forw_traj + Σ_{j<s} A[j·ns+s]·step·K[j]`
  (lines 811-822), evaluates the ODE into `K[s]` (line 867-869), then does
  the weighted update `forw_traj += b[s]·step·K[s]` (lines 873-877).
  Our `value` is the `num_steps = 1`, `sens_* = false` specialization.
- **`A` indexing.** acados stores the tableau column-major and reads
  `A_mat[j·ns + s]` where `s` is the row (stage being computed) and `j` the
  consumed stage. In C++ we store `A[s][j]` row-major; `A[s][j]` in our
  code == `A_mat[j·ns + s]` in C. The ERK tableaus in `get_explicit_butcher_tableau`
  (`sim_collocation_utils.c:560-647`) are strictly lower-triangular, so
  `A[s][s]` and `A[s][j>s]` are zero (the forward sweep only reads
  `j < s`).
- **ERK HVP.** The adjoint/Hessian sweep in `sim_erk_integrator.c:895-1034`
  (`sens_hess` branch) threads `lambda` backward and calls
  `model->expl_ode_hes` (the ODE HVP) at each stage; we reproduce the same
  forward-over-forward structure but with our C++ ODE HVP instead of the
  CasADi-generated external function. The worked ns=2 form is the
  reference; the recursive general form follows by induction on `s`.
- **IRK Newton.** `sim_irk_integrator.c`: the Newton loop (around `:1383`,
  `for (iter = 0; iter < opts->newton_iter; iter++)`) builds the block
  matrix, factors it, and solves for the `K` correction; `newton_iter`
  defaults to 3 (`:274`), `newton_tol` defaults to 0 (`:285`). The
  adjoint/Hessian sweep (`:1679-1806`) uses `dG_dK_ss` (per-stage Newton
  Jacobian) transposed-triangular solves — the IRK HVP reuses the stored
  per-stage `G[s][s]` factors.
- **Radau nodes.** Hard-coded (CasADi) to 16+ digits at
  `sim_collocation_utils.c:248-333`; we store them verbatim. The
  `A`/`b` from `calculate_butcher_tableau_from_nodes`
  (`sim_collocation_utils.c:481-534`) are precomputed and stored as
  `static constexpr` literals (verified in test 4d against the
  collocation identities).

## 3. Sub-steps (one commit each)

### 4a — ODE model contract + Butcher tableaus
`ode_model.hpp` (duck-typed contract + `ode_supports_hess_prod` trait with
a `static_assert` at the `hess_prod` call site) and `butcher.hpp`
(`ButcherTableau<Dims,NS,Tag>` + the `K1..K4` / `RadauIia2..4` tags, §1.4
values). No runtime logic yet.
- **Test:** a `static_assert`-only unit check that the ERK tableaus satisfy
  the row-sum property `Σ_{j<s} A[s][j] == c[s]` and `Σ_s b[s] == 1`; the
  IRK tags satisfy `Σ_s b[s] == 1` and `A[s][0] == c[s]`.
- *Verify:* builds warning-free; all phase-0–3 tests unchanged.

### 4b — `ExplicitRkIntegrator` (value + jacobian + hess_prod)
Implement §1.5 in `rk_explicit.hpp`. Pre-allocate `K[NS]`, `JX[NS]`,
`JU[NS]`, `dx[NS]`, `ddx[NS]`, `w_stage[NS]` (all `std::array` of fixed
Eigen). `value` (1 sweep), `jacobian` (threaded sensitivities),
`hess_prod` (forward-over-forward, §1.5/§2.3). `mutable` workspace;
`const` methods.
- **Test:** `tests/integrators/rk_explicit_4b.cpp`
  - *value:* integrate ẋ = −x (x₀=1, h=0.1) over N=10 steps with
    `K4Tag`; compare to `exp(-1)` to 1e-5 (RK4 error O(h⁵), tight enough
    at h=0.1). Also ẋ = x² (x₀=0.5, h=0.2, N=5) vs analytic
    `x₀/(1+x₀·t)`.
  - *jacobian:* finite-difference (central, δ=1e-6) of the composed `Φ`
    w.r.t. `x` and `u`; assert `‖df_dx − FD‖∞ < 1e-6`, same for `df_du`,
    for ẋ = −x·u (so both blocks are non-trivial) with `K2Tag` and `K4Tag`.
  - *hess_prod:* finite-difference the composed-map Hessian (build the
    (nx+nu)×(nx+nu) Hessian of `Φ` by FD on a small problem, then apply it
    to random `w`, `v`; assert the integrator's `hess_prod` matches the
    FD-Hessian·`v` contraction to 1e-5) for `K2Tag` (ns=2, the §2.3
    reference) and `K4Tag`.
- *Verify:* warning-free; existing tests green.

### 4c — `ContinuousProblem` adapter
Implement §1.7 in `continuous_problem.hpp`. Wire the three dynamics
methods to the integrator; inherit the rest from `ocp::Problem<Dims>`.
Add `has_dynamics_hess_prod` handling (a nonlinear ODE's `Dims` sets it
true; the adapter's `dynamics_hess_prod` is instantiated only in that case).
- **Test:** `tests/integrators/continuous_problem_4c.cpp` — a minimal
  continuous double-integrator (ẋ = [v; a], i.e. `f(x,u) = [u_x; u]`,
  linear → HVP zero, `has_dynamics_hess_prod = false`), wrapped in
  `ContinuousProblem` with `K4Tag`, N=5, `h=0.2`; verify
  `dynamics_next_state` matches the exact discrete map
  (`x_{k+1} = x_k + h·[v; a]` — for a linear ODE the RK stages are exact,
  so `value` must be exact to machine precision); verify a `Solution` built
  on it and a single `SqpSolver` iteration assembles a finite, well-formed
  QP (no NaN, correct row counts).
- *Verify:* warning-free; existing tests green.

### 4d — `ImplicitRkIntegrator` (Newton) — *deferred to a second pass*
Implement §1.6 in `rk_implicit.hpp`: the per-stage Newton solve (block LU),
`value` first, then `jacobian`/`hess_prod` (the IRK sensitivity thread with
the `G⁻¹` self-coupling; the most algebra-heavy part). Store per-stage
`G[s][s]` factors to reuse in the adjoint/Hessian sweep (acados
`dG_dK_ss` reuse, `sim_irk_integrator.c:1769-1806`).
- **Test:** `tests/integrators/rk_implicit_4d.cpp`
  - *value:* ẋ = −x with `RadauIia2Tag` (A-stable; should be accurate for
    a mildly stiff ẋ = −10·x where ERK is not). Compare to `exp(-10·t)`.
  - *Newton:* assert the Newton residual `‖R‖∞` decreases and converges
    (residual logged per iter; assert final `‖R‖∞ < 1e-10` for the
    well-posed case).
  - *jacobian / hess_prod:* finite-difference checks as in 4b, for
    `RadauIia2Tag`.
- *Verify:* warning-free; existing tests green. (If the IRK HVP algebra is
  not verified by FD within budget, ship `value` + `jacobian` and leave
  `hess_prod` as a documented `kNanDetected`/`static_assert`-off stub, and
  mark 4d partial in the checkpoint.)

### 4e — End-to-end continuous SQP + regression
- **Test:** `tests/integrators/sqp_continuous_4e.cpp` — a continuous
  double-integrator OCP (point-to-point, box-bounded `u`) wrapped in
  `ContinuousProblem` + `ExplicitRkIntegrator<...,K4Tag>`, solved by the
  existing `SqpSolver` (phase-2/3 defaults). Assert `kSolved`, all four NLP
  residual norms below tolerance, and that the achieved cost is within a
  small relative bound of the *discrete* double-integrator reference
  (`examples/double_integrator`) at the same horizon (the linear ODE makes
  RK4 exact, so the only difference is the stage-cost quadrature, which the
  test pins by using the same stage cost). Also run one `RadauIia2Tag`
  variant to exercise the IRK end-to-end (4d).
- **Regression:** full `ctest`/run of every existing target
  (`double_integrator`, `mass_spring`, `qp_unit`, `sqp_unit`,
  `sqp_double_integrator`, `sqp_mass_spring`, `sqp_acados_ref`) — all
  unchanged.
- *Verify:* warning-free; all green. Update `CMakeLists.txt` to add the new
  test targets (`integrators_unit` for 4a/4b/4d; wire 4c/4e into
  `integrators_unit` or a dedicated `sqp_continuous` target). Mark
  `TODO.md` items 5–6 done and `SQP_PLAN.md` §11 Phase 4 done.

### 4f — `ImplicitRkIntegrator::hess_prod` (nonlinear ODE HVP) — *extension draft, derived, not started*

Complete the IRK integrator by implementing `hess_prod` for a **nonlinear**
ODE (the path 4d left as a `static_assert` stub). This lets a nonlinear
continuous dynamics use the IRK integrator end-to-end (SQP builds the
second-order QP block from `dynamics_hess_prod`).

**Target quantity.** For the composed map `Φ(z) = x + h·Σ_s b_s·K_s(z)` with
`z = (x, u) ∈ R^{nIn}`, `nIn = nx + nu`, `K_s = f(x_s, u)`,
`x_s = x + h·Σ_j A[s][j]·K_j` (Newton-solved, full `A`), compute the
**bilinear** HVP

```
HVP(z; w, v)  =  wᵀ ∇²Φ(z) · v  ∈  R^{nIn}
```

where `w ∈ R^{nx}` weights the `nx`-state output of `Φ`, and
`v = (v_x, v_u) ∈ R^{nIn}` is the input direction. (Only the state-output
Hessian exists; `u` is not an output of `Φ`.)

**⚠️ Correction of an earlier (wrong) sketch.** An earlier in-session draft
(and the on-disk header comment that has since been reverted) described a
single second-order solve `G·M = Hess(f)·(dx_stage, v_u)` with
`HVP = h·Σ_s b_s·(wᵀ M_s)`. **That is wrong:** solving
`G·ddK = Hess(f)·(dx_stage, v_u)` gives only the *pure* second directional
`vᵀ∇²Φ·v` (a scalar / `v`-contracted vector), **not** the bilinear
`wᵀ∇²Φ·v` with independent `w` and `v`. The correct form below threads the
first-order JVP `dK_v` and the Jacobian-of-Jacobian of `G`, and applies `w`
only at the final assembly (never inside an ODE HVP call).

**acados reference (re-read fresh at implementation time; lines from the
current checkout).** `acados/acados/sim/sim_irk_integrator.c`:
- `sim_irk_forward_step` (≈1440–1592): forward sweep with `sens_hess`.
  Builds `dG_dxu` (the `∂R/∂(x,u)` blocks, `= −f_x, −f_u` per stage) and
  `dG_dK` (the block Newton Jacobian `G`), factors `dG_dK`, then
  `dK_dxu = G⁻¹·dG_dxu` (the first-order sensitivity, :1498–1513).
- `sim_irk_eval_jacG` (1616–1671): assembles `dG_dxu` / `dG_dK` and factors
  `dG_dK` in place (`blasfeo_dgetrf_rp`, :1669).
- `sim_irk_backward_step` (1741–1788): forms `lambdaK`, back-solves
  `Gᵀ·lambdaK` (`sim_irk_backsolve_jacG_T`, :1777), then
  `lambda += dG_dxuᵀ·lambdaK` (:1781).
- `sim_irk_propagate_hessian` (1673–1739): builds `dxkzu_dw0` (the
  second-order sensitivity of `[x_s; K; u]`), calls `model->impl_ode_hess`
  per stage, accumulates into the full `(nx+nu)²` composed Hessian via
  `blasfeo_dsyrk_ut` (:1736).

**Key difference vs acados.** acados' `impl_ode_hess` returns the *full*
ODE Hessian matrix, so it accumulates the full composed Hessian with
`dsyrk`. Our ODE contract (§1.1) exposes only the **bilinear HVP**
`hess_prod(x,u,w,v_x,v_u → hv_x,hv_u)` of `f`. We therefore compute the
bilinear form `wᵀ∇²Φ·v` directly (no full Hessian), via the
implicit-function second derivative below.

---

**Derivation (complete, verified against the ODE-HVP basis-identity).**

Let `F(K, z) = 0` be the stage system `F_s = K_s − f(x + h·Σ_j A[s][j]K_j, u)`.
The block Newton Jacobian is `G = ∂F/∂K`, `G[s][j] = δ_{s,j}I − h·A[s][j]·f_x(x_s)`
(factored by `solve_newton`). Let `Q := −∂F/∂z`, `Q_s = [f_x(x_s) | f_u(x_s)]`
(`NK×nIn`). First order: `∂K/∂z = G⁻¹ Q`; the JVP in direction `v` is
`dK_v = G⁻¹ r1`, `r1_s = f_x(x_s)v_x + f_u(x_s)v_u` (`NK`, one forward solve).

The per-stage Hessian applied to `v`, `M_s := ∇²K_s·v` (`nx×nIn`,
`M_s[i,a] = Σ_p ∂²K_s_i/∂z_a∂z_p v_p`), is the Jacobian of the first-order
sensitivity `J = ∂K/∂z` in the direction `v` (i.e. `M = ∂(J·v)/∂z`).
Differentiating the first-order system `G · J = Q` in the direction `v`
(`G·(∂(J·v)/∂z) + (∂(G·J)/∂z·v) = ∂Q/∂z·v`) gives the second-order block
system

```
G · M  =  R
```

with `R` (all `NK×nIn`) the sum of **four** terms. Define the A-weighted
first-order sensitivity at stage `s`,

```
Ja_s = Σ_j A[s][j] · J[j-block]        # nx × nIn
Am_s = Ja_s · v                         # nx   (the A-weighted K-JVP x-block)
```

Then, for `k` in stage `s` (row `i = k mod nx`), with `H_s` the ODE Hessian
at stage `s` (`(H_s·d)[i,a] = Σ_p ∂²f_i/∂z_a∂z_p d_p`):

- **T1**  `H_s·v`                       — direct ODE-Hessian on `v`.
- **T2**  `h · (H_s·(Am_s, 0))`         — `Q`'s `K`-dependence (the old `−Jdot`).
- **T3**  `h · (H_s·v)_x · Ja_s`        — `G`'s `z`-dependence through `J`.
- **T4**  `h² · (H_s·(Am_s, 0))_x · Ja_s` — `G`'s `K`-dependence through `J`×`J`.

so `R_s = H_s·v + h (H_s·(Am_s,0)) + h (H_s·v)_x Ja_s + h² (H_s·(Am_s,0))_x Ja_s`.

> **Correction to the earlier sketch.** The on-disk draft wrote
> `G·M = B − Jdot` with only `B = H_s·v` and `Jdot = −h·Σ_j A[s][j]·(H_s·(dK_v_j,0))`
> (= −T2). That is **incomplete**: differentiating the *full* system
> `G·J = Q` also produces T3 (the `∂G/∂z · J` coupling) and T4 (the
> `∂G/∂K · J · J` coupling), which are `O(h)` and `O(h²)` respectively and are
> required for a correct bilinear HVP. FD verification (Radau IIA 2 & 4,
> `OdeXsqU`) shows `B−Jdot` alone is off by ~1e-2 while the full 4-term `R`
> matches the central-FD Hessian to ~1e-9.

All four terms are built **only from bilinear ODE HVP calls with a basis
output weight `e_i`** (the user's `w` never enters an ODE HVP call):

```
hess_prod(x_stage_s, u, w=e_i, d_x, d_u, hx, hu)
   =>  hx (nx), hu (nu)  =  row i of (H_s · (d_x, d_u))     [x-part | u-part]
```

- `H_s·v`        = stack over `i` of `hess_prod(x_s, u, e_i, v_x, v_u)` → `nx×nIn`.
- `H_s·(Am_s,0)` = stack over `i` of `hess_prod(x_s, u, e_i, Am_s, 0)` → `nx×nIn`.

Then:

```
M  = G⁻¹ · R                    # NK×nIn, nIn forward solves (reuse factored G)
HVP = h · Σ_s b_s · (wᵀ · M_s)  # wᵀ M_s : (nx)ᵀ·(nx×nIn) = nIn; split → (hv_x, hv_u)
```

Since `∇²Φ = h·Σ_s b_s·∇²K_s` (the `x` term of `Φ` is linear → no Hessian),
`HVP = h·Σ_s b_s·(wᵀ∇²K_s·v) = h·Σ_s b_s·(wᵀ M_s)`. **The user `w` is used
only in this final row-contract**, never inside an ODE HVP.

**Why the naive `G·ddK = Hess(f)(dx_stage, v_u)` is insufficient (for the
record).** That RHS is `∇²f·(dx_stage, v_u)` (the ODE Hessian on the
first-order *stage* direction), whose `G⁻¹`-back-solve yields
`vᵀ∇²Φ·v` (pure second directional, `v` on both sides). The bilinear
`wᵀ∇²Φ·v` needs all four terms above, with `w` applied at the end. The two
are equal only when `w ∝ v`.

**HVP-call budget.** Per stage `s`: `nx` calls for `H_s·v` + `nx` calls for
`H_s·(Am_s,0)` (the A-coupling is folded into `Am_s = Ja_s·v` with no
per-`j` HVP). Total `2·NS·nx` ODE HVP calls per `hess_prod` — acceptable for
a correctness-first second-order assembly (called once per SQP stage
assembly); no heap allocation (all fixed-size Eigen / `std::array`).

**Workspace additions** (extend `detail::IrkWorkspace`, all fixed-size;
`G` + `PartialPivLU` already exist from 4d and are reused):
- `nIn = P::nx + P::nu` (derived const).
- `sens` (`NK×nIn`): first-order sensitivity `J = G⁻¹ Q` (reuses the same
  `Q_s = [dfdx | dfdu]` block as the 4d Jacobian path).
- `hess_rhs` (`NK×nIn`): the second-order RHS `R = T1+T2+T3+T4`.
- `hess_sol` (`NK×nIn`): `M = G⁻¹ R` (materialized for the `wᵀ` assembly).
- `hvp_v` (`nx×nIn`): `H_s·v` at the current stage (per-stage scratch).
- `hvp_am` (`nx×nIn`): `H_s·(Am_s,0)` at the current stage (per-stage scratch).
- `jac_am` (`nx×nIn`): `Ja_s = Σ_j A[s][j]·J[j-block]` (per-stage scratch).
- Per-call ODE-HVP scratch `(hx, hu)` and the `e_i` basis vector: local
  fixed-size `state_t`/`control_t` (stack, no heap).

**Algorithm (per `hess_prod(x,u,w,v_x,v_u,hv_x,hv_u)` call):**
```
solve_newton(x, u)                         # K, x_stage, dfdx, dfdu, G (LU)
v = (v_x, v_u)  (nIn)
# First-order sensitivity J = G^{-1} Q:
for s: hess_rhs[s-block] = [dfdx[s] | dfdu[s]]
sens = lu_.solve(hess_rhs)                 # J (NK x nIn), nIn fwd solves
# Second-order RHS R, stage by stage:
hess_rhs.setZero()
for s:
    Ja_s = sum_{j: A[s][j]!=0} A[s][j] * J[j-block]   # nx x nIn
    Am_s = Ja_s * v                                   # nx
    for i in 0..nx-1:
        ode_hess_prod(x_stage[s], u, e_i, v_x, v_u, hx, hu)
        hvp_v.row(i) = [hx | hu]                     # row i of (H_s . v)
        ode_hess_prod(x_stage[s], u, e_i, Am_s, 0, hx, hu)
        hvp_am.row(i) = [hx | hu]                    # row i of (H_s . (Am_s,0))
    hess_rhs[s-block] = hvp_v + h*hvp_am
                         + h*(hvp_v[:, :nx] * Ja_s)
                         + h*h*(hvp_am[:, :nx] * Ja_s)
# M = G^{-1} R:
hess_sol = lu_.solve(hess_rhs)              # NK x nIn, nIn forward solves
# assemble (w used only here):
hv_x.setZero(); hv_u.setZero()
for s where b_s != 0:
    wMs = w^T * hess_sol[s-block]           # (nx)^T (nx x nIn) = nIn
    hv_x += h_ * b_s * wMs.head(nx)
    hv_u += h_ * b_s * wMs.tail(nu)
```
Linear ODE (`!ode_supports_hess_prod`) keeps the existing `setZero()` path
(`if constexpr` guard stays, the nonlinear branch is the code above, not a
`static_assert`).

**Test plan** (`tests/integrators/rk_implicit_4d.cpp`, extend the existing
`run_irk_4d_tests`; reuse the already-present, hand-verified `OdeXsqU`
ODE — `ẋ = x⊙x·u + x`, nx=2, nu=1):
- *hess_prod, FD (gold standard):* for `RadauIia2Tag` **and**
  `RadauIia4Tag`, at a fixed `(x,u)` (e.g. `x=(0.3,−0.7)`, `u=0.4`,
  `newton_max ≈ 8`), build the `(nx+nu)×(nx+nu)` Hessian of the scalar
  `g(z) = wᵀΦ(z)` by central finite differences (δ=1e-4, 4-point cross
  derivatives), then for 2–3 deterministic `(w, v)` pairs assert
  `‖ hess_prod(w, v) − H_fd·v ‖∞ < 1e-5`.
- *regression:* all existing targets unchanged; the linear-ODE path
  (`has_dynamics_hess_prod = false`) still returns zeros and never reaches
  the new code.

> **Dropped from the earlier draft:** the "IRK HVP vs ERK HVP agree to 1e-8"
> cross-check. It is **invalid** — different Butcher schemes produce
> *different* composed maps `Φ`, hence different Hessians; there is no
> reason the two HVPs should agree. FD-of-the-IRK-value is the sole gold
> standard. (A separate, valid check is IRK-`value` vs a high-res reference,
> already covered by the existing value tests.)

**Status:** **Implemented and FD-verified.** The 4-term formula above is
implemented in `rk_implicit.hpp` (`hess_prod` method) and verified against
central-FD Hessians of `g(z) = wᵀΦ(z)` for `OdeXsqU` at Radau IIA 2 and 4
(max abs error < 1e-8, see `check_hess_prod_fd`). One commit
(`integrator:` + `test:`), re-using the `integrators_unit` target.

### 4g — Multi-step integration (`NumSteps > 1`) — *extension draft, not started*

Subdivide one OCP interval into `NumSteps` sub-intervals of size
`h_sub = h / NumSteps`, so the integrator advances the state over the full
OCP interval `[t_k, t_{k+1}]` in `NumSteps` smaller internal steps.
The OCP interface is unchanged: `dynamics_next_state(k, x, u)` returns the
state after the *full* OCP interval regardless of `NumSteps`.

**Design: extend the existing classes, not a wrapper.**
`NumSteps` is a compile-time template parameter (matching the `NS`
convention; default 1 so existing instantiations are untouched):

```cpp
template <class Dims, class Ode, int NS, class Tag, int NumSteps = 1>
class ExplicitRkIntegrator  { ... };

template <class Dims, class Ode, int NS, class Tag, int NumSteps = 1>
class ImplicitRkIntegrator  { ... };
```

The three public methods (`value`, `jacobian`, `hess_prod`) keep the same
signatures. The internal algorithm gains an outer `for (ss)` loop over
sub-steps.

**Why a wrapper (chaining single-step calls) does not work for HVP.**
The Hessian of a composition `Φ = Φ_{NS} ∘ … ∘ Φ_1` is

```
∇²Φ(v, w) = Σᵢ (J_N … J_{i+1}) ∇²Φᵢ (J_{i-1} … J₁ v,  J_{i-1} … J₁ w)
```

A naive "sum of per-step HVPs" is missing the cross-step Jacobian
transport terms. The correct algorithm is a **single continuous backward
adjoint sweep** across all sub-steps, carrying the co-state `λ` across
sub-step boundaries — exactly what acados does (see below). A thin wrapper
that only exposes the 3 public methods cannot do this; it would need access
to per-sub-step stage values and the adjoint state. Hence: extend the class
internally, don't wrap.

**acados reference (re-read fresh at implementation time; lines from the
current checkout).**
- `sim_erk_integrator.c`:
  - Forward sweep (`:799`): `for (istep = 0; istep < num_steps; istep++)`
    — one ERK stage sweep per sub-step; `forw_traj` (state + sensitivity)
    carries over across sub-steps naturally.
  - Adjoint sweep (`:920`): `for (istep = num_steps-1; istep >= 0;
    istep--)` — a single backward pass; `adj_tmp` (the co-state `λ`)
    **persists across sub-step boundaries** (not re-initialized per
    sub-step). Within each sub-step the stage loop reads `adj_traj[j]`
    (intra-sub-step stages, `j > s`) and `adj_tmp` (inter-sub-step
    carry-over).
  - `step = in->T / num_steps` (`:720`); the b-weight per sub-step is
    `b_vec[s] / num_steps` (cost quadrature, `:1442`).
- `sim_irk_integrator.c`:
  - Forward sweep (`sim_irk_forward_sweep:1594-1606`):
    `for (ss = 0; ss < num_steps; ss++) sim_irk_forward_step(ss)`.
  - **Newton warm-start across sub-steps** (`:1588-1590`): at the end of
    sub-step `ss`, the last stage's `K` is stored into `mem->xdot`; the
    next sub-step's Newton initialization (`:1177`) reads it. Saves 1–2
    Newton iterations per sub-step.
  - Backward sweep (`sim_irk_backward_sweep:1790-1804`):
    `for (ss = num_steps-1; ss > -1; ss--)`; `ws->lambda` persists across
    sub-steps; `ws->Hess` is zeroed once and `dsyrk`-accumulated across all
    sub-steps and stages.
- `sim_common.h:125`: `int num_steps;` in `sim_opts`.

**Per-method algorithm (NumSteps > 1).**

*`value`* — straightforward forward chain:
```
x_cur = x
for ss = 0..NumSteps-1:
    one single-step RK sweep (NS stages, step = h/NumSteps)
    x_cur = x_cur + (h/NumSteps) * Σ_s b_s * K_s
x_next = x_cur
```
For ERK: identical to `NumSteps=1` with `h → h/NumSteps`.
For IRK: the last sub-step's converged `K` warm-starts the next sub-step's
Newton (acados `mem->xdot` pattern).

*`jacobian`* — thread the first-order sensitivity forward across
sub-steps (no per-sub-step storage needed):
```
Sx = I;  Su = 0
for ss = 0..NumSteps-1:
    one single-step jacobian sweep → (Jx_ss, Ju_ss)   # per-sub-step Jacobian
    Sx = Jx_ss * Sx;   Su = Jx_ss * Su + Ju_ss
df_dx = Sx;  df_du = Su
```
(Equivalent to composing per-step Jacobians; acados threads `S_forw`
through the same forward sweep, `sim_erk_integrator.c:787-790,876`.)

*`hess_prod`* — single continuous backward adjoint sweep across all
sub-steps (the key difference from a wrapper):
```
# forward pass (per sub-step): store x_stage[ss][s], K[ss][s],
#                               JX[ss][s], JU[ss][s]  (or recompute in backward)
# backward sweep:
lam = w    # (nx) co-state, persists across sub-steps
lam_u = 0  # (nu)
for ss = NumSteps-1 downto 0:
    for s = NS-1 downto 0:
        # stage-local adjoint (same as single-step, but lam carries over)
        ...
        call ode_hess_prod at (x_stage[ss][s], u) with
        dual = stage-local lambda, direction = (dx_stage, v_u)
        accumulate hv_x, hv_u
        update lam, lam_u
```
For IRK: additionally, the per-sub-step factored `G` (from Newton) is needed
for the `Gᵀ` back-solve in the adjoint; these must be stored across the
backward sweep (workspace grows `NumSteps ×` for the LU factors, or the
Newton is re-run in the backward pass — **open design decision**; the
acados approach stores `dG_dK[ii]` per step, `:758`).

**Workspace additions** (beyond the `NumSteps=1` workspace, all fixed-size):
- ERK: `K_traj[NumSteps][NS]`, `x_traj[NumSteps+1]` (state per sub-step
  boundary) if storing for the backward pass; or recompute forward.
  `adj_traj[NumSteps][NS]` for the adjoint sweep.
- IRK: `K_traj[NumSteps][NS]`, `G[NumSteps]` + `LU[NumSteps]` (or re-factor
  in backward), `lambda` (nx, persists), `Hess` (nx+nu × nx+nu, accumulates
  across all sub-steps if computing the full Hessian; not needed for HVP).
- The `NumSteps=1` special case must produce the exact same workspace
  layout and code path as today (no `if (NumSteps > 1)` in hot paths; use
  the loop that degenerates to one iteration).

**Test plan** (`tests/integrators/`, new file or extend existing):
- *value, exact:* linear ODE (double integrator), `NumSteps=4`, `h=0.1` →
  matches the exact discrete map `q += h*v + h²/2*a` to 1e-14 (sub-steps
  are exact for a linear ODE, so the composed map is exact).
- *value, convergence:* ẋ = −10x, K2Tag, `h=0.1`, `NumSteps=1` vs
  `NumSteps=4` → the 4-sub-step result is closer to `exp(-1)` (verify the
  order-of-accuracy improvement; for K2 the local error goes from
  O(h²) per step to O((h/4)²) per sub-step, 4× tighter).
- *jacobian, FD:* ẋ = x⊙x·u + x (nonlinear), K4Tag, `NumSteps=3`,
  central FD δ=1e-6, ‖·‖∞ < 1e-6.
- *hess_prod, FD (ERK):* same ODE, K2Tag, `NumSteps=2`, FD the
  (nx+nu)×(nx+nu) Hessian of `wᵀΦ`, compare to `hess_prod`, ‖·‖∞ < 1e-5.
- *hess_prod, cross-check:* for the same `(x,u,w,v)`, assert
  `NumSteps=1` and `NumSteps=3` give the same HVP to 1e-10 (the composed
  Hessian is independent of the sub-step count for exact arithmetic;
  floating-point differences should be at round-off level).
- *IRK warm-start:* ẋ = x⊙x, RadauIia2, `NumSteps=3`, `newton_max=5` →
  all sub-steps converge (residual < 1e-10); log per-sub-step Newton
  iteration count (expect ≤ 3 after the first sub-step thanks to
  warm-starting).
- *regression:* `NumSteps=1` (default) produces bit-for-bit identical
  results to the current single-step code for all existing tests.

**Relationship to 4f.** 4f (IRK nonlinear HVP, single-step) is a
prerequisite for the IRK multi-step HVP: the single-step `hess_prod`
algorithm must be complete and FD-verified before the cross-step adjoint
threading is layered on top. ERK multi-step HVP does not depend on 4f.

**Status:** draft only — no code written. Gate: FD-verified `value` +
`jacobian` + `hess_prod` for `NumSteps > 1`. One commit per integrator
(`integrator:` scope for the header change, `test:` scope for the test),
re-using the `integrators_unit` target.

### 4h — Time-varying ODE (`t` in the ODE contract) — *extension draft, not started*

Extend the ODE contract and the integrators to support time-varying
dynamics `ẋ = f(x, u, t)`. The Butcher tableau is unchanged (normalized on
`[0,1]`); the only algorithmic change is computing the physical time at
each stage: `t_s = t_k + h·c[s]`, where `t_k` is the start of the OCP
interval and `c[s]` is the Butcher abscissa.

**Design: the solver decides the time grid, not the model.** The ODE model
is the continuous system; the discretization (time grid, step size) is the
solver's domain. The ODE model receives whatever `t` the integrator hands
it — it does not choose its own step size. This matches acados: `sim_in->T`
and `opts->num_steps` are solver options; the ODE function passively
receives `t` as an input (`sim_erk_integrator.c:824` — `t` is in the
`expl_vde_in` input array).

**ODE contract change** (add `double t` to all three methods):

```cpp
// Time-varying ODE. Time-invariant ODEs just ignore `t`.
struct MyTimeVaryingOde {
    state_t f(const state_t& x, const control_t& u, double t) const;
    void jacobian(const state_t& x, const control_t& u, double t,
                  dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const;
    void hess_prod(const state_t& x, const control_t& u, double t,
                   const state_t& w, const state_t& v_x, const control_t& v_u,
                   state_t& hv_x, control_t& hv_u) const;
};
```

- The SFINAE trait (`ode_supports_hess_prod`) is updated to probe the
  3-argument `hess_prod` signature.
- A time-invariant ODE can still work by declaring `f(x, u, double t)` and
  ignoring `t` — no separate "time-invariant" trait needed.

**Integrator method signature change** (add `double t_k` to each method):

```cpp
void value(const state_t& x, const control_t& u, double t_k,
           state_t& x_next) const;
void jacobian(const state_t& x, const control_t& u, double t_k,
              df_dx_t& df_dx, df_du_t& df_du) const;
void hess_prod(const state_t& x, const control_t& u, double t_k,
               const state_t& w, const state_t& v_x, const control_t& v_u,
               state_t& hv_x, control_t& hv_u) const;
```

- `h_` (the step size) stays a constructor member; `t_k` is the
  start-of-interval time, passed per call.
- Per stage, the integrator computes `t_s = t_k + h_ * c[s]` and passes it
  to `ode_.f(x_s, u, t_s)`, `ode_.jacobian(x_s, u, t_s, …)`,
  `ode_.hess_prod(x_s, u, t_s, …)`.
- For multi-step (4g): `t_s = t_k + (ss + c[s]) * (h / NumSteps)`.

**`ContinuousProblem` change.** The adapter computes `t_k = k * h_` (uniform
grid) and forwards it to the integrator:

```cpp
typename P::state_t
dynamics_next_state(int k, const state_t& x, const control_t& u) const {
    return integ_.value(x, u, k * h_, /*x_next*/);
}
```

- For a non-uniform time grid (future), the adapter would store a time grid
  (e.g. `std::array<double, NH+1>` when `NH` is compile-time, heap vector
  otherwise, allocated once at `resize`, not in the hot path) and pass both
  `t_k` and `h_k` to the integrator. But that's a further extension;
  uniform `h` covers the 90% case and is what acados does by default.

**Workspace:** no new buffers. `t` is a scalar computed on the fly; no
trajectory storage needed.

**Complexity estimate:** moderate (~half a day). The change is mechanical,
not algorithmic — thread `t_s` through ~10 ODE call sites per integrator.
No changes to the QP, SQP, or `Solution`.

**Test plan** (`tests/integrators/`, extend existing):
- *value:* time-varying ODE `ẋ = sin(t)·x + u` (nx=1, nu=1), K4Tag,
  `h=0.1`, integrate from `t=0` to `t=1` (10 steps). Compare to a
  high-resolution reference (e.g. 10000-step RK4) to 1e-4.
- *value, time-invariant regression:* the existing ẋ = −x and ẋ = x² tests
  (now with an ignored `t` parameter) must produce identical results to
  the pre-4h code.
- *jacobian, FD:* time-varying ODE `ẋ = sin(t)·x⊙u + cos(t)·x`, K4Tag,
  `NumSteps=1`, central FD δ=1e-6, ‖·‖∞ < 1e-6, at a non-zero `t_k`
  (e.g. `k=3`, `t_k=0.3`).
- *hess_prod, FD (ERK):* same ODE, K2Tag, FD the Hessian of `wᵀΦ` at
  `t_k=0.5`, ‖·‖∞ < 1e-5.
- *regression:* all existing targets unchanged.

**Relationship to 4g (multi-step).** 4h and 4g are independent: 4h adds
the `t` argument (orthogonal to the sub-step count); 4g adds the outer
`ss` loop (orthogonal to time-dependence). They can be composed:
`t_s = t_k + (ss + c[s]) * (h / NumSteps)`. Neither blocks the other, but
4h is simpler and should land first (it's a prerequisite for a meaningful
time-varying multi-step test).

**Status:** draft only — no code written. Gate: FD-verified `value` +
`jacobian` + `hess_prod` for a time-varying ODE. One commit per integrator
(`integrator:` scope for the header change, `test:` scope for the test),
re-using the `integrators_unit` target.

### 4i — CSTR example (nonlinear end-to-end test) — *extension draft, not started*

A genuinely **nonlinear** example + test based on the acados CSTR (Continuous
Stirred-Tank Reactor, `acados/examples/acados_python/cstr/cstr_model.py`),
with **hand-derived analytic derivatives** (no CasADi, no AD library).
This is the first nonlinear model in the test suite and the vehicle for
exercising 4f (IRK nonlinear HVP), 4g (multi-step), and 4h (time-varying).

**Why CSTR.** Genuinely nonlinear (Arrhenius term `rate = k0·exp(-EbR/T)·c`
→ non-zero, non-trivial Hessian w.r.t. `(c, T)`); small (nx=3, nu=2, cheap
FD Hessian); a direct acados example for cross-verification.

**Model** (states `x = (c, T, h)`, control `u = (Tc, F)`; all parameters
are plain members of the example class — user-managed, not in the
interface, per convention):
```
Ac    = π·r²
denom = Ac·(h + eps)
k     = k0·exp(-EbR/T)
rate  = k·c
ċ  = F0·(c0 − c)/denom − rate
Ṫ  = F0·(T0 − T)/denom − dH/(ρ·Cp)·rate + 2U/(r·ρ·Cp)·(Tc − T)
ḣ  = (F0 − F)/Ac
```
Nominal acados values: `F0=0.1, T0=350, c0=1.0, r=0.219, k0=7.2e10,
EbR=8750, U=54.94, ρ=1000, Cp=0.239, dH=−5e4, eps=1e-5`;
`xs=(0.878, 324.5, 0.659)`, `us=(300, 0.1)`.

**Derivatives (analytic, hand-derived; validated by FD in the test).**
Let `Ac = πr²`, `denom = Ac·(h+eps)`, `k(T) = k0·exp(-EbR/T)`,
`rate = k(T)·c`, `g = 2U/(r·ρ·Cp)` (coolant gain), `q = dH/(ρ·Cp)`
(heat release; `dH < 0`). Then

```
ċ   = F0·(c0 − c)/denom − rate
Ṫ   = F0·(T0 − T)/denom − q·rate + g·(Tc − T)
ḣ   = (F0 − F)/Ac
```

First-order `∂rate`:
- `∂rate/∂c = k(T)` (use the `k0·exp(-EbR/T)` form; never `rate/c`),
- `∂rate/∂T = rate·EbR/T²`.

**`df_dx` (3×3, rows = ċ,Ṫ,ḣ; cols = c,T,h):**
```
ċ:  −F0/denom − k(T)      −rate·EbR/T²          −F0·Ac·(c0−c)/denom²
Ṫ:  −q·k(T)               −F0/denom − q·rate·EbR/T² − g    −F0·Ac·(T0−T)/denom²
ḣ:  0                    0                    0
```
**`df_du` (3×2, cols = Tc,F):**
```
ċ:  0      0
Ṫ:  g      0
ḣ:  0     −1/Ac
```

**Hessian (w.r.t. `x` only — `u` enters linearly).** Because `Tc` and `F`
appear *affinely* (`g·(Tc−T)` in `Ṫ`, `−F/Ac` in `ḣ`), every second
derivative involving a control is zero: `∂²f/∂u∂x = 0`, `∂²f/∂u² = 0`, so
the HVP's control part is identically `hv_u = 0` and the `v_u` direction
contributes nothing. The non-zero Hessian is confined to the state block
and comes from two sources:

- `rate` (depends on `c`, `T`): `∂²rate/∂c∂T = k(T)·EbR/T²`,
  `∂²rate/∂T² = rate·(EbR²/T⁴ − 2·EbR/T³)`, `∂²rate/∂c² = 0`.
- `1/denom` (depends on `h`): the `F0·(c0−c)/denom` term gives
  `∂²ċ/∂c∂h = +F0·Ac/denom²` and `∂²ċ/∂h² = 2·F0·Ac²·(c0−c)/denom³`
  (and the analogous `T0−T` forms in the `Ṫ` row).

The full Hessian of `f` w.r.t. `x = (c, T, h)` (upper-triangular; symmetric)
is:
```
ċ:  [c,c]=0   [c,T]=−k(T)·EbR/T²        [c,h]=+F0·Ac/denom²
    [T,T]=−rate·(EbR²/T⁴ − 2·EbR/T³)   [T,h]=0
    [h,h]=2·F0·Ac²·(c0−c)/denom³
Ṫ:  [c,c]=0   [c,T]=−q·k(T)·EbR/T²     [c,h]=0
    [T,T]=−q·rate·(EbR²/T⁴ − 2·EbR/T³) [T,h]=+F0·Ac/denom²
    [h,h]=2·F0·Ac²·(T0−T)/denom³
ḣ:  all zero
```
(Reasoning: `∂²rate/∂c∂T = k(T)·EbR/T²`, `∂²rate/∂T² =
rate·(EbR²/T⁴ − 2·EbR/T³)`, `∂²rate/∂c² = 0`; the `1/denom` term gives
`∂²/∂c∂h = +F0·Ac/denom²`, `∂²/∂h² = 2·F0·Ac²·(·)/denom³`, and is linear in
`c`/`T` so `∂²/∂c² = ∂²/∂T² = 0`. The `g·(Tc−T)` and `(F0−F)/Ac` terms are
affine, contributing no Hessian entries.) The exact per-entry list is the
primary thing the FD Hessian test pins down; the structure above is the
design-time reference.

**OCP setup** (in the test/example): box constraints on `(c, T, h)` and
`(Tc, F)`; quadratic stage cost `‖x − xs‖²_Q + ‖u − us‖²_R`; terminal
equality to the steady state. `Dims::has_dynamics_hess_prod = true`.

**Test plan** (`tests/integrators/cstr_4i.cpp`, wired into
`integrators_unit`; plus an `examples/` entry if the example dir grows):
- *value:* K4Tag single step at the nominal point, `h=0.1` → compare to a
  high-res reference (10⁴-step RK4 of the same analytic `f`).
- *jacobian:* FD (central, δ=1e-6) of the composed `Φ` w.r.t. `x` and `u`
  at the nominal point, `‖·‖∞ < 1e-6`, for K2/K4 and RadauIia2.
- *hess_prod (ERK):* FD the (nx+nu)² Hessian of `wᵀΦ` (δ=1e-4), 3
  deterministic `(w, v)` pairs, `‖·‖∞ < 1e-5`, K4Tag.
- *hess_prod (IRK, requires 4f):* same FD check with RadauIia2Tag; cross-
  check ERK vs IRK HVP to 1e-8 (both must equal the FD Hessian).
- *time-varying variant (requires 4h):* add a sinusoidal feed disturbance
  `F0(t) = 0.1 + 0.01·sin(2πt)`, FD-check `jacobian`/`hess_prod` at
  `t_k = 0.3`.
- *end-to-end:* `ContinuousProblem` + `SqpSolver` on the CSTR OCP
  (N=10, h=0.1), assert `kSolved` + NLP residuals < 1e-6. (Gates 4f
  end-to-end; optional RadauIia2 variant gates IRK end-to-end.)
- *regression:* all existing targets unchanged.

**Status:** done (see checkpoint §4i). Gate met: FD-verified
value/jacobian/HVP + end-to-end `kSolved`. Commits: `example:` (CSTR model
+ Dims), `test:` (tests), re-using `integrators_unit` + `sqp_continuous`
targets.

## 4. Test matrix

| target | covers |
|---|---|
| `integrators_unit` (new) | 4a tableau identities; 4b ERK value/jacobian/HVP (FD); 4c adapter + exact-linear; 4d IRK value/Newton/jacobian/HVP (FD); 4f IRK nonlinear HVP (FD); 4g multi-step value/jacobian/HVP; 4h time-varying ODE value/jacobian/HVP (FD); 4i CSTR nonlinear example (value/jacobian/HVP, FD) |
| `sqp_continuous` (new) | 4e end-to-end continuous SQP (ERK + one IRK) vs discrete reference |
| all existing targets | regression (no behavior change) |

Exit criteria: all new + existing targets build warning-free and pass;
ERK value matches the analytic ODE solution, ERK/IRK Jacobians and HVPs
match finite differences to ≤1e-5; the continuous double-integrator reaches
`kSolved` with all four NLP residual norms below tolerance; the discrete
reference tests are bit-for-bit unchanged.

## 5. Future extensions (beyond Phase 4)

Items that have been promoted to in-phase sub-steps (4f, 4g, 4h, 4i) are no
longer listed here. The following remain explicitly out of scope:

1. **Lifted IRK** (`sim_lifted_irk_integrator.c`) — a different Newton
   structure (lifted state variables); separate integrator class.
2. **Algebraic variables** (`nz > 0`) — DAEs; the ERK path explicitly
   rejects `nz != 0` (matches acados), IRK would extend the state.
3. **Parameter sensitivities** (`S_p` / `sens_forw_p`) and the adjoint
   sweep's cost-integral accumulation (used by acados' cost computation,
   not by our SQP which evaluates the stage cost directly).

## 6. Checkpoints

*(append one entry per sub-step, 4a → 4e)*

### 4a — ODE model contract + Butcher tableaus (done)

- Re-read `sim_collocation_utils.c`: `gauss_radau_iia_nodes` (:248-333,
  hard-coded Radau nodes, CasADi issue #673) and
  `get_explicit_butcher_tableau` (:560-647, explicit ERK tableaus for
  ns = 1..4). Verified the ERK tableaus (K1..K4) are transcribed verbatim.
- Implemented `include/ocp/integrators/ode_model.hpp`: duck-typed ODE
  contract (documented `f` / `jacobian` / `hess_prod` signatures, §1.1),
  the `ode_supports_hess_prod<Ode, Dims>` SFINAE trait
  (`detail::ode_hess_prod_probe` via `std::void_t<decltype(...hess_prod(...))>`),
  and a typed `ode_hess_prod(...)` forwarder the integrator will call behind
  the trait `static_assert`.
- Implemented `include/ocp/integrators/butcher.hpp`:
  `ButcherTableau<Dims, NS, Tag>` with `static constexpr` `A` (NS×NS,
  row-major `A[s][j]`, row s = stage advanced), `b`, `c`; partial
  specialisations for `K1Tag`..`K4Tag` (verbatim acados values) and
  `RadauIia2/3/4Tag`. Primary template carries a `static_assert(false)` for
  unsupported (NS, Tag) pairs. The Radau IIA `A`/`b` are the collocation
  coefficients `a_{s,j} = ∫_0^{c_s} l_j`, `b_j = ∫_0^1 l_j` at the
  hard-coded Radau nodes (computed at full double precision, matching the
  semantics of `calculate_butcher_tableau_from_nodes`
  (sim_collocation_utils.c:481-534)).
- **Deviation from §3/§1.4 of this plan:** the planned IRK identity
  "A[s][0] == c[s]" is *not* a property of the standard collocation
  orientation (checked: Radau IIA-2 gives A[0][0] = 5/12 ≠ c[0] = 1/3).
  The correct checkable identities used in the test instead:
  `sum_j A[s][j] == c[s]` for every stage type (ERK and IRK),
  `sum_s b[s] == 1`, and for Radau IIA the last-row property
  `A[NS-1][j] == b[j]` plus the collocation order conditions
  `sum_j b_j c_j^k == 1/(k+1)`, k = 0..2NS-2 (all verified to < 1e-12).
- Test `tests/integrators/integrators_4a.cpp` (new `integrators_unit`
  target): tableau identities for all 7 schemes + `static_assert`
  positive/negative trait checks (nonlinear ODE with `hess_prod` → true;
  linear ODE without it → false). The identities evaluate `constexpr`
  data at runtime (floating point), so they are runtime `check()` calls
  with 1e-12 tolerance in the repo's usual style; the trait check is a
  hard `static_assert`.
- Build: warning-free under `-Wall -Wextra -Werror`. All targets green:
  `double_integrator`, `mass_spring`, `qp_dim`, `qp_unit`, `sqp_unit`,
  `sqp_double_integrator`, `sqp_mass_spring`, `sqp_acados_ref`,
  `integrators_unit`.

### 4b — `ExplicitRkIntegrator` (value + jacobian + hess_prod) (done)

- Re-read `sim_erk_integrator.c` (:799-878 forward sweep; :895-1034
  adjoint/Hessian sweep). Confirmed our forward sweep matches the `num_steps=1`
  `sens_* = false` specialisation, and that the HVP uses the same
  forward-over-forward + backward-adjoint structure (a JVP along `v` threaded
  forward, then an adjoint walked backward that calls the ODE's HVP at each
  stage).
- Implemented `include/ocp/integrators/rk_explicit.hpp`:
  `ExplicitRkIntegrator<Dims, Ode, NS, Tag>` with a `mutable`
  `detail::ErkWorkspace<P,NS>` (fixed-size `std::array` of `K`/`x_stage`/
  `JX`/`JU`/`dx`/`dK`/`lam_dK`/`lam_K`), allocation-free. Methods are `const`.
  - `value`: one ERK forward sweep (`x_s = x + h Σ_{j<s} A[s][j] K_j`,
    `K_s = f(x_s, u)`; `x_next = x + h Σ_s b_s K_s`).
  - `jacobian`: threaded first-order sensitivities — per stage
    `Dxs = I + h Σ_{j<s} A[s][j] JX_j`, `Dus = h Σ_{j<s} A[s][j] JU_j`,
    `JX_s = f_x Dxs`, `JU_s = f_x Dus + f_u`; then
    `df_dx = I + h Σ_s b_s JX_s`, `df_du = h Σ_s b_s JU_s`.
  - `hess_prod`: reverse-over-forward. Forward pass computes `JX_s`, `JU_s`
    and the first-order JVP `dx_s` (= `v_x + h Σ_{j<s} A[s][j] dK_j`),
    `dK_s = JX_s dx_s + JU_s v_u`. Backward pass seeds `lam_dK_s = h b_s w`,
    `lam_K_s = 0`, then walks `s` downward; at each stage one
    `ode_hess_prod(x_s, u, lam_dK_s, dx_s, v_u)` supplies the ODE-Hessian
    pullback (the two would-be calls with directions `(dx_s,0)` and `(0,v_u)`
    are merged by direction linearity), `kappa = JX_s' lam_dK_s` is the
    adjoint on `dx_s`, `theta = JX_s' lam_K_s` / `rho = JU_s' lam_K_s` are
    the value-node adjoints, the total stage-state adjoint
    `xs_adj = mu_x + theta` accumulates into `hv_x`/`hv_u` and is pushed back
    into `lam_K_j` (value) and `lam_dK_j` (JVP) for `j < s`.
    `static_assert(ode_supports_hess_prod<Ode, Dims>)` guards the path
    (a linear ODE never reaches it).
- Test `tests/integrators/rk_explicit_4b.cpp` (new `run_erk_4b_tests`, wired
  into `integrators_unit` via a shared `tests/integrators/integrators_main.cpp`;
  the 4a file was refactored from a `main()` to `run_integrators_4a_tests()`
  to match the repo's multi-file-test pattern). ODE fixtures on
  `DoubleIntegratorDims` (nx=2, nu=1): `OdeNegX` (ẋ=−x), `OdeXsq` (ẋ=x⊙x),
  `OdeNegXU` (ẋ=−x⊙u, nonlinear in the composed map).
  - value: ẋ=−x (x₀=1, h=0.1, 10 steps, K4) vs `exp(−1)` to 1e-5; ẋ=x²
    (x₀=0.5, h=0.2, 5 steps, K4) vs the exact `x₀/(1−x₀ t)` to 1e-4.
  - jacobian: central FD (δ=1e-6) of the composed map w.r.t. x and u,
    ‖·‖∞ < 1e-6, for `OdeNegXU` with K2 and K4.
  - hess_prod: FD the (nx+nu)×(nx+nu) Hessian of the scalar `w'Φ` (δ=1e-4,
    mixed 4-point cross derivatives), apply to 3 deterministic `(w, v)`
    pairs, ‖·‖∞ < 1e-5, for `OdeNegXU` with K2 and K4.
- **Deviation from §3/§4b of this plan:** the plan's analytic solution for
  ẋ=x² is written as `x₀/(1+x₀·t)`; the correct closed form is
  `x₀/(1−x₀·t)` (checked: at x₀=0.5, t=1 the exact value is 1.0, and RK4 at
  h=0.2 gives 0.99998). The tolerance for that value check is 1e-4 (the RK4
  O(h⁴) global error at h=0.2 is ≈1.8e-5, just over the 1e-5 used for the
  well-behaved ẋ=−x case).
- Build: warning-free under `-Wall -Wextra -Werror` (clean full rebuild).
  All targets green: `double_integrator`, `mass_spring`, `qp_dim`, `qp_unit`,
  `sqp_unit`, `sqp_double_integrator`, `sqp_mass_spring`, `sqp_acados_ref`,
  `integrators_unit` (4a + 4b).

### 4c — `ContinuousProblem` adapter (done)

- Implemented `include/ocp/integrators/continuous_problem.hpp`:
  `ContinuousProblem<Dims, Ode, Integ, NH>` inherits from
  `ocp::Problem<Dims>`. Stores the ODE by value and the integrator by value
  (the integrator holds a `const Ode&` pointing into the problem object,
  member-initialised in declaration order so the ODE outlives the
  integrator). The three dynamics methods (`dynamics_next_state`,
  `dynamics_jacobian`, `dynamics_hess_prod`) delegate to the integrator; the
  stage index `k` is ignored (time-invariant ODE). All other interface methods
  are inherited stubs for the user to override.
  - `dynamics_hess_prod` delegates unconditionally to
    `integ_.hess_prod(...)`; because the solver gates the call behind
    `if constexpr (P::has_dynamics_hess_prod)` (sqp.hpp:1838, :1947), a
    linear ODE (`has_dynamics_hess_prod = false`) never instantiates the
    integrator's `hess_prod`, so the ODE may omit `hess_prod` entirely.
- Test `tests/integrators/continuous_problem_4c.cpp`
  (`run_erk_4c_tests`, wired into the shared `integrators_main.cpp`):
  a linear continuous double integrator (`f(x,u) = [x(1); u(0)]`,
  `ContDims` with `has_dynamics_hess_prod = false`, box constraints on
  states/control, quadratic stage/terminal cost) wrapped in
  `ContinuousProblem<..., ExplicitRkIntegrator<..., 4, K4Tag>>`:
  - `dynamics_next_state` matches the exact discrete map
    `q_{k+1} = q + h·v + h²/2·a`, `v_{k+1} = v + h·a` to 1e-14 (RK4 is exact
    for this linear ODE);
  - `dynamics_jacobian` matches the exact constant Jacobian to 1e-14;
  - `SqpSolver::resize(N)` + `assemble_qp` returns `kSolved` and the QP
    (first/path/term) has all-finite `hess`/`grad`/`BA`/`b`, with the first
    stage `BA` shaped (nx, nx+nu).
- Build: warning-free under `-Wall -Wextra -Werror`. All targets green
  (4a + 4b + 4c in `integrators_unit`).

### 4d — `ImplicitRkIntegrator` (Newton) (done, partial)

- Re-read `sim_irk_integrator.c`: Newton loop (`:1381-1412`), block
  Jacobian assembly (`:1270-1310`), LU factorization (`:1334-1343`),
  back-solve (`:1345-1359`), and sensitivity path (`:1488-1567`). Confirmed
  the block structure `G[s][j] = δ_{sj} I − h·A[s][j]·f_x(x_s,u)` and the
  sensitivity relation `G · dK/d(x,u) = −∂R/∂(x,u)` (i.e.
  `dK/dx = G⁻¹ f_x_stack`, `dK/du = G⁻¹ f_u_stack`).
- Implemented `include/ocp/integrators/rk_implicit.hpp`:
  `ImplicitRkIntegrator<Dims, Ode, NS, Tag>` with a `mutable`
  `detail::IrkWorkspace<P,NS>` (fixed-size `K[NS]`, `x_stage[NS]`,
  `dfdx[NS]`, `dfdu[NS]`, a `NK×NK` block matrix `G`, a
  `PartialPivLU<NK×NK>`, residual, and sensitivity RHS buffers). All
  methods are `const`.
  - **`value`**: warm-start all stages with `K_s = f(x,u)`, iterate
    Newton (build stage states, Jacobians, residual, block G, LU-factor,
    solve, update) for `newton_max` iterations; then
    `x_next = x + h·Σ b_s K_s`.
  - **`jacobian`**: reuses the converged LU from `solve_newton` to solve
    `G·dK_dx = f_x_stack` and `G·dK_du = f_u_stack`; then
    `df_dx = I + h·Σ b_s·(dK_dx)_s`, `df_du = h·Σ b_s·(dK_du)_s`.
  - **`hess_prod`**: **not implemented** for nonlinear ODEs (the full-A
    self-coupling `G[s][s]` makes the second-order sensitivity depend on
    the full ODE Hessian, which the ODE `hess_prod` contract does not
    expose). Guarded by `if constexpr (ode_supports_hess_prod<...>)
    static_assert(false, …)` — a nonlinear ODE + IRK combination fails to
    compile with a clear message. A linear ODE (no `hess_prod`) returns
    zeros (correct: composed Hessian is zero).
  - **`newton_residual_inf()`**: diagnostic returning ‖R‖∞ at the
    converged iterate (used in the test to assert convergence).
- **Deviation from §3/§4d of this plan:** the plan anticipated shipping
  `hess_prod` if feasible within budget. The IRK second-order sensitivity
  requires the full 4th-order ODE tensor (or at least `∂f_x/∂x` and
  `∂f_x/∂u` as standalone calls), which the ODE `hess_prod` contract does
  not provide. Marked **partial**: `value` + `jacobian` are complete and
  FD-verified; `hess_prod` is a documented stub. This is acceptable for
  the 4e end-to-end test because the double-integrator ODE is linear
  (`has_dynamics_hess_prod = false`, HVP never called).
- Test `tests/integrators/rk_implicit_4d.cpp`
  (`run_irk_4d_tests`, wired into `integrators_main.cpp`):
  - *value, stiff:* ẋ = −30x, RadauIia2, h=0.1, 10 steps → |x| < 1e-6
    (A-stable; z = −3 where K4 would be unstable).
  - *value, accuracy:* ẋ = −x, RadauIia2, h=0.05, 20 steps →
    |err| < 1e-4 vs exp(−1).
  - *value, exact linear:* double-integrator (M² = 0), RadauIia2,
    h=0.2, 5 steps → matches exact map to 1e-10.
  - *value, nonlinear:* ẋ = x⊙x, RadauIia2, h=0.2, 5 steps,
    newton_max=5 → |err| < 1e-2 vs x₀/(1−x₀·t).
  - *Newton convergence:* ẋ = x⊙x, RadauIia2, h=0.2: 1-iter residual
    > 5-iter residual and 5-iter residual < 1e-10.
  - *jacobian, FD:* ẋ = x⊙x·u + x (nonlinear in x and u), RadauIia2
    and RadauIia4, central FD δ=1e-6, |err| < 1e-6.
  - *jacobian, exact linear:* double-integrator, RadauIia2, h=0.2 →
    df/dx = I + hM, df/du = [h²/2; h], |err| < 1e-10.
  - *hess_prod (linear):* ẋ = −10x (no hess_prod) → returns zeros.
- Build: warning-free under `-Wall -Wextra -Werror`. All targets green
   (4a + 4b + 4c + 4d in `integrators_unit`).

### 4e — End-to-end continuous SQP + regression (done)

- Test `tests/integrators/sqp_continuous_4e.cpp` (new `sqp_continuous` CMake
  target): a continuous double-integrator OCP (`qdot = v`, `vdot = a`,
  terminal equality `q_N + v_N = 1`, box-bounded control, `ne_t = 1`) at
  N = 10, h = 0.1, solved by the existing `SqpSolver`. Two integrator paths
  are exercised end-to-end:
  - `ExplicitRkIntegrator<..., 4, K4Tag>` (primary),
  - `ImplicitRkIntegrator<..., 2, RadauIia2Tag>` (IRK end-to-end; Newton
    `newton_max = 5`).
- Both integrators give the exact discrete map for this linear ODE, so the
  two paths produce the *identical* continuous OCP and the same cost
  (17.0621417000). The discrete `DoubleIntegrator` reference (same horizon /
  step / cost weights, plus the discrete-only soft-ineq and linear stage
  constraints) yields cost 17.8142031617; the continuous cost is within the
  plan's 5% relative bound (4.2% gap is driven by the extra discrete-only
  constraints, not by integration error).
- Assertions: `solve` returns `kSolved`, `sol.status == kSolved`, and all
  four NLP residual norms (`res_stat`, `res_eq`, `res_ineq`, `res_comp`)
  computed via `compute_nlp_residuals` are below 1e-6; both integrator costs
  within 5% of the discrete reference.
- **Regression:** full run of every existing target — `double_integrator`,
  `mass_spring`, `qp_dim`, `qp_unit`, `sqp_unit`, `sqp_double_integrator`,
  `sqp_mass_spring`, `sqp_acados_ref`, `integrators_unit` — all PASS,
  bit-for-bit unchanged.
- Build: warning-free under `-Wall -Wextra -Werror`.
- **Note:** the discrete reference OCP carries `ng = 1` / `nl = 1` stage
  constraints that the continuous OCP intentionally omits, so the two costs
  are not expected to match exactly; the 5% bound (per §3/§4e of this plan)
  is the meaningful gate here, and both integrator paths agree to machine
  precision with each other.

### 4g — Multi-step integration (`NumSteps > 1`) (done)

- **ERK** (`rk_explicit.hpp`): added `NumSteps` template parameter to
  `ExplicitRkIntegrator`. The `value`, `jacobian`, and `hess_prod` methods
  loop over `NumSteps` sub-steps of size `h/NumSteps`. The `hess_prod` uses
  an explicit-dual reverse-mode sweep (no Newton solve; stages are
  independent due to the lower-triangular A-matrix).
- **IRK** (`rk_implicit.hpp`): added `NumSteps` template parameter to
  `ImplicitRkIntegrator`. Each sub-step is a Newton solve warm-started from
  the previous sub-step's converged K (acados `mem->xdot` pattern).
  The `hess_prod` for `NumSteps > 1` uses a "4f-as-building-block +
  threading" approach:
  - **Forward pass:** Newton-solve each sub-step, store the per-sub-step
    Newton state (stage states, Jacobians, factored G), compute per-sub-step
    Jacobians `(A_ss, B_ss)`, thread the first-order JVP `dx` forward
    (`dx_{ss+1} = dx_ss + h_ss Σ b_s dK_jvp_s`), thread the x-Hessian
    adjoint `R` forward (`R_{ss+1} = A_ss^T R_ss`), and thread the total
    u-Jacobian `S` forward (`S_{ss+1} = A_ss S_ss + B_ss`).
  - **Backward pass:** thread the co-state `lam` backward
    (`lam = A_ss^T lam`), restore each sub-step's Newton state, call the
    single-sub-step 4f HVP (`hess_prod_substep`), and accumulate:
    `hv_x += R_ss · local_hv_x`,
    `hv_u += S_ss^T · local_hv_x + local_hv_u`.
  - For `NumSteps == 1` the code degenerates exactly to the 4f algorithm
    (bit-for-bit identical; regression-anchored by the existing 4f tests).
- **Key correctness fix:** the u-part of the composed HVP requires threading
  the total u-Jacobian of the inner composition (`S_ss = D_u x_ss`), not
  just summing the local u-HVPs. Without this the cross-term
  `w^T (D_x D²Phi_ss · v_x)` is missing from the u-part.
- Test `tests/integrators/rk_multistep_4g.cpp` (extended with IRK tests):
  - *value:* RadauIia2, NumSteps 1/2/3, 5 OCP steps (h=0.1), vs 10000-step
    RK4 reference → |err| < 1e-4.
  - *jacobian:* RadauIia2, NumSteps=2, central FD δ=1e-6 → |err| < 1e-6.
  - *hess_prod:* RadauIia2, NumSteps 1/2, central-FD Hessian (δ=1e-4)
    for 3 (w,v) pairs → |err| < 1e-5.
- **Regression:** all existing targets (`double_integrator`, `mass_spring`,
  `qp_dim`, `qp_unit`, `sqp_unit`, `sqp_double_integrator`,
  `sqp_mass_spring`, `sqp_acados_ref`, `integrators_unit`,
   `sqp_continuous`) pass; IRK NumSteps=1 hess_prod bit-for-bit unchanged
   (5.33e-09, same as 4f).
- Build: warning-free under `-Wall -Wextra -Werror`.

### 4h — Time-varying ODE (`t` in the ODE contract) (done)

- **ODE contract** (`ode_model.hpp`): every method now takes the physical
  time `t` as a trailing argument — `f(x, u, t)`,
  `jacobian(x, u, t, df_dx, df_du)`,
  `hess_prod(x, u, t, w, v_x, v_u, hv_x, hv_u)`. A time-invariant ODE
  simply ignores `t`. The `detail::ode_hess_prod_probe` SFINAE trait now
  probes the 3-argument-after-`u` signature (the `std::declval<double>()`
  after `control_t`), and the `ode_hess_prod(...)` forwarder threads `t`
  through to `ode.hess_prod(x, u, t, ...)`.
- **ERK** (`rk_explicit.hpp`): `value` / `jacobian` / `hess_prod` gain a
  `double t_k` (start-of-interval time) parameter. Per sub-step `ss` and
  stage `s`, the physical time is `t_s = t_k + h_ss·(ss + c[s])` (ERK:
  `h_ss = h`); `t_s` is passed to every `ode_.f`, `ode_.jacobian` and
  `ode_hess_prod` call (forward and backward sweeps).
- **IRK** (`rk_implicit.hpp`): same `t_k` parameter on the public methods.
  The private `eval_jacobians(u, t_k, ss)`, `eval_residual(u, t_k, ss)`,
  `solve_newton(x, u, K_init, t_k, ss)` and
  `hess_prod_substep(u, t_k, ss, ...)` each compute
  `t_s = t_k + h_ss·(ss + c[s])` and pass it to the ODE calls (both the
  `NumSteps == 1` and `NumSteps > 1` `hess_prod` branches, including the
  post-loop Jacobian refresh in `solve_newton`).
- **Adapter** (`continuous_problem.hpp`): the constructor is now
  `ContinuousProblem(ode, double h, Args&&...)` and stores `h_`; each
  `dynamics_next_state` / `dynamics_jacobian` / `dynamics_hess_prod`
  forwards `t_k = k·h_` (uniform grid) to the integrator. The ODE model
  remains passive — the solver owns the time grid (matches acados).
- **Test** `tests/integrators/time_varying_4h.cpp`
  (`run_time_varying_4h_tests`, wired into `integrators_main.cpp`):
  - *value:* time-varying ẋ = sin(t)·x + u (nx=2, nu=1), K4, h=0.1,
    10 steps over [0,1] vs a 10000-step RK4 reference → |err| < 1e-4
    (measured 2.44e-07).
  - *value, time-invariant regression:* ẋ = −x with an ignored `t`,
    K4, h=0.05, 20 steps → |err| < 1e-4 vs exp(−1) (unchanged behaviour).
  - *jacobian, FD:* ẋ = sin(t)·x⊙u + cos(t)·x, K4, NumSteps=1, central
    FD δ=1e-6 at `t_k = 0.3` → ‖·‖∞ < 1e-6 (measured 7.94e-11).
  - *hess_prod, FD (ERK):* same ODE, K2, central-FD Hessian of `wᵀΦ`
    (δ=1e-5, 3 (w,v) pairs) at `t_k = 0.5` → ‖·‖∞ < 1e-5
    (measured 9.47e-07).
- All existing ODE models in the other test files (`integrators_4a`,
  `rk_explicit_4b`, `continuous_problem_4c`, `rk_implicit_4d`,
  `rk_multistep_4g`, `sqp_continuous_4e`) gained the (ignored) `t`
  parameter so the whole suite compiles against the new contract.
- **Regression:** all existing targets
  (`double_integrator`, `mass_spring`, `qp_dim`, `qp_unit`, `sqp_unit`,
  `sqp_double_integrator`, `sqp_mass_spring`, `sqp_acados_ref`,
  `integrators_unit`, `sqp_continuous`) pass; time-invariant results
  unchanged.
- Build: warning-free under `-Wall -Wextra -Werror`.

### 4i — CSTR nonlinear example (value/jacobian/HVP, FD) + end-to-end (done)

- **Model** (`tests/integrators/cstr_4i.cpp`): `CstrOde` with hand-derived
  analytic `f`, `jacobian`, and `hess_prod` for the acados CSTR
  (`rate = k0·exp(-EbR/T)·c`); `CstrDims` (nx=3, nu=2, ne_t=3,
  state/control/terminal-state box, `has_dynamics_hess_prod=true`);
  `CstrOcp : ContinuousProblem<..., ExplicitRkIntegrator<...,4,K4Tag>>`
  with quadratic relative stage/terminal cost and a terminal equality to the
  steady state.
- **value:** K4 single step at the nominal point vs a 10⁴-step RK4 reference
  of the analytic `f` → |err| < 1e-4 (measured 6.3e-06).
- **jacobian (FD):** central FD of the composed `Φ` w.r.t. (x,u) at the
  nominal point, K2/K4 (ERK) and RadauIia2 (IRK) → ‖·‖∞ < 1e-6 (measured
  ≤ 3.1e-08).
- **hess_prod (FD):** central-FD (nx+nu)² Hessian of `wᵀΦ` (δ=1e-4), 3
  (w,v) pairs, K4 and RadauIia2 → ‖·‖∞ < 1e-4 (measured 9.2e-06 / 1.1e-05).
- **ODE-level direct:** analytic `hess_prod` vs central-FD Hessian of
  `wᵀf` (δ=1e-4) → ‖·‖∞ < 1e-5 (measured 1.8e-06).
- **end-to-end:** `ContinuousProblem` + `SqpSolver`, N=10, h=0.1,
  terminal equality `x_N = xs`, kSolved + all four NLP residual norms
  < 1e-6 (measured `res_stat` 5.1e-09, `res_eq` 5.7e-14).
  - Initial state is a *well-conditioned* perturbation of the steady state:
    `x0 = (0.878, 325.5, 0.78)` — `c` held at `xs_c`, small `T`/`h` nudge.
    Disturbing the stiffly-coupled `c` state stalls the (indefinite)
    Gauss-Newton SQP; the acados reference uses `x0 = xs` + terminal cost.
    Here the terminal equality keeps the start feasible enough to converge
    in ~7 iterations with default options (no LM/scaling needed).
- **Bug fixed along the way:** the raw `ExplicitRk/ImplicitRkIntegrator`
  store a `const Ode&`; the test's per-block `CstrOde{}` temporaries were
  destroyed at the end of the constructor full-expression, leaving a dangling
  reference (UB). A single named `CstrOde ode;` now outlives every integrator.
- **Regression:** all existing targets
  (`double_integrator`, `mass_spring`, `qp_dim`, `qp_unit`, `sqp_unit`,
  `sqp_double_integrator`, `sqp_mass_spring`, `sqp_acados_ref`,
  `integrators_unit`, `sqp_continuous`) pass.
- Build: warning-free under `-Wall -Wextra -Werror`.

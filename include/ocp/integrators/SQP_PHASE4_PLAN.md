# Phase 4 plan: RK integrator + continuous-time problem interface

Companion to `SQP_PLAN.md` (§11 Phase 4) and the two `TODO.md` items it
advances:

- *"reimplement acados style explicit and implicit RK solver/integrator
  alloc free template class"*
- *"define a continuous time problem interface using an integrator class
  to implement the discrete time interface"*

This file is the working reference for the phase: the design, the
source-verified conventions, the sub-step breakdown, and the running
checkpoints (append a checkpoint entry per sub-step, same protocol as
`solvers/SQP_PHASE3_PLAN.md` / `solvers/acados/SQP_PHASE2_PLAN.md`).

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
  time-varying ODEs (explicit `t_k` argument), lifted IRK
  (`sim_lifted_irk_integrator.c`), algebraic variables (`nz > 0`),
  multi-step (`num_steps > 1`) integration, parameter sensitivities
  (`S_p` / `sens_forw_p`), and the adjoint sweep's cost-integral use.

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
    void dynamics_jacobian(int, const typename P::state_t& x,
                           const typename P::control_t& u,
                           typename P::dyn_df_dx_t& df_dx,
                           typename P::dyn_df_du_t& df_du) const;
    void dynamics_hess_prod(int, const typename P::state_t& x,
                            const typename P::control_t& u,
                            const typename P::state_t& w,
                            const typename P::state_t& v_x,
                            const typename P::control_t& v_u,
                            typename P::state_t& hv_x,
                            typename P::control_t& hv_u) const;
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
- Because `dynamics_hess_prod` is only called when
  `Dims::has_dynamics_hess_prod` is true, a linear ODE (zero HVP) may set
  `has_dynamics_hess_prod = false` and the adapter's method is never
  instantiated (the `static_assert` trait in §1.1 is not fired).

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

## 4. Test matrix

| target | covers |
|---|---|
| `integrators_unit` (new) | 4a tableau identities; 4b ERK value/jacobian/HVP (FD); 4c adapter + exact-linear; 4d IRK value/Newton/jacobian/HVP (FD) |
| `sqp_continuous` (new) | 4e end-to-end continuous SQP (ERK + one IRK) vs discrete reference |
| all existing targets | regression (no behavior change) |

Exit criteria: all new + existing targets build warning-free and pass;
ERK value matches the analytic ODE solution, ERK/IRK Jacobians and HVPs
match finite differences to ≤1e-5; the continuous double-integrator reaches
`kSolved` with all four NLP residual norms below tolerance; the discrete
reference tests are bit-for-bit unchanged.

## 5. Future extensions (not in Phase 4)

1. **Time-varying ODE** — add a `t_k = k·h` argument to `OdeModel::f` and
   its derivatives; thread it through the integrator (currently
   time-invariant, `k` ignored).
2. **Multi-step** (`num_steps > 1`) — the ERK/IRK loops already have the
   `istep` outer loop shape; expose `num_steps` and accumulate.
3. **Lifted IRK** (`sim_lifted_irk_integrator.c`) — a different Newton
   structure; separate integrator.
4. **Algebraic variables** (`nz > 0`) — DAEs; the ERK path explicitly
   rejects `nz != 0` (matches acados), IRK would extend the state.
5. **Parameter sensitivities** (`S_p` / `sens_forw_p`) and the adjoint
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

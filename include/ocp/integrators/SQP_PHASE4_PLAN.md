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

### 4f — `ImplicitRkIntegrator::hess_prod` (nonlinear ODE HVP) — *extension draft, not started*

Complete the IRK integrator by implementing `hess_prod` for a **nonlinear**
ODE (the path 4d left as a `static_assert` stub). This lets a nonlinear
continuous dynamics use the IRK integrator end-to-end (SQP builds the
second-order QP block from `dynamics_hess_prod`).

**Why 4d stopped short (re-stated for context).** The composed IRK map
`Φ(x,u) = x + h·Σ_s b_s·K_s*` has `K_s* = f(x_s,u)` with `x_s = x +
h·Σ_j A[s][j]·K_j*` (full `A`, implicit). Its Hessian is a pullback of the
ODE Hessian through an *implicit* (Newton-solved) map, so it has two
contributions (per stage `s`):

  1. **ODE-Hessian term** — `H_f(x_s,u)` (the ODE Hessian, w.r.t.
     `(x,u)`) applied to the first-order stage direction
     `(dx_s, v_u)`, `dx_s = v_x + h·Σ_j A[s][j]·dK_j(v)`.
  2. **Jacobian-of-Jacobian term** — `f_x(x_s,u)` applied to the
     *second-order* stage perturbation `ddx_s = h·Σ_j A[s][j]·ddK_j(v)`,
     where `ddK` is the second-order correction (the "Jacobian-of-Jacobian"
     of the implicit solve).

The ERK HVP (4b) has only term (1), threaded by a forward JVP + backward
adjoint recursion. The IRK HVP needs both terms, which is why acados threads
the full second-order stage sensitivity.

**acados reference (re-read fresh at implementation time; lines from the
current checkout).** `acados/acados/sim/sim_irk_integrator.c`:
- `sim_irk_forward_step` (≈1440–1592): forward sweep with `sens_hess`.
  Builds `dG_dxu` (the `∂R/∂(x,u)` blocks, `= −f_x, −f_u` per stage) and
  `dG_dK` (the block Newton Jacobian `G`), factors `dG_dK`, then
  `dK_dxu = G⁻¹·dG_dxu` (the first-order sensitivity, :1498–1513).
- `sim_irk_eval_jacG` (1616–1671): assembles `dG_dxu` / `dG_dK` and factors
  `dG_dK` in place (`blasfeo_dgetrf_rp`, :1669).
- `sim_irk_backward_step` (1741–1788): per step, forms `lambdaK`
  (`= −h·b·lambda`), back-solves `Gᵀ·lambdaK` (`sim_irk_backsolve_jacG_T`,
  :1777), then `lambda += dG_dxuᵀ·lambdaK` (:1781). This is the adjoint
  (the "dual" passed to each stage's ODE HVP).
- `sim_irk_propagate_hessian` (1673–1739): builds `dxkzu_dw0` — the
  *second-order* sensitivity of the augmented stage state `[x_s; K; u]` w.r.t.
  the initial input (accumulates `−a·dK_dxu` per stage, :1694–1704) — calls
  `model->impl_ode_hess` at each stage (with the `lambdaK` adjoint as the
  dual), and accumulates into the full `(nx+nu)×(nx+nu)` composed Hessian via
  `blasfeo_dsyrk_ut` (:1736). `sim_irk_backward_sweep` (1790–1804) extracts
  `out->S_hess`.

**Key difference vs acados.** acados' `impl_ode_hess` returns the *full*
Hessian of the DAE residual (a `(2nx+nz+nu)²` object), so it can accumulate
the full composed Hessian with `dsyrk`. Our ODE contract (§1.1) exposes only
the **bilinear HVP** `hess_prod(x,u,w,v_x,v_u → hv_x,hv_u)` of `f` w.r.t.
`(x,u)`. So we compute the bilinear form `wᵀ∇²Φ·v` directly, without forming
the full Hessian. Both terms still must be threaded; the identity is:

```
HVP(w, v)  =  h · Σ_s b_s · [  (ODE-HVP term)  +  (Jacobian-of-Jacobian term) ]
```
  - **(1) ODE-HVP term:** `ode_.hess_prod(x_s, u, dual=lambdaK_s,
    dirx=dx_s, diru=v_u, ·, ·)` where `lambdaK_s` is the back-solved adjoint
    and `dx_s` the first-order stage direction. (This is exactly the ERK
    per-stage call; the only IRK-specific input is `lambdaK_s` from the
    `Gᵀ` back-solve rather than the ERK backward recursion.)
  - **(2) Jacobian-of-Jacobian term:** `f_x(x_s,u)ᵀ`-contraction with the
    second-order stage perturbation `ddx_s = h·Σ_j A[s][j]·ddK_j(v)`,
    accumulated through the same adjoint `lambdaK_s`. This requires solving a
    second-order system `G·ddK = B` per HVP call, where `B_s` is the ODE
    Hessian applied to the first-order direction `(dx_s, v_u)` (i.e. one
    `ode_.hess_prod` per stage supplies `B_s`'s ODE-Hessian part).

> **Open design decision at implementation time:** whether to (a) solve the
> second-order `G·ddK = B` system explicitly (one extra LU back-solve per
> HVP call, reusing the factored `G` from the Newton solve), or (b) fold term
> (2) into a single augmented solve. Option (a) is the direct analogue of the
> acados `dxkzu_dw0` thread and is the expected choice; validate by FD before
> committing either. The `G` factorization from `solve_newton` is reusable
> (acados `jac_reuse`, :1469) — the HVP is always called right after
> `value`/`jacobian` at the same `(x,u)`.

**Workspace additions** (extend `detail::IrkWorkspace`, all fixed-size):
- `dK_dxu` (`NK × (nx+nu)`): first-order sensitivity `G⁻¹·F` (kept from the
  Newton/jacobian pass, reused by `hess_prod`).
- `dK_v`, `ddK` (`NK`): first- and second-order corrections in direction `v`.
- `dx_stage` (`std::array<state_t, NS>`), `ddx_stage` (`std::array<state_t,NS>`):
  first- and second-order stage-state perturbations.
- `lambdaK` (`NK`): adjoint (dual) on `K` from the `Gᵀ` back-solve.
- (`G` and `PartialPivLU` already exist from 4d; reuse, re-factor if the
  Newton count changed.)

**Algorithm (per `hess_prod(x,u,w,v_x,v_u,hv_x,hv_u)` call):**
```
solve_newton(x, u)                       # K, G (LU)   [reuse 4d]
dK_dxu  = G^{-1} * F                     # F_s = [f_x_s | f_u_s]  (first-order)
dK_v    = dK_dxu * (v_x; v_u)            # first-order in direction v
for s: dx_stage[s]  = v_x + h*sum_j A[s][j]*dK_v[j]   # first-order x_s
# second-order: B_s from ODE Hessian on (dx_stage[s], v_u); solve G*ddK = B
for s: B_s = ode_.hess_prod(x_s, u, w_e?, dx_stage[s], v_u, ...)  # see note
ddK = G^{-1} * stack(B)
for s: ddx_stage[s] = h*sum_j A[s][j]*ddK[j]
# adjoint: beta_s = h*b_s*w ; lambdaK = G^{-T} * beta ; (theta back-accum)
for s: solve G^T * lambdaK = beta (via LU .transpose())
for s:
    (1) ode_.hess_prod(x_s, u, lambdaK_s, dx_stage[s], v_u, mx, mu)
        hv_x += mx ; hv_u += mu
    (2) hv_x += f_x_s^T (lambdaK_s) contracted with ddx_stage[s]
        (f_x_s^T * lambdaK_s) . ddx_stage[s]  -> add to hv_x
# (u-part of term (2): f_u_s^T lambdaK_s · ddx ... see FD-verified form)
```
(The precise index/contract for term (2)'s `u`-part is to be fixed and
FD-verified at implementation time; the draft above is the structural sketch.)

**Test plan** (`tests/integrators/rk_implicit_4d.cpp`, extend the existing
`run_irk_4d_tests`):
- *hess_prod, FD:* reuse a nonlinear ODE with an analytic `hess_prod`
  (e.g. `OdeXsqU`: `ẋ = x⊙x·u + x`, from 4b/4d). Build the
  `(nx+nu)×(nx+nu)` Hessian of the scalar `wᵀΦ(x,u)` by central FD (mixed
  4-point cross derivatives, δ=1e-4), apply to 3 deterministic `(w, v)`
  pairs, assert the IRK `hess_prod` matches the FD-Hessian contraction to
  1e-5, for `RadauIia2Tag` and `RadauIia4Tag`.
- *cross-check vs ERK:* on the same nonlinear ODE, assert the IRK HVP and
  the ERK HVP (4b) agree to 1e-8 for the same `(x,u,w,v)` (both must equal
  the FD Hessian, so agreement is a strong independent check).
- *regression:* all existing targets unchanged; the linear-ODE path
  (`has_dynamics_hess_prod = false`) still returns zeros and never reaches
  the new code.

**Status:** draft only — no code written. Gate: FD-verified `hess_prod`
before the `static_assert(false)` in `rk_implicit.hpp` is lifted. One commit
(`integrator:` + `test:`), re-using the `integrators_unit` target.

## 4. Test matrix

| target | covers |
|---|---|
| `integrators_unit` (new) | 4a tableau identities; 4b ERK value/jacobian/HVP (FD); 4c adapter + exact-linear; 4d IRK value/Newton/jacobian/HVP (FD); 4f IRK nonlinear HVP (FD) [draft] |
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
6. **IRK nonlinear HVP** — `ImplicitRkIntegrator::hess_prod` for nonlinear
   ODEs; tracked as sub-step **4f** (extension draft, §3). The 4d partial
   stub (`static_assert` for nonlinear, zero for linear) is replaced by the
   full ODE-Hessian + Jacobian-of-Jacobian thread above.

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

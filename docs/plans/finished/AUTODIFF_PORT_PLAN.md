# Plan: Port the cppduals AutoDiffOde to master (fused model evaluation)

Status: **in progress** (2026-10-04). This file is the working reference for
the implementation. It ports the first cppduals-based derivative
implementation from branch `dev` (commits `4bb0a17`, `6254aaa`, `6ff16bf`,
merge-base `6c524f5`) onto the current master, where the model layers have
since evolved: the ODE contract gained the time argument `t` (phase 4h), the
problem contract moved to fused value/grad/hess triplets, and
`ContinuousProblem` / the integrators were remapped accordingly.

The dev-branch plan docs are **superseded and ignored**:
`MODEL_API_PLAN.md` (already implemented on master, see
`docs/plans/finished/MODEL_API_PLAN.md`) and
`include/ocp/integrators/ODE_AUTODIFF_PLAN.md` (written against the
pre-`t`, pre-fusion ODE contract). The mechanics from the latter (AD
seeding, vendoring, Eigen glue) carry over; everything signature-shaped is
rederived for master.

## 0. What dev brings and what must be redone

| dev commit | content | disposition |
|---|---|---|
| `4bb0a17` | vendor `thirdparty/cppduals` (LICENSE, README, `duals/dual`), `duals_eigen3.hpp`, CMake include | **cherry-pick as-is** (no dependency on the old API) |
| `6254aaa` | `ode_autodiff.hpp`: `AutoDiffOde<Derived, Dims>` with `jacobian` / `hess_prod` | **rewrite**: old signatures (no `t`), no fused entries; new base exposes the fused `value_jac` / `value_jac_hess_prod` entries, with `jacobian` / `hess_prod` as thin wrappers |
| `6ff16bf` | `ode_autodiff_4j.cpp` tests | **adapt**: `t` argument everywhere; add checks for the fused entries, the integrator forward cache, and the AD integrator paths |
| `ce317fc`, `adc8ed3` | plan docs | ignore (superseded) |

## 1. Current master state (why a mechanical port is not enough)

**ODE contract** (`include/ocp/integrators/ode_model.hpp`), duck-typed:

```
state_t f(const state_t& x, const control_t& u, double t) const;
void jacobian(const state_t& x, const control_t& u, double t,
              dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const;
void hess_prod(const state_t& x, const control_t& u, double t,
               const state_t& w, const state_t& v_x, const control_t& v_u,
               state_t& hv_x, control_t& hv_u) const;   // optional (nonlinear)
```

Presence of `hess_prod` is probed by `ode_supports_hess_prod<Ode, Dims>` and
`static_assert`ed at the integrator `hess_prod` call sites.

**Problem contract** (`problem.hpp`): fused triplets per model part.
Dynamics entries:

```
state_t dynamics_next_state(int k, const state_t& x, const control_t& u) const;
void dynamics_value_jac(int k, x, u, state_t& x_next, df_dx&, df_du&) const;
void dynamics_value_jac_hess(int k, x, u, const state_t& lam, x_next,
                             df_dx&, df_du&, dyn_hess_t& hess) const;
```

`dyn_hess_t` is the **multiplier-contracted** Hessian of the stage map:
`hess v = sum_i lam_i D^2(Phi_i) v`, `(nx+nu) x (nx+nu)`, `[x; u]` layout.

**`ContinuousProblem` today** (`continuous_problem.hpp`):

- `dynamics_value_jac`  = `integ_.value(...) + integ_.jacobian(...)` (two passes)
- `dynamics_value_jac_hess` = the above **plus** `nx+nu` unit-vector calls to
  `integ_.hess_prod(...)`, each of which redoes the full forward pass.

**`ExplicitRkIntegrator` internals** (`rk_explicit.hpp`, `NumSteps`
sub-steps, `NS` Butcher stages per sub-step):

- `value`: `NS` plain `ode_.f` calls per sub-step.
- `jacobian`: `NS` plain `ode_.f` + `NS` `ode_.jacobian` calls + explicit
  sensitivity threading (`JX = Jf_s * Dxs`, `JU = Jf_s * Dus + Jf_u`,
  sub-step composition `Jx_ss`, `Ju_ss`).
- `hess_prod`: forward pass (stage states `x_stage`, stage derivs `K`,
  threaded `JX`/`JU`, first-order JVP `dx`/`dK` along `v`) + an explicit-dual
  reverse sweep that injects `ode_.hess_prod` once per stage (weight = local
  adjoint). The forward is **recomputed on every call** — the `nx+nu` HVP
  calls from `ContinuousProblem` therefore redo it `nx+nu` times.

**`ImplicitRkIntegrator` internals** (`rk_implicit.hpp`): Newton-based stage
solve; `hess_prod` (NumSteps==1) runs the Newton forward + `G` factor +
`sens = G^{-1} Q` (all v-independent) and **re-runs it for every HVP
direction**; the v-dependent part is the `R` assembly (`NS * 2*nx` ODE HVP
injections per stage loop) + `G^{-1} R` solve + b-contraction. NumSteps>1
threads the Newton states / Jacobians / co-state across sub-steps with the
same per-call forward recompute.

**Cost today** (per OCP stage, AD ODE, `NS` stages, `nIn = nx+nu`; one pass
of `f` at precision `p` counts as `p` "f-units" — plain = 1,
`dual<double, nIn>` = `nIn+1`, nested `dual<dual<double, nIn>, 1>` =
`2(nIn+1)`):

| entry | ODE passes (today) |
|---|---|
| `dynamics_value_jac` | `NS` plain + `NS` dual1 |
| `dynamics_value_jac_hess` | above + `nIn` x (`NS` plain + `NS` dual1 + `NS` nested + JVP + adjoint) |

## 2. AD mechanics (unchanged from the dev PoC)

- **Jacobian**: one forward pass of `duals::dual<scalar_t, nIn>`; seed
  `z_k = variable(z_k, k)`; read `fd(i).dpart(k) = df_i/dz_k`. The **value
  is carried for free**: `fd(i).rpart() = f_i`.
- **HVP**: one nested pass of `duals::dual<dual<scalar_t, nIn>, 1>`; seed
  inner `= variable(z_k, k)`, outer `= v_k`. Then
  `g = w^T f` (state-space weight applied **only at the output contraction**,
  never inside `f`) and `hv_j = g.dpart(0).dpart(j) = e_j^T D^2 g v`. The
  outer dimension stays `N = 1` because the multiplier contraction happens
  outside `f` — this is the contraction gain: the Hessian is never
  materialized in the dual lattice. The nested pass also carries value +
  full first derivatives for free (`fd(i).rpart()` is the dual1 row).
- **Full composed-map map**: for an *explicit* RK scheme the stage recursion
  is pure arithmetic, so running it in the nested dual scalar gives
  `x_next` with value, Jacobian, and the HVP `D^2(x_next_i) v` for every
  output `i` in **one pass** (no adjoint sweep at all):
  `x_next_dual(i).rpart().rpart()` = value,
  `x_next_dual(i).rpart().dpart(j)` = `dPhi_i/dz_j`,
  `x_next_dual(i).dpart(0).dpart(j)` = `(D^2 Phi_i v)_j`.
- **Vendoring**: `thirdparty/cppduals/duals/dual` only (pinned master commit
  `7b2a6d1`, MPL-2.0, self-contained C++17 header, no Eigen dependency).
  Upstream `duals/dual_eigen` `#error`s on Eigen 3 and only adds Eigen-5
  GEMM/SIMD packets; not needed — every dual-valued product in this codebase
  is a small ODE `f` / stage expression, all large linear algebra stays
  plain `double`.
- **Eigen glue** (`duals_eigen3.hpp`): `Eigen::NumTraits<duals::dual<T,N>>`
  (inherited from the value type, `RequireInitialization = false` — the dual
  default-ctor zero-initializes) + `ScalarBinaryOpTraits` promotion for
  mixed dual/arithmetic operands (both orders, per arithmetic type).
  Targets Eigen 3.x (system has 3.4.90).

## 3. Design decisions

### D1 — Vendoring + glue: cherry-pick as-is

- `thirdparty/cppduals/{LICENSE.txt, README.md, duals/dual}` verbatim from dev.
- `include/ocp/integrators/duals_eigen3.hpp` verbatim from dev.
- CMake: project-wide `thirdparty/cppduals` include dir (any target that
  compiles `rk_explicit.hpp` transitively includes `duals_eigen3.hpp`).

### D2 — Fused ODE contract entries + capabilities (`ode_model.hpp`)

Two **optional** duck-typed entries, alongside (not replacing) the existing
ones — hand-written ODEs stay untouched:

```
void value_jac(const state_t& x, const control_t& u, double t,
               state_t& f_val, dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const;

void value_jac_hess_prod(const state_t& x, const control_t& u, double t,
                         state_t& f_val, dyn_df_dx_t& df_dx,
                         dyn_df_du_t& df_du, const state_t& w,
                         const state_t& v_x, const control_t& v_u,
                         state_t& hv_x, control_t& hv_u) const;
```

Semantics: the same single `f` pass as the corresponding pair of legacy
calls, with `f_val` (and for the HVP entry, `df_dx`/`df_du`) produced for
free. Backends that cannot fuse may still provide them as `f` + `jacobian`
(`+ hess_prod`) composites — the entries are contract-level, not
optimization-mandatory.

New SFINAE probes + traits (mirroring `ode_hess_prod_probe`):

```
ode_supports_value_jac<Ode, Dims>
ode_supports_value_jac_hess_prod<Ode, Dims>
```

plus the `ode_value_jac(...)` / `ode_value_jac_hess_prod(...)` inline
helpers (same pattern as `ode_hess_prod`).

A **generic scalar-capability probe** (no duals dependency in the contract
header; the dual type `D` is passed in):

```
ode_f_evaluable_at<Ode, Dims, D>  // f evaluable with Matrix<D, nx/nu, 1> args
                                  // AND returning Matrix<D, nx, 1>
```

The return-type `is_same` check is essential: Eigen 3's converting
`Matrix(const EigenBase<OtherDerived>&)` constructor is *not* SFINAE-gated on
the scalar type, so a call `ode.f(Matrix<dual>, ...)` would otherwise be
"valid" (via a converting temporary) even for a hand-written
`f(state_t, control_t, double)` ODE. Requiring the return type to be the
`D`-typed state vector discriminates a genuinely scalar-templated `f`
(AutoDiffOde) from a hand-written one.

### D3 — `AutoDiffOde<Derived, Dims>` CRTP base (`ode_autodiff.hpp`)

Derived supplies **only** the scalar-templated RHS:

```
template <class D>
Eigen::Matrix<D, nx, 1> f(const Eigen::Matrix<D, nx, 1>& x,
                          const Eigen::Matrix<D, nu, 1>& u, double t) const;
```

Base implements (allocation-free, fixed-size stack buffers, copied
`head(nx)`/`tail(nu)` splits — a `VectorBlock` does not match the derived
`f` signature):

- `value_jac`: one `dual<scalar_t, nIn>` pass; `f_val` from `rpart()`.
- `value_jac_hess_prod`: one nested pass; `f_val` / Jacobian from
  `fd(i).rpart()`; HVP via the `w`-output contraction.
- `jacobian` / `hess_prod`: thin wrappers over the fused entries that drop
  `f_val` (respectively `f_val` + Jacobian). Keeping the legacy entries means
  `ode_supports_hess_prod` stays true and **every existing integrator code
  path keeps working unmodified** (drop-in for both integrators, as in dev).

The base does **not** define `f` (no name hiding). Header doc carries over
the dev caveats, updated for master names: set `Dims::has_dynamics_hess` to
`true`; a purely linear ODE is better off skipping `AutoDiffOde` and setting
`has_dynamics_hess = false` (the inherited HVP would correctly return zero
but pay one wasted nested pass per call).

### D4 — `ExplicitRkIntegrator`: AD front paths + non-AD forward cache

The integrator runs the dual passes **itself** (seeds the duals, calls
`ode_.f` with them, extracts) — it does not rely on the ODE-level fused
entries. Path selection is compile-time via `ode_f_evaluable_at`.

New workspace state (in the existing `mutable ErkWorkspace`):
`last_x`, `last_u`, `last_t`, `fwd_valid` — the non-AD forward cache key.
The invariant: whenever `fwd_valid`, the workspace forward state
(`x_stage`, `K`, `JX`, `JU`) corresponds exactly to
`(last_x, last_u, last_t)`; every forward that (re)writes them refreshes the
key, and a hit (key match) is only taken from a state written under that key.

New private `ad_forward<D>(x0_dual, u_dual, t_k, x_next_dual)`: the RK
recursion (NumSteps sub-steps) evaluated in scalar `D`, seeded by the
caller. Pure arithmetic — no sensitivity threading, no adjoint.

Public entries:

- **`value_jac(x, u, t_k, x_next, df_dx, df_du)`** (new):
  - AD ODE (`ode_f_evaluable_at<Ode, Dims, dual1_t>`): seed
    `z = [x; u]` with the `nIn`-basis, one `ad_forward` pass; extract
    value + full Jacobian from the final dual state. One `NS`-stage dual1
    pass replaces today's `NS` plain + `NS` dual1.
  - hand-written ODE: `forward_jac` (the legacy `jacobian` body, refactored
    to also emit `x_next` — it already accumulates it and discards it) and
    set the forward cache.
- **`jacobian(...)`**: same path, `x_next` discarded. For the AD ODE this is
  the fused dual1 pass; for hand-written ODEs it *is* `forward_jac`.
- **`hess_prod(x, u, t_k, w, v_x, v_u, hv_x, hv_u)`** (signature unchanged):
  - AD ODE (`ode_f_evaluable_at<Ode, Dims, dual2_t>`): seed inner basis +
    outer direction `v`, one `ad_forward` pass in the nested dual; extract
    the HVP `hv_j = sum_i w_i x_next_dual(i).dpart(0).dpart(j)`. **No adjoint
    sweep, no per-stage ODE HVP calls** — the `w` contraction happens at
    extraction (the contraction gain).
  - hand-written ODE: legacy algorithm, but the forward is
    **cache-checked** (key `(x, u, t_k)`); on a hit only the v-dependent JVP
    threading + backward sweep run. `static_assert(ode_supports_hess_prod)`
    moves into this branch only.
- **`value_jac_hess_prod(...)`** (new; the ODE-level analogue of the
  problem-level triplet, composed over the stage map):
  - AD ODE: the nested `ad_forward` pass with full extraction (value +
    Jacobian + `w`-contracted HVP).
  - hand-written ODE: `value_jac` + `hess_prod` composite.
- **`supports_value_jac_hess_prod`**: `static constexpr` capability flag
  (`ode_f_evaluable_at<Ode, Dims, dual2_t>::value` for the explicit
  integrator; `false` for the implicit one) so `ContinuousProblem` can
  dispatch without including the duals header.

`value(...)` (plain) is unchanged and does not touch the cache.

### D5 — `ImplicitRkIntegrator`: value_jac composite + hess forward cache

The implicit stage solve is Newton-based (dual-Newton is out of scope, §8),
so no AD fusion happens inside the stage solve; the win is **caching the
v-independent forward**:

New workspace state: `last_x`, `last_u`, `last_t`, `fwd_valid`
(single-sub-step Newton state valid for the key — NumSteps==1 only),
`sens_valid` (`ws_.sens = G^{-1} Q` cached), `traj_valid` (multi-step
trajectory arrays + `R_traj`/`S_traj` valid for the key) + `sens_traj[ss]`
(per-sub-step `G^{-1} Q`).

- **`value_jac(x, u, t_k, x_next, df_dx, df_du)`** (new): `value()` +
  `jacobian()`; with the cache, the second call reuses the first's Newton
  state instead of re-running it.
- **`value()`**: NumSteps==1 cache hit → `x_next = x + h sum b K` from the
  cached `ws_.K` (no Newton); miss → today's sweep + cache update.
- **`jacobian()`**: NumSteps==1 cache hit → skip `solve_newton`, reuse
  `ws_.dfdx/dfdu/lu_` for the `G^{-1} dR/d(x,u)` solves; the per-sub-step
  post-Newton body is factored into `substep_jac(...)` shared with the
  miss path.
- **`hess_prod()`**:
  - NumSteps==1: cache hit skips the Newton forward; `sens = G^{-1} Q`
    computed once per key (`sens_valid`); the v-dependent `R` assembly,
    `G^{-1} R` solve and b-contraction always run (per direction).
  - NumSteps>1: cache hit skips the whole multi-step Newton forward
    (trajectory arrays, `A/B/R/S` threading all v-independent and stored);
    only the v-dependent JVP threading (`dx_traj`) + backward sweep re-run.
    The per-sub-step `sens` comes from `sens_traj`.
  - Any `value()` / `jacobian()` forward invalidates the multi-step
    trajectory cache (they don't write the traj arrays).

### D6 — `ContinuousProblem` remap

```
dynamics_value_jac      -> integ_.value_jac(x, u, k*h_, x_next, df_dx, df_du)
                           (was: integ_.value + integ_.jacobian)

dynamics_value_jac_hess -> if Integ::supports_value_jac_hess_prod
                             nIn x integ_.value_jac_hess_prod(..., e_j)
                             (value + Jacobian extracted on the first column;
                              one nested pass per Hessian column, no forward
                              recomputation, no adjoint sweep)
                           else
                             integ_.value_jac(...)   // populates the cache
                             nIn x integ_.hess_prod(...)   // cache hits
```

The problem contract itself (`problem.hpp`) is **unchanged** — only the
adapter implementation and the integrator internals change. QP/SQP layers
see the same `dyn_hess_t`.

## 4. Efficiency summary

Per OCP stage, AD ODE, `NS` stages, `nIn` inputs (f-units, §1; `NS = 4`,
`nIn = 3` example values in parentheses):

| entry | master today | after this plan |
|---|---|---|
| `dynamics_value_jac` | `NS` plain + `NS` dual1 (20) | `NS` dual1 (16) |
| `dynamics_value_jac_hess` | above + `nIn` x (`NS` plain + `NS` dual1 + `NS` nested + JVP + adjoint) (180) | `nIn` x `NS` nested (96) — one nested pass per Hessian column carries value + Jacobian for free |

For hand-written ODEs the AD fusion is absent, but the forward cache alone
cuts the `dynamics_value_jac_hess` forward cost from `nIn + 1` forwards to
`1` forward (NumSteps==1 implicit: `(nIn - 1)` full Newton sweeps saved).
Nested-pass storage is `(nIn+1) * 2` doubles per state component — linear in
`nIn`, unlike a full-basis pass.

## 5. Sub-steps (one commit each)

1. **`build: vendor cppduals duals/dual + Eigen-3 glue`** — cherry-pick
   `4bb0a17` file-wise (thirdparty tree, `duals_eigen3.hpp`, CMake include).
   [DONE]
2. **`integrator: fused value_jac / value_jac_hess_prod ODE entries`** —
   extend `ode_model.hpp` with the two contract entries, the SFINAE probes,
   traits, `ode_f_evaluable_at`, and helper functions (§D2). No behavior
   change for existing ODEs. [DONE]
3. **`integrator: add AutoDiffOde CRTP base (multidual AD, fused entries)`** —
   `ode_autodiff.hpp` per §D3 (rewritten from dev's `6254aaa` for the `t`
   argument + fused entries). [DONE]
4. **`integrator: ExplicitRk AD front paths + hess forward cache`** — §D4:
   `ad_forward`, `value_jac`, restructured `hess_prod`,
   `value_jac_hess_prod`, workspace cache key, capability flag. [DONE]
5. **`integrator: ImplicitRk value_jac + hess forward cache`** — §D5.
   [DONE; composite `value_jac_hess_prod` added for API uniformity]
6. **`problem: ContinuousProblem uses fused integrator entries`** — §D6.
   [DONE]
7. **`test: AutoDiffOde fused-entry + integrator checks (4j)`** — adapt
   dev's `ode_autodiff_4j.cpp` to master (§7 below), wire into
   `integrators_main.cpp` + CMake. [DONE]
8. Full build; run every test target; confirm zero regressions. [DONE —
   all 11 targets (double_integrator, mass_spring, masses_chain,
   pendulum_on_cart, qp_unit, integrators_unit, sqp_* x5) build
   warning-free under `-Wall -Wextra -Werror` and pass.]

## 6. Risks / notes

- **Eigen 3.4.90** confirmed on this machine; the glue targets Eigen 3.x and
  has no SIMD/packet content, so no vectorization regression is possible for
  plain-`double` paths (the dual paths are new code).
- **Probe correctness**: `ode_f_evaluable_at` must check the return type
  (§D2) or every hand-written ODE would take the AD path and fail deep in
  Eigen's converting constructor.
- **Cache correctness**: the key is `(x, u, t_k)` exact-equality; a stale hit
  is impossible because every workspace-rewriting forward refreshes the key
  (invariant §D4/§D5). `NumSteps > 1` uses the same key (`h` fixed per
  integrator instance).
- **Implicit + AD**: the implicit integrator stays on the legacy ODE entries
  (Newton-based stage solve; dual-Newton is a separate, costlier concern,
  §8). The forward cache (§D5) is the efficiency win there.
- **Compile-time**: `rk_explicit.hpp` now includes `duals_eigen3.hpp`
  (`<duals/dual>`, ~1.9k lines of templates) — every TU that compiles the
  explicit integrator pays it; the CMake include dir is project-wide.
- **Linear ODEs through `AutoDiffOde`**: see §D3 caveat; tests use a
  nonlinear ODE (`x.*x*u + x`, dev's `OdeXsqU`, plus a `t`-dependent term)
  and keep the linear-ODE regression ODEs hand-written.
- **No solver-side changes**: `problem.hpp`, QP, SQP untouched; the port is
  fully contained in `include/ocp/integrators/`, `continuous_problem.hpp`,
  `thirdparty/`, and the integrator test target.

## 7. Test plan (sub-step 7)

Port of dev's 4j suite with the `t` argument. Reference ODE
(`OdeXsqURef`): `xdot = x.*x*u + x + [sin(t); cos(t)]` — the additive
`t`-dependent term makes the *value* time-varying (exercising the `t`
plumbing through the dual paths) while leaving the hand-derived Jacobian /
HVP references identical to dev's. AD ODE (`OdeXsqUAd`): same `f`, scalar-
templated, no derivative methods.

1. **Traits**: `static_assert` on `ode_supports_hess_prod`,
   `ode_supports_value_jac`, `ode_supports_value_jac_hess_prod`, and
   `ode_f_evaluable_at<OdeXsqUAd, Dims, dual1/dual2>`; `false` for
   `OdeXsqURef` (hand-written) — guards the return-type probe.
2. **ODE-level** (AD vs `OdeXsqURef` + FD):
   - `jacobian` / `hess_prod` (legacy): exact vs hand-derived, FD
     `< 1e-6` / `< 1e-5` (dev checks, with `t`);
   - `value_jac`: `f_val` vs hand `f`, Jacobian vs hand-derived;
   - `value_jac_hess_prod`: value + Jacobian as above; HVP vs hand-derived
     and vs central-FD Hessian of `g = w^T f` applied to `v` (3 `(w, v)`
     cases, as in dev).
3. **Integrator-level** (wrapped in `ExplicitRkIntegrator<..., 4, K4Tag>` and
   `ImplicitRkIntegrator<..., 2, RadauIia2Tag>`):
   - `value_jac` of the composed map vs central FD of `value` (`< 1e-6`);
   - `hess_prod` vs central-FD Hessian of `w^T Phi` applied to `v`
     (`< 1e-5`, 3 cases) — dev's integration checks, with `t`;
   - `value_jac_hess_prod` vs the same FD references (explicit: AD path;
     implicit: composite path);
   - **cache check**: cold `hess_prod` result == warm `hess_prod` result
     (same `(x, u, t)`, different `v`); `value_jac` then `hess_prod` on both
     integrators (cache populated by `value_jac` on the explicit non-AD path
     — verified with a hand-written ODE);
   - **cross-backend check**: `OdeXsqURef` (hand-written, same `f`) through
     the same integrator yields `value_jac` / `hess_prod` matching the AD
     ODE's to `< 1e-12` (guards the AD glue end-to-end).
4. **Regression**: all existing targets (`double_integrator`, `mass_spring`,
   `qp_*`, `sqp_*`, `integrators_unit`, `sqp_continuous`) build
   warning-free and pass.

## 8. Future (explicitly out of scope)

- **Full-basis nested pass**: seeding the outer dual with the full `nIn`
  basis (`dual<dual<double, nIn>, nIn>`) yields the *entire* contracted
  Hessian matrix in one pass (cost `(nIn+1)^2` f-units per stage vs
  `2*nIn*(nIn+1)` for the unit-direction scheme — a win for `nIn > 1`), at
  the price of `(nIn+1)^2` doubles per state component of scratch (quadratic
  in `nIn`). Revisit for large `nIn` with a size threshold.
- **Dual-Newton fused stage solves** for the implicit integrator (dual1 /
  nested Newton + dual LU) — would let `ImplicitRkIntegrator` expose a true
  fused `value_jac` / HVP entry instead of the composite.
- Upstreaming `duals/dual` updates / adopting `duals/dual_eigen` with an
  Eigen-5 migration (would enable vectorized dual GEMM for large problems).

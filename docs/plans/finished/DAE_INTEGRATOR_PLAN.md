# DAE-Native Implicit Integrator Plan

CLEAN_UP_TODO item 3: "refactor to use implicit ode formulation for implicit
integrators."

## Interpretation

For implicit integrators (IRK: Radau IIA, Gauss-Legendre), the integrator
should consume the **DAE residual** `F(x, xdot, u, t)` directly (the form the
generated CasADi C code provides), instead of the current **pseudo-ODE**
`f = -F(x, 0, u) / c` manufactured in the generated adapters, where the
per-row coefficient `c_i` is calibrated at construction from the `dF/dxdot`
diagonal and snapped to ±1.  That calibration is exact only when `dF/dxdot`
is diagonal full-rank (i.e. an ODE written in residual form) and is the
"hack" this item removes.  After the refactor the implicit integrator is
DAE-native (acados `sim_irk_integrator` style), and the generated adapters
are thin hand-throughs of `dae_fun` / `dae_fun_jac` with no runtime
calibration.

Explicit integrators (ERK) are **out of scope** and stay ODE-based
(`p5probe`, `double_integrator`, …).

## Current state (facts)

- ODE duck-type: `include/ocp/integrators/ode_model.hpp`
  (`f`, `jacobian`, optional `hess_prod`, optional fused `value_jac`,
  `value_jac_hess_prod`; SFINAE probes `ode_supports_*`).
- Implicit integrator: `include/ocp/integrators/rk_implicit.hpp`,
  `ImplicitRkIntegrator<Dims, Ode, NS, Tag, NumSteps>`.  Newton on the
  stage derivatives `K_s` with `x_s = x_k + h Σ_j A[s][j] K_j`;
  residual `R_s = K_s − f(x_s, u)`; block Jacobian
  `G[s][j] = δ_{s,j} I − h A[s][j] f_x(s)`.  Forward cache keyed on
  `(x, u, t_k)`; multi-step (NumSteps > 1) trajectory cache; 4-term
  second-order HVP formula from the ODE Hessian.
  `supports_value_jac_hess_prod = false` (Newton-based, no dual-Newton).
- Generated IRK adapters (4 of them) contain the c_ hack:
  - `examples/masses_chain/generated/MassesChainGen.hpp`
    (`GaussLegendre4Tag`, 4 stages × 2 sub-steps)
  - `examples/pendulum_on_cart/generated/PendulumGen.hpp`
    (`GaussLegendre4Tag`, 4 × 1)
  - `examples/unicycle/generated/UnicycleGen.hpp` (`GaussLegendre4Tag`,
    4 × 1)
  - `examples/p5probe_b/generated/P5probeB.hpp` (`GaussLegendre2Tag`,
    2 × 1)
  Pattern: `Ode` struct holds `detail::XDyn` (F eval) + `detail::XDynJac`
  (F jac), calibrates `c_` from the `dFxdot` diagonal in the constructor
  (`assert |c| ≥ 0.5`, snap to ±1), then `f = −F(x, 0, u)/c_` and
  `jacobian = −dF/dz / c_` (row-wise).
- All 4 generated Dims have `has_dynamics_hess = false` (Gauss-Newton QP
  Hessian) → the second-order path is only exercised by the hand-written
  ODE test models, not by the generated examples.
- Hand-written twin: `examples/masses_chain/masses_chain.hpp`
  `MassesChainOde` — `f = F(x, 0, u)` via the C residual, FD Jacobian,
  no `hess_prod`.
- Implicit-integrator consumers in tests:
  - `tests/integrators/rk_implicit_4d.cpp` (value/jac/hess_prod, FD checks)
  - `tests/integrators/rk_multistep_4g.cpp` (NumSteps > 1)
  - `tests/integrators/cstr_4i.cpp` (constraint + IRK, SQP)
  - `tests/integrators/ode_autodiff_4j.cpp` (AutoDiffOde, AD fused entries)
  - `tests/integrators/sqp_continuous_4e.cpp` (line 219, IRK + SQP)
- Generator: `tools/acados2ocp_pp.py`, ODE-struct emission for the IRK
  branch (the block that emits `struct *Ode` with the c_ calibration and
  the `f` / `jacobian` methods).

## Target design

### 3.1 New file `include/ocp/integrators/dae_model.hpp`

DAE model duck-type (all `const`, fixed-size Eigen):

```cpp
// DAE residual  F(x, xdot, u, t)  (nx-vector).
state_t F(const state_t& x, const state_t& xdot, const control_t& u,
          double t) const;
// Jacobian blocks at (x, xdot, u, t):
//   dF/dx   (nx × nx), dF/dxdot (nx × nx), dF/du (nx × nu).
void F_jac(const state_t& x, const state_t& xdot, const control_t& u,
           double t, dyn_df_dx_t& dFdx, dyn_df_dx_t& dFxdot,
           dyn_df_du_t& dFdu) const;
// Optional (nonlinear DAE only): bilinear Hessian-vector product of F
// w.r.t. (x, xdot, u):
//   hv_x   = w^T d²F/dxdz · (v_x, v_xdot, v_u)  x-part
//   hv_xdot = … xdot-part,  hv_u = … u-part
void F_hess_prod(const state_t& x, const state_t& xdot,
                 const control_t& u, double t, const state_t& w,
                 const state_t& v_x, const state_t& v_xdot,
                 const control_t& v_u, state_t& hv_x, state_t& hv_xdot,
                 control_t& hv_u) const;
```

SFINAE probes `dae_supports_hess_prod<Dae, Dims>` (mirrors
`ode_supports_hess_prod`); `F` / `F_jac` are required (static_assert in
the integrator).

`OdeToDae<Ode, Dims>` wrapper: turns any ODE model into a DAE model with
the residual `F = xdot − f(x, u, t)` (affine in xdot):
- `F(x, xdot, u, t) = xdot − ode.f(x, u, t)`
- `F_jac: dFdx = −dfdx, dFxdot = I, dFdu = −dfdu`
- `F_hess_prod(x, xdot, u, t, w, vx, vxdot, vu, …)`: `hv_x = w^T H_f(vx, vu)`,
  `hv_xdot = 0` (F affine in xdot → no xdot-Hessian blocks),
  `hv_u = w^T H_f(…, vu)`; only called when `ode_supports_hess_prod` is
  true (otherwise the wrapper omits `F_hess_prod` and the integrator takes
  the zero-HVP path, mirroring the current linear-ODE behavior).

The wrapper is dual-type transparent (`F` with dual `x`/`u` composes with a
dual `f`), so AD-based ODE models keep working through it.

### 3.2 DAE-native `ImplicitRkIntegrator` (in-place refactor of
`rk_implicit.hpp`)

Template parameter renamed `Ode` → `Dae` (duck-type changed).  Public API
(`value`, `value_jac`, `jacobian`, `hess_prod`,
`value_jac_hess_prod`, `newton_residual_inf`,
`supports_value_jac_hess_prod = false`) is unchanged, so
`ContinuousProblem` and all problem-side code are untouched.

Newton unknowns remain the stage derivatives `K_s`; stage states are
`x_s = x_k + h_ss Σ_j A[s][j] K_j` (full Butcher matrix — valid for both
Radau IIA and Gauss-Legendre tags).

- Residual: `R_s = F(x_s, K_s, u, t_s)`  (was `K_s − f(x_s, u)`).
- Block Jacobian: `G[s][j] = δ_{s,j} F_xdot_s + h_ss A[s][j] F_x_s`
  (was `δ_{s,j} I − h A f_x`).
- Warm start `K_init`: per stage, solve the small `nx × nx` system
  `F_xdot_s · K0_s = −F(x_s, 0, u, t_s)` (stage states at `K = 0`).
  For an ODE residual (`F = xdot − f`, `F_xdot = I`) this recovers the
  current `f(x_k, u)` broadcast exactly.  (For the GaussLegendre tags the
  stage states at `K = 0` are all `x_k`.)
- First-order sensitivity: identical structure as today, with
  `Q = [stack_s(F_x_s) | stack_s(F_u_s)]` and `G` as above;
  `x_next = x_k + h Σ_s b_s K_s` gives
  `∂x_next/∂x_k = I + h Σ_s b_s (G^{-1} Q_x)_s`,
  `∂x_next/∂u = h Σ_s b_s (G^{-1} Q_u)_s`.
- Second-order HVP (replaces the ODE 4-term formula; derived in the
  appendix below): with `η_s = (v_x + h_ss A[s,:]·v_K, v_K_s, v_u)`
  (where `v_K = (∂K/∂z)·v` is the first-order K-direction) and the
  mixed Hessian matrices at stage s
  - `Mxx  = ∂²F/∂x∂(·)  (x-slot of ∂²F/∂(x,·) in direction η_s)`
  - `Mxxd = ∂²F/∂x∂xdot (·)` (in direction η_s)
  - `Mxu  = ∂²F/∂x∂u (·)` (in direction η_s)

  the stage-s right-hand side for `K'' v` is

  ```
  b_s = [D Fx_s | D Fu_s]
        − (Mxx^T η_x + Mxxd η_xd + Mxu^T η_u) · K'_s
        − h_ss Σ_j A[s][j] (Mxx η_x + Mxxd η_xd + Mxu η_u) · K'_j
  ```

  where `D Fx_s = Mxx η_x + Mxxd η_xd + Mxu η_u` and
  `D Fu_s = Mux η_x + Muxd η_xd + Muu η_u` (the `u`-slot rows of the same
  HVP evaluations, `nx × nu`).  `Mxx`, `Mxxd`, `Mxu` (and their
  transposes for the xdot/u slots) come from `nx` calls of
  `F_hess_prod(x_s, K_s, u, t_s, w = e_i, v_x, v_xdot, v_u)` per stage.
  `K'' v = G^{-1} b`;  `hv = h Σ_s b_s (K'' v)_s` split into
  `(hv_x, hv_u)`.  The multi-step (NumSteps > 1) backward sweep threads the
  same per-sub-step `b_s` with the existing R/S co-state machinery (the
  per-sub-step local HVP formula is the single change; the thread
  equations `R_{ss+1} = A_ss^T R_ss`, `S_{ss+1} = A_ss S_ss + B_ss` are
  unchanged).

  Special case check: for `F = xdot − f` all xdot-Hessian blocks vanish
  (`Mxxd = 0`, `Mxdot* = 0`), `Mxx = −H_fxx`, `Mxu = −H_fxu`, and the
  formula reduces (up to the sign flip of `G` vs. today) to the existing
  4-term ODE expression `Hv + h·HAm + h(Hv_x)Ja + h²(HAm_x)Ja` — verified
  by the migrated `rk_implicit_4d` / `rk_multistep_4g` tests (FD-based).

- Forward cache / multi-step trajectory cache: unchanged (key `(x, u, t_k)`).
- `newton_residual_inf()`: `‖F(x_s, K_s, u)‖∞` at the converged iterate.

### 3.3 Generated adapters (`tools/acados2ocp_pp.py`)

The ODE-struct emission for the IRK branch is replaced by a DAE struct:

```cpp
struct XDae
{
    // members: detail::XDyn dyn_;  detail::XDynJac dyn_j_;  double p_;
    state_t F(const state_t& x, const state_t& xdot, const control_t& u,
              double t) const;                     // dyn_.eval(x, xdot, u, p_, t)
    void F_jac(const state_t& x, const state_t& xdot, const control_t& u,
               double t, dyn_df_dx_t& dFdx, dyn_df_dx_t& dFxdot,
               dyn_df_du_t& dFdu) const;           // densify from dyn_j_.eval
};
```

- `c_` calibration, `#include <cassert>`, `<cmath>` and the `assert`
  removed entirely.
- The struct name (`*Ode` → `*Dae`) and the `ContinuousProblem` /
  `ImplicitRkIntegrator` template arguments in the generated file are
  updated accordingly.
- The ERK branch (explicit ODE, `vde_forw`) is untouched.
- Densify layout for `F_jac`: the capi `DynJac` struct stores
  `dFdx_[nx*nx]` (column-major: `dFdx[j*nx+i] = ∂F_i/∂x_j`),
  `dFxdot_[nx*nx]`, `dFdu_[nu*nx]` — copy row-wise into the Eigen
  blocks (same convention already used by the current `jacobian()`).

### 3.4 Consumer migration

- 4 generated IRK examples: regenerated from the updated generator.
- `examples/masses_chain/masses_chain.hpp` `MassesChainOde` →
  `MassesChainDae`: `F = masses_chain_residual(x, xdot, u, h_)`
  (hand-through), `F_jac` central-FD (h = 1e-6, 3 blocks; the twin is a
  reference implementation, FD is acceptable — the gen-equivalence test
  tolerance absorbs FD error).  No `F_hess_prod` (GN, as today).
- Test ODE models are **not** rewritten; the 5 implicit-integrator test
  files switch `ImplicitRkIntegrator<Dims, OdeX, …>` →
  `ImplicitRkIntegrator<Dims, ocp::OdeToDae<OdeX, Dims>, …>` (or a local
  alias).  The AutoDiffOde models in `ode_autodiff_4j.cpp` wrap the same
  way (dual-transparency of `OdeToDae`).
- `ContinuousProblem`, `Problem`, `rk_explicit.hpp`, `ode_autodiff.hpp`:
  no changes.

## Math appendix (second-order derivation)

Per stage s, with `z = (x_k, u)` the OCP inputs and `K` the stage
derivatives (unknowns of the stage system),
`R_s(z, K) = F(x_s(z, K), K_s, u)` with `x_s = x_k + h A[s,:] K`:

```
∂R_s/∂K_j  = h A[s][j] F_x + δ_{s,j} F_xdot
∂R_s/∂x_k  = F_x
∂R_s/∂u    = F_u
```

Newton: `G δK = −R` with `G[s][j]` as above.  First order:
`G K' = Q`, `Q = [∂R/∂x_k | ∂R/∂u]`.

Second order: `R(z, K(z)) = 0` with `G K' = −∂R/∂z = Q`.  Differentiate in
direction `v = (v_x, v_u)`:
`G (K'' v) + (∂G/∂z v) K' = −∂²R/∂z² v`, i.e.
`G (K'' v) = −∂²R/∂z² v − (∂G/∂z v) K'`, where

```
chain direction at stage s:
  η_s = (η_x, η_xd, η_u) = (v_x + h A[s,:] v_K,  v_K_s,  v_u),
  v_K = K' v
  −∂²R/∂z² v |_s = −[D Fx_s | D Fu_s],
  D Fx_s = Mxx η_x + Mxxd η_xd + Mxu η_u
  D Fu_s = Mux η_x + Muxd η_xd + Muu η_u
  ∂G[s][j]/∂z v = h A[s][j] D Fx_s + δ_{s,j} D Fxdot_s,
  D Fxdot_s = Mxx^T η_x + Mxxd η_xd + Mxu^T η_u
```

(`M*` are the Hessian blocks of `F` at stage s with the second slot
contracted with `η_s`, obtained from `nx` `F_hess_prod` calls per stage
with unit output weights; symmetry `H_{x,xdot} = H_{xdot,x}^T` gives the
`Mxx^T`-form.)  The stage-s RHS for `K'' v` is then

```
b_s = −[D Fx_s | D Fu_s]
      − D Fxdot_s K'_s  −  h Σ_j A[s][j] D Fx_s K'_j
```

Sign check: for `F = xdot − f` this reduces to
`b_s = Hv + h·HAm + h(Hv_x)Ja + h²(HAm_x)Ja`, the existing 4-term ODE
expression (same `G`, so identical `K'' v`).

and `K'' v = G^{-1} b`, `hv_x/u = h Σ_s b_s (K'' v)_s |_{x/u}`.

## Phases

### Phase 0 — Baseline snapshot (gate)
Record `cost=… iters=…` for all 19 test/example binaries and the 3
`compare.sh` outputs; save to
`docs/plans/dae_integrator_baseline.txt`.  Every later phase must
reproduce these (small Newton-iteration differences allowed; cost must
match to ~1e-6 relative).

### Phase 1 — DAE model + wrapper (gate: new unit test)
- Add `include/ocp/integrators/dae_model.hpp` (duck type, probes,
  `OdeToDae`).
- New test `tests/integrators/dae_model_4k.cpp` (+ CMake target
  `integrators_unit` already aggregates; add file to the list):
  - `OdeToDae(NonlinearOde)`: `F` / `F_jac` vs. the ODE `f` / `jacobian`
    (exact, since F = xdot − f) and `F_hess_prod` vs. `hess_prod`
    (exact for the x/u parts, xdot part zero).
  - A DAE written directly (e.g. `F = sin(x) − xdot + u xdot²`) with FD
    checks of `F` / `F_jac` / `F_hess_prod` (5-point FD, h = 1e-5,
    tol 1e-8) to prove the duck type is usable standalone.

### Phase 2 — DAE-native integrator, first-order (gate: value/jac)
- Refactor `rk_implicit.hpp` to the DAE formulation (§3.2) for
  `value` / `value_jac` / `jacobian` / warm-start / caches.
  `hess_prod` temporarily keeps the ODE formula compiled behind
  `if constexpr (false)` → returns zeros (second order lands in Phase 3).
- Migrate the 5 test files to `OdeToDae` (§3.4).
- Gates:
  - New equivalence test: for 3 ODE models (linear, nonlinear,
    time-varying) × {RadauIia2, GaussLegendre2/4}, the DAE integrator on
    `OdeToDae(ode)` reproduces the legacy ODE integrator's `value` and
    `value_jac` within 1e-12 (bit-identical modulo FMA; the Newton
    sequences should be identical: same `G` up to a sign, same residual).
  - Existing FD-based `value` / `jacobian` checks in `rk_implicit_4d` /
    `rk_multistep_4g` pass unchanged.

### Phase 3 — Second-order HVP (gate: existing hess_prod tests)
- Implement the DAE second-order formula (§3.2, appendix) using
  `F_hess_prod`; zero-HVP path when `dae_supports_hess_prod` is false.
- Gate: `rk_implicit_4d` (hess_prod vs FD), `rk_multistep_4g`
  (multi-step hess_prod vs FD), `ode_autodiff_4j` (AutoDiffOde through
  `OdeToDae`, fused-entry checks) all pass unchanged.

### Phase 4 — Generated adapters (gate: regenerated examples)
- Update the generator ODE-struct emission for the IRK branch to the DAE
  struct (§3.3); keep the ERK branch as is.
- Regenerate all 5 examples (the 3 ERK ones must be byte-identical or
  comment-only changed; verify with `git diff`).
- Wire the 4 generated IRK examples (their `.hpp` already carries the
  integrator + `ContinuousProblem`; only the model struct changes).
- Gate: `sqp_masses_chain`, `sqp_pendulum`, `sqp_unicycle`,
  `sqp_p5probe_b` pass; costs/iters match Phase 0.

### Phase 5 — Hand-written twin + full regression (gate: all green)
- `MassesChainOde` → `MassesChainDae` in
  `examples/masses_chain/masses_chain.hpp` (§3.4).
- Full regression: all 19 binaries + 3 × `compare.sh`; compare costs
  against the Phase 0 baseline.

### Phase 6 — Docs + close-out
- Update `AGENTS.md`/`ARCHITECTURE.md` references to the ODE integrator
  contract if any mention the c_ hack.
- Mark item 3 in `docs/CLEAN_UP_TODO.md` (strike-through, per the
  established pattern).
- Move this plan to `docs/plans/finished/`.

## Verification strategy

1. **Equivalence gate** (Phase 2): DAE integrator on `OdeToDae(ode)` vs.
   legacy ODE integrator, `value` + `value_jac`, 1e-12, multiple
   tableaux/stage-counts.
2. **FD cross-checks** (Phases 1–3): every new analytic quantity
   (`F`, `F_jac` blocks, `F_hess_prod`, composed `value_jac`,
   `hess_prod`) vs. 5-point central FD (h = 1e-5, tol 1e-8) at ≥ 3
   random points — same standard as the GENERATED_DERIVATIVES item.
3. **Regression**: all 19 test/example binaries; costs within ~1e-6 rel.
   of the Phase 0 baseline; 3 × `compare.sh` PASS.
4. **Build**: warning-free under `-Wall -Wextra -Werror`.
5. **Generator round-trip**: re-run on all 5 JSONs; diff shows only the
   DAE-struct changes in the 4 IRK adapters.

## Risks / open questions

- **Gauss-Legendre stage-derivative layout**: the formulation uses
  `x_s = x_k + h A[s,:] K` with the full Butcher matrix (no reliance on
  the Radau adjacent-state difference).  All 4 generated examples are
  GaussLegendre, so this path is fully exercised; RadauIia tags are
  exercised by the unit tests.  If a tag's `A` has structural zeros the
  block-Jacobian assembly skips them (existing `A[s][j] == 0` check).
- **`F_xdot` singular/non-diagonal**: the warm-start solve
  `F_xdot_s K0 = −F(x_s, 0, u)` uses a `PartialPivLU` of the `nx × nx`
  block; for the current examples `F_xdot = ±I`.  A genuinely singular
  `F_xdot` would surface as a Newton failure (non-finite `K0`) —
  documented limitation, no extra handling needed for current scope.
- **Newton count changes**: the DAE Newton (`G` contains `F_xdot` on the
  diagonal) may take a different number of iterations than the ODE
  Newton for the same ODE-residual (mathematically identical systems up
  to row scaling; iteration count is not guaranteed to match).  Phase 0
  baseline therefore compares cost, and allows small iteration drift.
- **`sqp_pendulum` / `sqp_unicycle`** currently use the generated
  adapters only through the `sqp_*` tests (no hand-written twin) — no
  extra migration beyond regeneration.
- **AutoDiffOde through `OdeToDae`**: dual-transparency of the wrapper
  must be verified in `ode_autodiff_4j` (the AD fused-entry tests run on
  the explicit integrator; the implicit side only uses `f` /
  `jacobian` / `hess_prod`, all wrapped).

## Post-implementation notes (completed)

Two bugs surfaced during the Phase 5 regression and were fixed in-tree:

- **`OdeToDae::F_hess_prod` sign**: `F = xdot - f` implies `D2 F = -D2 f`,
  so the wrapper must negate the ODE HVP (`hv_x = -hv_x; hv_u = -hv_u;`);
  it originally passed it through.  The integrator's second-order formula
  (`b_s = -[Mx|Mu] - Mxdot J_s - h sum_j A Mx J_j`, `M = G^{-1} b`,
  `HVP = sum_s h b_s w^T M_s`) was verified correct index-by-index against
  the legacy FD-verified ODE formula (commit `88abc81`) once the F-Hessian
  sign is right.
- **4k wrap check was self-consistent**: it compared `OdeToDae::F_hess_prod`
  to the ODE's `hess_prod` (same sign on both sides), so it could not catch
  the wrapper sign bug.  It now asserts `F_hess_prod == -ODE HVP`.

Final state: 19/19 binaries green, `integrators_unit` all suites green
(HVP errors at FD precision, 1e-6..1e-9), 3x `compare.sh` PASS at baseline
magnitudes.

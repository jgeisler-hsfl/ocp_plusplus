# Plan: Fused model-evaluation entries (value / value+first / value+first+hess)

Status: **implemented** (2026-10-04, branch `model-api`).
This file was the working reference for the implementation. It reworks the
model-evaluation part of the problem contract (`include/ocp/problem.hpp`):
the stage/terminal cost, the dynamics map, and the nonlinear (in)equality
constraints move from separate value / gradient / HVP entries to nested
**triplets**, following the acados grouping.

> Originally drafted on `dev` (2026-10-04 morning) at merge-base `6c524f5`.
> Master has since advanced 27 commits (JSON code generator, five generated
> examples, hessmode/CSTR/time-varying/multi-step tests, GN-Hessian docs,
> slack-relaxed residuals). The §11 re-evaluation section records what
> changed and how this plan was adapted.

## 1. Motivation

Current contract (per model part): three *disjoint* entries —
`*_value`, `*_gradient` (or `*_jacobian`), `*_hess_prod` (HVP). The SQP
solver needs, at the working iterate `(x_k, u_k)` (one SQP iteration, per
stage):

| demand (call site, current master) | value | first deriv | hess part |
|---|---|---|---|
| cost — QP assembly (`sqp.hpp:1977,2034,2089,2144,2189,2238`) | | x | x |
| cost — NLP residual (`sqp.hpp:108,195`) | | x | |
| cost — merit, current + trial points (`sqp.hpp:1431`, `globalize.hpp:581,1573`) | x | | |
| dyn — QP assembly BA + b (`sqp.hpp:2045,2057,2155,2167`) | x | x | |
| dyn — residual `res_eq` (`sqp.hpp:238`) | x | | |
| dyn — exact Hessian, if `compute_hess` (`build_dyn_hvp`, `sqp.hpp:1839`) | | | x (HVP × (nu+nx)) |
| constr — QP DC fill (`sqp.hpp:2317,2342,2446,2469,2550,2568`) | | x | |
| constr — d-mask / feasibility (`sqp.hpp:2630,2746,2851`, `sqp.hpp:3171`) | x | | |
| constr — merit trial points (`globalize.hpp:947,1171,1637`) | x | | |
| constr — exact Hessian, if `compute_hess` (`build_ineq/eq_hvp`, `sqp.hpp:1872,1909`) | | | x (HVP × (nu+nx)) |

Problems with the disjoint entries:

1. **Redundant evaluations at the working iterate.** Cost gradient is
   evaluated twice (assembly + residual); dynamics value+Jacobian twice
   (assembly + residual); constraint values+Jacobians 2-3 times (d-mask,
   DC fill, residual). All at the same `(x_k, u_k)`.
2. **HVP loops defeat shared intermediates.** The dynamics/constraint
   Hessian contribution is rebuilt with `nu+nx` HVP calls per stage
   (`build_dyn_hvp` / `build_ineq_hvp` / `build_eq_hvp`,
   `sqp.hpp:1839-1962`). Model functions are `const`/pure, so an
   implementation cannot cache the `D²f` / `D²g` setup across those
   calls — it is redone `nu+nx` times.
3. No way to share intermediates between value, first derivatives, and
   second derivatives, which is the whole point of acados's fused entries
   (CasADi emits one function computing `[f, g, H]` in a single AD pass).

Acados (verified in source) groups each model part into a **nested triplet**
`{value} ⊂ {value, first} ⊂ {value, first, hess}`:

- cost: `ext_cost_fun` / `ext_cost_fun_jac` / `ext_cost_fun_jac_hess`
  (`ocp_nlp_cost_external.c:403-628`)
- dynamics: `disc_dyn_fun` / `disc_dyn_fun_jac` / `disc_dyn_fun_jac_hess`
  (`ocp_nlp_dynamics_disc.c:644-687`); the hess output is the
  **multiplier-contracted** `(nu+nx)²` matrix with `π` passed in as input
  (`casadi_function_generation.py:233-247`)
- nonlinear constraints: `nl_constr_phi_o_r_fun` / `nl_constr_r_fun_jac` /
  `..._fun_phi_jac_ux_z_phi_hess_r_jac_ux`
  (`ocp_nlp_constraints_bgp.c:1263,1418`)

Acados never re-evaluates: its NLP residual reads the cost gradient stored
during QP assembly (`ocp_nlp_common.c:3760`), and merit at line-search trial
points uses the value-only entry.

## 2. Resolved decisions

| # | Decision |
|---|---|
| D1 | Nested triplet per model part: `value` / `value_grad` (cost) or `value_jac` (dynamics, constraints) / `value_grad_hess` / `value_jac_hess`. No bitmasks, no flag arguments. |
| D2 | Multi-output entries return `void`; the value is an out-ref like the other outputs (house style: multi-output = `void` + out-refs; single-output = return by value). Return slot left free for future signaling (`Status`). |
| D3 | Dynamics/constraint "hess part" = **full multiplier-contracted matrix in one call** (`M v = Σ_i lam_i D²(·)_i v`), replacing the HVP entries. Multiplier is an input argument (acados style). |
| D4 | Model-side Hessians stay `[x; u]`-layout (documented convention); the solver permutes into the QP's `(u; x)` layout, exactly like the cost Hessian today. Terminal constraint Hessian is `nx × nx`. |
| D5 | Stage / terminal remain **separate entries** (terminal is state-only, different types). Equality groups mirror inequality groups. Linear and box groups are unchanged (no derivatives, Hessian ≡ 0). |
| D6 | Full symmetric square Hessian matrices for now (no upper/lower-triangle storage); symmetry is stated in the contract. Triangle storage is a follow-up, orthogonal to this change (§9). |
| D7 | **Deferred to a follow-up** (was: solver reuses quantities assembled at the working iterate for the residual check). Re-evaluation (§11.4): the current `compute_nlp_residuals` also evaluates the dynamics/constraint *Jacobians* (res_stat), so true reuse would need a per-stage workspace holding Jacobian blocks, and the free-function signature `compute_nlp_residuals(problem, sol)` (used standalone by tests `residuals_2c`, `driver_2g`) would have to change. Instead: the residual keeps its signature and simply switches to the fused non-hess entries (`stage_cost_value_grad`, `dynamics_value_jac`, `*_value_jac`), which already removes the value/derivative duplication *inside* the residual. Assembly→residual duplication stays for now. |
| D8 | `Dims::has_dynamics_hess_prod` → `Dims::has_dynamics_hess`, `Dims::has_constr_hess_prod` → `Dims::has_constr_hess` (the entries are no longer HVPs). `false` ⇒ map is linear, the hess entry is absent and never called. All doc mentions (problem.hpp Dims block + notes, `GN_HESSIAN_PLAN.md`, `sqp.hpp` `SqpOptions` comment, generator comment, phase plans) must follow the rename. |
| D9 | ODE / integrator level is **unchanged** (`ode_model.hpp`, `rk_explicit.hpp`, `rk_implicit.hpp`: `value` / `jacobian` / `hess_prod` stay, including the time-varying `t` argument and `NumSteps` added after the branch point). Only the `ContinuousProblem` adapter is remapped to the new Problem-level entries (§5). Note: `ode_autodiff.hpp` exists only on the `dev` branch (cppduals work, untouched by this change); its comment rename happens when dev is merged. |
| D10 | Existing concrete problems keep `has_dynamics_hess = true` / `has_constr_hess = true` with zero matrices where the map is linear (they already return zero HVPs today), so the fused path stays exercised by the existing test suite and all results must stay numerically identical. `hessmode_5a` (nonzero dynamics HVP) and `cstr_4i` (nonlinear ODE through `ContinuousProblem`) give genuine FD-checkable coverage of the contracted-Hessian path. |
| D11 | **New.** Soft-penalty method names keep the `_constr_` infix (`stage_inequality_constr_soft_penalty`, ...) — explicitly out of scope for this change despite the value-entry rename, to keep the diff bounded. A later cleanup may rename to `stage_inequality_soft_penalty` etc. |

## 3. New contract

All new/changed entries, with the existing (unchanged) ones for reference.
Terminal mirrors stage, state-only.

```cpp
// --- cost (stage k) ---
Scalar  stage_cost_value(int k, const state_t& x, const control_t& u) const;       // unchanged (merit, trial points)
void    stage_cost_value_grad(int k, const state_t& x, const control_t& u,
                              Scalar& value, stage_grad_t& grad) const;            // NEW (replaces stage_cost_gradient)
void    stage_cost_value_grad_hess(int k, const state_t& x, const control_t& u,
                                   Scalar& value, stage_grad_t& grad,
                                   stage_hess_t& hess) const;                      // NEW (replaces stage_cost_hessian)

// --- terminal cost (x only) ---
Scalar  terminal_cost_value(const state_t& x) const;                               // unchanged
void    terminal_cost_value_grad(const state_t& x, Scalar& value,
                                 term_grad_t& grad) const;                         // NEW
void    terminal_cost_value_grad_hess(const state_t& x, Scalar& value,
                                      term_grad_t& grad, term_hess_t& hess) const; // NEW

// --- dynamics (stage k; (x, u) -> x_{k+1}) ---
state_t dynamics_next_state(int k, const state_t& x, const control_t& u) const;    // unchanged (SOC trial points, merit gaps)
void    dynamics_value_jac(int k, const state_t& x, const control_t& u,
                           state_t& x_next, dyn_df_dx_t& df_dx,
                           dyn_df_du_t& df_du) const;                              // NEW (replaces dynamics_jacobian)
void    dynamics_value_jac_hess(int k, const state_t& x, const control_t& u,
                                const state_t& lam, state_t& x_next,
                                dyn_df_dx_t& df_dx, dyn_df_du_t& df_du,
                                dyn_hess_t& hess) const;                           // NEW (replaces dynamics_hess_prod)
//   hess v = sum_i lam_i D²f_i v,   dyn_hess_t = Matrix<scalar_t, nx+nu, nx+nu>, [x;u] layout

// --- stage nonlinear inequality (k); equality mirrors with eq_t / eq_de_dx_t / ... ---
ineq_t  stage_inequality_value(int k, const state_t& x, const control_t& u) const; // renamed from stage_inequality_constr
void    stage_inequality_value_jac(int k, const state_t& x, const control_t& u,
                                   ineq_t& g, ineq_dg_dx_t& g_dx,
                                   ineq_dg_du_t& g_du) const;                      // NEW (replaces ..._constr_jacobian)
void    stage_inequality_value_jac_hess(int k, const state_t& x, const control_t& u,
                                        const ineq_t& lam, ineq_t& g,
                                        ineq_dg_dx_t& g_dx, ineq_dg_du_t& g_du,
                                        constr_hess_t& hess) const;               // NEW (replaces ..._constr_hess_prod)
//   hess v = sum_i lam_i D²g_i v,   constr_hess_t = Matrix<scalar_t, nx+nu, nx+nu>, [x;u] layout

// --- terminal nonlinear (in)equality (x only) ---
ineq_t  terminal_inequality_value(const state_t& x) const;
void    terminal_inequality_value_jac(const state_t& x, ineq_t& g,
                                      ineq_term_dg_dx_t& g_dx) const;
void    terminal_inequality_value_jac_hess(const state_t& x, const ineq_t& lam,
                                           ineq_t& g, ineq_term_dg_dx_t& g_dx,
                                           term_constr_hess_t& hess) const;
//   hess v = sum_i lam_i D²g_i v (state only), term_constr_hess_t = Matrix<scalar_t, nx, nx>
//   (terminal equality mirrors with eq_term_t / eq_term_de_dx_t)

// --- unchanged: initial_state, stage/terminal linear specs, box specs,
//     *_constr_soft_penalty (D11) ---
```

New type aliases in `Problem`:

```cpp
using dyn_hess_t         = Eigen::Matrix<Scalar, nx + nu, nx + nu>;  // [x;u], multiplier-contracted
using constr_hess_t      = Eigen::Matrix<Scalar, nx + nu, nx + nu>;  // [x;u], multiplier-contracted
using term_constr_hess_t = Eigen::Matrix<Scalar, nx, nx>;            // state only
```

Contract notes to add to the header block:

- The hess parts are **multiplier-conditioned**: the supplied multiplier is
  the net KKT multiplier of the corresponding group (dynamics:
  `lambda_dyn[k]`; constraints: `lambda_ineq_stage[k]` /
  `lambda_eq_stage[k]` / terminal multipliers). The implementation may
  compute per-row Hessians and contract, or take the adjoint path — the
  contract only fixes the output.
- Cost Hessians may be exact, Gauss-Newton, or any valid approximation
  (unchanged, ARCHITECTURE.md §6); they are symmetric by contract.
- When `has_dynamics_hess` / `has_constr_hess` is false the corresponding
  `*_value_jac_hess` entry need not be implemented and is never called
  (map is linear; the Hessian contribution is zero). `has_dynamics_hess`
  keeps the EXACT / GAUSS_NEWTON semantics documented in `GN_HESSIAN_PLAN.md`
  and the Dims block (rename only, no semantic change).

## 4. Solver changes

`include/ocp/solvers/acados/sqp.hpp` (line numbers = current master):

1. **Assembly** (`assemble_first` :1968, `assemble_path` :2080,
   `assemble_term` :2179):
   - cost: one `stage_cost_value_grad_hess` / `terminal_cost_value_grad_hess`
     call replaces the separate `stage_cost_hessian` (:1977, :2089, :2189) +
     `stage_cost_gradient` (:2034, :2144, :2238); `value` + `grad` outputs
     feed the QP `grad`/`hess` (permuted `[x;u] → (u;x)` element loop as
     today).
   - dynamics: `if constexpr (P::has_dynamics_hess)` + `opts_.compute_hess`
     gate (:1992, :2102) → `dynamics_value_jac_hess(k, x, u,
     sol.lambda_dyn[k], ...)`; else `dynamics_value_jac`. `st.hess` (dyn
     part), `st.BA`, `st.b` all come from this single call; the separate
     `dynamics_next_state` calls (:2057, :2167) and `dynamics_jacobian`
     calls (:2045, :2155) are removed (value is the `x_next` out-ref,
     Jacobian the `df_dx`/`df_du` out-refs).
   - constraints (ineq + eq, stage + terminal): analogous —
     `*_value_jac_hess` (multiplier = stage/terminal lambda) or
     `*_value_jac` when `P::has_constr_hess == false` or
     `!opts_.compute_hess` (:2003, :2113, :2199). The `g`/`e` out-refs feed
     the d-mask fill, so the separate value calls in
     `fill_d_mask_first/path/term` (:2630-2632, :2746-2748, :2851-2853) are
     removed and `fill_dc_*` / `fill_d_mask_*` receive the already-computed
     Jacobians/values as parameters (pass-through; they run after
     `assemble_*` in the `assemble_qp` driver).
   - Hessian contribution: insert the returned `constr_hess_t` /
     `dyn_hess_t` with the `[x;u] → (u;x)` permutation, replacing the
     `build_*_hvp` matrix-build loops.
   - **Delete** `build_dyn_hvp` (:1839) / `build_ineq_hvp` (:1872) /
     `build_eq_hvp` (:1909) / `build_term_hvp` (:1946).
   - `check_first_degeneracy` (:3162, called from `assemble_first` :2071)
     keeps its own `stage_inequality_value` + `stage_inequality_value_jac`
     calls (fixed-x0 path, once per iteration; the stage-0 fused data is
     not available at that point yet).

2. **Residual** (D7 deferred): `compute_nlp_residuals` (:87) keeps its
   signature `compute_nlp_residuals(problem, sol, sl, tau_min)` (tests
   `residuals_2c`, `driver_2g` call it standalone) but switches its model
   calls to the fused non-hess entries:
   - `stage_cost_gradient` (:108) → `stage_cost_value_grad` (drop `value`)
   - `terminal_cost_gradient` (:195) → `terminal_cost_value_grad`
   - `dynamics_jacobian` + `dynamics_next_state` (:112, :238) → one
     `dynamics_value_jac`
   - `stage_inequality_constr_jacobian` + `stage_inequality_constr`
     (:118, :269, :551) → one `stage_inequality_value_jac` (per stage,
     computed once, shared by res_stat / res_ineq / res_comp — note res_comp
     currently re-evaluates the value at :551; use the stored `g`)
   - `stage_equality_constr_jacobian` + `stage_equality_constr`
     (:123, :283) → `stage_equality_value_jac`
   - terminal analogues (:201, :207, :382, :395, :649) →
     `terminal_{inequality,equality}_value_jac`

3. **Merit / cost**: `raw_cost` (:1425) and `globalize.hpp` merit/funnel
   keep the value-only entries — names unchanged (`stage_cost_value`,
   `terminal_cost_value`, `dynamics_next_state`). Constraint values at
   trial points rename to `stage_inequality_value` /
   `stage_equality_value` / terminal versions (`globalize.hpp:947,949,
   1075,1077, 1171,1173, 1284,1286, 1637,1639, 1744,1746`).

4. **SOC** (:1656-1830): unchanged in substance — `soc_rewrite_rhs`
   (:1689) and `soc_rewrite_d_*` (:1745-1747, :1798-1800) call the
   value-only entries at trial iterates; renames only.

5. **Flags / docs**: rename `P::has_dynamics_hess_prod` /
   `has_constr_hess_prod` at :1992, :2003, :2102, :2113, :2199; update the
   `SqpOptions::compute_hess` comment (:718-721).

`include/ocp/solvers/acados/globalize.hpp`: value-entry renames only
(table in 4.3); no derivative entries are used there.

`include/ocp/solvers/hpipm/` and `acados/{qpscaling,regularize}.hpp`:
**unchanged** (QP data model and post-assembly passes do not touch model
functions).

## 5. Integrator adapter

`include/ocp/integrators/continuous_problem.hpp` (post-4h/4g: ODE is
time-varying `f(x,u,t)`, `NumSteps` supported; adapter forwards
`t_k = k * h_`):

- `dynamics_next_state` — unchanged.
- `dynamics_jacobian` → `dynamics_value_jac`: call `integ_.value` +
  `integ_.jacobian` (two integrator calls; a fused integrator-level entry
  is a follow-up, §9).
- `dynamics_hess_prod` → `dynamics_value_jac_hess`: value + Jacobian as
  above, then build the `(nx+nu)²` `[x;u]` matrix with `nx + nu` calls to
  `integ_.hess_prod` (columns: `v = e_j`, `w = lam`). This unit-vector
  loop moves from the solver (`build_dyn_hvp`) into the adapter — same
  number of integrator HVPs, but now one `const` call where the ODE can
  cache stage data (e.g. a dual-based `hess_prod` setup on the `dev`
  branch).
- Update the file's header comment (lines 8-23) and the `has_dynamics_hess`
  wording.

`ode_model.hpp`, `rk_explicit.hpp`, `rk_implicit.hpp`, `butcher.hpp`:
**unchanged** (ODE-level API, incl. `t` and `NumSteps`).

## 5b. Generator (NEW — not in the original plan)

`tools/acados2ocp_pp.py` (2008 lines, JSON-driven) is now the primary
producer of concrete problems (five generated examples, regenerated by
`bfad1bd`). It emits the old contract names and must be updated:

- Cost dispatch f-strings: `stage_dispatch` (:1327-1369) emits
  `stage_cost_value` / `stage_cost_gradient` / `stage_cost_hessian`;
  `term_dispatch` (:1371-1384) the terminal analogues. Rewrite to emit
  `stage_cost_value` (unchanged), `stage_cost_value_grad`,
  `stage_cost_value_grad_hess` (and terminal). The per-scope bodies
  (`cost_scope_body`, :940-1302: `cp_*`/`c0_*`/`ct_*` for LINEAR_LS,
  NONLINEAR_LS, EXTERNAL) are unchanged — only the wrapping signatures
  become `void` + out-refs.
- Constraint templates (:1407-1661): `stage_inequality_constr` →
  `stage_inequality_value`, `stage_inequality_constr_jacobian` →
  `stage_inequality_value_jac` (gain a `g` out-ref, fed by the existing
  `hval` helper), same for equality and both terminal groups. Soft-penalty
  method names unchanged (D11).
- Dims flag emission (:814-815): `has_dynamics_hess = false;`
  `has_constr_hess = false;` (values stay false).
- Generated-file header preamble comment (:1805): flag-name update.
- No new templates needed today: all five generated examples set both
  flags false, and the generator emits no dynamics/constraint HVP. When a
  future JSON enables `has_dynamics_hess`/`has_constr_hess`, the new
  `*_value_jac_hess` emission must be written against the triplet
  contract (GENERATOR_PLAN.md P5 item).
- **Regenerate all five examples** (manual CLI, no build hook):
  `masses_chain`, `pendulum_on_cart`, `unicycle`, `p5probe`, `p5probe_b`
  (specs: `examples/*/acados_codegen/acados_ocp_nlp.json`; commands
  documented in `CMakeLists.txt` :69-181). The CMake regeneration
  comments need no change (method names don't appear in them).
- `tools/GENERATOR_PLAN.md` (:109-124, :164, :214, :319-320) mentions the
  old names → docs pass.

## 6. Consumer migrations (mechanical, per file)

Hand-written examples:

| file | work |
|---|---|
| `examples/double_integrator/double_integrator.hpp` | cost → 2 triplets (quadratic, trivial); `dynamics_jacobian` → `dynamics_value_jac`; `dynamics_hess_prod` → `dynamics_value_jac_hess` (zero matrix; D10); ineq value/jac/hess → triplet (zero Hessian); flag renames (2) |
| `examples/double_integrator/main.cpp` | FD sanity: value for FD, `stage_cost_value_grad` for the gradient check, `stage_cost_value_grad_hess` for the Hessian check (:76, :92-117) |
| `examples/mass_spring/mass_spring.hpp` + `main.cpp` | same pattern (zero-matrix entries, flags stay true per D10) |
| `examples/mass_spring_ref/mass_spring_ref.hpp` | two problem classes (Box, Term), same pattern; Term class also has a `terminal_equality_constr_hess_prod` → `terminal_equality_value_jac_hess` (zero) |
| `examples/masses_chain/masses_chain.hpp` + `main.cpp` | hand-written reference problem (used by `gen_equivalence`): same pattern |

Generated examples (via §5b, one commit after the generator):

| file | work |
|---|---|
| `examples/masses_chain/generated/MassesChainGen.hpp` | regen |
| `examples/pendulum_on_cart/generated/PendulumGen.hpp` | regen |
| `examples/unicycle/generated/UnicycleGen.hpp` | regen |
| `examples/p5probe/generated/P5probe.hpp` | regen (NONLINEAR_LS cost, stage/terminal ineq) |
| `examples/p5probe_b/generated/P5probeB.hpp` | regen (EXTERNAL FD cost) |

Tests:

| file | work |
|---|---|
| `tests/sqp/hessmode_5a.cpp` | embedded problem (both Dims sets, **nonzero** dynamics HVP `w(0)*Ts*v_u(0)` — the EXACT-mode FD asset): cost triplets, `dynamics_value_jac(_hess)`, ineq/terminal-eq triplets, flag renames (:65-66, :92-93, :148, :216, :267, :347) |
| `tests/sqp/assemble_2f.cpp` | three embedded problems (QuadTest :93, DegProbe :262, SoftBoxProbe :390): cost triplets + flags |
| `tests/sqp/driver_2g.cpp` | embedded problem (:320-373): cost triplet; `compute_nlp_residuals` call (:185) unchanged |
| `tests/sqp/residuals_2c.cpp` | standalone `compute_nlp_residuals` calls (:123, :182, :223) unchanged (signature stable per D7-defer); verify |
| `tests/sqp/apply_step_2b.cpp` | direct `dynamics_jacobian` call → `dynamics_value_jac` (:60 area); flag renames |
| `tests/sqp/merit_2d.cpp` | flag renames only (:71-72); value entries unchanged |
| `tests/hpipm/suite_1i.cpp` | flag renames only (Dims :110, :150; empty problem shells) |
| `tests/sqp/{regularize_2a,adaptive_lm_3g,warm_start_3c,timeout_3f,qpscaling_3h,funnel_3i,soc_3j}` | use example problems; verify no direct model calls; flag renames in any local Dims |
| `tests/sqp_double_integrator/sqp_double_integrator.cpp` | :93-119, :264-265 direct entry calls (cost value/grad/hess, dynamics) → triplets |
| `tests/sqp_mass_spring/sqp_mass_spring.cpp` | :75, :95 same |
| `tests/sqp_acados_ref/sqp_acados_ref.cpp` | uses MassSpringRef; verify |
| `tests/sqp_masses_chain/sqp_masses_chain.cpp` | :145, :180 `stage_cost_gradient` → `stage_cost_value_grad` |
| `tests/sqp_masses_chain/gen_equivalence.cpp` | includes both problem headers; verify (no direct model calls found) |
| `tests/sqp_pendulum/sqp_pendulum.cpp`, `tests/sqp_unicycle/sqp_unicycle.cpp` | generated problem; driver uses only `initial_state`/`cost_reference`/solver — verify |
| `tests/sqp_p5probe/sqp_p5probe.cpp` | :77, :78, :86 value entries (names unchanged) + :132 `terminal_inequality_constr` → `terminal_inequality_value` |
| `tests/sqp_p5probe_b/sqp_p5probe_b.cpp` | :46, :48, :51 value entries; verify |
| `tests/integrators/continuous_problem_4c.cpp` | `dynamics_next_state` (:217), `dynamics_jacobian` (:239) → `dynamics_value_jac`; flag renames (:33-34) |
| `tests/integrators/sqp_continuous_4e.cpp` | two embedded problems: cost triplets; `terminal_equality_constr_jacobian/hess_prod` (:196-203, :315-322) → `value_jac(_hess)`; flag renames (:50-51) |
| `tests/integrators/cstr_4i.cpp` | CSTR problem class (:41-42 flags, :547-643 cost/terminal-eq entries) → triplets; ODE-level `CstrOde::hess_prod` and integrator checks **unchanged** (D9) |
| `tests/integrators/{integrators_4a,rk_explicit_4b,rk_implicit_4d,rk_multistep_4g,time_varying_4h,integrators_main}` | ODE/integrator-level API only — **no change** |

## 7. Docs

- `include/ocp/problem.hpp`: interface-contract block (L252-379), Dims
  example comment (L25-38), notes (L344-379), all new/changed stubs with
  `static_assert(false, ...)` messages, the `has_*_hess` members (L401-414).
- `include/ocp/solvers/acados/GN_HESSIAN_PLAN.md`: flag-name mentions →
  `has_dynamics_hess` (rename only; the EXACT/GN semantics table stays).
- `include/ocp/solvers/acados/SQP_PLAN.md` (:320-333 area) +
  `SQP_PHASE3_PLAN.md`: add a "superseded by MODEL_API_PLAN.md" note at the
  model-entry mentions; do not rewrite history.
- `include/ocp/integrators/SQP_PHASE4_PLAN.md`: old-name mentions.
- `tools/GENERATOR_PLAN.md`: Ode/Problem contract mentions (:109-124,
  :164, :214, :319-320).
- `ARCHITECTURE.md` §6 stays valid (no derivative-source distinction).
- `AGENTS.md`: no cost-entry conventions to update (re-check at commit time).
- `TODO.md`: roadmap — not edited without asking (AGENTS.md rule).

## 8. Verification gates (per commit + final)

1. `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build` —
   warning-free under `-Wall -Wextra -Werror`.
2. Run **all** binaries: `double_integrator`, `mass_spring`,
   `masses_chain`, `masseschaingen_gen`, `sqp_masses_chain`,
   `sqp_masses_chain_gen_equiv`, `pendulum_on_cart`, `unicycle`,
   `sqp_pendulum`, `sqp_unicycle`, `sqp_p5probe`, `sqp_p5probe_b`,
   `qp_dim`, `qp_unit`, `sqp_unit` (incl. `hessmode_5a`),
   `sqp_double_integrator`, `sqp_mass_spring`, `sqp_acados_ref`
   (acados parity: cost rel. err < 1e-4, must be unchanged),
   `integrators_unit` (incl. `cstr_4i`, `time_varying_4h`,
   `rk_multistep_4g`), `sqp_continuous`.
3. Expected outcome: **numerically identical** results on all existing
   tests — the assembled QP matrices must be bit-close to today's (the
   Hessian contributions `Σ lam_i D²(·)_i` are the same numbers, just
   computed inside the problem entry instead of the solver's HVP loop).
   Any deviation ⇒ layout/permutation bug (check `[x;u] → (u;x)` in 4.1
   and the terminal `nx × nx` path).
4. FD spot checks that must stay green (they exercise the new
   contracted-Hessian path end-to-end): `hessmode_5a` EXACT mode (nonzero
   dynamics HVP vs GN), `cstr_4i` (nonlinear ODE HVP through
   `ContinuousProblem`), `sqp_masses_chain_gen_equiv` (generated vs
   hand-written QP equivalence).
5. Generator diff check: after regeneration, `git diff` of the five
   generated headers must show only the expected renames/signature
   changes (no numeric/structural drift).
6. Call-count spot check (optional, debug build): instrument one example
   to count `stage_cost_value_grad_hess` calls per SQP iteration — expect
   exactly `N + 1` (one per stage incl. terminal) at the working iterate.

## 9. Out of scope / follow-ups

- **D7 residual reuse** (workspace cache of assembly outputs;
  `compute_nlp_residuals` reads it instead of re-calling the fused
  entries): deferred (see D7 rationale).
- **Triangle-only Hessian storage** (symmetric, upper/lower triangle):
  deferred (D6). Would need a `SymMatrix<N>` wrapper and QP-side support.
- **Fused integrator entry** (value + Jacobian in one `Integ::` call):
  today `dynamics_value_jac` issues two integrator calls; add
  `Integ::value_jac` later.
- **Cost scaling** (acados cost-module `scaling` factor): not in our
  contract.
- **Soft-penalty renames** (`*_constr_soft_penalty` → `*_soft_penalty`):
  cleanup after the value-entry rename (D11).
- **Generator HVP emission**: writing the `*_value_jac_hess` templates for
  JSON models with `has_dynamics_hess`/`has_constr_hess = true` (all
  current JSONs are false; GENERATOR_PLAN P5).
- `dev` branch `ode_autodiff.hpp` comment rename (cppduals work): happens
  when `dev` is merged on top of this.

## 10. Suggested commit order

Branch: `model-api` off `master` (`bfad1bd`). The `dev` branch
(cppduals + original plan draft) is left untouched.

1. `docs:` re-evaluated MODEL_API_PLAN.md (this file).
2. `problem:` new contract — types (`dyn_hess_t`, `constr_hess_t`,
   `term_constr_hess_t`), flag renames, triplet stubs, header block.
   (Build breaks for all consumers; expected, resolved by 3-7.)
3. `solver:` `sqp.hpp` / `globalize.hpp` on the new entries; delete
   `build_*_hvp`; residual on fused non-hess entries.
4. `integrator:` `continuous_problem.hpp` adapter (`dynamics_value_jac`,
   `dynamics_value_jac_hess` with the unit-vector HVP loop; header
   comment).
5. `tools:` generator templates updated + regenerate the five generated
   examples (diff-check per §8.5).
6. `example:` migrate hand-written problems (`double_integrator`,
   `mass_spring`, `mass_spring_ref`, `masses_chain`) + `main.cpp` FD
   checks. Full build green from here.
7. `test:` migrate embedded test problems + flag renames (§6 table).
8. `docs:` plan cross-references (GN_HESSIAN_PLAN, SQP_PLAN, phase plans,
   GENERATOR_PLAN).

Each of 3-7 lands build-green on the files it touches; the full build is
green after 7 (intermediate states may not compile end-to-end between 2
and 7 — acceptable because the commits land as one reviewed series).

## 11. Re-evaluation vs the branch point (`6c524f5` → `bfad1bd`)

What changed on master after the plan was drafted, and the plan's response:

1. **JSON code generator + 5 generated examples** (`7abb7a2`, `fe05063`,
   `f87cbbd`, `bfad1bd`): the original plan (§6) only listed hand-written
   examples. Response: new §5b (generator templates + regeneration) and
   §6 rows for the five generated headers and their test drivers.
2. **hessmode_5a** (`104dc70`, `d1fc2e5`, `ccad3ed`): embedded problem
   with a *nonzero* dynamics HVP and EXACT/GN flag pairs; also new
   `GN_HESSIAN_PLAN.md` with the EXACT/GN semantics. Response: added to
   §6 (migration) and §8.3 (primary FD asset for the new
   `dynamics_value_jac_hess`); D8 rename now covers the GN docs.
3. **GN/EXACT semantics documented** (`ccad3ed`): the
   `has_dynamics_hess_prod` doc block in `problem.hpp` was rewritten.
   Response: the D8 rename must preserve that wording (rename only, no
   semantic change) — noted in §3 and §7.
4. **Slack-relaxed NLP residuals + fixed-x0 box-row guard** (`b9346fe`):
   `compute_nlp_residuals` gained the `SqpSlacks*` / `tau_min` arguments
   and a `res_comp` block that re-evaluates constraint values (:551,
   :649). Response: D7 re-scoped (deferred) — full assembly→residual
   reuse would now also need Jacobian-block storage and a signature
   change; the residual instead switches to the fused non-hess entries and
   computes each `value_jac` once per stage, shared by res_stat /
   res_ineq / res_comp (§4.2).
5. **Time-varying ODE (4h) + multi-step (4g)** (`51b1e47`, `fe716d7`,
   `a99e6a9`, `15e36da`): the ODE contract gained `t` and `NumSteps`;
   `ContinuousProblem` forwards `t_k = k*h`. Response: D9/§5 updated —
   the adapter's new fused entries forward `t` exactly like today's; the
   ODE level stays untouched.
6. **CSTR example (4i)** (`a848729`): nonlinear ODE with hand-derived
   analytic `hess_prod`, end-to-end through `ContinuousProblem` +
   `SqpSolver` with both hess flags true. Response: added to §6 and §8.3
   (second genuine-coverage asset for the contracted-Hessian path).
7. **Stale line numbers**: every `sqp.hpp` / `globalize.hpp` reference in
   the original plan is off by up to ~200 lines (385 lines changed).
   Response: §1 table, §4, and §6 re-anchored to current-master line
   numbers (verified 2026-10-04).
8. **`ode_autodiff.hpp`** (`ode_autodiff` exists only on `dev` via the
   cppduals work `6254aaa`): D9's file list corrected to the master
   integrator set; the dev-side comment rename is a merge-time task (§9).

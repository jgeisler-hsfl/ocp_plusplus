# Plan: parameter-vector concept + NLP-level parameter sensitivity (Phase 5, item 5i)

Status: **DONE (2026-10-14)** — full `*_sens_p` interface (box/linear/
initial included), staged-only `sens_param` (dense/block seed condensing
deferred to 5j), DoubleIntegrator `np = 6` with
`p = (Ts, v_max, wq, wv, wa, wf)`. Unblocks the open item
`SQP_HPIPM_PARITY_PLAN.md` §9.3 ("NLP-level (parameter) sensitivities:
blocked on a parameter-vector concept in `problem.hpp`"). This doc scopes
the interface + a staged `AcadosSolver::sens_param`; dense/block seed
condensing are deferred to 5j.

## 0. Implementation outcome (added 2026-10-14, after the build-out)

- **FD verification (DI N=10, central FD of the full SQP re-solve):**
  max |du/dp − FD| / |dx/dp − FD|: p_0 (Ts) 1.7e-6 (h = 1e-5), p_1 (v_max)
  1.1e-12, p_2 (wq) 6.1e-10, p_3 (wv) 3.2e-11, p_4 (wa) 1.3e-8,
  p_5 (wf) 5.1e-11. All pass (`tests/sqp/sens_param_5i.cpp`).
- **Active-set gating (design decision, plan §2.3 below):** the d-seed on
  constraint sides that are inactive at the converged solution is dropped
  (`fill_param_seed` masks sides with converged slack `t >= 1e-3`).
  Without it, a parameter that only moves an *inactive* constraint (here
  `v_max`, margin ~1.5) leaks an O(mu_exit) artifact through the
  nearly-singular barrier row (observed: dx_0 = 1e-6, du = 8.5e-6 instead
  of the exact 0); with gating the response is ~1e-12. The first-order
  sensitivity of an inactive constraint's value / bound is exactly zero
  (active-set theory), so the gated result is the physically correct one;
  the `(dJ/dp)' lambda` grad couplings are left ungated (for an inactive
  row lambda ~ mu/t ~ O(mu_exit), negligible).
- **FD step / tolerance calibration:** `Ts` (p_0) enters the dynamics
  quadratically, so central-FD truncation at h = 1e-4 is ~5e-5; the test
  uses h = 1e-5 (observed 1.7e-6) with tol 1e-5 for p_0 only. All other
  parameters enter linearly: h = 1e-4·max(1,|p|), tol 1e-6.
- **Eigen portability:** the installed Eigen (< 3.3) has no `.reshaped()`,
  and 1x1 products are matrices (no vector methods): the `(dJ/dp·dp)' lam`
  accumulation therefore uses the scalar-loop helper
  `detail::jacp_t_lam_add` (acados.hpp) instead of `reshape`/`dot`.
- **Interface fix found during review:** the value p-derivatives needed
  their own `(ng, np)` / `(ne, np)` / `(ng_t, np)` / `(ne_t, np)` aliases
  (`ineq_dg_dp_t`, `eq_de_dp_t`, `ineq_term_dg_dp_t`,
  `eq_term_de_dp_t`); the stubs had initially reused the value types
  (`ineq_t` etc., which are `(ng, 1)`).

## 1. Goal

First-order sensitivity of the *converged OCP solution* w.r.t. a parameter
vector `p`:

- API: `AcadosSolver::sens_param(problem, sol, dp, dsol)` — reassemble the
  staged QP at the converged SQP iterate (`sol`), build the QP RHS seed from
  the problem's new `*_sens_p` derivative entries, and run
  `HpipmQpSolver::sens_forward` (the Phase-4 / 5f seed mechanism).
- Returns the QP-space response `dsol` (ux / pi / lam / t), i.e.
  `d(x*, u*)/dp · dp` to first order, in the same layout as `sens_x0`.
- Test: `tests/sqp/sens_param_5i.cpp` (DoubleIntegrator, N=10, central FD of
  the full SQP re-solve for each parameter; rel. err < 1e-6), mirroring
  `sens_x0_5f.cpp`.

## 2. Formulation — why the seed needs no Hessian p-derivative

At a converged SQP iterate `(z*, λ*)` at parameter `p*`, the SQP QP *is* the
exact linearization of the NLP (the QP Hessian `H_k` is the Lagrangian Hessian
`∇²_{zz} L_NLP`, the QP `BA`/`DC` are the constraint Jacobians, `b`/`d` are
the constraint value offsets). The NLP KKT residuals, as functions of
`(z, p)`, are:

- stationarity:  `∇_z L_cost(z,p)  +  BA'(z,p) μ_dyn  +  DC'(z,p) μ_c  = 0`
- dynamics:       `BA(z,p) z  +  b(z,p)  −  x_{k+1}  = 0`
- box/ineq/lin:  `DC(z,p) z  +  d(z,p)  ≥ 0`   (offset form, `d_lo = lo − w`, `d_hi = w − hi`)

Differentiate w.r.t. `p` **holding `z*` and `λ*` fixed** (standard
linearization; valid under LICQ / non-degeneracy at the active set):

```
[ H  BA' ] [ dz ]    [ ∂p ∇L_cost  +  (∂p BA)' π  +  (∂p DC)' Δλ ]
[ BA  DC ] [ dλ ]  =  [ ∂p b ]
                    [ ∂p d ]
```

Key point: `H` (the Lagrangian Hessian) is the **LHS matrix** — it is *not*
differentiated, so no third-order `∂p ∇²L` term appears. The only
second-derivative information needed is through the *first* derivatives of the
values (`∂p f`, `∂p g`, `∂p e`) and the *Jacobian* p-derivatives
(`∂p A`, `∂p B`, `∂p g_dx`, …). This is exactly why HPIPM's
`d_ocp_qp_seed` carries only the RHS fields (`g`, `b`, `d`, `m`) and **no
Hessian seed**: NLP parameter sensitivity reuses the converged KKT LHS.

The existing `HpipmQpSolver::sens_solve` (hpipm.hpp, 5f) builds a QP with the
converged LHS (`hess`, `BA`, `DC`, `d_mask` from `in`) and the seed as the new
RHS (`grad`, `b`, `d`, `m`); its solution equals the sensitivity `(dz, dλ)`.
So `sens_param` is: assemble the converged QP → fill `QpSeed` from the
problem's `*_sens_p` outputs → `sens_forward`.

### 2.1 Per-field seed recipe (staged layout: `z_k = (u; x; s)`)

**Sign conventions verified against `compute_residuals` / `globalize.hpp`:**
`res_g = H z + g + DC'(λ_hi − λ_lo) + BA' π − pin(π_prev)`;
`res_b = BA z + b − x_{k+1}`; and `sol.lambda_dyn[k] = −π_qp[k]`
(globalize.hpp:148). The seeds below are in the **QP multiplier convention**
(π, Δλ = λ_hi − λ_lo):

**`seed.grad` (stationarity RHS), per stage, `(u; x; s)` layout:**

```
(u block) =  ∂p(∂L/∂u)  +  (∂p B_k)' π_k  +  (∂p g_du)' Δλ_ineq[k]
            +  (∂p e_du)' Δλ_eq[k]  +  (∂p B_lin)' Δλ_lin[k]
(x block) =  ∂p(∂L/∂x)  +  (∂p A_k)' π_k  +  (∂p g_dx)' Δλ_ineq[k]
            +  (∂p e_dx)' Δλ_eq[k]  +  (∂p A_lin)' Δλ_lin[k]
(s block) =  0                       (v1: p-independent slack penalties)
```

where in the QP convention `π_k = −sol.lambda_dyn[k]`,
`Δλ_ineq = sol.lambda_ineq_stage[k]` (hi-side), `Δλ_eq = sol.lambda_eq_stage[k]`,
`Δλ_lin = sol.lambda_lin_stage[k]` (all already the hi−lo differences or
hi-side values per `globalize.hpp`).

Box rows have unit-column `DC` (p-independent) → no grad contribution.
The `(∂p ·)' λ` terms are the differentiating-of-the-Lagrangian constraint
terms; they are O(1) and **must** be included (not negligible), even when the
row is currently inactive (their multiplier may be nonzero).

**`seed.b` (dynamics RHS), stage k:**  `∂p f_k(z*, p*)` = `f_p` from
`dynamics_sens_p`. (Here `b_k = f(x*_k,u*_k,p) − x*_{k+1}`; `x*_{k+1}` is a
fixed current-iterate value, so `∂p b_k = ∂p f_k`.)

**`seed.d` (bound offsets), per side; `d_lo = lo − w`, `d_hi = w − hi`,
`w = row value at z*`:**

| group | lo side seed | hi side seed |
|-------|-------------|-------------|
| pin (fixed x0) | `+ ∂p x0` | `− ∂p x0` |
| box (bx/bu)    | `+ ∂p lo`  | `− ∂p hi`  |
| ineq (hi only) | —          | `+ ∂p g`   |
| eq (both 0)    | `− ∂p e`   | `+ ∂p e`   |
| lin            | `∂p lo − (∂p A)x* − (∂p B)u*` | `− ∂p hi + (∂p A)x* + (∂p B)u*` |
| slack sides    | 0          | 0          |

(`w` for box/lin depends on the p-dependent `A`, `B`, `lo`, `hi`, hence the
extra `(∂p A)x* + (∂p B)u*` shift on lin; box `w` is a variable value,
p-independent at fixed `z*`.)

**`seed.m`:** zero (v1; carried for parity, used only by adjoint scaling).

**Sign verification:** the `(∂p ·)' π` grad terms and the lin `d` shift carry
the QP row sign convention (`res_b = BA z + b − x_{k+1}`); the FD test is the
safety net for the final signs, following the 5f precedent
("sign to be verified numerically").

## 3. Interface change (`include/ocp/problem.hpp`)

### 3.1 Dims + types

- `Dims` gains `static constexpr int np = 0;` (0 = no parameter vector; the
  `*_sens_p` entries are then optional and never called). Every existing
  concrete `Dims` tag is updated with `np = 0` (mechanical, no behavior
  change).
- `Problem` gains `using param_t = Eigen::Matrix<Scalar, np, 1>;` and the
  p-derivative matrix types (fixed-size, `np` columns), e.g.:
  - `dyn_df_dp_t   = Eigen::Matrix<Scalar, nx, np>`            (`∂f/∂p`)
  - `dyn_dA_dp_t   = Eigen::Matrix<Scalar, nx*nx, np>`         (`∂(∂f/∂x)/∂p`, col-major per p)
  - `dyn_dB_dp_t   = Eigen::Matrix<Scalar, nx*nu, np>`         (`∂(∂f/∂u)/∂p`)
  - `stage_grad_dp_t = Eigen::Matrix<Scalar, nx+nu, np>`       (`∂(∇L)/∂p`, `[g_x; g_u]` layout, col per p)
  - `term_grad_dp_t  = Eigen::Matrix<Scalar, nx, np>`
  - `ineq_dg_dx_dp_t = Eigen::Matrix<Scalar, ng*nx, np>`; `ineq_dg_du_dp_t = Eigen::Matrix<Scalar, ng*nu, np>` (and eq / terminal analogues)
  - box/lin bound + `A`/`B` p-derivative vectors.

### 3.2 `*_sens_p` stubs (dimension-gated, mirroring the `ne == 0` pattern)

Each is required iff its group dimension is nonzero **and** `np > 0`; a
zero-dimension group leaves the stub unimplemented (same contract as the
existing `stage_equality_value` when `ne == 0`):

- dynamics (if `nx > 0`): `void dynamics_sens_p(int k, x, u, state_t& f_p, dyn_dA_dp_t& A_p, dyn_dB_dp_t& B_p) const;`
- stage cost (always, if `np > 0`): `void stage_cost_sens_p(int k, x, u, param_t& L_p, stage_grad_dp_t& grad_p) const;`
- terminal cost: `void terminal_cost_sens_p(x, param_t& L_p, term_grad_dp_t& grad_p) const;`
- stage ineq (if `ng > 0`): `void stage_inequality_sens_p(int k, x, u, ineq_t& g_p, ineq_dg_dx_dp_t& gdx_p, ineq_dg_du_dp_t& gdu_p) const;`
- stage eq (if `ne > 0`): `void stage_equality_sens_p(int k, x, u, eq_t& e_p, eq_de_dx_dp_t& edx_p, eq_de_du_dp_t& edu_p) const;`
- terminal ineq / eq (if `ng_t` / `ne_t > 0`): state-only analogues.
- stage linear (if `nl > 0`): `stage_linear_sens_p(int k) const` returning a spec-like struct `{ dA_dp, dB_dp, dlo_dp, dhi_dp }`.
- terminal linear (if `nl_t > 0`): `terminal_linear_sens_p() const`.
- box: `stage_state_box_sens_p(int k)` / `stage_control_box_sens_p(int k)` /
  `terminal_state_box_sens_p()`, each returning `{ dlo_dp, dhi_dp }`.
- initial state (if `fixed_initial_state`): `Eigen::Matrix<Scalar, nx, np> initial_state_sens_p() const;`

Every stub carries a `static_assert(false, ...)` with the exact required
signature, consistent with the rest of the interface.

## 4. Solver change (`include/ocp/solvers/acados/`)

- `AcadosSolver::sens_param(const P& problem, const Solution<P, NH>& sol,
  const param_t& dp, QpSol<P, NH>& dsol)` (public, like `sens_x0`):
  `static_assert(P::np > 0)`; dispatch on `opts_.qp_mode`:
  - **kStaged (5i scope):** assemble the QP, fill `QpSeed` per §2.1 (new
    helper `assemble_param_seed`, reusing the `StageEval`/`TermEval` bundles +
    one `*_sens_p` call per module per stage), then `sens_qp.sens_forward`.
  - kFullCondensed / kPartialCondensed: return `kAborted` with a message in
    5i (deferred to 5j — see §6).
- The seed assembly reads the converged multipliers from `sol`
  (`lambda_dyn`, `lambda_ineq_stage`, `lambda_eq_stage`, `lambda_lin_stage`,
  terminal analogues) for the `(∂p ·)' λ` grad terms, applying the
  `π = −λ_dyn` sign from `globalize.hpp`.

No change to `hpipm.hpp` (the seed mechanism is generic and already in place).

## 5. Example: `DoubleIntegrator` (np = 6)

Parameter vector `p = (Ts, v_max, wq, wv, wa, wf)`; implement all the
`*_sens_p` entries the dimensions require. Nonzero entries:

- `dynamics_sens_p`: `f_p[:,Ts] = [v; u]`, `A_p[:,Ts] = [[0,1],[0,0]]`,
  `B_p[:,Ts] = [0;1]` (cols for the weight params are 0).
- `stage_cost_sens_p`: `grad_p` for `wq/wv/wa` → `grad_p[:,wq] = [x0;0;0]`,
  `[:,wv] = [0;x1;0]`, `[:,wa] = [0;0;u]` (in `[g_x; g_u]` layout); `L_p`
  nonzero for all three.
- `terminal_cost_sens_p`: `grad_p[:,wf] = [2 x0; 2 x1]`.
- `stage_inequality_sens_p`: `g_p[:,v_max] = −1` (`g = v − v_max`),
  `gdx_p = gdu_p = 0`.
- box / linear / initial `sens_p`: all zero.

## 6. Out of scope (5j)

- **Dense / block seed condensing:** the staged seed (per-stage `grad`/`b`/`d`)
  must be condensed into a `DenseQpSeed` / block `QpSeed` (the `b` seed flows
  through the same Riccati/flow-map elimination as the data; the dense QP has
  no `b` field, so the dynamics p-derivative is folded into the dense `seed_g`).
  This is a clean follow-up once the staged recipe is FD-verified.
- **p-dependent slack penalties** (the `s` grad block / `m` seed): zero in v1.
- **Adjoint** parameter sensitivity (`sens_adj`): the mechanism exists
  (Phase 4); not in 5i's scope.

## 7. Open questions (verify at implementation time)

1. Sign of the `(∂p ·)' π` grad terms and the lin `d` shift under the
   `res_b = BA z + b − x_{k+1}` row convention — FD test is the safety net.
2. `param_t` when `np == 0`: the type is `Eigen::Matrix<Scalar,0,1>`;
   confirm `sens_param` is simply unavailable (`static_assert(P::np > 0)`)
   rather than a zero-size no-op.
3. Cost of the extra `*_sens_p` calls (one per module per stage, N+1 stages):
   negligible vs. the QP re-solve; confirm no need to fold into `eval_stage`.

## 8. Execution phases (all done 2026-10-14)

- **A. Scope pass** — done (this doc, scope confirmed 2026-10-14).
- **B. Interface** (`problem.hpp`) — done: `np` (optional in Dims, detection
  default 0 via `detail::dims_np`), `param_t` + p-derivative types, ~13
  dimension-gated `*_sens_p` stubs + value p-derivative aliases, contract
  block updated; `np = 0` added to every existing `Dims` tag (+ the acados2ocp
  generator emits `np = 0`).
- **C. DoubleIntegrator** (`np = 6`) — done: all required `*_sens_p` entries.
- **D. Solver** (`acados.hpp`) — done: `fill_param_seed` (generic-lambda
  `fill_stage` covering first/path stage-type differences) + public
  `sens_param` (staged; kAborted for the condensing modes) + active-set
  gating (§2.3 / §0) + `detail::jacp_t_lam_add` helper.
- **E. Test** — done: `tests/sqp/sens_param_5i.cpp` wired into `sqp_unit`
  (zero-dp, non-staged kAborted, per-parameter central FD).
- **F. Regression** — done: warning-free full build; all 11 test binaries +
  examples pass (the `np == 0` path is bit-identical).
- **G. Docs + commits** — docs done (this doc, parity plan §8/§9.3,
  `problem.hpp` contract); commits pending.

## 9. Verification gate (met 2026-10-14)

- [x] Warning-free build under `-Wall -Wextra -Werror`.
- [x] Full existing test suite passes; no silently weakened tests (the
  v_max "dx_0 == 0" check now passes at 1e-12 because active-set gating
  makes its seed exactly zero; the Ts check uses a documented
  truncation-aware step/tolerance, see §0).
- [x] `tests/sqp/sens_param_5i.cpp` green: for each `p_j`, central FD of
  the full SQP re-solve vs. `sens_param(..., dp = e_j, ...)`; errors listed
  in §0 (all <= 1e-6 except Ts at 1.7e-6 with h = 1e-5, tol 1e-5).
- [x] A/B: all existing example SQP solutions unchanged (`np == 0` path
  bit-identical; the new code is only reached when `np > 0`).
- [x] Docs: `problem.hpp` interface-contract block, this plan, main parity
  plan §9.3 → done, §8 row.

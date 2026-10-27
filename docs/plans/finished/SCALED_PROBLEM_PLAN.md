# ProblemScaling: CRTP wrapper for state/input diagonal scaling

TODO item B: "add a ProblemScaling crtp class for scaling states and inputs.
(how to recover the unscaled values from Solution?)" (docs/plans/TODO.md).

## Motivation

When a problem has widely different state/input scales (e.g. position ~ 100 m
vs. velocity ~ 5 m/s, or an input that ranges from 1e-3 to 1e3), the solver
works better on a *scaled* problem where all variables are O(1). The wrapper
lets the user define the model in natural units and let the wrapper handle
the coordinate transform, keeping the solver on well-conditioned variables.

## Scaling model (settled)

**Diagonal (element-wise) scaling only, at the NLP level.** For scaling
vectors `sx` (nx) and `su` (nu) with all entries > 0:

    x_s = sx ⊙ x          (scaled state)
    u_s = su ⊙ u          (scaled control)

No affine shift (offset) in v1 — keep it minimal; offset can be a later
extension. The solver sees the scaled problem; the user sees the unscaled
model.

Positioning vs. acados: acados has *no* NLP-level variable scaler; its
`ocp_nlp_qpscaling` module is a different, solver-internal mechanism that
scales the *QP subproblem's* objective (Gershgorin eigenvalue estimate) and
constraint rows (Jacobian/bound norms) per iteration, with duals rescaled
after the solve — the NLP variables never change coordinates. Users who want
NLP-level x/u scaling in acados do it by hand (typically in CasADi). This
wrapper provides that missing NLP-level facility; it is independent of (and
complementary to) any future QP-level scaling inside the solver.

### Transformation rules

For a general diagonal `S = diag(s)`, with unscaled variable `z` and scaled
variable `z_s = S z` (so `z = S⁻¹ z_s`):

| Quantity | Scaled form | Transform |
|---|---|---|
| **Value** (cost, constraint) | unchanged | evaluate base at `S⁻¹ z_s` |
| **Gradient** / row-Jacobian | `∇_s = S⁻ᵀ ∇` | left-multiply base result by `S⁻ᵀ` |
| **Column-Jacobian** (dynamics) | `Sx·J·S⁻¹` (w.r.t. state) and `Sx·J·Su⁻¹` (w.r.t. input) — `f_s = Sx·f(...)` carries a left factor for *both* blocks | conjugate / right-multiply |
| **Hessian** (cost, multiplier-contracted) | `H_s = D⁻ᵀ H D⁻¹` where `D = blkdiag(Sx, Su)` | congruence by `D⁻¹` |
| **Box bounds** | `lo_s = S ⊙ lo`, `hi_s = S ⊙ hi` | element-wise multiply |
| **Linear A / B** | `A_s = A Sx⁻¹`, `B_s = B Su⁻¹` | right-multiply by `S⁻¹` |
| **Soft-penalty weights** | box rows: `quad_s = quad/s²`, `lin_s = lin/s` (slack is in scaled units); linear & nonlinear rows: unchanged (constraint values are scale-invariant) | per-row element-wise |

For diagonal S, `S⁻ᵀ = S⁻¹` (element-wise reciprocal), so "left-multiply by
S⁻ᵀ" is just element-wise division of the gradient by `s`.

### Per-interface-function transforms

Let `Sx = diag(sx)`, `Su = diag(su)`, `D = blkdiag(Sx, Su)` (nx+nu).

| Function | Transform |
|---|---|
| `initial_state()` | return `Sx ⊙ base.initial_state()` |
| `dynamics_next_state` | `Sx ⊙ base.dynamics_next_state(k, Sx⁻¹x_s, Su⁻¹u_s)` |
| `dynamics_value_jac` | x_next as above; `df_dx_s = Sx·df_dx·Sx⁻¹`; `df_du_s = Sx·df_du·Su⁻¹` |
| `dynamics_value_jac_hess` | value/jac as above; `hess_s = D⁻ᵀ · hess_base · D⁻¹` |
| `stage_cost_value` | `base.stage_cost_value(k, Sx⁻¹x_s, Su⁻¹u_s)` |
| `stage_cost_value_grad` | value as above; `grad_s = [Sx⁻¹⊙grad_x; Su⁻¹⊙grad_u]` |
| `stage_cost_value_grad_hess` | value/grad as above; `hess_s = D⁻ᵀ H D⁻¹` |
| `terminal_cost_value` | `base.terminal_cost_value(Sx⁻¹x_s)` |
| `terminal_cost_value_grad` | `grad_s = Sx⁻¹ ⊙ base_grad` |
| `terminal_cost_value_grad_hess` | `hess_s = Sx⁻ᵀ H Sx⁻¹` |
| `stage_inequality_value` | `base.stage_inequality_value(k, Sx⁻¹x_s, Su⁻¹u_s)` (value unchanged) |
| `stage_inequality_value_jac` | `g_dx_s = g_dx·Sx⁻¹`; `g_du_s = g_du·Su⁻¹` |
| `stage_inequality_value_jac_hess` | value/jac as above; `hess_s = D⁻ᵀ hess D⁻¹` |
| `stage_equality_*` | identical pattern to inequality |
| `stage_linear_constr` | `A_s = A·Sx⁻¹`; `B_s = B·Su⁻¹`; bounds unchanged |
| `stage_state_box_constr` | `lo_s = sx[state_box_idx] ⊙ lo`; `hi_s = sx[state_box_idx] ⊙ hi`; `penalty.quad /= s²`, `penalty.lin /= s` per row (slack is in scaled units) |
| `stage_control_box_constr` | `lo_s = su[control_box_idx] ⊙ lo`; `hi_s = su[control_box_idx] ⊙ hi`; `penalty.quad /= s²`, `penalty.lin /= s` per row |
| `stage_{ineq,eq}_constr_soft_penalty` | unchanged (pass through) |
| `terminal_state_box_constr` | `lo_s = sx[terminal_state_box_idx] ⊙ lo`; `hi_s = sx[terminal_state_box_idx] ⊙ hi`; `penalty.quad /= s²`, `penalty.lin /= s` per row |
| `terminal_inequality_*` / `terminal_equality_*` | same pattern as stage (state only) |
| `terminal_linear_constr` | `A_s = A·Sx⁻¹`; bounds unchanged |
| `terminal_{ineq,eq}_constr_soft_penalty` | unchanged |
| All `*_sens_p` entries | same transforms as their non-sens counterparts, applied column-wise |

## Recovering unscaled values from Solution

The solver produces `Solution<ScaledProblem>` with all quantities in *scaled*
coordinates. Recovery is a pure element-wise transform (diagonal scaling is
its own inverse up to reciprocals):

| Solution field | Unscaled = ? |
|---|---|
| `x[k]` (N+1) | `x_unscaled = Sx⁻¹ ⊙ x_s` (divide each component by `sx`) |
| `u[k]` (N) | `u_unscaled = Su⁻¹ ⊙ u_s` (divide each component by `su`) |
| `lambda_dyn[k]` (N) | `λ_dyn_unscaled = Sx ⊙ λ_dyn_s` (multiply by `sx`) |
| `lambda_box_state[k]` (N+1, layout `[lo;hi]`, rows in `state_box_idx` order) | multiply each row `i` by `sx[state_box_idx[i]]` |
| `lambda_box_control[k]` (N, layout `[lo;hi]`, rows in `control_box_idx` order) | multiply each row `i` by `su[control_box_idx[i]]` |
| `lambda_ineq_stage`, `lambda_eq_stage`, `lambda_lin_stage` | unchanged (constraint values are scale-invariant) |
| `lambda_ineq_term`, `lambda_eq_term`, `lambda_lin_term` | unchanged |
| `cost_value` | unchanged (scalar objective) |
| `status` | unchanged |

**Rationale for multiplier transforms:**
- *Dynamics multipliers* co-vary with the state (they are the Lagrange
  multiplier of `x_{k+1} = f_k(x,u)`, and the constraint residual transforms
  as `Sx·(x_{k+1} - f)`, so `λ_s = Sx⁻ᵀ λ`).
- *Box multipliers on x/u* co-vary with the bounded variable (the box
  residual `x_i - lo_i` becomes `x_s,i - lo_s,i = sx_i(x_i - lo_i)`, so
  `λ_s,i = λ_i / sx_i` → `λ_i = sx_i · λ_s,i`).
- *(In)Equality / linear multipliers* multiply dimensionless constraint
  values `g(x,u)`, `e(x,u)`, `Ax+Bu`; these values are unchanged by the
  coordinate transform, so the multipliers are unchanged.

### API for recovery

Provide a **member method** on the wrapper:

```cpp
/// Return the solution with all primal/dual quantities converted back
/// to the user's (unscaled) coordinates. The input solution is not
/// modified; a new Solution is returned by value.
Solution<ScaledProblem> unscale() const;
```

Implementation: copy the solution, then apply the element-wise transforms
above. Since scaling is diagonal this is O(N·(nx+nu)) with no heap
allocation (for fixed-horizon mode) or O(N·(nx+nu)) copies (dynamic mode).

Alternatively (lighter-weight, no copy): expose `sx()` and `su()` as
const-accessors and document the recovery formula. The user can then do:

```cpp
for (int k = 0; k <= sol.N; ++k)
    sol.x[k].array() /= scaling.sx().array();
for (int k = 0; k < sol.N; ++k)
    sol.u[k].array() /= scaling.su().array();
```

**Decision:** provide *both* — the accessor for simple x/u recovery (most
common need) and the `unscale()` method for full KKT recovery.

## Class design

```cpp
// include/ocp/scaling.hpp  (new file)

namespace ocp {

/// Diagonal state/input scaling wrapper.
///
/// Wraps a concrete problem `Base` (which satisfies
/// `ocp::Problem<Dims>` for some `Dims`) and presents the same
/// interface to the solver, but in scaled coordinates:
///
///   x_s = sx ⊙ x,   u_s = su ⊙ u
///
/// All model evaluations are forwarded to the wrapped problem after
/// converting from scaled to unscaled coordinates, and the results
/// are converted back.
///
/// The scaling vectors are user-managed members (like problem
/// parameters); they take effect immediately (no "commit" step).
///
/// @tparam Base  the unscaled concrete problem type
template <class Base>
class ScaledProblem
{
    // Extract Dims from Base (requires Base to re-expose it, or we
    // duplicate the Dims tag — see Phase 0 below).
    //
    // Members:
    Base&    base_;       // reference to the wrapped problem
    state_t  sx_;         // state scaling (nx), all > 0
    control_t su_;        // input scaling (nu), all > 0

public:
    // Inherit the Dims from Base (via a new `dims_t` alias added to
    // ocp::Problem in Phase 0).
    using dims_t = typename Base::dims_t;
    // Inherit all type aliases from Base (they are identical since Dims match).
    using scalar_t  = typename Base::scalar_t;
    using state_t   = typename Base::state_t;
    using control_t = typename Base::control_t;
    // ... (all other type aliases, same as Base)

    static constexpr int nx = Base::nx;
    static constexpr int nu = Base::nu;
    // ... (all dimensions, flags, index sets — same as Base)

    // Constructors
    explicit ScaledProblem(Base& base,
                           const state_t& sx = state_t::Ones(),
                           const control_t& su = control_t::Ones());

    // Accessors
    const state_t&   sx() const { return sx_; }
    const control_t& su() const { return su_; }
    state_t&         sx()       { return sx_; }
    control_t&       su()       { return su_; }

    /// Convert a scaled solution back to unscaled coordinates (by value).
    template <int NH = Eigen::Dynamic>
    auto unscale(const Solution<Base, NH>& sol_scaled) const;
    // Actually: Solution<ScaledProblem> since the solver produces that type.

    // --- All interface functions (see transform table above) ---
    state_t initial_state() const;
    state_t dynamics_next_state(int k, const state_t& x, const control_t& u) const;
    void dynamics_value_jac(int k, const state_t& x, const control_t& u,
                            state_t& x_next, dyn_df_dx_t& df_dx,
                            dyn_df_du_t& df_du) const;
    // ... (every interface function, including *_sens_p)
};

}  // namespace ocp
```

### Inheritance strategy

The wrapper does **not** use CRTP. It is a *composition* wrapper:
- It holds a `Base&` (reference to the user's unscaled problem).
- It **itself** satisfies the `ocp::Problem<Dims>` interface (by re-declaring
  all the same type aliases, dimensions, and implementing every function).
- The solver templates over `ScaledProblem<Base>` exactly as they would over
  the raw `Base`.

This is cleaner than CRTP because:
1. The solver's template parameter is the concrete type; no `Derived` indirection.
2. The user can still use the raw `Base` without scaling (no forced wrapper).
3. The `Solution<ScaledProblem>` type naturally carries the scaling context.

However, to avoid duplicating 30+ type aliases and `static constexpr` members,
the wrapper will **inherit from `ocp::Problem<dims_t>`** (the same base class
the user's problem inherits from) and only override the interface functions.
The dimensions/index-sets/type-aliases come for free from the base.

### Phase 0 — add `dims_t` alias to `ocp::Problem`

One-line change in `problem.hpp`:

```cpp
template <class Dims, class Scalar = double>
class Problem
{
public:
    using dims_t = Dims;   // NEW: expose the Dims tag for wrappers
    // ... rest unchanged
};
```

This allows `ScaledProblem<Base>` to write `ocp::Problem<typename Base::dims_t>`
without the user repeating the Dims tag.

## Execution plan

### Phase 0 — `dims_t` alias in problem.hpp

- Add `using dims_t = Dims;` to the public section of `ocp::Problem`.
- No other changes; all existing code compiles unchanged.
- **Verify:** build + run `double_integrator` (exit 0).

### Phase 1 — Core wrapper (`include/ocp/scaling.hpp`)

- New header `include/ocp/scaling.hpp`.
- `ocp::ScaledProblem<Base>` inheriting `ocp::Problem<typename Base::dims_t>`.
- Store `Base& base_`, `state_t sx_`, `control_t su_`.
- Implement all **value-level** interface functions:
  - `initial_state`, `dynamics_next_state`, `stage_cost_value`,
    `terminal_cost_value`, `stage_inequality_value`,
    `stage_equality_value`, `stage_linear_constr`,
    `stage_state_box_constr`, `stage_control_box_constr`,
    `terminal_state_box_constr`, `terminal_inequality_value`,
    `terminal_equality_value`, `terminal_linear_constr`,
    soft-penalty pass-throughs.
- Implement all **gradient/Jacobian-level** functions:
  - `dynamics_value_jac`, `stage_cost_value_grad`,
    `terminal_cost_value_grad`, `stage_inequality_value_jac`,
    `stage_equality_value_jac`, `terminal_inequality_value_jac`,
    `terminal_equality_value_jac`.
- Implement all **Hessian-level** functions:
  - `dynamics_value_jac_hess`, `stage_cost_value_grad_hess`,
    `terminal_cost_value_grad_hess`, `stage_inequality_value_jac_hess`,
    `stage_equality_value_jac_hess`, `terminal_inequality_value_jac_hess`,
    `terminal_equality_value_jac_hess`.
- Implement all **`*_sens_p`** entries (same transforms, column-wise).
- **No `*_sens_p` in this phase** — gate with `if constexpr (Base::np > 0)`.
  (Actually implement them; they follow the same pattern. Include in this
  phase for completeness.)
- Helper: a small internal utility for the diagonal congruence
  `D⁻ᵀ H D⁻¹` on `[x;u]` block matrices (avoid temporary allocation;
  use in-place row/column scaling for fixed-size Eigen).
- **Verify:** build warning-free (`-Wall -Wextra -Werror`).

### Phase 2 — Solution recovery

- Implement `unscale(const Solution<ScaledProblem<Base>, NH>&) const`
  returning `Solution<Base, NH>` with all fields converted.
- Expose `sx()` / `su()` const accessors for the common "just divide x and u"
  use case.
- Document the multiplier transform table (in the header and in the plan).
- **Verify:** unit test (Phase 3).

### Phase 3 — Test

New file: `tests/scaling/scaling_test.cpp` (CMake target `scaling_test`;
implemented as written below, with DoubleIntegrator as the base problem).

Test strategy — use `DoubleIntegrator` as the base problem:

1. **Identity scaling (sx = 1, su = 1):** the wrapper's output must
   bit-match the raw problem's output for every interface function.
2. **Known scaling (sx = [2; 0.5], su = [3]):**
   - Pick random (x, u) points.
   - Call every interface function on the scaled wrapper and the raw problem.
   - Verify:
     - Values (cost, constraint) match.
     - Gradients: `grad_s = S⁻ᵀ grad_base` (element-wise).
     - Dynamics Jacobian: `A_s = Sx A Sx⁻¹`, `B_s = B Su⁻¹`.
     - Hessians: `H_s = D⁻ᵀ H D⁻¹`.
     - Box bounds: `lo_s = S ⊙ lo`.
     - Linear A/B: right-multiplied by S⁻¹.
   - Use a tolerance of 1e-12 (the transforms are exact for fixed-size
     Eigen with double).
3. **Recovery test:**
   - Fill a `Solution<ScaledProblem>` with known scaled values.
   - Call `unscale()`.
   - Verify `x_unscaled = Sx⁻¹ ⊙ x_s`, `u_unscaled = Su⁻¹ ⊙ u_s`.
   - Verify multiplier transforms.
4. **Solver integration test (e2e):**
   - Solve the double-integrator with `AcadosSolver<ScaledProblem<DoubleIntegrator>>`
     using a non-trivial scaling.
   - Solve the same problem without scaling.
   - Compare: the unscaled solution from the scaled run must match the
     direct unscaled solution to solver tolerance (~1e-6 on x/u, cost).
   - This validates the full pipeline end-to-end.

### Phase 4 — Docs

- `docs/ARCHITECTURE.md`: add a paragraph on scaling under §2 Design
  Principles or a new §9, describing the wrapper pattern and the
  unscale-recovery formula.
- `docs/tutorial/`: add a section (or a new chapter) showing usage:
  ```cpp
  DoubleIntegrator raw;
  DoubleIntegrator::state_t sx; sx << 1.0, 0.1;
  DoubleIntegrator::control_t su; su << 0.5;
  ocp::ScaledProblem<DoubleIntegrator> scaled(raw, sx, su);
  // ... solve with `scaled`, then:
  auto sol_unscaled = scaled.unscale(sol);
  ```
- `docs/plans/TODO.md`: mark item B (line 34) as done with pointer to this
  plan.

## Status

- [x] Phase 0 — `dims_t` alias in problem.hpp
- [x] Phase 1 — core wrapper in `include/ocp/scaling.hpp`
- [x] Phase 2 — solution recovery (`unscale` + accessors)
- [x] Phase 3 — unit tests + e2e solver test
- [x] Phase 4 — docs (ARCHITECTURE §9, README, TODO)

## Implementation findings (log)

1. **C++ name lookup in dependent bases:** unqualified lookup does not find
   type aliases / statics inherited from the *dependent* base
   `Problem<typename Base::dims_t, ...>` inside the `ScaledProblem` class
   template, and a `using`-declaration may only name members of a base
   class (the wrapper's base is `Problem<...>`, not `Base`). Solution:
   re-expose all type aliases used in the body as local
   `using X = typename Base::X;` aliases, and the static dimensions /
   index arrays as `static constexpr` copies. Verified with a minimal
   repro before touching the real header.
2. **Dynamics multiplier transform (derived & verified in 1-D):** the base
   `dynamics_value_jac_hess` must receive the *unscaled* multiplier
   `lam = Sx * lam_s` (i.e. the wrapper multiplies, not divides) because
   `lam_s'(x_next_s - f_s) = (Sx' lam_s)' (x_next - f)`. Confirmed with
   f(x)=x^2: base returns 2*s*lam_s, the scaled Hessian must be
   2*lam_s/s.
3. **e2e test warm start:** solving from an already-converged trajectory
   makes the SQP take a ~1e-6 Newton step and return `kMinStep` instead of
   `kSolved` (this happens through the wrapper and directly, i.e. it is
   not a wrapper bug). The e2e test therefore builds the same
   terminal-equality-feasible constant-control warm start for both solves
   (mirroring `tests/sqp_double_integrator/`).
4. **Zero-dim groups:** forwarding is unconditional; solver code only ever
   calls the groups the concrete problem implements, so the
   `static_assert(false)` stubs in `problem.hpp` are never instantiated
   through the wrapper.
5. **Scaling vectors as public members:** the plan suggested const
   `sx()` / `su()` accessors; implemented as public members `sx_` / `su_`
   (trailing underscore, user-managed, like `DoubleIntegrator::Ts_`).
   Reason: the scaling vectors are user data that can change between
   solves, and public read/write access is simpler than accessors plus
   mutators; the trailing-underscore convention marks the distinction
   from the scaled/unscaled coordinate names used in the transform
   comments.

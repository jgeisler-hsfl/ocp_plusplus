# Plan: Replace Finite Differences with Generated CasADi Derivatives

## Problem

All five generated examples use central finite differences (h=1e-6) for
dynamics Jacobians, cost Jacobians/gradients/Hessians, and constraint
Jacobianians. The acados codegen already emits exact analytic derivatives as
separate C functions. The generated adapter should call those instead.

## Verified C-Function Inventory

### Dynamics — IRK / DAE

| File | ABI (verified) | Outputs |
|------|----------------|---------|
| `*_impl_dae_fun.c` | `(x, xdot, u, p, t)` | F (dense, 1 output) |
| `*_impl_dae_fun_jac_x_xdot_u.c` | `(x, xdot, u, p, t)` | F, dF/dx (nx×nx), dF/dxdot (nx×nx), dF/du (nx×nu) — all sparse |

- All outputs use CasADi sparse layout (see Sparsity Format below).
- dF/dxdot is diagonal ±I (the c_i vector). This replaces the runtime
  two-call calibration currently in every generated ODE constructor.
- f(x,u) = -F(x, 0, u) / c; df/dx = -(dF/dx) / c (row-wise); df/du = -(dF/du) / c.

### Dynamics — ERK / explicit ODE

| File | ABI (verified) | Outputs |
|------|----------------|---------|
| `*_expl_ode_fun.c` | `(x, u)` | f (dense, 1 output) |
| `*_expl_vde_forw.c` | `(x, Sx[nx×nx], Su[nx×nu], u)` | f, Sx_out, Su_out (dense) |
| `*_expl_vde_adj.c` | `(x, y[nx], v[nx×nu], u)` | adjoint intermediates |

**Key semantic finding**: vde_forw is NOT a pointwise HVP. It is a
stage-sensitivity propagation operator. Given Sx = d(stage_x)/d(x0) and
Su = d(stage_x)/du, it returns:
  - res[0] = f(x, u)  (the ODE value at the stage)
  - res[1] = Jx · Sx + (stage corrections)  → with Sx=I, Su=0: res[1] ≈ Jx
  - res[2] = Ju · Sx + Jx · Su             → with Sx=I, Su=0: res[2] ≈ Ju

**Plan**: Call `vde_forw(x, Sx=I, Su=0, u)`. The outputs should give
(f, Jx, Ju) in one call. **Must verify by cross-checking against FD before
wiring into the ODE struct.** If the semantics don't match (e.g. the Sx_out
is not simply Jx when Sx=I), fallback: assemble J by calling vde_forw with
unit-vector Sx (nx+nu calls).

No `expl_ode_fun_jac_x_u.c` is generated. If vde_forw proves unsuitable,
the alternative is to add a codegen request for a plain Jacobian.

### Cost — NONLINEAR_LS

| File | ABI (verified) | Outputs |
|------|----------------|---------|
| `*_cost_y_fun.c` | `(x, u, p, t)` | y[ny] |
| `*_cost_y_fun_jac_ut_xt.c` | `(x, u, p, t)` | y[ny], Jᵀ[(nx+nu)×ny] sparse, dummy[0] |
| `*_cost_y_e_fun_jac_ut_xt.c` | `(x, p, t)` | y[ny], Jᵀ[nx×ny] sparse, dummy[0] |

**Critical layout finding**: The Jacobian output o1 is **transposed** and in
**[u; x] block order**:
  - Rows 0..nu-1: ∂y/∂u transposed → Ju (ny × nu)
  - Rows nu..nx+nu-1: ∂y/∂x transposed → Jx (ny × nx)

In C++:
  ```cpp
  Ju = o1.topRows(nu).transpose();      // (ny × nu)
  Jx = o1.bottomRows(nx).transpose();   // (ny × nx)
  ```

GN Hessian = Jxᵀ W Jx, Jxᵀ W Ju, Juᵀ W Jx, Juᵀ W Ju (no d²y needed).

### Cost — EXTERNAL

| File | ABI (verified) | Outputs |
|------|----------------|---------|
| `*_cost_ext_cost_fun.c` | `(x, u, p)` | value (scalar) |
| `*_cost_ext_cost_fun_jac.c` | `(x, u, p)` | value, grad[(nx+nu)] dense |
| `*_cost_ext_cost_fun_jac_hess.c` | `(x, u, p)` | value, grad[(nx+nu)], hess[(nx+nu)×(nx+nu)] sparse, o3[0], o4[0×(nx+nu)] |

- Stage-0 variant: `*_cost_ext_cost_0_fun{,_jac,_jac_hess}.c`
- Terminal variant: `*_cost_ext_cost_e_fun{,_jac,_jac_hess}.c`
- Gradient order: **[gu; gx]** — casadi emits the gradient in [u; x] block
  order, i.e. o1 = [gu0..gu(nu-1), gx0..gx(nx-1)] (verified numerically for
  p5probe_b; the [gx; gu] reading is provably not a gradient field of the
  value function). The framework expects [gx; gu], so the generated adapter
  reorders: gx(i) = o1[nu+i], gu(i) = o1[i].
- Hessian o2 is sparse ((nx+nu)×(nx+nu)), **lower-triangular, in [u; x]
  block order**; o3 and o4 can be nullptr (CasADi guards writes with
  `if(res[i]!=0)`). The generated adapter densifies, symmetrizes and
  reorders to [x; u].
- The Hessian is the **exact** Hessian of the external cost value (not a GN
  approximation); for p5probe_b W and yref are empty, so the external
  function output is the cost itself.

### Constraints — h (inequality/equality)

| File | ABI (verified) | Outputs |
|------|----------------|---------|
| `*_constr_h_fun.c` | `(x, u, t)` | h[nh] |
| `*_constr_h_fun_jac_uxt_zt.c` | `(x, u, t)` | h[nh], Jᵀ[(nu+nx)×nh] sparse, dummy |
| `*_constr_h_e_fun_jac_uxt_zt.c` | `(x, t)` | h[nh], Jᵀ[nx×nh] sparse, dummy |

**Same transposed + [u;x] layout as NONLINEAR_LS cost**:
  - Rows 0..nu-1: ∂h/∂u transposed → Ju (nh × nu)
  - Rows nu..nu+nx-1: ∂h/∂x transposed → Jx (nh × nx)

## Hard Constraints

1. **No sparse matrices.** All algebra stays on dense, fixed-size Eigen
   (AGENTS.md convention). The CasADi sparse layout is only the C-ABI wire
   format: a workspace struct holds a fixed integer pattern (queried once at
   construction) and a flat `double[nnz]` raw buffer; `eval()` densifies
   immediately into fixed-size dense buffers. No `Eigen::Sparse*` type is
   introduced anywhere; downstream code sees dense matrices only.

2. **DAE scope: ODE-in-residual-form only.** The framework's ODE contract is
   an explicit RHS `xdot = f(x, u, t)`. The generated adapter reduces the
   CasADi residual `F(x, xdot, u) = c⊙xdot − f_cas(x, u)` by solving for
   `xdot`: `f = −F(x, 0, u)/c`, `df/dx = −(dF/dx)/c`, `df/du = −(dF/du)/c`
   (row-wise). This is exact only when dF/dxdot is a full-rank diagonal —
   true for all five current examples (algebraic part is
   zero-dimensional). Problem-level algebraic constraints (linear/nonlinear
   h, box) live at the OCP constraint level and are unaffected. The ODE
   constructor takes `c_` from the dFxdot diagonal and asserts
   `|c_i| >= 0.5` before snapping to ±1, so a genuine DAE with algebraic
   rows fails loudly at construction instead of dividing by zero.

## Sparsity Format (Verified from acados/utils/external_function_generic.c)

CasADi sparse output sparsity array `sp[]` (confirmed against
`acados/acados/utils/external_function_generic.c`, `casadi_nnz` /
`casadi_is_dense` / `d_cvt_colmaj_to_casadi`):
- `sp[0]` = nrow
- `sp[1]` = ncol
- `sp[2]` = dense flag (0 = sparse, 1 = dense)
- If dense (sp[2]==1): data is nrow*ncol contiguous doubles (column-major).
- If sparse (sp[2]==0):
  - `idxcol = sp + 2` → (ncol+1) cumulative column-end indices
  - `row = sp + ncol + 3` → nnz row indices (column-major / CSC order)
  - Note: a "fake-sparse" output (sp[2]==0 but the pattern is actually
    fully dense, nnz == nrow*ncol) is still written in this CSC layout, and
    the CSC scatter below reproduces the exact column-major dense order, so
    no separate `casadi_is_dense` branch is required.
- raw buffer size per output = `casadi_nnz(sp)`: dense → nrow*ncol,
  sparse → sum over columns of (idxcol[c+1]-idxcol[c]).

Densification algorithm:
```cpp
void densify(const int* sp, const double* data, double* out, int nrow, int ncol)
{
    // out is column-major (Eigen layout): element (row, col) at out[col*nrow+row]
    if (sp[2] == 1) {  // dense
        std::memcpy(out, data, nrow * ncol * sizeof(double));
    } else {           // sparse (CSC)
        std::fill(out, out + nrow * ncol, 0.0);
        const int* col_idx = sp + 2;   // ncol+1 cumulative column ends
        const int* rows = sp + ncol + 3;
        for (int col = 0; col < ncol; ++col)
            for (int i = col_idx[col]; i < col_idx[col + 1]; ++i)
                out[col * nrow + rows[i]] = data[i];
    }
}
```

## Current FD Usage (All 5 Examples)

| Example | ODE jacobian | Cost deriv | Constraint deriv |
|---------|-------------|------------|-----------------|
| pendulum_on_cart (IRK, LINEAR_LS) | FD of f | none (closed-form) | none |
| unicycle (IRK, LINEAR_LS) | FD of f | none (closed-form) | none |
| masses_chain (IRK, LINEAR_LS) | FD of f | none (closed-form) | none |
| p5probe (ERK, NONLINEAR_LS, h) | FD of f | FD of y (cost_jy) | FD of h (hjac, htjac) |
| p5probe_b (IRK, EXTERNAL) | FD of f | FD grad + FD hess | none |

## Design

### 1. C-API Layer (`*_capi.hpp`)

For each multi-output C function, emit a dedicated workspace struct:

```cpp
struct PendulumGenDynJac
{
    mutable std::vector<int> iw_;
    mutable std::vector<double> w_;
    // Output buffers (dense after densification):
    mutable double F_[4];
    mutable double dFdx_[16];  // nx*nx
    mutable double dFxdot_[16];
    mutable double dFdu_[4];   // nx*nu
    // Sparsity descriptors (queried once at construction):
    int spF_[/* nrow+ncol+3+nnz */];
    int spdFdx_[...];
    int spdFxdot_[...];
    int spdFdu_[...];

    PendulumGenDynJac() { /* query sparsity_out for each output */ }

    void eval(const double* x, const double* xdot, const double* u,
              double p, double t) const
    {
        double* F_raw_, *dFdx_raw_, *dFxdot_raw_, *dFdu_raw_;
        // allocate raw sparse buffers, call C function, densify
    }
};
```

For sparse outputs, the raw buffer is allocated at `nnz` size (from the
sparsity array), then densified into the fixed-size dense buffer.

### 2. ODE Struct

**IRK path** (pendulum, unicycle, masses_chain, p5probe_b):
- `f()`: unchanged (calls `Dyn::eval` for F, computes -F/c).
- `jacobian()`: call `DynJac::eval()` once; extract dFdx, dFdu; scale rows
  by -1/c_i.
- `value_jac()`: call `DynJac::eval()` once; extract F → f, dFdx → df_dx,
  dFdu → df_du (all scaled). This is the fused entry the integrator can
  use to avoid a separate f() call.
- `c_`: derive from dFxdot diagonal at construction (one `DynJac::eval()`
  call at xref/uref), snap to ±1. Replaces the two-value-function
  calibration. **Assert `|c_i| >= 0.5` before snapping** (fails loudly for a
  genuine DAE with algebraic rows, see Hard Constraints §2).
- `hess_prod()`: not available from CasADi DAE codegen. All current examples
  use Gauss-Newton (no dynamics Hessian), so this is never called. Omit.

**ERK path** (p5probe):
- `f()`: unchanged (calls `expl_ode_fun`).
- `jacobian()`: call `vde_forw(x, Sx=I, Su=0, u)` → (f, Jx, Ju).
  **Gate**: verify against 5-point FD on 3+ random points before wiring.
  If verification fails, fallback: assemble J via nx+nu unit-vector calls.
- No `c_` (explicit ODE, no DAE residual).

### 3. Cost Methods

**LINEAR_LS** (pendulum, unicycle, masses_chain, p5probe c0/ct):
- No change. Closed-form Vx/W/Vu already exact.

**NONLINEAR_LS** (p5probe cp/ct):
- Replace `cp_jy()` / `ct_jy()` (FD) with C call to `CostJac::eval()` /
  `CostEJac::eval()`.
- Unpack transposed + [u;x] layout:
  ```cpp
  Ju = raw.topRows(nu).transpose();  // (ny × nu)
  Jx = raw.bottomRows(nx).transpose(); // (ny × nx)
  ```
- `cp_grad()`, `cp_hess()`: same formulas as now but Jx/Ju from C call.

**EXTERNAL** (p5probe_b stage/c0/terminal):
- `value_grad`: call `CostJac::eval()` → (value, grad[(nx+nu)]).
  Split grad into gx/head and gu/tail.
- `value_grad_hess`: call `CostJacHess::eval()` → (value, grad, hess_sparse).
  Densify hess into (nx+nu)×(nx+nu) matrix. o3/o4 passed as nullptr.

### 4. Constraint Methods (h)

**p5probe**:
- `hval()` + `hjac()`: single C call to `HJac::eval()` → (h[nh], Jᵀ[(nu+nx)×nh]).
  Unpack: Ju = topRows(nu).transpose(), Jx = bottomRows(nx).transpose().
- `htval()` + `htjac()`: same pattern with `HEJac`.
- The `stage_inequality_value_jac` / `terminal_inequality_value_jac` methods
  call these and assemble g_dx, g_du.

### 5. Generator Changes (`tools/acados2ocp_pp.py`)

**`parse_c_sig` / `add_cfun`**:
- Extend token parser to handle `RxC` (e.g. `2x2`) and `RxC,nz` (sparse)
  tokens in output specs.
- Remove the `len(outs) != 1` assertion; allow multi-output registration.
- Register derivative files with a `role` field:
  `"dyn_jac"`, `"cost_jac"`, `"cost_jac_hess"`, `"h_jac"`, `"vde_forw"`.
- Store per-output sparsity info (nrow, ncol, dense flag) in the spec dict.

**`emit_capi`**:
- For each registered C function, emit a workspace struct.
- For sparse outputs: query `sym_sparsity_out` at construction, store the
  pattern, allocate raw buffer at nnz size.
- `eval()` method: set up arg/res pointer arrays, call C function,
  densify sparse outputs into fixed-size dense arrays.

**`emit_header` (ODE section)**:
- IRK: emit `value_jac()` calling DynJac; emit `jacobian()` as wrapper.
  Derive c_ from DynJac's dFxdot at construction.
- ERK: emit `jacobian()` calling vde_forw with Sx=I, Su=0.

**`emit_header` (cost section)**:
- NONLINEAR_LS: emit `cp_jy()` / `ct_jy()` calling CostJac/CostEJac;
  unpack transposed [u;x] layout.
- EXTERNAL: emit `ext_grad()` / `ext_hess()` calling CostJac/CostJacHess.

**`emit_header` (constraint section)**:
- Emit `hjac()` / `htjac()` calling HJac/HEJac; unpack transposed [u;x].

**`emit_cmake`**:
- Include all derivative `.c` files in the source list.

### 6. CMake (root `CMakeLists.txt`)

New per-example derivative source variables:
```cmake
set(PC_DERIV_C  examples/pendulum_on_cart/acados_codegen/pendulum_model/pendulum_impl_dae_fun_jac_x_xdot_u.c)
set(UN_DERIV_C  examples/unicycle/acados_codegen/unicycle_model/unicycle_impl_dae_fun_jac_x_xdot_u.c)
set(MC_DERIV_C  examples/masses_chain/acados_codegen/masses_chain_model/masses_chain_impl_dae_fun_jac_x_xdot_u.c)
set(P5_DERIV_C  examples/p5probe/acados_codegen/p5probe_model/p5probe_expl_vde_forw.c
                examples/p5probe/acados_codegen/p5probe_cost/p5probe_cost_y_fun_jac_ut_xt.c
                examples/p5probe/acados_codegen/p5probe_cost/p5probe_cost_y_e_fun_jac_ut_xt.c
                examples/p5probe/acados_codegen/p5probe_constraints/p5probe_constr_h_fun_jac_uxt_zt.c
                examples/p5probe/acados_codegen/p5probe_constraints/p5probe_constr_h_e_fun_jac_uxt_zt.c)
set(P5B_DERIV_C examples/p5probe_b/acados_codegen/p5probe_b_model/p5probe_b_impl_dae_fun_jac_x_xdot_u.c
                examples/p5probe_b/acados_codegen/p5probe_b_cost/p5probe_b_cost_ext_cost_fun_jac.c
                examples/p5probe_b/acados_codegen/p5probe_b_cost/p5probe_b_cost_ext_cost_fun_jac_hess.c
                examples/p5probe_b/acados_codegen/p5probe_b_cost/p5probe_b_cost_ext_cost_0_fun_jac.c
                examples/p5probe_b/acados_codegen/p5probe_b_cost/p5probe_b_cost_ext_cost_0_fun_jac_hess.c
                examples/p5probe_b/acados_codegen/p5probe_b_cost/p5probe_b_cost_ext_cost_e_fun_jac.c
                examples/p5probe_b/acados_codegen/p5probe_b_cost/p5probe_b_cost_ext_cost_e_fun_jac_hess.c)
```

Add `${X_DERIV_C}` to targets that use generated headers:
- `pendulum_on_cart`, `sqp_pendulum`
- `unicycle`, `sqp_unicycle`
- `masseschaingen_gen`, `sqp_masses_chain_gen_equiv`
- `sqp_p5probe`
- `sqp_p5probe_b`

Do NOT add to hand-written port targets (`masses_chain`) — they use their
own capi.

## Phased Execution

Each phase is independently committable and testable.

### Phase 1: Parser + C-API Foundation + IRK Dynamics Jacobian
- Extend `parse_c_sig` for multi-output and RxC tokens.
- Extend `add_cfun` (remove single-output assertion).
- Emit DynJac workspace structs in `*_capi.hpp`.
- Replace FD `jacobian()` with C call in all 4 IRK examples
  (pendulum, unicycle, masses_chain, p5probe_b).
- Derive `c_` from dFxdot (replaces two-call calibration).
- Add `value_jac()` fused entry.
- Update CMakeLists.txt (add DAE jac .c files).
- Regenerate, build, run all tests.

### Phase 2: EXTERNAL Cost (p5probe_b) — DONE
- General multi-output emitter `_emit_multi_out_struct` (replaces the
  dyn_jac-only struct); role→member table `_MULTI_OUT`; `load_spec`
  registers `cost{,0,}_e?_jac{,_hess}` cfuns for EXTERNAL cost scopes.
- C++ adapter: reorder grad `[u;x]→[x;u]`; Hessian densify + symmetrize
  (lower-triangular) + reorder `[u;x]→[x;u]`.
- CMake: 6 cost-deriv `.c` files added to `P5B_DERIV_C`.
- Test: `sqp_p5probe_b` gained an analytic-vs-FD cross-check (5-point FD of
  the C value function for grad, central-difference of the analytic grad for
  hess) at a nonzero point.
- Verified: FD cross-check err ~1e-12 (grad) / ~1e-13 (hess);
  `sqp_p5probe_b` cost=0.0 iters=2 status=1 (same as FD baseline);
  all 6 sqp_* + 8 example/unit binaries pass.

### Phase 3: NONLINEAR_LS Cost (p5probe) — DONE
- Emit CostYJac / Cost0YJac / CostEYJac workspace structs (o1 only;
  o0=y and o2=dummy skipped via nullptr).
- Replace FD `cp_jy` / `ct_jy` with C calls to `cost_y{,_e}_fun_jac_ut_xt`.
- Unpack transposed [u;x] layout: stage `J = [Jx | Ju]` with
  `Jx = o1.bottomRows(nx).transpose()`, `Ju = o1.topRows(nu).transpose()`;
  terminal `J = o1.transpose()`.
- Add `cost_y{,_e}_fun_jac_ut_xt.c` to CMakeLists (`P5_DERIV_C`).
- Regenerate, build, run sqp_p5probe (cost=0.771744, 36 iters, status=1)
  and full regression (all 19 binaries green).

### Phase 4: h-Constraints (p5probe) — DONE
- Emit HJac / HEJac workspace structs.
- Replace FD `hjac` / `htjac` with C calls.
- Unpack transposed [u;x] layout.
- Add .c files to CMakeLists.
- Regenerate, build, run sqp_p5probe.
- A3 FD cross-check added to sqp_p5probe test.
- Note: Jacobian is sized by `nh_raw` (= raw h rows from JSON, `len(h["lh"])`)
  rather than the split-row count `nh`, since the C function returns the raw
  Jacobian.  The split-row mapping (lo/hi sign flips) is applied by the
  `*_value_jac` methods on top.

### Phase 5: ERK ODE Jacobian via vde_forw (p5probe) — DONE
- **Gate**: (A4) section in `sqp_p5probe` compares the analytic
  `jacobian()` (vde_forw called with Sx=I, Su=0) against 5-point FD of
  `expl_ode_fun` at 3 points (incl. the origin). Tolerance 1e-8. Passes.
- Emitted `jacobian()` calls the `OdeJac` workspace struct
  (custom-emitted; non-standard signature `(x, Sx, Su, u, [p]) -> (f, Sx',
  Su')`, bypasses `plan_args`). `Sx' = (df/dx) Sx`, `Su' = (df/dx) Su +
  df/du`; with Sx=I, Su=0 the outputs are the raw Jacobians.
- Fallback: if no `*_vde_forw.c` in the model files, the generator keeps
  the FD `jacobian()` (no example currently needs this).
- `p5probe_expl_vde_forw.c` added to `P5_DERIV_C` in CMakeLists.
- Regenerated, built, sqp_p5probe passes (cost=0.771744, 36 iters).

### Phase 6: Final Regression + Cleanup — DONE
- All 5 example binaries + all sqp_* test targets: 19/19 pass.
- `compare.sh` for pendulum, unicycle, masses_chain: all PASS.
- Full `cmake --build build` clean under `-Wall -Wextra -Werror`.
- Regenerated all 5 JSONs; `git diff` shows only derivative-related
  changes (plus the pre-existing AGENTS.md doc update, unrelated).
- Item 2 in `docs/CLEAN_UP_TODO.md` marked complete.

## Verification Strategy

1. **Analytic vs FD cross-check** (per phase): For each newly wired C-call
   derivative, compare against 5-point central FD (h=1e-5) on 3+ random
   points. Tolerance: 1e-8 relative. (Analytic is exact; FD error is
   O(h²) + O(ε/h) ≈ 1e-10.)
2. **SQP regression**: Run all `sqp_*` test targets. Solutions should be
   numerically identical or slightly more accurate vs pre-change.
3. **compare.sh**: Run for pendulum, unicycle, masses_chain. KKT residuals
   and solution trajectory must match acados within existing tolerances.
4. **Build**: Warning-free under `-Wall -Wextra -Werror`.
5. **Generator round-trip**: Re-run on all 5 JSONs; diff shows only
   expected derivative-related changes.

## Follow-up (separate task): Implicit ODE Formulation for ImplicitRkIntegrator

Target (user-requested, deferred to its own plan/task): the implicit
integrator works on the DAE residual `F(x, xdot, u, t)` directly instead of
the reduced explicit RHS `xdot = f(x, u)`. The ODE duck-type contract splits:

| | explicit RK (today, unchanged) | implicit RK (after rewrite) |
|---|---|---|
| value | `f(x, u, t)` → xdot | `residual(x, xdot, u, t)` → F |
| Jacobian | `jacobian(x, u, t, df_dx, df_du)` | `residual_jacobian(x, xdot, u, t, dF_dx, dF_dxdot, dF_du)` |
| 2nd order | `hess_prod` of f (optional) | HVP of F (optional; F is affine in xdot, so mixed xdot terms vanish) |

### Formulation

Stage equations (same Butcher tableau as today):
  x_s = x + h·Σ_j A[s][j]·K_j
  R_s = F(x_s, K_s, u, t_s) = 0
Newton over K:
  G[s][j] = h·A[s][j]·dF_dx(s) + δ_sj·dF_dxdot(s)
  dK = G⁻¹·(−R)

No reduction by `c`, no division: for a residual `F = c⊙xdot − f_cas` the
implicit system is exactly the row-scaled version of the current explicit
system (`R_implicit = diag(c)·R_explicit`,
`G_implicit = diag(c)·G_explicit`), so Newton iterates are identical —
a clean regression gate (all five examples must stay bit-equal up to
rounding). The payoff: non-diagonal `dF/dxdot` (mass matrices) and index-1
DAEs with algebraic rows (rows with `dF/dxdot` singular but the coupled
block `G` regular) become expressible, matching acados's
`sim_irk_integrator.c`, which works in DAE form.

### Scope of the rewrite (`include/ocp/integrators/`)

1. `rk_implicit.hpp`:
   - `eval_residual`: `K_s − f(x_s, u)` → `F(x_s, K_s, u, t_s)`.
   - `eval_block_jacobian`: `I − h·A·f_x` → `h·A·dF_dx + δ·dF_dxdot`.
   - Workspace: per-stage `dF_dx`/`dF_dxdot`/`dF_du` replace `dfdx`/`dfdu`.
   - First-order sensitivity (dK/d(x,u) = G⁻¹Q) and the multi-step cache:
     same structure, new Jacobian members; cache keying (x, u, t_k) unchanged.
   - Second-order 4f `hess_prod`: re-derive from the DAE residual, following
     acados's 4f in DAE form (`../acados/docs/algorithm/`). Gate: the new
     HVP must reproduce the current f-space 4f on all ODE examples.
   - `supports_value_jac_hess_prod = false` stays.
2. `ode_model.hpp`: traits split — explicit ODE (`f` + `jacobian` [+
   `hess_prod`]) vs implicit ODE (`residual` + `residual_jacobian` [+
   residual HVP]). `static_assert` that the IRK integrator's ODE provides the
   implicit set (mirror of the existing missing-method failure pattern).
3. `rk_explicit.hpp`: unchanged (explicit RHS contract stays).
4. `continuous_problem.hpp` / solver: unchanged — they consume only the
   integrator's discrete-map interface.
5. Generated IRK ODE structs (`*Gen.hpp`): shrink to pass-throughs —
   `residual()` = `Dyn::eval` (F), `residual_jacobian()` = `DynJac::eval`
   (F, dFdx, dFxdot, dFdu); the `c_` member, calibration, and all scaling
   disappear from the generated code.
6. Tests: update `tests/integrators/rk_implicit_4d.cpp`,
   `rk_multistep_4g.cpp`, `time_varying_4h.cpp`, `sqp_continuous_4e.cpp`
   (and the hand-written `masses_chain` port) to the implicit contract; add
   a new test with a non-unit diagonal residual (scaled ODE,
   `F = c⊙xdot − f` with c ≠ ±1) and, ideally, a mass-matrix residual
   (non-diagonal `dF_dxdot`) to prove the general path.

### Why it is a separate task

- It is a self-contained rewrite of `rk_implicit.hpp` with its own
  regression risk (Newton, 1st- and 2nd-order sensitivity, multi-step
  cache); mixing it with the FD-removal work would make test failures hard
  to attribute.
- This plan's C-API layer is built to be neutral to it: the `DynJac`
  workspace struct exposes `F, dFdx, dFxdot, dFdu` as plain dense outputs.
  The `c`-scaling lives in the *ODE struct*, not the capi layer — after the
  rewrite the ODE struct shrinks to a pass-through and the capi layer is
  untouched (zero rework).

Suggested order: finish this plan (Phases 1–6), then start the implicit-ODE
task, which can then drop the `c_` machinery the generator emitted in Phase 1.

## Out of Scope / Follow-ups

- **EXACT Hessian for DAE**: No CasADi Hessian C function is generated for
  DAE models. All current examples use Gauss-Newton (cost Hessian only;
  dynamics enter via linearized BA). A future exact-Hessian path would
  need reverse-mode AD in C++ or a new CasADi codegen pass.
- **ERK hess_prod**: If vde_forw/adj can be used for the composed Hessian,
  this would enable exact dynamics Hessian for ERK. Out of scope for this
  task (all examples use GN).
- **Multi-step IRK (nstep > 1)**: The DAE jac call is per-stage; the
  integrator already loops over stages. No additional work needed.
- **Hand-written masses_chain port**: `examples/masses_chain/masses_chain.hpp`
  also uses FD. This is a hand-written port (not generated) — separate task.

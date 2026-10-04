# acados Example Port + JSON Generator Plan

> This document is the planning record for the acados→ocp++ porting workflow
> (archived as of P5; kept for reference and the generator scope matrix).
> The generator is JSON-driven: it consumes `acados_ocp_nlp.json`, which any
> acados codegen interface (MATLAB or Python) emits identically.

## 1. Goal

A repeatable pipeline that turns an acados OCP example into an ocp++ problem
and proves the ocp++ SQP reproduces the acados solution:

```
 acados OCP (any interface)  ──►  generated C (model/cost/constr + derivatives)
                                     + standalone acados main  +  acados_ocp_nlp.json
                                 │
                                 ├──► compiled acados main  ──►  xref / uref / cost_ref  (REFERENCE)
                                 │
                                 └──► ocp++ wrapper (.hpp) + demo main ──► x / u / cost
                                                                            │
                                            compare(x,u,cost)  pass if within tol
```

Two deliverables:
1. **ocp++ wrapper** for a chosen example: a `Problem`/`Ode` that **links the
   generated CasADi C** (values + derivatives), a demo `main`, and a
   comparison against the compiled acados main.
2. **The generator**: a Python tool `tools/acados2ocp_pp.py` that, given the
   codegen artifact `acados_ocp_nlp.json`, emits the ocp++ wrapper `.hpp` +
   `main` automatically — reusing the generated C for any model / cost /
   constraint and emitting the cost/box code itself when acados uses a native
   (no-code-file) cost/constraint form.

## 2. Environment (current state, 2026-10-03)

| Component | Status |
|---|---|
| acados C lib | **built** (`/home/jgeisler/repos/acados/lib`: libacados, hpipm, blasfeo) |
| acados Python (`acados_toolbox` + casadi 3.8.1) | **available** — `example_ocp.py` codegen verified (pendulum_on_cart, unicycle, p5probe, p5probe_b) |
| ocp++ | builds clean (`cmake -B build && cmake --build build`) |

Any acados codegen interface (MATLAB or Python) emits the same
`acados_ocp_nlp.json`; the generator consumes that JSON and needs no live
problem object.

## 3. What the codegen run produces

`codegen_ocp_<name>_<hash>/` contains:
- **Standalone reference main** `main_ocp_<name>_<hash>.c` — builds and runs
  on its own (`make example` / the emitted `CMakeLists.txt`), links
  `-lacados -lhpipm -lblasfeo`, prints full `xtraj` (NX×(N+1)), `utraj`
  (NU×N), status, `sqp_iter`, KKT residual. **This is the reference.**
- **Model C** `masses_chain_model/*.c` + `masses_chain_model.h`:
  - `masses_chain_impl_dae_fun` — implicit residual `F(x, xdot, u, z)`
  - `masses_chain_impl_dae_fun_jac_x_xdot_u` — `∂F/∂(x, xdot, u)`
  - `masses_chain_impl_dae_jac_x_xdot_u_z` — `∂F/∂(x, xdot, u, z)`
  - `masses_chain_impl_dae_fun_jac_x_xdot_z` — `∂F/∂(x, xdot, z)`
  - all in the acados external-function form
    `int f(const real_t**arg, real_t**res, int*iw, real_t*w, void*mem)`
    + `_work` / `_sparsity_in` / `_sparsity_out` / `_n_in` / `_n_out`.
- **Cost C** `masses_chain_cost/masses_chain_cost.h` — **empty** (cost is
  native LINEAR_LS, no external cost function).
- **Constraints C** `masses_chain_constraints/masses_chain_constraints.h`
  — **empty** (constraints are native box, no external functions).
- **`acados_ocp_nlp.json`** — complete structured OCP dump (dims, cost,
  bounds, integrator + collocation, QP solver, solver options). This is the
  structured input the generator consumes.
- `acados_solver_*.c`, `acados_sim_solver_*.c`, mex + sfunction files
  (not needed for the C++ comparison).

**Consequence:** for this example the only generated model code worth reusing
is the implicit DAE residual + Jacobians; the cost and constraint code must be
**emitted by the generator** (native LINEAR_LS cost + native box). This is
exactly the "no cost-function code file" case the generator must handle.

## 4. Target OCP: masses_chain (from `acados_ocp_nlp.json`)

| Quantity | Value |
|---|---|
| `nx`, `nu` | 24 (4 masses × (p∈R³,v∈R³)), 3 (force on last mass) |
| `N`, `T`, `dt` | 40, 8.0 s, 0.2 s |
| Cost | **LINEAR_LS**: `½‖[x;u]−yref‖²_W`, terminal `½‖x−x_ref‖²_{W_e}`; `W=10·diag` (u-rows = 1e-2), `W_e=10·I₂₄`, `yref=[x_ref;0]` |
| State box | `−0.01 ≤ pᵢ.ᵧ ≤ 1e4`, i=1..4 → indices `{1,7,13,19}` (nbx=4) |
| Control box | `−1 ≤ u ≤ 1` (nbu=3) |
| Initial state | **fixed**, `nbx_0=24` (lo=hi=x0) → `fixed_initial_state=true` |
| `ng=ne=nl=ng_t=ne_t=nl_t` | 0 |
| Dynamics | **IRK, 4 stages × 2 sub-steps, Gauss-Legendre collocation**, 3 Newton iters, tol 0 |
| Solver | SQP, PARTIAL_CONDENSING_HPIPM, `cond_N=5`, `hessian_approx=GAUSS_NEWTON`, `globalization=FIXED_STEP` (step 1), `regularize=NO_REGULARIZE`, tol 1e-6, max_iter 100 |

State layout per mass i: `(px,py,pz, vx,vy,vz)`; mass i occupies
`[6(i-1) .. 6(i-1)+5]`. The constrained DOF is `py` at index `6(i-1)+1`.

Physics (from `masses_chain_model.m`): free masses on a nonlinear spring chain,
`F_i = D·(1 − L/‖d_i‖)·d_i`, `d_1 = p_1`, `d_i = p_i − p_{i-1}`;
`ṗ_i = v_i`; `v̇_i = (1/m)(F_{i+1}−F_i) − [0,0,g]` for i<last, `v̇_last = u`.
Constants `g=9.81, L=0.033, D=1.0, m=0.03`. `x_ref`, `x0` are Newton-solved
rest positions (constants in the .m).

## 5. ocp++ wrapper architecture

Location: **`examples/masses_chain/`** (the codegen folder, the
ocp++ wrapper, the demo main, and the reference artifacts all live here —
no symlinks / no vendoring outside the tree).

```
examples/masses_chain/
  acados_codegen/                # generated (committed): *.c, *.h, acados_ocp_nlp.json
  capi.hpp                       # extern "C" decls + caller shim for the generated C
  masses_chain_ode.hpp           # Ode over the implicit DAE residual (f + jacobian)
  masses_chain.hpp               # Dims + ContinuousProblem + LINEAR_LS cost + box + x0
  main.cpp                       # solve, print, write ocp_pp.csv, compare vs ref
  ref_acados.csv                 # x/u captured from the compiled acados main
```

### 5.1 Dynamics — implicit DAE → explicit Ode bridge

ocp++'s `Ode` contract is **explicit** (`f(x,u,t)→xdot` + `jacobian`). The
generated C is the **implicit residual** `F(x, xdot, u, z) = f_expl(x,u,z) −
xdot`. Because `∂F/∂xdot = −I` (invertible), the two formulations are
equivalent and acados's IRK on the DAE is identical to IRK on `xdot=f_expl`:

```
Ode::f(x, u, t)          =  masses_chain_impl_dae_fun([x; 0; u; z])   # = f_expl(x,u)
Ode::jacobian(x,u,t,...) =  x/u blocks of
                           masses_chain_impl_dae_fun_jac_x_xdot_u([x; 0; u; z])
                           # ∂F/∂x = ∂f/∂x,  ∂F/∂u = ∂f/∂u  (∂F/∂xdot = −I ignored)
```

`z` (np=1, value 0) is unused by the residual (only the unused RK4 `phi` map
used `DT`); confirmed via `sparsity_in`. The caller shim fills `arg` in the
acados order `[x; xdot; u; z]`, packs sparsity, calls the function, and
copies the dense blocks into Eigen matrices.

**Sign convention (found 2026-10-04, systemic bug fix).** The residual
convention is *not* uniform across acados codegen paths:

| codegen path | residual | `∂F/∂xdot` | `F(x,0,u)` |
|---|---|---|---|
| MATLAB codegen | `f − xdot` | `−I` | `+f` |
| Python / casadi (`acados_toolbox`) | `xdot − f` | `+I` | `−f` |

The generated Ode therefore must not hard-code the sign. The emitted Ode
calibrates `c = dF/dxdot` once in its constructor (a two-point probe, snapped
to ±1) and computes `f(x,u) = −F(x,0,u)/c` and `J = Jx,Ju / c`. masses_chain
yields `c=−1` (bit-identical `f` to the pre-fix wrapper); pendulum/unicycle
yield `c=+1` (pre-fix they had negated dynamics, `max|dx| ≈ 19`).

### 5.2 Cost / box / x0 — emitted, not reused

- **LINEAR_LS cost** (native, no C file): `stage_cost_value = ½‖V y − yref‖²_W`
  with `y=[x;u]`; gradient `= Jyᵀ W (Jy y − yref)`; Hessian (exact) `= Jyᵀ W Jy`
  (constant). `Jy` is the known `[I_nx 0; 0 I_nu]`-ish selection from the
  `Vx`/`Vu`/`W`/`yref` in the JSON. Terminal analogous with `W_e`, `yref_e`.
- **Box** (`BoxSpec`): state rows `{1,7,13,19}`, lo=−0.01, hi=1e4; control
  rows all, lo=hi=±1.
- **Fixed x0**: `Dims::fixed_initial_state=true`, `initial_state()` = the
  Newton-solved `x0` (constant from the .m).

### 5.3 Integrator + solver — must match acados exactly

- `ImplicitRkIntegrator<Dims, MassesChainOde, 4, <GL4Tag>, 2>` via
  `ContinuousProblem` → a discrete `Problem` with `dt = 0.2`.
- `SqpSolver` with `PartialCondensingHpipm`, `cond_N=5`,
  `hessian_approx = GAUSS_NEWTON` → set `Dims::has_dynamics_hess_prod = false`
  (drop the dynamics HVP; keep the exact LINEAR_LS cost Hessian), matching
  acados. Globalization set to fixed-step (full step) to mirror the JSON.
- 3 Newton iters per sub-step, tol 0 (ocp++ defaults) = acados defaults.

### 5.4 ocp++ prerequisite: add a Gauss-Legendre 4-stage tableau

acados's generated IRK uses `collocation_type = GAUSS_LEGENDRE` (4 stages);
ocp++'s `butcher.hpp` currently only ships Radau IIA (2/3/4). The
`ImplicitRkIntegrator` is **fully generic over `Tag`** (plain Newton on the
collocation system, no Radau-specific assumption), so adding
`GaussLegendre4Tag` + a `ButcherTableau<Dims,4,GaussLegendre4Tag>`
specialization (nodes/weights/A from acados `sim_collocation_utils.c`) is a
small, isolated change. **Required for an exact integrator match** (§9 Q1).

## 6. Reference capture (no codegen needed at solve time)

The reference is the **compiled acados main**, per your preference:
```
cd examples/masses_chain/codegen_masses_chain
make example                       # links -lacados -lhpipm -lblasfeo
./main_ocp_masses_chain_<hash>     # prints xtraj (24×41), utraj (3×40), status
```
Capture `xtraj`/`utraj`/cost into `ref_acados.csv` (add a small CSV-dump block
to the main, or parse the `d_print_exp_tran_mat` block — §9 Q3).

## 7. Comparison harness

- Both sides solve the **identical** OCP (same N, dt, weights, bounds, x0,
  integrator, solver options) from the same init (x=x_ref all stages, u=0).
- ocp++ `main` writes `ocp_pp.csv` (x, u, cost, status) in the same layout as
  `ref_acados.csv`.
- A small check (in `main` or a test) asserts:
  - cost relative diff < 1e-4
  - `‖Δx‖∞ < 1e-3` and `‖Δu‖∞ < 1e-3` over the horizon.

**Tolerance reality check (learned 2026-10-04).** The strict §7 tolerances
hold only when both SQPs converge to the *same* KKT point:

| example | cost rel. diff | `‖Δx‖∞` | `‖Δu‖∞` | note |
|---|---|---|---|---|
| masses_chain | ~4e-7 | ~1e-8 | ~1e-9 | smooth, convex-ish → identical KKT point |
| unicycle | ~3e-6 | 4.5e-2 | 1.1e-1 | smooth but non-convex; distinct KKT points |
| pendulum_on_cart | ~1.7e-4 | 9.7e-2 | 0.78 | bang-bang controls; distinct KKT points |

The ocp++ SQP uses the *cost-only* Hessian (`has_dynamics_hess_prod=false`),
while acados uses Gauss-Newton (cost + J′J). For non-convex OCPs the two
iterations can land on **different local KKT points** (ocp++ even finds a
*lower* cost for the pendulum: 44834.14 vs acados 44841.68). The per-example
CI tests therefore assert, in addition to loose cost/trajectory tolerances:
1. **integrator-map exactness** (ocp++ IRK on the acados reference
   trajectory must reproduce the acados stage values to ~1e-9 — this is the
   true "same problem" guard), and
2. **ocp++ KKT residual small** (the ocp++ solution must actually solve the
   OCP it is solving).

## 8. The generator (`tools/acados2ocp_pp.py`, Phase 2)

**Pivot (2026-10-03):** the generator is a **Python script** reading the
`acados_ocp_nlp.json` artifact. Rationale:
- the JSON is a complete, self-sufficient OCP spec (dims, full cost matrices
  `Vx/Vu/W/yref(_e)`, all box idx/bounds + the `lbx_0==ubx_0` fixed-x0 marker,
  integrator + collocation + Newton config, and the generated C residual file
  names). Verified on `masses_chain`.
- `acados_ocp_nlp.json` is emitted identically by **both** the MATLAB and the
  Python (`acados_toolbox`) codegen interfaces, so consuming it decouples the
  generator from the codegen front-end and works for code produced by either.
- JSON parsing + templated C++ emission is idiomatic in Python (stdlib `json`
  + f-strings); no MATLAB licence, trivial to test (run → diff).

`tools/acados2ocp_pp.py --json <...json> --name <Cls> --out <dir> [--csrc <c>]`:
- validates the **Phase-1 scope** (LINEAR_LS cost; box-only constraints;
  fixed x0 via `lbx_0==ubx_0`; IRK + GAUSS_LEGENDRE 2..4 stages; casadi DAE,
  `nz==0`); anything else → `Unsupported` (extend later);
- emits `<Name>.hpp` (Dims + Ode [CasADi residual + central-FD Jacobian] +
  integrator alias + LINEAR_LS cost/box/initial-state, all from the JSON),
  `<Name>_capi.hpp` (extern-C wrapper over the generated DAE residual),
  `main.cpp` (solve + CSV driver), and a CMake target snippet.
- The generated C residual (`<model>_impl_dae_fun.c`, listed in the JSON) is
  the shared dynamics backend and is linked in, not regenerated.

## 9. Phases

| Phase | Deliverable |
|---|---|
| **P0** | This plan + decisions locked (§10). ✅ |
| **P1a** | Add `GaussLegendre4Tag` + tableau to `butcher.hpp` (ocp++ prereq). Build clean. ✅ |
| **P1b** | Run the `masses_chain` codegen → `codegen_masses_chain/`; move the example + codegen into `examples/masses_chain/`. ✅ |
| **P1c** | Build the compiled acados reference main; capture `ref_acados.csv`. ✅ |
| **P1d** | Hand-write the ocp++ wrapper (`capi.hpp`, `*_ode.hpp`, `masses_chain.hpp`, `main.cpp`) + CMake; build warning-free. ✅ |
| **P1e** | Run ocp++ main; compare to reference; iterate until within §7 tolerances; emit CSV. ✅ (cost rel 4e-7) |
| **P2** | Build `tools/acados2ocp_pp.py` (Python, JSON-driven); verify it reproduces the P1d wrapper for `masses_chain` via `gen_equivalence.cpp`. ✅ |
| **P3** | Apply generator to more examples, one `examples/<name>/` each: **pendulum_on_cart ✅, unicycle ✅** (both Python-codegen; sign-convention fix §5.1); furuta / control_rates / swarming need P5 features (NONLINEAR_LS / ERK) — moved to P5. |
| **P4** | (later) mex interface for ocp++ solvers; direct MATLAB comparison. |
| **P5** | Extend the generator to the full JSON feature space (§12): nonlinear/external cost, stage-0 cost, free x0, linear + nonlinear constraints, soft (slack) constraints, ERK + Radau-IRK integrators. No example needed; validate via unit-level emission tests + any new example ported afterward. |

## 10. Decisions (locked 2026-10-03)

1. **Integrator tableau** — add a **Gauss-Legendre 4-stage** (`GaussLegendre4Tag`)
   tableau to `butcher.hpp` so the IRK matches acados exactly (P1a).
2. **Phase ordering** — hand-written wrapper first (P1d) to de-risk the ocp++
    side end-to-end; the JSON generator (P2) then reproduces it.
3. **Reference capture** — add a CSV-dump block to the generated
   `main_ocp_masses_chain_*.c` → `ref_acados.csv`.
4. **Reproducible comparison** — `examples/masses_chain/compare.sh` builds+runs
   both sides and diffs the CSVs. ocp++ core stays acados-lib-free; only the
   reference target links acados.

## 11. Immediate next steps (after §10 answers)

1. P1a: `GaussLegendre4Tag` + `ButcherTableau` specialization (copy GL4 nodes /
   A / weights from acados `sim_collocation_utils.c`); add a unit test in
   `tests/integrators/` sanity-checking the collocation identities.
2. P1b: run the codegen; relocate the example + codegen into
    `examples/masses_chain/`.
3. P1c: build + run the reference main; produce `ref_acados.csv`.
4. P1d/e: wrapper + demo + comparison; iterate to tolerance.

## 12. Generator scope matrix (updated 2026-10-04)

Full `acados_ocp_nlp.json` feature space → ocp++ support status. The generator
must accept every row marked **supported**, and reject every row marked
**deferred / out-of-scope** with a precise `Unsupported` message naming the
offending JSON field.

| # | acados JSON feature | ocp++ support | Status |
|---|---|---|---|
| 1 | `sim_method = IRK`, `collocation_type = GAUSS_LEGENDRE`, stages 2/3/4 | `ImplicitRkIntegrator` + `GaussLegendre{2,3,4}Tag` | ✅ done |
| 2 | `sim_method = IRK`, `collocation_type = RADAU`, stages 2/3/4 | `RadauIia{2,3,4}Tag` | P5 |
| 3 | `sim_method = ERK` (K1–K4), `sim_num_stages` | `ExplicitRkIntegrator` K1–K4 | P5 |
| 4 | `sim_method = DISCRETE` | no discrete-Problem base yet | deferred |
| 5 | `sim_method = GENERIC` (external sim) | no external sim hook | deferred |
| 6 | `num_stages` per-stage list (variable stages over horizon) | single-stage IRK config only | deferred |
| 7 | `cost_type / cost_type_t = LINEAR_LS` | emitted `stage_cost_*` | ✅ done |
| 8 | `cost_type_0 = LINEAR_LS_0` (stage-0 cost) | `stage_cost_*(k)` already takes k | P5 |
| 9 | `cost_type = EXTERNAL / NONLINEAR_LS / CONVEX_OVER_NONLINEAR` | cost C files (fun/jac/hess) via capi | P5 |
| 10 | box constraints (`idxbx/lbx/ubx`, `idxbu`, `idxbx_e`) | `BoxSpec` | ✅ done |
| 11 | linear path/terminal constr (`Cx,Cu,lb,ub`, `Cxe,Cue,lbe,ube`) | `LinearSpec` / `TerminalLinearSpec` | P5 |
| 12 | nonlinear path/terminal constr (`constr_ext_fun_type=casadi`) | constraint C files via capi | P5 |
| 13 | soft/slack constraints (`ns, nsbx, nsg, ...`, soft-penalty `Zl/Zu`) | `*_soft_idx` arrays + `soft_penalty` in `BoxSpec` | P5 |
| 14 | fixed initial state (`lbx_0 == ubx_0`) | `fixed_initial_state=true` | ✅ done |
| 15 | free initial state (`lbx_0 != ubx_0`) | `fixed_initial_state=false` + stage-0 box | P5 |
| 16 | DAE with algebraic vars (`nz > 0`) | solver must handle z in KKT — not yet | deferred |
| 17 | time-varying model (`t` in model signature) | Ode contract has `t` | P5 (check) |
| 18 | non-uniform `time_steps` | `ContinuousProblem` assumes uniform dt | deferred |
| 19 | `hessian_approx = GAUSS_NEWTON` | `has_dynamics_hess_prod=false` | ✅ done |
| 20 | `hessian_approx = EXACT / ROSEN / NONE` | exact/ROSEN HVP for DAE residual | deferred |
| 21 | `globalization = SQP_STEP / FILTER / TRUST_REGION` | fixed-step SQP only | deferred |
| 22 | `qp_solver = FULL_CONDENSING_HPIPM / SQR_METHOD / QPOASES` | `PartialCondensingHpipm` only | deferred |

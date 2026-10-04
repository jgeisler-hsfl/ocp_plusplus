# GN Hessian for the dynamics (TODO item "implement GAUSS_NEWTON")

> TODO item: *"implement GAUSS_NEWTON for dynamics part of the cost function.
> add switch in dims to activate."*
>
> Companion to MATLAB_GENERATOR_PLAN.md §5.3 / §7 / §12 (row 19). This plan
> records what acados's `hessian_approx` actually does (verified against the
> acados source at `/home/jgeisler/repos/acados`), what ocp++ already does,
> and the remaining work (QP-level verification + interface docs).

## 1. What acados `hessian_approx` actually does (source-verified)

Current acados interface: `hessian_approx ∈ {GAUSS_NEWTON, EXACT}`
(`acados_ocp_options.py:462`; the historical enum also had `NONE` and
`ROSEN`, no longer exposed). Per-module overrides: `exact_hess_cost`,
`exact_hess_dyn`, `exact_hess_constr` (`ocp_nlp_common.c:1515-1544`).

The QP stage Hessian `RSQrq[k]` (over the stage variables `(u_k, x_k)`,
i.e. the same layout ocp++ uses: `st.hess` in (u;x) order) is assembled as:

| acados mode | cost contribution | dynamics contribution | constraint contribution |
|---|---|---|---|
| `EXACT` | `J_yᵀ W J_y` **+** 2nd-order residual term (the generated `nls_y_hess` fn, `ocp_nlp_cost_nls.c:645-660`) | `+ S_hess`: multiplier-weighted flow-map Hessian from the collocation integrator (`ocp_nlp_dynamics_cont.c:758-768`) | `+ λᵀ H_c` HVP (`ocp_nlp_constraints_bgh.c:1310-1372`; **not supported** in `bgp`, `ocp_nlp_constraints_bgp.c:1333-1337`) |
| `GAUSS_NEWTON` | `J_yᵀ W J_y` only (`ocp_nlp_cost_nls.c:630-635`: `dsyrk` of `tmp_nv_ny = Jᵀ W_chol`) | **nothing** — the dynamics enter the QP only through the linearized `BA` matrix | **nothing** |
| `ROSEN` (historical) | same as `EXACT` (exact cost Hessian) | same as `GAUSS_NEWTON` | same as `GAUSS_NEWTON` |

Key facts:

1. **ROSEN vs GAUSS_NEWTON differ only in the cost Hessian**: ROSEN keeps the
   exact `∇²L_cost` (incl. the 2nd-order residual Hessian term), GAUSS_NEWTON
   uses the GN form `J_yᵀ W J_y`. For least-squares costs (LINEAR_LS /
   NONLINEAR_LS with affine `y`) the two are **identical**.
2. **No mode adds a `J_dynᵀ J_dyn` term to the QP Hessian.** In the
   structured (partial-condensing) QP the linearized dynamics live in the
   `BA` rows of the KKT system; the only *second-order* dynamics
   contribution is the exact flow-map Hessian, and only in `EXACT` mode.
   (Verified: no `dsyrk`/`dgemm` on the dynamics Jacobian anywhere in the
   `ocp_nlp_*` dynamics modules; and empirically — see §3.)
3. `compute_hess` / `exact_hess` are the runtime gates; `hessian_approx`
   selects the combination at solver creation.

## 2. ocp++ current state (already implements the GN behavior)

- **Cost Hessian** is user-provided per stage (`stage_cost_hessian` /
  `terminal_cost_hessian`): the user decides exact vs GN form. The generated
  wrappers (masses_chain, pendulum, unicycle) all provide the GN form
  `Vᵀ W V`, which for LINEAR_LS equals the exact cost Hessian — matching
  acados in both `ROSEN` and `GAUSS_NEWTON` for these examples.
- **Dynamics second-order term**: gated by the compile-time
  `Dims::has_dynamics_hess_prod` **and** the runtime
  `SqpOptions::compute_hess` (sqp.hpp `assemble_first` / `assemble_path`):
  `st.hess -= build_dyn_hvp(...)`, i.e. the exact term `−λ_dynᵀ H_f`
  (constraint residual `x_{k+1} − f_k(x_k,u_k)`, Lagrangian Hessian w.r.t.
  `(x_k,u_k)` is `−λᵀ H_f`).
- **Constraint HVPs**: gated by `Dims::has_constr_hess_prod` + `compute_hess`.

Mapping to acados:

| ocp++ setting | QP Hessian | acados equivalent |
|---|---|---|
| `has_dynamics_hess_prod=true`, `compute_hess=true`, exact cost H | `H_cost − λᵀH_f` (+ constr HVPs) | `EXACT` |
| no dynamics HVP, exact cost H | `H_cost` | `ROSEN` |
| no dynamics HVP, GN cost H (`VᵀWV`) | `H_cost = JᵀWJ` | `GAUSS_NEWTON` |

So the "switch in dims" from the TODO item **exists**
(`has_dynamics_hess_prod`), and the GAUSS_NEWTON dynamics behavior (no
second-order dynamics term; linearized `BA` only) is **implemented**. All
three ported acados examples declare `hessian_approx: GAUSS_NEWTON` in their
JSON and ocp++ sets `has_dynamics_hess_prod = false`; masses_chain matches
acados to cost rel. 4e-7 / ‖Δx‖∞ 1e-8, which is only possible if the QP
Hessians agree (a missing dynamics JᵀJ term would be a systematic O(1)
difference, not 1e-8).

The ROSEN/GN distinction for ocp++ lives entirely on the cost-Hessian side
(user's choice per `stage_cost_hessian`); no separate solver switch is
needed (and current acados no longer exposes ROSEN at all).

## 3. Work (status: done, 2026-10-04)

### 3a. QP-level verification

1. **Unit test `tests/sqp/hessmode_5a.cpp`** (added to the `sqp_unit`
   target in `CMakeLists.txt`) — DONE. A small synthetic problem with
   known analytic `H_f` and `H_cost` and nonzero fixed multipliers:
   - GN mode (`has_dynamics_hess_prod=false`): `H_qp == H_cost` exactly
     (0.0 error), `BA == [B|A]`, `b == f(x,u) − x̄` — with nonzero
     `λ_dyn`, so the absence of the `−λᵀH_f` term is discriminating.
   - EXACT mode (`has_dynamics_hess_prod=true`, `compute_hess=true`):
     `H_qp == H_cost − λᵀH_f` + constraint HVP terms, checked against
     analytic values.
   - **FD cross-check of the EXACT path**: central-FD of the Lagrangian
     gradient `∇[L_cost + λ_dynᵀ(x̄ − f)]` w.r.t. `(u_k, x_k)` matches
     `H_qp·v` — the QP-assembly sign/layout is now FD-verified
     (the CSTR 4i test only FD-checks `hess_prod` at the integrator level).
   - All checks pass (all reported errors 0.0).
2. **QP-level equivalence for masses_chain
   (`tests/sqp_masses_chain/sqp_masses_chain.cpp`, check (E))** — DONE.
   Assembles the QP at the common warm start with **nonzero** `λ_dyn` and
   asserts, to 1e-12, that every stage/terminal Hessian equals the analytic
   cost Hessian (`10·I₂₄ ⊕ 1e-2·I₃` per stage, `10·I₂₄` terminal), the
   gradient equals the permuted cost gradient, and `BA`/`b` equal
   `[B|A]` / `Φ(x,u) − x_{k+1}` from the problem's own Jacobian/map.
   Since masses_chain is LINEAR_LS (cost Hessian exact and
   mode-independent) and acados GN adds no dynamics term (§1,
   source-verified), this proves the ocp++ QP Hessian equals the acados-GN
   QP Hessian exactly.
   - *Optional follow-up:* a byte-level dump of acados' iter-0
     `RSQrq`/`BAbt`/`b` (no built-in dump in the codegen main; would need a
     small C driver against the codegen lib in
     `examples/masses_chain/acados_codegen`) for a direct matrix diff.
   - Note: the (B)/(C) final-solution gaps vs acados are *not* a Hessian
     difference — both sides build the same cost-only QP Hessian; they come
     from the central-FD dynamics Jacobian (vs CasADi analytic) amplified
     by the weakly-convex KKT system.

### 3b. Interface / documentation cleanup ("switch in dims") — DONE

- `problem.hpp`: `has_dynamics_hess_prod` docs rewritten (Dims example,
  interface note, flag comment): `false` = no exact dynamics HVP, the SQP
  uses the Gauss-Newton Hessian (dynamics only via the linearized BA;
  acados `GAUSS_NEWTON`/historical `ROSEN`); `true` = exact Lagrangian
  Hessian (acados `EXACT`).
- `sqp.hpp`: `compute_hess` documented as the runtime gate mirroring
  acados' `exact_hess`/`compute_hess`, selecting EXACT vs pure cost
  (Gauss-Newton) QP Hessian.
- `MATLAB_GENERATOR_PLAN.md`: §7 corrected (both QPs carry the same
  cost-only Hessian; gaps come from the dynamics Jacobian, not the
  Hessian model); §12 row 19 → done, row 20 → EXACT supported by the
  solver (generator emits the HVP wrapper in P5).
- `examples/masses_chain/masses_chain.hpp` +
  `tests/sqp_masses_chain/sqp_masses_chain.cpp`: the stale
  "acados = cost + dynamics J'J" deviation notes replaced with the
  corrected explanation (dynamics-Jacobian accuracy).
- `TODO.md` item 8: marked `[x]`.

### 3c. Explicit non-goals (documented, not implemented)

- **Textbook `J_dynᵀJ_dyn` dynamics GN term**: not what acados does in the
  structured QP (the BA rows already carry the dynamics in the KKT system;
  adding JᵀJ to `H` would double-count the linearization). Do not add.
- **Per-group GN/EXACT switches** (acados `exact_hess_cost/dyn/constr`):
  ocp++'s single `compute_hess` gate suffices for now; note as follow-up.
- **Collocation cost sensitivity** (`J_stage = J_y·S_forw_stage`, acados
  `cost_computation` path): applies to *continuous-time integrated* costs;
  the ocp++ discrete interface evaluates the cost on `(x_k, u_k)`, so it
  does not occur here.

## 4. Planned commits

1. `solver: add FD-checked QP Hessian assembly test (exact vs GN)`
2. `test: add masses_chain QP-level acados equivalence check`
3. `problem: document GN/EXACT semantics of has_dynamics_hess_prod`
4. `docs: correct plan §7 Hessian claim; update GN status`

## 5. Verification

- `cmake -B build && cmake --build build` warning-free
  (`-Wall -Wextra -Werror`).
- Full test suite passes (incl. new `hessmode_5a`, updated
  `sqp_masses_chain`).
- `examples/masses_chain/compare.sh` still green.

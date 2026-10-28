# Receding-Horizon (MPC) Shift Helper

Status: DESIGN SETTLED (consulted 2026-10-28), ready to implement.

## 1. Goal

Advance TODO.md item B:

> add receding horizon shifting to Solution, what about the solver-internal slacks?

Provide a framework-level helper that advances a previously solved
`ocp::Solution` (and the SQP driver's internal slack state) by one stage
so it can be used as the warm start of the next MPC step, instead of the
hand-rolled copy loops currently shown in tutorial ch. 07 §7.3.

## 2. Background / current state

- The MPC loop is currently documented (not implemented) as manual code in
  `docs/tutorial/07-advanced-topics.md` §7.3: copy a `Solution`, shift
  `x`/`u`/`lambda_dyn`/`lambda_box_state` one stage up, set `x[0]` to the
  new measurement, zero the last `u`, re-solve.
- `ocp::Solution<P, NH>` (in `include/ocp/problem.hpp`) is the shared,
  solver-independent warm-start/result container:
  - `x` (N+1), `u` (N)
  - `lambda_dyn` (N), `lambda_ineq_stage` / `lambda_eq_stage` /
    `lambda_lin_stage` (N each)
  - `lambda_ineq_term` / `lambda_eq_term` / `lambda_lin_term` (terminal)
  - `lambda_box_state` (N+1), `lambda_box_control` (N)
  - `cost_value`, `status`
- The SQP driver (`AcadosSolver`, `include/ocp/solvers/acados/acados.hpp`)
  owns solver-internal state that persists across `solve()` calls:
  - `slacks_` (`SqpSlacks<P, NH>` in `globalize.hpp`): NLP slacks,
    warm-started across solves **only when the horizon matches**
    (`slack_N_ != N` → `slacks_.setZero()`).
  - QP dual warm start: persistent `qp_out_` / `dense_qp_out_` /
    `block_qp_.out` carry the last QP's `lam`/`t` into the next QP solve
    when `HpipmOptions::warm_start` is 2 (primal+dual) or 3 (hot);
    the primal step is always zeroed before the QP.
  - Globalizer (`glob_`) and adaptive LM state are **reset at the start of
    every `solve()`** — nothing to shift.
- Note: the persistent QP duals live in the QP's *scaled, step-form,
  tau-shifted* space (the QP is re-assembled from the NLP problem each
  SQP iteration). The NLP-level multipliers inside `ocp::Solution` are the
  natural shiftable quantity.
- `Dims::fixed_initial_state`: when true, the solver pins `x_0` from
  `problem.initial_state()` (user-managed problem state); `sol.x[0]` is
  then informational only. A shift helper cannot update the problem's
  internal state — the user must do that separately.

## 3. Settled design (consulted 2026-10-28)

1. **Scope**: `Solution::shift()` (problem.hpp) **plus** an
   `AcadosSolver::shift()` that also shifts the driver's internal
   `slacks_` one stage. QP-level duals (`qp_out_`, `dense_qp_out_`,
   `block_qp_.out`) are **not** shifted — they stay warm-started as-is
   through the existing `HpipmOptions::warm_start` option (they are a
   second-order warm start in scaled step-space; the NLP multipliers are
   the primary one).
2. **Cold start**: warm by default (status quo); add an explicit
   `AcadosSolver::reset_warm_start()` that zeros `slacks_` (via
   `slack_N_ = 0`, so the next `solve()` re-zeroes if the horizon
   changed) and the persistent QP duals (`qp_out_` lam/t/pi,
   `dense_qp_out_` lam/t, `block_qp_.out` lam/t) — a one-shot cold
   start, callable after `shift()` when a clean slate is wanted.
3. **New last stage**: policy enum `ShiftFill` with two values; default is
   rollover of the primal variables + zeroed multipliers (the sensible
   choice per consultation):
   - `kRollover` (default): `x[N]` keeps the old terminal state,
     `u[N-1]` keeps the old last control; the new stage-N-1 multipliers
     (`lambda_dyn`, stage ineq/eq/lin, `lambda_box_control`) are zeroed;
     terminal multipliers are unchanged.
   - `kZero`: `x[N]` and `u[N-1]` are zeroed; same multiplier treatment.
4. **New initial state**: `shift()` takes it as an argument —
   `sol.shift(x0_new, fill)` writes `sol.x[0]` in the same call.

### 3.1 The shift itself

For horizon N, one-stage receding-horizon shift of `Solution`:

| quantity                    | rule                                      |
|-----------------------------|-------------------------------------------|
| `x[k]` (k=0..N-1)           | `x[k] = x[k+1]`                           |
| `x[0]`                      | `= x0_new` (after the shift)              |
| `u[k]` (k=0..N-2)           | `u[k] = u[k+1]`                           |
| `u[N-1]`                    | keep old value (kRollover) / 0 (kZero)    |
| `lambda_dyn[k]` (k=0..N-2)  | `= lambda_dyn[k+1]`                       |
| stage ineq/eq/lin multipliers (k=0..N-2) | `= [k+1]`  |
| stage-N-1 multipliers       | zeroed                                    |
| `lambda_box_state[k]` (k=0..N-1) | `= lambda_box_state[k+1]`      |
| `lambda_box_control[k]` (k=0..N-2) | `= lambda_box_control[k+1]` |
| terminal multipliers        | unchanged (terminal stage stays terminal) |
| `x[N]`                      | keep old value (kRollover) / 0 (kZero)    |
| `cost_value`, `status`      | reset (0, `kUnset`)                       |

Works in both dynamic (`NH = Eigen::Dynamic`, in-place vector moves) and
fixed (`NH >= 1`, in-place `std::array` moves) extent modes.

Driver-side slack shift (same mapping, `slacks_` only):

| quantity            | rule                                    |
|---------------------|------------------------------------------|
| `slacks_.first`     | `= slacks_.path[0]` (if N > 1), else zeroed |
| `slacks_.path[k]` (k=0..N-3) | `= slacks_.path[k+1]`        |
| `slacks_.path[N-2]` | zeroed (old terminal slacks have a different extent) |
| `slacks_.term`      | unchanged                                |

Only applied when `slack_N_ == sol.N` (i.e. the driver has been solved at
this horizon before); otherwise the next `solve()` zeroes the slacks as
today.

### 3.2 API sketch

```cpp
// problem.hpp
enum class ShiftFill
{
    kRollover,  // x[N], u[N-1] keep their previous values
    kZero       // new last-stage x[N], u[N-1] zeroed
};

// ocp::Solution<P, NH>
void shift(const state_t& x0_new, ShiftFill fill = ShiftFill::kRollover);

// AcadosSolver<P, NH, ...>
void shift(Solution<P, NH>& sol, const P::state_t& x0_new,
           ShiftFill fill = ShiftFill::kRollover);   // solution + slacks_
void reset_warm_start();   // one-shot cold start (slacks_ + QP duals)
const SqpSlacks<P, NH>& slacks() const;  // read-only, for tests/diagnostics
```

`RtiSolver` carries the same `slacks_`/`slack_N_` members; `shift` /
`reset_warm_start` / `slacks()` can be mirrored there cheaply, but RTI is
a single-shot LQR-style solver — noted as an optional follow-up, not part
of this change unless trivial.

## 4. Implementation blocks

- [x] **1. `Solution::shift`** — enum + member in
      `include/ocp/problem.hpp`; in-place, both NH modes; Doxygen with the
      mapping table; keep the header self-contained (Eigen only).
- [x] **2. Driver integration** — `AcadosSolver::shift`,
      `reset_warm_start`, `slacks()` accessor in
      `include/ocp/solvers/acados/acados.hpp` (mirror in `rti.hpp` only
      if trivial).
      * Finding: `SqpSlacks::first` and `path[0]` have different sizes when
        `fixed_initial_state && nbx_soft > 0` (the first stage drops the
        state-box block, see `QpDim::nslack_first`). `shift()` therefore maps
        `first <- path[0].segment(2*nbx_soft, nslack_first)` instead of a
        plain assignment (compiles for all combinations).
      * `reset_warm_start()` zeros `slacks_` + `slack_N_ = 0` and the
        persistent QP duals (staged `qp_out_` pi/lam/t, dense `dense_qp_out_`
        lam/t, block `block_qp_.out` pi/lam/t when NB > 0). Primal QP steps
        are always zeroed before the QP anyway (`solve_qp`), so they need no
        handling.
- [x] **3. Unit tests** —
      - `tests/problem/solution_shift_unit.cpp`: dynamic-NH and
        fixed-NH; verify every shifted entry, `x[0]` set, both fill
        policies, terminal multipliers untouched, cost/status reset.
      - `tests/sqp/rh_shift.cpp`: solve double integrator → shift →
        verify slacks mapping via the accessor → re-solve (warm) and
        solve after `reset_warm_start()` (cold); both must converge,
        warm expected to need <= SQP iterations.
      - Register both in CMakeLists.txt (pattern: existing unit tests).
      * Findings:
        - `AcadosSolver::shift` needed `typename P::state_t` in the
          signature (dependent scope) and `P::nbx_soft` /
          `P::fixed_initial_state` (soft counts live on the problem, not
          on `QpDim`).
        - `warm_start_first_qp=true` on a fresh solver (no persistent QP
          duals yet) makes the first QP clip zero duals to 0.1 and
          diverge (`res_comp = inf`); it only makes sense once the solver
          has dual history. The test therefore enables it only before
          the post-shift warm solve.
        - No IPM-level iteration assertion: the driver always
          cold-starts the primal step, and for the small, well-conditioned
          DI QP the ws=0 heuristic primal init is competitive with the
          dual-warm start (cf. warm_start_3c). The warm/cold comparison
          is asserted at the SQP level only.
- [x] **4. Docs** — tutorial ch. 07 §7.3: replace the manual snippet with
      the helper (keep a short manual version for reference); note
      `reset_warm_start`; update `docs/ARCHITECTURE.md` if the
      solver-internal-state note needs the new accessor; mark the TODO
      item done in `docs/plans/TODO.md` before committing.

## 5. Test plan

- `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`
  (warning-free under `-Wall -Wextra -Werror`).
- New unit tests from §4.3 + the existing `double_integrator` sanity
  test and `status_unit` / `inspect_unit` still pass.

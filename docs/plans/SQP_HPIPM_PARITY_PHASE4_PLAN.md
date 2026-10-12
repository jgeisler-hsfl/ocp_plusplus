# QP sensitivities (HPIPM seed module) — Phase 4

Status: **active** (no blocks done yet). Implements the missing master-plan
row "QP sensitivities (forward/adjoint, HPIPM seed module)". Ports the
HPIPM seed mechanism of the *staged* OCP-QP IPM solver into our
`HpipmQpSolver`. NLP-level parameter sensitivities remain out of scope
(master plan §9 item 3: needs a parameter-vector concept in `problem.hpp`).
Dense-QP and partial-condensing seeds (HPIPM `d_dense_qp_seed` /
part-cond variants) are deferred — the staged seed module only.

Reference files (read at implementation time; do not modify):
- `../acados/external/hpipm/include/hpipm_d_ocp_qp_seed.h` — `d_ocp_qp_seed`
  data model: `seed_g` (per stage, `nu+nx+2ns`), `seed_b` (per dynamics
  step, `nx`), `seed_d` / `seed_m` (per stage, `2nb+2ng+2ns` sides).
- `../acados/external/hpipm/ocp_qp/x_ocp_qp_seed.c` — create / set_zero /
  `set_seed_*` setters (note: `set_seed_ub/ug/us` apply a `-1` scale
  because HPIPM's `d_hi = v - hi`).
- `../acados/external/hpipm/ocp_qp/x_ocp_qp_ipm.c` — the actual
  computation (NOT in the seed files):
  - `OCP_QP_IPM_SENS_FRW` (line ~3383): alias the sensitivity vectors
    into the core workspace, load the converged solution from the `*_bkp`
    copies, build a `tmp_qp` that keeps the LHS (`BAbt`, `RSQrq`, `DCt`,
    `Z`, `d_mask`, `idxs_rev`, `diag_H_flag`, `idxb`) but takes the seeds
    as RHS (`b = seed_b`, `rqz = seed_g`, `d = seed_d`, `m = seed_m`),
    set `ws->use_Pb = 0`, call `OCP_QP_SOLVE_KKT_STEP(&tmp_qp, sens, ...)`.
    The result left in `sens` (converged solution + step) is the forward
    sensitivity `d(sol)/d(param)`.
  - `OCP_QP_IPM_SENS_ADJ` (line ~3455): identical except (a) `seed_m` is
    scaled by `t_bkp` (per side) before the KKT step and (b) the
    resulting `sens->t` is scaled by `1/t_bkp` afterwards.
- `../acados/external/hpipm/ocp_qp/x_ocp_qp_kkt.c` —
  `OCP_QP_SOLVE_KKT_STEP` (line 2306): the re-solve that reuses the
  factorization left on exit. It recomputes `COMPUTE_GAMMA_QP` from the
  current `d`/`m` and the solution's `lam`/`t` (Gamma = lam/t is
  factor-consistent; the lowercase gamma forcing uses the current
  `res_m - lam*res_d`), then runs the two-pass Riccati without any
  refactoring.
- `../acados/external/hpipm/examples/c/example_d_ocp_qp_sens.c` — usage
  pattern: solve once, create a zero seed, set a single unit seed, call
  `d_ocp_qp_ipm_sens_frw` on the same `workspace` (factors still
  resident), read the sensitivities from the `sens` solution struct.

## Algorithm (summary)

Both forward and adjoint are a *single* KKT re-solve with the factorization
left on exit by the converged `solve()`; no IPM loop, no line search.

1. Build the seeded QP: copy the LHS (hess, BA, DC, d_mask) from `in`,
   replace the RHS fields (`grad`, `b`, `d`, `m`) with the seed vectors.
2. Force term: the seed vectors ARE the KKT force (HPIPM aliases them into
   `res_g/res_b/res_d/res_m` inside `SOLVE_KKT_STEP`); no centering shift
   (the sens call happens outside the IPM loop, sigma = 0).
3. Solve the KKT system with the stored factors:
   `solve_kkt_reuse_gamma(in_seed, sol, rhs, step)` — recomputes
   Gamma = lam*/t* (from `sol`), runs the backward-reduced / forward /
   closed-form passes over the stored L/P factors.
4. **`sens` = the step itself (ux, pi, lam, t deltas).** HPIPM's
   `SOLVE_KKT_STEP` overwrites the sens vector with the linear-system
   solution (`VECCP(res_g, dux)`, x_ocp_qp_kkt.c:2357); the bkp load in
   `SENS_FRW` only supplies `cws->lam/t` for gamma + the closed-form dual
   update. The caller adds: `new_sol ≈ sol + sens` for a unit parameter
   change. (Not a preloaded solution plus step.)
5. Factorization: our `solve()` does NOT refactor at exit (the
   `update_fact_exit` option is declared, not yet implemented — the stored
   L/P factors are from the last *predictor* of the final iteration, i.e.
   one update stale). Therefore `sens_forward` performs a fresh
   `fact_solve_kkt(in_seed, sol, rhs, step)` on the FIRST call after a
   `solve()` (the HPIPM `update_fact_exit` refactor at the final iterate);
   later calls with other seeds reuse the factors via
   `solve_kkt_reuse_gamma` (factors depend only on LHS + final lam/t, not
   on the seed). A `has_factors_` latch tracks this; `solve()` resets it.
6. Adjoint only: before the solve, scale `in_seed.m` by `sol.t` (per
   side); after, scale `sens.t` by `1/sol.t` (per side).

Field mapping (HPIPM `OCP_QP` / seed  ->  ocp++ `Qp` / `QpSeed`):

| HPIPM            | seed vector | ocp++ Qp field | QpSeed field |
|------------------|-------------|----------------|--------------|
| `rqz` (per stage)| `seed_g`    | `*.grad`       | `grad_*`     |
| `b` (per dyn step)| `seed_b`   | `*.b`          | `b_first/path` |
| `d` (sides)      | `seed_d`    | `*.d`          | `d_*`        |
| `m` (sides)      | `seed_m`    | `*.m`          | `m_*`        |

Side layout: `QpSeed` uses our native side layout (lo block, hi block,
slack sides per `QpLayout`), NOT HPIPM's `[lb;lg;ub;ug;ls;us]`. The
hi-side sign convention is the same as `Qp::d` (`w_cur - hi`): a seed of
an *upper* bound value enters `d` with a minus sign, mirroring HPIPM's
`set_seed_ub/ug/us` `-1` scaling.

## API design

`include/ocp/solvers/hpipm/seed.hpp` (new):

```
template <class P, int NH = Eigen::Dynamic, class Dim = QpDim<P>>
struct QpSeed
{
    int N;
    // rqz seeds: linear cost term over (u;x;s) / (x;s), per stage type
    grad_first / grad_path (N-1) / grad_term
    // dynamics offset seeds (k = 0..N-1)
    b (N entries)
    // constraint offset seeds (side layout, same as Qp::d)
    d_first / d_path (N-1) / d_term
    // complementarity RHS seeds (side layout, same as Qp::m)
    m_first / m_path (N-1) / m_term
    QpSeed() / QpSeed(int n_stages) / resize(int) / setZero()
};
```

`include/ocp/solvers/hpipm/hpipm.hpp` (`HpipmQpSolver`, public):

```
Status sens_forward(const QpT& in, const QpSolT& sol,
                    const QpSeed<P, NH, Dim>& seed, QpSolT& sens);
Status sens_adj(const QpT& in, const QpSolT& sol,
                const QpSeed<P, NH, Dim>& seed, QpSolT& sens);
```

Contract:
- `in` must be the QP that was last successfully `solve()`d on this
  instance (same LHS); `sol` its converged solution.
- Factorization: `solve()` resets a private `bool has_factors_` latch.
  The FIRST sens call after a `solve()` does a fresh `fact_solve_kkt`
  at the final iterate (the HPIPM `update_fact_exit` refactor, which our
  `solve()` does not do on exit); subsequent sens calls (any seed) reuse
  the factors via `compute_gamma` + `solve_kkt_reuse_gamma` (factors
  depend only on LHS + final lam/t + reg, not on the seed). A subsequent
  `solve()` refactors and resets the latch.
- Unconstrained QP (all d_mask = 0): no gamma/factor path — a single
  `solve_kkt_unconstr` on the seeded QP; the adjoint t-scaling is
  skipped (no sides; avoids 0/0 in the m = 0 case).
- Staged solver only (`Dim = QpDim<P>`, compile-time `static_assert`);
  `HpipmDenseQpSolver` / block condensing get no seed support in this
  phase.

## Blocks

1. **4a — seed data model.** `seed.hpp` with `QpSeed<P, NH, Dim>`
   (fixed-size Eigen, same NH mechanism as `Qp`/`QpSol`), `setZero()`.
   Compile smoke via the 4d test (no behavior yet).
2. **4b — `sens_forward`.** Private `sens_solve<Adj>` helper implementing
   the algorithm above (seeded QP + forced QpRes, lazy exit-refactor on
   first use, `solve_kkt_reuse_gamma` with explicit `compute_gamma`
   thereafter); public `sens_forward`; `has_factors_` latch (reset in
   `solve()`, set after the first successful sens factorization).
3. **4c — `sens_adj`.** Adjoint m/t scaling on top of the shared helper.
4. **4d — tests** `tests/hpipm/seed_4d.cpp` (in the `qp_unit` target,
   `run_seed_4d_tests()` called from `qp_unit.cpp` main):
   small staged QP (N = 5, nx = 2, nu = 1, box bounds on u and x so
   active/inactive sides exist), solved with `HpipmQpSolver`.
   - FD cross-check (central differences, h = 1e-6, rel err < 1e-5):
     - `b` seed (stage k, one component) vs perturb `in.b` ± h, re-solve
       (fresh solver), compare `sens.ux` / `sens.pi`.
     - `grad` (rqz) seed vs perturb `in.grad` ± h.
     - Hessian seed: perturb `in.hess` ± h E by a symmetric direction E;
       central difference must match `sens_forward(grad_k = E_k * sol.ux_k)`
       directly: d(sol)/d(H) = -K^{-1} (E z*) and the seed machinery
       applies the same internal negative sign to a grad seed as a grad
       perturbation (d(sol)/d(g) = -K^{-1} e_g), so both are
       -K^{-1} (E z*) with no extra flip.
     - `d` seed: perturb a box lower bound ± h (seed = +e_j on the lo
       side) and an upper bound ± h (seed = -e_j on the hi side).
   - Adjoint: for the same seeds as above, `sens_adj(...).ux ==
     sens_forward(...).ux` (ux part is identical by construction); plus
     the duality identity on two different seeds s1, s2:
     `<ux(s1), v2> == <ux(s2), v1>` (symmetry of K^{-1}).
   - State hygiene: `solve` -> `sens_forward` -> `solve` again on the
     same instance must still converge to the same solution (factors are
     refactored, no stale state).
   - Guard: `sens_forward` on a fresh solver (no prior solve) returns
     `kQpFailure`.
5. **4e — docs.** Master plan parity matrix "QP sensitivities" ->
   done (staged, forward + adjoint; dense / part-cond deferred).
   ARCHITECTURE.md: one paragraph in the QP-solver section (seed module,
   factorization reuse). README: mention in the HPIPM bullet.
   `DUAL_ACCESS_PLAN.md`: no options struct added — note that `QpSeed`
   is plain data (no registry involvement).

Gate for every block: `cmake --build build` warning-free
(`-Wall -Wextra -Werror`), full test suite green (byte-identical output
for the pre-existing targets, modulo wall-clock prints).

## Open items (verify during implementation, not design)

1. Exact force-term semantics: our `solve_kkt_reuse_gamma` recomputes
   gamma from the passed `res` (res_m - lam*res_d over 1/t). HPIPM's
   `SOLVE_KKT_STEP` does the same from `qp->d/m` (the seeds) with no
   centering shift. Confirm by reading our `backward_reduced_stage` /
   `forward_stage` force assembly and, decisively, by the 4d FD tests
   (any semantic slip shows up there).
2. Whether the stored factors remain valid when the last `solve()` ended
   on the LQ auto-switch path (`force_lq`): the L factors come from the
   LQ wide-matrix factorization; the reuse pass must be the LQ-consistent
   one. The 4d test runs with the default `lq_fact = 0` (Cholesky);
   optionally add an LQ-mode FD check if the reuse path differs.
3. `ws_.Pb_*`: declared/resized but apparently never read (the LQ path
   computes the D' L product inline into the wide matrix). If confirmed
   dead, leave as-is (no cleanup in this phase; note for 5-polish).
4. Hessian seeds are expressed as an equivalent `grad` seed in the 4d
   test (d(sol)/d(H) = -K^{-1} (E v*)); the API stays HPIPM-faithful
   (RHS seeds only) — no matrix-seed field in `QpSeed`.
5. `m` seeds: not FD-tested (v1 has m = 0 everywhere; complementarity
   m_relax is not implemented). The API carries `m_*` for completeness
   (HPIPM parity, adjoint scaling path exercised by the m = 0 case).

# Plan: merge `irk_hess` (final sensitivity/parity work) into `master`

Status: **planned (2026-10-25)**. This plan documents how to bring the
`irk_hess` branch — the *final* version of the SQP/HPIPM parity +
sensitivity work (phases 5f–5j) — into `master`, which currently carries the
later docs/license/doxygen rework but **none** of the 5f–5j code.

## Why a merge (not drop / not cherry-pick)

- **Drop** would lose the entire sensitivity feature, its FD-verified tests,
  the new `hpipm` headers, and the plan docs that record the intentional
  equality-constraint deviation. Not acceptable.
- **Cherry-pick**: the 18 `irk_hess` commits are a sequential, interdependent
  chain (5f→5g→5h→5i→5j). Picking them one by one hits the same conflicts with
  more friction and no benefit. Not worth it.
- **Merge**: a dry-run (`git merge-tree --write-tree irk_hess master`)
  conflicts in exactly **3 files / 11 hunks**; everything else auto-merges,
  including the `ARCHITECTURE.md → docs/ARCHITECTURE.md` rename (the 68 new
  lines land correctly at the new path). This preserves the linear 5f→5j
  history as a single reviewable unit.

### What each side contributes
| Area | `irk_hess` | `master` |
|---|---|---|
| 5f–5j sensitivity code | ✅ all | ❌ none |
| Sens tests (`sens_x0_5f`, `sens_param_5i/5j`, `dense_seed_5g`, hpipm units) | ✅ | ❌ |
| New headers (`seed.hpp`, `block_condense.hpp`, `condense.hpp`) | ✅ | ❌ |
| Parity plan (`docs/plans/SQP_HPIPM_PARITY_PLAN.md`) | ✅ **final** (5f–5j done, §9.1/§9.2 verified, §9.3 closed) | ⚠️ stale snapshot ("deferred"/"verify"/"blocked") |
| Equality-constraint deviation (conditioning decision) | ✅ documented (5F/5G plans) + in code (`barrier_mask`) | ❌ |
| Docs reorg into `docs/`, tutorial fixes, `TODO.md` | ❌ | ✅ (10-24/25) |
| GPL `LICENSE` / `NOTICE` (root + per-component) | ❌ | ✅ (10-24) |
| Doxygen "named I/O" comment rework (docu_clean) | ❌ | ✅ (10-10) |

Note: the earlier "parity merge" into `master` (`306d988`) was a **no-op for
code** — `sqp_hpipm_parity` sat at the stale merge base `9440d15`, so it added
no new commits (only a 1-line Allman-brace fix, `709292c`).

### The two genuine code deltas (both are the intended 5f deviation — keep `irk_hess`)
1. `update_vars(in, out, step)` (3-arg staged) vs `master`'s 2-arg
   `update_vars(iter, step)`.
2. `res_m = barr.cwiseProduct(...)` where `barr = barrier_mask(d_mask)`, vs
   `master`'s `res_m = d_mask.cwiseProduct(...)`. Hard-equality sides
   (`d_mask = 2`) carry **no** complementarity.

---

## Execution

> Run in the **master worktree** (`ocp++`, currently at `709292c`). The
> `ocp_plusplus-irk_hess` repo stays on `irk_hess` as the source.

### Phase 0 — Pre-flight
- [ ] Confirm both working trees are clean (`git status`).
- [ ] Safety tag on master: `git tag backup/pre-irk-hess-merge master`.
- [ ] Confirm merge base = `9440d15`; operation = `git merge irk_hess` from `master`.

### Phase 1 — Perform the merge
- [ ] `git merge --no-ff irk_hess` in the master worktree.
- [ ] Expect 3 conflicted files: `include/ocp/problem.hpp`,
      `include/ocp/solvers/hpipm/hpipm.hpp`, `include/ocp/solvers/hpipm/qp.hpp`
      (11 hunks total).

### Phase 2 — Resolve the 11 conflict hunks
Resolution rule of thumb: **take `irk_hess`** (it holds the newer, correct
code + its coherent inline notes), except where a hunk is a pure comment
rewording of the *same* old content or where a stale path must be updated.

**`include/ocp/solvers/hpipm/qp.hpp` (3 hunks)**
- [ ] d_mask-values + "Assembly notes" banner → take `irk_hess` (full
      `0 = absent / 1 = barrier / 2 = hard-equality` semantics + "phase 2").
- [ ] `idxe` doc → take `irk_hess` (carries the `HPIPM idxe` reference; fits
      the parity style).
- [ ] soft-slack vs hard-equality-rows block → take `irk_hess` (the whole
      hard-equality infrastructure — `eq_row_soft`, `term_eq_row_soft`,
      `n_eq_first/path/term`, `has_eq`, `make_eq_rows_*`, `eq_rows_*` — is new
      and required by the 5f code; keep its "Soft-slack column mapping
      (HPIPM `idxs`/`idxs_rev` analogue)" banner).

**`include/ocp/solvers/hpipm/hpipm.hpp` (5 hunks)**
- [ ] `update_vars` call in the staged loop → take `irk_hess`
      (`update_vars(in, out, step)`). **Verify** the 3-arg *definition*
      `void update_vars(const QpT& in, QpSolT& iter, ...)` survives and no
      stale 2-arg staged definition remains.
- [ ] `res_mu` comment + mask loop → take `irk_hess` (barrier sides only,
      `d_mask > 0.5 && d_mask < 1.5`).
- [ ] hard-equality comment at the centering/correction site → take `irk_hess`.
- [ ] `present_mask` / `barrier_mask` + `update_lam_t` doc → take `irk_hess`
      (both helpers are new and used across `solve`, residuals, and init).
- [ ] `res_m` in `init_point` → take `irk_hess`
      (`res_m = barr.cwiseProduct(lam.cwiseProduct(t) - m)`; `barr =
      barrier_mask(d_mask)`).

**`include/ocp/problem.hpp` (3 hunks)**
- [ ] `Dims` example banner → take `irk_hess` (adds `static constexpr int np = 0`
      + richer flag comments).
- [ ] `SoftPenalty` doc + Eigen 3.3 NOTE → take `irk_hess` (the
      "`{}` does not zero Eigen 3.3.9 members → use `::Zero()`" pitfall note is
      new and important).
- [ ] "See also" + `*_sens_p` contract + Notes → **merge both sides**: keep
      `irk_hess`'s `*_sens_p` block and the Notes, but update the "See also"
      cross-reference to the **new `docs/ARCHITECTURE.md` path** (master moved
      `ARCHITECTURE.md` → `docs/ARCHITURE.md`). Do not leave a dangling root
      `ARCHITECTURE.md` reference.

### Phase 3 — Comment/style reconciliation (explicit item)
Taking `irk_hess` for the code means the touched regions carry its **older
inline-comment style**, while the rest of the tree uses the docu_clean
**doxygen "named I/O"** style. Bring the *new public surface* into the doxygen
convention — **scope guard: only merge-touched regions + the new sensitivity
API; no whole-file reformat** (AGENTS.md forbids drive-by reformatting).
- [ ] `problem.hpp`: `*_sens_p` interface stubs, the new `*_sensp_t` / sens
      type aliases, and the `np` flag → `///` doxygen with named I/O, matching
      the surrounding entries. Keep the top **Interface contract** block in
      sync with the stubs (AGENTS.md rule).
- [ ] `acados.hpp`: `sens_x0` and `sens_param` public methods → `///` doxygen
      with *what / dimensions / layout* (`[x; u]` order, stage extents
      `x: N+1`, `u: N`), matching the neighboring methods' style.
- [ ] `qp.hpp`: new hard-equality members (`eq_rows_*`, `n_eq_*`, `has_eq`,
      d_mask semantics) → concise comments consistent with the file.
- [ ] **Preserve** the HPIPM source-line cross-references (e.g.
      `x_ocp_qp_res.c:513-516`, `x_core_qp_ipm_aux.c:545-578`) — they are the
      parity documentation; do not strip them during the style pass.

### Phase 4 — `ARCHITECTURE.md` + `README.md` (explicit item)
- [ ] `docs/ARCHITECTURE.md`: review the 68 new lines from `irk_hess`
      (sensitivity / §8 content); fix any links that now point at a root
      `ARCHITECTURE.md` instead of `docs/`; check against the
      "keep ARCHITECTURE, README and tutorial in sync on merge" note
      (`d8f2fdd`).
- [ ] `README.md`: confirm the feature list mentions the sensitivity
      capabilities (`sens_x0`, `sens_param`, hard-equality QP, all three QP
      modes); add if missing.
- [ ] Verify `docs/plans/finished/SQP_HPIPM_PARITY_PLAN.md` now carries the
      **final** content (5f–5j done, §9.1/§9.2 verified, §9.3 closed) as a
      result of the auto-merged rename+modify.

### Phase 5 — Plan-file housekeeping
- [ ] Move the 5F–5J plan files from `docs/plans/` → `docs/plans/finished/`
      (all are marked completed/done), matching the phase 1–4 move from
      `c921f54`.
      - `SQP_HPIPM_PARITY_PHASE5_5F_SENSX0_PLAN.md`
      - `SQP_HPIPM_PARITY_PHASE5_5G_DENSE_SENS_PLAN.md`
      - `SQP_HPIPM_PARITY_PHASE5_5H_BLOCK_SENS_PLAN.md`
      - `SQP_HPIPM_PARITY_PHASE5_5I_SENS_PARAM_PLAN.md`
      - `SQP_HPIPM_PARITY_PHASE5_5J_SENS_PARAM_CONDENSING_PLAN.md`
      (Leave this merge plan `SQP_HPIPM_PARITY_MERGE_PLAN.md` active until
      done, then move it too.)
- [ ] Update `docs/tutorial/README.md` / `README.md` index if they list plan
      file locations.

### Phase 6 — New tutorial section (explicit item)
**Goal:** add a section deriving a *local linear state feedback control law*
from the x0 sensitivities.
- [ ] Location: a new section in `docs/tutorial/07-advanced-topics.md`
      (which already covers sensitivities). *Alternative:* a new chapter
      `08-sensitivities.md` if the content outgrows one section.
- [ ] Content outline:
      1. Recap: after `solver.solve(p, sol)`, the optimal `(x*, u*)` depends on
         the data; a small perturbation `dx0` of the initial state shifts the
         whole trajectory.
      2. API: `solver.sens_x0(p, sol, dx0, dsol)` → `dsol` is the **first-order**
         response of `(x, u)` to `dx0`.
      3. Gain derivation: one `sens_x0` call per basis vector `e_i` of `x0`
         yields column *i* of `J = d(x*, u*)/dx0`; stack the columns → assemble
         `K = ∂u₀*/∂x₀` (first-stage control sensitivity to the initial state).
      4. **Local linear state feedback law:** `u₀ ≈ u₀* + K·(x₀ − x₀*)`
         (optionally the per-stage open-loop gains `K_k`).
      5. Runnable snippet on `DoubleIntegrator` (mirrors
         `tests/sqp/sens_x0_5f.cpp`, simplified for a tutorial).
      6. Validity note: first-order / local around the solved iterate, **not**
         a global controller.
- [ ] (Optional, if a new chapter is created) add a companion example under
      `examples/tutorial/`.

### Phase 7 — Build & test
- [ ] `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`
      — must stay **warning-free** under `-Wall -Wextra -Werror`.
- [ ] Run the full suite, especially the new sensitivity tests:
      `sqp_unit` (includes `sens_x0_5f`, `sens_param_5i`, `sens_param_5j`),
      `dense_seed_5g`, `qp_unit`, `split_step_3b`, `alpha_1f`,
      `sqp_double_integrator`.
- [ ] Verify the auto-merged `acados.hpp` retains **both** the `709292c`
      Allman-brace fix in `assemble_first` **and** the new `sens_*` methods
      (this file auto-merged — confirm no content was silently dropped).
- [ ] If a tutorial companion example was added, build + run it.

### Phase 8 — Commit, push, cleanup
- [ ] Commit in logical units (one concern each, AGENTS.md imperative
      subjects):
      1. the merge with all conflict resolutions;
      2. `docs: move 5F–5J parity plans to finished/` (+ index updates);
      3. `docs: add tutorial section on local linear feedback from
         sensitivities`;
      4. `docs: sync ARCHITECTURE/README with the sensitivity feature`
         (if changed);
      5. comment-style reconciliation (if kept as a separate commit).
- [ ] Push `master` (local `master` is ahead of `origin/master` at `34d0c70`
      and has **not** been pushed yet — one push covers the parity no-op merge
      + this merge).
- [ ] Decide the fate of the `irk_hess` branch once it is fully in `master`
      (keep for now; delete only after confirmation).

---

## Risks / watch-items
- **Comment drift:** taking `irk_hess` code makes Phase 3 mandatory for style
  consistency — this is the main reason the merge is not a plain "take theirs".
- **Do not lose master-only fixes:** `acados.hpp` auto-merges, so the `709292c`
  fix is preserved; verify in Phase 7.
- **Stale paths:** the `*_sens_p` "See also" block and any `ARCHITECTURE.md`
  links must point at `docs/…` after the reorg (Phase 2 + Phase 4).

# Item 5 — Linear + quadratic slack penalty weights (acados `Z`/`z` parity)

TODO item 5: "add linear soft constraint penalties, just like in acados/hpipm".

## Semantics (verified against the acados reference)

acados models the soft-constraint slack cost as

```
0.5 s' Z s + r_s' s        (Z diagonal, one weight per slack side; r_s a
                            per-side gradient weight)
```

- `ocp_nlp_cost_common`: `Z_usr` (Zl/Zu, "diagonal Hessian of slacks") and
  `z_usr` (zl/zu, "gradient of slacks"); `cost_common_*_with_slacks_and_scale`
  computes `fun += 0.5 s'(2 z + Z .* s) s`, `grad_s = z + Z .* s`.
- HPIPM `d_ocp_qp`: `Z` is a `blasfeo_dvec` (diagonal only — no weight matrix
  even in HPIPM dense), `rqz` carries the slack gradient.
- Standard acados soft constraints use `Z = w`, `z = 0`; the linear field
  exists in the interface for generality.
- The complementarity RHS `m` stays 0 for slack sides (our "v1 m = 0" is
  correct): the quadratic weight enters through the QP Hessian, the linear
  weight only through the QP gradient.

Our current state: quadratic weight `w` implemented everywhere (QP Hessian
slack diagonal + NLP cost `0.5 w s^2` + funnel weights); **linear weight
`r_s` missing** (QP slack gradient hard-wired to 0, NLP cost lacks `r s`).
No weight-matrix support needed (not in HPIPM dense either).

## Design (settled with the user)

- **Unified struct** `SoftPenalty<Scalar, N> { quad, lin }` — single source
  of truth; replaces the bare penalty vectors.
- **Per-row** granularity: one `(quad, lin)` pair per constraint row, applied
  identically to its lo-side and hi-side slacks (mirrors today's per-row
  `soft_penalty`; acados' per-side split is almost always symmetric).

### `problem.hpp`

- New `SoftPenalty<Scalar, N>`: two `Eigen::Matrix<Scalar, N, 1>` members
  `quad` (Hessian weight, `0.5 w s^2`) and `lin` (gradient weight, `r s`,
  default 0); default ctor zeroes both; 1-arg (quad only) and 2-arg ctors.
- `BoxSpec`: `soft_penalty` (vector) -> `SoftPenalty<Scalar, Dim> penalty`.
- Type aliases become structs (method signatures keep their names):
  `ineq_pen_t = SoftPenalty<Scalar, ng_soft>`, `eq_pen_t`,
  `ineq_term_pen_t`, `eq_term_pen_t` likewise.
- Contract block + field comments updated.

### `sqp.hpp`

- `fill_slack_diag` / `fill_slack_diag_term`: additionally take the stage
  `grad`; Hessian diagonal from `.quad`, slack grad entries from `.lin`
  (same offset formulas).
- `slack_pair` / `slack_penalty` (NLP cost for merit/backtracking):
  `0.5 w s^2` -> `0.5 w s^2 + r s` per slack side.
- All `spec.soft_penalty(...)` / `spec.bounds.soft_penalty(...)` /
  `pen(i)` reads -> `.penalty.quad(...)` / `pen.quad(i)` etc.

### `globalize.hpp` (funnel)

- 5 soft-penalty reads -> `.penalty.quad` / `.quad` (the funnel uses the
  quadratic weight only).

### Migration (breaking, single commit)

- Examples: `double_integrator`, `mass_spring_ref` (BoxSpec fields);
  `hessmode_5a` (`HessModeProb` penalty methods), `assemble_2f`
  (QuadTest + BoxSpec sites), `merit_2d`, `driver_2g`, `apply_step_2b`,
  `qp_dim`/`suite_1i` (Dims only, no change), integrator tests
  (`continuous_problem_4c`, `cstr_4i`, `sqp_continuous_4e`).
- Generated adapters (`masses_chain`, `pendulum_on_cart`, `unicycle`,
  `p5probe`, `p5probe_b`) regenerated from the updated
  `tools/acados2ocp_pp.py` (BoxSpec sites + the 4 penalty methods emit
  `{ quad, 0 }` structs).

### `qp.hpp` doc

- "grad slack part = 0" -> "grad slack part = linear slack weight
  (0 unless the problem supplies one)".

## Tests

- Update all existing soft-penalty sites to the struct API.
- New value-level checks (hessmode_5a):
  - box row with `penalty.lin != 0`: QP slack grad == lin, slack Hessian
    diag == quad (first + path stage and terminal);
  - NLP `slack_penalty` includes the `r s` term (via `compute_cost`);
  - FD cross-check of the slack objective (quadratic + linear).
- Regression gate: clean `-Werror` build, all 20 test binaries, 3x
  `compare.sh` vs baselines (defaults unchanged: lin = 0 everywhere).

## Follow-up

Strike item 5 in `docs/CLEAN_UP_TODO.md`; move this plan to `finished/`.

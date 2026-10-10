# Item 4 — Per-source Hessian switches (acados `exact_hess_dyn` / `exact_hess_constr`)

TODO item 4: "add an explicit switch to use GAUSS_NEWTON or no dynamics hessian
at all, if none is implemented."

## Semantics (settled in discussion)

This solver is a **dense collocation** SQP: states are decision variables,
dynamics are linearized equality constraints `BA z_k = b`, and the QP Hessian
is block-diagonal per stage. In that formulation the dynamics' second-order
information enters the QP Hessian through exactly one term, the
multiplier-weighted dynamics Hessian `lambda_k^T D2f(x_k, u_k)` (the `M_dyn`
HVP). There are therefore exactly two dynamics-Hessian models, matching
acados `hessian_approx`:

| mode | stage Hessian | acados |
|---|---|---|
| EXACT | `H_cost + lambda^T D2f` (+ constraint Hessians) | `EXACT` |
| GAUSS_NEWTON | `H_cost` (dynamics enter only via linearized `BA`) | `GAUSS_NEWTON` |

The literature `J^T H_cost J` term belongs to **reduced** (state-eliminated /
shooting) formulations; it is the Schur complement of the dense KKT system
and must **not** be added to a dense stage Hessian block.

## Change

The compile-time per-source switches already exist and are kept as-is:
`Dims::has_dynamics_hess` / `Dims::has_constr_hess` (capability flags).

The single runtime gate `SqpOptions::compute_hess` (which demoted **both**
sources at once) is split into two independent runtime flags, mirroring
acados' `exact_hess_dyn` / `exact_hess_constr`:

```cpp
bool compute_hess_dyn = true;    // include lambda^T D2f term (needs has_dynamics_hess)
bool compute_hess_constr = true; // include constraint HVP terms (needs has_constr_hess)
```

Effective term inclusion:
- dynamics Hessian term  <=> `has_dynamics_hess && compute_hess_dyn`
- constraint Hessian term <=> `has_constr_hess && compute_hess_constr`

LM damping (`add_lm_term`) is applied when **either** runtime gate is on
(preserves current default behavior; with both off the QP Hessian is the pure
cost Hessian + slack diagonal, which is PSD and needs no damping).

### Gating sites in `include/ocp/solvers/acados/sqp.hpp`

- `SqpOptions::compute_hess` -> `compute_hess_dyn` + `compute_hess_constr`
  (+ doc comment, ~line 728).
- `add_lm_term` gate (~line 936):
  `!opts_.compute_hess` -> `(!opts_.compute_hess_dyn && !opts_.compute_hess_constr)`.
- `assemble_first` (~1904/1924/1942/1971):
  - dynamics `dynamics_value_jac_hess` call: `compute_hess` -> `compute_hess_dyn`
  - stage ineq/eq `*_value_jac_hess` calls: -> `compute_hess_constr`
  - `add_permuted_hess` block: split into a dyn-gated add of `M_dyn` and a
    constr-gated add of `M_ineq`/`M_eq`.
- `assemble_path` (~2049/2069/2087/2114): same split.
- `assemble_term` (~2183/2198/2220): all three are constraint terms ->
  `compute_hess_constr`.

### Contract docs

- `problem.hpp` contract block (~377-388): note the two independent runtime
  gates in `SqpOptions`.
- `sqp.hpp` `SqpOptions` / `add_lm_term` / `SqpSolver` doc comments.

## Tests

- `tests/sqp/options_2e.cpp`: default-check both new flags.
- `tests/sqp/assemble_2f.cpp`: LM no-op test sets **both** flags false.
- `tests/sqp/hessmode_5a.cpp`:
  - existing check 3 (gate off -> GN) now sets both flags false;
  - new check: `compute_hess_dyn=false, compute_hess_constr=true` ->
    stage hess `hux = 0.5 + 0.4 = 0.9`, terminal hess `3.0` (dyn term dropped,
    constr terms kept);
  - new check: `compute_hess_dyn=true, compute_hess_constr=false` ->
    stage hess `hux = 0.5 - 0.3 = 0.2`, terminal hess `8.0`.

## Regression gate

Build with `-Werror`; run all 19 test binaries + 3 `compare.sh`
(masses_chain, pendulum_on_cart, unicycle) vs baselines. Defaults are
unchanged (both flags true), so no example behavior should move.

## Follow-up

Strike item 4 in `docs/CLEAN_UP_TODO.md`; move this plan to `finished/`.

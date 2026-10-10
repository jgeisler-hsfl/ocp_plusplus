# Rename sqp.hpp -> acados.hpp, SqpSolver -> AcadosSolver

CLEAN_UP item: the SQP driver lives in `include/ocp/solvers/acados/sqp.hpp`
but other SQP-based solvers may follow, so the `Sqp*` namespace should stay
reserved for generic SQP concepts.

## Scope decision (confirmed with user)

Rename only the acados-specific driver and its directly coupled types:

| old | new | reason |
|-----|-----|--------|
| file `sqp.hpp` | `acados.hpp` | the driver is the acados port |
| `SqpSolver` | `AcadosSolver` | the driver class (item wording) |
| `SqpOptions` | `AcadosOptions` | driver-coupled options struct |
| `SqpStatistics` | `AcadosStatistics` | per-solve statistics of the driver |
| `SqpIteration` | (kept) | generic SQP iteration-row concept |
| `NlpResiduals` | (kept) | already prefix-free, generic |
| `print_sqp_iteration` | (kept) | prints a generic `SqpIteration` row |
| `SqpSlacks` (globalize.hpp) | (kept) | generic SQP slack trajectory |

## Public API review

Every public `AcadosSolver` method is exercised by unit tests and
intentionally documented as exposed for testability/inspection: `solve`,
`resize`, `options`, `statistics`, `last_qp`, `last_qp_sol`, `assemble_qp`,
`add_lm_term`, `update_adaptive_lm_mu`, `adaptive_lm_mu(_bar)`,
`compute_cost`, `soc_count`, `check_termination`. All stay public.

One change: `make_scaler_opts` (static, options -> `QpScaler::Opts` glue)
has zero external callers (constructor-internal only) and is moved to the
private section.

## Migration surface

- `include/ocp/solvers/acados/acados.hpp` — renames + header comments +
  `make_scaler_opts` -> private.
- `include/ocp/solvers/acados/regularize.hpp`,
  `include/ocp/problem.hpp` — comment-only references.
- All tests (`tests/sqp/**`, `tests/sqp_*/`, `tests/integrators/`) and
  hand-written examples — type names + include path.
- `tools/acados2ocp_pp.py` — generated-main template (include line,
  `AcadosOptions`, `AcadosSolver`); the five generated mains
  (p5probe, p5probe_b, unicycle, masses_chain, pendulum_on_cart) updated
  accordingly.
- `docs/plans/finished/**` — historical plans, intentionally not touched.

## Verification

Clean `-Werror` build; all 20 test binaries pass; the three
`examples/*/compare.sh` baselines pass.

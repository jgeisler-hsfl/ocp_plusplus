# TODO

Commit hashes and dates refer to `master`.

## Open items

### B. Problem interface / design ergonomics

- [x] why does ocp::ContinuousProblem need Dims and ODE as template parameters, can it not access these from the integrator? (done — the integrators re-expose their `Dims` and model type as `dims_t` / `model_t`; `ContinuousProblem` is now spelled `ContinuousProblem<Integ>` (plus optional `Derived` / `NH`), with all hand-written call sites, the generator, and regenerated examples updated; plan in `docs/plans/finished/CP_FROM_INTEG_PLAN.md`) (2026-10-28, branch `feature/cp-from-integ`, merged `883f241`)
- [ ] add a linear and a nonlinear least sqquares tracking problem crtp class with gauss newton approximation option

### C. Tooling / generator

- [ ] add FD class and "check wrapper" to quickly verify user supplied jecobians and hessians

### E. Optimization

- [ ] [OPTIMIZATION] ErkWorkspace split by derivative source: most fields are used by exactly one branch of ExplicitRkIntegrator. The AD path (ad_forward, Dual1/Dual2) uses only stack-local dual vectors and none of the workspace; value() only needs K (stage-derivative scratch, shared by all ODE kinds); everything else (x_stage, JX, JU, dx, dK, ddK_dual, dKval_dual, dv_ss, dx_ss, du, and the (x,u,t_k)-keyed forward cache last_x/last_u/last_t/fwd_valid) is used only by the hand-written-ODE paths (forward_jac / thread_jvp / backward_sweep). Design: split the workspace via std::conditional_t on !ode_f_evaluable_at<Ode, Dims, Dual1> (a base holding just K, plus a legacy extension with the rest); legacy-only member functions never instantiate for AD ODEs, so no further gating is needed. Caveat: pure RAM (no runtime or compile-time cost today — the Dual2 codegen overhead is already gone via the ad_hess_prod opt-out); savings scale with JX ~ NS*NumSteps*nx^2 per integrator instance (negligible at the current example sizes, ~2.5 MB at nx=100, NumSteps=8; multiplies with SQP worker instances).

## Done

### Core roadmap (original items 1–17)

- [x] Define problem interface (2026-09-27, `2b788cd`)
- [x] Port selected acados problems to new interface for comparisons (2026-09-27, `eab6096`)
- [x] Write simple test for problem interface using a simple example from the acados repo: test discrete time simulation of model and evaluation of cost and constraint functions, output as csv file (2026-09-27, `eab6096`; ticked in `2b53672`)
- [x] Test acados reimplementation with simple example from acados repo (2026-10-01, `49efcf4` standalone e2e SQP tests + `08b1d4a` acados HPIPM reference diff test)
- [x] reimplement acados style explicit and implicit RK solver/integrator alloc free template class (2026-10-03, phases 4a–4i `93a2973`…`a848729`; ticked in `70eef97`)
- [x] Define a continuous time problem interface using an integrator class to implement the discrete time interface (2026-10-03, `6c2fa1b` phase 4c; ticked in `70eef97`)
- [x] write JSON generator (tools/acados2ocp_pp.py) to write interface definition for new framework from acados_ocp_nlp.json, reusing CasADi generated dynamics, cost and constraint code (2026-10-04, introduced in `445afde`; updates `847bcfa`, `7356fcf`, `e3338c0`)
- [x] implement GAUSS_NEWTON for dynamics part of the cost function. add switch in dims to activate (2026-10-04, merge of branch `dev-gn` in `35122ac`; key commits `6bef82b`, `9963055`, `ccd071b`, `2cc70f3`)
- [x] write project README (2026-10-04, `b7dfaf0`)
- [x] rework problem contract to use nested triplets for value/grad/hess (2026-10-04, `52608f9`; migration in `15f7cfa`, `a99b6b1`, `847bcfa`, `e2e3970`)
- [x] cppduals calculation of grad and hess (2026-10-04, `872a385` vendored duals + `df11b9c` AutoDiffOde; later `f1720c5` AutoDiffProblem mix-in, `4585c76` AutoDiffDae)
- [x] Test more examples from acados repo (2026-10-04, `bfb69e0` ticked; masses_chain/pendulum_on_cart `445afde`, unicycle `8012bbc`, p5probe* `a901190`)
- [x] We should start to write some documentation/how to use, that explains these things, maybe even a tutorial with increasing levels of difficulty. (2026-10-10, tutorial added in `65e7b06`; extended through `bee4b4f` 2026-10-26, ch. 8)
- [x] rework all code documentation to be more precise, informative and not too long (doxygen style), name inputs and outputs. leave the genesis (reference to the implementation phases) and the acados original code only as supplementary info. (2026-10-10, `2aab016`)
- [x] Reimplement acados SQP solver with HPIPM SQ solver (2026-10-12, phases 1–3 `b72f399`…`42f01be`; `42f01be` = 3e, SQP_RTI reference + docs)
- [x] rewrite options and statistics to use a dual access approach: direct via struct/class members (like currently) or by name via a reference map. aim: make values "discoverable" and settable in an interactive environment. (done — the inspect protocol, `docs/plans/finished/OPTION_INSPECT_PLAN.md`; history in `docs/plans/finished/DUAL_ACCESS_PLAN.md` — 2026-10-26, `c8589c7`…`52e8b70`)
- [x] investigate a way to implement a central "discovery"-map where all parts of a problem/solver-object can "register" their run-time mutable values (lists) like options and parameters. all values shall be alloc-free class members somewhere, only the "by-name"-access refs them from a map. every part of the problem has its own local list or lists (maps that ref values). the central registry maps theses lists/maps forming a hierarchie. maybe add a special accessor that takes complete value-"paths" and resolves them through the maps (of course handling nonexistent entries gracefully), like get("solver/hpimp/options/i_max") would return the value i_max that is an option in the hpipm solver. (done — per-type `inspect` with dotted paths, e.g. `qp.options.warm_start`, plus the `ocp::inspect` walkers; the MEX `options`/`opt_dump` commands are the discovery front-end, `docs/plans/finished/OPTION_INSPECT_PLAN.md` — 2026-10-26, `52e8b70`; MEX front-end `7d876fd`, merged in `a847f67` branch `matlab_mex`)

### A. Numerics / solver internals

- [x] why is there not qp_warm_start=1? (answered: acados zeroes the primal `ux` before every QP solve, so HPIPM `warm_start = 1` (primal-only) behaves like `0`; only `{2, 3}` are effective — `docs/plans/finished/SQP_PHASE3_PLAN.md` (3c); see the `RtiOptions::qp_warm_start` comment) (2026-10-02, `c71c2fa`)
- [x] why is lm damping not applied ("a no-op"), when both `compute_hess_dyn` and `compute_hess_constr` are false? (answered: with both gates off the QP Hessian is the pure cost Hessian + slack diagonal, which is PSD and needs no damping — `add_lm_term` doc comment, tutorial 05 §5.5, `docs/plans/finished/HESSIAN_SWITCH_PLAN.md`; covered by the no-op test in `tests/sqp/assemble_2f.cpp`) (2026-10-10, `3f9dc4f`)
- [x] make clear, how Levenberg-Marquardt and its options correspond to the different convexification options in acados (done — tutorial ch. 5: §5.1 model table with the acados `hessian_approx` equivalents, §5.4 runtime gates mirroring `exact_hess_dyn` / `exact_hess_constr`, §5.5 fixed + adaptive LM with acados-matching option names, §5.6 summary) (2026-10-10, `65e7b06`)
- [x] optional fixed size statistics array with iteration number field and ring buffer filling or ignoring new entries, when the buffer is full (which is better?) (done — compile-time `NSlots` FIFO ring via `ocp::RingBuffer`, drop-oldest policy, per-row `iter` stamp; `HpipmOptions::stat_max` removed; `docs/plans/finished/STATS_RING_PLAN.md`) (2026-10-28, branch `feature/stats-ring-buffer`, `e633456`)

### B. Problem interface / design ergonomics

- [x] add a ProblemScaling class for scaling states and inputs (done: `ocp::ScaledProblem<Base>` composition wrapper in `include/ocp/scaling.hpp`; unscaled values are recovered via `scaled.unscale(sol)`; plan in `docs/plans/finished/SCALED_PROBLEM_PLAN.md`) (2026-10-27, `5977f73`, docs `d4d5793`)
- [x] to be examined: how to share parameters between ODE/DAE and cost and constraints (done — duck-typed `params()` hook: a user-defined parameter struct read by the model bodies, optional override on `Problem` with an `EmptyParams` base stub, inspect/MEX by-name access; plan in `docs/plans/finished/DUCK_TYPED_PARAMS_PLAN.md`, pattern in tutorial ch. 02/06, MEX `param_*` commands in ch. 08) (2026-10-28, branch `feature/duck-typed-params`, merged `03c7280`)
- [x] add receding horizon shifting to Solution, what about the solver-internal slacks? (done — `ocp::Solution::shift(x0_new, ShiftFill)` in `problem.hpp` plus `AcadosSolver::shift` / `reset_warm_start` / `slacks()` in `acados.hpp`; the driver shifts the internal slacks one stage (first <- stage-1 path, path up one, new last path stage zeroed, terminal unchanged) while the persistent QP duals stay in place as a second-order warm start; tutorial 07 §7.3 updated, `tests/problem/solution_shift_unit.cpp` + `tests/sqp/rh_shift.cpp`) (2026-10-28, branch `feature/receding-horizon-shift`, merged `57c13ef`)
- [x] do we really need the NS number of stage in the integrators, is that not already encoded in the tags? (yes, redundant — dropped: tags now carry `ns`, `ButcherTableau<Dims, Tag>`, `ExplicitRkIntegrator<Dims, Ode, Tag, NumSteps>`, `ImplicitRkIntegrator<Dims, Dae, Tag, NumSteps>`; see docs/plans/finished/DROP_NS_PLAN.md) (2026-10-28, branch `feature/drop-integrator-ns`, `ebd95ef`)
- [x] make it possible to set the integrator step size per stage from the problem (done: `h` moved from the integrator constructor to every entry point as a per-call argument; `ocp::ContinuousProblem` gained the CRTP hook `StepSpec get_step_size(int k) const` with the `ConstantStepSize` fallback so all existing problems keep compiling; the grid schedule idiomatically lives in the duck-typed `params()` struct, see tutorial ch. 06 §6.10; `tests/integrators/step_grid_4m.cpp`, plan in `docs/plans/finished/PER_STAGE_STEP_SIZE_PLAN.md`) (2026-10-28, branch `feature/per-stage-step-size`, `7d549bf`)

### C. Tooling / generator

- [x] does ocp::Solution have a to_csv? if not, add it and add a simple mecanism to plot it png or preferably html (probably best use python and plotly, leave it to the user to install) (done — `ocp::Solution::to_csv` in `problem.hpp` with optional `Dims::state_names` / `control_names` column headers via `Problem::state_name` / `control_name`, `tools/plot_solution.py` for HTML/PNG plotting via plotly (user-installed); `tests/problem/solution_unit.cpp`, tutorial 02 + 03 §3.9) (2026-10-28, branch `feature/solution-names-csv`, merged `fd6ad6f`)
- [x] add support for more than one parameter in acados2ocp_pp.py (done — `np >= 0` accepted; the generator emits a shared `{name}Params` struct with a fixed-size `p` vector leaf exposed via the duck-typed `params()` hook, shared by dynamics and cost/constraints; the old `np == 1` scalar path was unified onto it; end-to-end fixture in `tests/params/np2/` + `params_np2` test, plan in `docs/plans/finished/MULTI_PARAM_PLAN.md`) (2026-10-28, branch `feature/multi-param-generator`, merged `e23d7d3`)

### D. Docs / tutorial / README / license

- [x] does the implicit integrator have options to limit the number of newton iterations and the frequency of reevaluated jacobians? (answered — `ImplicitRkIntegrator(dae, h, newton_max = 3, newton_tol = 0.0)`; the DAE Jacobians are re-evaluated at every Newton iteration (no frequency knob) — `solve_newton` in `include/ocp/integrators/rk_implicit.hpp`) (2026-10-03, `d31925c`)
- [x] check about this "hess * v  =  sum_i lam_i * D^2 f_i(x, u) * v" in 05-second-order-models.md. i believe the "v" is no longer there (checked — 05 §5.3 now reads `hess = sum_i lam_i * D^2 f_i(x, u)`) (2026-10-24, `6e7b260`)
- [x] definition of IrkBase is missing in the example in 06-continuous-time-and-integrators.md (fixed — `using IrkBase = ocp::ContinuousProblem<DuffingDims, Dae, IrkGl4>;` in 06 §6.4) (2026-10-24, `6e7b260`)
- [x] in 06-continuous-time-and-integrators.md, 6.5 Automatic differentiation, add a link to an explanation of duals and the original repo (done — 06 §6.5 links the Wikipedia dual-numbers AD article and names the vendored multidual header) (2026-10-24, `6e7b260`)
- [x] add license and notice to it in the README (done — README "License" section linking LICENSE and NOTICE) (2026-10-24, `3f5192e` + `3632308`)
- [x] add anotice to AGENTS.md to check/update the ARCHITECTURE, README and tutorial on ever merge (done — AGENTS.md "Documentation": "On every merge, check and update `docs/ARCHITECTURE.md`, `README.md` and the tutorial in `docs/tutorial/`") (2026-10-24, `d8f2fdd`)
- [x] add ocp::Status conversion to string (done — `ocp::to_string(Status)` in `problem.hpp`, tested in `tests/problem/status_unit.cpp`) (2026-10-27, `02c95f9`)
- [x] in 06-continuous-time-and-integrators.md, add a usage example for AutoDiffDae (done — 06 §6.6 "DAE derivatives via AutoDiffDae" with a DuffingDae snippet, WithHess explanation, and pointer to `dae_autodiff_4j.cpp`) (2026-10-27, `8514285`)
- [x] is the funnel globalization adequately addressed in the tutorial? (done — 07 §7.6 now documents the `f`/`h`/`b`/`p` accept semantics, a concrete `Funnel<P, NH>` instantiation, and all 9 funnel parameters; funnel params are settable via the new `AcadosSolver::globalizer()` accessor, tested in `tests/sqp/funnel_3i.cpp`) (2026-10-27, `58da6b1`)

### Sweep items

- [x] dont pass options to the solver. let the solver have a default set options member and expose it to the use via the solver object. (done — `AcadosSolver(AcadosOptions opts = {})` / `RtiSolver(RtiOptions opts = {})` hold a default-constructed options member; `solve()` takes no options; the live options are exposed via the inspect protocol (`options.*`) and `options()`; checked off in the 2026-10-27 sweep, `b8c8c48`; code landed 2026-10-10, `05ff146`)

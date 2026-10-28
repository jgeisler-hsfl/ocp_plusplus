- [x] Define problem interface
- [x] Port selected acados problems to new interface for comparisons
- [x] Write simple test for problem interface using a simple example from the acados repo: test discrete time simulation of model and evaluation of cost and constraint functions, output as csv file
- [x] Reimplement acados SQP solver with HPIPM SQ solver
- [x] reimplement acados style explicit and implicit RK solver/integrator alloc free template class
- [x] Define a continuous time problem interface using an integrator class to implement the discrete time interface
- [x] write JSON generator (tools/acados2ocp_pp.py) to write interface definition for new framework from acados_ocp_nlp.json, reusing CasADi generated dynamics, cost and constraint code
- [x] implement GAUSS_NEWTON for dynamics part of the cost function. add switch in dims to activate
- [x] write project README
- [x] rework problem contract to use nested triplets for value/grad/hess
- [x] cppduals calculation of grad and hess
- [x] rewrite options and statistics to use a dual access approach: direct via struct/class members (like currently) or by name via a reference map. aim: make values "discoverable" and settable in an interactive environment. (done — the inspect protocol, `docs/plans/finished/OPTION_INSPECT_PLAN.md`; history in `docs/plans/finished/DUAL_ACCESS_PLAN.md`)
- [x] investigate a way to implement a central "discovery"-map where all parts of a problem/solver-object can "register" their run-time mutable values (lists) like options and parameters. all values shall be alloc-free class members somewhere, only the "by-name"-access refs them from a map. every part of the problem has its own local list or lists (maps that ref values). the central registry maps theses lists/maps forming a hierarchie. maybe add a special accessor that takes complete value-"paths" and resolves them through the maps (of course handling nonexistent entries gracefully), like get("solver/hpimp/options/i_max") would return the value i_max that is an option in the hpipm solver. (done — per-type `inspect` with dotted paths, e.g. `qp.options.warm_start`, plus the `ocp::inspect` walkers; the MEX `options`/`opt_dump` commands are the discovery front-end, `docs/plans/finished/OPTION_INSPECT_PLAN.md`)
- [x] Test acados reimplementation with simple example from acados repo
- [x] Test more examples from acados repo
- [x] We should start to write some documentation/how to use, that explains these things, maybe even a tutorial with increasing levels of difficulty.
- [x] rework all code documentation to be more precise, informative and not too long (doxygen style), name inputs and outputs. leave the genesis (reference to the implementation phases) and the acados original code only as supplementary info.

## Open items

### A. Numerics / solver internals

- [ ] optional fixed size statistics array with iteration number field and ring buffer filling or ignoring new entries, when the buffer is full (which is better?)
- [x] why is there not qp_warm_start=1? (answered: acados zeroes the primal `ux` before every QP solve, so HPIPM `warm_start = 1` (primal-only) behaves like `0`; only `{2, 3}` are effective — `docs/plans/finished/SQP_PHASE3_PLAN.md` (3c); see the `RtiOptions::qp_warm_start` comment)
- [x] make clear, how Levenberg-Marquardt and its options correspond to the different convexification options in acados (done — tutorial ch. 5: §5.1 model table with the acados `hessian_approx` equivalents, §5.4 runtime gates mirroring `exact_hess_dyn` / `exact_hess_constr`, §5.5 fixed + adaptive LM with acados-matching option names, §5.6 summary)
- [x] why is lm damping not applied ("a no-op"), when both `compute_hess_dyn` and `compute_hess_constr` are false? (answered: with both gates off the QP Hessian is the pure cost Hessian + slack diagonal, which is PSD and needs no damping — `add_lm_term` doc comment, tutorial 05 §5.5, `docs/plans/finished/HESSIAN_SWITCH_PLAN.md`; covered by the no-op test in `tests/sqp/assemble_2f.cpp`)

### B. Problem interface / design ergonomics

- [ ] to be examined: how to share parameters between ODE/DAE and cost and constraints
- [ ] do we really need the NS number of stage in the integrators, is that not already encoded in the tags?
- [ ] why does ocp::ContinuousProblem need Dims and ODE as template parameters, can it not access these from the integrator?
- [ ] make it possible to set the integrator step size per stage from the problem
- [x] add a ProblemScaling class for scaling states and inputs (done: `ocp::ScaledProblem<Base>` composition wrapper in `include/ocp/scaling.hpp`; unscaled values are recovered via `scaled.unscale(sol)`; plan in `docs/plans/finished/SCALED_PROBLEM_PLAN.md`)
- [ ] add a linear and a nonlinear least sqquares tracking problem crtp class with gauss newton approximation option
- [ ] add receding horizon shifting to Solution, what about the solver-internal slacks?

### C. Tooling / generator

- [ ] add FD class and "check wrapper" to quickly verify user supplied jecobians and hessians
- [x] does ocp::Solution have a to_csv? if not, add it and add a simple mecanism to plot it png or preferably html (probably best use python and plotly, leave it to the user to install) (done — `ocp::Solution::to_csv` in `problem.hpp` with optional `Dims::state_names` / `control_names` column headers via `Problem::state_name` / `control_name`, `tools/plot_solution.py` for HTML/PNG plotting via plotly (user-installed); `tests/problem/solution_unit.cpp`, tutorial 02 + 03 §3.9)
- [ ] add support for more than one parameter in acados2ocp_pp.py

### D. Docs / tutorial / README / license

- [x] add ocp::Status conversion to string (done — `ocp::to_string(Status)` in `problem.hpp`, tested in `tests/problem/status_unit.cpp`)
- [x] in 06-continuous-time-and-integrators.md, add a usage example for AutoDiffDae (done — 06 §6.6 "DAE derivatives via AutoDiffDae" with a DuffingDae snippet, WithHess explanation, and pointer to `dae_autodiff_4j.cpp`)
- [x] is the funnel globalization adequately addressed in the tutorial? (done — 07 §7.6 now documents the `f`/`h`/`b`/`p` accept semantics, a concrete `Funnel<P, NH>` instantiation, and all 9 funnel parameters; funnel params are settable via the new `AcadosSolver::globalizer()` accessor, tested in `tests/sqp/funnel_3i.cpp`)
- [x] check about this "hess * v  =  sum_i lam_i * D^2 f_i(x, u) * v" in 05-second-order-models.md. i believe the "v" is no longer there (checked — 05 §5.3 now reads `hess = sum_i lam_i * D^2 f_i(x, u)`)
- [x] definition of IrkBase is missing in the example in 06-continuous-time-and-integrators.md (fixed — `using IrkBase = ocp::ContinuousProblem<DuffingDims, Dae, IrkGl4>;` in 06 §6.4)
- [x] in 06-continuous-time-and-integrators.md, 6.5 Automatic differentiation, add a link to an explanation of duals and the original repo (done — 06 §6.5 links the Wikipedia dual-numbers AD article and names the vendored multidual header)
- [x] add license and notice to it in the README (done — README "License" section linking LICENSE and NOTICE)
- [x] does the implicit integrator have options to limit the number of newton iterations and the frequency of reevaluated jacobians? (answered — `ImplicitRkIntegrator(dae, h, newton_max = 3, newton_tol = 0.0)`; the DAE Jacobians are re-evaluated at every Newton iteration (no frequency knob) — `solve_newton` in `include/ocp/integrators/rk_implicit.hpp`)
- [x] add anotice to AGENTS.md to check/update the ARCHITECTURE, README and tutorial on ever merge (done — AGENTS.md "Documentation": "On every merge, check and update `docs/ARCHITECTURE.md`, `README.md` and the tutorial in `docs/tutorial/`")

### E. Optimization

- [ ] [OPTIMIZATION] ErkWorkspace split by derivative source: most fields are used by exactly one branch of ExplicitRkIntegrator. The AD path (ad_forward, Dual1/Dual2) uses only stack-local dual vectors and none of the workspace; value() only needs K (stage-derivative scratch, shared by all ODE kinds); everything else (x_stage, JX, JU, dx, dK, ddK_dual, dKval_dual, dv_ss, dx_ss, du, and the (x,u,t_k)-keyed forward cache last_x/last_u/last_t/fwd_valid) is used only by the hand-written-ODE paths (forward_jac / thread_jvp / backward_sweep). Design: split the workspace via std::conditional_t on !ode_f_evaluable_at<Ode, Dims, Dual1> (a base holding just K, plus a legacy extension with the rest); legacy-only member functions never instantiate for AD ODEs, so no further gating is needed. Caveat: pure RAM (no runtime or compile-time cost today — the Dual2 codegen overhead is already gone via the ad_hess_prod opt-out); savings scale with JX ~ NS*NumSteps*nx^2 per integrator instance (negligible at the current example sizes, ~2.5 MB at nx=100, NumSteps=8; multiplies with SQP worker instances).

### Done (this sweep, 2026-10-27)

- [x] dont pass options to the solver. let the solver have a default set options member and expose it to the use via the solver object. (done — `AcadosSolver(AcadosOptions opts = {})` / `RtiSolver(RtiOptions opts = {})` hold a default-constructed options member; `solve()` takes no options; the live options are exposed via the inspect protocol (`options.*`) and `options()`)

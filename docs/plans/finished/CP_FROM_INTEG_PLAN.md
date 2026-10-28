# Plan: derive ContinuousProblem Dims/Ode from the integrator

TODO §B item: *"why does ocp::ContinuousProblem need Dims and ODE as template
parameters, can it not access these from the integrator?"*

## Answer

Yes. The integrator is already templated on both:

- `ExplicitRkIntegrator<Dims, Ode, Tag, NumSteps>`
- `ImplicitRkIntegrator<Dims, Dae, Tag, NumSteps>`

but neither re-exposes them, so `ContinuousProblem<Dims, Ode, Integ, ...>`
forces every call site to spell Dims and the model twice. Same redundancy
pattern as the `NS` argument dropped in `ebd95ef` (the scheme tag already
carries `ns`). Fix: expose `using dims_t` / `using model_t` on both
integrators and let the adapter derive them:

```cpp
template <class Integ, class Derived = ConstantStepSize,
          int NH = Eigen::Dynamic>
class ContinuousProblem : public Problem<typename Integ::dims_t>
```

`NH` is currently unused in the adapter body (template declaration only);
it stays to avoid a broader signature change than this item requires.

## Scope

- `include/ocp/integrators/rk_explicit.hpp`, `rk_implicit.hpp` — add the two
  public aliases each (with Doxygen).
- `include/ocp/integrators/continuous_problem.hpp` — new signature, `Dims` /
  `Ode` aliases derived from `Integ`; header-doc update.
- Hand-written call sites (spelling change only, no behaviour change):
  - `examples/masses_chain/masses_chain.hpp`
  - `examples/tutorial/continuous.cpp`
  - `tests/integrators/step_grid_4m.cpp` (5 spellings, 2 with `Derived`)
  - `tests/integrators/sqp_continuous_4e.cpp` (2)
  - `tests/integrators/problem_autodiff_4l.cpp` (4)
  - `tests/integrators/dae_autodiff_4j.cpp` (3)
  - `tests/integrators/cstr_4i.cpp` (1)
  - `tests/integrators/continuous_problem_4c.cpp` (1)
  - `include/ocp/autodiff_problem.hpp` (comment example)
- Generator: `tools/acados2ocp_pp.py` `class_block` emits
  `ocp::ContinuousProblem<{name}Integ>`; regenerate the 5 examples
  (masses_chain, pendulum_on_cart, unicycle, p5probe_erk, p5probe_irk) and
  the `tests/params/np2` fixture with the CMakeLists-documented commands.
  Regenerated diffs must be limited to the base-class spelling lines.
- Docs: `docs/tutorial/06-continuous-time-and-integrators.md` (7 spellings),
  `docs/plans/TODO.md` §B item → `[x]` with date/commit annotation.

## Constraints

- Warning-free build under `-Wall -Wextra -Werror`; full test suite passes
  (all 31 targets, incl. `sqp_masses_chain`, `sqp_masses_chain_gen_equiv`,
  `params_np2`).
- One logical change per commit; plan archival + AGENTS.md unlink happen with
  the final commit.

## Blocks

1. [x] Integrator aliases: `dims_t` / `model_t` on `ExplicitRkIntegrator`
       and `ImplicitRkIntegrator` (public section, Doxygen one-liners).
2. [x] Slim `ContinuousProblem`: `template <class Integ, class Derived =
       ConstantStepSize, int NH = Eigen::Dynamic>`, derive
       `Dims = Integ::dims_t`, `Ode = Integ::model_t`; update the file-header
       doc.
3. [x] Hand-written call sites: 8 files listed above.
4. [x] Generator `class_block` + regenerate 5 examples + NP2 fixture;
       verified diffs are spelling-only (6 files, base-class lines only).
5. [x] Docs: tutorial 06 (7 spellings + the §6.10 "4th position" note),
       TODO §B done-item. Plan archive to `docs/plans/finished/` and
       AGENTS.md item 4 back to "(none)" happen with the final commit.

   Findings: `NH` is unused in the `ContinuousProblem` body (template
   declaration only); kept as-is to keep the change minimal. No
   SFINAE/wrapper code (MEX, ScaledProblem, has_params) references the
   adapter's template parameters, so only call-site spellings change.

## Commit split

- C1 `410f8ec` integrator: expose dims_t and model_t aliases
- C2 `1bba0d1` problem: derive ContinuousProblem dims from integrator
- C3 `c7de1c5` tools: emit slim ContinuousProblem spelling
- C4 docs: document integrator-derived ContinuousProblem

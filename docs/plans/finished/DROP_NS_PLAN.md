# Drop the redundant `NS` template argument from the RK integrators

## 1. Motivation

The scheme tag already encodes the number of stages:

| Tag | NS |
|---|---|
| `K1Tag` | 1 |
| `K2Tag` | 2 |
| `K3Tag` | 3 |
| `K4Tag` | 4 |
| `RadauIia2Tag` / `RadauIia3Tag` / `RadauIia4Tag` | 2 / 3 / 4 |
| `GaussLegendre2Tag` / `GaussLegendre3Tag` / `GaussLegendre4Tag` | 2 / 3 / 4 |

`ButcherTableau` only has partial specializations for exactly these
`(NS, Tag)` pairs; any other combination fails the primary-template
`static_assert`. `NS` therefore carries no information the tag does not
already carry — it is repeated at ~60 call sites and in three template
signatures (`ButcherTableau<Dims, NS, Tag>`,
`ExplicitRkIntegrator<Dims, Ode, NS, Tag, NumSteps>`,
`ImplicitRkIntegrator<Dims, Dae, NS, Tag, NumSteps>`).

Goal: derive the stage count from the tag and drop `NS` from all public
signatures. `NumSteps` is genuinely independent (sub-intervals per OCP
step) and stays.

Nothing of value is lost: the only current "safety" is the
`static_assert` on mismatched `(NS, Tag)` pairs, i.e. the ability to
*spell a wrong NS* — removing the argument removes the error class
outright.

## 2. Design

- `include/ocp/integrators/butcher.hpp`
  - Each tag gains `static constexpr int ns = N;` (tags stay pure
    marker structs + this one constant).
  - `ButcherTableau<Dims, NS, Tag>` → `ButcherTableau<Dims, Tag>`
    (`Dims` kept for interface symmetry with the integrators, as
    documented in the header; `NS` dropped). The 10 partial
    specializations become keyed on `Tag` alone; the primary-template
    `static_assert` message is updated.
- `include/ocp/integrators/rk_explicit.hpp`
  - `template <class Dims, class Ode, class Tag, int NumSteps = 1>`.
  - Public `static constexpr int NS = Tag::ns;` (so users/tests can
    still query the stage count, e.g. for diagnostics).
  - `detail::ErkWorkspace<P, NS, NumSteps>` unchanged (int parameter is
    fine internally; the integrator passes its derived `NS`).
  - Header doc comments updated (signature, `NS` description).
- `include/ocp/integrators/rk_implicit.hpp`
  - Same: `template <class Dims, class Dae, class Tag, int NumSteps = 1>`,
    public `static constexpr int NS = Tag::ns;`,
    `detail::IrkWorkspace` unchanged, header docs updated.
- Call sites: drop the `NS` argument. Test helpers templated on
  `<int NS, class Tag>` become `<class Tag>` (loops use the derived
  count via `ButcherTableau<...>::...` or `Tag::ns`).
- `tools/acados2ocp_pp.py`: the emitted integrator alias loses the
  `{nst},` argument (line ~1287). `spec["num_stages"]` stays (still used
  for validation, the generated header comment, and the ERK/IRK tag
  choice).
- Regenerate the 5 checked-in generated headers with the updated tool:
  `examples/{pendulum_on_cart,masses_chain,unicycle,p5probe_erk,
  p5probe_irk}/generated/*Gen.hpp`. Expected diff: alias line only
  (the generator comment line `(N stages x M sub-steps)` stays — it is
  prose, not code).
- Docs: `docs/tutorial/06-continuous-time-and-integrators.md`
  (signature lines ~67/74, examples ~118/125/220). `docs/ARCHITECTURE.md`
  and `README.md` were checked: no integrator signatures referenced.
  `docs/plans/TODO.md` has no item referencing the NS argument (re-check
  before committing).

## 3. Execution blocks

- [x] B1 `butcher.hpp`: `ns` on the 10 tags; `ButcherTableau<Dims, Tag>`;
      updated comments/assert message.
- [x] B2 `rk_explicit.hpp`: drop `NS` parameter, add derived public
      constant, update header docs.
- [x] B3 `rk_implicit.hpp`: same.
- [x] B4 tests: `tests/integrators/{integrators_4a, rk_explicit_4b,
      rk_implicit_4d, rk_multistep_4g, time_varying_4h, cstr_4i,
      dae_autodiff_4j, problem_autodiff_4l, continuous_problem_4c,
      sqp_continuous_4e}.cpp` — drop NS at every call site; convert
      `template <int NS, class Tag>` helpers (bodies use a local
      `const int NS = Tag::ns;`).
- [x] B5 hand-written examples: `examples/masses_chain/masses_chain.hpp`,
      `examples/tutorial/continuous.cpp`.
- [x] B6 generator: update alias emission in `tools/acados2ocp_pp.py`,
      regenerate the 5 generated headers, verify the diff is the alias
      line + the pre-existing np-comment drift only.
- [x] B7 docs: tutorial 06 signatures/examples.
- [x] B8 build + test: full warning-free build; ran `integrators_unit`
      (all phase-4 checks), `double_integrator`, `tutorial_continuous`,
      `masses_chain`, `masseschaingen_gen`, `pendulum_on_cart`,
      `unicycle`, `sqp_p5probe_erk`, `sqp_p5probe_irk`,
      `sqp_masses_chain_gen_equiv`, `sqp_continuous` — all pass.

## 4. Verification

- `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`
  stays warning-free under `-Wall -Wextra -Werror`.
- `./build/integrators_unit` and `./build/double_integrator` exit 0.
- `git diff` of the regenerated headers shows only the removed `NS`
  argument (no other churn).
- No remaining references to the old 4/5-argument integrator
  spellings: `grep -rn "Integrator<.*,.*, *[1-4],"` over include/,
  tests/, examples/, tools/ (finished plans in
  `docs/plans/finished/` are historical and are *not* rewritten).

## 5. Findings / notes

- `tests/sqp_masses_chain/ref_masses_chain.h` is a reference snapshot of
  the *problem* interface only — it contains no integrator alias, no
  change needed.
- `sqp_continuous_4e.cpp`, `continuous_problem_4c.cpp` and the other
  tests use the full `<Dims, Model, NS, Tag>` spelling; all are plain
  call-site edits.
- The generated-header comment line
  `// integrator: IRK GaussLegendre4Tag (4 stages x 2 sub-steps)` is
  emitted from `spec["num_stages"]` and stays as-is.
- Regeneration also re-emits an `np` comment that was added to the
  generator in `b17c82a` but never regenerated into the checked-in
  headers (pre-existing drift). The regenerated headers now match the
  current tool exactly: alias line + that comment, no other churn.

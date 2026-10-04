# cppduals (vendored)

Vendored copy of the **pure, dependency-free** `duals/dual` header from the
[cppduals](https://gitlab.com/tesch1/cppduals) project.

- **License:** Mozilla Public License 2.0 (see `LICENSE.txt`, verbatim from upstream).
- **Pinned commit:** `7b2a6d149a85c74019d3ef804cd40953a2350119` (master).

## Why only `duals/dual` (not the full repo)

The project is built on **Eigen 3.x**. Upstream master's `duals/dual_eigen`
`#error`s on Eigen 3 (it targets Eigen 5, whose GEMM/SIMD packet path vectorizes
large dual-valued GEMMs by decomposing them into a few native-precision GEMMs).
None of that is needed here: the only dual-valued products in this codebase are
the small ODE `f` expressions and the integrator's `nx x nx` stage Jacobians,
while all large dense linear algebra (QP / SQP / KKT / LU) is plain `double`.

So we vendor only the self-contained `duals/dual` header (the multivariate
`dual<T, N>` scalar: `variable(v,i)`, `dpart(i)`, `set_dpart(v,i)`, nesting
`dual<dual<double,N>,1>` for 2nd order) and provide our own minimal Eigen-3 glue
in `include/ocp/integrators/duals_eigen3.hpp`.

## Usage

Include via the project include path (`-I thirdparty/cppduals` is added to the
`integrators_unit` target):

```cpp
#include <duals/dual>   // duals::dual<T, N>
```

// ocp/integrators/duals_eigen3.hpp
//
// Minimal Eigen-3.x glue for the vendored multivariate dual scalar
// `duals::dual<T, N>` (thirdparty/cppduals/duals/dual).
//
// Eigen 3.x ships no traits for `duals::dual`, so without this header any
// Eigen expression built from a dual scalar fails either in `NumTraits` (the
// primary template reads `std::numeric_limits<dual>` defaults) or in scalar
// type promotion (mixing a dual with a plain arithmetic scalar, e.g. the
// `w * f` contraction in an HVP). This header supplies exactly the two pieces
// needed and nothing more:
//
//   * `Eigen::NumTraits<duals::dual<T, N>>`  -- duals are regular, non-integer,
//                                               non-complex, non-vectorizable
//                                               scalars, already zero-initialized
//                                               by their default constructor.
//   * `Eigen::ScalarBinaryOpTraits` promotion -- a cwise binary op between a
//                                               `dual<T,N>` and a plain
//                                               arithmetic scalar promotes back
//                                               to `dual<T,N>`.
//
// In Eigen 3.x every cwise binary operator (`+`, `-`, `*`, `/`) derives its
// result scalar from `Eigen::ScalarBinaryOpTraits<L,R,op>`. We specialize it
// for each common arithmetic scalar type (both operand orders). The fixed
// scalar type in each specialization keeps it from matching the built-in
// same-type `ScalarBinaryOpTraits<T,T,op>` (which covers `dual op dual`), so
// the two never become ambiguous; only the mixed dual/scalar case is handled
// here.
//
// Master's Eigen-5 `duals/dual_eigen` is intentionally NOT vendored: it
// `#error`s on Eigen 3 and only adds SIMD GEMM packet kernels (vectorizing
// large dual-valued GEMMs). This project has no such need -- its dual-valued
// products are the small ODE `f` expressions and the integrator's `nx x nx`
// stage Jacobians; all large dense linear algebra is plain `double`.
//
// C++17, header-only, no dependencies beyond Eigen and <duals/dual>.

#pragma once

#include <Eigen/Core>

#include <duals/dual>

#include <limits>
#include <type_traits>

namespace Eigen
{

/// `duals::dual<T, N>` is a regular scalar: not integer, not complex (unless
/// its value type is complex), and its default constructor already
/// zero-initializes, so Eigen need not initialize storage itself. Inheriting
/// from `NumTraits<T>` (the value type) picks up the correct `epsilon` /
/// `highest` / `lowest` / `digits10` / cost model; the recursive definition
/// terminates at the base arithmetic type, so a nested `dual<dual<double, N>,
/// 1>` resolves through `NumTraits<dual<double, N>>` down to `NumTraits<double>`.
template <class T, int N>
struct NumTraits<duals::dual<T, N>> : NumTraits<T>
{
    typedef duals::dual<T, N> Real;
    typedef duals::dual<T, N> Literal;
    typedef duals::dual<T, N> Nested;
    enum
    {
        IsInteger = false,
        IsSigned  = false,
        IsComplex = duals::is_complex<T>::value,
        RequireInitialization = false,
        ReadCost = 2 * NumTraits<T>::ReadCost,
        AddCost  = 2 * NumTraits<T>::AddCost,
        MulCost  = 3 * NumTraits<T>::MulCost + NumTraits<T>::AddCost
    };

    static inline Real epsilon()         { return Real(NumTraits<T>::epsilon()); }
    static inline Real dummy_precision() { return Real(NumTraits<T>::dummy_precision()); }
    static inline Real highest()         { return Real(NumTraits<T>::highest()); }
    static inline Real lowest()          { return Real(NumTraits<T>::lowest()); }
    static inline int  digits10()        { return NumTraits<T>::digits10(); }
};

}  // namespace Eigen

// Promote a cwise binary op between a `duals::dual<T, N>` and a plain
// arithmetic scalar back to `duals::dual<T, N>`. One specialization is
// needed per arithmetic type (both operand orders); the fixed scalar type in
// each keeps it from matching the same-type `ScalarBinaryOpTraits<T, T, op>`
// (dual x dual) or the plain `T x T` arithmetic cases, so there is no
// overload ambiguity. This covers, e.g., `0.5 * x` on a `Matrix<dual<...>>`
// and the HVP contraction where a `double` weights a nested-dual vector.
namespace Eigen
{
#define OCP_DUAL_MIX_TRAITS(SCALAR)                                     \
    template <class T, int N, class BinaryOp>                          \
    struct ScalarBinaryOpTraits<duals::dual<T, N>, SCALAR, BinaryOp>   \
    { typedef duals::dual<T, N> ReturnType; };                         \
    template <class T, int N, class BinaryOp>                          \
    struct ScalarBinaryOpTraits<SCALAR, duals::dual<T, N>, BinaryOp>   \
    { typedef duals::dual<T, N> ReturnType; };

OCP_DUAL_MIX_TRAITS(double)
OCP_DUAL_MIX_TRAITS(float)
OCP_DUAL_MIX_TRAITS(long double)
OCP_DUAL_MIX_TRAITS(int)
OCP_DUAL_MIX_TRAITS(long)
OCP_DUAL_MIX_TRAITS(long long)
OCP_DUAL_MIX_TRAITS(short)
OCP_DUAL_MIX_TRAITS(char)
OCP_DUAL_MIX_TRAITS(signed char)
OCP_DUAL_MIX_TRAITS(unsigned)
OCP_DUAL_MIX_TRAITS(unsigned long)
OCP_DUAL_MIX_TRAITS(unsigned long long)
OCP_DUAL_MIX_TRAITS(unsigned short)
OCP_DUAL_MIX_TRAITS(unsigned char)

#undef OCP_DUAL_MIX_TRAITS

}  // namespace Eigen

// ocp/integrators/butcher.hpp
//
// Compile-time Butcher tableaus for the phase-4 integrators.
//
// Each scheme is selected by a tag type. The tableau stores the coefficients
// as `static constexpr` arrays:
//
//   A : NS x NS, row-major A[s][j]. Row s = the stage being advanced, column
//       j = the stage whose K_j is consumed. The stage update is
//           x_s = x + h * sum_j A[s][j] * K_j.
//   b : (NS,)  quadrature weights, x_next = x + h * sum_s b[s] * K_s.
//   c : (NS,)  stage abscissas.
//
// Orientation matches acados: `A[s][j]` here equals `A_mat[j*ns + s]` read in
// sim_erk_integrator.c / sim_irk_integrator.c. The explicit tableaus are
// transcribed verbatim from `get_explicit_butcher_tableau`
// (sim_collocation_utils.c:560-647). The Radau IIA tableaus are the standard
// collocation coefficients a_{s,j} = int_0^{c_s} l_j, b_j = int_0^1 l_j
// (l_j = Lagrange basis at the Radau IIA nodes; nodes hard-coded from
// sim_collocation_utils.c:248-333, matching CasADi).
//
// Every stored tableau satisfies the consistency identities:
//   sum_j A[s][j] == c[s]        (all s)
//   sum_s b[s]    == 1
// The Radau IIA tableaus additionally satisfy the collocation order
// conditions sum_j b_j c_j^k == 1/(k+1), k = 0..2*NS-2, and the last-row
// property A[NS-1][j] == b[j].

#pragma once

#include <array>

#include "ocp/problem.hpp"

namespace ocp
{

// =========================================================================
//  Scheme tags
// =========================================================================

/// Explicit Euler (ns = 1).
struct K1Tag
{
};

/// Explicit RK2 / Heun (ns = 2).
struct K2Tag
{
};

/// Explicit RK3 (ns = 3).
struct K3Tag
{
};

/// Classic explicit RK4 (ns = 4).
struct K4Tag
{
};

/// Radau IIA collocation, 2 stages (implicit, L-stable).
struct RadauIia2Tag
{
};

/// Radau IIA collocation, 3 stages (implicit, L-stable).
struct RadauIia3Tag
{
};

/// Radau IIA collocation, 4 stages (implicit, L-stable).
struct RadauIia4Tag
{
};

// =========================================================================
//  ButcherTableau
// =========================================================================

/// A Butcher tableau for a fixed Runge-Kutta scheme.
///
/// `Dims` is carried for interface symmetry with the integrator template
/// (which is also parameterised by `Dims`); the tableau values depend only
/// on `NS` and `Tag`. Supported (NS, Tag) pairs are given by the partial
/// specialisations below; any other combination fails to compile.
template <class Dims, int NS, class Tag>
struct ButcherTableau
{
    static_assert(false,
        "ocp::ButcherTableau: unsupported (NS, Tag); use K1Tag..K4Tag "
        "(explicit) or RadauIia2Tag..RadauIia4Tag (Radau IIA) with the "
        "matching stage count.");
};

// ---- explicit RK tableaus (acados get_explicit_butcher_tableau) ---------

template <class Dims>
struct ButcherTableau<Dims, 1, K1Tag>
{
    static constexpr std::array<std::array<double, 1>, 1> A = {
        { { 0.0 } } };
    static constexpr std::array<double, 1> b = { 1.0 };
    static constexpr std::array<double, 1> c = { 0.0 };
};

template <class Dims>
struct ButcherTableau<Dims, 2, K2Tag>
{
    static constexpr std::array<std::array<double, 2>, 2> A = {
        { { 0.0, 0.0 }, { 0.5, 0.0 } } };
    static constexpr std::array<double, 2> b = { 0.0, 1.0 };
    static constexpr std::array<double, 2> c = { 0.0, 0.5 };
};

template <class Dims>
struct ButcherTableau<Dims, 3, K3Tag>
{
    static constexpr std::array<std::array<double, 3>, 3> A = {
        { { 0.0, 0.0, 0.0 },
          { 0.5, 0.0, 0.0 },
          { -1.0, 2.0, 0.0 } } };
    static constexpr std::array<double, 3> b = { 1.0 / 6.0, 2.0 / 3.0,
                                                 1.0 / 6.0 };
    static constexpr std::array<double, 3> c = { 0.0, 0.5, 1.0 };
};

template <class Dims>
struct ButcherTableau<Dims, 4, K4Tag>
{
    static constexpr std::array<std::array<double, 4>, 4> A = {
        { { 0.0, 0.0, 0.0, 0.0 },
          { 0.5, 0.0, 0.0, 0.0 },
          { 0.0, 0.5, 0.0, 0.0 },
          { 0.0, 0.0, 1.0, 0.0 } } };
    static constexpr std::array<double, 4> b = { 1.0 / 6.0, 1.0 / 3.0,
                                                 1.0 / 3.0, 1.0 / 6.0 };
    static constexpr std::array<double, 4> c = { 0.0, 0.5, 0.5, 1.0 };
};

// ---- Radau IIA collocation tableaus (a_{s,j} = int_0^{c_s} l_j) ---------

template <class Dims>
struct ButcherTableau<Dims, 2, RadauIia2Tag>
{
    static constexpr std::array<std::array<double, 2>, 2> A = {
        { { 5.0 / 12.0, -1.0 / 12.0 },
          { 0.75, 0.25 } } };
    static constexpr std::array<double, 2> b = { 0.75, 0.25 };
    static constexpr std::array<double, 2> c = { 0.33333333333333337034, 1.0 };
};

template <class Dims>
struct ButcherTableau<Dims, 3, RadauIia3Tag>
{
    static constexpr std::array<std::array<double, 3>, 3> A = {
        { { 0.19681547722366044, -0.065535425850198434, 0.023770974348220168 },
          { 0.39442431473908734, 0.29207341166522816, -0.041548752125997845 },
          { 0.37640306270046719, 0.51248582618842131, 0.11111111111111116 } } };
    static constexpr std::array<double, 3> b = { 0.37640306270046719,
                                                 0.51248582618842131,
                                                 0.11111111111111116 };
    static constexpr std::array<double, 3> c = { 0.15505102572168222297,
                                                 0.64494897427831787695, 1.0 };
};

template <class Dims>
struct ButcherTableau<Dims, 4, RadauIia4Tag>
{
    static constexpr std::array<std::array<double, 4>, 4> A = {
        { { 0.11299947932315661, -0.040309220723522464, 0.025802377420336538,
            -0.0099046765072664748 },
          { 0.23438399574740051, 0.20689257393535834, -0.047857128048540282,
            0.016047422806516214 },
          { 0.21668178462325033, 0.40612326386737307, 0.18903651817005729,
            -0.024182104899832746 },
          { 0.22046221117676867, 0.38819346884317207, 0.32884431998005947,
            0.0625 } } };
    static constexpr std::array<double, 4> b = { 0.22046221117676867,
                                                 0.38819346884317207,
                                                 0.32884431998005947,
                                                 0.0625 };
    static constexpr std::array<double, 4> c = { 0.08858795951270420632,
                                                 0.40946686444073465694,
                                                 0.78765946176084700170, 1.0 };
};

}  // namespace ocp

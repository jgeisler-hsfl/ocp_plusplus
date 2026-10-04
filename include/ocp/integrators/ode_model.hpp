// ocp/integrators/ode_model.hpp
//
// Continuous-time ODE model contract for the integrators (phase 4).
//
// The ODE model is a *duck-typed* contract: the user writes a plain struct
// with the methods below; there is no `OdeModel` base class to inherit (the
// `Dims` tag plays that role for the discrete problem). The integrator
// (rk_explicit.hpp / rk_implicit.hpp) dispatches statically on the concrete
// ODE type, exactly like `ocp::Problem` dispatches on the concrete problem.
//
// Contract (all methods `const` and pure; parameters are user-managed state
// of the concrete ODE, not interface arguments). Phase 4h makes the ODE
// *time-varying*: every method takes the physical time `t` at the evaluation
// point. The solver (integrator) decides the time grid and hands `t` to the
// model; a time-invariant ODE simply ignores `t`.
//
//   state_t f(const state_t& x, const control_t& u, double t) const;
//
//       ODE right-hand side  xdot = f(x, u, t).  `state_t` / `control_t` are
//       the fixed-size Eigen vectors of `ocp::Problem<Dims>` (nx / nu
//       components). `t` is the physical time at which the RHS is evaluated
//       (the integrator computes the per-stage time from the Butcher abscissa).
//
//   void jacobian(const state_t& x, const control_t& u, double t,
//                 dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const;
//
//       Dynamics Jacobian blocks  df/dx (nx x nx) and  df/du (nx x nu) at
//       (x, u, t).
//
//   void hess_prod(const state_t& x, const control_t& u, double t,
//                  const state_t& w, const state_t& v_x, const control_t& v_u,
//                  state_t& hv_x, control_t& hv_u) const;
//
//       Bilinear Hessian-vector product of the ODE map  f  (the *ODE's* HVP,
//       not the composed discrete map's), evaluated at (x, u, t):
//
//           hv_x = sum_i w_i (df_i/dx^2  v_x + df_i/dux v_u)
//           hv_u = sum_i w_i (df_i/dxd u v_x + df_i/du^2 v_u)
//
//       with `w` in state space (dual to the ODE output). Only required when
//       the ODE is nonlinear; a linear ODE may omit it and the integrator's
//       `hess_prod` returns zero (the Hessian of a linear ODE map is zero).
//
//   Fused entries (optional; the value/first/second triplets of the ODE map,
//   mirroring the problem-level fused contract). Backends that can compute
//   the requested derivatives in one pass (e.g. forward-mode AD) provide
//   them; others may omit them or provide them as composites of the
//   separate entries above:
//
//   void value_jac(const state_t& x, const control_t& u, double t,
//                  state_t& f_val, dyn_df_dx_t& df_dx,
//                  dyn_df_du_t& df_du) const;
//
//       f(x, u, t) and the Jacobian blocks in one evaluation: `f_val` is
//       the ODE right-hand side at (x, u, t).
//
//   void value_jac_hess_prod(const state_t& x, const control_t& u, double t,
//                            state_t& f_val, dyn_df_dx_t& df_dx,
//                            dyn_df_du_t& df_du, const state_t& w,
//                            const state_t& v_x, const control_t& v_u,
//                            state_t& hv_x, control_t& hv_u) const;
//
//       f, the Jacobian blocks, and the bilinear Hessian-vector product
//       (same semantics as `hess_prod` above) in one evaluation.
//
// Presence of `hess_prod` / `value_jac` / `value_jac_hess_prod` is detected
// at compile time by the `ode_supports_*` traits below and `static_assert`ed
// or branch-selected at the integrator call sites (mirrors the `Problem`
// stub approach): a nonlinear ODE that forgets `hess_prod` fails to compile.

#pragma once

#include <type_traits>
#include <utility>

#include "ocp/problem.hpp"

namespace ocp
{

namespace detail
{

// SFINAE probe: is `Ode::hess_prod(x, u, t, w, v_x, v_u, hv_x&, hv_u&)` a
// valid expression with the Problem<Dims> vector types? The probe is only
// instantiated when the trait is queried, so it never forces the ODE's other
// methods to exist. Phase 4h: the probe matches the time-varying signature
// (the `double t` after `u`).
template <class Ode, class Dims, class = std::void_t<>>
struct ode_hess_prod_probe : std::false_type
{
};

template <class Ode, class Dims>
struct ode_hess_prod_probe<
    Ode, Dims,
    std::void_t<decltype(std::declval<const Ode&>().hess_prod(
        std::declval<const typename Problem<Dims>::state_t&>(),
        std::declval<const typename Problem<Dims>::control_t&>(),
        std::declval<double>(),
        std::declval<const typename Problem<Dims>::state_t&>(),
        std::declval<const typename Problem<Dims>::state_t&>(),
        std::declval<const typename Problem<Dims>::control_t&>(),
        std::declval<typename Problem<Dims>::state_t&>(),
        std::declval<typename Problem<Dims>::control_t&>()))>> : std::true_type
{
};

// SFINAE probe: is `Ode::value_jac(x, u, t, f_val, df_dx, df_du)` a valid
// expression with the Problem<Dims> vector types?
template <class Ode, class Dims, class = std::void_t<>>
struct ode_value_jac_probe : std::false_type
{
};

template <class Ode, class Dims>
struct ode_value_jac_probe<Ode, Dims,
    std::void_t<decltype(std::declval<const Ode&>().value_jac(
        std::declval<const typename Problem<Dims>::state_t&>(),
        std::declval<const typename Problem<Dims>::control_t&>(),
        std::declval<double>(),
        std::declval<typename Problem<Dims>::state_t&>(),
        std::declval<typename Problem<Dims>::dyn_df_dx_t&>(),
        std::declval<typename Problem<Dims>::dyn_df_du_t&>()))>> : std::true_type
{
};

// SFINAE probe: is `Ode::value_jac_hess_prod(x, u, t, f_val, df_dx, df_du,
// w, v_x, v_u, hv_x, hv_u)` a valid expression?
template <class Ode, class Dims, class = std::void_t<>>
struct ode_value_jac_hess_probe : std::false_type
{
};

template <class Ode, class Dims>
struct ode_value_jac_hess_probe<Ode, Dims,
    std::void_t<decltype(std::declval<const Ode&>().value_jac_hess_prod(
        std::declval<const typename Problem<Dims>::state_t&>(),
        std::declval<const typename Problem<Dims>::control_t&>(),
        std::declval<double>(),
        std::declval<typename Problem<Dims>::state_t&>(),
        std::declval<typename Problem<Dims>::dyn_df_dx_t&>(),
        std::declval<typename Problem<Dims>::dyn_df_du_t&>(),
        std::declval<const typename Problem<Dims>::state_t&>(),
        std::declval<const typename Problem<Dims>::state_t&>(),
        std::declval<const typename Problem<Dims>::control_t&>(),
        std::declval<typename Problem<Dims>::state_t&>(),
        std::declval<typename Problem<Dims>::control_t&>()))>> : std::true_type
{
};

// SFINAE probe: can `Ode::f` be called with a dual scalar type `D`?
// Requires both that the call is well-formed AND that the return type is
// `Matrix<D, nx, 1>` (guards against Eigen's unconstrained converting
// constructor making a hand-written double-f appear dual-evaluable).
template <class Ode, class Dims, class D, class = void>
struct ode_f_evaluable_at : std::false_type
{
};

template <class Ode, class Dims, class D>
struct ode_f_evaluable_at<Ode, Dims, D,
    std::void_t<decltype(std::declval<const Ode&>().f(
        std::declval<const Eigen::Matrix<D, Problem<Dims>::nx, 1>&>(),
        std::declval<const Eigen::Matrix<D, Problem<Dims>::nu, 1>&>(),
        std::declval<double>()))>>
    : std::is_same<std::decay_t<decltype(std::declval<const Ode&>().f(
          std::declval<const Eigen::Matrix<D, Problem<Dims>::nx, 1>&>(),
          std::declval<const Eigen::Matrix<D, Problem<Dims>::nu, 1>&>(),
          std::declval<double>()))>,
                   Eigen::Matrix<D, Problem<Dims>::nx, 1>>
{
};

}  // namespace detail

/// True when the ODE model provides the `hess_prod` method (nonlinear ODE);
/// false when it is linear and the ODE HVP is omitted.
template <class Ode, class Dims>
constexpr bool ode_supports_hess_prod =
    detail::ode_hess_prod_probe<Ode, Dims>::value;

/// True when the ODE model provides the fused `value_jac` entry.
template <class Ode, class Dims>
constexpr bool ode_supports_value_jac =
    detail::ode_value_jac_probe<Ode, Dims>::value;

/// True when the ODE model provides the fused `value_jac_hess_prod` entry.
template <class Ode, class Dims>
constexpr bool ode_supports_value_jac_hess_prod =
    detail::ode_value_jac_hess_probe<Ode, Dims>::value;

/// True when `Ode::f` can be called with scalar type `D` and returns a
/// `Matrix<D, nx, 1>`. Used by integrators to detect AD-capable ODEs.
template <class Ode, class Dims, class D>
constexpr bool ode_f_evaluable_at =
    detail::ode_f_evaluable_at<Ode, Dims, D>::value;

/// Invoke the ODE's Hessian-vector product with the `Problem<Dims>` vector
/// types. Only usable when `ode_supports_hess_prod<Ode, Dims>` is true (the
/// integrator `static_assert`s this before calling).
template <class Ode, class Dims>
inline void ode_hess_prod(const Ode& ode, const typename Problem<Dims>::state_t& x,
                           const typename Problem<Dims>::control_t& u, double t,
                           const typename Problem<Dims>::state_t& w,
                           const typename Problem<Dims>::state_t& v_x,
                           const typename Problem<Dims>::control_t& v_u,
                           typename Problem<Dims>::state_t& hv_x,
                           typename Problem<Dims>::control_t& hv_u)
{
    ode.hess_prod(x, u, t, w, v_x, v_u, hv_x, hv_u);
}

/// Invoke the ODE's fused `value_jac` entry.
template <class Ode, class Dims>
inline void ode_value_jac(const Ode& ode,
                          const typename Problem<Dims>::state_t& x,
                          const typename Problem<Dims>::control_t& u, double t,
                          typename Problem<Dims>::state_t& f_val,
                          typename Problem<Dims>::dyn_df_dx_t& df_dx,
                          typename Problem<Dims>::dyn_df_du_t& df_du)
{
    ode.value_jac(x, u, t, f_val, df_dx, df_du);
}

/// Invoke the ODE's fused `value_jac_hess_prod` entry.
template <class Ode, class Dims>
inline void ode_value_jac_hess_prod(
    const Ode& ode, const typename Problem<Dims>::state_t& x,
    const typename Problem<Dims>::control_t& u, double t,
    typename Problem<Dims>::state_t& f_val,
    typename Problem<Dims>::dyn_df_dx_t& df_dx,
    typename Problem<Dims>::dyn_df_du_t& df_du,
    const typename Problem<Dims>::state_t& w,
    const typename Problem<Dims>::state_t& v_x,
    const typename Problem<Dims>::control_t& v_u,
    typename Problem<Dims>::state_t& hv_x,
    typename Problem<Dims>::control_t& hv_u)
{
    ode.value_jac_hess_prod(x, u, t, f_val, df_dx, df_du, w, v_x, v_u,
                           hv_x, hv_u);
}

}  // namespace ocp

// ocp/integrators/dae_model.hpp
//
// DAE model contract for the implicit (IRK) integrator (rk_implicit.hpp).
//
// Where the explicit integrator consumes the ODE right-hand side
//  xdot = f(x, u, t)  (ode_model.hpp), the implicit integrator consumes the
// DAE residual
//
//     F(x, xdot, u, t) = 0
//
// directly — the form the generated CasADi code (dae_fun / dae_fun_jac)
// provides.  An ODE written in residual form is F = xdot - f (or f - xdot)
// with dF/dxdot = +/-(I); the integrator's Newton solve is exact for any
// (regular) dF/dxdot, not just the diagonal ODE case.
//
// The DAE model is a *duck-typed* contract (like the ODE model in
// ode_model.hpp): a plain struct with the methods below; the integrator
// (rk_implicit.hpp) dispatches statically on the concrete type.  All
// methods are `const`; vectors are the fixed-size Eigen types of
// `Problem<Dims>`.
//
//   state_t F(const state_t& x, const state_t& xdot,
//             const control_t& u, double t) const;
//
//       DAE residual (nx-vector).
//
//   void F_jac(const state_t& x, const state_t& xdot,
//              const control_t& u, double t, dyn_df_dx_t& dFdx,
//              dyn_df_dx_t& dFxdot, dyn_df_du_t& dFdu) const;
//
//       Jacobian blocks of F at (x, xdot, u, t):  dFdx (nx x nx),
//       dFxdot (nx x nx),  dFdu (nx x nu).
//
//   void F_hess_prod(const state_t& x, const state_t& xdot,
//                    const control_t& u, double t, const state_t& w,
//                    const state_t& v_x, const state_t& v_xdot,
//                    const control_t& v_u, state_t& hv_x,
//                    state_t& hv_xdot, control_t& hv_u) const;
//
//       Optional (nonlinear DAE only; presence probed by
//       dae_supports_hess_prod).  Bilinear Hessian-vector product of F
//       w.r.t. z = (x, xdot, u) with output weight `w` and direction
//       (v_x, v_xdot, v_u):
//
//           hv_x    = (sum_i w_i d²F_i/dz²) · v  (x-part)
//           hv_xdot = … (xdot-part)
//           hv_u    = … (u-part)
//
//       A DAE affine in xdot (in particular an ODE residual) has
//       hv_xdot = 0.

#pragma once

#include "ocp/integrators/ode_model.hpp"
#include "ocp/problem.hpp"

#include <type_traits>
#include <utility>

namespace ocp
{

namespace detail
{

// SFINAE probe: is `Dae::F_hess_prod(x, xdot, u, t, w, v_x, v_xdot, v_u,
// hv_x, hv_xdot, hv_u)` a valid expression with the Problem<Dims> types?
template <class Dae, class Dims, class = std::void_t<>>
struct dae_hess_prod_probe : std::false_type
{
};

template <class Dae, class Dims>
struct dae_hess_prod_probe<
    Dae, Dims,
    std::void_t<decltype(std::declval<const Dae&>().F_hess_prod(
        std::declval<const typename Problem<Dims>::state_t&>(),
        std::declval<const typename Problem<Dims>::state_t&>(),
        std::declval<const typename Problem<Dims>::control_t&>(),
        std::declval<double>(),
        std::declval<const typename Problem<Dims>::state_t&>(),
        std::declval<const typename Problem<Dims>::state_t&>(),
        std::declval<const typename Problem<Dims>::state_t&>(),
        std::declval<const typename Problem<Dims>::control_t&>(),
        std::declval<typename Problem<Dims>::state_t&>(),
        std::declval<typename Problem<Dims>::state_t&>(),
        std::declval<typename Problem<Dims>::control_t&>()))>>
    : std::true_type
{
};

// Base for OdeToDae: holds the wrapped ODE.  The partial specialization for
// ODEs that provide hess_prod adds F_hess_prod; it is absent otherwise so
// the probe above correctly reports false and the integrator takes the
// zero-HVP path (linear-ODE fallback).
template <class Ode, class Dims, bool HasHess>
struct OdeToDaeBase
{
    Ode ode_;
    OdeToDaeBase(const Ode& o) : ode_(o) {}
};

template <class Ode, class Dims>
struct OdeToDaeBase<Ode, Dims, true>
{
    Ode ode_;
    OdeToDaeBase(const Ode& o) : ode_(o) {}

    using state_t   = typename Problem<Dims>::state_t;
    using control_t = typename Problem<Dims>::control_t;

    // F = xdot - f is affine in xdot: no xdot Hessian blocks, and
    // D2 F = -D2 f, so the ODE HVP is negated.
    void F_hess_prod(const state_t& x, const state_t& xdot,
                     const control_t& u, double t, const state_t& w,
                     const state_t& v_x, const state_t& v_xdot,
                     const control_t& v_u, state_t& hv_x,
                     state_t& hv_xdot, control_t& hv_u) const
    {
        (void)xdot;
        (void)v_xdot;
        ode_.hess_prod(x, u, t, w, v_x, v_u, hv_x, hv_u);
        hv_x = -hv_x;
        hv_u = -hv_u;
        hv_xdot.setZero();
    }
};

}  // namespace detail

/// True when the DAE model provides `F_hess_prod` (nonlinear DAE); false
/// when the HVP entry is omitted (the integrator then returns a zero
/// composed HVP, i.e. the Gauss-Newton Hessian model).
template <class Dae, class Dims>
constexpr bool dae_supports_hess_prod =
    detail::dae_hess_prod_probe<Dae, Dims>::value;

/// Invoke the DAE's Hessian-vector product with the `Problem<Dims>` vector
/// types.  Only usable when `dae_supports_hess_prod<Dae, Dims>` is true
/// (the integrator `static_assert`s this before calling).
template <class Dae, class Dims>
inline void dae_hess_prod(const Dae& dae,
                          const typename Problem<Dims>::state_t& x,
                          const typename Problem<Dims>::state_t& xdot,
                          const typename Problem<Dims>::control_t& u,
                          double t,
                          const typename Problem<Dims>::state_t& w,
                          const typename Problem<Dims>::state_t& v_x,
                          const typename Problem<Dims>::state_t& v_xdot,
                          const typename Problem<Dims>::control_t& v_u,
                          typename Problem<Dims>::state_t& hv_x,
                          typename Problem<Dims>::state_t& hv_xdot,
                          typename Problem<Dims>::control_t& hv_u)
{
    dae.F_hess_prod(x, xdot, u, t, w, v_x, v_xdot, v_u, hv_x, hv_xdot, hv_u);
}

/// Adapts an ODE model (ode_model.hpp contract) to the DAE contract with
/// the residual  F(x, xdot, u, t) = xdot - f(x, u, t),  so hand-written
/// ODE models can be used with the implicit integrator unchanged.
///
/// The ODE is held by value; aggregate-initialize with the model:
///   OdeToDae<MyOde, Dims> dae{MyOde{...}};
/// `F_hess_prod` is present only when the wrapped ODE provides
/// `hess_prod` (see dae_supports_hess_prod).
template <class Ode, class Dims>
struct OdeToDae
    : detail::OdeToDaeBase<Ode, Dims, ode_supports_hess_prod<Ode, Dims>>
{
    using P           = Problem<Dims>;
    using state_t     = typename P::state_t;
    using control_t   = typename P::control_t;
    using dyn_df_dx_t = typename P::dyn_df_dx_t;
    using dyn_df_du_t = typename P::dyn_df_du_t;

    /// Implicit conversion from the wrapped ODE model, so integrator
    /// construction wraps at the call site:
    ///   ImplicitRkIntegrator<Dims, OdeToDae<Ode, Dims>, ...> integ(ode);
    OdeToDae(const Ode& ode)
        : detail::OdeToDaeBase<Ode, Dims, ode_supports_hess_prod<Ode, Dims>>(
              ode)
    {
    }

    /// DAE residual:  xdot - f(x, u, t).
    state_t F(const state_t& x, const state_t& xdot, const control_t& u,
              double t) const
    {
        return xdot - this->ode_.f(x, u, t);
    }

    /// Jacobian blocks:  dFdx = -df/dx,  dFxdot = I,  dFdu = -df/du.
    void F_jac(const state_t& x, const state_t& xdot, const control_t& u,
               double t, dyn_df_dx_t& dFdx, dyn_df_dx_t& dFxdot,
               dyn_df_du_t& dFdu) const
    {
        (void)xdot;  // F is affine in xdot.
        dyn_df_dx_t df_dx;
        dyn_df_du_t df_du;
        this->ode_.jacobian(x, u, t, df_dx, df_du);
        dFdx   = -df_dx;
        dFxdot = dyn_df_dx_t::Identity();
        dFdu   = -df_du;
    }
};

}  // namespace ocp

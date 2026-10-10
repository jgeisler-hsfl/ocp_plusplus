// ocp/integrators/dae_autodiff.hpp
//
// CRTP base providing the DAE derivative contract (dae_model.hpp) from the
// DAE residual `F` alone, via forward-mode AD on the vendored multidual
// scalar (thirdparty/cppduals, Eigen-3 glue in duals_eigen3.hpp).
//
// The implicit integrator (rk_implicit.hpp) has no dual-Newton path, so its
// derivatives must come from the model's `F_jac` / `F_hess_prod` entries.
// This base generates them from a single scalar-templated residual:
//
//   template <class D>
//   Eigen::Matrix<D, nx, 1> F(const Eigen::Matrix<D, nx, 1>& x,
//                             const Eigen::Matrix<D, nx, 1>& xdot,
//                             const Eigen::Matrix<D, nu, 1>& u,
//                             double t) const;
//
// where nx / nu are the Dims dimensions (re-exposed below).  The same
// method is called with D = double (value pass) and with the dual scalar
// types below (derivative passes), so it must be written with plain
// arithmetic / Eigen cwise operations on D (and may use t).  The base does
// not define its own F, so there is no name hiding.
//
// Derivative cost (allocation-free; fixed-size Eigen buffers on the stack):
//   F_jac -- one forward pass of dual<scalar_t, nIn>  (nIn = 2 nx + nu);
//            the value F_val is carried for free (rpart of each component).
//   F_hess_prod -- one forward-over-forward pass of
//            dual<dual<scalar_t, nIn>, 1>, giving  r(j) = (d2g/dz^2 v)_j
//            for  g(z) = w^T F(z),  z = [x; xdot; u];  the value and the
//            full first derivatives are carried for free;  w is applied
//            only in the final output contraction, never inside F.
//
// The `WithHess` switch selects the Hessian model of the whole solver:
//   WithHess == true (default): F_hess_prod is provided, so
//     dae_supports_hess_prod is true and ImplicitRkIntegrator computes the
//     full second-order sensitivities (exact-Newton Hessian model).
//   WithHess == false: F_hess_prod is not provided, so
//     dae_supports_hess_prod is false and ImplicitRkIntegrator::hess_prod
//     returns zeros (Gauss-Newton Hessian model; the dynamics second-order
//     term is dropped).  Set Dims::has_dynamics_hess = false to skip the
//     dynamics Hessian assembly in the SQP as well.
//
// Usage:
//
//   struct MyDae : ocp::AutoDiffDae<MyDae, MyDims>
//   {
//       template <class D>
//       Eigen::Matrix<D, nx, 1> F(const Eigen::Matrix<D, nx, 1>& x,
//                                 const Eigen::Matrix<D, nx, 1>& xdot,
//                                 const Eigen::Matrix<D, nu, 1>& u,
//                                 double t) const
//       {
//           return xdot - x.cwiseProduct(x) * u(0) - x;
//       }
//   };
//
// An ODE is expressed as the residual  F = xdot - f(x, u, t)  (dFxdot = I);
// the formulation is exact for any (regular) dFxdot, so implicit DAEs work
// unchanged.  A purely linear ODE needs no AD at all: hand-write the
// residual with has_dynamics_hess = false (OdeToDae over a hand-written ODE,
// or a hand-written linear DAE).

#pragma once

#include "ocp/integrators/dae_model.hpp"
#include "ocp/integrators/duals_eigen3.hpp"

namespace ocp
{

/// CRTP base providing the dae_model.hpp derivative entries for a DAE that
/// defines only a scalar-templated `F` (see the file-level contract above).
/// `Derived` is the concrete DAE; `Dims` is the same dimensions tag the
/// integrator and `ContinuousProblem` use, so all vector types match the
/// dae_model.hpp contract verbatim.
///
/// The primary template is the Hessian-suppressed variant (`WithHess ==
/// false`): it exposes `F_jac` only, so `dae_supports_hess_prod` is false
/// and ImplicitRkIntegrator::hess_prod returns zeros (Gauss-Newton model).
/// The `true` specialization below adds the fused `F_hess_prod` (nested-
/// dual forward-over-forward pass), giving the exact-Newton Hessian model.
template <class Derived, class Dims, bool WithHess = true>
class AutoDiffDae
{
public:
    // Re-exposed Problem<Dims> types / dimensions (layout: z = [x; xdot; u]).
    using P         = Problem<Dims>;
    using scalar_t  = typename P::scalar_t;
    using state_t   = typename P::state_t;
    using control_t = typename P::control_t;
    using df_dx_t   = typename P::dyn_df_dx_t;
    using df_du_t   = typename P::dyn_df_du_t;

    static constexpr int nx  = P::nx;
    static constexpr int nu  = P::nu;
    static constexpr int nIn = 2 * nx + nu;

    using dual1_t = duals::dual<scalar_t, nIn>;

    /// The concrete DAE (CRTP accessor).
    const Derived& self() const
    {
        return static_cast<const Derived&>(*this);
    }

    /// DAE Jacobian blocks dF/dx (nx x nx), dF/dxdot (nx x nx), dF/du
    /// (nx x nu) at (x, xdot, u, t), from a single `dual<scalar_t, nIn>`
    /// pass over z = [x; xdot; u].  Input k carries the unit vector e_k in
    /// its nIn-partial storage, so fd(i).dpart(k) = dF_i/dz_k; the residual
    /// value (fd(i).rpart()) is carried for free and dropped (the
    /// dae_model.hpp contract has no fused value + Jacobian entry).
    void F_jac(const state_t& x, const state_t& xdot, const control_t& u,
               double t, df_dx_t& dFdx, df_dx_t& dFxdot, df_du_t& dFdu) const
    {
        Eigen::Matrix<dual1_t, nIn, 1> zbar;
        for (int k = 0; k < nx; ++k)
        {
            zbar(k) = dual1_t::variable(x(k), k);
        }
        for (int k = 0; k < nx; ++k)
        {
            zbar(nx + k) = dual1_t::variable(xdot(k), nx + k);
        }
        for (int k = 0; k < nu; ++k)
        {
            zbar(2 * nx + k) = dual1_t::variable(u(k), 2 * nx + k);
        }

        // Copy out the exact-size argument blocks: a segment view does not
        // match the derived F signature.
        const Eigen::Matrix<dual1_t, nx, 1> xd    = zbar.head(nx);
        const Eigen::Matrix<dual1_t, nx, 1> xdotd = zbar.segment(nx, nx);
        const Eigen::Matrix<dual1_t, nu, 1> ud    = zbar.tail(nu);

        const Eigen::Matrix<dual1_t, nx, 1> fd = self().F(xd, xdotd, ud, t);
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                dFdx(i, j) = fd(i).dpart(j);
            }
            for (int j = 0; j < nx; ++j)
            {
                dFxdot(i, j) = fd(i).dpart(nx + j);
            }
            for (int j = 0; j < nu; ++j)
            {
                dFdu(i, j) = fd(i).dpart(2 * nx + j);
            }
        }
    }
};

/// `WithHess == true` specialization: adds the fused bilinear
/// Hessian-vector product of the DAE residual (dae_model.hpp contract) from
/// a single nested `dual<dual<scalar_t, nIn>, 1>` pass, so
/// `dae_supports_hess_prod` is true and ImplicitRkIntegrator computes the
/// full second-order sensitivities.
template <class Derived, class Dims>
class AutoDiffDae<Derived, Dims, true>
    : public AutoDiffDae<Derived, Dims, false>
{
public:
    using Base      = AutoDiffDae<Derived, Dims, false>;
    using state_t   = typename Base::state_t;
    using control_t = typename Base::control_t;
    using dual1_t   = typename Base::dual1_t;
    using dual2_t   = duals::dual<dual1_t, 1>;

    /// Bilinear Hessian-vector product of the DAE map (dae_model.hpp
    /// contract): with  g(z) = w^T F(z),  z = [x; xdot; u],  v = (v_x,
    /// v_xdot, v_u),
    ///
    ///     [hv_x; hv_xdot; hv_u] = D^2 g(x, xdot, u, t) * v
    ///
    /// from a single forward-over-forward nested-dual pass.  Seeding per
    /// input k: inner multidual rpart = z_k with partials e_k (the full
    /// nIn gradient basis); outer single dual rpart = v_k with zero
    /// partials (the HVP direction).  The value and the full first
    /// derivatives are carried for free; `w` is in state space and is
    /// applied only at the output contraction.
    void F_hess_prod(const state_t& x, const state_t& xdot,
                     const control_t& u, double t, const state_t& w,
                     const state_t& v_x, const state_t& v_xdot,
                     const control_t& v_u, state_t& hv_x, state_t& hv_xdot,
                     control_t& hv_u) const
    {
        constexpr int nx  = Base::nx;
        constexpr int nu  = Base::nu;
        constexpr int nIn = Base::nIn;

        Eigen::Matrix<dual2_t, nIn, 1> zbar;
        for (int k = 0; k < nx; ++k)
        {
            zbar(k) = dual2_t(dual1_t::variable(x(k), k), dual1_t(v_x(k)));
        }
        for (int k = 0; k < nx; ++k)
        {
            zbar(nx + k) = dual2_t(dual1_t::variable(xdot(k), nx + k),
                                   dual1_t(v_xdot(k)));
        }
        for (int k = 0; k < nu; ++k)
        {
            zbar(2 * nx + k) = dual2_t(dual1_t::variable(u(k), 2 * nx + k),
                                       dual1_t(v_u(k)));
        }

        const Eigen::Matrix<dual2_t, nx, 1> xd    = zbar.head(nx);
        const Eigen::Matrix<dual2_t, nx, 1> xdotd = zbar.segment(nx, nx);
        const Eigen::Matrix<dual2_t, nu, 1> ud    = zbar.tail(nu);

        const Eigen::Matrix<dual2_t, nx, 1> fd =
            this->self().F(xd, xdotd, ud, t);

        // g = w^T F: apply the state-space weights only at the output.
        dual2_t g(0.0);
        for (int i = 0; i < nx; ++i)
        {
            g += w(i) * fd(i);
        }
        for (int j = 0; j < nx; ++j)
        {
            hv_x(j) = g.dpart(0).dpart(j);
        }
        for (int j = 0; j < nx; ++j)
        {
            hv_xdot(j) = g.dpart(0).dpart(nx + j);
        }
        for (int j = 0; j < nu; ++j)
        {
            hv_u(j) = g.dpart(0).dpart(2 * nx + j);
        }
    }
};

}  // namespace ocp

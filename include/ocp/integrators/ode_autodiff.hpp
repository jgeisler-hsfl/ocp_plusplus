// ocp/integrators/ode_autodiff.hpp
//
// CRTP base providing the ODE derivative contract (ode_model.hpp) from the
// ODE right-hand side `f` alone, via forward-mode AD on the vendored
// multidual scalar (thirdparty/cppduals, Eigen-3 glue in duals_eigen3.hpp).
//
// The derived ODE defines ONLY a scalar-templated right-hand side:
//
//   template <class D>
//   Eigen::Matrix<D, nx, 1> f(const Eigen::Matrix<D, nx, 1>& x,
//                             const Eigen::Matrix<D, nu, 1>& u,
//                             double t) const;
//
// where nx / nu are the Dims dimensions (re-exposed below).  The same
// method is called with D = double (value pass) and with the dual scalar
// types below (derivative passes), so it must be written with plain
// arithmetic / Eigen cwise operations on D (and may use t).  The base does
// not define its own f, so there is no name hiding.
//
// Derivative cost (allocation-free; fixed-size Eigen buffers on the stack):
//   value_jac -- one forward pass of dual<scalar_t, nIn>  (nIn = nx + nu);
//                the value f_val is carried for free (rpart of each
//                component);
//   value_jac_hess_prod -- one forward-over-forward pass of
//                dual<dual<scalar_t, nIn>, 1>, giving  r(j) = (d2g/dz^2 v)_j
//                for  g(z) = w^T f(z),  z = [x; u];  value and the full
//                first derivatives are carried for free;  w is applied only
//                in the final output contraction, never inside f.
//
// The legacy entries jacobian / hess_prod are thin wrappers over the fused
// ones (dropping the value, respectively the value + Jacobian), so an
// AutoDiffOde satisfies ode_model.hpp exactly and drops into
// ExplicitRkIntegrator / ImplicitRkIntegrator / ContinuousProblem
// unchanged.
//
// Usage:
//
//   struct MyOde : ocp::AutoDiffOde<MyOde, MyDims>
//   {
//       template <class D>
//       Eigen::Matrix<D, nx, 1> f(const Eigen::Matrix<D, nx, 1>& x,
//                                 const Eigen::Matrix<D, nu, 1>& u,
//                                 double t) const
//       {
//           return x.cwiseProduct(x) * u(0) + x;
//       }
//   };
//
// Set Dims::has_dynamics_hess = true for a nonlinear AutoDiffOde.
//
// Linear ODEs: the inherited hess_prod is present and returns exactly zero
// (a linear f carries no mixed second-order dual terms), so setting
// has_dynamics_hess = true is safe but pays one wasted nested pass per
// stage.  For a purely linear ODE it is cleaner to skip AutoDiffOde and
// hand-write jacobian with has_dynamics_hess = false (the OdeLin /
// OdeDoubleInt pattern in the 4b/4d tests).

#pragma once

#include "ocp/integrators/ode_model.hpp"
#include "ocp/integrators/duals_eigen3.hpp"

namespace ocp
{

/// CRTP base providing the ode_model.hpp derivative entries (fused
/// `value_jac` / `value_jac_hess_prod` plus the legacy `jacobian` /
/// `hess_prod` wrappers) for an ODE that defines only a scalar-templated
/// `f` (see the file-level contract above). `Derived` is the concrete ODE;
/// `Dims` is the same dimensions tag the integrators and
/// `ContinuousProblem` use, so all vector types match the ode_model.hpp
/// contract verbatim.
template <class Derived, class Dims>
class AutoDiffOde
{
public:
    // Re-exposed Problem<Dims> types / dimensions (layout: z = [x; u]).
    using P         = Problem<Dims>;
    using scalar_t  = typename P::scalar_t;
    using state_t   = typename P::state_t;
    using control_t = typename P::control_t;
    using df_dx_t   = typename P::dyn_df_dx_t;
    using df_du_t   = typename P::dyn_df_du_t;

    static constexpr int nx  = P::nx;
    static constexpr int nu  = P::nu;
    static constexpr int nIn = nx + nu;

    using dual1_t = duals::dual<scalar_t, nIn>;
    using dual2_t = duals::dual<dual1_t, 1>;

    /// The concrete ODE (CRTP accessor).
    const Derived& self() const
    {
        return static_cast<const Derived&>(*this);
    }

    /// Fused ODE value + Jacobian in one `dual<scalar_t, nIn>` pass:
    /// input k carries the unit vector e_k in its nIn-partial storage, so
    /// fd(i).rpart() = f_i and fd(i).dpart(k) = df_i/dz_k with
    /// z = [x; u] (columns 0..nx-1 are df/dx, nx..nIn-1 are df/du).
    void value_jac(const state_t& x, const control_t& u, double t,
                   state_t& f_val, df_dx_t& df_dx, df_du_t& df_du) const
    {
        Eigen::Matrix<dual1_t, nIn, 1> zbar;
        for (int k = 0; k < nx; ++k)
        {
            zbar(k) = dual1_t::variable(x(k), k);
        }
        for (int k = 0; k < nu; ++k)
        {
            zbar(nx + k) = dual1_t::variable(u(k), nx + k);
        }

        // Copy out the exact-size state / control parts: a head/tail
        // VectorBlock view does not match the derived f signature.
        Eigen::Matrix<dual1_t, nx, 1> xd = zbar.head(nx);
        Eigen::Matrix<dual1_t, nu, 1> ud = zbar.tail(nu);

        const Eigen::Matrix<dual1_t, nx, 1> fd = self().f(xd, ud, t);
        for (int i = 0; i < nx; ++i)
        {
            f_val(i) = fd(i).rpart();
            for (int j = 0; j < nx; ++j)
            {
                df_dx(i, j) = fd(i).dpart(j);
            }
            for (int j = 0; j < nu; ++j)
            {
                df_du(i, j) = fd(i).dpart(nx + j);
            }
        }
    }

    /// Fused ODE value + Jacobian + bilinear Hessian-vector product in one
    /// nested `dual<dual<scalar_t, nIn>, 1>` pass.  Seeding per input k:
    /// inner multidual rpart = z_k with partials e_k (the full nIn gradient
    /// basis); outer single dual rpart = v_k with zero partials (the HVP
    /// direction).  Then g = sum_i w_i f_i(z) and
    /// r(j) = g.dpart(0).dpart(j) = e_j^T D^2g v;  the value and the first
    /// derivatives are read from fd(i).rpart() for free.  `w` is in state
    /// space and is applied only at the output contraction.
    void value_jac_hess_prod(const state_t& x, const control_t& u, double t,
                             state_t& f_val, df_dx_t& df_dx,
                             df_du_t& df_du, const state_t& w,
                             const state_t& v_x, const control_t& v_u,
                             state_t& hv_x, control_t& hv_u) const
    {
        Eigen::Matrix<dual2_t, nIn, 1> zbar;
        for (int k = 0; k < nx; ++k)
        {
            zbar(k) = dual2_t(dual1_t::variable(x(k), k), dual1_t(v_x(k)));
        }
        for (int k = 0; k < nu; ++k)
        {
            zbar(nx + k) =
                dual2_t(dual1_t::variable(u(k), nx + k), dual1_t(v_u(k)));
        }

        Eigen::Matrix<dual2_t, nx, 1> xd = zbar.head(nx);
        Eigen::Matrix<dual2_t, nu, 1> ud = zbar.tail(nu);

        const Eigen::Matrix<dual2_t, nx, 1> fd = self().f(xd, ud, t);
        for (int i = 0; i < nx; ++i)
        {
            f_val(i) = fd(i).rpart().rpart();
            for (int j = 0; j < nx; ++j)
            {
                df_dx(i, j) = fd(i).rpart().dpart(j);
            }
            for (int j = 0; j < nu; ++j)
            {
                df_du(i, j) = fd(i).rpart().dpart(nx + j);
            }
        }

        // g = w^T f: apply the state-space weights only at the output.
        dual2_t g(0.0);
        for (int i = 0; i < nx; ++i)
        {
            g += w(i) * fd(i);
        }
        for (int j = 0; j < nx; ++j)
        {
            hv_x(j) = g.dpart(0).dpart(j);
        }
        for (int j = 0; j < nu; ++j)
        {
            hv_u(j) = g.dpart(0).dpart(nx + j);
        }
    }

    /// ODE Jacobian blocks df/dx (nx x nx) and df/du (nx x nu) at (x, u, t),
    /// from a single dual<scalar_t, nIn> pass (value dropped).
    void jacobian(const state_t& x, const control_t& u, double t,
                  df_dx_t& df_dx, df_du_t& df_du) const
    {
        state_t f_val;
        value_jac(x, u, t, f_val, df_dx, df_du);
    }

    /// Bilinear Hessian-vector product of the ODE map (ode_model.hpp
    /// contract): with  g(z) = w^T f(z),  v = (v_x, v_u),
    ///
    ///     [hv_x; hv_u] = D^2 g(x, u, t) * v
    ///
    /// from a single nested dual pass (value and Jacobian dropped).
    void hess_prod(const state_t& x, const control_t& u, double t,
                   const state_t& w, const state_t& v_x, const control_t& v_u,
                   state_t& hv_x, control_t& hv_u) const
    {
        state_t f_val;
        df_dx_t df_dx;
        df_du_t df_du;
        value_jac_hess_prod(x, u, t, f_val, df_dx, df_du, w, v_x, v_u, hv_x,
                            hv_u);
    }
};

}  // namespace ocp

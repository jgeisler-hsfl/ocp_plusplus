// ocp/integrators/rk_implicit.hpp
//
// Allocation-free implicit Runge-Kutta / collocation integrator (phase 4d).
//
// ImplicitRkIntegrator turns a time-invariant ODE  xdot = f(x, u)  into the
// discrete dynamics  x_next = Phi(x, u)  via a fixed implicit RK scheme
// (ButcherTableau<Dims, NS, Tag>, Radau IIA). Because the Butcher matrix A is
// full (not strictly lower triangular), the stage equations
//
//     K_s = f(x + h * sum_j A[s][j] * K_j, u),   s = 0..NS-1
//
// form a coupled nonlinear system in the NS*nx unknowns K_1..K_NS. Each OCP
// step solves it by Newton iteration with a single block LU factorization of
// the NS*nx-by-NS*nx Jacobian (matches acados sim_irk_integrator.c: the block
// matrix dG_dK_ss is factored in place and reused for the correction and, on
// the sensitivity path, for the first-order solves).
//
// value    -- Newton solve, then x_next = x + h * sum_s b_s * K_s.
// jacobian -- implicit first-order sensitivity: dK/d(x,u) = G^{-1} * dR/d(x,u),
//             then the same b-weighted accumulation as the explicit path.
// hess_prod -- NOT implemented for the implicit path in this phase (the
//             full-A self-coupling G[s][j] = -h A[s][j] f_x needs the full ODE
//             Hessian, not just the ODE hess_prod). A linear ODE (no
//             hess_prod) has a zero composed Hessian and returns zeros; a
//             nonlinear ODE reaches a static_assert (see the plan, 4d).
//
// Allocation-free: every scratch buffer is a fixed-size Eigen object or
// std::array, sized once on NS and Dims::nx/nu. Methods are const; the ODE
// model is held by const reference; the per-call scratch is `mutable`.
//
// Newton options mirror acados: newton_max (opts->newton_iter, default 3)
// and newton_tol (opts->newton_tol, default 0.0 = fixed-iteration count, no
// early termination).

#pragma once

#include "ocp/integrators/butcher.hpp"
#include "ocp/integrators/ode_model.hpp"

#include <array>

#include <Eigen/Dense>

namespace ocp
{

namespace detail
{

/// Pre-allocated scratch for one implicit RK step (NS stages; sizes from P).
template <class P, int NS>
struct IrkWorkspace
{
    using scalar_t  = typename P::scalar_t;
    using state_t   = typename P::state_t;
    using control_t = typename P::control_t;
    using df_dx_t   = typename P::dyn_df_dx_t;
    using df_du_t   = typename P::dyn_df_du_t;
    static constexpr int NK = NS * P::nx;

    std::array<state_t, NS> K{};        // stage derivatives (Newton unknowns)
    std::array<state_t, NS> x_stage{};  // stage states x_s
    std::array<df_dx_t, NS> dfdx{};     // df/dx at stage s
    std::array<df_du_t, NS> dfdu{};     // df/du at stage s

    Eigen::Matrix<scalar_t, NK, NK> G{};   // block Newton Jacobian
    Eigen::PartialPivLU<Eigen::Matrix<scalar_t, NK, NK>> lu_;  // factored G
    Eigen::Matrix<scalar_t, NK, 1>  res{}; // Newton residual / correction
    Eigen::Matrix<scalar_t, NK, P::nx> rhs_x{};  // sensitivity RHS (dK/dx)
    Eigen::Matrix<scalar_t, NK, P::nu> rhs_u{};  // sensitivity RHS (dK/du)

    double last_residual_ = 0.0;  // ‖R‖∞ at the converged iterate (diagnostic)
};

}  // namespace detail

/// Implicit Runge-Kutta (collocation) integrator for a time-invariant ODE.
///
/// `NS` is the number of RK stages (2..4 for the Radau IIA tags). `Tag`
/// selects the Butcher scheme (RadauIia2Tag..RadauIia4Tag). `h` is the fixed
/// step size. The ODE model must provide `f` and `jacobian`; `hess_prod` is
/// only relevant to the (not-yet-implemented) nonlinear HVP path.
template <class Dims, class Ode, int NS, class Tag>
class ImplicitRkIntegrator
{
    using P   = ocp::Problem<Dims>;
    using Tab = ocp::ButcherTableau<Dims, NS, Tag>;

    static constexpr int nx = P::nx;
    static constexpr int nu = P::nu;
    static constexpr int NK = NS * nx;

public:
    using state_t   = typename P::state_t;
    using control_t = typename P::control_t;
    using df_dx_t   = typename P::dyn_df_dx_t;
    using df_du_t   = typename P::dyn_df_du_t;

    /// @param ode        ODE model (held by const ref; must outlive this).
    /// @param h          fixed step size.
    /// @param newton_max max Newton iterations per step (acados default 3).
    /// @param newton_tol early-termination residual tolerance; 0 disables
    ///                   (fixed-iteration count, the acados default).
    explicit ImplicitRkIntegrator(const Ode& ode, double h, int newton_max = 3,
                                  double newton_tol = 0.0)
        : ode_(ode), h_(h), newton_max_(newton_max), newton_tol_(newton_tol)
    {
    }

    /// Discrete map x_next = Phi(x, u), obtained from the Newton solve.
    void value(const state_t& x, const control_t& u, state_t& x_next) const
    {
        solve_newton(x, u);
        x_next = x;
        for (int s = 0; s < NS; ++s)
        {
            if (Tab::b[s] != 0.0)
            {
                x_next += h_ * Tab::b[s] * ws_.K[s];
            }
        }
    }

    /// Composed-map Jacobian (df_dx, df_du) via the implicit first-order
    /// sensitivity: dK/d(x,u) = G^{-1} dR/d(x,u), then the b-weighted sum.
    void jacobian(const state_t& x, const control_t& u, df_dx_t& df_dx,
                  df_du_t& df_du) const
    {
        solve_newton(x, u);
        for (int s = 0; s < NS; ++s)
        {
            ws_.rhs_x.block(s * nx, 0, nx, nx) = ws_.dfdx[s];
            ws_.rhs_u.block(s * nx, 0, nx, nu) = ws_.dfdu[s];
        }
        const auto dK_dx = ws_.lu_.solve(ws_.rhs_x);  // NK x nx
        const auto dK_du = ws_.lu_.solve(ws_.rhs_u);  // NK x nu
        df_dx = df_dx_t::Identity();
        df_du.setZero();
        for (int s = 0; s < NS; ++s)
        {
            if (Tab::b[s] != 0.0)
            {
                df_dx += h_ * Tab::b[s] * dK_dx.block(s * nx, 0, nx, nx);
                df_du += h_ * Tab::b[s] * dK_du.block(s * nx, 0, nx, nu);
            }
        }
    }

    /// Bilinear Hessian-vector product of the composed map.
    ///
    /// Not implemented for the implicit path in phase 4d: the full-A
    /// self-coupling makes the second-order sensitivity depend on the full ODE
    /// Hessian, which the ODE `hess_prod` contract does not expose. A linear
    /// ODE (no `hess_prod`) has a zero composed Hessian and returns zeros; a
    /// nonlinear ODE fails to compile here (use ExplicitRkIntegrator instead).
    void hess_prod(const state_t& x, const control_t& u, const state_t& w,
                   const state_t& v_x, const control_t& v_u,
                   state_t& hv_x, control_t& hv_u) const
    {
        if constexpr (ode_supports_hess_prod<Ode, Dims>)
        {
            static_assert(
                false,
                "ImplicitRkIntegrator::hess_prod is not implemented in phase "
                "4d: the full-A stage coupling needs the full ODE Hessian, "
                "not just the ODE hess_prod. Use ExplicitRkIntegrator for a "
                "nonlinear dynamics HVP, or a linear ODE (zero Hessian).");
        }
        else
        {
            (void)x;
            (void)u;
            (void)w;
            (void)v_x;
            (void)v_u;
            hv_x.setZero();
            hv_u.setZero();
        }
    }

    /// ‖R‖∞ of the Newton residual at the last converged solve (diagnostic).
    double newton_residual_inf() const
    {
        return ws_.last_residual_;
    }

private:
    // Stage states from the current K:  x_s = x + h * sum_j A[s][j] K_j.
    void eval_stage_states(const state_t& x) const
    {
        for (int s = 0; s < NS; ++s)
        {
            state_t xs = x;
            for (int j = 0; j < NS; ++j)
            {
                if (Tab::A[s][j] != 0.0)
                {
                    xs += h_ * Tab::A[s][j] * ws_.K[j];
                }
            }
            ws_.x_stage[s] = xs;
        }
    }

    // ODE Jacobians at each stage state.
    void eval_jacobians(const control_t& u) const
    {
        for (int s = 0; s < NS; ++s)
        {
            ode_.jacobian(ws_.x_stage[s], u, ws_.dfdx[s], ws_.dfdu[s]);
        }
    }

    // Newton residual R_s = K_s - f(x_s, u).
    void eval_residual(const control_t& u) const
    {
        for (int s = 0; s < NS; ++s)
        {
            ws_.res.segment(s * nx, nx) = ws_.K[s] - ode_.f(ws_.x_stage[s], u);
        }
    }

    // Block Newton Jacobian: G[s][j] = delta_{s,j} I - h A[s][j] f_x(s).
    void eval_block_jacobian() const
    {
        ws_.G.setZero();
        for (int s = 0; s < NS; ++s)
        {
            for (int j = 0; j < NS; ++j)
            {
                if (Tab::A[s][j] == 0.0 && s != j)
                {
                    continue;
                }
                df_dx_t blk = -h_ * Tab::A[s][j] * ws_.dfdx[s];
                if (s == j)
                {
                    blk += df_dx_t::Identity();
                }
                ws_.G.block(s * nx, j * nx, nx, nx) = blk;
            }
        }
    }

    // Newton solve over the coupled stage system; leaves ws_.K converged and
    // ws_.lu_ factored (with dfdx/dfdu and last_residual_ refreshed) at the
    // converged point, ready for value() or jacobian().
    void solve_newton(const state_t& x, const control_t& u) const
    {
        const state_t K0 = ode_.f(x, u);
        for (int s = 0; s < NS; ++s)
        {
            ws_.K[s] = K0;
        }
        for (int it = 0; it < newton_max_; ++it)
        {
            eval_stage_states(x);
            eval_jacobians(u);
            eval_residual(u);
            eval_block_jacobian();
            ws_.lu_.compute(ws_.G);
            Eigen::Matrix<typename P::scalar_t, NK, 1> dK =
                ws_.lu_.solve(-ws_.res);
            for (int s = 0; s < NS; ++s)
            {
                ws_.K[s] += dK.segment(s * nx, nx);
            }
            if (newton_tol_ > 0.0 &&
                ws_.res.cwiseAbs().maxCoeff() < newton_tol_)
            {
                break;
            }
        }
        // Refresh the factorization and residual at the converged K so the
        // sensitivity solves and the diagnostic are consistent.
        eval_stage_states(x);
        eval_jacobians(u);
        eval_residual(u);
        ws_.last_residual_ = ws_.res.cwiseAbs().maxCoeff();
        eval_block_jacobian();
        ws_.lu_.compute(ws_.G);
    }

    const Ode& ode_;
    double h_;
    int newton_max_;
    double newton_tol_;
    mutable detail::IrkWorkspace<P, NS> ws_;
};

}  // namespace ocp

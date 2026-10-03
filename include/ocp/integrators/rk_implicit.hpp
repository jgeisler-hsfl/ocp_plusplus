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
// value     -- Newton solve, then x_next = x + h * sum_s b_s * K_s.
// jacobian  -- implicit first-order sensitivity: dK/d(x,u) = G^{-1} * dR/d(x,u),
//              then the same b-weighted accumulation as the explicit path.
// hess_prod -- second-order implicit sensitivity (4f): differentiates the
//              first-order system G . J = Q in the direction v.  The second-
//              order RHS R = B + h (H . (Am,0)) + h (H.v)_x Ja + h^2 (H.(Am,0))_x Ja
//              (four terms, all derived from the ODE Hessian H_s via hess_prod
//              with basis output weights e_i); M = G^{-1} R; HVP = sum_s h b_s (w^T M_s).
//              A linear ODE (no hess_prod) has a zero composed Hessian and returns zeros.
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
    static constexpr int nx  = P::nx;
    static constexpr int NK  = NS * P::nx;
    static constexpr int nIn = P::nx + P::nu;

    std::array<state_t, NS> K{};        // stage derivatives (Newton unknowns)
    std::array<state_t, NS> x_stage{};  // stage states x_s
    std::array<df_dx_t, NS> dfdx{};     // df/dx at stage s
    std::array<df_du_t, NS> dfdu{};     // df/du at stage s

    Eigen::Matrix<scalar_t, NK, NK> G{};   // block Newton Jacobian
    Eigen::PartialPivLU<Eigen::Matrix<scalar_t, NK, NK>> lu_;  // factored G
    Eigen::Matrix<scalar_t, NK, 1>  res{}; // Newton residual / correction
    Eigen::Matrix<scalar_t, NK, P::nx> rhs_x{};  // sensitivity RHS (dK/dx)
    Eigen::Matrix<scalar_t, NK, P::nu> rhs_u{};  // sensitivity RHS (dK/du)

    // Nonlinear hess_prod (4f): J = dK/dz (first-order sensitivity), R the
    // second-order RHS, M = G^{-1} R; per-stage ODE-HVP and A-weighted blocks.
    Eigen::Matrix<scalar_t, NK, nIn>  sens{};    // J = G^{-1} Q  (NK x nIn)
    Eigen::Matrix<scalar_t, NK, nIn>  hess_rhs{}; // R (accumulates B+T2+T3+T4)
    Eigen::Matrix<scalar_t, NK, nIn>  hess_sol{}; // M = G^{-1} R
    Eigen::Matrix<scalar_t, nx, nIn> hvp_v{};    // (H_s . v) at stage s (nx x nIn)
    Eigen::Matrix<scalar_t, nx, nIn> hvp_am{};   // (H_s . (Am_s,0)) at stage s
    Eigen::Matrix<scalar_t, nx, nIn> jac_am{};   // Ja_s = sum_j A[s][j] J[j-block]

    double last_residual_ = 0.0;  // ‖R‖∞ at the converged iterate (diagnostic)
};

}  // namespace detail

/// Implicit Runge-Kutta (collocation) integrator for a time-invariant ODE.
///
/// `NS` is the number of RK stages (2..4 for the Radau IIA tags). `Tag`
/// selects the Butcher scheme (RadauIia2Tag..RadauIia4Tag). `h` is the fixed
/// step size. The ODE model must provide `f` and `jacobian`; a nonlinear ODE
/// must also provide `hess_prod` (used by the 4f nonlinear HVP path).
template <class Dims, class Ode, int NS, class Tag>
class ImplicitRkIntegrator
{
    using P   = ocp::Problem<Dims>;
    using Tab = ocp::ButcherTableau<Dims, NS, Tag>;

    static constexpr int nx  = P::nx;
    static constexpr int nu  = P::nu;
    static constexpr int NK  = NS * nx;
    static constexpr int nIn = nx + nu;

public:
    using scalar_t  = typename P::scalar_t;
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

    /// Bilinear Hessian-vector product of the composed map,
    /// `hv = w^T d2Phi . v` (see SQP_PHASE4_PLAN.md, 4f).
    ///
    /// Nonlinear ODE: second-order implicit sensitivity via the block Newton
    /// Jacobian `G = dF/dK`.  With `J = G^{-1} Q` (first-order sensitivity)
    /// the per-stage block of the second-order RHS `R` is
    ///
    ///   R_s = H_s.v + h (H_s.(Am_s,0)) + h (H_s.v)_x Ja_s
    ///         + h^2 (H_s.(Am_s,0))_x Ja_s
    ///
    /// where `H_s` is the ODE Hessian at stage `s`, `Am_s = Ja_s . v` and
    /// `Ja_s = sum_j A[s][j] J[j-block]` (the A-weighted first-order
    /// sensitivities); all four terms arise from differentiating
    /// `G . J = Q` in the direction `v`.  The bilinear form `w^T d2Phi . v`
    /// is then `sum_s h b_s (w^T M_s)` with `M = G^{-1} R`.  The ODE's own
    /// `hess_prod` is used only with basis output weights `e_i` (the user's
    /// `w` enters solely in the final row contract).
    ///
    /// A linear ODE (no `hess_prod`) has a zero composed Hessian and returns
    /// zeros.
    void hess_prod(const state_t& x, const control_t& u, const state_t& w,
                   const state_t& v_x, const control_t& v_u,
                   state_t& hv_x, control_t& hv_u) const
    {
        if constexpr (ode_supports_hess_prod<Ode, Dims>)
        {
            using nIn_vec = Eigen::Matrix<scalar_t, nIn, 1>;

            solve_newton(x, u);

            // Full input direction v = (v_x, v_u) in R^{nIn}.
            const nIn_vec v = [this](const state_t& vx, const control_t& vu) {
                nIn_vec r;
                r.head(nx) = vx;
                r.tail(nu) = vu;
                return r;
            }(v_x, v_u);

            // First-order sensitivity J = G^{-1} Q, Q_s = [dfdx | dfdu].
            for (int s = 0; s < NS; ++s)
            {
                ws_.hess_rhs.block(s * nx, 0, nx, nx) = ws_.dfdx[s];
                ws_.hess_rhs.block(s * nx, nx, nx, nu) = ws_.dfdu[s];
            }
            ws_.sens = ws_.lu_.solve(ws_.hess_rhs);

            // Second-order RHS R, accumulated stage by stage.
            ws_.hess_rhs.setZero();
            const control_t zero_u = control_t::Zero();
            for (int s = 0; s < NS; ++s)
            {
                // Ja_s = sum_j A[s][j] J[j-block]   (nx x nIn).
                ws_.jac_am.setZero();
                for (int j = 0; j < NS; ++j)
                {
                    if (Tab::A[s][j] != 0.0)
                    {
                        ws_.jac_am += Tab::A[s][j] *
                            ws_.sens.block(j * nx, 0, nx, nIn);
                    }
                }
                // Am_s = Ja_s . v  (the A-weighted K-JVP x-block, nx).
                const Eigen::Matrix<scalar_t, nx, 1> am_s = ws_.jac_am * v;

                // ODE HVPs at stage s: (H_s . v) and (H_s . (Am_s, 0)),
                // one basis output weight e_i each.
                for (int i = 0; i < nx; ++i)
                {
                    const state_t e_i = state_t::Unit(i);
                    state_t hx;
                    control_t hu;
                    ode_hess_prod<Ode, Dims>(ode_, ws_.x_stage[s], u, e_i,
                                             v_x, v_u, hx, hu);
                    ws_.hvp_v.row(i).head(nx) = hx;
                    ws_.hvp_v.row(i).tail(nu) = hu;
                    ode_hess_prod<Ode, Dims>(ode_, ws_.x_stage[s], u, e_i,
                                             am_s, zero_u, hx, hu);
                    ws_.hvp_am.row(i).head(nx) = hx;
                    ws_.hvp_am.row(i).tail(nu) = hu;
                }

                // R_s = Hv + h HAm + h (Hv_x Ja_s) + h^2 (HAm_x Ja_s).
                ws_.hess_rhs.block(s * nx, 0, nx, nIn) =
                    ws_.hvp_v + h_ * ws_.hvp_am +
                    h_ * (ws_.hvp_v.leftCols(nx) * ws_.jac_am) +
                    h_ * h_ * (ws_.hvp_am.leftCols(nx) * ws_.jac_am);
            }

            // M = G^{-1} R  (nIn forward solves on the factored G), then the
            // b-weighted row contract (the user's w is used only here).
            ws_.hess_sol = ws_.lu_.solve(ws_.hess_rhs);
            hv_x.setZero();
            hv_u.setZero();
            for (int s = 0; s < NS; ++s)
            {
                if (Tab::b[s] == 0.0)
                {
                    continue;
                }
                const auto wms = w.transpose() *
                    ws_.hess_sol.block(s * nx, 0, nx, nIn);
                hv_x += h_ * Tab::b[s] * wms.head(nx);
                hv_u += h_ * Tab::b[s] * wms.tail(nu);
            }
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

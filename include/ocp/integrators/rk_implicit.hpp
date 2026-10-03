// ocp/integrators/rk_implicit.hpp
//
// Allocation-free implicit Runge-Kutta / collocation integrator (phases 4d,
// 4g, 4h).
//
// ImplicitRkIntegrator turns a (possibly time-varying) ODE
//  xdot = f(x, u, t)  into the discrete dynamics  x_next = Phi(x, u, t_k)
//  via a fixed implicit RK scheme
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

/// Pre-allocated scratch for one implicit RK step over NumSteps sub-steps.
/// The first block is the single-sub-step Newton scratch (the "current"
/// sub-step); the remaining arrays hold the per-sub-step trajectory (needed to
/// warm-start the next sub-step's Newton and to run the multi-step hess_prod
/// reverse sweep) and the reverse-mode duals (phase 4g).
template <class P, int NS, int NumSteps>
struct IrkWorkspace
{
    using scalar_t  = typename P::scalar_t;
    using state_t   = typename P::state_t;
    using control_t = typename P::control_t;
    using df_dx_t   = typename P::dyn_df_dx_t;
    using df_du_t   = typename P::dyn_df_du_t;
    static constexpr int nx  = P::nx;
    static constexpr int nu  = P::nu;
    static constexpr int NK  = NS * P::nx;
    static constexpr int nIn = P::nx + P::nu;

    using Gmat = Eigen::Matrix<scalar_t, NK, NK>;
    using LU   = Eigen::PartialPivLU<Gmat>;
    using Kvec = Eigen::Matrix<scalar_t, NK, 1>;

    // --- single-sub-step Newton scratch (the "current" sub-step) -----------
    std::array<state_t, NS> K{};        // stage derivatives (Newton unknowns)
    std::array<state_t, NS> x_stage{};  // stage states x_s
    std::array<df_dx_t, NS> dfdx{};     // df/dx at stage s
    std::array<df_du_t, NS> dfdu{};     // df/du at stage s

    Gmat   G{};                          // block Newton Jacobian
    LU     lu_;                          // factored G
    Kvec   res{};                        // Newton residual / correction
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

    // --- multi-step (4g) trajectory: one entry per sub-step ---------------
    // The multi-step hess_prod reuses the single-sub-step 4f HVP as a
    // building block: the forward pass stores each sub-step's Newton state
    // (stage states, Jacobians, factored G) plus the per-sub-step Jacobians
    // and the forward JVP of the boundary states; the backward pass threads
    // the co-state and the x-Hessian adjoint across the sub-step boundaries.
    std::array<std::array<state_t, NS>, NumSteps> x_stage_traj{};
    std::array<std::array<df_dx_t, NS>, NumSteps> dfdx_traj{};
    std::array<std::array<df_du_t, NS>, NumSteps> dfdu_traj{};
    std::array<LU, NumSteps> lu_traj{};          // per-sub-step factored G
    std::array<df_dx_t, NumSteps> A_traj{};      // per-sub-step x-Jacobian
    std::array<df_du_t, NumSteps> B_traj{};      // per-sub-step u-Jacobian
    std::array<state_t, NumSteps + 1> dx_traj{}; // JVP of the boundary states
    std::array<df_dx_t, NumSteps> R_traj{};      // (A_{ss-1}..A_0)^T threading
    // S_traj[ss] = D_u x_{ss}: total u-Jacobian of the composition of
    // sub-steps 0..ss-1 (forward-threaded: S_{ss+1} = A_ss S_ss + B_ss).
    std::array<df_du_t, NumSteps> S_traj{};
};

}  // namespace detail

/// Implicit Runge-Kutta (collocation) integrator for a (possibly
/// time-varying) ODE.
///
/// `NS` is the number of RK stages (2..4 for the Radau IIA tags). `Tag`
/// selects the Butcher scheme (RadauIia2Tag..RadauIia4Tag). `h` is the fixed
/// step size for the full OCP interval; `NumSteps` (phase 4g) is the number of
/// sub-intervals one OCP step of size `h` is divided into (h/NumSteps per
/// sub-step), the default of 1 being the plain single-step scheme. The ODE
/// model must provide `f` and `jacobian`; a nonlinear ODE must also provide
/// `hess_prod` (used by the nonlinear HVP path).
///
/// Time-varying (phase 4h): each method receives `t_k` (the physical time at
/// the start of the OCP interval); the per-stage time
///   t_s = t_k + h_ss * (ss + c[s])
/// is forwarded to every ODE evaluation (f, jacobian, hess_prod).
template <class Dims, class Ode, int NS, class Tag, int NumSteps = 1>
class ImplicitRkIntegrator
{
    using P   = ocp::Problem<Dims>;
    using Tab = ocp::ButcherTableau<Dims, NS, Tag>;

    static constexpr int nx  = P::nx;
    static constexpr int nu  = P::nu;
    static constexpr int NK  = NS * nx;
    static constexpr int nIn = nx + nu;
    using Gmat = Eigen::Matrix<typename P::scalar_t, NK, NK>;
    using Kvec = Eigen::Matrix<typename P::scalar_t, NK, 1>;

public:
    using scalar_t  = typename P::scalar_t;
    using state_t   = typename P::state_t;
    using control_t = typename P::control_t;
    using df_dx_t   = typename P::dyn_df_dx_t;
    using df_du_t   = typename P::dyn_df_du_t;

    /// @param ode        ODE model (held by const ref; must outlive this).
    /// @param h          fixed step size for the full OCP interval
    ///                   (h/NumSteps per sub-step).
    /// @param newton_max max Newton iterations per sub-step (acados default 3).
    /// @param newton_tol early-termination residual tolerance; 0 disables
    ///                   (fixed-iteration count, the acados default).
    explicit ImplicitRkIntegrator(const Ode& ode, double h, int newton_max = 3,
                                  double newton_tol = 0.0)
        : ode_(ode), h_(h), h_ss_(h / (double)NumSteps),
          newton_max_(newton_max), newton_tol_(newton_tol)
    {
    }

    /// Discrete map x_next = Phi(x, u) for the full OCP interval, advanced in
    /// NumSteps sub-steps of size h/NumSteps. Each sub-step is a Newton solve,
    /// warm-started from the previous sub-step's converged K (acados
    /// `mem->xdot` pattern).
    void value(const state_t& x, const control_t& u, double t_k,
               state_t& x_next) const
    {
        state_t x_cur = x;
        Kvec K_init;
        bool first = true;
        for (int ss = 0; ss < NumSteps; ++ss)
        {
            if (first)
            {
                const state_t K0 =
                    ode_.f(x_cur, u, t_k + ss * h_ss_);
                for (int s = 0; s < NS; ++s)
                {
                    K_init.segment(s * nx, nx) = K0;
                }
                first = false;
            }
            else
            {
                for (int s = 0; s < NS; ++s)
                {
                    K_init.segment(s * nx, nx) = ws_.K[s];
                }
            }
            solve_newton(x_cur, u, K_init, t_k, ss);
            for (int s = 0; s < NS; ++s)
            {
                if (Tab::b[s] != 0.0)
                {
                    x_cur += h_ss_ * Tab::b[s] * ws_.K[s];
                }
            }
        }
        x_next = x_cur;
    }

    /// Composed-map Jacobian (df_dx, df_du): per sub-step, the implicit
    /// first-order sensitivity dK/d(x,u) = G^{-1} dR/d(x,u) gives the
    /// per-sub-step Jacobian (Jx_ss, Ju_ss); the composed Jacobian threads
    /// Sx = Jx_ss*Sx, Su = Jx_ss*Su + Ju_ss across the sub-steps.
    void jacobian(const state_t& x, const control_t& u, double t_k,
                  df_dx_t& df_dx, df_du_t& df_du) const
    {
        state_t x_cur = x;
        df_dx_t Sx = df_dx_t::Identity();
        df_du_t Su;
        Su.setZero();
        Kvec K_init;
        bool first = true;
        for (int ss = 0; ss < NumSteps; ++ss)
        {
            if (first)
            {
                const state_t K0 = ode_.f(x_cur, u, t_k + ss * h_ss_);
                for (int s = 0; s < NS; ++s)
                {
                    K_init.segment(s * nx, nx) = K0;
                }
                first = false;
            }
            else
            {
                for (int s = 0; s < NS; ++s)
                {
                    K_init.segment(s * nx, nx) = ws_.K[s];
                }
            }
            solve_newton(x_cur, u, K_init, t_k, ss);
            for (int s = 0; s < NS; ++s)
            {
                ws_.rhs_x.block(s * nx, 0, nx, nx) = ws_.dfdx[s];
                ws_.rhs_u.block(s * nx, 0, nx, nu) = ws_.dfdu[s];
            }
            const auto dK_dx = ws_.lu_.solve(ws_.rhs_x);  // NK x nx
            const auto dK_du = ws_.lu_.solve(ws_.rhs_u);  // NK x nu
            df_dx_t Jx_ss = df_dx_t::Identity();
            df_du_t Ju_ss;
            Ju_ss.setZero();
            for (int s = 0; s < NS; ++s)
            {
                if (Tab::b[s] != 0.0)
                {
                    Jx_ss += h_ss_ * Tab::b[s] * dK_dx.block(s * nx, 0, nx, nx);
                    Ju_ss += h_ss_ * Tab::b[s] * dK_du.block(s * nx, 0, nx, nu);
                    x_cur += h_ss_ * Tab::b[s] * ws_.K[s];
                }
            }
            Sx = Jx_ss * Sx;
            Su = Jx_ss * Su + Ju_ss;
        }
        df_dx = Sx;
        df_du = Su;
    }

    /// Bilinear Hessian-vector product of the composed map,
    /// `hv = w^T d2Phi . v` (see SQP_PHASE4_PLAN.md, 4f/4g).
    ///
    /// NumSteps == 1: the single-sub-step 4f algorithm (bit-for-bit identical
    /// to the phase-4f implementation).
    ///
    /// NumSteps > 1: the forward pass Newton-solves each sub-step (warm-
    /// started from the previous sub-step's K), stores the per-sub-step Newton
    /// state and per-sub-step Jacobians, and threads the first-order JVP (dx)
    /// and the x-Hessian adjoint (R) forward. The backward pass threads the
    /// co-state (lambda) and, per sub-step, re-runs the 4f HVP with the
    /// stored Newton state, accumulating the contributions.
    ///
    /// A linear ODE (no `hess_prod`) has a zero composed Hessian and returns
    /// zeros.
    void hess_prod(const state_t& x, const control_t& u, double t_k,
                   const state_t& w, const state_t& v_x, const control_t& v_u,
                   state_t& hv_x, control_t& hv_u) const
    {
        if constexpr (ode_supports_hess_prod<Ode, Dims>)
        {
            if constexpr (NumSteps == 1)
            {
                Kvec K_init;
                const state_t K0 = ode_.f(x, u, t_k);
                for (int s = 0; s < NS; ++s)
                {
                    K_init.segment(s * nx, nx) = K0;
                }
                solve_newton(x, u, K_init, t_k, 0);
                hess_prod_substep(u, t_k, 0, w, v_x, v_u, hv_x, hv_u);
            }
            else
            {
                // Forward: Newton-solve each sub-step, store state, thread
                // the JVP (dx) and the x-Hessian adjoint (R) forward.
                state_t x_cur = x;
                state_t dx_cur = v_x;
                ws_.R_traj[0] = df_dx_t::Identity();
                ws_.S_traj[0].setZero();
                ws_.dx_traj[0] = v_x;
                Kvec K_init;
                bool first = true;
                for (int ss = 0; ss < NumSteps; ++ss)
                {
                    if (first)
                    {
                        const state_t K0 =
                            ode_.f(x_cur, u, t_k + ss * h_ss_);
                        for (int s = 0; s < NS; ++s)
                        {
                            K_init.segment(s * nx, nx) = K0;
                        }
                        first = false;
                    }
                    else
                    {
                        for (int s = 0; s < NS; ++s)
                        {
                            K_init.segment(s * nx, nx) = ws_.K[s];
                        }
                    }
                    solve_newton(x_cur, u, K_init, t_k, ss);

                    // Store per-sub-step Newton state.
                    for (int s = 0; s < NS; ++s)
                    {
                        ws_.x_stage_traj[ss][s] = ws_.x_stage[s];
                        ws_.dfdx_traj[ss][s] = ws_.dfdx[s];
                        ws_.dfdu_traj[ss][s] = ws_.dfdu[s];
                    }
                    ws_.lu_traj[ss] = ws_.lu_;

                    // Per-sub-step Jacobians (A_ss, B_ss) and state advance.
                    for (int s = 0; s < NS; ++s)
                    {
                        ws_.rhs_x.block(s * nx, 0, nx, nx) = ws_.dfdx[s];
                        ws_.rhs_u.block(s * nx, 0, nx, nu) = ws_.dfdu[s];
                    }
                    const auto dK_dx = ws_.lu_.solve(ws_.rhs_x);
                    const auto dK_du = ws_.lu_.solve(ws_.rhs_u);
                    ws_.A_traj[ss] = df_dx_t::Identity();
                    ws_.B_traj[ss].setZero();
                    for (int s = 0; s < NS; ++s)
                    {
                        if (Tab::b[s] != 0.0)
                        {
                            ws_.A_traj[ss] +=
                                h_ss_ * Tab::b[s] *
                                dK_dx.block(s * nx, 0, nx, nx);
                            ws_.B_traj[ss] +=
                                h_ss_ * Tab::b[s] *
                                dK_du.block(s * nx, 0, nx, nu);
                            x_cur += h_ss_ * Tab::b[s] * ws_.K[s];
                        }
                    }

                    // Thread R (x-Hessian adjoint) and S (total u-Jacobian of
                    // the composition of sub-steps 0..ss-1) forward.
                    if (ss + 1 < NumSteps)
                    {
                        ws_.R_traj[ss + 1] =
                            ws_.A_traj[ss].transpose() * ws_.R_traj[ss];
                        ws_.S_traj[ss + 1] =
                            ws_.A_traj[ss] * ws_.S_traj[ss] + ws_.B_traj[ss];
                    }

                    // Thread the JVP forward: dx_{ss+1} = dx_ss + h*sum b*dK.
                    for (int s = 0; s < NS; ++s)
                    {
                        ws_.res.segment(s * nx, nx) =
                            ws_.dfdx[s] * dx_cur + ws_.dfdu[s] * v_u;
                    }
                    const Kvec dK_jvp = ws_.lu_.solve(ws_.res);
                    state_t dx_next = dx_cur;
                    for (int s = 0; s < NS; ++s)
                    {
                        if (Tab::b[s] != 0.0)
                        {
                            dx_next += h_ss_ * Tab::b[s] *
                                dK_jvp.segment(s * nx, nx);
                        }
                    }
                    ws_.dx_traj[ss + 1] = dx_next;
                    dx_cur = dx_next;
                }

                // Backward: thread the co-state, compute local HVPs.
                hv_x.setZero();
                hv_u.setZero();
                state_t lam = w;  // co-state on x_{NumSteps}
                for (int ss = NumSteps - 1; ss >= 0; --ss)
                {
                    // Restore sub-step ss's Newton state.
                    for (int s = 0; s < NS; ++s)
                    {
                        ws_.x_stage[s] = ws_.x_stage_traj[ss][s];
                        ws_.dfdx[s] = ws_.dfdx_traj[ss][s];
                        ws_.dfdu[s] = ws_.dfdu_traj[ss][s];
                    }
                    ws_.lu_ = ws_.lu_traj[ss];

                    // Local HVP: lam^T D2Phi_ss (dx_traj[ss], v_u).
                    state_t local_hv_x;
                    control_t local_hv_u;
                    hess_prod_substep(u, t_k, ss, lam, ws_.dx_traj[ss], v_u,
                                      local_hv_x, local_hv_u);

                    // Accumulate: x-part threaded via R_ss; u-part gets the
                    // B-comp chain (S_ss^T applied to the x-part) plus the
                    // local u-HVP.
                    hv_x += ws_.R_traj[ss] * local_hv_x;
                    hv_u += ws_.S_traj[ss].transpose() * local_hv_x + local_hv_u;

                    // Thread the co-state back: lam = A_ss^T * lam.
                    lam = ws_.A_traj[ss].transpose() * lam;
                }
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
    // Stage states from the current K:  x_s = x + h_ss * sum_j A[s][j] K_j.
    void eval_stage_states(const state_t& x) const
    {
        for (int s = 0; s < NS; ++s)
        {
            state_t xs = x;
            for (int j = 0; j < NS; ++j)
            {
                if (Tab::A[s][j] != 0.0)
                {
                    xs += h_ss_ * Tab::A[s][j] * ws_.K[j];
                }
            }
            ws_.x_stage[s] = xs;
        }
    }

    // ODE Jacobians at each stage state.
    void eval_jacobians(const control_t& u, double t_k, int ss) const
    {
        for (int s = 0; s < NS; ++s)
        {
            const double t_s = t_k + (ss + Tab::c[s]) * h_ss_;
            ode_.jacobian(ws_.x_stage[s], u, t_s, ws_.dfdx[s], ws_.dfdu[s]);
        }
    }

    // Newton residual R_s = K_s - f(x_s, u).
    void eval_residual(const control_t& u, double t_k, int ss) const
    {
        for (int s = 0; s < NS; ++s)
        {
            const double t_s = t_k + (ss + Tab::c[s]) * h_ss_;
            ws_.res.segment(s * nx, nx) =
                ws_.K[s] - ode_.f(ws_.x_stage[s], u, t_s);
        }
    }

    // Block Newton Jacobian: G[s][j] = delta_{s,j} I - h_ss A[s][j] f_x(s).
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
                df_dx_t blk = -h_ss_ * Tab::A[s][j] * ws_.dfdx[s];
                if (s == j)
                {
                    blk += df_dx_t::Identity();
                }
                ws_.G.block(s * nx, j * nx, nx, nx) = blk;
            }
        }
    }

    // Newton solve over the coupled stage system of size h_ss, starting from
    // K_init (warm-start; the f(x,u) broadcast for the first sub-step, or the
    // previous sub-step's converged K). Leaves ws_.K converged and ws_.lu_
    // factored (with dfdx/dfdu and last_residual_ refreshed) at the converged
    // point, ready for value(), jacobian(), or the hess_prod forward sweep.
    void solve_newton(const state_t& x, const control_t& u,
                      const Kvec& K_init, double t_k, int ss) const
    {
        for (int s = 0; s < NS; ++s)
        {
            ws_.K[s] = K_init.segment(s * nx, nx);
        }
        for (int it = 0; it < newton_max_; ++it)
        {
            eval_stage_states(x);
            eval_jacobians(u, t_k, ss);
            eval_residual(u, t_k, ss);
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
        eval_jacobians(u, t_k, ss);
        eval_residual(u, t_k, ss);
        ws_.last_residual_ = ws_.res.cwiseAbs().maxCoeff();
        eval_block_jacobian();
        ws_.lu_.compute(ws_.G);
    }

    // Single-sub-step HVP (4f algorithm): assumes the Newton state
    // (ws_.x_stage, ws_.dfdx, ws_.dfdu, ws_.lu_) is set up for the current
    // sub-step. Computes w^T D2Phi_ss (v_x, v_u) = (hv_x, hv_u).
    void hess_prod_substep(const control_t& u, double t_k, int ss,
                           const state_t& w, const state_t& v_x,
                           const control_t& v_u, state_t& hv_x,
                           control_t& hv_u) const
    {
        using nIn_vec = Eigen::Matrix<scalar_t, nIn, 1>;

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

            // ODE HVPs at stage s: (H_s . v) and (H_s . (Am_s, 0)).
            const double t_s = t_k + (ss + Tab::c[s]) * h_ss_;
            for (int i = 0; i < nx; ++i)
            {
                const state_t e_i = state_t::Unit(i);
                state_t hx;
                control_t hu;
                ode_hess_prod<Ode, Dims>(ode_, ws_.x_stage[s], u, t_s, e_i,
                                         v_x, v_u, hx, hu);
                ws_.hvp_v.row(i).head(nx) = hx;
                ws_.hvp_v.row(i).tail(nu) = hu;
                ode_hess_prod<Ode, Dims>(ode_, ws_.x_stage[s], u, t_s, e_i,
                                         am_s, zero_u, hx, hu);
                ws_.hvp_am.row(i).head(nx) = hx;
                ws_.hvp_am.row(i).tail(nu) = hu;
            }

            // R_s = Hv + h HAm + h (Hv_x Ja_s) + h^2 (HAm_x Ja_s).
            ws_.hess_rhs.block(s * nx, 0, nx, nIn) =
                ws_.hvp_v + h_ss_ * ws_.hvp_am +
                h_ss_ * (ws_.hvp_v.leftCols(nx) * ws_.jac_am) +
                h_ss_ * h_ss_ * (ws_.hvp_am.leftCols(nx) * ws_.jac_am);
        }

        // M = G^{-1} R, then the b-weighted row contract.
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
            hv_x += h_ss_ * Tab::b[s] * wms.head(nx);
            hv_u += h_ss_ * Tab::b[s] * wms.tail(nu);
        }
    }

    const Ode& ode_;
    double h_;
    double h_ss_;
    int newton_max_;
    double newton_tol_;
    mutable detail::IrkWorkspace<P, NS, NumSteps> ws_;
};

}  // namespace ocp

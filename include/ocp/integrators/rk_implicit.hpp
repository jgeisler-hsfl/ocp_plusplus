// ocp/integrators/rk_implicit.hpp
//
// Allocation-free implicit Runge-Kutta / collocation integrator in DAE
// (implicit) form (phases 4d, 4g, 4h, 4i; DAE-native since 4k,
// CLEAN_UP_TODO item 3).
//
// ImplicitRkIntegrator turns the DAE residual
//
//     F(x, xdot, u, t) = 0
//
// (duck type: dae_model.hpp — the same contract the generated CasADi
// dae_fun / dae_fun_jac code exposes) into the discrete dynamics
//  x_next = Phi(x, u, t_k)  via a fixed implicit RK scheme
// (ButcherTableau<Dims, NS, Tag>, Radau IIA or Gauss-Legendre).
//
// Stage equations (sub-step size h_ss = h / NumSteps, Butcher A, b, c):
//
//     x_s = x + h_ss * sum_j A[s][j] K_j,        s = 0..NS-1
//     F(x_s, K_s, u, t_s) = 0,   t_s = t_k + h_ss (ss + c[s])
//     x_next = x + h_ss * sum_s b_s K_s
//
// with the stage derivatives K_s as the unknowns.  Because the Butcher
// matrix A is full (not strictly lower triangular), the stage equations
// form a coupled nonlinear system in the NS*nx unknowns K_0..K_NS-1.  Each
// sub-step solves it by Newton iteration with a single block LU
// factorization of the NS*nx-by-NS*nx Jacobian (matches acados
// sim_irk_integrator.c: the block matrix is factored in place and reused
// for the correction and, on the sensitivity path, for the first-order
// solves):
//
//     G[s][j] = delta_{s,j} F_xdot_s + h_ss A[s][j] F_x_s
//
// An ODE written in residual form (F = xdot - f, F_xdot = I) is a
// specialization; the formulation is exact for any (regular) F_xdot, not
// just the diagonal ODE case.
//
// value     -- Newton solve, then x_next = x + h_ss * sum_s b_s K_s.
// jacobian  -- implicit first-order sensitivity:  G K' = -dR/dz, then the
//              same b-weighted accumulation as the explicit path.
// value_jac -- x_next + Jacobian from a single Newton sweep (NumSteps == 1
//              reuses the forward cache; NumSteps > 1 extracts both from one
//              multi-step sweep instead of two).
// hess_prod -- second-order implicit sensitivity: differentiates
//              G K' = -dR/dz in direction v.  The stage-s second-order RHS
//              is  b_s = -[Mx | Mu] - Mxdot K'_s - h sum_j A[s][j] Mx K'_j,
//              where Mx/Mxdot/Mu are the stage-s mixed Hessian blocks of F
//              w.r.t. z = (x, xdot, u) with the second slot contracted with
//              the chain-rule stage direction (see hess_prod_substep);
//              M = G^{-1} b; HVP = sum_s h b_s (w^T M_s).  A DAE without
//              F_hess_prod (in particular all generated adapters) has a
//              zero composed Hessian and returns zeros.
//
// Warm start: the first sub-step initializes each stage from the small
// per-stage linear solve  F_xdot(x, 0, u, t_s) K0_s = -F(x, 0, u, t_s)
// (for an ODE residual this is exactly K0_s = f(x, u, t_s)); later
// sub-steps warm-start from the previous sub-step's converged K (acados
// `mem->xdot` pattern).
//
// Forward cache (phase 4i): the v-independent forward state is keyed on
// (x, u, t_k).  NumSteps == 1: the converged Newton state (K, stage
// Jacobians, factored G) is reused by a second value / jacobian / value_jac /
// hess_prod call with the same key, and the first-order sensitivity
// sens = G^{-1} Q is computed once per key.  NumSteps > 1: the whole
// multi-step Newton trajectory (stage states, K, Jacobians, factored G,
// per-sub-step Jacobians, R/S threading, per-sub-step sens) is cached; a
// second hess_prod call with the same key re-runs only the v-dependent JVP
// threading and the backward sweep.  value() / jacobian() / value_jac()
// invalidate the multi-step trajectory cache.  The cache is
// per-integrator-instance.
//
// No AD fusion: the stage solve is Newton-based, so unlike the explicit
// integrator there is no dual-Newton path; supports_value_jac_hess_prod is
// false and ContinuousProblem falls back to value_jac + hess_prod per
// Hessian column (which hits the cache above).
//
// Allocation-free: every scratch buffer is a fixed-size Eigen object or
// std::array, sized once on NS and Dims::nx/nu.  Methods are const; the DAE
// model is stored by value; the per-call scratch is `mutable`.
//
// Newton options mirror acados: newton_max (opts->newton_iter, default 3)
// and newton_tol (opts->newton_tol, default 0.0 = fixed-iteration count, no
// early termination).

#pragma once

#include "ocp/integrators/butcher.hpp"
#include "ocp/integrators/dae_model.hpp"

#include <array>

#include <Eigen/Dense>

namespace ocp
{

namespace detail
{

/// Pre-allocated scratch for one implicit RK step over NumSteps sub-steps.
/// The first block is the single-sub-step Newton scratch (the "current"
/// sub-step); the remaining arrays hold the per-sub-step trajectory (needed
/// to warm-start the next sub-step's Newton and to run the multi-step
/// hess_prod reverse sweep) and the reverse-mode duals (phase 4g).
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

    using Gmat  = Eigen::Matrix<scalar_t, NK, NK>;
    using LU    = Eigen::PartialPivLU<Gmat>;
    using Kvec  = Eigen::Matrix<scalar_t, NK, 1>;
    using MatNx = Eigen::Matrix<scalar_t, nx, nx>;

    // --- single-sub-step Newton scratch (the "current" sub-step) -----------
    std::array<state_t, NS> K{};        // stage derivatives (Newton unknowns)
    std::array<state_t, NS> x_stage{};  // stage states x_s
    std::array<df_dx_t, NS> dFdx{};     // dF/dx at stage s
    std::array<df_dx_t, NS> dFxdot{};   // dF/dxdot at stage s
    std::array<df_du_t, NS> dFdu{};     // dF/du at stage s

    Gmat   G{};                          // block Newton Jacobian
    LU     lu_;                          // factored G
    Kvec   res{};                        // Newton residual / correction
    Eigen::Matrix<scalar_t, NK, P::nx> rhs_x{};  // sensitivity RHS (-dR/dx)
    Eigen::Matrix<scalar_t, NK, P::nu> rhs_u{};  // sensitivity RHS (-dR/du)

    // Warm-start scratch: per-stage solve  F_xdot K0 = -F(x, 0, u).
    MatNx                   kinit_A{};
    Eigen::PartialPivLU<MatNx> kinit_lu_;

    // Nonlinear hess_prod: J = dK/dz (first-order sensitivity), R the
    // second-order RHS, M = G^{-1} R; per-stage mixed-Hessian blocks and
    // A-weighted first-order blocks.
    Eigen::Matrix<scalar_t, NK, nIn>  sens{};     // J = G^{-1} Q  (NK x nIn)
    Eigen::Matrix<scalar_t, NK, nIn>  hess_rhs{}; // b (accumulates b_s)
    Eigen::Matrix<scalar_t, NK, nIn>  hess_sol{}; // M = G^{-1} b
    // hvp_v.row(i)  = [ (Mx_s)_irow | (Mu_s)_irow ]  (nx x nIn)
    // hvp_am.row(i) = (Mxdot_s)_irow (leftCols(nx) used)
    // jac_am        = Ja_s = sum_j A[s][j] J_j-block  (nx x nIn)
    Eigen::Matrix<scalar_t, nx, nIn> hvp_v{};
    Eigen::Matrix<scalar_t, nx, nIn> hvp_am{};
    Eigen::Matrix<scalar_t, nx, nIn> jac_am{};

    double last_residual_ = 0.0;  // ‖F‖∞ at the converged iterate (diagnostic)

    // --- multi-step (4g) trajectory: one entry per sub-step ---------------
    // The multi-step hess_prod reuses the single-sub-step HVP as a building
    // block: the forward pass stores each sub-step's Newton state (stage
    // states, K, Jacobians, factored G) plus the per-sub-step Jacobians and
    // the forward JVP of the boundary states; the backward pass threads the
    // co-state and the x-Hessian adjoint across the sub-step boundaries.
    std::array<std::array<state_t, NS>, NumSteps> x_stage_traj{};
    std::array<std::array<state_t, NS>, NumSteps> K_traj{};
    std::array<std::array<df_dx_t, NS>, NumSteps> dFdx_traj{};
    std::array<std::array<df_dx_t, NS>, NumSteps> dFxdot_traj{};
    std::array<std::array<df_du_t, NS>, NumSteps> dFdu_traj{};
    std::array<LU, NumSteps> lu_traj{};          // per-sub-step factored G
    std::array<df_dx_t, NumSteps> A_traj{};      // per-sub-step x-Jacobian
    std::array<df_du_t, NumSteps> B_traj{};      // per-sub-step u-Jacobian
    std::array<state_t, NumSteps + 1> dx_traj{}; // JVP of the boundary states
    std::array<df_dx_t, NumSteps> R_traj{};      // (A_{ss-1}..A_0)^T threading
    // S_traj[ss] = D_u x_{ss}: total u-Jacobian of the composition of
    // sub-steps 0..ss-1 (forward-threaded: S_{ss+1} = A_ss S_ss + B_ss).
    std::array<df_du_t, NumSteps> S_traj{};
    // Phase 4i: per-sub-step first-order sensitivity G^{-1} Q (v-independent,
    // cached by the multi-step forward, consumed by the backward sweep).
    std::array<Eigen::Matrix<scalar_t, NK, nIn>, NumSteps> sens_traj{};

    // --- forward cache (phase 4i): the v-independent forward state above is
    //     valid for the (x, u, t_k) recorded here.
    state_t   last_x{};
    control_t last_u{};
    double    last_t = 0.0;
    bool fwd_valid  = false;  // NumSteps==1: Newton state matches the key
    bool sens_valid = false;  // NumSteps==1: sens = G^{-1} Q valid for the key
    bool traj_valid = false;  // NumSteps>1: traj arrays + R/S valid for the key
};

}  // namespace detail

/// Implicit Runge-Kutta (collocation) integrator for a (possibly
/// time-varying) DAE.
///
/// `NS` is the number of RK stages (2..4 for the Radau IIA / Gauss-
/// Legendre tags).  `Tag` selects the Butcher scheme (RadauIia2Tag..
/// RadauIia4Tag, GaussLegendre2Tag..GaussLegendre4Tag).  `h` is the fixed
/// step size for the full OCP interval; `NumSteps` (phase 4g) is the number
/// of sub-intervals one OCP step of size `h` is divided into (h/NumSteps per
/// sub-step), the default of 1 being the plain single-step scheme.  The DAE
/// model must provide `F` and `F_jac` (dae_model.hpp); a nonlinear DAE
/// must also provide `F_hess_prod` (used by the nonlinear HVP path).
///
/// Time-varying (phase 4h): each method receives `t_k` (the physical time at
/// the start of the OCP interval); the per-stage time
///   t_s = t_k + h_ss * (ss + c[s])
/// is forwarded to every DAE evaluation (F, F_jac, F_hess_prod).
///
/// No dual-scalar stage solve: the stage equations are Newton-solved at
/// scalar_t, so the fused HVP entry is not available
/// (`supports_value_jac_hess_prod == false`); the forward cache is the
/// efficiency mechanism for repeated HVP calls with the same (x, u, t_k).
template <class Dims, class Dae, int NS, class Tag, int NumSteps = 1>
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

    /// Always false: the implicit stage solve is Newton-based (no dual-
    /// Newton), so the composed-map value + Jacobian + HVP cannot be produced
    /// by a single forward pass; ContinuousProblem dispatches to value_jac +
    /// per-column hess_prod (which reuse the forward cache).
    static constexpr bool supports_value_jac_hess_prod = false;

    /// @param dae        DAE model (stored by value; safe with temporary
    ///                   OdeToDae wrappers).
    /// @param h          fixed step size for the full OCP interval
    ///                   (h/NumSteps per sub-step).
    /// @param newton_max max Newton iterations per sub-step (acados default 3).
    /// @param newton_tol early-termination residual tolerance; 0 disables
    ///                   (fixed-iteration count, the acados default).
    explicit ImplicitRkIntegrator(const Dae& dae, double h, int newton_max = 3,
                                  double newton_tol = 0.0)
        : dae_(dae), h_(h), h_ss_(h / (double)NumSteps),
          newton_max_(newton_max), newton_tol_(newton_tol)
    {
    }

    /// Discrete map x_next = Phi(x, u) for the full OCP interval, advanced in
    /// NumSteps sub-steps of size h/NumSteps.  Each sub-step is a Newton
    /// solve, warm-started from the previous sub-step's converged K (acados
    /// `mem->xdot` pattern).
    void value(const state_t& x, const control_t& u, double t_k,
               state_t& x_next) const
    {
        if constexpr (NumSteps == 1)
        {
            if (!fwd_hit(x, u, t_k))
            {
                newton_single(x, u, t_k);
                set_fwd_cache(x, u, t_k);
            }
            x_next_from_cached_K(x, x_next);
        }
        else
        {
            ws_.traj_valid = false;
            multi_step_sweep(x, u, t_k, x_next, nullptr, nullptr);
        }
    }

    /// x_next + composed-map Jacobian (df_dx, df_du) from a single sweep:
    /// NumSteps == 1 reuses the forward cache when present, NumSteps > 1
    /// extracts both from one multi-step sweep.
    void value_jac(const state_t& x, const control_t& u, double t_k,
                   state_t& x_next, df_dx_t& df_dx, df_du_t& df_du) const
    {
        if constexpr (NumSteps == 1)
        {
            if (!fwd_hit(x, u, t_k))
            {
                newton_single(x, u, t_k);
                set_fwd_cache(x, u, t_k);
            }
            state_t x_cur = x;
            df_dx_t Sx = df_dx_t::Identity();
            df_du_t Su;
            Su.setZero();
            substep_jac(0, x_cur, Sx, Su);
            x_next = x_cur;
            df_dx  = Sx;
            df_du  = Su;
        }
        else
        {
            ws_.traj_valid = false;
            multi_step_sweep(x, u, t_k, x_next, &df_dx, &df_du);
        }
    }

    /// Composed-map Jacobian (df_dx, df_du): per sub-step, the implicit
    /// first-order sensitivity K' = G^{-1} (-dR/dz) gives the per-sub-step
    /// Jacobian (Jx_ss, Ju_ss); the composed Jacobian threads
    /// Sx = Jx_ss*Sx, Su = Jx_ss*Su + Ju_ss across the sub-steps.
    /// (Same single sweep as value_jac; x_next is discarded.)
    void jacobian(const state_t& x, const control_t& u, double t_k,
                  df_dx_t& df_dx, df_du_t& df_du) const
    {
        state_t x_next;
        value_jac(x, u, t_k, x_next, df_dx, df_du);
    }

    /// x_next + Jacobian + HVP, as a composite of value_jac + hess_prod (no
    /// dual-Newton stage solve, so this is not a single fused pass; the
    /// forward cache lets hess_prod reuse value_jac's Newton state).  Provided
    /// so both integrators expose a uniform public API.
    void value_jac_hess_prod(const state_t& x, const control_t& u, double t_k,
                             state_t& x_next, df_dx_t& df_dx, df_du_t& df_du,
                             const state_t& w, const state_t& v_x,
                             const control_t& v_u, state_t& hv_x,
                             control_t& hv_u) const
    {
        value_jac(x, u, t_k, x_next, df_dx, df_du);
        hess_prod(x, u, t_k, w, v_x, v_u, hv_x, hv_u);
    }

    /// Bilinear Hessian-vector product of the composed map,
    /// `hv = w^T d2Phi . v` (see SQP_PHASE4_PLAN.md, 4f/4g).
    ///
    /// NumSteps == 1: the single-sub-step algorithm; the Newton forward and
    /// the v-independent sensitivity sens = G^{-1} Q are taken from the
    /// forward cache when the key (x, u, t_k) matches; the v-dependent b
    /// assembly, G^{-1} b solve and b-contraction always run.
    ///
    /// NumSteps > 1: the forward pass Newton-solves each sub-step (warm-
    /// started from the previous sub-step's K), stores the per-sub-step Newton
    /// state, Jacobians and sens (cached for the key); the v-dependent JVP
    /// threading (dx) and the backward sweep thread the co-state and the
    /// x-Hessian adjoint (R) across the sub-step boundaries and re-run per
    /// direction.
    ///
    /// A DAE without `F_hess_prod` (in particular the generated adapters)
    /// has a zero composed Hessian and returns zeros.
    void hess_prod(const state_t& x, const control_t& u, double t_k,
                   const state_t& w, const state_t& v_x, const control_t& v_u,
                   state_t& hv_x, control_t& hv_u) const
    {
        if constexpr (dae_supports_hess_prod<Dae, Dims>)
        {
            if constexpr (NumSteps == 1)
            {
                if (!fwd_hit(x, u, t_k))
                {
                    newton_single(x, u, t_k);
                    set_fwd_cache(x, u, t_k);
                }
                if (!ws_.sens_valid)
                {
                    fill_first_order_rhs();
                    ws_.sens = ws_.lu_.solve(ws_.hess_rhs);
                    ws_.sens_valid = true;
                }
                hess_prod_substep(u, t_k, 0, w, v_x, v_u, hv_x, hv_u);
            }
            else
            {
                if (!traj_hit(x, u, t_k))
                {
                    forward_traj(x, u, t_k);
                }
                thread_jvp_traj(v_x, v_u);
                backward_traj(w, v_u, hv_x, hv_u);
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

    /// ‖F(x_s, K_s, u)‖∞ of the Newton residual at the last converged solve
    /// (diagnostic).
    double newton_residual_inf() const
    {
        return ws_.last_residual_;
    }

private:
    // ---- forward cache (phase 4i) -----------------------------------------

    /// True when the cached forward state was produced for exactly this
    /// (x, u, t_k) (bit-identical inputs).
    bool key_match(const state_t& x, const control_t& u, double t_k) const
    {
        return ws_.last_t == t_k
            && (x - ws_.last_x).squaredNorm() == 0.0
            && (u - ws_.last_u).squaredNorm() == 0.0;
    }

    bool fwd_hit(const state_t& x, const control_t& u, double t_k) const
    {
        return ws_.fwd_valid && key_match(x, u, t_k);
    }

    bool traj_hit(const state_t& x, const control_t& u, double t_k) const
    {
        return ws_.traj_valid && key_match(x, u, t_k);
    }

    void set_fwd_cache(const state_t& x, const control_t& u, double t_k) const
    {
        ws_.last_x = x;
        ws_.last_u = u;
        ws_.last_t = t_k;
        ws_.fwd_valid  = true;
        ws_.sens_valid = false;
    }

    // x_next = x + h_ss * sum_s b_s K_s from the cached converged K
    // (NumSteps == 1, where h_ss == h and ws_.K is the only sub-step).
    void x_next_from_cached_K(const state_t& x, state_t& x_next) const
    {
        x_next = x;
        for (int s = 0; s < NS; ++s)
        {
            if (Tab::b[s] != 0.0)
            {
                x_next += h_ss_ * Tab::b[s] * ws_.K[s];
            }
        }
    }

    // Single-sub-step Newton sweep (NumSteps == 1): warm-start the K init,
    // solve; leaves ws_.K / dFdx / dFxdot / dFdu / lu_ converged at (x, u).
    void newton_single(const state_t& x, const control_t& u, double t_k) const
    {
        Kvec K_init;
        warm_start(0, x, u, t_k, K_init);
        solve_newton(x, u, K_init, t_k, 0);
    }

    // Warm-start the K init of sub-step ss: the per-stage linear solve
    // F_xdot(x, 0, u, t_s) K0_s = -F(x, 0, u, t_s) for the first sub-step
    // (an ODE residual reduces this to K0_s = f(x, u, t_s)), the previous
    // sub-step's converged K otherwise.
    void warm_start(int ss, const state_t& x_cur, const control_t& u,
                    double t_k, Kvec& K_init) const
    {
        if (ss == 0)
        {
            const state_t zero_xdot = state_t::Zero();
            for (int s = 0; s < NS; ++s)
            {
                const double t_s = t_k + (ss + Tab::c[s]) * h_ss_;
                const state_t F0 = dae_.F(x_cur, zero_xdot, u, t_s);
                df_dx_t dFdx0, dFxdot0;
                df_du_t dFdu0;
                dae_.F_jac(x_cur, zero_xdot, u, t_s, dFdx0, dFxdot0, dFdu0);
                ws_.kinit_lu_.compute(dFxdot0);
                K_init.segment(s * nx, nx) = ws_.kinit_lu_.solve(-F0);
            }
        }
        else
        {
            for (int s = 0; s < NS; ++s)
            {
                K_init.segment(s * nx, nx) = ws_.K[s];
            }
        }
    }

    // Fill ws_.rhs_x / rhs_u / hess_rhs with -dR/dz = [stack(-dFdx) |
    // stack(-dFdu)] (the first-order RHS; also the sensitivity RHS used by
    // substep_jac's rhs_x/rhs_u).
    void fill_first_order_rhs() const
    {
        for (int s = 0; s < NS; ++s)
        {
            ws_.rhs_x.block(s * nx, 0, nx, nx) = -ws_.dFdx[s];
            ws_.rhs_u.block(s * nx, 0, nx, nu) = -ws_.dFdu[s];
            ws_.hess_rhs.block(s * nx, 0, nx, nx) = -ws_.dFdx[s];
            ws_.hess_rhs.block(s * nx, nx, nx, nu) = -ws_.dFdu[s];
        }
    }

    // Post-Newton per-sub-step Jacobian body: K' = G^{-1} (-dR/dz),
    // b-weighted accumulation into (Jx_ss, Ju_ss), state advance, and the
    // Sx/Su thread update.  Requires the current sub-step's Newton state
    // (ws_.dFdx / dFdu / lu_ / K) to be set up (converged, from the cache or
    // the live sweep).
    void substep_jac(int ss, state_t& x_cur, df_dx_t& Sx, df_du_t& Su) const
    {
        fill_first_order_rhs();
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
        (void)ss;
    }

    // NumSteps > 1 Newton sweep over all sub-steps.  When df_dx / df_du are
    // non-null, the per-sub-step Jacobian body runs as well (value_jac);
    // otherwise only the state advances (value).
    void multi_step_sweep(const state_t& x, const control_t& u, double t_k,
                          state_t& x_next, df_dx_t* df_dx, df_du_t* df_du) const
    {
        state_t x_cur = x;
        df_dx_t Sx = df_dx_t::Identity();
        df_du_t Su;
        Su.setZero();
        Kvec K_init;
        for (int ss = 0; ss < NumSteps; ++ss)
        {
            warm_start(ss, x_cur, u, t_k, K_init);
            solve_newton(x_cur, u, K_init, t_k, ss);
            if (df_dx != nullptr)
            {
                substep_jac(ss, x_cur, Sx, Su);
            }
            else
            {
                for (int s = 0; s < NS; ++s)
                {
                    if (Tab::b[s] != 0.0)
                    {
                        x_cur += h_ss_ * Tab::b[s] * ws_.K[s];
                    }
                }
            }
        }
        x_next = x_cur;
        if (df_dx != nullptr)
        {
            *df_dx = Sx;
            *df_du = Su;
        }
    }

    // ---- multi-step (NumSteps > 1) hess_prod decomposition (phase 4g/4i) ----

    // v-independent multi-step forward: Newton-solve each sub-step, store the
    // per-sub-step Newton state, per-sub-step Jacobians, R/S threading and
    // per-sub-step sens = G^{-1} Q; fills the trajectory cache.
    void forward_traj(const state_t& x, const control_t& u, double t_k) const
    {
        state_t x_cur = x;
        ws_.R_traj[0] = df_dx_t::Identity();
        ws_.S_traj[0].setZero();
        Kvec K_init;
        for (int ss = 0; ss < NumSteps; ++ss)
        {
            warm_start(ss, x_cur, u, t_k, K_init);
            solve_newton(x_cur, u, K_init, t_k, ss);

            // Store per-sub-step Newton state.
            for (int s = 0; s < NS; ++s)
            {
                ws_.x_stage_traj[ss][s] = ws_.x_stage[s];
                ws_.K_traj[ss][s] = ws_.K[s];
                ws_.dFdx_traj[ss][s] = ws_.dFdx[s];
                ws_.dFxdot_traj[ss][s] = ws_.dFxdot[s];
                ws_.dFdu_traj[ss][s] = ws_.dFdu[s];
            }
            ws_.lu_traj[ss] = ws_.lu_;

            // Per-sub-step Jacobians (A_ss, B_ss) and state advance.
            for (int s = 0; s < NS; ++s)
            {
                ws_.rhs_x.block(s * nx, 0, nx, nx) = -ws_.dFdx[s];
                ws_.rhs_u.block(s * nx, 0, nx, nu) = -ws_.dFdu[s];
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

            // Thread R (x-Hessian adjoint) and S (total u-Jacobian of the
            // composition of sub-steps 0..ss-1) forward.
            if (ss + 1 < NumSteps)
            {
                ws_.R_traj[ss + 1] =
                    ws_.A_traj[ss].transpose() * ws_.R_traj[ss];
                ws_.S_traj[ss + 1] =
                    ws_.A_traj[ss] * ws_.S_traj[ss] + ws_.B_traj[ss];
            }

            // Per-sub-step first-order sensitivity (v-independent).
            ws_.hess_rhs.setZero();
            for (int s = 0; s < NS; ++s)
            {
                ws_.hess_rhs.block(s * nx, 0, nx, nx) = -ws_.dFdx[s];
                ws_.hess_rhs.block(s * nx, nx, nx, nu) = -ws_.dFdu[s];
            }
            ws_.sens_traj[ss] = ws_.lu_.solve(ws_.hess_rhs);
        }
        ws_.last_x = x;
        ws_.last_u = u;
        ws_.last_t = t_k;
        ws_.traj_valid = true;
    }

    // v-dependent JVP threading along the cached trajectory:
    // dx_{ss+1} = dx_ss + h_ss * sum_s b_s dK_s with dK = G^{-1}(-dR/dz .
    // (dx, v_u)).  Requires forward_traj for the same key.
    void thread_jvp_traj(const state_t& v_x, const control_t& v_u) const
    {
        state_t dx_cur = v_x;
        ws_.dx_traj[0] = v_x;
        for (int ss = 0; ss < NumSteps; ++ss)
        {
            // Restore sub-step ss's Jacobians and factored G.
            for (int s = 0; s < NS; ++s)
            {
                ws_.dFdx[s] = ws_.dFdx_traj[ss][s];
                ws_.dFdu[s] = ws_.dFdu_traj[ss][s];
            }
            ws_.lu_ = ws_.lu_traj[ss];

            // Thread the JVP forward: dx_{ss+1} = dx_ss + h*sum b*dK.
            for (int s = 0; s < NS; ++s)
            {
                ws_.res.segment(s * nx, nx) =
                    -(ws_.dFdx[s] * dx_cur + ws_.dFdu[s] * v_u);
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
    }

    // Backward sweep over the cached trajectory: thread the co-state, compute
    // the local per-sub-step HVPs with the stored Newton state and sens,
    // accumulate.  Requires thread_jvp_traj for the same key and direction.
    void backward_traj(const state_t& w, const control_t& v_u,
                       state_t& hv_x, control_t& hv_u) const
    {
        hv_x.setZero();
        hv_u.setZero();
        state_t lam = w;  // co-state on x_{NumSteps}
        for (int ss = NumSteps - 1; ss >= 0; --ss)
        {
            // Restore sub-step ss's Newton state.
            for (int s = 0; s < NS; ++s)
            {
                ws_.x_stage[s] = ws_.x_stage_traj[ss][s];
                ws_.K[s] = ws_.K_traj[ss][s];
                ws_.dFdx[s] = ws_.dFdx_traj[ss][s];
                ws_.dFxdot[s] = ws_.dFxdot_traj[ss][s];
                ws_.dFdu[s] = ws_.dFdu_traj[ss][s];
            }
            ws_.lu_ = ws_.lu_traj[ss];
            ws_.sens = ws_.sens_traj[ss];

            // Local HVP: lam^T D2Phi_ss (dx_traj[ss], v_u).  The cached key
            // (last_u, last_t) is the (u, t_k) the trajectory was built for.
            state_t local_hv_x;
            control_t local_hv_u;
            hess_prod_substep(ws_.last_u, ws_.last_t, ss, lam,
                              ws_.dx_traj[ss], v_u, local_hv_x, local_hv_u);

            // Accumulate: x-part threaded via R_ss; u-part gets the
            // B-comp chain (S_ss^T applied to the x-part) plus the
            // local u-HVP.
            hv_x += ws_.R_traj[ss] * local_hv_x;
            hv_u += ws_.S_traj[ss].transpose() * local_hv_x + local_hv_u;

            // Thread the co-state back: lam = A_ss^T * lam.
            lam = ws_.A_traj[ss].transpose() * lam;
        }
    }

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

    // DAE Jacobian blocks at each stage point (x_s, K_s).
    void eval_jacobians(const control_t& u, double t_k, int ss) const
    {
        for (int s = 0; s < NS; ++s)
        {
            const double t_s = t_k + (ss + Tab::c[s]) * h_ss_;
            dae_.F_jac(ws_.x_stage[s], ws_.K[s], u, t_s, ws_.dFdx[s],
                       ws_.dFxdot[s], ws_.dFdu[s]);
        }
    }

    // Newton residual R_s = F(x_s, K_s, u, t_s).
    void eval_residual(const control_t& u, double t_k, int ss) const
    {
        for (int s = 0; s < NS; ++s)
        {
            const double t_s = t_k + (ss + Tab::c[s]) * h_ss_;
            ws_.res.segment(s * nx, nx) =
                dae_.F(ws_.x_stage[s], ws_.K[s], u, t_s);
        }
    }

    // Block Newton Jacobian: G[s][j] = delta_{s,j} F_xdot_s + h_ss A[s][j]
    // F_x_s.
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
                df_dx_t blk =
                    h_ss_ * Tab::A[s][j] * ws_.dFdx[s];
                if (s == j)
                {
                    blk += ws_.dFxdot[s];
                }
                ws_.G.block(s * nx, j * nx, nx, nx) = blk;
            }
        }
    }

    // Newton solve over the coupled stage system of size h_ss, starting from
    // K_init (warm-start; the per-stage F_xdot solve for the first sub-step,
    // or the previous sub-step's converged K).  Leaves ws_.K converged and
    // ws_.lu_ factored (with dFdx/dFxdot/dFdu and last_residual_ refreshed)
    // at the converged point, ready for value(), jacobian(), or the
    // hess_prod forward sweep.
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
            Kvec dK = ws_.lu_.solve(-ws_.res);
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

    // Single-sub-step HVP: assumes the Newton state (ws_.x_stage, ws_.K,
    // ws_.lu_) is set up for the current sub-step and that ws_.sens =
    // G^{-1} Q is filled (NumSteps == 1: the lazy sens_valid cache;
    // NumSteps > 1: restored from sens_traj by backward_traj).  Computes
    // w^T D2Phi_ss (v_x, v_u) = (hv_x, hv_u).
    //
    // The stage-s second-order RHS is
    //   b_s = -[Mx | Mu] - Mxdot K'_s - h sum_j A[s][j] Mx K'_j,
    // where Mx / Mxdot / Mu are the mixed Hessian blocks of F w.r.t.
    // z = (x, xdot, u) (rows: d2F_i/dx_j..., d2F_i/dxdot_j..., d2F_i/du_j...)
    // with the second slot contracted with the chain-rule stage direction
    //   (dx_s, dK_s, v_u) = (v_x + h sum_j A[s][j] (K'_z v)_j,  (K'_z v)_s,
    // v_u).
    // For an ODE residual (F = xdot - f) this reduces to the 4-term
    // expression Hv + h HAm + h (Hv_x) Ja + h^2 (HAm_x) Ja (up to the sign
    // of G, which is identical in both forms).
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

        // Second-order RHS b, accumulated stage by stage.
        ws_.hess_rhs.setZero();
        for (int s = 0; s < NS; ++s)
        {
            // Ja_s = sum_j A[s][j] J_j-block   (nx x nIn), J = K'_z.
            ws_.jac_am.setZero();
            for (int j = 0; j < NS; ++j)
            {
                if (Tab::A[s][j] != 0.0)
                {
                    ws_.jac_am += Tab::A[s][j] *
                        ws_.sens.block(j * nx, 0, nx, nIn);
                }
            }
            // Chain-rule stage direction at stage s.
            const state_t v_K_s =
                ws_.sens.block(s * nx, 0, nx, nIn) * v;
            const state_t dx_s = v_x + h_ss_ * (ws_.jac_am * v);

            // Mixed Hessian at stage s via unit output weights:
            // hvp_v.row(i)  = [ (Mx_s)_irow | (Mu_s)_irow ]
            // hvp_am.row(i) = (Mxdot_s)_irow (leftCols(nx))
            const double t_s = t_k + (ss + Tab::c[s]) * h_ss_;
            for (int i = 0; i < nx; ++i)
            {
                const state_t e_i = state_t::Unit(i);
                state_t hx;
                state_t hxd;
                control_t hu;
                dae_hess_prod<Dae, Dims>(dae_, ws_.x_stage[s], ws_.K[s], u,
                                         t_s, e_i, dx_s, v_K_s, v_u, hx, hxd,
                                         hu);
                ws_.hvp_v.row(i).head(nx) = hx;
                ws_.hvp_v.row(i).tail(nu) = hu;
                ws_.hvp_am.row(i).head(nx) = hxd;
            }

            const auto Mx    = ws_.hvp_v.leftCols(nx);
            const auto Mxdot = ws_.hvp_am.leftCols(nx);
            Eigen::Matrix<scalar_t, nx, nIn> b_s = -ws_.hvp_v;
            b_s -= Mxdot * ws_.sens.block(s * nx, 0, nx, nIn);
            for (int j = 0; j < NS; ++j)
            {
                if (Tab::A[s][j] != 0.0)
                {
                    b_s -= h_ss_ * Tab::A[s][j] *
                        Mx * ws_.sens.block(j * nx, 0, nx, nIn);
                }
            }
            ws_.hess_rhs.block(s * nx, 0, nx, nIn) = b_s;
        }

        // M = G^{-1} b, then the b-weighted row contract.
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

    Dae dae_;
    double h_;
    double h_ss_;
    int newton_max_;
    double newton_tol_;
    mutable detail::IrkWorkspace<P, NS, NumSteps> ws_;
};

}  // namespace ocp

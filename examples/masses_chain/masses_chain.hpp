// masses_chain.hpp
//
// ocp++ concrete problem for the acados masses_chain OCP
// (acados_matlab_octave/masses_chain_model, codegen 21ef639b).
//
//   state  x = [ p1 v1 ; p2 v2 ; p3 v3 ; p4 v4 ]  (p_i, v_i in R^3),  nx = 24
//   control u = force on the last mass (R^3),       nu = 3
//   horizon  N = 40,  T = 8.0,  dt = T/N = 0.2
//
//   dynamics: nonlinear spring-chain ODE, integrated with the acados-matching
//             implicit-RK collocation scheme (Gauss-Legendre, 4 stages,
//             2 sub-steps per shooting interval).  The ODE right-hand side and
//             its Jacobian are the CasADi-generated residual
//             masses_chain_impl_dae_fun (see masses_chain_capi.hpp); the
//             Jacobian is obtained by central finite differences (phase-1).
//
//   stage cost    L_k(x,u) = 0.5 [ 10 ||x - x_ref||^2 + 1e-2 ||u||^2 ]
//   terminal cost L_N(x)    = 0.5 * 10 ||x - x_ref||^2
//   state box (stages 1..N-1, py of each mass):  -0.01 <= py_i <= 1e4
//   control box (all stages):                    -1     <= u    <= 1
//   initial state: fixed (x_0 = the hanging rest configuration)
//
// Deviations from the generated acados solver (benign, verified by
// tests/sqp_masses_chain):
//   * acados carries 24 terminal state-box rows (idxbxe = all 24) with default
//     +/-inf bounds, i.e. fully inactive.  They are dropped here
//     (terminal_state_box_idx empty); inactive rows do not affect the KKT
//     point.
//   * Dynamics Jacobian.  The acados reference uses the CasADi analytic
//     Jacobian; ocp++ uses central finite differences (h = 1e-6) of the
//     CasADi residual.  The QP Hessian model is identical on both sides:
//     Gauss-Newton, cost Hessian only (has_dynamics_hess = false;
//     acados hessian_approx = GAUSS_NEWTON adds no dynamics Hessian term —
//     see tests/sqp_masses_chain check (E) and
//     docs/plans/finished/GN_HESSIAN_PLAN.md).  On this weakly-convex
//     OCP (control weight 1e-2, nonlinear spring-chain dynamics) the tiny
//     Jacobian differences are amplified by the ill-conditioned KKT system
//     into a bounded trajectory difference (|du| up to ~8e-3, |dx| up to
//     ~1e-3); not a porting error (the dynamics map itself is exact: the
//     ocp++ collocation map reproduces acados's x_{k+1} from (x_k, u_k) to
//     1e-13, and the cost matches to 4e-7).

#pragma once

#include "ocp/integrators/continuous_problem.hpp"
#include "ocp/integrators/rk_implicit.hpp"

#include "masses_chain_capi.hpp"

#include <Eigen/Dense>

namespace ocp
{

// =========================================================================
//  Dims
// =========================================================================

struct MassesChainDims
{
    static constexpr int nx = 24;
    static constexpr int nu = 3;
    static constexpr int ng = 0;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 0;
    static constexpr int nl_t = 0;

    static constexpr bool fixed_initial_state = true;
    static constexpr bool has_dynamics_hess = false;  // GN: cost Hessian only
    static constexpr bool has_constr_hess = false;    // only (linear) box rows

    // py of mass i lives at state index 6*(i-1) + 1  ->  {1, 7, 13, 19}
    static constexpr std::array<int, 4> state_box_idx = {1, 7, 13, 19};
    static constexpr std::array<int, 3> control_box_idx = {0, 1, 2};
    // acados terminal box (24 rows, +/-inf) is inactive -> dropped
    static constexpr std::array<int, 0> terminal_state_box_idx = {};

    static constexpr std::array<int, 0> ineq_soft_idx = {};
    static constexpr std::array<int, 0> eq_soft_idx = {};
    static constexpr std::array<int, 0> lin_soft_idx = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {};
    static constexpr std::array<int, 0> state_box_soft_idx = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

// =========================================================================
//  ODE model (explicit RHS + finite-difference Jacobian from the residual)
// =========================================================================

struct MassesChainOde
{
    using P = ocp::Problem<MassesChainDims>;
    using state_t = typename P::state_t;
    using control_t = typename P::control_t;
    using dyn_df_dx_t = typename P::dyn_df_dx_t;
    using dyn_df_du_t = typename P::dyn_df_du_t;

    double h_ = 0.2;  // OCP step (forwarded to the C residual as p; unused)

    // xdot = f(x, u, t): the residual with xdot = 0 equals f_expl(x, u).
    state_t f(const state_t& x, const control_t& u, double /*t*/) const
    {
        state_t F;
        state_t xdot = state_t::Zero();
        detail::masses_chain_residual(x.data(), xdot.data(), u.data(), h_,
                                      F.data());
        return F;
    }

    // Central finite differences of f(x, u) w.r.t. x and u.
    void jacobian(const state_t& x, const control_t& u, double t,
                  dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
    {
        const double h = 1.0e-6;
        for (int j = 0; j < P::nx; ++j)
        {
            state_t xp = x, xm = x;
            xp(j) += h;
            xm(j) -= h;
            df_dx.col(j) = (f(xp, u, t) - f(xm, u, t)) / (2.0 * h);
        }
        for (int j = 0; j < P::nu; ++j)
        {
            control_t up = u, um = u;
            up(j) += h;
            um(j) -= h;
            df_du.col(j) = (f(x, up, t) - f(x, um, t)) / (2.0 * h);
        }
    }

    // No hess_prod: a nonlinear ODE whose composed-Hessian HVP the phase-1
    // wrapper does not supply (the solver therefore uses the Gauss-Newton /
    // cost-only QP Hessian via has_dynamics_hess = false).
};

// =========================================================================
//  Concrete problem (ContinuousProblem: integrator supplies the dynamics)
// =========================================================================

using MassesChainInteg =
    ocp::ImplicitRkIntegrator<MassesChainDims, MassesChainOde, 4,
                              ocp::GaussLegendre4Tag, 2>;

class MassesChain
    : public ocp::ContinuousProblem<MassesChainDims, MassesChainOde,
                                    MassesChainInteg>
{
    using Base =
        ocp::ContinuousProblem<MassesChainDims, MassesChainOde, MassesChainInteg>;

public:
    MassesChain() : Base(MassesChainOde{}, /*h=*/0.2)
    {
        // x_ref: the "spread" rest configuration (acados yref / x_ref).
        x_ref_ << 0.2437127723791751, 0.0, -0.4707556301562673, 0.0, 0.0, 0.0,
                  0.5, 0.0, -0.6357704467945128, 0.0, 0.0, 0.0,
                  0.7562872276208249, 0.0, -0.4707556301562673, 0.0, 0.0, 0.0,
                  1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        // x_0: the "hanging" rest configuration (acados lbx0/ubx0).
        x0_ << 0.0, 0.3740500601637552, -0.344560900973575, 0.0, 0.0, 0.0,
               0.0, 0.7567239589410515, -0.3750878206748779, 0.0, 0.0, 0.0,
               0.0, 1.132755245684027, -0.08869592957430722, 0.0, 0.0, 0.0,
               0.0, 1.5, 0.5, 0.0, 0.0, 0.0;
    }

    // ---------------------------------------------------------------
    // initial state (fixed)
    // ---------------------------------------------------------------
    state_t initial_state() const
    {
        return x0_;
    }

    // The cost / terminal reference (the "spread" configuration); exposed so
    // a driver can warm-start the interior states with it (as the acados
    // reference does with init_x = repmat(x_ref)).
    const state_t& cost_reference() const
    {
        return x_ref_;
    }

    // ---------------------------------------------------------------
    // stage cost  L = 0.5 [ 10 ||x - x_ref||^2 + 1e-2 ||u - 0||^2 ]
    // ---------------------------------------------------------------
    double stage_cost_value(int /*k*/, const state_t& x,
                            const control_t& u) const
    {
        const auto dx = x - x_ref_;
        return 0.5 * (10.0 * dx.dot(dx) + 1.0e-2 * u.dot(u));
    }

    void stage_cost_value_grad(int /*k*/, const state_t& x,
                               const control_t& u, double& value,
                               stage_grad_t& grad) const
    {
        value = 0.5 * (10.0 * (x - x_ref_).squaredNorm()
                       + 1.0e-2 * u.squaredNorm());
        grad.head<MassesChainDims::nx>() = 10.0 * (x - x_ref_);
        grad.tail<MassesChainDims::nu>() = 1.0e-2 * u;
    }

    void stage_cost_value_grad_hess(int /*k*/, const state_t& x,
                                    const control_t& u, double& value,
                                    stage_grad_t& grad,
                                    stage_hess_t& hess) const
    {
        stage_cost_value_grad(0, x, u, value, grad);
        hess.setZero();
        hess.block(0, 0, MassesChainDims::nx, MassesChainDims::nx) =
            10.0 * Eigen::Matrix<double, MassesChainDims::nx,
                                 MassesChainDims::nx>::Identity();
        hess.block(MassesChainDims::nx, MassesChainDims::nx,
                   MassesChainDims::nu, MassesChainDims::nu) =
            1.0e-2 * Eigen::Matrix<double, MassesChainDims::nu,
                                   MassesChainDims::nu>::Identity();
    }

    // ---------------------------------------------------------------
    // terminal cost  L_N = 0.5 * 10 ||x - x_ref||^2
    // ---------------------------------------------------------------
    double terminal_cost_value(const state_t& x) const
    {
        const auto dx = x - x_ref_;
        return 0.5 * 10.0 * dx.dot(dx);
    }

    void terminal_cost_value_grad(const state_t& x, double& value,
                                  term_grad_t& grad) const
    {
        value = 0.5 * 10.0 * (x - x_ref_).squaredNorm();
        grad = 10.0 * (x - x_ref_);
    }

    void terminal_cost_value_grad_hess(const state_t& x, double& value,
                                       term_grad_t& grad,
                                       term_hess_t& hess) const
    {
        terminal_cost_value_grad(x, value, grad);
        hess = 10.0 * term_hess_t::Identity();
    }

    // ---------------------------------------------------------------
    // box constraints (the only active groups; ng = ne = nl = 0)
    // ---------------------------------------------------------------
    state_box_t stage_state_box_constr(int /*k*/) const
    {
        state_box_t spec;
        spec.lo.setConstant(-0.01);
        spec.hi.setConstant(1.0e4);
        spec.soft_penalty.setZero();
        return spec;
    }

    control_box_t stage_control_box_constr(int /*k*/) const
    {
        control_box_t spec;
        spec.lo.setConstant(-1.0);
        spec.hi.setConstant(1.0);
        spec.soft_penalty.setZero();
        return spec;
    }

private:
    state_t x_ref_{};  // cost reference (spread configuration)
    state_t x0_{};     // fixed initial state (hanging configuration)
};

}  // namespace ocp

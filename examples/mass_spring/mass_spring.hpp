// Example concrete problem for the ocp++ problem interface:
// the 4-mass spring chain of acados/examples/acados_python/generic_dyn_disc/.
//
//   x = [q; v]  (q: 4 positions, v: 4 velocities),  u: 3 forces
//
//   continuous time (sampling time Ts = 0.5):
//        q_dot = v
//        v_dot = T q + Bc u,   T = tridiag(1, -3, 1),  Bc = [I3; 0]
//
//   discrete time:  x_{k+1} = A x_k + B u_k,
//        A = expm(Ts Ac),  B = (A - I) Ac^{-1} Bc,  Ac = [0  I; T  0]
//
//   stage cost:    0.5 * (x'x + 2 u'u)
//   terminal cost: 0.5 * x_N' x_N
//
//   box (stages 0..N-1):  -0.5 <= u_i <= 0.5,  -4 <= x_i <= 4
//   terminal box:         -4 <= x_N,i <= 4
//   initial state:        x_0 = [2.5, 2.5, 0, 0, 0, 0, 0, 0]'
//
// All box rows are active: state_box_idx = {0..7}, control_box_idx = {0, 1, 2},
// terminal_state_box_idx = {0..7}. No nonlinear or linear constraints
// (ng = ne = nl = ng_t = ne_t = nl_t = 0) and no soft rows, so the
// corresponding interface functions stay unimplemented on purpose.
//
// A and B are derived from the continuous-time system exactly as in
// generic_disc_dyn.c (mass_spring_system + disc_dyn_fun), the generic
// dynamics of the acados example (its default run), so this problem matches
// the acados reference for later solution comparisons. Note: main.py's
// *symbolic* variant uses a -2 diagonal in T; the generic C dynamics use -3.

#pragma once

#include "ocp/problem.hpp"

#include <unsupported/Eigen/MatrixFunctions>

struct MassSpringDims
{
    static constexpr int nx = 8;
    static constexpr int nu = 3;
    static constexpr int ng = 0;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 0;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = true;  // x_0 = [2.5, 2.5, 0, ...]'
    static constexpr bool has_dynamics_hess = true;  // HVP implemented (zero; linear map)
    static constexpr bool has_constr_hess   = true;  // no nonlinear constraints (vacuously true)
    static constexpr std::array<int, 8> state_box_idx   = {0, 1, 2, 3, 4, 5, 6, 7};
    static constexpr std::array<int, 3> control_box_idx = {0, 1, 2};
    static constexpr std::array<int, 8> terminal_state_box_idx = {0, 1, 2, 3, 4, 5, 6, 7};
    // no soft constraint rows: all `*_soft_idx` sets are empty
    static constexpr std::array<int, 0> ineq_soft_idx   = {};
    static constexpr std::array<int, 0> eq_soft_idx     = {};
    static constexpr std::array<int, 0> lin_soft_idx    = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx   = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx  = {};
    static constexpr std::array<int, 0> state_box_soft_idx   = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

class MassSpring : public ocp::Problem<MassSpringDims>
{
public:
    static constexpr int nmass = nx / 2;  // 4 masses

    // user-managed problem parameters (not part of the interface)
    double Ts_ = 0.5;  // sampling time
    Eigen::Matrix<double, nmass, nmass> T_{};  // coupling tridiag(1, -3, 1)
    dyn_df_dx_t A_{};  // discrete state matrix
    dyn_df_du_t B_{};  // discrete input matrix

    MassSpring()
    {
        // continuous-time system  Ac = [0  I; T  0],  Bc = [0; I3]
        T_.setZero();
        for (int i = 0; i < nmass; ++i)
        {
            T_(i, i) = -3.0;
            if (i + 1 < nmass)
            {
                T_(i, i + 1) = 1.0;
                T_(i + 1, i) = 1.0;
            }
        }
        dyn_df_dx_t Ac = dyn_df_dx_t::Zero();
        for (int i = 0; i < nmass; ++i)
        {
            Ac(i, nmass + i) = 1.0;  // q_dot = v
        }
        Ac.block(nmass, 0, nmass, nmass) = T_;
        dyn_df_du_t Bc = dyn_df_du_t::Zero();
        for (int i = 0; i < nu; ++i)
        {
            Bc(nmass + i, i) = 1.0;  // forces act on the first 3 masses
        }

        const dyn_df_dx_t E = (Ts_ * Ac).exp();
        A_ = E;
        // top-right block of expm([Ts Ac, Ts Bc; 0, 0]); E and Ac commute, so
        // this equals the generic_disc_dyn.c computation Ac^{-1} (A - I) Bc
        B_ = (E - dyn_df_dx_t::Identity()) * Ac.inverse() * Bc;
    }

    // ---------------------------------------------------------------
    // initial state
    // ---------------------------------------------------------------
    state_t initial_state() const
    {
        state_t x0;
        x0.setZero();
        x0(0) = 2.5;
        x0(1) = 2.5;
        return x0;
    }

    // ---------------------------------------------------------------
    // dynamics
    // ---------------------------------------------------------------
    state_t dynamics_next_state(int, const state_t& x, const control_t& u) const
    {
        return A_ * x + B_ * u;
    }

    void dynamics_value_jac(int, const state_t& x, const control_t& u,
                            state_t& x_next,
                            dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
    {
        x_next = A_ * x + B_ * u;
        df_dx = A_;
        df_du = B_;
    }

    void dynamics_value_jac_hess(int k, const state_t& x, const control_t& u,
                                 const state_t&, state_t& x_next,
                                 dyn_df_dx_t& df_dx, dyn_df_du_t& df_du,
                                 dyn_hess_t& hess) const
    {
        // linear dynamics map: Hessian is zero
        dynamics_value_jac(k, x, u, x_next, df_dx, df_du);
        hess.setZero();
    }

    // ---------------------------------------------------------------
    // stage cost  (zero reference, weights dWu = 2*1, dWx = 1)
    // ---------------------------------------------------------------
    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return 0.5 * (x.dot(x) + 2.0 * u.dot(u));
    }

    void stage_cost_value_grad(int, const state_t& x, const control_t& u,
                               double& value, stage_grad_t& g) const
    {
        value = 0.5 * (x.dot(x) + 2.0 * u.dot(u));
        g.head<nx>() = x;
        g.tail<nu>() = 2.0 * u;
    }

    void stage_cost_value_grad_hess(int, const state_t& x, const control_t& u,
                                    double& value, stage_grad_t& g,
                                    stage_hess_t& H) const
    {
        stage_cost_value_grad(0, x, u, value, g);
        H.setZero();
        H.block(0, 0, nx, nx) = dyn_df_dx_t::Identity();
        H.block(nx, nx, nu, nu) = 2.0 * Eigen::Matrix<double, nu, nu>::Identity();
    }

    // ---------------------------------------------------------------
    // terminal cost  (zero reference, weights dWx = 1)
    // ---------------------------------------------------------------
    double terminal_cost_value(const state_t& x) const
    {
        return 0.5 * x.dot(x);
    }

    void terminal_cost_value_grad(const state_t& x, double& value,
                                  term_grad_t& g) const
    {
        value = 0.5 * x.dot(x);
        g = x;
    }

    void terminal_cost_value_grad_hess(const state_t& x, double& value,
                                       term_grad_t& g, term_hess_t& H) const
    {
        terminal_cost_value_grad(x, value, g);
        H = term_hess_t::Identity();
    }

    // ---------------------------------------------------------------
    // box constraints  (the only active constraint groups)
    // ---------------------------------------------------------------
    state_box_t stage_state_box_constr(int) const
    {
        state_box_t spec;
        spec.lo.setConstant(-4.0);
        spec.hi.setConstant(4.0);
        spec.soft_penalty.setZero();
        return spec;
    }

    control_box_t stage_control_box_constr(int) const
    {
        control_box_t spec;
        spec.lo.setConstant(-0.5);
        spec.hi.setConstant(0.5);
        spec.soft_penalty.setZero();
        return spec;
    }

    term_state_box_t terminal_state_box_constr() const
    {
        term_state_box_t spec;
        spec.lo.setConstant(-4.0);
        spec.hi.setConstant(4.0);
        spec.soft_penalty.setZero();
        return spec;
    }

    // ng = ne = nl = 0 (and all terminal groups zero): all remaining
    // constraint functions stay unimplemented.
};

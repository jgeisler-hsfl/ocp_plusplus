// Example concrete problem for the ocp++ problem interface:
// a discrete-time double integrator.
//
//   x = [q; v]  (position, velocity),  u = [a]  (acceleration)
//
//   dynamics:    q_{k+1} = q_k + Ts * v_k
//                v_{k+1} = v_k + Ts * a_k
//
//   stage cost:  0.5 * (wq*q^2 + wv*v^2 + wa*a^2)
//   terminal:    wf * (q_N^2 + v_N^2)
//
//   box:         -10 <= q <= 10,  -10 <= v <= 10   (all stages incl. terminal)
//                 -1  <= a <= 1                     (stages 0..N-1)
//   inequality:  v - v_max <= 0                    (stages 0..N-1, soft row)
//   linear:      -3 <= 0.5*q + v <= 3              (stages 0..N-1)
//   terminal:    q_N + v_N = 1
//   initial:     x_0 = [1; 0.5]
//
// All box rows are active: state_box_idx = {0, 1}, control_box_idx = {0},
// terminal_state_box_idx = {0, 1}. The velocity cap is the only soft
// constraint: ineq_soft_idx = {0} (slack penalty weight 100); all other
// `*_soft_idx` sets are empty. No terminal inequality / terminal linear
// constraints (ng_t = nl_t = 0) and no stage equality constraints
// (ne = 0), so the corresponding interface functions stay unimplemented
// on purpose.

#pragma once

#include "ocp/problem.hpp"

struct DoubleIntegratorDims
{
    static constexpr int nx = 2;
    static constexpr int nu = 1;
    static constexpr int ng = 1;
    static constexpr int ne = 0;
    static constexpr int nl = 1;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 1;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = true;  // x_0 = [1; 0.5] given
    static constexpr bool has_dynamics_hess = true;  // HVP implemented (zero; linear map)
    static constexpr bool has_constr_hess   = true;  // HVPs implemented (zero; linear maps)
    static constexpr std::array<int, 2> state_box_idx   = {0, 1};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 2> terminal_state_box_idx = {0, 1};
    // soft rows: the velocity cap (ineq row 0) is the only soft constraint
    static constexpr std::array<int, 1> ineq_soft_idx   = {0};
    static constexpr std::array<int, 0> eq_soft_idx     = {};
    static constexpr std::array<int, 0> lin_soft_idx    = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx   = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx  = {};
    static constexpr std::array<int, 0> state_box_soft_idx   = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

class DoubleIntegrator : public ocp::Problem<DoubleIntegratorDims>
{
public:
    DoubleIntegrator() = default;

    // user-managed problem parameters (not part of the interface)
    double Ts_    = 0.1;
    double v_max_ = 2.0;
    double wq_    = 1.0;
    double wv_    = 0.1;
    double wa_    = 0.01;
    double wf_    = 10.0;

    // ---------------------------------------------------------------
    // initial state
    // ---------------------------------------------------------------
    state_t initial_state() const
    {
        state_t x0;
        x0 << 1.0, 0.5;
        return x0;
    }

    // ---------------------------------------------------------------
    // dynamics
    // ---------------------------------------------------------------
    state_t dynamics_next_state(int, const state_t& x, const control_t& u) const
    {
        state_t xn;
        xn(0) = x(0) + Ts_ * x(1);
        xn(1) = x(1) + Ts_ * u(0);
        return xn;
    }

    void dynamics_value_jac(int, const state_t& x, const control_t& u,
                            state_t& x_next,
                            dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
    {
        x_next = dynamics_next_state(0, x, u);
        df_dx.setZero();
        df_dx(0, 0) = 1.0;
        df_dx(0, 1) = Ts_;
        df_dx(1, 1) = 1.0;
        df_du.setZero();
        df_du(1, 0) = Ts_;
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
    // stage cost
    // ---------------------------------------------------------------
    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return 0.5 * (wq_ * x(0) * x(0) + wv_ * x(1) * x(1) + wa_ * u(0) * u(0));
    }

    void stage_cost_value_grad(int, const state_t& x, const control_t& u,
                               double& value, stage_grad_t& g) const
    {
        value = 0.5 * (wq_ * x(0) * x(0) + wv_ * x(1) * x(1) + wa_ * u(0) * u(0));
        g(0) = wq_ * x(0);
        g(1) = wv_ * x(1);
        g(2) = wa_ * u(0);
    }

    void stage_cost_value_grad_hess(int, const state_t& x, const control_t& u,
                                    double& value, stage_grad_t& g,
                                    stage_hess_t& H) const
    {
        stage_cost_value_grad(0, x, u, value, g);
        H.setZero();
        H(0, 0) = wq_;
        H(1, 1) = wv_;
        H(2, 2) = wa_;
    }

    // ---------------------------------------------------------------
    // terminal cost
    // ---------------------------------------------------------------
    double terminal_cost_value(const state_t& x) const
    {
        return wf_ * (x(0) * x(0) + x(1) * x(1));
    }

    void terminal_cost_value_grad(const state_t& x, double& value,
                                  term_grad_t& g) const
    {
        value = wf_ * (x(0) * x(0) + x(1) * x(1));
        g(0) = 2.0 * wf_ * x(0);
        g(1) = 2.0 * wf_ * x(1);
    }

    void terminal_cost_value_grad_hess(const state_t& x, double& value,
                                       term_grad_t& g, term_hess_t& H) const
    {
        terminal_cost_value_grad(x, value, g);
        H.setZero();
        H(0, 0) = 2.0 * wf_;
        H(1, 1) = 2.0 * wf_;
    }

    // ---------------------------------------------------------------
    // stage constraints
    // ---------------------------------------------------------------
    ineq_t stage_inequality_value(int, const state_t& x, const control_t&) const
    {
        ineq_t g;
        g(0) = x(1) - v_max_;
        return g;
    }

    void stage_inequality_value_jac(int, const state_t& x, const control_t&,
                                    ineq_t& g, ineq_dg_dx_t& g_dx,
                                    ineq_dg_du_t& g_du) const
    {
        g = stage_inequality_value(0, x, control_t{});
        g_dx.setZero();
        g_dx(0, 1) = 1.0;
        g_du.setZero();
    }

    void stage_inequality_value_jac_hess(int k, const state_t& x,
                                         const control_t& u,
                                         const ineq_t&, ineq_t& g,
                                         ineq_dg_dx_t& g_dx, ineq_dg_du_t& g_du,
                                         constr_hess_t& hess) const
    {
        // linear constraint: Hessian is zero
        stage_inequality_value_jac(k, x, u, g, g_dx, g_du);
        hess.setZero();
    }

    ineq_pen_t stage_inequality_constr_soft_penalty(int) const
    {
        ineq_pen_t pen;
        pen(0) = 100.0;
        return pen;
    }

    // ne == 0: stage_equality_value / stage_equality_value_jac stay unimplemented.

    stage_linear_t stage_linear_constr(int) const
    {
        stage_linear_t spec;
        spec.A.setZero();
        spec.A(0, 0) = 0.5;
        spec.A(0, 1) = 1.0;
        spec.B.setZero();
        spec.bounds.lo(0) = -3.0;
        spec.bounds.hi(0) = 3.0;
        spec.bounds.soft_penalty.setZero();
        return spec;
    }

    state_box_t stage_state_box_constr(int) const
    {
        state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.soft_penalty.setZero();
        return spec;
    }

    control_box_t stage_control_box_constr(int) const
    {
        control_box_t spec;
        spec.lo(0) = -1.0;
        spec.hi(0) = 1.0;
        spec.soft_penalty.setZero();
        return spec;
    }

    // ---------------------------------------------------------------
    // terminal constraints
    // ---------------------------------------------------------------
    term_state_box_t terminal_state_box_constr() const
    {
        term_state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.soft_penalty.setZero();
        return spec;
    }

    // ng_t == 0: terminal_inequality_value / _value_jac stay unimplemented.

    eq_term_t terminal_equality_value(const state_t& x) const
    {
        eq_term_t e;
        e(0) = x(0) + x(1) - 1.0;
        return e;
    }

    void terminal_equality_value_jac(const state_t& x, eq_term_t& e,
                                     eq_term_de_dx_t& e_dx) const
    {
        e(0) = x(0) + x(1) - 1.0;
        e_dx.setZero();
        e_dx(0, 0) = 1.0;
        e_dx(0, 1) = 1.0;
    }

    void terminal_equality_value_jac_hess(const state_t& x, const eq_term_t&,
                                          eq_term_t& e,
                                          eq_term_de_dx_t& e_dx,
                                          term_constr_hess_t& hess) const
    {
        // linear constraint: Hessian is zero
        terminal_equality_value_jac(x, e, e_dx);
        hess.setZero();
    }

    // nl_t == 0: terminal_linear_constr stays unimplemented.
};

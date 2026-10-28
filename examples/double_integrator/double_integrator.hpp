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
//
// All model parameters live in a single shared DiParams object, exposed
// through the duck-typed params() hook (ocp/problem.hpp, "shared
// parameters"): Ts feeds the dynamics, v_max the velocity cap, w_stage
// the stage cost weights and wf the terminal cost. The object exposes
// the inspect protocol, so every leaf is enumerable and settable by name
// (e.g. from the MEX wrapper) while the model code reads the same live
// object.

#pragma once

#include <optional>

#include "ocp/problem.hpp"
#include "ocp/utils/inspect.hpp"

/// Shared model parameters of the double integrator (duck-typed params()
/// hook, see ocp/problem.hpp "shared parameters"). A single object is
/// read by all model evaluations: Ts by the dynamics, v_max by the
/// velocity-cap constraint, w_stage by the stage cost and wf by the
/// terminal cost. The inspect protocol exposes every leaf by name (Ts
/// is bounded to stay strictly positive, like the step sizes).
struct DiParams
{
    double Ts = 0.1;                    // time step (dynamics)
    double v_max = 2.0;                 // velocity cap (inequality row)
    Eigen::Matrix<double, 3, 1> w_stage;  // stage cost weights [wq; wv; wa]
    double wf = 10.0;                   // terminal cost weight

    DiParams()
    {
        w_stage << 1.0, 0.1, 0.01;
    }

    template <class Visitor>
    static constexpr void inspect(Visitor&& vis)
    {
        vis("Ts", &DiParams::Ts, ocp::inspect::kMinPositive, std::nullopt);
        vis("v_max", &DiParams::v_max);
        vis("w_stage", &DiParams::w_stage);
        vis("wf", &DiParams::wf);
    }
};

/// Dims tag for the double integrator: structural dimensions and box index sets.
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
    static constexpr int np   = 6;  // p = (Ts, v_max, wq, wv, wa, wf)
    static constexpr bool fixed_initial_state = true;  // x_0 = [1; 0.5] given
    static constexpr bool has_dynamics_hess = true;  // HVP implemented (zero; linear map)
    static constexpr bool has_constr_hess   = true;  // HVPs implemented (zero; linear maps)
    // component names for CSV headers / plot labels
    static constexpr std::array<const char*, 2> state_names   = {"q", "v"};
    static constexpr std::array<const char*, 1> control_names = {"a"};
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

    /// User-managed shared model parameters (not part of the interface);
    /// the single live object read by all model evaluations.
    DiParams params_;

    // ---------------------------------------------------------------
    // shared parameters (duck-typed params() hook)
    // ---------------------------------------------------------------
    DiParams& params() { return params_; }
    const DiParams& params() const { return params_; }

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
        xn(0) = x(0) + params_.Ts * x(1);
        xn(1) = x(1) + params_.Ts * u(0);
        return xn;
    }

    void dynamics_value_jac(int, const state_t& x, const control_t& u,
                            state_t& x_next,
                            dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
    {
        x_next = dynamics_next_state(0, x, u);
        df_dx.setZero();
        df_dx(0, 0) = 1.0;
        df_dx(0, 1) = params_.Ts;
        df_dx(1, 1) = 1.0;
        df_du.setZero();
        df_du(1, 0) = params_.Ts;
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
        return 0.5 * (params_.w_stage(0) * x(0) * x(0) + params_.w_stage(1) * x(1) * x(1) + params_.w_stage(2) * u(0) * u(0));
    }

    void stage_cost_value_grad(int, const state_t& x, const control_t& u,
                               double& value, stage_grad_t& g) const
    {
        value = 0.5 * (params_.w_stage(0) * x(0) * x(0) + params_.w_stage(1) * x(1) * x(1) + params_.w_stage(2) * u(0) * u(0));
        g(0) = params_.w_stage(0) * x(0);
        g(1) = params_.w_stage(1) * x(1);
        g(2) = params_.w_stage(2) * u(0);
    }

    void stage_cost_value_grad_hess(int, const state_t& x, const control_t& u,
                                    double& value, stage_grad_t& g,
                                    stage_hess_t& H) const
    {
        stage_cost_value_grad(0, x, u, value, g);
        H.setZero();
        H(0, 0) = params_.w_stage(0);
        H(1, 1) = params_.w_stage(1);
        H(2, 2) = params_.w_stage(2);
    }

    // ---------------------------------------------------------------
    // terminal cost
    // ---------------------------------------------------------------
    double terminal_cost_value(const state_t& x) const
    {
        return params_.wf * (x(0) * x(0) + x(1) * x(1));
    }

    void terminal_cost_value_grad(const state_t& x, double& value,
                                  term_grad_t& g) const
    {
        value = params_.wf * (x(0) * x(0) + x(1) * x(1));
        g(0) = 2.0 * params_.wf * x(0);
        g(1) = 2.0 * params_.wf * x(1);
    }

    void terminal_cost_value_grad_hess(const state_t& x, double& value,
                                       term_grad_t& g, term_hess_t& H) const
    {
        terminal_cost_value_grad(x, value, g);
        H.setZero();
        H(0, 0) = 2.0 * params_.wf;
        H(1, 1) = 2.0 * params_.wf;
    }

    // ---------------------------------------------------------------
    // stage constraints
    // ---------------------------------------------------------------
    ineq_t stage_inequality_value(int, const state_t& x, const control_t&) const
    {
        ineq_t g;
        g(0) = x(1) - params_.v_max;
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
        pen.quad(0) = 100.0;
        return pen;
    }

    // ne == 0: stage_equality_value / stage_equality_value_jac stay unimplemented.

    /// Linear stage constraint: -3 <= 0.5*q + v <= 3.
    stage_linear_t stage_linear_constr(int) const
    {
        stage_linear_t spec;
        spec.A.setZero();
        spec.A(0, 0) = 0.5;
        spec.A(0, 1) = 1.0;
        spec.B.setZero();
        spec.bounds.lo(0) = -3.0;
        spec.bounds.hi(0) = 3.0;
        spec.bounds.penalty = {};
        return spec;
    }

    /// State box: -10 <= q, v <= 10.
    state_box_t stage_state_box_constr(int) const
    {
        state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.penalty = {};
        return spec;
    }

    /// Control box: -1 <= a <= 1.
    control_box_t stage_control_box_constr(int) const
    {
        control_box_t spec;
        spec.lo(0) = -1.0;
        spec.hi(0) = 1.0;
        spec.penalty = {};
        return spec;
    }

    // ---------------------------------------------------------------
    // terminal constraints
    // ---------------------------------------------------------------
    /// Terminal state box: -10 <= q, v <= 10.
    term_state_box_t terminal_state_box_constr() const
    {
        term_state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.penalty = {};
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

    // ---------------------------------------------------------------
    // Parameter vector  p = (Ts, v_max, wq, wv, wa, wf)
    // ---------------------------------------------------------------

    void dynamics_sens_p(int, const state_t& x, const control_t& u,
                         dyn_df_dp_t& f_p, dyn_dA_dp_t& A_p,
                         dyn_dB_dp_t& B_p) const
    {
        // f = [q + Ts*v; v + Ts*u];  A = [[1, Ts],[0, 1]];  B = [0; Ts]
        f_p.setZero();
        f_p(0, 0) = x(1);
        f_p(1, 0) = u(0);
        A_p.setZero();
        A_p(1, 0) = 1.0;  // dA/dTs = [[0,1],[0,0]], row-major flattened
        B_p.setZero();
        B_p(1, 0) = 1.0;  // dB/dTs = [0; 1]
    }

    void stage_cost_sens_p(int, const state_t& x, const control_t& u,
                           param_t& L_p, stage_grad_dp_t& grad_p) const
    {
        // L = 0.5 (wq q^2 + wv v^2 + wa u^2);  grad = [wq q; wv v; wa u]
        // p = (Ts, v_max, wq, wv, wa, wf) -> dL/dp nonzero at wq, wv, wa
        L_p.setZero();
        L_p(2) = 0.5 * x(0) * x(0);
        L_p(3) = 0.5 * x(1) * x(1);
        L_p(4) = 0.5 * u(0) * u(0);
        grad_p.setZero();
        grad_p(0, 2) = x(0);
        grad_p(1, 3) = x(1);
        grad_p(2, 4) = u(0);
    }

    void terminal_cost_sens_p(const state_t& x, param_t& L_p,
                              term_grad_dp_t& grad_p) const
    {
        // L_t = wf (q^2 + v^2);  grad = [2 wf q; 2 wf v]
        L_p.setZero();
        L_p(5) = x(0) * x(0) + x(1) * x(1);
        grad_p.setZero();
        grad_p(0, 5) = 2.0 * x(0);
        grad_p(1, 5) = 2.0 * x(1);
    }

    void stage_inequality_sens_p(int, const state_t&, const control_t&,
                                 ineq_dg_dp_t& g_p, ineq_dg_dx_dp_t& gdx_p,
                                 ineq_dg_du_dp_t& gdu_p) const
    {
        // g = v - v_max
        g_p.setZero();
        g_p(0, 1) = -1.0;
        gdx_p.setZero();
        gdu_p.setZero();
    }

    // ne == 0: stage_equality_sens_p stays unimplemented.

    // linear constraint coefficients/bounds are p-independent
    stage_linear_sensp_t stage_linear_sens_p(int) const
    {
        return stage_linear_sensp_t{};
    }

    stage_state_box_sensp_t stage_state_box_sens_p(int) const
    {
        return stage_state_box_sensp_t{};
    }

    stage_control_box_sensp_t stage_control_box_sens_p(int) const
    {
        return stage_control_box_sensp_t{};
    }

    term_state_box_sensp_t terminal_state_box_sens_p() const
    {
        return term_state_box_sensp_t{};
    }

    // ng_t == 0: terminal_inequality_sens_p stays unimplemented.

    void terminal_equality_sens_p(const state_t&, eq_term_de_dp_t& e_p,
                                  eq_term_de_dx_dp_t& edx_p) const
    {
        // e = q + v - 1 (p-independent)
        e_p.setZero();
        edx_p.setZero();
    }

    // nl_t == 0: terminal_linear_sens_p stays unimplemented.

    x0_dp_t initial_state_sens_p() const
    {
        // x0 = [1; 0.5] (p-independent).  Explicit ::Zero(): value-init
        // (`x0_dp_t{}`) does not zero Eigen 3.3 fixed-size matrices.
        return x0_dp_t::Zero();
    }
};

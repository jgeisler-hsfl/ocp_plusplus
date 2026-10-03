// Phase 4c: verify the ContinuousProblem adapter delegates the dynamics to
// the integrator and produces a well-formed QP through the SQP assembler.
//
// Test problem: a continuous double integrator (qdot = v, vdot = a) wrapped
// in ContinuousProblem with K4Tag. The ODE is linear, so has_dynamics_hess_prod
// is false and the ODE does not provide hess_prod. For a linear ODE the RK4
// stages are exact, so dynamics_next_state must reproduce the exact discrete
// map to machine precision.

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/integrators/continuous_problem.hpp"
#include "ocp/solvers/acados/sqp.hpp"

namespace
{

// --- Dims for the continuous double integrator -------------------------

struct ContDims
{
    static constexpr int nx = 2;
    static constexpr int nu = 1;
    static constexpr int ng = 0;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 0;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = true;
    static constexpr bool has_dynamics_hess_prod = false;  // linear ODE
    static constexpr bool has_constr_hess_prod = false;    // linear constraints
    static constexpr std::array<int, 2> state_box_idx = {0, 1};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 2> terminal_state_box_idx = {0, 1};
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

using CP = ocp::Problem<ContDims>;
using state_t = CP::state_t;
using control_t = CP::control_t;

// --- ODE: qdot = v, vdot = a (linear) ----------------------------------

struct DoubleIntegratorOde
{
    state_t f(const state_t& x, const control_t& u) const
    {
        state_t r;
        r(0) = x(1);
        r(1) = u(0);
        return r;
    }
    void jacobian(const state_t&, const control_t&,
                  CP::dyn_df_dx_t& df_dx, CP::dyn_df_du_t& df_du) const
    {
        df_dx.setZero();
        df_dx(0, 1) = 1.0;
        df_du.setZero();
        df_du(1, 0) = 1.0;
    }
    // No hess_prod: the ODE is linear (has_dynamics_hess_prod = false).
};

// --- Concrete problem --------------------------------------------------

using Integ = ocp::ExplicitRkIntegrator<ContDims, DoubleIntegratorOde, 4,
                                          ocp::K4Tag>;
using Base = ocp::ContinuousProblem<ContDims, DoubleIntegratorOde, Integ>;

struct ContDoubleIntegrator : Base
{
    double wq_ = 1.0;
    double wv_ = 0.1;
    double wa_ = 0.01;
    double wf_ = 10.0;
    double h_  = 0.2;

    ContDoubleIntegrator(double h = 0.2) : Base(DoubleIntegratorOde{}, h), h_(h) {}

    state_t initial_state() const
    {
        state_t x0;
        x0 << 1.0, 0.5;
        return x0;
    }

    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return 0.5 * (wq_ * x(0) * x(0) + wv_ * x(1) * x(1) +
                      wa_ * u(0) * u(0));
    }

    CP::stage_grad_t
    stage_cost_gradient(int, const state_t& x, const control_t& u) const
    {
        CP::stage_grad_t g;
        g(0) = wq_ * x(0);
        g(1) = wv_ * x(1);
        g(2) = wa_ * u(0);
        return g;
    }

    CP::stage_hess_t
    stage_cost_hessian(int, const state_t&, const control_t&) const
    {
        CP::stage_hess_t H;
        H.setZero();
        H(0, 0) = wq_;
        H(1, 1) = wv_;
        H(2, 2) = wa_;
        return H;
    }

    double terminal_cost_value(const state_t& x) const
    {
        return wf_ * (x(0) * x(0) + x(1) * x(1));
    }

    CP::term_grad_t terminal_cost_gradient(const state_t& x) const
    {
        CP::term_grad_t g;
        g(0) = 2.0 * wf_ * x(0);
        g(1) = 2.0 * wf_ * x(1);
        return g;
    }

    CP::term_hess_t terminal_cost_hessian(const state_t&) const
    {
        CP::term_hess_t H;
        H.setZero();
        H(0, 0) = 2.0 * wf_;
        H(1, 1) = 2.0 * wf_;
        return H;
    }

    CP::state_box_t stage_state_box_constr(int) const
    {
        CP::state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.soft_penalty.setZero();
        return spec;
    }

    CP::control_box_t stage_control_box_constr(int) const
    {
        CP::control_box_t spec;
        spec.lo(0) = -1.0;
        spec.hi(0) = 1.0;
        spec.soft_penalty.setZero();
        return spec;
    }

    CP::term_state_box_t terminal_state_box_constr() const
    {
        CP::term_state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.soft_penalty.setZero();
        return spec;
    }
};

// --- helpers -----------------------------------------------------------

int failures = 0;

void check(bool cond, const char* msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

void check_close(double a, double b, double tol, const char* msg)
{
    if (std::fabs(a - b) > tol)
    {
        std::fprintf(stderr, "FAIL: %s (%.15e vs %.15e)\n", msg, a, b);
        ++failures;
    }
}

}  // namespace

int run_erk_4c_tests()
{
    const ContDoubleIntegrator prob;
    const double h = prob.h_;

    // ---------------------------------------------------------------
    //  dynamics_next_state: for the linear double integrator, RK4 is exact
    //  (the ODE has polynomial trajectories of degree <= 2, well within RK4's
    //  4th-order accuracy). The exact discrete map is:
    //      q_{k+1} = q_k + h * v_k + h^2/2 * a_k
    //      v_{k+1} = v_k + h * a_k
    // ---------------------------------------------------------------
    {
        state_t x;
        x << 0.7, -0.3;
        control_t u;
        u << 0.4;

        state_t xn = prob.dynamics_next_state(0, x, u);
        // exact
        const double q_exp = x(0) + h * x(1) + 0.5 * h * h * u(0);
        const double v_exp = x(1) + h * u(0);
        check_close(xn(0), q_exp, 1e-14, "4c dynamics q exact");
        check_close(xn(1), v_exp, 1e-14, "4c dynamics v exact");
        std::printf("  dynamics_next_state exact  OK\n");
    }

    // ---------------------------------------------------------------
    //  dynamics_jacobian: for the linear ODE the composed-map Jacobian is
    //  constant and equal to the exact discrete Jacobian
    //      df/dx = [[1, h + h^2/2], [0, 1]]  (RK4 exact for linear ODE)
    //      df/du = [[h^2/2], [h]]
    // ---------------------------------------------------------------
    {
        state_t x;
        x << 1.0, 0.0;
        control_t u;
        u << 0.5;
        CP::dyn_df_dx_t Jx;
        CP::dyn_df_du_t Ju;
        prob.dynamics_jacobian(0, x, u, Jx, Ju);
        // exact: for linear ODE, df/dx = I + h*A_discrete where A_discrete
        // is the state-transition matrix of the discrete map.
        // For the double integrator with exact integration:
        //   q_{k+1} = q + h*v + h^2/2*a  => dq/dv = h, dq/da = h^2/2
        //   v_{k+1} = v + h*a           => dv/da = h
        // So:  df/dx = [[1, h], [0, 1]],  df/du = [[h^2/2], [h]]
        check_close(Jx(0, 0), 1.0, 1e-14, "4c Jx(0,0)");
        check_close(Jx(0, 1), h, 1e-14, "4c Jx(0,1)");
        check_close(Jx(1, 0), 0.0, 1e-14, "4c Jx(1,0)");
        check_close(Jx(1, 1), 1.0, 1e-14, "4c Jx(1,1)");
        check_close(Ju(0, 0), 0.5 * h * h, 1e-14, "4c Ju(0,0)");
        check_close(Ju(1, 0), h, 1e-14, "4c Ju(1,0)");
        std::printf("  dynamics_jacobian exact  OK\n");
    }

    // ---------------------------------------------------------------
    //  assemble_qp: verify the SQP assembler produces a finite, well-formed
    //  QP for the continuous problem.
    // ---------------------------------------------------------------
    {
        using Solver = ocp::SqpSolver<ContDoubleIntegrator>;
        Solver solver;
        ocp::Solution<ContDoubleIntegrator> sol(5);
        solver.resize(5);

        // Warm start: x_0 = initial_state, u_k = 0, forward-simulate states.
        sol.x[0] = prob.initial_state();
        for (int k = 0; k < sol.N; ++k)
        {
            control_t u_k;
            u_k.setZero();
            sol.u[k] = u_k;
            sol.x[k + 1] =
                prob.dynamics_next_state(k, sol.x[k], sol.u[k]);
        }

        const auto status = solver.assemble_qp(prob, sol);
        check(status == ocp::Status::kSolved,
              "4c assemble_qp returns kSolved");

        const auto& qp = solver.last_qp();
        // Every stage (first / path / term) carries a Hessian + gradient;
        // only the first/path stages additionally carry the dynamics matrix
        // (BA, b) — the terminal stage has no control and no dynamics row.
        auto check_finite = [&](const auto& m, const char* what)
        {
            check(m.allFinite(),
                  (std::string("4c QP ") + what + " finite").c_str());
        };

        check_finite(qp.first.hess, "first hess");
        check_finite(qp.first.grad, "first grad");
        check_finite(qp.first.BA, "first BA");
        check_finite(qp.first.b, "first b");

        for (int k = 0; k < qp.N - 1; ++k)
        {
            check_finite(qp.path[k].hess, "path hess");
            check_finite(qp.path[k].grad, "path grad");
            check_finite(qp.path[k].BA, "path BA");
            check_finite(qp.path[k].b, "path b");
        }

        check_finite(qp.term.hess, "term hess");
        check_finite(qp.term.grad, "term grad");

        // Correct row counts (nx = 2, nu = 1).
        check(qp.first.BA.rows() == 2, "4c first BA rows == nx");
        check(qp.first.BA.cols() == 3, "4c first BA cols == nx+nu");
        std::printf("  assemble_qp finite QP  OK\n");
    }

    if (failures != 0)
    {
        std::fprintf(stderr, "[4c] %d check(s) failed.\n", failures);
    }
    return failures;
}

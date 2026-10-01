// Phase 2 sub-step 2g test: SQP driver (SqpSolver::solve,
// SqpSolver::check_termination) from include/ocp/solvers/acados/sqp.hpp.
//
// Checks (SQP_PHASE2_PLAN.md sec. 2g; acados ocp_nlp_sqp.c:351-430,
// :538-800):
//   - check_termination: all five branches + the kUnset fallthrough, in
//     the acados order.
//   - solve() on DoubleIntegrator (N = 10, zero-control forward
//     simulation warm start; N >= 10 keeps the terminal equality
//     q_N + v_N = 1 feasible under the control box |a| <= 1, since the
//     minimum of q_N + v_N at N = 5 is 1.15): kSolved, independently
//     recomputed NLP
//     residuals below tolerances, x[0] = initial state, terminal
//     equality and control box satisfied, cost_value = problem cost +
//     0.5 w s^2 slack penalty (sec. 1.6).
//   - repeated solve on the same object: warm start from the converged
//     iterate terminates at iteration 0 (kSolved, no QP solved).
//   - max_iter = 0: kMaxIterations, one statistics row.
//   - DegProbe (fixed x_0, hard ineq violated at the pinned state,
//     zero Jacobian w.r.t. u_0; assemble_2f sec. 2f.6): kInfeasible.
//   - FailQp / ZeroStepQp duck-typed QP solvers: kQpFailure and
//     kMinStep driver paths.

#include <cmath>
#include <cstdio>

#include "ocp/solvers/acados/sqp.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

namespace
{

using namespace ocp;

int failures = 0;

void check(bool ok, const char* msg)
{
    if (!ok)
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

// ---------------------------------------------------------------------
// check_termination branch coverage
// ---------------------------------------------------------------------

void test_check_termination()
{
    SqpSolver<DoubleIntegrator> solver;
    const SqpOptions& o = solver.options();

    NlpResiduals res;
    res.res_stat = 1.0;
    res.res_eq = 1.0;
    res.res_ineq = 1.0;
    res.res_comp = 1.0;

    // 1. NaN dominates (iter 0, no max_iter trigger)
    NlpResiduals nan = res;
    nan.res_ineq = std::nan("");
    check(solver.check_termination(0, nan, 1.0, 0.0)
              == Status::kNanDetected,
          "check_termination: NaN -> kNanDetected");

    // 2. iter >= max_iter with !eval_residual_at_max_iter
    check(solver.check_termination(o.max_iter, res, 1.0, 0.0)
              == Status::kMaxIterations,
          "check_termination: iter >= max_iter -> kMaxIterations");

    // 3. all four norms strictly below the tolerances
    NlpResiduals ok = res;
    ok.res_stat = 0.5 * o.tol_stat;
    ok.res_eq = 0.5 * o.tol_eq;
    ok.res_ineq = 0.5 * o.tol_ineq;
    ok.res_comp = 0.5 * o.tol_comp;
    check(solver.check_termination(0, ok, 1.0, 0.0) == Status::kSolved,
          "check_termination: all norms < tols -> kSolved");

    // 4. tiny step at iter > 0
    check(solver.check_termination(1, res, 0.5 * o.tol_min_step_norm, 0.0)
              == Status::kMinStep,
          "check_termination: step < tol_min_step_norm -> kMinStep");

    // 4b. the same tiny step at iter 0 is NOT kMinStep
    check(solver.check_termination(0, res, 0.5 * o.tol_min_step_norm, 0.0)
              == Status::kUnset,
          "check_termination: min-step only for iter > 0");

    // 5. unbounded cost
    check(solver.check_termination(0, res, 1.0, o.tol_unbounded)
              == Status::kUnbounded,
          "check_termination: cost <= tol_unbounded -> kUnbounded");

    // 6. iter >= max_iter with eval_residual_at_max_iter = true and
    //    residuals above tolerance: still kMaxIterations (last branch)
    {
        SqpOptions o2 = o;
        o2.eval_residual_at_max_iter = true;
        SqpSolver<DoubleIntegrator> s2(o2);
        check(s2.check_termination(o2.max_iter, res, 1.0, 0.0)
                  == Status::kMaxIterations,
              "check_termination: eval_residual_at_max_iter -> "
              "kMaxIterations after the tolerance check");
    }

    // fallthrough: nothing triggered
    check(solver.check_termination(1, res, 1.0, 0.0) == Status::kUnset,
          "check_termination: no trigger -> kUnset");
}

// ---------------------------------------------------------------------
// warm-start iterate for DoubleIntegrator: x[0] = initial state,
// u = 0, forward simulation
// ---------------------------------------------------------------------

Solution<DoubleIntegrator>
make_warm_start(const DoubleIntegrator& p, int n_stages, double a = 0.0)
{
    Solution<DoubleIntegrator> sol(n_stages);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < n_stages; ++k)
    {
        sol.u[k](0) = a;
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }
    return sol;
}

// Constant control that makes the terminal equality q_N + v_N = 1 hold
// exactly at the warm start: with x0 = [1; 0.5] and Euler dynamics,
// q_N + v_N = (q0 + v0) + Ts*N*v0 + a * (Ts^2*N*(N-1)/2 + Ts*N),
// so a = (1 - (q0 + v0) - Ts*N*v0) / (Ts^2*N*(N-1)/2 + Ts*N).
// Since the equality is linear in the trajectory, the feasibility is
// preserved through every SQP iteration.
double terminal_eq_feasible_control(const DoubleIntegrator& p, int n_stages)
{
    const double Ts = p.Ts_;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * n_stages * (n_stages - 1) / 2.0 + Ts * n_stages;
    return (1.0 - (q0 + v0) - Ts * n_stages * v0) / den;
}

// ---------------------------------------------------------------------
// end-to-end solve on DoubleIntegrator (N = 10)
// ---------------------------------------------------------------------

void test_solve_double_integrator()
{
    DoubleIntegrator prob;
    // N = 10: at N = 5 the terminal equality q_N + v_N = 1 is infeasible
    // (min of q_N + v_N over |a| <= 1 is 1.15). The warm start satisfies
    // the terminal equality exactly (terminal_eq_feasible_control); since
    // the equality is linear in the trajectory, every SQP iterate keeps
    // it feasible.
    const int N = 10;

    SqpOptions opts;
    opts.print_level = 0;

    SqpSolver<DoubleIntegrator> solver(opts);
    Solution<DoubleIntegrator> sol =
        make_warm_start(prob, N, terminal_eq_feasible_control(prob, N));

    const Status st = solver.solve(prob, sol);
    check(st == Status::kSolved, "DI solve: status kSolved");
    check(sol.status == Status::kSolved, "DI solve: sol.status kSolved");
    check(solver.statistics().iter >= 1, "DI solve: at least one row");

    // independent residual recheck (free function, plan sec. 2c)
    const NlpResiduals res =
        compute_nlp_residuals(prob, sol, opts.tau_min);
    check(res.res_stat < opts.tol_stat, "DI solve: res_stat < tol");
    check(res.res_eq < opts.tol_eq, "DI solve: res_eq < tol");
    check(res.res_ineq < opts.tol_ineq, "DI solve: res_ineq < tol");
    check(res.res_comp < opts.tol_comp, "DI solve: res_comp < tol");

    // initial state preserved (fixed x_0)
    const auto x0 = prob.initial_state();
    for (int i = 0; i < 2; ++i)
    {
        check_close(sol.x[0](i), x0(i), 1e-10, "DI solve: x[0] preserved");
    }

    // terminal equality q + v = 1
    check_close(sol.x[N](0) + sol.x[N](1), 1.0, 1e-6,
                "DI solve: terminal equality");

    // control box -1 <= u <= 1
    for (int k = 0; k < N; ++k)
    {
        check(sol.u[k](0) >= -1.0 - 1e-8 && sol.u[k](0) <= 1.0 + 1e-8,
              "DI solve: control box");
    }

    // cost_value = problem cost + 0.5 w s^2 (sec. 1.6). The soft
    // velocity cap v <= 2 is inactive at the optimum (v stays ~ 0.5),
    // so the slack penalty is zero and cost_value equals the raw cost.
    double cost = 0.0;
    for (int k = 0; k < N; ++k)
    {
        cost += prob.stage_cost_value(k, sol.x[k], sol.u[k]);
    }
    cost += prob.terminal_cost_value(sol.x[N]);
    check(sol.cost_value >= cost - 1e-10, "DI solve: cost >= raw cost");
    check_close(sol.cost_value, cost, 1e-6,
                "DI solve: cost = raw cost (soft ineq inactive)");

    // statistics: the last row is the accepted iteration
    const auto& rows = solver.statistics().rows;
    check(rows.size() == static_cast<std::size_t>(solver.statistics().iter),
          "DI solve: row count == iter");
    check(rows.back().alpha > 0.0, "DI solve: last alpha > 0");
    check(rows.back().qp_iter >= 0, "DI solve: last qp_iter >= 0");

    // last_qp / last_qp_sol accessors expose the final assembled QP
    check(solver.last_qp().N == N, "DI solve: last_qp horizon");
    check(solver.last_qp_sol().N == N, "DI solve: last_qp_sol horizon");
}

// ---------------------------------------------------------------------
// repeated solve on the same object: the converged iterate terminates
// at iteration 0 without solving a QP (slacks warm-started, sec. 2g.1)
// ---------------------------------------------------------------------

void test_solve_warm_start()
{
    DoubleIntegrator prob;
    const int N = 10;

    SqpOptions opts;
    opts.print_level = 0;

    SqpSolver<DoubleIntegrator> solver(opts);
    Solution<DoubleIntegrator> sol =
        make_warm_start(prob, N, terminal_eq_feasible_control(prob, N));
    check(solver.solve(prob, sol) == Status::kSolved,
          "DI warm: first solve kSolved");
    const int cost_iters_1 = solver.statistics().iter;

    const Status st2 = solver.solve(prob, sol);
    check(st2 == Status::kSolved, "DI warm: second solve kSolved");
    check(solver.statistics().iter == 1, "DI warm: second solve "
          "terminates at iteration 0");
    check(solver.statistics().rows.back().qp_iter == 0,
          "DI warm: no QP solved in the second solve");
    check(cost_iters_1 >= 2, "DI warm: first solve ran the loop");
}

// ---------------------------------------------------------------------
// max_iter = 0: the termination check at iteration 0 returns
// kMaxIterations before any QP solve
// ---------------------------------------------------------------------

void test_solve_max_iter_zero()
{
    DoubleIntegrator prob;
    const int N = 5;

    SqpOptions opts;
    opts.max_iter = 0;
    opts.print_level = 0;

    SqpSolver<DoubleIntegrator> solver(opts);
    Solution<DoubleIntegrator> sol = make_warm_start(prob, N);

    const Status st = solver.solve(prob, sol);
    check(st == Status::kMaxIterations, "DI maxiter: kMaxIterations");
    check(sol.status == Status::kMaxIterations, "DI maxiter: sol.status");
    check(solver.statistics().iter == 1, "DI maxiter: one row recorded");
    check(solver.statistics().rows[0].qp_iter == 0,
          "DI maxiter: no QP solved");
}

// ---------------------------------------------------------------------
// DegProbe: fixed x_0 pinned at a state that violates a hard ineq
// whose Jacobian w.r.t. (u_0; s_0) is zero -> infeasible (plan 2f.6)
// ---------------------------------------------------------------------

struct DegProbeDims
{
    static constexpr int nx = 1;
    static constexpr int nu = 1;
    static constexpr int ng = 1;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 0;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = true;
    static constexpr bool has_dynamics_hess_prod = false;
    static constexpr bool has_constr_hess_prod = false;
    static constexpr std::array<int, 0> state_box_idx = {};
    static constexpr std::array<int, 0> control_box_idx = {};
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

class DegProbe : public ocp::Problem<DegProbeDims>
{
public:
    state_t initial_state() const
    {
        return state_t::Constant(1, 2.0);
    }

    state_t dynamics_next_state(int, const state_t& x, const control_t&) const
    {
        return x;
    }

    void dynamics_jacobian(int, const state_t&, const control_t&,
                           dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
    {
        df_dx(0, 0) = 1.0;
        df_du.setZero();
    }

    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return x(0) * x(0) + u(0) * u(0);
    }

    stage_grad_t stage_cost_gradient(int, const state_t& x,
                                     const control_t& u) const
    {
        stage_grad_t g;
        g << 2.0 * x(0), 2.0 * u(0);
        return g;
    }

    stage_hess_t stage_cost_hessian(int, const state_t&,
                                    const control_t&) const
    {
        stage_hess_t H;
        H.setZero();
        H(0, 0) = 2.0;
        H(1, 1) = 2.0;
        return H;
    }

    double terminal_cost_value(const state_t& x) const
    {
        return x(0) * x(0);
    }

    term_grad_t terminal_cost_gradient(const state_t& x) const
    {
        return term_grad_t::Constant(1, 2.0 * x(0));
    }

    term_hess_t terminal_cost_hessian(const state_t&) const
    {
        term_hess_t H;
        H(0, 0) = 2.0;
        return H;
    }

    // x - 1 <= 0: violated at the pinned x_0 = 2, independent of u
    ineq_t stage_inequality_constr(int, const state_t& x,
                                   const control_t&) const
    {
        return ineq_t::Constant(1, x(0) - 1.0);
    }

    void stage_inequality_constr_jacobian(
        int, const state_t&, const control_t&, ineq_dg_dx_t& g_dx,
        ineq_dg_du_t& g_du) const
    {
        g_dx(0, 0) = 1.0;
        g_du.setZero();
    }
};

void test_solve_infeasible()
{
    DegProbe prob;
    const int N = 2;

    SqpOptions opts;
    opts.print_level = 0;

    SqpSolver<DegProbe> solver(opts);
    Solution<DegProbe> sol(N);
    sol.x[0] = prob.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = prob.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }

    const Status st = solver.solve(prob, sol);
    check(st == Status::kInfeasible, "DegProbe: kInfeasible");
    check(sol.status == Status::kInfeasible, "DegProbe: sol.status");
}

// ---------------------------------------------------------------------
// duck-typed QP solvers: kQpFailure and kMinStep driver paths
// ---------------------------------------------------------------------

// Duck-typed QP solver mock: the driver only needs solve() and
// statistics() (for the iteration count in the stat row).
struct MockStats
{
    int iter = 0;
};

class FailQp
{
public:
    Status solve(const Qp<DoubleIntegrator>&,
                 QpSol<DoubleIntegrator>&)
    {
        return Status::kQpFailure;
    }
    MockStats statistics() const { return MockStats{}; }
};

class ZeroStepQp
{
public:
    Status solve(const Qp<DoubleIntegrator>& in,
                 QpSol<DoubleIntegrator>& out)
    {
        out.resize(in.N);
        out.ux_first.setZero();
        for (int k = 0; k < in.N - 1; ++k)
        {
            out.ux_path[k].setZero();
        }
        out.ux_term.setZero();
        for (int k = 0; k < in.N; ++k)
        {
            out.pi[k].setZero();
        }
        out.lam_first.setZero();
        for (int k = 0; k < in.N - 1; ++k)
        {
            out.lam_path[k].setZero();
        }
        out.lam_term.setZero();
        out.t_first.setZero();
        for (int k = 0; k < in.N - 1; ++k)
        {
            out.t_path[k].setZero();
        }
        out.t_term.setZero();
        return Status::kSolved;
    }
    MockStats statistics() const { return MockStats{}; }
};

void test_solve_qp_failure()
{
    DoubleIntegrator prob;
    const int N = 5;

    SqpOptions opts;
    opts.max_iter = 3;
    opts.print_level = 0;

    SqpSolver<DoubleIntegrator, Eigen::Dynamic, FailQp> solver(opts);
    Solution<DoubleIntegrator> sol = make_warm_start(prob, N);

    const Status st = solver.solve(prob, sol);
    check(st == Status::kQpFailure, "FailQp: kQpFailure");
    check(sol.status == Status::kQpFailure, "FailQp: sol.status");
    check(solver.statistics().iter == 1, "FailQp: one row recorded");
    check(solver.statistics().rows[0].qp_status
              == static_cast<int>(Status::kQpFailure),
          "FailQp: qp_status in the row");
}

void test_solve_min_step()
{
    DoubleIntegrator prob;
    const int N = 5;

    SqpOptions opts;
    opts.max_iter = 3;
    opts.print_level = 0;

    SqpSolver<DoubleIntegrator, Eigen::Dynamic, ZeroStepQp> solver(opts);
    Solution<DoubleIntegrator> sol = make_warm_start(prob, N);

    const Status st = solver.solve(prob, sol);
    check(st == Status::kMinStep, "ZeroStepQp: kMinStep");
    check(sol.status == Status::kMinStep, "ZeroStepQp: sol.status");
    // zero merit improvement at every trial -> the line search hits
    // alpha_min at iteration 0 (globalizer kMinStep, acados :894)
    check(solver.statistics().iter == 1, "ZeroStepQp: one row "
          "(kMinStep at iter 0)");
    check(solver.statistics().rows[0].step_norm == 0.0,
          "ZeroStepQp: step norm recorded");
    check(solver.statistics().rows[0].alpha > 0.0
              && solver.statistics().rows[0].alpha < 1.0,
          "ZeroStepQp: shrunken alpha recorded");
}

}  // namespace

int run_driver_2g_tests()
{
    failures = 0;
    test_check_termination();
    test_solve_double_integrator();
    test_solve_warm_start();
    test_solve_max_iter_zero();
    test_solve_infeasible();
    test_solve_qp_failure();
    test_solve_min_step();
    return failures;
}

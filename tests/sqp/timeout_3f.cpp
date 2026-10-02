// Phase 3 sub-step 3f test: timeout (ocp_nlp_sqp.c:351-443, 519-520,
// 538-539, 606-644).
//
// Covers (SQP_PHASE3_PLAN.md sec. 3f):
//   - check_termination: the timeout branch fires only when
//     timeout_max_time > 0 and timeout_max_time <= elapsed + estimate.
//   - solve(): tiny timeout_max_time -> kTimeout at the first check;
//     timeout_max_time = 0 (default) -> inert, normal solve.
//   - All four heuristic values (0=ZERO, 1=LAST, 2=MAX, 3=AVERAGE) run
//     without spurious timeout when the budget is large.

#include <cstdio>

#include "ocp/solvers/acados/sqp.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

namespace
{

using namespace ocp;
using DI = DoubleIntegrator;

int failures = 0;

void check(bool ok, const char* msg)
{
    if (!ok)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

Solution<DI> make_warm_start(const DI& p, int n_stages, double a = 0.0)
{
    Solution<DI> sol(n_stages);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < n_stages; ++k)
    {
        sol.u[k](0) = a;
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }
    return sol;
}

double terminal_eq_feasible_control(const DI& p, int n_stages)
{
    const double Ts = p.Ts_;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * n_stages * (n_stages - 1) / 2.0 + Ts * n_stages;
    return (1.0 - (q0 + v0) - Ts * n_stages * v0) / den;
}

// ---------------------------------------------------------------------
// check_termination: timeout branch
// ---------------------------------------------------------------------

void test_check_termination_timeout()
{
    // Solver with an active timeout budget.
    SqpOptions opts;
    opts.timeout_max_time = 1.0;  // 1-second budget
    opts.timeout_heuristic = 0;   // ZERO: estimate stays 0

    SqpSolver<DI> solver(opts);
    NlpResiduals res;
    res.res_stat = 1.0;
    res.res_eq = 1.0;
    res.res_ineq = 1.0;
    res.res_comp = 1.0;

    // elapsed = 2.0 > budget 1.0 -> kTimeout
    check(solver.check_termination(1, res, 1.0, 0.0, 2.0) == Status::kTimeout,
          "check_termination: elapsed > budget -> kTimeout");

    // elapsed = 0.5 < budget 1.0 -> kUnset (keep iterating)
    check(solver.check_termination(1, res, 1.0, 0.0, 0.5) == Status::kUnset,
          "check_termination: within budget -> kUnset");

    // Exactly at the boundary: 1.0 <= 1.0 + 0 -> kTimeout
    check(solver.check_termination(1, res, 1.0, 0.0, 1.0) == Status::kTimeout,
          "check_termination: elapsed == budget -> kTimeout");

    // Default options (timeout_max_time = 0): the branch is inert
    SqpSolver<DI> solver_def;
    check(
        solver_def.check_termination(1, res, 1.0, 0.0, 1e9) == Status::kUnset,
        "check_termination: timeout_max_time = 0 -> inert");
}

// ---------------------------------------------------------------------
// solve(): tiny timeout triggers kTimeout
// ---------------------------------------------------------------------

void test_solve_timeout()
{
    DI prob;
    const int N = 5;

    SqpOptions opts;
    opts.print_level = 0;
    opts.timeout_max_time = 1e-9;  // smaller than any iteration's elapsed
    opts.timeout_heuristic = 1;    // LAST (irrelevant: triggers at iter 0)

    SqpSolver<DI> solver(opts);
    Solution<DI> sol = make_warm_start(prob, N);

    const Status st = solver.solve(prob, sol);
    check(st == Status::kTimeout, "solve: tiny timeout -> kTimeout");
    check(sol.status == Status::kTimeout, "solve: sol.status kTimeout");
    check(solver.statistics().status == Status::kTimeout,
          "solve: stat.status kTimeout");
    check(solver.statistics().iter == 1,
          "solve: timeout at iter 0 -> one stat row");
}

// ---------------------------------------------------------------------
// solve(): large timeout (all heuristics) -> normal kSolved
// ---------------------------------------------------------------------

void test_solve_timeout_large_inert()
{
    DI prob;
    const int N = 10;
    const double a = terminal_eq_feasible_control(prob, N);

    for (int h = 0; h <= 3; ++h)
    {
        SqpOptions opts;
        opts.print_level = 0;
        opts.timeout_heuristic = h;
        opts.timeout_max_time = 1e9;  // effectively no timeout

        SqpSolver<DI> solver(opts);
        Solution<DI> sol = make_warm_start(prob, N, a);

        const Status st = solver.solve(prob, sol);
        char buf[64];
        std::snprintf(buf, sizeof(buf),
                      "heuristic=%d: kSolved (no spurious timeout)", h);
        check(st == Status::kSolved, buf);
    }
}

}  // namespace

int run_timeout_3f_tests()
{
    failures = 0;
    test_check_termination_timeout();
    test_solve_timeout();
    test_solve_timeout_large_inert();

    if (failures == 0)
    {
        std::printf("All timeout (3f) checks passed.\n");
    }
    return failures;
}

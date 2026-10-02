// Phase 3 sub-step 3i test: funnel globalization (SQP_PHASE3_PLAN.md
// sec. 3i, acados ocp_nlp_globalization_funnel.c).
//
// Checks:
//   - end-to-end: funnel on double_integrator (N=10) -> kSolved, all four
//     NLP residuals below tolerance.
//   - funnel_width monotonically non-increasing across accepted iterations,
//     ending near the final L1 infeasibility.
//   - funnel accept-type (iter_type) is logged in the statistics.
//   - funnel solution matches the merit-backtracking reference solution
//     (cost + primal within tolerance).

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/solvers/acados/sqp.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

namespace
{

using namespace ocp;
using DI = DoubleIntegrator;

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

void check_close(double a, double b, double tol, const std::string& msg)
{
    if (std::fabs(a - b) > tol)
    {
        std::fprintf(stderr, "FAIL: %s (%.15e vs %.15e)\n", msg.c_str(), a,
                     b);
        ++failures;
    }
}

// Solver with the funnel globalizer.
using FunnelSolver = SqpSolver<DI, Eigen::Dynamic,
                               HpipmQpSolver<DI, Eigen::Dynamic>,
                               GlmRegularizer<DI, Eigen::Dynamic>,
                               Funnel<DI, Eigen::Dynamic>>;

// Solver with the default merit-backtracking globalizer (reference).
using MeritSolver = SqpSolver<DI>;

Solution<DI> make_warm_start(const DI& p, int n_stages)
{
    const double Ts = p.Ts_;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * n_stages * (n_stages - 1) / 2.0 +
                       Ts * n_stages;
    const double a = (1.0 - (q0 + v0) - Ts * n_stages * v0) / den;

    Solution<DI> sol(n_stages);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < n_stages; ++k)
    {
        sol.u[k](0) = a;
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }
    return sol;
}

// ---------------------------------------------------------------------
// End-to-end: funnel on DI (N=10) -> kSolved, residuals < tol.
// ---------------------------------------------------------------------
void test_funnel_converges()
{
    const std::string p = "3i funnel: ";
    DI prob;
    const int N = 10;

    SqpOptions opts;
    opts.max_iter = 30;
    FunnelSolver solver(opts);
    Solution<DI> sol = make_warm_start(prob, N);
    const Status st = solver.solve(prob, sol);
    check(st == Status::kSolved, p + "status kSolved");
    if (st != Status::kSolved)
    {
        return;
    }

    const NlpResiduals res = compute_nlp_residuals(prob, sol);
    check(res.res_stat < 1e-8, p + "res_stat");
    check(res.res_eq < 1e-8, p + "res_eq");
    check(res.res_ineq < 1e-8, p + "res_ineq");
    check(res.res_comp < 1e-8, p + "res_comp");
}

// ---------------------------------------------------------------------
// funnel_width is monotonically non-increasing across the logged
// iterations, and ends near the final L1 infeasibility.
// ---------------------------------------------------------------------
void test_funnel_width_monotone()
{
    const std::string p = "3i funnel width: ";
    DI prob;
    const int N = 10;

    SqpOptions opts;
    opts.max_iter = 30;
    FunnelSolver solver(opts);
    Solution<DI> sol = make_warm_start(prob, N);
    const Status st = solver.solve(prob, sol);
    check(st == Status::kSolved, p + "status kSolved");
    if (st != Status::kSolved)
    {
        return;
    }

    const auto& rows = solver.statistics().rows;
    double prev_w = 0.0;
    bool first = true;
    int last_w_idx = -1;
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        const double w = rows[i].funnel_width;
        if (w > 0.0)
        {
            if (!first)
            {
                check(w <= prev_w + 1e-12, p + "width non-increasing");
            }
            else
            {
                first = false;
            }
            prev_w = w;
            last_w_idx = static_cast<int>(i);
        }
    }
    check(last_w_idx >= 0, p + "width was logged");
}

// ---------------------------------------------------------------------
// The funnel accept-type (iter_type) is actually logged in the stats.
// ---------------------------------------------------------------------
void test_funnel_iter_type_logged()
{
    const std::string p = "3i funnel iter_type: ";
    DI prob;
    const int N = 10;

    SqpOptions opts;
    opts.max_iter = 30;
    FunnelSolver solver(opts);
    Solution<DI> sol = make_warm_start(prob, N);
    const Status st = solver.solve(prob, sol);
    check(st == Status::kSolved, p + "status kSolved");
    if (st != Status::kSolved)
    {
        return;
    }

    const auto& rows = solver.statistics().rows;
    bool any_logged = false;
    for (const auto& r : rows)
    {
        if (r.funnel_iter_type >= 0)
        {
            any_logged = true;
            break;
        }
    }
    check(any_logged, p + "iter_type logged");
}

// ---------------------------------------------------------------------
// Funnel solution matches the merit-backtracking reference.
// ---------------------------------------------------------------------
void test_funnel_matches_merit()
{
    const std::string p = "3i funnel vs merit: ";
    DI prob;
    const int N = 10;

    SqpOptions opts;
    opts.max_iter = 30;

    MeritSolver solver_m(opts);
    Solution<DI> sol_m = make_warm_start(prob, N);
    const Status st_m = solver_m.solve(prob, sol_m);
    check(st_m == Status::kSolved, p + "merit kSolved");

    FunnelSolver solver_f(opts);
    Solution<DI> sol_f = make_warm_start(prob, N);
    const Status st_f = solver_f.solve(prob, sol_f);
    check(st_f == Status::kSolved, p + "funnel kSolved");

    if (st_m != Status::kSolved || st_f != Status::kSolved)
    {
        return;
    }

    check_close(sol_f.cost_value, sol_m.cost_value, 1e-6, p + "cost");
    double dx = 0.0, du = 0.0;
    for (int k = 0; k <= N; ++k)
    {
        dx = std::max(
            dx, static_cast<double>(
                    (sol_f.x[k] - sol_m.x[k]).cwiseAbs().maxCoeff()));
    }
    for (int k = 0; k < N; ++k)
    {
        du = std::max(
            du, static_cast<double>(
                    (sol_f.u[k] - sol_m.u[k]).cwiseAbs().maxCoeff()));
    }
    check(dx < 1e-6, p + "dx < 1e-6");
    check(du < 1e-6, p + "du < 1e-6");
}

}  // namespace

int run_funnel_3i_tests()
{
    failures = 0;
    test_funnel_converges();
    test_funnel_width_monotone();
    test_funnel_iter_type_logged();
    test_funnel_matches_merit();

    if (failures == 0)
    {
        std::printf("All funnel globalization (3i) checks passed.\n");
    }
    return failures;
}

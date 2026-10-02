// Phase 3 sub-step 3g test: adaptive Levenberg-Marquardt.
//
// Covers (SQP_PHASE3_PLAN.md sec. 3g):
//   - mu schedule: full steps decay by 1/lam (with mu_bar lag),
//     truncated steps grow by *lam (capped at 1.0), floor at mu_min.
//   - effective damping: the Hessian (u;x) block increment equals
//     obj_scalar * raw_cost * mu.

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/solvers/acados/sqp.hpp"
#include "ocp/solvers/acados/regularize.hpp"

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

// ---------------------------------------------------------------------
// Test 1: mu schedule (full steps decay, truncated steps grow)
// ---------------------------------------------------------------------

void test_mu_schedule()
{
    SqpOptions opts;
    opts.with_adaptive_lm = true;
    opts.adaptive_lm_mu0 = 1e-3;
    opts.adaptive_lm_lam = 5.0;
    opts.adaptive_lm_mu_min = 1e-16;

    SqpSolver<DI> solver(opts);

    // iter 0: init
    solver.update_adaptive_lm_mu(0, 0.0);
    check_close(solver.adaptive_lm_mu(), 1e-3, 1e-18,
                "iter 0: mu = mu0");
    check_close(solver.adaptive_lm_mu_bar(), 1e-3, 1e-18,
                "iter 0: mu_bar = mu0");

    // iter 1: full -> mu = max(min, mu_bar/lam) = 1e-3/5 = 2e-4
    solver.update_adaptive_lm_mu(1, 1.0);
    check_close(solver.adaptive_lm_mu(), 2e-4, 1e-18,
                "iter 1 full: mu = 1e-3/5 = 2e-4");
    check_close(solver.adaptive_lm_mu_bar(), 1e-3, 1e-18,
                "iter 1 full: mu_bar = old mu = 1e-3");

    // iter 2: full -> mu = max(min, 1e-3/5) = 2e-4 (mu_bar still 1e-3)
    solver.update_adaptive_lm_mu(2, 1.0);
    check_close(solver.adaptive_lm_mu(), 2e-4, 1e-18,
                "iter 2 full: mu = mu_bar/lam = 1e-3/5 = 2e-4");
    check_close(solver.adaptive_lm_mu_bar(), 2e-4, 1e-18,
                "iter 2 full: mu_bar = old mu = 2e-4");

    // iter 3: full -> mu = max(min, 2e-4/5) = 4e-5
    solver.update_adaptive_lm_mu(3, 1.0);
    check_close(solver.adaptive_lm_mu(), 4e-5, 1e-18,
                "iter 3 full: mu = 2e-4/5 = 4e-5");
    check_close(solver.adaptive_lm_mu_bar(), 2e-4, 1e-18,
                "iter 3 full: mu_bar = old mu = 2e-4");

    // iter 4: full -> mu = max(min, 2e-4/5) = 4e-5 (mu_bar now 2e-4)
    solver.update_adaptive_lm_mu(4, 1.0);
    check_close(solver.adaptive_lm_mu(), 4e-5, 1e-18,
                "iter 4 full: mu = 2e-4/5 = 4e-5");
    check_close(solver.adaptive_lm_mu_bar(), 4e-5, 1e-18,
                "iter 4 full: mu_bar = old mu = 4e-5");

    // iter 5: truncated (alpha=0.5) -> mu = min(lam*4e-5, 1) = 2e-4
    solver.update_adaptive_lm_mu(5, 0.5);
    check_close(solver.adaptive_lm_mu(), 2e-4, 1e-18,
                "iter 5 trunc: mu = 5*4e-5 = 2e-4");
    check_close(solver.adaptive_lm_mu_bar(), 4e-5, 1e-18,
                "iter 5 trunc: mu_bar unchanged = 4e-5");

    // iter 6: truncated (alpha=0.7) -> mu = min(lam*2e-4, 1) = 1e-3
    solver.update_adaptive_lm_mu(6, 0.7);
    check_close(solver.adaptive_lm_mu(), 1e-3, 1e-18,
                "iter 6 trunc: mu = 5*2e-4 = 1e-3");

    // iter 7: full -> mu = max(min, mu_bar/lam) = max(min, 4e-5/5) = 8e-6
    // (mu_bar is still 4e-5 from iter 5 truncation)
    solver.update_adaptive_lm_mu(7, 1.0);
    check_close(solver.adaptive_lm_mu(), 8e-6, 1e-18,
                "iter 7 full: mu = mu_bar/lam = 4e-5/5 = 8e-6");
    check_close(solver.adaptive_lm_mu_bar(), 1e-3, 1e-18,
                "iter 7 full: mu_bar = old mu = 1e-3");
}

// ---------------------------------------------------------------------
// Test 2: mu floor (mu_min)
// ---------------------------------------------------------------------

void test_mu_min_floor()
{
    SqpOptions opts;
    opts.adaptive_lm_mu0 = 1e-3;
    opts.adaptive_lm_lam = 5.0;
    opts.adaptive_lm_mu_min = 1e-16;

    SqpSolver<DI> solver(opts);
    solver.update_adaptive_lm_mu(0, 0.0);

    // 50 full steps: mu decays by 1/lam per pair of steps, eventually
    // hitting the mu_min floor.
    for (int i = 1; i <= 50; ++i)
    {
        solver.update_adaptive_lm_mu(i, 1.0);
    }
    check_close(solver.adaptive_lm_mu(), 1e-16, 1e-20,
                "mu floor: after 50 full steps mu == mu_min");
}

// ---------------------------------------------------------------------
// Test 3: mu cap at 1.0
// ---------------------------------------------------------------------

void test_mu_cap_at_one()
{
    SqpOptions opts;
    opts.adaptive_lm_mu0 = 0.8;
    opts.adaptive_lm_lam = 5.0;
    opts.adaptive_lm_mu_min = 1e-16;

    SqpSolver<DI> solver(opts);
    solver.update_adaptive_lm_mu(0, 0.0);
    check_close(solver.adaptive_lm_mu(), 0.8, 1e-15,
                "cap: iter 0 mu = mu0 = 0.8");

    // truncated: mu = min(5 * 0.8, 1.0) = 1.0 (capped)
    solver.update_adaptive_lm_mu(1, 0.5);
    check_close(solver.adaptive_lm_mu(), 1.0, 1e-15,
                "cap: mu = min(5*0.8, 1) = 1.0");

    // another truncated: mu = min(5 * 1.0, 1.0) = 1.0
    solver.update_adaptive_lm_mu(2, 0.3);
    check_close(solver.adaptive_lm_mu(), 1.0, 1e-15,
                "cap: stays at 1.0");
}

// ---------------------------------------------------------------------
// Test 4: effective damping on Hessian (end-to-end via solve loop)
// ---------------------------------------------------------------------

void test_effective_damping()
{
    DI prob;
    const int N = 5;
    const double a = 0.0;
    Solution<DI> sol = make_warm_start(prob, N, a);

    // Compute raw cost (stage + terminal, no slack penalty)
    double raw = 0.0;
    for (int k = 0; k < N; ++k)
    {
        raw += prob.stage_cost_value(k, sol.x[k], sol.u[k]);
    }
    raw += prob.terminal_cost_value(sol.x[N]);

    const double mu0 = 1e-3;
    const double obj_scalar = 2.0;
    const double expected = obj_scalar * raw * mu0;

    // Use NoRegularizer so the Hessian difference is purely the LM term.
    using SolverT = SqpSolver<DI, Eigen::Dynamic, HpipmQpSolver<DI>,
                              NoRegularizer<DI>>;

    // Baseline: no LM
    SqpOptions optsA;
    optsA.max_iter = 0;
    optsA.with_adaptive_lm = false;
    SolverT solverA(optsA);
    Solution<DI> solA = make_warm_start(prob, N, a);
    solverA.solve(prob, solA);
    const auto& qpA = solverA.last_qp();

    // Adaptive LM (mu at iter 0 = mu0)
    SqpOptions optsB;
    optsB.max_iter = 0;
    optsB.with_adaptive_lm = true;
    optsB.adaptive_lm_mu0 = mu0;
    optsB.adaptive_lm_obj_scalar = obj_scalar;
    SolverT solverB(optsB);
    Solution<DI> solB = make_warm_start(prob, N, a);
    solverB.solve(prob, solB);
    const auto& qpB = solverB.last_qp();

    constexpr int nux = DI::nx + DI::nu;  // 3

    // First stage: (u;x) diagonal
    for (int i = 0; i < nux; ++i)
    {
        const double diff =
            static_cast<double>(qpB.first.hess(i, i)) -
            static_cast<double>(qpA.first.hess(i, i));
        check_close(diff, expected, 1e-12,
                    "first hess diag [" + std::to_string(i) + "] diff");
    }

    // First stage: off-diagonals unchanged
    for (int i = 0; i < nux; ++i)
    {
        for (int j = i + 1; j < nux; ++j)
        {
            const double diff =
                static_cast<double>(qpB.first.hess(i, j)) -
                static_cast<double>(qpA.first.hess(i, j));
            check_close(diff, 0.0, 1e-14,
                        "first hess off-diag (" + std::to_string(i) + "," +
                            std::to_string(j) + ")");
        }
    }

    // Path stages (k = 1..N-1)
    for (int k = 1; k < N; ++k)
    {
        for (int i = 0; i < nux; ++i)
        {
            const double diff =
                static_cast<double>(qpB.path[k - 1].hess(i, i)) -
                static_cast<double>(qpA.path[k - 1].hess(i, i));
            check_close(diff, expected, 1e-12,
                        "path[" + std::to_string(k) + "] hess diag [" +
                            std::to_string(i) + "]");
        }
    }

    // Terminal stage: (x) diagonal (nx = 2)
    for (int i = 0; i < DI::nx; ++i)
    {
        const double diff =
            static_cast<double>(qpB.term.hess(i, i)) -
            static_cast<double>(qpA.term.hess(i, i));
        check_close(diff, expected, 1e-12,
                    "term hess diag [" + std::to_string(i) + "]");
    }
}

// ---------------------------------------------------------------------
// Test 5: end-to-end solve with adaptive LM converges
// ---------------------------------------------------------------------

void test_solve_converges()
{
    DI prob;
    const int N = 10;
    const double Ts = prob.Ts_;
    const double q0 = prob.initial_state()(0);
    const double v0 = prob.initial_state()(1);
    const double den =
        Ts * Ts * N * (N - 1) / 2.0 + Ts * N;
    const double a = (1.0 - (q0 + v0) - Ts * N * v0) / den;

    SqpOptions opts;
    opts.print_level = 0;
    opts.with_adaptive_lm = true;

    SqpSolver<DI> solver(opts);
    Solution<DI> sol = make_warm_start(prob, N, a);

    const Status st = solver.solve(prob, sol);
    check(st == Status::kSolved,
          "adaptive LM end-to-end: kSolved");
    check(solver.statistics().iter > 0,
          "adaptive LM: at least 1 iteration recorded");
}

}  // namespace

int run_adaptive_lm_3g_tests()
{
    failures = 0;
    test_mu_schedule();
    test_mu_min_floor();
    test_mu_cap_at_one();
    test_effective_damping();
    test_solve_converges();

    if (failures == 0)
    {
        std::printf("All adaptive LM (3g) checks passed.\n");
    }
    return failures;
}

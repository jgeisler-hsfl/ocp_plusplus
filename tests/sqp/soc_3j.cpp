// Phase 3 sub-step 3j test: second-order-correction pre-pass
// (SQP_PHASE3_PLAN.md sec. 3j; acados ocp_nlp_common.c:4290
//  ocp_nlp_perform_second_order_correction, and the SOC gate in
//  ocp_nlp_globalization_merit_backtracking.c:569-602 / Waechter 2006).
//
// Checks:
//   - regression: use_soc=false (default) on the double integrator matches
//     the pre-3j behaviour (kSolved, residuals below tolerance, soc_count==0).
//   - use_soc=true converges (kSolved, residuals below tolerance), reaches
//     the same cost as the no-SOC reference, and actually fires the
//     second-order correction at least once (soc_count() > 0) -- on the
//     warm-start double integrator the full step is rejected on the first
//     iteration (it does not lower the L-infinity violation), so the QP is
//     re-solved with the corrected RHS.

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

// Merit-backtracking globalizer (the default; the SOC pre-pass lives here).
using SocSolver = SqpSolver<DI>;

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
// 1. regression: use_soc=false -> kSolved, soc_count == 0.
// ---------------------------------------------------------------------
void test_no_soc_regression()
{
    const std::string p = "3j soc-off: ";
    DI prob;
    const int N = 10;

    SqpOptions opts;
    opts.max_iter = 30;
    opts.glob.use_soc = false;
    SocSolver solver(opts);
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
    check(solver.soc_count() == 0, p + "soc_count == 0");
}

// ---------------------------------------------------------------------
// 2. use_soc=true converges, matches the no-SOC cost, and fires the SOC.
// ---------------------------------------------------------------------
void test_soc_converges()
{
    const std::string p = "3j soc-on: ";
    DI prob;
    const int N = 10;

    SqpOptions opts_ref;
    opts_ref.max_iter = 30;
    opts_ref.glob.use_soc = false;
    SocSolver ref(opts_ref);
    Solution<DI> sol_ref = make_warm_start(prob, N);
    const Status st_ref = ref.solve(prob, sol_ref);
    check(st_ref == Status::kSolved, p + "ref kSolved");
    if (st_ref != Status::kSolved)
    {
        return;
    }

    SqpOptions opts;
    opts.max_iter = 30;
    opts.glob.use_soc = true;
    SocSolver solver(opts);
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

    check_close(sol.cost_value, sol_ref.cost_value, 1e-6, p + "cost vs ref");

    // Mechanism: the warm-start full step is rejected (it does not lower the
    // L-infinity violation), so the SOC pre-pass re-solves the QP at least
    // once.
    check(solver.soc_count() > 0, p + "soc_count > 0 (SOC fired)");
}

}  // namespace

int run_soc_3j_tests()
{
    failures = 0;
    test_no_soc_regression();
    test_soc_converges();

    if (failures == 0)
    {
        std::printf("All second-order-correction (3j) checks passed.\n");
    }
    return failures;
}

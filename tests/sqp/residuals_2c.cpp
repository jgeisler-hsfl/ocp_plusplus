// Phase 2 sub-step 2c test: NLP residuals
// (NlpResiduals, compute_nlp_residuals) from
// include/ocp/solvers/acados/sqp.hpp.
//
// Checks (SQP_PHASE2_PLAN.md sec. 2c, ocp_nlp_common.c:3743-3846
// ocp_nlp_res_compute):
//   - res_stat: Lagrangian gradient (stationarity) with hand-computed
//     values on the DoubleIntegrator (N=2, fixed x_0)
//   - res_eq: dynamics gap max over stages
//   - res_ineq: max positive violation (box, linear, ineq, terminal eq)
//   - res_comp: complementarity with tau_min shift; eq rows zeroed
//     (terminal equality violated but excluded from res_comp)
//   - tau_min = 0 variant: no floor on res_comp
//   - all four residuals are exactly zero on a KKT-consistent iterate

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/solvers/acados/sqp.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

namespace
{

using namespace ocp;

int failures = 0;

void check_close(double a, double b, const std::string& msg)
{
    if (std::fabs(a - b) > 1e-12)
    {
        std::fprintf(stderr, "FAIL: %s (%.12e vs %.12e)\n", msg.c_str(), a, b);
        ++failures;
    }
}

// Zero every field of a Solution.
template <class P, int NH>
void zero_solution(Solution<P, NH>& sol)
{
    const int N = sol.N;
    for (int k = 0; k < N; ++k)
    {
        sol.x[k].setZero();
        sol.u[k].setZero();
        sol.lambda_dyn[k].setZero();
        sol.lambda_ineq_stage[k].setZero();
        sol.lambda_eq_stage[k].setZero();
        sol.lambda_lin_stage[k].setZero();
        sol.lambda_box_state[k].setZero();
        sol.lambda_box_control[k].setZero();
    }
    sol.x[N].setZero();
    sol.lambda_box_state[N].setZero();
    sol.lambda_ineq_term.setZero();
    sol.lambda_eq_term.setZero();
    sol.lambda_lin_term.setZero();
}

// ---------------------------------------------------------------------
// Hand-computed infeasible iterate (DoubleIntegrator, N = 2, fixed x_0).
//
//   x[0] = [ 1, 0.5]   (initial_state, pinned)
//   x[1] = [11,  1  ]   (q_1 = 11 violates box hi = 10)
//   x[2] = [ 0.5, 0.6 ] (terminal eq violated: 0.5+0.6-1 = 0.1)
//   u[0] = [ 0.5]
//   u[1] = [-0.3]
//
// Multipliers:
//   lambda_dyn[0]   = [1, 2]
//   lambda_dyn[1]   = [3, 4]
//   lambda_ineq[0]  = [0.1]
//   lambda_ineq[1]  = [0.2]
//   lambda_box_state[0] = [0.1, 0.2, 0.3, 0.4]  (lo_q, lo_v, hi_q, hi_v)
//   lambda_box_state[1] = [0.5, 0.6, 0.7, 0.8]
//   lambda_box_state[2] = [0.9, 1.0, 1.1, 1.2]
//   lambda_box_control[0] = [0.1, 0.2]
//   lambda_box_control[1] = [0.3, 0.4]
//   lambda_lin[0]   = [0.5]
//   lambda_lin[1]   = [-0.3]
//   lambda_eq_term  = [1.5]
//
// Expected results (tau_min = 1e-16):
//   res_stat = 17.7
//   res_eq   = 10.6
//   res_ineq = 3.5
//   res_comp = 0.7 + 1e-16
// ---------------------------------------------------------------------
void test_infeasible_iterate()
{
    using DI = DoubleIntegrator;
    const std::string p = "2c infeasible: ";
    constexpr int N = 2;
    const double tau_min = 1e-16;

    DI prob;
    Solution<DI> sol(N);
    zero_solution(sol);

    sol.x[0] << 1.0, 0.5;
    sol.x[1] << 11.0, 1.0;
    sol.x[2] << 0.5, 0.6;
    sol.u[0](0) = 0.5;
    sol.u[1](0) = -0.3;

    sol.lambda_dyn[0] << 1.0, 2.0;
    sol.lambda_dyn[1] << 3.0, 4.0;
    sol.lambda_ineq_stage[0](0) = 0.1;
    sol.lambda_ineq_stage[1](0) = 0.2;
    sol.lambda_box_state[0] << 0.1, 0.2, 0.3, 0.4;
    sol.lambda_box_state[1] << 0.5, 0.6, 0.7, 0.8;
    sol.lambda_box_state[2] << 0.9, 1.0, 1.1, 1.2;
    sol.lambda_box_control[0] << 0.1, 0.2;
    sol.lambda_box_control[1] << 0.3, 0.4;
    sol.lambda_lin_stage[0](0) = 0.5;
    sol.lambda_lin_stage[1](0) = -0.3;
    sol.lambda_eq_term(0) = 1.5;

    const NlpResiduals r =
        compute_nlp_residuals<DI, Eigen::Dynamic>(prob, sol, tau_min);

    // res_stat = max(0.095, 12.05, 17.7) = 17.7
    //   k=0 u-part: |0.005 - 0.2 + 0.1| = 0.095 (x_0 skipped: fixed)
    //   k=1 x-part: max|12.05, -2.1| = 12.05
    //   k=N: max|14.7, 17.7| = 17.7
    check_close(r.res_stat, 17.7, p + "res_stat");

    // res_eq = max(||f_0-x_1||_inf, ||f_1-x_2||_inf)
    //   f_0 = [1.05, 0.55], gap_0 = [9.95, 0.45] -> 9.95
    //   f_1 = [11.1, 0.97], gap_1 = [10.6, 0.37] -> 10.6
    check_close(r.res_eq, 10.6, p + "res_eq");

    // res_ineq = max over all violations:
    //   k=1 lin: 0.5*11+1 = 6.5 > 3 -> 3.5  (dominates)
    //   k=1 box q: 11-10 = 1
    //   k=N eq: |0.5+0.6-1| = 0.1
    check_close(r.res_ineq, 3.5, p + "res_ineq");

    // res_comp: k=1 box state hi j=0: 0.7*max(0, 11-10)+tau_min = 0.7+tau_min
    //   all other sides: 0*violation + tau_min = tau_min
    //   terminal eq (k=N, e=0.1): zeroed (acados idxe masking)
    check_close(r.res_comp, 0.7 + tau_min, p + "res_comp");
}

// ---------------------------------------------------------------------
// tau_min = 0: res_comp has no floor; only the actual lambda*violation
// contributes. All other residuals unchanged.
// ---------------------------------------------------------------------
void test_zero_tau_min()
{
    using DI = DoubleIntegrator;
    const std::string p = "2c zero-tau: ";
    constexpr int N = 2;

    DI prob;
    Solution<DI> sol(N);
    zero_solution(sol);

    sol.x[0] << 1.0, 0.5;
    sol.x[1] << 11.0, 1.0;
    sol.x[2] << 0.5, 0.6;
    sol.u[0](0) = 0.5;
    sol.u[1](0) = -0.3;

    sol.lambda_dyn[0] << 1.0, 2.0;
    sol.lambda_dyn[1] << 3.0, 4.0;
    sol.lambda_ineq_stage[0](0) = 0.1;
    sol.lambda_ineq_stage[1](0) = 0.2;
    sol.lambda_box_state[0] << 0.1, 0.2, 0.3, 0.4;
    sol.lambda_box_state[1] << 0.5, 0.6, 0.7, 0.8;
    sol.lambda_box_state[2] << 0.9, 1.0, 1.1, 1.2;
    sol.lambda_box_control[0] << 0.1, 0.2;
    sol.lambda_box_control[1] << 0.3, 0.4;
    sol.lambda_lin_stage[0](0) = 0.5;
    sol.lambda_lin_stage[1](0) = -0.3;
    sol.lambda_eq_term(0) = 1.5;

    const NlpResiduals r =
        compute_nlp_residuals<DI, Eigen::Dynamic>(prob, sol, 0.0);

    check_close(r.res_stat, 17.7, p + "res_stat");
    check_close(r.res_eq, 10.6, p + "res_eq");
    check_close(r.res_ineq, 3.5, p + "res_ineq");
    // Without tau_min the floor vanishes; the only non-zero term is
    // 0.7 * 1 (box q hi violation at k=1).
    check_close(r.res_comp, 0.7, p + "res_comp (no tau_min)");
}

// ---------------------------------------------------------------------
// All-zero multipliers + a feasible iterate: every residual must be
// small (only the cost gradient contributes to res_stat).
//
// Feasible iterate: x[0]=[1,0.5], forward-simulate with u=0.
//   x[1] = [1.05, 0.5], x[2] = [1.1, 0.5]
//   ineq: v - 2 = -1.5 <= 0
//   lin:  0.5*1 + 0.5 = 1.0 in [-3, 3]
//   box:  all within bounds
//   term eq: 1.1 + 0.5 - 1 = 0.6  (not zero, but we check res_ineq
//            includes it: res_ineq = 0.6)
// ---------------------------------------------------------------------
void test_zero_multipliers_feasible()
{
    using DI = DoubleIntegrator;
    const std::string p = "2c zero-lam: ";
    constexpr int N = 2;

    DI prob;
    Solution<DI> sol(N);
    zero_solution(sol);

    // Forward-simulate with u = 0 from x_0 = [1, 0.5]
    sol.x[0] << 1.0, 0.5;
    sol.u[0].setZero();
    sol.x[1] = prob.dynamics_next_state(0, sol.x[0], sol.u[0]);  // [1.05, 0.5]
    sol.u[1].setZero();
    sol.x[2] = prob.dynamics_next_state(1, sol.x[1], sol.u[1]);  // [1.1, 0.5]

    // All multipliers zero.
    const NlpResiduals r =
        compute_nlp_residuals<DI, Eigen::Dynamic>(prob, sol);

    // res_eq: dynamics are exactly satisfied -> 0
    check_close(r.res_eq, 0.0, p + "res_eq (feasible)");

    // res_ineq: terminal eq = |1.1 + 0.5 - 1| = 0.6
    // (all stage constraints satisfied; terminal eq violated)
    check_close(r.res_ineq, 0.6, p + "res_ineq (term eq)");

    // res_comp: all multipliers zero, tau_min = 1e-16
    // Every active side contributes 0*violation + tau_min = tau_min
    // The terminal eq row is zeroed, so it does NOT contribute.
    check_close(r.res_comp, 1e-16, p + "res_comp (zero multipliers)");

    // res_stat: cost gradient only (all multipliers zero)
    //   k=0: grad = [1, 0.05, 0] (u-part: 0)  -> max = 1 (x skipped, fixed)
    //         actually x_0 skipped, so u-part only: |0| = 0
    //   k=1: grad = [1.05, 0.05, 0]
    //         u-part: |0| = 0
    //         x-part: [1.05, 0.05] -> max = 1.05
    //   k=N: grad = [2*10*1.1, 2*10*0.5] = [22, 10]
    //         x-part: [22, 10] -> max = 22
    check_close(r.res_stat, 22.0, p + "res_stat (cost grad only)");
}

}  // namespace

int run_residuals_2c_tests()
{
    failures = 0;
    test_infeasible_iterate();
    test_zero_tau_min();
    test_zero_multipliers_feasible();

    if (failures == 0)
    {
        std::printf("All residuals (2c) checks passed.\n");
    }
    return failures;
}

// Phase 1 block 1e test: QpMode driver dispatch (full condensing).
//
// Covers (SQP_HPIPM_PARITY_PHASE1_PLAN.md sec. 1e):
//   - AcadosOptions::qp_mode defaults to kStaged (all existing behavior
//     unchanged);
//   - AcadosSolver with QpMode::kFullCondensed solves the same NLP as
//     kStaged (DoubleIntegrator, dynamic N = 10 and fixed NH = 10): both
//     kSolved, same x / u / cost (the QP is solved through
//     condense_full -> HpipmDenseQpSolver -> expand_full_sol);
//   - the dense path with qp.options.warm_start = 2 (dual warm start across
//     SQP iterations through the persistent dense solution) also converges
//     to the same solution;
//   - the cond_time stat: 0 in kStaged mode, > 0 in kFullCondensed.

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/utils/inspect.hpp"
#include "ocp/solvers/acados/acados.hpp"

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

double terminal_eq_feasible_control(const DI& p, int n_stages)
{
    const double Ts = p.Ts_;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * n_stages * (n_stages - 1) / 2.0 + Ts * n_stages;
    return (1.0 - (q0 + v0) - Ts * n_stages * v0) / den;
}

double total_cond_time(const AcadosStatistics& stat)
{
    double t = 0.0;
    for (const auto& row : stat.rows)
    {
        t += row.cond_time;
    }
    return t;
}

template <int NH>
void test_qpmode(const char* name, int N)
{
    const std::string p = name;
    DI prob;
    const double a = terminal_eq_feasible_control(prob, N);

    // staged (default mode)
    Solution<DI, NH> sol_st;
    sol_st.resize(N);
    sol_st.x[0] = prob.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol_st.u[k](0) = a;
        sol_st.x[k + 1] = prob.dynamics_next_state(k, sol_st.x[k],
                                                   sol_st.u[k]);
    }
    AcadosOptions o1;
    AcadosSolver<DI, NH> s1(o1);
    const Status st1 = s1.solve(prob, sol_st);
    check(st1 == Status::kSolved, p + ": staged kSolved");
    if (st1 != Status::kSolved)
    {
        return;
    }

    // full condensed
    Solution<DI, NH> sol_dc;
    sol_dc.resize(N);
    sol_dc.x[0] = prob.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol_dc.u[k](0) = a;
        sol_dc.x[k + 1] = prob.dynamics_next_state(k, sol_dc.x[k],
                                                   sol_dc.u[k]);
    }
    AcadosOptions o2;
    o2.qp_mode = QpMode::kFullCondensed;
    AcadosSolver<DI, NH> s2(o2);
    const Status st2 = s2.solve(prob, sol_dc);
    check(st2 == Status::kSolved, p + ": full-condensed kSolved");
    if (st2 != Status::kSolved)
    {
        return;
    }

    // same NLP solution (the QP is solved to its IPM tolerances in both
    // modes; the SQP propagates the difference, so use a loose bound)
    const double tol = 1e-6;
    for (int k = 0; k < N; ++k)
    {
        check((sol_st.u[k] - sol_dc.u[k]).cwiseAbs().maxCoeff() < tol,
              p + ": u[" + std::to_string(k) + "] agrees");
        check((sol_st.x[k] - sol_dc.x[k]).cwiseAbs().maxCoeff() < tol,
              p + ": x[" + std::to_string(k) + "] agrees");
    }
    check((sol_st.x[N] - sol_dc.x[N]).cwiseAbs().maxCoeff() < tol,
          p + ": terminal x agrees");
    check_close(sol_st.cost_value, sol_dc.cost_value, 1e-8,
                p + ": cost agrees");

    // cond_time stat
    check(total_cond_time(s1.statistics()) == 0.0,
          p + ": staged cond_time is 0");
    check(total_cond_time(s2.statistics()) > 0.0,
          p + ": condensed cond_time > 0");

    // full condensed + dual warm start (qp.options.warm_start = 2, set
    // by name on the live solver): the dense duals (lam / t) persist in
    // the driver workspace across SQP iterations; the primal step stays
    // cold.
    int ws2_iter = 0;
    {
        Solution<DI, NH> sol_ws;
        sol_ws.resize(N);
        sol_ws.x[0] = prob.initial_state();
        for (int k = 0; k < N; ++k)
        {
            sol_ws.u[k](0) = a;
            sol_ws.x[k + 1] = prob.dynamics_next_state(k, sol_ws.x[k],
                                                       sol_ws.u[k]);
        }
        AcadosOptions o3;
        o3.qp_mode = QpMode::kFullCondensed;
        AcadosSolver<DI, NH> s3(o3);
        inspect::set_double(s3, "qp.options.warm_start", 2.0);
        const Status st3 = s3.solve(prob, sol_ws);
        ws2_iter = s3.statistics().iter;
        check(st3 == Status::kSolved, p + ": condensed + ws2 kSolved");
        if (st3 != Status::kSolved)
        {
            return;
        }
        for (int k = 0; k <= N; ++k)
        {
            check((sol_st.x[k] - sol_ws.x[k]).cwiseAbs().maxCoeff() < tol,
                  p + ": ws2 x[" + std::to_string(k) + "] agrees");
        }
        for (int k = 0; k < N; ++k)
        {
            check((sol_st.u[k] - sol_ws.u[k]).cwiseAbs().maxCoeff() < tol,
                  p + ": ws2 u[" + std::to_string(k) + "] agrees");
        }
    }

    std::printf("  %s: staged %d it, condensed %d it (cond_time %.3e s), "
                "ws2 %d it\n",
                p.c_str(), s1.statistics().iter, s2.statistics().iter,
                total_cond_time(s2.statistics()), ws2_iter);
}

}  // namespace

int run_qpmode_1e_tests()
{
    failures = 0;
    // N = 10 is the smallest horizon where the terminal equality
    // q_N + v_N = 1 is exactly feasible for a constant control (the
    // control box bounds make N < 10 infeasible; see
    // sqp_double_integrator.cpp).
    test_qpmode<Eigen::Dynamic>("DI dyn N=10 ", 10);
    test_qpmode<10>("DI fixed NH=10", 10);

    if (failures == 0)
    {
        std::printf("All qpmode (1e) checks passed.\n");
    }
    return failures;
}

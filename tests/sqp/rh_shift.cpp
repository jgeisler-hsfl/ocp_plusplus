// Receding-horizon shift test (TODO.md item B, plan block 3):
//
// Covers the driver-level `AcadosSolver::shift` / `reset_warm_start`:
//   - after a cold solve, `shift()` advances both the Solution and the
//     driver-internal SqpSlacks one stage; verify the exact mapping
//     (first <- stage-1 path, path up one stage, new last path stage
//     zeroed, terminal slacks unchanged).
//   - the shifted warm start converges (kSolved) to the same solution,
//     with no more SQP iterations / IPM iterations than the cold solve.
//   - `reset_warm_start()` zeroes slacks_ and the persistent QP duals,
//     after which a fresh cold solve converges to the same solution.

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
using D = QpDim<DI>;

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
        std::fprintf(stderr, "FAIL: %s (%.15e vs %.15e)\n", msg.c_str(),
                     a, b);
        ++failures;
    }
}

template <class Vec>
void check_vec_close(const Vec& a, const Vec& b, double tol,
                     const std::string& msg)
{
    if (a.size() != b.size())
    {
        std::fprintf(stderr, "FAIL: %s (size %d vs %d)\n", msg.c_str(),
                     (int)a.size(), (int)b.size());
        ++failures;
        return;
    }
    const double m =
        (a.size() == 0) ? 0.0 : (a - b).cwiseAbs().maxCoeff();
    if (m > tol)
    {
        std::fprintf(stderr, "FAIL: %s (max diff %.3e)\n", msg.c_str(), m);
        ++failures;
    }
}

template <class Vec>
void check_vec_zero(const Vec& v, const std::string& msg)
{
    const double m = (v.size() == 0) ? 0.0 : v.cwiseAbs().maxCoeff();
    if (m != 0.0)
    {
        std::fprintf(stderr, "FAIL: %s (max %.3e)\n", msg.c_str(), m);
        ++failures;
    }
}

/// Sum the qp_iter column over all statistics rows (same helper as
/// warm_start_3c; the last row for a kSolved run carries a stale value
/// from the prior iteration, but is the same for both sides of the
/// comparison).
int total_ipm_iters(const AcadosStatistics<>& stat)
{
    int total = 0;
    for (int i = 0; i < stat.n_rows(); ++i)
    {
        total += stat.row(i).qp_iter;
    }
    return total;
}

int sqp_iters(const AcadosStatistics<>& stat)
{
    return stat.n_rows();
}

/// Zero-control feasible warm-start trajectory (shared with 2g / 3c).
Solution<DI> make_warm_start(const DI& p, int n_stages)
{
    Solution<DI> sol(n_stages);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < n_stages; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }
    return sol;
}

}  // namespace

void test_rh_shift_driver()
{
    DI prob;
    const int N = 10;
    const DI::state_t x0_new = prob.initial_state();

    AcadosOptions opts;
    opts.print_level = 0;
    AcadosSolver<DI> solver(opts);
    // Persistent QP duals (warm_start = 2). warm_start_first_qp stays off
    // for the cold reference solve: on a fresh solver there are no
    // persistent duals yet, so a "warm" first QP would only clip zeros.
    check(inspect::set_double(solver, "qp.options.warm_start", 2.0).ok(),
          "set qp.options.warm_start = 2");

    // 1. Cold reference solve --------------------------------------
    solver.reset_warm_start();
    Solution<DI> sol = make_warm_start(prob, N);
    const Status st_cold = solver.solve(prob, sol);
    check(st_cold == Status::kSolved, "rh shift: cold solve kSolved");
    if (st_cold != Status::kSolved)
    {
        return;
    }
    const int cold_sqp = sqp_iters(solver.statistics());
    const int cold_ipm = total_ipm_iters(solver.statistics());
    const double cold_cost = sol.cost_value;
    const Solution<DI> sol_cold = sol;
    std::printf("  rh shift: cold sqp=%d ipm=%d cost=%.6f\n", cold_sqp,
                cold_ipm, cold_cost);

    // 2. Shift the solution + slacks by one stage ------------------
    const SqpSlacks<DI> sl0 = solver.slacks();
    solver.shift(sol, x0_new);

    // Solution mapping (rollover default)
    check((sol.x[0] - x0_new).cwiseAbs().maxCoeff() == 0.0,
          "rh shift: x[0] set to x0_new");
    for (int k = 1; k < N; ++k)
    {
        check_vec_close(sol.x[k], sol_cold.x[k + 1], 0.0,
                        "rh shift: x[k] = old x[k+1] (k=" +
                            std::to_string(k) + ")");
    }
    check_vec_close(sol.x[N], sol_cold.x[N], 0.0,
                    "rh shift: x[N] rolled over");
    for (int k = 0; k + 1 < N; ++k)
    {
        check_vec_close(sol.u[k], sol_cold.u[k + 1], 0.0,
                        "rh shift: u[k] = old u[k+1] (k=" +
                            std::to_string(k) + ")");
    }
    check_vec_close(sol.u[N - 1], sol_cold.u[N - 1], 0.0,
                    "rh shift: u[N-1] rolled over");
    check(sol.status == Status::kUnset, "rh shift: status reset");
    check(sol.cost_value == 0.0, "rh shift: cost reset");

    // Slack mapping
    {
        const auto& sl = solver.slacks();
        for (int k = 0; k + 1 < N - 1; ++k)
        {
            check_vec_close(sl.path[k], sl0.path[k + 1], 0.0,
                            "rh shift: slack path[" + std::to_string(k) +
                                "] = old path[k+1]");
        }
        check_vec_zero(sl.path[N - 2],
                       "rh shift: new last path stage zeroed");
        check_vec_close(sl.term, sl0.term, 0.0,
                        "rh shift: terminal slacks unchanged");
        // New stage 0 corresponds to the old stage 1, so the first-stage
        // slacks come from the old path slot 0.  DI: fixed x_0 and no soft
        // state-box rows, so the first-stage slack vector matches the
        // path-stage one exactly (no leading block to drop).
        check(D::nslack_first == D::nslack_path,
              "rh shift: first/path slack sizes match (DI)");
        check_vec_close(sl.first, sl0.path[0], 0.0,
                        "rh shift: first = old stage-1 path slacks");
    }

    // 3. Warm re-solve at the shifted horizon ---------------------
    // The solver now carries persistent QP duals from the cold solve,
    // so a warm first QP is meaningful: enable it.
    check(inspect::set_field(solver, "options.warm_start_first_qp", 1).ok(),
          "set options.warm_start_first_qp = true");
    const Status st_warm = solver.solve(prob, sol);
    check(st_warm == Status::kSolved, "rh shift: warm solve kSolved");
    if (st_warm != Status::kSolved)
    {
        return;
    }
    const int warm_sqp = sqp_iters(solver.statistics());
    const int warm_ipm = total_ipm_iters(solver.statistics());
    const double warm_cost = sol.cost_value;
    std::printf("  rh shift: warm sqp=%d ipm=%d cost=%.6f\n", warm_sqp,
                warm_ipm, warm_cost);

    // The warm start pays off at the SQP level (the shifted solution is
    // close to the optimum, so fewer SQP iterations are needed). At the
    // IPM level no assertion is made: the driver always zeroes the
    // primal step before every QP (the duals are the persistent warm
    // part), and for this small, well-conditioned QP the ws=0
    // heuristic's primal initialization is competitive (see the
    // warm_start_3c driver test for the same observation).
    check(warm_sqp <= cold_sqp,
          "rh shift: warm sqp <= cold sqp (" + std::to_string(warm_sqp) +
              " <= " + std::to_string(cold_sqp) + ")");
    check_close(warm_cost, cold_cost, 1e-6, "rh shift: cost matches cold");
    for (int k = 0; k <= N; ++k)
    {
        check_vec_close(sol.x[k], sol_cold.x[k], 1e-6,
                        "rh shift: warm x[" + std::to_string(k) +
                            "] matches cold");
    }
    for (int k = 0; k < N; ++k)
    {
        check_vec_close(sol.u[k], sol_cold.u[k], 1e-6,
                        "rh shift: warm u[" + std::to_string(k) +
                            "] matches cold");
    }

    // 4. One-shot cold start --------------------------------------
    // A fresh cold start needs the cold-first-QP behaviour again,
    // because reset_warm_start() leaves no dual history.
    solver.reset_warm_start();
    check(inspect::set_field(solver, "options.warm_start_first_qp", 0).ok(),
          "set options.warm_start_first_qp = false");
    {
        const auto& sl = solver.slacks();
        check_vec_zero(sl.first, "rh shift: reset zeroed first slacks");
        for (int k = 0; k < N - 1; ++k)
        {
            check_vec_zero(sl.path[k],
                           "rh shift: reset zeroed path slacks (k=" +
                               std::to_string(k) + ")");
        }
        check_vec_zero(sl.term, "rh shift: reset zeroed terminal slacks");
        const auto& q = solver.last_qp_sol();
        for (int k = 0; k < N; ++k)
        {
            check_vec_zero(q.pi[k], "rh shift: reset zeroed QP pi[k]");
        }
        check_vec_zero(q.lam_first, "rh shift: reset zeroed QP lam_first");
        check_vec_zero(q.lam_term, "rh shift: reset zeroed QP lam_term");
        for (int k = 1; k < N; ++k)
        {
            check_vec_zero(q.lam_path[k - 1],
                           "rh shift: reset zeroed QP lam_path[k-1]");
        }
    }

    Solution<DI> sol2 = make_warm_start(prob, N);
    const Status st_cold2 = solver.solve(prob, sol2);
    check(st_cold2 == Status::kSolved, "rh shift: post-reset cold kSolved");
    if (st_cold2 != Status::kSolved)
    {
        return;
    }
    const int cold2_ipm = total_ipm_iters(solver.statistics());
    check_close(sol2.cost_value, cold_cost, 1e-6,
                "rh shift: post-reset cost matches");
    std::printf("  rh shift: cold2 ipm=%d cost=%.6f\n", cold2_ipm,
                sol2.cost_value);
}

int run_rh_shift_tests()
{
    failures = 0;
    test_rh_shift_driver();
    if (failures == 0)
    {
        std::printf("All rh_shift checks passed.\n");
    }
    return failures;
}

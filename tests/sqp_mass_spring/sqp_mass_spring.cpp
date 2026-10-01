// Phase 2 sub-step 2h test: standalone end-to-end SQP run on the MassSpring
// (N = 20) with the default SqpSolver (HPIPM QP + GLM regularizer + merit
// backtracking). Box-only problem (no ineq/eq/lin, no soft rows): verifies
// the stage-type layout at a larger horizon with 3 controls and 8 states.
//
// Warm start: x[0] = initial_state(), u = 0, forward simulation (feasible:
// the state box [-4, 4]^8 is respected by the damped free response, and
// there is no terminal equality).
//
// Independent verification (recomputed from the problem interface, not from
// the solver): dynamics gap, box violations, complementarity with the
// solution multipliers, and stationarity via a finite-difference Lagrangian
// gradient (the SQP_PHASE2_PLAN sec. 1.3 sign-convention gate).
//
// Writes sqp_mass_spring.csv (SqpStatistics::write_csv) and prints the stat
// rows (print_level = 1).

#include <cmath>
#include <cstdio>
#include <cstring>

#include "ocp/solvers/acados/sqp.hpp"

#include "../../examples/mass_spring/mass_spring.hpp"

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

constexpr int N = 20;

Solution<MassSpring> warm_start(const MassSpring& p)
{
    Solution<MassSpring> sol(N);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }
    return sol;
}

// sec. 1.3 NLP Lagrangian with the multipliers frozen at `s`:
//   L = sum_k 0.5 (x'x + 2 u'u) + 0.5 x_N' x_N
//     + sum_k pi_k' (x_{k+1} - f_k)
//     + box sides:  lam_lo (lo - w) + lam_hi (w - hi)
// x_0 is pinned (no stage-0 state box); no ineq/eq/lin rows.
double lagrangian(const MassSpring& p, const Solution<MassSpring>& s)
{
    double L = 0.0;
    for (int k = 0; k < N; ++k)
    {
        L += p.stage_cost_value(k, s.x[k], s.u[k]);
        const auto f = p.dynamics_next_state(k, s.x[k], s.u[k]);
        L += s.lambda_dyn[k].dot(s.x[k + 1] - f);

        // control box, 3 rows
        for (int j = 0; j < 3; ++j)
        {
            L +=  s.lambda_box_control[k](j) * (-0.5 - s.u[k](j));
            L += s.lambda_box_control[k](3 + j) * (s.u[k](j) - 0.5);
        }
        // state box, 8 rows (absent at k = 0: x_0 pinned)
        if (k > 0)
        {
            for (int j = 0; j < 8; ++j)
            {
                L +=  s.lambda_box_state[k](j) * (-4.0 - s.x[k](j));
                L += s.lambda_box_state[k](8 + j) * (s.x[k](j) - 4.0);
            }
        }
    }
    L += p.terminal_cost_value(s.x[N]);
    // terminal state box, 8 rows
    for (int j = 0; j < 8; ++j)
    {
        L +=  s.lambda_box_state[N](j) * (-4.0 - s.x[N](j));
        L += s.lambda_box_state[N](8 + j) * (s.x[N](j) - 4.0);
    }
    return L;
}

// Inf-norm of the finite-difference Lagrangian gradient w.r.t. every free
// primal variable: u_k (k = 0..N-1, 3 each) and x_k (k = 1..N, 8 each;
// x_0 pinned).
double fd_lagrangian_grad_inf(const MassSpring& p,
                              const Solution<MassSpring>& s)
{
    const double h = 1e-5;
    double gmax = 0.0;
    for (int k = 0; k < N; ++k)
    {
        for (int j = 0; j < 3; ++j)
        {
            Solution<MassSpring> sp = s;
            Solution<MassSpring> sm = s;
            sp.u[k](j) += h;
            sm.u[k](j) -= h;
            gmax = std::max(gmax, std::fabs((lagrangian(p, sp) -
                                            lagrangian(p, sm)) / (2.0 * h)));
        }
    }
    for (int k = 1; k <= N; ++k)
    {
        for (int j = 0; j < 8; ++j)
        {
            Solution<MassSpring> sp = s;
            Solution<MassSpring> sm = s;
            sp.x[k](j) += h;
            sm.x[k](j) -= h;
            gmax = std::max(gmax, std::fabs((lagrangian(p, sp) -
                                            lagrangian(p, sm)) / (2.0 * h)));
        }
    }
    return gmax;
}

void check_residuals(const MassSpring& p, const Solution<MassSpring>& sol)
{
    // dynamics gap
    double gap = 0.0;
    for (int k = 0; k < N; ++k)
    {
        const auto f = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
        gap = std::max(gap,
            static_cast<double>((f - sol.x[k + 1]).cwiseAbs().maxCoeff()));
    }
    check(gap < 1e-8, "dynamics gap < 1e-8");

    // box violations (state incl. terminal, control)
    double viol = 0.0;
    for (int k = 1; k <= N; ++k)
    {
        for (int j = 0; j < 8; ++j)
        {
            const double w = sol.x[k](j);
            viol = std::max(viol, std::max(0.0, std::max(w - 4.0, -4.0 - w)));
        }
    }
    for (int k = 0; k < N; ++k)
    {
        for (int j = 0; j < 3; ++j)
        {
            const double w = sol.u[k](j);
            viol = std::max(viol, std::max(0.0, std::max(w - 0.5, -0.5 - w)));
        }
    }
    check(viol < 1e-8, "box violations < 1e-8");

    // complementarity with the solution multipliers
    double comp = 0.0;
    for (int k = 0; k < N; ++k)
    {
        for (int j = 0; j < 3; ++j)
        {
            const double w = sol.u[k](j);
            comp = std::max(comp, std::fabs(
                sol.lambda_box_control[k](j) * std::max(0.0, -0.5 - w)));
            comp = std::max(comp, std::fabs(
                sol.lambda_box_control[k](3 + j) * std::max(0.0, w - 0.5)));
        }
    }
    for (int k = 1; k <= N; ++k)
    {
        for (int j = 0; j < 8; ++j)
        {
            const double w = sol.x[k](j);
            comp = std::max(comp, std::fabs(
                sol.lambda_box_state[k](j) * std::max(0.0, -4.0 - w)));
            comp = std::max(comp, std::fabs(
                sol.lambda_box_state[k](8 + j) * std::max(0.0, w - 4.0)));
        }
    }
    check(comp < 1e-8, "complementarity < 1e-8");

    // stationarity: FD Lagrangian gradient (sec. 1.3 sign-convention gate)
    const double fd = fd_lagrangian_grad_inf(p, sol);
    check(fd < 1e-5, "FD Lagrangian gradient < 1e-5");
}

}  // namespace

int main()
{
    MassSpring p;
    SqpOptions opts;
    opts.print_level = 1;

    SqpSolver<MassSpring> solver(opts);
    Solution<MassSpring> sol = warm_start(p);

    const Status st = solver.solve(p, sol);
    check(st == Status::kSolved, "solve returned kSolved");
    check(sol.status == Status::kSolved, "sol.status kSolved");
    check(solver.statistics().iter >= 1, "at least one statistics row");

    check_residuals(p, sol);

    // x_0 preserved (fixed initial state)
    const auto x0 = p.initial_state();
    for (int i = 0; i < 8; ++i)
    {
        check_close(sol.x[0](i), x0(i), 1e-10, "x[0] preserved");
    }

    // box bounds
    for (int k = 0; k < N; ++k)
    {
        for (int j = 0; j < 3; ++j)
        {
            check(sol.u[k](j) >= -0.5 - 1e-8 && sol.u[k](j) <= 0.5 + 1e-8,
                  "control box holds");
        }
    }
    for (int k = 1; k <= N; ++k)
    {
        for (int j = 0; j < 8; ++j)
        {
            check(sol.x[k](j) >= -4.0 - 1e-8 && sol.x[k](j) <= 4.0 + 1e-8,
                  "state box holds");
        }
    }

    // CSV output
    solver.statistics().write_csv("sqp_mass_spring.csv");
    std::FILE* f = std::fopen("sqp_mass_spring.csv", "r");
    check(f != nullptr, "CSV file written");
    if (f != nullptr)
    {
        char hdr[128];
        check(std::fgets(hdr, sizeof(hdr), f) != nullptr &&
                  std::strstr(hdr, "res_stat") != nullptr,
              "CSV header");
        int rows = 0;
        char buf[256];
        while (std::fgets(buf, sizeof(buf), f) != nullptr)
        {
            ++rows;
        }
        check(rows == static_cast<int>(solver.statistics().iter),
              "CSV row count matches statistics");
        std::fclose(f);
    }

    if (failures > 0)
    {
        std::fprintf(stderr, "sqp_mass_spring: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("sqp_mass_spring: OK (%d SQP iterations, cost = %.10e)\n",
                solver.statistics().iter, sol.cost_value);
    return 0;
}

// sqp_p5probe.cpp
//
// p5probe (Phase 5): port of the acados "ocp_p5probe_ecaceb6d" OCP
// (ERK K4, N = 3, soft stage ineq h in [0,1], hard terminal ineq,
// non-convex h_t). Verifies:
//   (A) model fidelity at the acados reference trajectory (raw cost,
//       dynamics gap, constraint values);
//   (B) SQP convergence to a valid KKT point from a generic warm start
//       (final residuals from the solver's own slack-aware residuals).
//
// Reference: examples/p5probe/acados_codegen/ref_A_traj.txt.  The acados
// reference was run with FIXED_STEP globalization and dt-scaled stage
// costs; our port follows the codebase convention (stage cost without an
// explicit dt factor, merit-backtracking globalization), so a different
// local KKT point is expected (the terminal ineq makes the problem
// non-convex).  Hence (B) checks KKT validity, not trajectory agreement.

#include <cmath>
#include <cstdio>

#include "ocp/solvers/acados/sqp.hpp"
#include "P5probe.hpp"

namespace
{

int failures = 0;

void check(bool cond, const char* what)
{
    if (!cond)
    {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

void check_close(double v, double ref, double tol, const char* what)
{
    if (std::fabs(v - ref) > tol)
    {
        ++failures;
        std::fprintf(stderr, "FAIL: %s: %.6e vs %.6e (tol %.1e)\n", what,
                     v, ref, tol);
    }
}

}  // namespace

int main()
{
    using P5 = ocp::P5probe;
    P5 p;
    constexpr int N = 3;

    // acados reference trajectory (ref_A_traj.txt)
    const double Xref[4][2] = {
        { 8.933036e-05, 3.694790e-04 },
        { 1.254670e-04, 3.532551e-04 },
        { 7.063627e-05, -1.449872e-03 },
        { -1.410963e-06, 8.941362e-06 },
    };
    const double Uref[3][2] = {
        { 4.112374e-05, -4.069026e-04 },
        { 1.826072e-03, -3.971484e-02 },
        { 1.219550e-02, 4.785542e-03 },
    };

    // (A) model fidelity at the reference trajectory
    double raw = 0.0;
    for (int k = 0; k < N; ++k)
    {
        P5::state_t x;
        x << Xref[k][0], Xref[k][1];
        P5::control_t u;
        u << Uref[k][0], Uref[k][1];
        raw += p.stage_cost_value(k, x, u);
        const P5::state_t xn = p.dynamics_next_state(k, x, u);
        const double gap =
            std::max(std::abs(static_cast<double>(xn(0) - Xref[k + 1][0])),
                     std::abs(static_cast<double>(xn(1) - Xref[k + 1][1])));
        check(gap < 1e-7, "ref dynamics gap");
    }
    P5::state_t xe;
    xe << Xref[3][0], Xref[3][1];
    raw += p.terminal_cost_value(xe);
    check_close(raw, 9.614245e-07, 1e-9, "ref raw cost");

    // (B) SQP solve from a generic warm start
    ocp::SqpOptions opts;
    opts.print_level = 0;
    opts.max_iter = 100;
    opts.tol_stat = 1e-6;
    opts.tol_eq = 1e-6;
    opts.tol_ineq = 1e-6;
    opts.tol_comp = 1e-6;
    ocp::SqpSolver<P5> solver(opts);
    ocp::Solution<P5> sol(N);
    sol.x[0] << 0.5, 0.5;
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = p.cost_reference();
    }

    const ocp::Status st = solver.solve(p, sol);
    check(st == ocp::Status::kSolved, "status kSolved");
    check(std::isfinite(sol.cost_value) && sol.cost_value > 0.0,
          "cost finite and positive");

    const auto& rows = solver.statistics().rows;
    check(!rows.empty(), "statistics non-empty");
    if (!rows.empty())
    {
        const auto& r = rows.back();
        check(r.res_stat < 1e-6, "final res_stat < 1e-6");
        check(r.res_eq < 1e-6, "final res_eq < 1e-6");
        check(r.res_ineq < 1e-6, "final res_ineq < 1e-6");
        check(r.res_comp < 1e-6, "final res_comp < 1e-6");
    }

    // raw dynamics gap at the solution
    for (int k = 0; k < N; ++k)
    {
        const P5::state_t xn =
            p.dynamics_next_state(k, sol.x[k], sol.u[k]);
        const double gap = (xn - sol.x[k + 1]).cwiseAbs().maxCoeff();
        check(gap < 1e-7, "solution dynamics gap");
    }

    // hard terminal ineq must be satisfied (raw, no slacks)
    const auto ge = p.terminal_inequality_constr(sol.x[N]);
    check(static_cast<double>(ge(0)) < 1e-9, "terminal ineq lo satisfied");
    check(static_cast<double>(ge(1)) < 1e-9, "terminal ineq hi satisfied");

    std::printf("sqp_p5probe: cost=%.6f iters=%d status=%d\n",
                sol.cost_value, solver.statistics().iter, static_cast<int>(st));
    if (failures > 0)
    {
        std::fprintf(stderr, "sqp_p5probe: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("sqp_p5probe: OK\n");
    return 0;
}

// sqp_p5probe_b.cpp
//
// p5probe_b (Phase 5b): port of the acados "ocp_p5probe_b" OCP
// (IRK Gauss-Legendre 2 stages, N = 3, EXTERNAL cost, no constraints,
// all-zeros optimum). Verifies:
//   (A) model fidelity at the acados reference (all zeros): cost ~ 0,
//       dynamics residual ~ 0;
//   (B) SQP from a small nonzero warm start converges to the zero
//       trajectory with valid KKT residuals.

#include <cmath>
#include <cstdio>

#include "ocp/solvers/acados/sqp.hpp"
#include "P5probeB.hpp"

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

}  // namespace

int main()
{
    using P5B = ocp::P5probeB;
    P5B p;
    constexpr int N = 3;

    // (A) model fidelity at the all-zeros reference
    P5B::state_t xz;
    xz.setZero();
    P5B::control_t uz;
    uz.setZero();
    for (int k = 0; k < N; ++k)
    {
        const double c = p.stage_cost_value(k, xz, uz);
        check(std::fabs(c) < 1e-12, "ref stage cost ~ 0");
        const P5B::state_t xn = p.dynamics_next_state(k, xz, uz);
        check(xn.cwiseAbs().maxCoeff() < 1e-12, "ref dynamics ~ 0");
    }
    check(std::fabs(p.terminal_cost_value(xz)) < 1e-12,
          "ref terminal cost ~ 0");

    // (B) SQP from a small nonzero warm start
    ocp::SqpOptions opts;
    opts.print_level = 0;
    opts.max_iter = 100;
    opts.tol_stat = 1e-6;
    opts.tol_eq = 1e-6;
    opts.tol_ineq = 1e-6;
    opts.tol_comp = 1e-6;
    ocp::SqpSolver<P5B> solver(opts);
    ocp::Solution<P5B> sol(N);
    sol.x[0] << 0.1, 0.2;
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = p.cost_reference();
    }

    const ocp::Status st = solver.solve(p, sol);
    check(st == ocp::Status::kSolved, "status kSolved");
    check(std::isfinite(sol.cost_value) && sol.cost_value >= 0.0,
          "cost finite non-negative");
    check(sol.cost_value < 1e-4, "cost near zero optimum");

    const auto& rows = solver.statistics().rows;
    check(!rows.empty(), "statistics non-empty");
    if (!rows.empty())
    {
        const auto& r = rows.back();
        check(r.res_stat < 1e-6, "final res_stat < 1e-6");
        check(r.res_eq < 1e-6, "final res_eq < 1e-6");
    }

    // trajectory should be (near) the zero solution
    for (int k = 0; k <= N; ++k)
        check(sol.x[k].cwiseAbs().maxCoeff() < 1e-3,
              "x_k near zero");
    for (int k = 0; k < N; ++k)
        check(sol.u[k].cwiseAbs().maxCoeff() < 1e-3, "u_k near zero");

    std::printf("sqp_p5probe_b: cost=%.6f iters=%d status=%d\n",
                sol.cost_value, solver.statistics().iter,
                static_cast<int>(st));
    if (failures > 0)
    {
        std::fprintf(stderr, "sqp_p5probe_b: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("sqp_p5probe_b: OK\n");
    return 0;
}

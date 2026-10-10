// sqp_p5probe_irk.cpp
//
// p5probe_irk (Phase 5b, implicit RK): port of the acados "ocp_p5probe_b" OCP
// (IRK Gauss-Legendre 2 stages, N = 3, EXTERNAL cost, no constraints,
// all-zeros optimum). Verifies:
//   (A) model fidelity at the acados reference (all zeros): cost ~ 0,
//       dynamics residual ~ 0;
//   (B) SQP from a small nonzero warm start converges to the zero
//       trajectory with valid KKT residuals.

#include <cmath>
#include <cstdio>

#include "ocp/solvers/acados/acados.hpp"
#include "P5probeIrk.hpp"

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
    using P5B = ocp::P5probeIrk;
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

    // (A2) analytic cost derivatives cross-checked against finite
    //      differences of the (independently generated) value functions
    {
        constexpr int nxf = P5B::nx;
        constexpr int nuf = P5B::nu;
        constexpr int m = nxf + nuf;
        P5B::state_t x;
        x << 0.31, -0.57;
        P5B::control_t u;
        u << 0.42;
        const double h = 1.0e-5;

        // stage (k=1 -> path cost path)
        const auto sval = [&](const P5B::state_t& xs,
                              const P5B::control_t& us) {
            return p.stage_cost_value(1, xs, us);
        };
        double sv;
        P5B::stage_grad_t sg;
        p.stage_cost_value_grad(1, x, u, sv, sg);
        for (int i = 0; i < nxf; ++i)
        {
            P5B::state_t a = x, b = x, c = x, d = x;
            a(i) += 2 * h; b(i) += h; c(i) -= h; d(i) -= 2 * h;
            const double fd = (-sval(a, u) + 8 * sval(b, u)
                              - 8 * sval(c, u) + sval(d, u)) / (12.0 * h);
            check(std::fabs(sg(i) - fd) < 1e-6, "stage grad x vs FD");
        }
        for (int j = 0; j < nuf; ++j)
        {
            P5B::control_t a = u, b = u, c = u, d = u;
            P5B::state_t x0 = x;
            a(j) += 2 * h; b(j) += h; c(j) -= h; d(j) -= 2 * h;
            const double fd = (-sval(x0, a) + 8 * sval(x0, b)
                              - 8 * sval(x0, c) + sval(x0, d)) / (12.0 * h);
            check(std::fabs(sg(nxf + j) - fd) < 1e-6, "stage grad u vs FD");
        }
        P5B::stage_hess_t SH;
        p.stage_cost_value_grad_hess(1, x, u, sv, sg, SH);
        const double h2 = 1.0e-4;
        for (int j = 0; j < m; ++j)
        {
            P5B::state_t xp = x, xm = x;
            P5B::control_t up = u, um = u;
            if (j < nxf) { xp(j) += h2; xm(j) -= h2; }
            else         { up(j - nxf) += h2; um(j - nxf) -= h2; }
            double vv;
            P5B::stage_grad_t gp, gm;
            p.stage_cost_value_grad(1, xp, up, vv, gp);
            p.stage_cost_value_grad(1, xm, um, vv, gm);
            for (int i = 0; i < m; ++i)
                check(std::fabs(SH(i, j) - (gp(i) - gm(i)) / (2.0 * h2)) < 1e-5,
                      "stage hess vs FD");
        }

        // terminal
        const auto tval = [&](const P5B::state_t& xs) {
            return p.terminal_cost_value(xs);
        };
        double tv;
        P5B::term_grad_t tg;
        p.terminal_cost_value_grad(x, tv, tg);
        for (int i = 0; i < nxf; ++i)
        {
            P5B::state_t a = x, b = x, c = x, d = x;
            a(i) += 2 * h; b(i) += h; c(i) -= h; d(i) -= 2 * h;
            const double fd = (-tval(a) + 8 * tval(b)
                              - 8 * tval(c) + tval(d)) / (12.0 * h);
            check(std::fabs(tg(i) - fd) < 1e-6, "term grad vs FD");
        }
        P5B::term_hess_t TH;
        p.terminal_cost_value_grad_hess(x, tv, tg, TH);
        for (int j = 0; j < nxf; ++j)
        {
            P5B::state_t xp = x, xm = x;
            xp(j) += h2; xm(j) -= h2;
            double vv;
            P5B::term_grad_t gp, gm;
            p.terminal_cost_value_grad(xp, vv, gp);
            p.terminal_cost_value_grad(xm, vv, gm);
            for (int i = 0; i < nxf; ++i)
                check(std::fabs(TH(i, j) - (gp(i) - gm(i)) / (2.0 * h2)) < 1e-5,
                      "term hess vs FD");
        }
    }

    // (B) SQP from a small nonzero warm start
    ocp::AcadosOptions opts;
    opts.print_level = 0;
    opts.max_iter = 100;
    opts.tol_stat = 1e-6;
    opts.tol_eq = 1e-6;
    opts.tol_ineq = 1e-6;
    opts.tol_comp = 1e-6;
    ocp::AcadosSolver<P5B> solver(opts);
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

    std::printf("sqp_p5probe_irk: cost=%.6f iters=%d status=%d\n",
                sol.cost_value, solver.statistics().iter,
                static_cast<int>(st));
    if (failures > 0)
    {
        std::fprintf(stderr, "sqp_p5probe_irk: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("sqp_p5probe_irk: OK\n");
    return 0;
}

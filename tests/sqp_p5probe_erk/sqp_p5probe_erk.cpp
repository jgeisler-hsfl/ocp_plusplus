// sqp_p5probe_erk.cpp
//
// p5probe_erk (Phase 5, explicit RK): port of the acados "ocp_p5probe_ecaceb6d" OCP
// (ERK K4, N = 3, soft stage ineq h in [0,1], hard terminal ineq,
// non-convex h_t). Verifies:
//   (A) model fidelity at the acados reference trajectory (raw cost,
//       dynamics gap, constraint values);
//   (B) SQP convergence to a valid KKT point from a generic warm start
//       (final residuals from the solver's own slack-aware residuals).
//
// Reference: examples/p5probe_erk/acados_codegen/ref_A_traj.txt.  The acados
// reference was run with FIXED_STEP globalization and dt-scaled stage
// costs; our port follows the codebase convention (stage cost without an
// explicit dt factor, merit-backtracking globalization), so a different
// local KKT point is expected (the terminal ineq makes the problem
// non-convex).  Hence (B) checks KKT validity, not trajectory agreement.

#include <cmath>
#include <cstdio>

#include "ocp/solvers/acados/acados.hpp"
#include "P5probeErk.hpp"

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
    using P5 = ocp::P5probeErk;
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

    // (A2) analytic cost gradient cross-checked against finite differences
    //      of the (independently generated) y value functions.  The NONLINEAR
    //      LS gradient is J' W (y - yref) = J' g, an exact first derivative, so
    //      a generic nonzero g pins down the full wired y-Jacobian.  (The
    //      stage Hessian is the Gauss-Newton J'WJ and is NOT a finite
    //      difference of the gradient for a nonlinear y, so it is not checked
    //      here.)
    {
        constexpr int nxf = P5::nx;
        constexpr int nuf = P5::nu;
        P5::state_t x;
        x << 0.31, -0.57;
        P5::control_t u;
        u << 0.42, -0.13;
        const double h = 1.0e-5;

        // stage (k = 1 -> path cost)
        const auto sval = [&](const P5::state_t& xs,
                              const P5::control_t& us) {
            return p.stage_cost_value(1, xs, us);
        };
        double sv;
        P5::stage_grad_t sg;
        p.stage_cost_value_grad(1, x, u, sv, sg);
        for (int i = 0; i < nxf; ++i)
        {
            P5::state_t a = x, b = x, c = x, d = x;
            a(i) += 2 * h; b(i) += h; c(i) -= h; d(i) -= 2 * h;
            const double fd = (-sval(a, u) + 8 * sval(b, u)
                              - 8 * sval(c, u) + sval(d, u)) / (12.0 * h);
            check(std::fabs(sg(i) - fd) < 1e-6, "stage grad x vs FD");
        }
        for (int j = 0; j < nuf; ++j)
        {
            P5::control_t a = u, b = u, c = u, d = u;
            P5::state_t x0 = x;
            a(j) += 2 * h; b(j) += h; c(j) -= h; d(j) -= 2 * h;
            const double fd = (-sval(x0, a) + 8 * sval(x0, b)
                              - 8 * sval(x0, c) + sval(x0, d)) / (12.0 * h);
            check(std::fabs(sg(nxf + j) - fd) < 1e-6, "stage grad u vs FD");
        }

        // terminal
        const auto tval = [&](const P5::state_t& xs) {
            return p.terminal_cost_value(xs);
        };
        double tv;
        P5::term_grad_t tg;
        p.terminal_cost_value_grad(x, tv, tg);
        for (int i = 0; i < nxf; ++i)
        {
            P5::state_t a = x, b = x, c = x, d = x;
            a(i) += 2 * h; b(i) += h; c(i) -= h; d(i) -= 2 * h;
            const double fd = (-tval(a) + 8 * tval(b)
                              - 8 * tval(c) + tval(d)) / (12.0 * h);
            check(std::fabs(tg(i) - fd) < 1e-6, "term grad vs FD");
        }
    }

    // (A3) analytic h-constraint Jacobians (wired from the casadi
    //      constr_h[_e]_fun_jac_uxt_zt functions) cross-checked against
    //      finite differences of the (independently generated) h values.
    {
        constexpr int nxf = P5::nx;
        constexpr int nuf = P5::nu;
        constexpr int ngr = P5::ng;
        P5::state_t x;
        x << 0.31, -0.57;
        P5::control_t u;
        u << 0.42, -0.13;
        const double h = 1.0e-5;

        // stage
        P5::ineq_t g;
        P5::ineq_dg_dx_t g_dx;
        P5::ineq_dg_du_t g_du;
        p.stage_inequality_value_jac(1, x, u, g, g_dx, g_du);
        for (int j = 0; j < nxf; ++j)
        {
            P5::state_t a = x, b = x, c = x, d = x;
            a(j) += 2 * h; b(j) += h; c(j) -= h; d(j) -= 2 * h;
            const auto ga = p.stage_inequality_value(1, a, u);
            const auto gb = p.stage_inequality_value(1, b, u);
            const auto gc = p.stage_inequality_value(1, c, u);
            const auto gd = p.stage_inequality_value(1, d, u);
            for (int i = 0; i < ngr; ++i)
            {
                const double fd =
                    (-ga(i) + 8 * gb(i) - 8 * gc(i) + gd(i)) / (12.0 * h);
                check(std::fabs(g_dx(i, j) - fd) < 1e-6,
                      "stage ineq dg/dx vs FD");
            }
        }
        for (int j = 0; j < nuf; ++j)
        {
            P5::control_t a = u, b = u, c = u, d = u;
            P5::state_t x0 = x;
            a(j) += 2 * h; b(j) += h; c(j) -= h; d(j) -= 2 * h;
            const auto ga = p.stage_inequality_value(1, x0, a);
            const auto gb = p.stage_inequality_value(1, x0, b);
            const auto gc = p.stage_inequality_value(1, x0, c);
            const auto gd = p.stage_inequality_value(1, x0, d);
            for (int i = 0; i < ngr; ++i)
            {
                const double fd =
                    (-ga(i) + 8 * gb(i) - 8 * gc(i) + gd(i)) / (12.0 * h);
                check(std::fabs(g_du(i, j) - fd) < 1e-6,
                      "stage ineq dg/du vs FD");
            }
        }

        // terminal
        P5::ineq_term_t gt;
        P5::ineq_term_dg_dx_t gt_dx;
        p.terminal_inequality_value_jac(x, gt, gt_dx);
        for (int j = 0; j < nxf; ++j)
        {
            P5::state_t a = x, b = x, c = x, d = x;
            a(j) += 2 * h; b(j) += h; c(j) -= h; d(j) -= 2 * h;
            const auto ga = p.terminal_inequality_value(a);
            const auto gb = p.terminal_inequality_value(b);
            const auto gc = p.terminal_inequality_value(c);
            const auto gd = p.terminal_inequality_value(d);
            for (int i = 0; i < ngr; ++i)
            {
                const double fd =
                    (-ga(i) + 8 * gb(i) - 8 * gc(i) + gd(i)) / (12.0 * h);
                check(std::fabs(gt_dx(i, j) - fd) < 1e-6,
                      "term ineq dg/dx vs FD");
            }
        }
    }

    // (A4) ODE RHS Jacobian gate: the analytic jacobian() (wired from
    //      expl_vde_forw, called with Sx=I, Su=0) must match 5-point FD of
    //      the (independently generated) expl_ode_fun at several points.
    {
        using Ode = ocp::P5probeErkOde;
        constexpr int nxf = P5::nx;
        constexpr int nuf = P5::nu;
        const double pts[3][4] = {
            { 0.31, -0.57, 0.42, -0.13 },
            { -1.7,  2.2,  0.0,  1.0 },
            { 0.0,   0.0,  0.0,  0.0 },
        };
        const double h = 1.0e-5;
        for (int r = 0; r < 3; ++r)
        {
            Ode ode;
            P5::state_t x;
            x << pts[r][0], pts[r][1];
            P5::control_t u;
            u << pts[r][2], pts[r][3];
            const double t = 0.05 * (r + 1);
            Ode::dyn_df_dx_t Jx;
            Ode::dyn_df_du_t Ju;
            ode.jacobian(x, u, t, Jx, Ju);
            for (int j = 0; j < nxf; ++j)
            {
                P5::state_t a = x, b = x, c = x, d = x;
                a(j) += 2 * h; b(j) += h; c(j) -= h; d(j) -= 2 * h;
                for (int i = 0; i < nxf; ++i)
                {
                    const double fd =
                        (-ode.f(a, u, t)(i) + 8 * ode.f(b, u, t)(i)
                         - 8 * ode.f(c, u, t)(i) + ode.f(d, u, t)(i))
                        / (12.0 * h);
                    check(std::fabs(Jx(i, j) - fd) < 1e-8, "ode Jx vs FD");
                }
            }
            for (int j = 0; j < nuf; ++j)
            {
                P5::control_t a = u, b = u, c = u, d = u;
                P5::state_t x0 = x;
                a(j) += 2 * h; b(j) += h; c(j) -= h; d(j) -= 2 * h;
                for (int i = 0; i < nxf; ++i)
                {
                    const double fd =
                        (-ode.f(x0, a, t)(i) + 8 * ode.f(x0, b, t)(i)
                         - 8 * ode.f(x0, c, t)(i) + ode.f(x0, d, t)(i))
                        / (12.0 * h);
                    check(std::fabs(Ju(i, j) - fd) < 1e-8, "ode Ju vs FD");
                }
            }
        }
    }

    // (B) SQP solve from a generic warm start
    ocp::AcadosOptions opts;
    opts.print_level = 0;
    opts.max_iter = 100;
    opts.tol_stat = 1e-6;
    opts.tol_eq = 1e-6;
    opts.tol_ineq = 1e-6;
    opts.tol_comp = 1e-6;
    ocp::AcadosSolver<P5> solver(opts);
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
    const auto ge = p.terminal_inequality_value(sol.x[N]);
    check(static_cast<double>(ge(0)) < 1e-9, "terminal ineq lo satisfied");
    check(static_cast<double>(ge(1)) < 1e-9, "terminal ineq hi satisfied");

    std::printf("sqp_p5probe_erk: cost=%.6f iters=%d status=%d\n",
                sol.cost_value, solver.statistics().iter, static_cast<int>(st));
    if (failures > 0)
    {
        std::fprintf(stderr, "sqp_p5probe_erk: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("sqp_p5probe_erk: OK\n");
    return 0;
}

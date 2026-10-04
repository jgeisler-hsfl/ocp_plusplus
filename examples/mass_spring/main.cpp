// Sanity test for the mass_spring example (ocp++ problem interface):
//
// - checks compile-time dimensions
// - verifies the discrete dynamics map A x + B u against an independent
//   RK4 simulation of the continuous-time system over one sampling interval
// - verifies the dynamics Jacobian and the stage / terminal cost
//   gradient and Hessian against central finite differences
// - forward-simulates the acados horizon (N = 20) and checks the
//   box-constraint specs and trajectory feasibility
// - writes the simulated trajectory (states, controls, stage / terminal
//   costs) to mass_spring_simulation.csv

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

#include "mass_spring.hpp"

static int nfail = 0;

static void check_close(const char* name, double a, double b, double tol = 1e-6)
{
    if (std::fabs(a - b) > tol * (1.0 + std::fabs(b)))
    {
        std::printf("FAIL %s: got %g, expected %g (diff %g)\n", name, a, b, a - b);
        ++nfail;
    }
    else
    {
        std::printf("ok   %s: %g\n", name, a);
    }
}

int main()
{
    // ---- compile-time dimension checks --------------------------------
    static_assert(MassSpring::nx == 8, "nx");
    static_assert(MassSpring::nu == 3, "nu");
    static_assert(MassSpring::ng == 0, "ng");
    static_assert(MassSpring::ne == 0, "ne");
    static_assert(MassSpring::nl == 0, "nl");
    static_assert(MassSpring::ng_t == 0, "ng_t");
    static_assert(MassSpring::ne_t == 0, "ne_t");
    static_assert(MassSpring::nl_t == 0, "nl_t");
    static_assert(MassSpring::nbx == 8, "nbx");
    static_assert(MassSpring::nbu == 3, "nbu");
    static_assert(MassSpring::nbx_t == 8, "nbx_t");
    static_assert(MassSpring::fixed_initial_state, "fixed x_0");
    static_assert(MassSpring::ng_soft == 0, "ng_soft");
    static_assert(MassSpring::ne_soft == 0, "ne_soft");
    static_assert(MassSpring::nl_soft == 0, "nl_soft");
    static_assert(MassSpring::nbx_soft == 0, "nbx_soft");
    static_assert(MassSpring::nbu_soft == 0, "nbu_soft");
    static_assert(MassSpring::ng_t_soft == 0, "ng_t_soft");
    static_assert(MassSpring::ne_t_soft == 0, "ne_t_soft");
    static_assert(MassSpring::nl_t_soft == 0, "nl_t_soft");
    static_assert(MassSpring::nbx_t_soft == 0, "nbx_t_soft");
    static_assert(MassSpring::stage_grad_t::RowsAtCompileTime == 11, "grad layout [x;u]");
    static_assert(MassSpring::stage_hess_t::RowsAtCompileTime == 11, "hess layout [x;u]");

    const int N = 20;  // same horizon as the acados example
    MassSpring problem;
    ocp::Solution<MassSpring> sol(N);  // constructor resizes all trajectories
    if (sol.x.size() != static_cast<std::size_t>(N + 1) ||
        sol.u.size() != static_cast<std::size_t>(N))
    {
        std::printf("FAIL: trajectory extents\n");
        return 1;
    }

    // ---- forward simulation over the acados horizon -------------------
    sol.x[0] = problem.initial_state();
    check_close("x0(0)", sol.x[0](0), 2.5, 1e-12);
    check_close("x0(1)", sol.x[0](1), 2.5, 1e-12);
    check_close("x0(2)", sol.x[0](2), 0.0, 1e-12);

    MassSpring::control_t u;
    u << 0.1, -0.2, 0.3;
    for (int k = 0; k < N; ++k)
    {
        sol.u[k] = u;
        sol.x[k + 1] = problem.dynamics_next_state(k, sol.x[k], u);
    }
    double max_x = 0.0;
    for (int k = 0; k <= N; ++k)
    {
        for (int i = 0; i < MassSpring::nx; ++i)
        {
            max_x = std::max(max_x, std::fabs(sol.x[k](i)));
        }
    }
    if (max_x > 4.0)
    {
        std::printf("FAIL: simulated trajectory leaves the state box (max |x| = %g)\n",
                    max_x);
        ++nfail;
    }
    else
    {
        std::printf("ok   simulated trajectory stays in the state box (max |x| = %g)\n",
                    max_x);
    }

    // ---- discrete dynamics map vs independent RK4 simulation ----------
    // The expm-based discretization (A_, B_) is checked against a fine RK4
    // integration of the continuous system q_dot = v, v_dot = T q + Bc u.
    {
        using Vec4 = Eigen::Matrix<double, MassSpring::nmass, 1>;
        MassSpring::state_t x;
        MassSpring::control_t uu;
        x << 0.3, -0.2, 1.1, 0.4, 0.7, -0.5, 0.2, 0.9;
        uu << 0.1, -0.2, 0.3;

        const MassSpring::state_t xd = problem.dynamics_next_state(0, x, uu);

        const int n = 2000;
        const double hs = problem.Ts_ / n;
        Vec4 q = x.head<MassSpring::nmass>();
        Vec4 v = x.tail<MassSpring::nmass>();
        auto deriv = [&](const Vec4& qq, const Vec4& vv)
        {
            Vec4 dv = problem.T_ * qq;
            for (int i = 0; i < MassSpring::nu; ++i)
            {
                dv(i) += uu(i);
            }
            return std::pair<Vec4, Vec4>(vv, dv);
        };
        for (int s = 0; s < n; ++s)
        {
            auto k1 = deriv(q, v);
            auto k2 = deriv(q + 0.5 * hs * k1.first, v + 0.5 * hs * k1.second);
            auto k3 = deriv(q + 0.5 * hs * k2.first, v + 0.5 * hs * k2.second);
            auto k4 = deriv(q + hs * k3.first, v + hs * k3.second);
            q += (hs / 6.0) * (k1.first + 2.0 * k2.first + 2.0 * k3.first + k4.first);
            v += (hs / 6.0) * (k1.second + 2.0 * k2.second + 2.0 * k3.second + k4.second);
        }
        char name[32];
        for (int i = 0; i < MassSpring::nmass; ++i)
        {
            std::snprintf(name, sizeof(name), "map q(%d) vs RK4", i);
            check_close(name, xd(i), q(i), 1e-9);
            std::snprintf(name, sizeof(name), "map v(%d) vs RK4", i);
            check_close(name, xd(MassSpring::nmass + i), v(i), 1e-9);
        }
    }

    // ---- dynamics Jacobian vs finite differences ----------------------
    {
        MassSpring::state_t x;
        MassSpring::control_t u;
        x << 0.3, -0.2, 1.1, 0.4, 0.7, -0.5, 0.2, 0.9;
        u << 0.1, -0.2, 0.3;

        MassSpring::dyn_df_dx_t A;
        MassSpring::dyn_df_du_t B;
        MassSpring::state_t x_next;
        problem.dynamics_value_jac(0, x, u, x_next, A, B);
        check_close("dyn A == A_", (A - problem.A_).cwiseAbs().maxCoeff(), 0.0, 1e-12);
        check_close("dyn B == B_", (B - problem.B_).cwiseAbs().maxCoeff(), 0.0, 1e-12);

        const double h = 1e-6;
        double maxdiff = 0.0;
        for (int j = 0; j < MassSpring::nx; ++j)
        {
            MassSpring::state_t xp = x, xm = x;
            xp(j) += h;
            xm(j) -= h;
            const auto fp = problem.dynamics_next_state(0, xp, u);
            const auto fm = problem.dynamics_next_state(0, xm, u);
            for (int i = 0; i < MassSpring::nx; ++i)
            {
                maxdiff = std::max(maxdiff,
                    std::fabs((fp(i) - fm(i)) / (2.0 * h) - A(i, j)));
            }
        }
        for (int j = 0; j < MassSpring::nu; ++j)
        {
            MassSpring::control_t up = u, um = u;
            up(j) += h;
            um(j) -= h;
            const auto fp = problem.dynamics_next_state(0, x, up);
            const auto fm = problem.dynamics_next_state(0, x, um);
            for (int i = 0; i < MassSpring::nx; ++i)
            {
                maxdiff = std::max(maxdiff,
                    std::fabs((fp(i) - fm(i)) / (2.0 * h) - B(i, j)));
            }
        }
        check_close("dyn jac max |fd - J|", maxdiff, 0.0, 1e-4);
    }

    // ---- stage cost gradient / Hessian vs finite differences ----------
    {
        const int k = 3;
        MassSpring::state_t x;
        MassSpring::control_t u;
        x << 0.3, -0.2, 1.1, 0.4, 0.7, -0.5, 0.2, 0.9;
        u << 0.1, -0.2, 0.3;

        MassSpring::stage_grad_t g;
        double lcost = 0.0;
        problem.stage_cost_value_grad(k, x, u, lcost, g);
        const double h = 1e-6;
        double maxdiff = 0.0;
        for (int j = 0; j < MassSpring::nx + MassSpring::nu; ++j)
        {
            MassSpring::state_t xp = x, xm = x;
            MassSpring::control_t up = u, um = u;
            if (j < MassSpring::nx)
            {
                xp(j) += h;
                xm(j) -= h;
            }
            else
            {
                const int jj = j - MassSpring::nx;
                up(jj) += h;
                um(jj) -= h;
            }
            const double fd = (problem.stage_cost_value(k, xp, up) -
                               problem.stage_cost_value(k, xm, um)) / (2.0 * h);
            maxdiff = std::max(maxdiff, std::fabs(fd - g(j)));
        }
        check_close("stage cost grad max |fd - g|", maxdiff, 0.0, 1e-4);

        MassSpring::stage_hess_t H;
        problem.stage_cost_value_grad_hess(k, x, u, lcost, g, H);
        double hmax = 0.0;
        for (int i = 0; i < MassSpring::nx + MassSpring::nu; ++i)
        {
            for (int j = 0; j < MassSpring::nx + MassSpring::nu; ++j)
            {
                const double ref = (i == j) ? (i < MassSpring::nx ? 1.0 : 2.0) : 0.0;
                hmax = std::max(hmax, std::fabs(H(i, j) - ref));
            }
        }
        check_close("stage cost hess max err (I, 2I blocks)", hmax, 0.0, 1e-12);

        // terminal cost: gradient = x, Hessian = I
        MassSpring::term_grad_t gt;
        double tval = 0.0;
        problem.terminal_cost_value_grad(x, tval, gt);
        double gmax = 0.0;
        for (int j = 0; j < MassSpring::nx; ++j)
        {
            MassSpring::state_t xp = x, xm = x;
            xp(j) += h;
            xm(j) -= h;
            const double fd = (problem.terminal_cost_value(xp) -
                               problem.terminal_cost_value(xm)) / (2.0 * h);
            gmax = std::max(gmax, std::fabs(fd - gt(j)));
        }
        check_close("term cost grad max |fd - g|", gmax, 0.0, 1e-4);
        MassSpring::term_hess_t Ht;
        problem.terminal_cost_value_grad_hess(x, tval, gt, Ht);
        check_close("term cost hess == I",
                    (Ht - MassSpring::term_hess_t::Identity()).cwiseAbs().maxCoeff(),
                    0.0, 1e-12);
    }

    // ---- box constraint specs ------------------------------------------
    {
        const auto cb = problem.stage_control_box_constr(0);
        double d = 0.0;
        for (int i = 0; i < MassSpring::nbu; ++i)
        {
            d = std::max(d, std::max(std::fabs(cb.lo(i) + 0.5),
                                     std::fabs(cb.hi(i) - 0.5)));
        }
        check_close("ctrl box bounds [-0.5; 0.5]", d, 0.0, 1e-12);
        check_close("ctrl box soft penalty", cb.soft_penalty.maxCoeff(), 0.0, 1e-12);

        const auto sb = problem.stage_state_box_constr(0);
        d = 0.0;
        for (int i = 0; i < MassSpring::nbx; ++i)
        {
            d = std::max(d, std::max(std::fabs(sb.lo(i) + 4.0),
                                     std::fabs(sb.hi(i) - 4.0)));
        }
        check_close("state box bounds [-4; 4]", d, 0.0, 1e-12);
        check_close("state box soft penalty", sb.soft_penalty.maxCoeff(), 0.0, 1e-12);

        const auto tsb = problem.terminal_state_box_constr();
        d = 0.0;
        for (int i = 0; i < MassSpring::nbx_t; ++i)
        {
            d = std::max(d, std::max(std::fabs(tsb.lo(i) + 4.0),
                                     std::fabs(tsb.hi(i) - 4.0)));
        }
        check_close("term state box bounds [-4; 4]", d, 0.0, 1e-12);
    }

    // ---- CSV output of the simulated trajectory ------------------------
    // one row per stage k = 0..N: state x_k, control u_k (blank at k = N),
    // stage cost L_k(x_k, u_k) for k < N, terminal cost L_N(x_N) for k = N
    {
        const char* csv_name = "mass_spring_simulation.csv";
        std::FILE* csv = std::fopen(csv_name, "w");
        if (csv == nullptr)
        {
            std::printf("FAIL: could not open %s for writing\n", csv_name);
            ++nfail;
        }
        else
        {
            std::fprintf(csv, "k");
            for (int i = 0; i < MassSpring::nx; ++i)
            {
                std::fprintf(csv, ",x%d", i);
            }
            for (int i = 0; i < MassSpring::nu; ++i)
            {
                std::fprintf(csv, ",u%d", i);
            }
            std::fprintf(csv, ",cost\n");
            for (int k = 0; k <= N; ++k)
            {
                std::fprintf(csv, "%d", k);
                for (int i = 0; i < MassSpring::nx; ++i)
                {
                    std::fprintf(csv, ",%.12e", sol.x[k](i));
                }
                if (k < N)
                {
                    for (int i = 0; i < MassSpring::nu; ++i)
                    {
                        std::fprintf(csv, ",%.12e", sol.u[k](i));
                    }
                    std::fprintf(csv, ",%.12e\n",
                                 problem.stage_cost_value(k, sol.x[k], sol.u[k]));
                }
                else
                {
                    for (int i = 0; i < MassSpring::nu; ++i)
                    {
                        std::fprintf(csv, ",");
                    }
                    std::fprintf(csv, ",%.12e\n",
                                 problem.terminal_cost_value(sol.x[k]));
                }
            }
            std::fclose(csv);
            std::printf("ok   wrote %s (%d rows)\n", csv_name, N + 1);
        }
    }

    if (nfail == 0)
        std::printf("\nAll interface checks passed.\n");
    else
        std::printf("\n%d check(s) FAILED.\n", nfail);
    return nfail == 0 ? 0 : 1;
}

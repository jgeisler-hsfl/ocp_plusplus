// Sanity test for the ocp++ problem interface.
//
// - checks compile-time dimensions
// - forward-simulates the dynamics
// - verifies stage cost gradient / Hessian and dynamics Jacobian
//   against central finite differences
// - evaluates all constraint groups

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "double_integrator.hpp"

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
    static_assert(DoubleIntegrator::nx == 2, "nx");
    static_assert(DoubleIntegrator::nu == 1, "nu");
    static_assert(DoubleIntegrator::ng == 1, "ng");
    static_assert(DoubleIntegrator::ne == 0, "ne");
    static_assert(DoubleIntegrator::nl == 1, "nl");
    static_assert(DoubleIntegrator::ng_t == 0, "ng_t");
    static_assert(DoubleIntegrator::ne_t == 1, "ne_t");
    static_assert(DoubleIntegrator::nl_t == 0, "nl_t");
    static_assert(DoubleIntegrator::nbx == 2, "nbx");
    static_assert(DoubleIntegrator::nbu == 1, "nbu");
    static_assert(DoubleIntegrator::nbx_t == 2, "nbx_t");
    static_assert(DoubleIntegrator::fixed_initial_state, "fixed x_0");
    static_assert(DoubleIntegrator::ng_soft == 1, "ng_soft");
    static_assert(DoubleIntegrator::ne_soft == 0, "ne_soft");
    static_assert(DoubleIntegrator::nl_soft == 0, "nl_soft");
    static_assert(DoubleIntegrator::nbx_soft == 0, "nbx_soft");
    static_assert(DoubleIntegrator::nbu_soft == 0, "nbu_soft");
    static_assert(DoubleIntegrator::ng_t_soft == 0, "ng_t_soft");
    static_assert(DoubleIntegrator::ne_t_soft == 0, "ne_t_soft");
    static_assert(DoubleIntegrator::nl_t_soft == 0, "nl_t_soft");
    static_assert(DoubleIntegrator::nbx_t_soft == 0, "nbx_t_soft");
    static_assert(DoubleIntegrator::stage_grad_t::RowsAtCompileTime == 3, "grad layout [x;u]");
    static_assert(DoubleIntegrator::stage_hess_t::RowsAtCompileTime == 3, "hess layout [x;u]");

    const int N = 10;
    DoubleIntegrator problem;
    ocp::Solution<DoubleIntegrator> sol(N);  // constructor resizes all trajectories
    if (sol.x.size() != static_cast<std::size_t>(N + 1) ||
        sol.u.size() != static_cast<std::size_t>(N))
    {
        std::printf("FAIL: trajectory extents\n");
        return 1;
    }

    // ---- forward simulation -------------------------------------------
    check_close("x0(0)", problem.initial_state()(0), 1.0, 1e-12);
    check_close("x0(1)", problem.initial_state()(1), 0.5, 1e-12);
    sol.x[0] = problem.initial_state();
    for (int k = 0; k < N; ++k)
    {
        DoubleIntegrator::control_t u;
        u << 0.1;
        sol.u[k] = u;
        sol.x[k + 1] = problem.dynamics_next_state(k, sol.x[k], u);
    }
    // v_k = 0.5 + k*Ts*a;  q_N = q_0 + Ts * sum_k v_k
    const double v0 = 0.5, a = 0.1, Ts = 0.1;
    const double qN = 1.0 + Ts * (N * v0 + Ts * a * N * (N - 1) / 2.0);
    check_close("x_N(0)", sol.x[N](0), qN, 1e-12);
    check_close("x_N(1)", sol.x[N](1), v0 + N * Ts * a, 1e-12);

    // ---- cost gradient / Hessian vs finite differences ----------------
    const double h = 1e-6;
    const int k = 3;
    DoubleIntegrator::state_t x;
    DoubleIntegrator::control_t u;
    x << 0.7, -0.2;
    u << 0.3;

    DoubleIntegrator::stage_grad_t g;
    DoubleIntegrator::stage_hess_t H;
    double lcost = 0.0;
    problem.stage_cost_value_grad_hess(k, x, u, lcost, g, H);

    // finite-difference gradient: variables = (q, v, a)
    double fd[3];
    {
        for (int i = 0; i < 3; ++i)
        {
            DoubleIntegrator::state_t xp = x, xm = x;
            DoubleIntegrator::control_t up = u, um = u;
            if (i < 2) { xp(i) += h; xm(i) -= h; }
            else       { up(0) += h; um(0) -= h; }
            fd[i] = (problem.stage_cost_value(k, xp, up) -
                     problem.stage_cost_value(k, xm, um)) / (2.0 * h);
        }
    }
    check_close("grad q", g(0), fd[0], 1e-4);
    check_close("grad v", g(1), fd[1], 1e-4);
    check_close("grad a", g(2), fd[2], 1e-4);

    // Hessian is constant here; check a few entries against finite differences
    check_close("Hess qq", H(0, 0), 1.0);
    check_close("Hess va", H(1, 2), 0.0, 1e-8);

    // terminal cost gradient
    DoubleIntegrator::term_grad_t gt;
    double tval = 0.0;
    problem.terminal_cost_value_grad(x, tval, gt);
    check_close("term grad q", gt(0), 2.0 * problem.wf_ * x(0), 1e-10);

    // ---- dynamics Jacobian vs finite differences ----------------------
    {
        DoubleIntegrator::dyn_df_dx_t A;
        DoubleIntegrator::dyn_df_du_t B;
        DoubleIntegrator::state_t x_next;
        problem.dynamics_value_jac(k, x, u, x_next, A, B);

        // verify A(0,1) = Ts against a central finite difference in x(1)
        DoubleIntegrator::state_t xf = x;
        xf(1) += h;
        DoubleIntegrator::state_t f_p = problem.dynamics_next_state(k, xf, u);
        xf(1) -= 2.0 * h;
        DoubleIntegrator::state_t f_m = problem.dynamics_next_state(k, xf, u);
        check_close("dyn A(0,1) fd", A(0, 1), (f_p(0) - f_m(0)) / (2.0 * h), 1e-5);
        check_close("dyn A(0,1)", A(0, 1), problem.Ts_, 1e-12);
        check_close("dyn A(0,0)", A(0, 0), 1.0, 1e-12);
        check_close("dyn A(1,1)", A(1, 1), 1.0, 1e-12);
        check_close("dyn A(1,0)", A(1, 0), 0.0, 1e-12);
        check_close("dyn B(0,0)", B(0, 0), 0.0, 1e-12);
        check_close("dyn B(1,0)", B(1, 0), problem.Ts_, 1e-12);
    }

    // ---- constraint evaluations ---------------------------------------
    {
        auto g_ineq = problem.stage_inequality_value(k, x, u);
        check_close("ineq value", g_ineq(0), x(1) - problem.v_max_);

        DoubleIntegrator::ineq_dg_dx_t Gx;
        DoubleIntegrator::ineq_dg_du_t Gu;
        DoubleIntegrator::ineq_t g_val;
        problem.stage_inequality_value_jac(k, x, u, g_val, Gx, Gu);
        check_close("ineq jac Gx(0,1)", Gx(0, 1), 1.0, 1e-12);

        check_close("ineq soft penalty",
                    problem.stage_inequality_constr_soft_penalty(k)(0), 100.0, 1e-12);

        const auto lin = problem.stage_linear_constr(k);
        const double val = (lin.A * x + lin.B * u)(0);
        check_close("linear value", val, 0.5 * x(0) + x(1));
        if (val < lin.bounds.lo(0) || val > lin.bounds.hi(0))
        {
            std::printf("FAIL: linear constraint violated: %g\n", val);
            ++nfail;
        }
        else
        {
            std::printf("ok   linear value %g within [%.1f, %.1f]\n",
                        val, lin.bounds.lo(0), lin.bounds.hi(0));
        }

        const auto sb = problem.stage_state_box_constr(k);
        const auto cb = problem.stage_control_box_constr(k);
        const auto tsb = problem.terminal_state_box_constr();
        check_close("state box lo(0)", sb.lo(0), -10.0, 1e-12);
        check_close("control box hi(0)", cb.hi(0), 1.0, 1e-12);
        check_close("term state box hi(1)", tsb.hi(1), 10.0, 1e-12);

        const auto et = problem.terminal_equality_value(sol.x[N]);
        check_close("term eq value", et(0), sol.x[N](0) + sol.x[N](1) - 1.0, 1e-12);
    }

    // ---- fixed-horizon mode -------------------------------------------
    // compile-time extents, stack-allocated, value-initialized (zeroed)
    {
        ocp::Solution<DoubleIntegrator, N> fsol;  // N is a compile-time constant
        if (fsol.N != N ||
            fsol.x.size() != static_cast<std::size_t>(N + 1) ||
            fsol.u.size() != static_cast<std::size_t>(N) ||
            !fsol.lambda_dyn[0].isZero())
        {
            std::printf("FAIL: fixed-horizon extents / zero-init\n");
            ++nfail;
        }
        else
        {
            std::printf("ok   fixed-horizon extents (N=%d, stack-allocated)\n",
                        fsol.N);
        }
        fsol.x[0] = problem.initial_state();
        for (int i = 0; i < N; ++i)
        {
            DoubleIntegrator::control_t uf;
            uf << 0.1;
            fsol.u[i] = uf;
            fsol.x[i + 1] = problem.dynamics_next_state(i, fsol.x[i], uf);
        }
        check_close("fixed x_N(0)", fsol.x[N](0), qN, 1e-12);
    }

    if (nfail == 0)
        std::printf("\nAll interface checks passed.\n");
    else
        std::printf("\n%d check(s) FAILED.\n", nfail);
    return nfail == 0 ? 0 : 1;
}

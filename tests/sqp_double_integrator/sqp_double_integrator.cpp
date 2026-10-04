// Phase 2 sub-step 2h test: standalone end-to-end SQP run on the
// DoubleIntegrator (N = 10) with the default SqpSolver
// (HPIPM QP + GLM regularizer + merit backtracking).
//
// Warm start: x[0] = initial_state(), constant control a* that keeps the
// terminal equality q_N + v_N = 1 exactly feasible (the equality is linear
// in the trajectory, so feasibility is preserved through every SQP iterate),
// forward-simulated states, zero multipliers.
//
// Independent verification (recomputed from the problem interface, not from
// the solver): dynamics gap, constraint violations, complementarity with the
// solution multipliers, and stationarity via a finite-difference Lagrangian
// gradient at the solution (the SQP_PHASE2_PLAN sec. 1.3 sign-convention
// gate: a wrong multiplier sign shows up as an O(1) FD gradient).
//
// Writes sqp_double_integrator.csv (SqpStatistics::write_csv) and prints
// the stat rows (print_level = 1).

#include <cmath>
#include <cstdio>
#include <cstring>

#include "ocp/solvers/acados/sqp.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

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

constexpr int N = 10;

// Constant control that makes the terminal equality q_N + v_N = 1 hold
// exactly at the warm start (N = 10 needed: at N = 5 the min of q_N+v_N
// over |a| <= 1 is 1.15, so the equality would be infeasible).
double terminal_eq_feasible_control(const DoubleIntegrator& p)
{
    const double Ts = p.Ts_;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * N * (N - 1) / 2.0 + Ts * N;
    return (1.0 - (q0 + v0) - Ts * N * v0) / den;
}

Solution<DoubleIntegrator> warm_start(const DoubleIntegrator& p)
{
    Solution<DoubleIntegrator> sol(N);
    const double a = terminal_eq_feasible_control(p);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol.u[k](0) = a;
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }
    return sol;
}

// sec. 1.3 NLP Lagrangian with multipliers frozen at `s`:
//   L = sum_k L_k + L_N
//     + sum_k pi_k' (x_{k+1} - f_k)
//     + sum_k lam_g' g_k
//     + sum_k net_lin * (A x + B u - ref)
//     + box:  lam_lo(lo - w) + lam_hi(w - hi)
//     + terminal eq:  mu * (q_N + v_N - 1)
// x_0 is pinned (not a decision variable); soft-ineq slack is 0 at optimum.
double lagrangian(const DoubleIntegrator& p,
                  const Solution<DoubleIntegrator>& s)
{
    double L = 0.0;
    for (int k = 0; k < N; ++k)
    {
        L += p.stage_cost_value(k, s.x[k], s.u[k]);

        const auto f = p.dynamics_next_state(k, s.x[k], s.u[k]);
        L += s.lambda_dyn[k].dot(s.x[k + 1] - f);

        const auto g = p.stage_inequality_value(k, s.x[k], s.u[k]);
        L += s.lambda_ineq_stage[k].dot(g);

        const auto lin = p.stage_linear_constr(k);
        const double v = (lin.A * s.x[k] + lin.B * s.u[k])(0);
        const double mu = s.lambda_lin_stage[k](0);
        L += mu * (v - (mu >= 0.0 ? lin.bounds.hi(0) : lin.bounds.lo(0)));

        // control box: lo = -1, hi = 1
        L +=  s.lambda_box_control[k](0) * (-1.0 - s.u[k](0));
        L +=  s.lambda_box_control[k](1) * (s.u[k](0) - 1.0);

        // state box (k >= 1 only; x_0 is pinned)
        if (k > 0)
        {
            L +=  s.lambda_box_state[k](0) * (-10.0 - s.x[k](0));
            L +=  s.lambda_box_state[k](2) * (s.x[k](0) - 10.0);
            L +=  s.lambda_box_state[k](1) * (-10.0 - s.x[k](1));
            L +=  s.lambda_box_state[k](3) * (s.x[k](1) - 10.0);
        }
    }
    L += p.terminal_cost_value(s.x[N]);
    L +=  s.lambda_box_state[N](0) * (-10.0 - s.x[N](0));
    L +=  s.lambda_box_state[N](2) * (s.x[N](0) - 10.0);
    L +=  s.lambda_box_state[N](1) * (-10.0 - s.x[N](1));
    L +=  s.lambda_box_state[N](3) * (s.x[N](1) - 10.0);
    L +=  s.lambda_eq_term(0) * (s.x[N](0) + s.x[N](1) - 1.0);
    return L;
}

// Inf-norm of the FD Lagrangian gradient w.r.t. free primal variables:
// u_k (k = 0..N-1), x_k (k = 1..N; x_0 pinned).
// L is quadratic in (x,u) with multipliers fixed, so central-difference
// error is pure round-off (~1e-10 for these magnitudes).
double fd_lagrangian_grad_inf(const DoubleIntegrator& p,
                              const Solution<DoubleIntegrator>& s)
{
    const double h = 1e-5;
    double gmax = 0.0;
    for (int k = 0; k < N; ++k)
    {
        Solution<DoubleIntegrator> sp = s, sm = s;
        sp.u[k](0) += h;
        sm.u[k](0) -= h;
        gmax = std::max(gmax,
            std::fabs((lagrangian(p, sp) - lagrangian(p, sm)) / (2.0 * h)));
    }
    for (int k = 1; k <= N; ++k)
    {
        for (int i = 0; i < 2; ++i)
        {
            Solution<DoubleIntegrator> sp = s, sm = s;
            sp.x[k](i) += h;
            sm.x[k](i) -= h;
            gmax = std::max(gmax,
                std::fabs((lagrangian(p, sp) - lagrangian(p, sm)) / (2.0 * h)));
        }
    }
    return gmax;
}

void check_residuals(const DoubleIntegrator& p,
                     const Solution<DoubleIntegrator>& sol)
{
    // 1. dynamics gap
    double gap = 0.0;
    for (int k = 0; k < N; ++k)
    {
        const auto f = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
        gap = std::max(gap,
            static_cast<double>((f - sol.x[k + 1]).cwiseAbs().maxCoeff()));
    }
    check(gap < 1e-8, "dynamics gap < 1e-8");

    // 2. hard constraint violations (soft ineq handled separately below)
    double viol = std::fabs(sol.x[N](0) + sol.x[N](1) - 1.0);
    for (int k = 1; k <= N; ++k)
    {
        for (int j = 0; j < 2; ++j)
        {
            const double w = sol.x[k](j);
            viol = std::max(viol, std::max(0.0, std::max(w - 10.0, -10.0 - w)));
        }
    }
    for (int k = 0; k < N; ++k)
    {
        const double a = sol.u[k](0);
        viol = std::max(viol, std::max(0.0, std::max(a - 1.0, -1.0 - a)));
        const double v = 0.5 * sol.x[k](0) + sol.x[k](1);
        viol = std::max(viol, std::max(0.0, std::max(v - 3.0, -3.0 - v)));
    }
    check(viol < 1e-8, "hard constraint violations < 1e-8");

    // 3. complementarity: |lambda * positive_violation| < tol per side
    double comp = 0.0;
    for (int k = 0; k < N; ++k)
    {
        const double a = sol.u[k](0);
        comp = std::max(comp, std::fabs(
            sol.lambda_box_control[k](0) * std::max(0.0, -1.0 - a)));
        comp = std::max(comp, std::fabs(
            sol.lambda_box_control[k](1) * std::max(0.0, a - 1.0)));
        const double g = sol.x[k](1) - p.v_max_;
        comp = std::max(comp, std::fabs(
            sol.lambda_ineq_stage[k](0) * std::max(0.0, g)));
        const double v = 0.5 * sol.x[k](0) + sol.x[k](1);
        const double net = sol.lambda_lin_stage[k](0);
        comp = std::max(comp, std::fabs(
            net >= 0.0 ? net * std::max(0.0, v - 3.0)
                       : -net * std::max(0.0, -3.0 - v)));
    }
    for (int k = 1; k <= N; ++k)
    {
        for (int j = 0; j < 2; ++j)
        {
            const double w = sol.x[k](j);
            comp = std::max(comp, std::fabs(
                sol.lambda_box_state[k](j) * std::max(0.0, -10.0 - w)));
            comp = std::max(comp, std::fabs(
                sol.lambda_box_state[k](2 + j) * std::max(0.0, w - 10.0)));
        }
    }
    check(comp < 1e-8, "complementarity < 1e-8");

    // 4. stationarity: FD Lagrangian gradient (sec. 1.3 sign-convention gate)
    const double fd = fd_lagrangian_grad_inf(p, sol);
    check(fd < 1e-5, "FD Lagrangian gradient < 1e-5");
}

}  // namespace

int main()
{
    DoubleIntegrator p;
    SqpOptions opts;
    opts.print_level = 1;

    SqpSolver<DoubleIntegrator> solver(opts);
    Solution<DoubleIntegrator> sol = warm_start(p);

    const Status st = solver.solve(p, sol);
    check(st == Status::kSolved, "solve returned kSolved");
    check(sol.status == Status::kSolved, "sol.status kSolved");
    check(solver.statistics().iter >= 1, "at least one statistics row");

    check_residuals(p, sol);

    // x_0 preserved (fixed initial state)
    const auto x0 = p.initial_state();
    check_close(sol.x[0](0), x0(0), 1e-10, "x[0].q preserved");
    check_close(sol.x[0](1), x0(1), 1e-10, "x[0].v preserved");

    // terminal equality q_N + v_N = 1
    check_close(sol.x[N](0) + sol.x[N](1), 1.0, 1e-6, "terminal q + v = 1");

    // control box bounds
    for (int k = 0; k < N; ++k)
    {
        check(sol.u[k](0) >= -1.0 - 1e-8 && sol.u[k](0) <= 1.0 + 1e-8,
              "control box holds");
    }

    // soft velocity cap: v <= v_max + slack. The ineq (w = 100) is the only
    // soft row, so per-row slack <= sqrt(2 * total_slack_penalty / w).
    double raw = 0.0;
    for (int k = 0; k < N; ++k)
        raw += p.stage_cost_value(k, sol.x[k], sol.u[k]);
    raw += p.terminal_cost_value(sol.x[N]);
    check(sol.cost_value >= raw - 1e-12, "cost_value >= raw cost");
    const double slack_bound =
        std::sqrt(std::max(0.0, 2.0 * (sol.cost_value - raw) / 100.0));
    for (int k = 0; k < N; ++k)
    {
        check(static_cast<double>(sol.x[k](1)) <= p.v_max_ + slack_bound + 1e-8,
              "soft row: v_k <= v_max + slack");
    }

    // CSV output
    solver.statistics().write_csv("sqp_double_integrator.csv");
    std::FILE* f = std::fopen("sqp_double_integrator.csv", "r");
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
            ++rows;
        check(rows == static_cast<int>(solver.statistics().iter),
              "CSV row count matches statistics");
        std::fclose(f);
    }

    if (failures > 0)
    {
        std::fprintf(stderr, "sqp_double_integrator: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("sqp_double_integrator: OK (%d SQP iterations, "
                "cost = %.10e)\n",
                solver.statistics().iter, sol.cost_value);
    return 0;
}

// Phase 1 block 1f test: end-to-end full-condensed vs staged agreement.
//
// Covers (SQP_HPIPM_PARITY_PHASE1_PLAN.md sec. 1f):
//   - DoubleIntegrator (N = 10) and MassesChain (N = 40) are solved in
//     both QpModes from the same warm start; the converged NLP solutions
//     agree to <= 1e-8 (x / u inf norm, cost relative).
//
// The acados-reference half of 1f (the mass_spring QPs solved with
// kFullCondensed against the FULL_CONDENSING_HPIPM references) lives in
// tests/sqp_acados_ref/sqp_acados_ref.cpp.

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/solvers/acados/acados.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"
#include "../../examples/masses_chain/masses_chain.hpp"

namespace
{

using namespace ocp;

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

template <class P>
void compare_solutions(const std::string& name, const Solution<P>& a,
                       const Solution<P>& b, int N, double tol)
{
    double dx = 0.0;
    double du = 0.0;
    for (int k = 0; k <= N; ++k)
    {
        dx = std::max(dx,
                      (double)(a.x[k] - b.x[k]).cwiseAbs().maxCoeff());
    }
    for (int k = 0; k < N; ++k)
    {
        du = std::max(du,
                      (double)(a.u[k] - b.u[k]).cwiseAbs().maxCoeff());
    }
    const double rel_cost =
        std::fabs(a.cost_value - b.cost_value) / std::fabs(a.cost_value);
    check(dx < tol, name + ": x inf-norm diff < " +
                        std::to_string(tol) + " (got " +
                        std::to_string(dx) + ")");
    check(du < tol, name + ": u inf-norm diff < " +
                        std::to_string(tol) + " (got " +
                        std::to_string(du) + ")");
    check(rel_cost < tol, name + ": cost rel diff < " +
                              std::to_string(tol) + " (got " +
                              std::to_string(rel_cost) + ")");
    std::printf("  %s: dx = %.3e, du = %.3e, rel_cost = %.3e\n", name.c_str(),
                dx, du, rel_cost);
}

// ---------------------------------------------------------------------
// DoubleIntegrator N = 10 (both modes, 1e-8)
// ---------------------------------------------------------------------

double di_feasible_control(const DoubleIntegrator& p, int n_stages)
{
    const double Ts = p.params().Ts;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * n_stages * (n_stages - 1) / 2.0 + Ts * n_stages;
    return (1.0 - (q0 + v0) - Ts * n_stages * v0) / den;
}

void test_di()
{
    const std::string name = "e2e DI N=10    ";
    const int N = 10;
    DoubleIntegrator p;
    const double a = di_feasible_control(p, N);

    auto make_start = [&]()
    {
        Solution<DoubleIntegrator> sol(N);
        sol.x[0] = p.initial_state();
        for (int k = 0; k < N; ++k)
        {
            sol.u[k](0) = a;
            sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
        }
        return sol;
    };

    AcadosOptions o1;
    AcadosSolver<DoubleIntegrator> s1(o1);
    Solution<DoubleIntegrator> sol1 = make_start();
    check(s1.solve(p, sol1) == Status::kSolved, name + ": staged kSolved");
    if (failures > 0)
    {
        return;
    }

    AcadosOptions o2;
    o2.qp_mode = QpMode::kFullCondensed;
    AcadosSolver<DoubleIntegrator> s2(o2);
    Solution<DoubleIntegrator> sol2 = make_start();
    check(s2.solve(p, sol2) == Status::kSolved,
          name + ": condensed kSolved");
    if (failures > 0)
    {
        return;
    }

    compare_solutions(name, sol1, sol2, N, 1e-8);
}

// ---------------------------------------------------------------------
// MassesChain N = 40 (both modes, 1e-8; NLP tolerances 1e-6 as in
// sqp_masses_chain: the weakly-convex Hessian stalls at the 1e-8 default)
// ---------------------------------------------------------------------

void test_masses_chain()
{
    const std::string name = "e2e MC N=40    ";
    const int N = 40;
    MassesChain p;

    auto make_start = [&]()
    {
        Solution<MassesChain> sol(N);
        sol.x[0] = p.initial_state();
        for (int k = 0; k < N; ++k)
        {
            sol.u[k].setZero();
            sol.x[k + 1] = p.cost_reference();
        }
        return sol;
    };

    AcadosOptions o1;
    o1.tol_stat = 1e-6;
    o1.tol_eq = 1e-6;
    o1.tol_ineq = 1e-6;
    o1.tol_comp = 1e-6;
    AcadosSolver<MassesChain> s1(o1);
    Solution<MassesChain> sol1 = make_start();
    const Status st1 = s1.solve(p, sol1);
    const bool ok1 = st1 == Status::kSolved || st1 == Status::kMinStep;
    check(ok1, name + ": staged converged (status " +
                   std::to_string((int)st1) + ")");
    if (!ok1)
    {
        return;
    }

    AcadosOptions o2 = o1;
    o2.qp_mode = QpMode::kFullCondensed;
    AcadosSolver<MassesChain> s2(o2);
    Solution<MassesChain> sol2 = make_start();
    const Status st2 = s2.solve(p, sol2);
    const bool ok2 = st2 == Status::kSolved || st2 == Status::kMinStep;
    check(ok2, name + ": condensed converged (status " +
                   std::to_string((int)st2) + ")");
    if (!ok2)
    {
        return;
    }

    compare_solutions(name, sol1, sol2, N, 1e-8);
}

}  // namespace

int run_e2e_1f_tests()
{
    failures = 0;
    test_di();
    test_masses_chain();

    if (failures == 0)
    {
        std::printf("All e2e (1f) checks passed.\n");
    }
    return failures;
}

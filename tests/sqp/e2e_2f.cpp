// Phase 2 block 2f test: end-to-end block-condensed vs staged agreement.
//
// Covers (SQP_HPIPM_PARITY_PHASE2_PLAN.md sec. 2f):
//   - DoubleIntegrator (N = 10) and MassesChain (N = 40) are solved in
//     kStaged and kPartialCondensed (NB = 2, and NB = 4 where N % NB == 0)
//     from the same warm start; the converged NLP solutions agree to
//     <= 1e-8 (x / u inf norm, cost relative) and the SQP iteration
//     counts are comparable.
//   - A solver compiled with a block size that does not match qp_cond_N
//     falls back to the plain staged path.

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

/// AcadosSolver with a compile-time block size NB (the trailing template
/// parameter of the 2e driver hook).
template <class P, int NH, int NB>
using BlockSolver = AcadosSolver<P, NH, HpipmQpSolver<P, NH>,
                                 GlmRegularizer<P, NH>,
                                 MeritBacktracking<P, NH>, NB>;

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
// DoubleIntegrator N = 10: staged vs NB = 2 (N % 4 != 0, so NB = 4 is
// not exercised), 1e-8
// ---------------------------------------------------------------------

double di_feasible_control(const DoubleIntegrator& p, int n_stages)
{
    const double Ts = p.Ts_;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * n_stages * (n_stages - 1) / 2.0 + Ts * n_stages;
    return (1.0 - (q0 + v0) - Ts * n_stages * v0) / den;
}

void test_di()
{
    const std::string name = "e2e DI N=10";
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
    const int iters1 = s1.statistics().iter;
    if (failures > 0)
    {
        return;
    }

    AcadosOptions o2;
    o2.qp_mode = QpMode::kPartialCondensed;
    o2.qp_cond_N = N / 2;  // 5 blocks of NB = 2
    BlockSolver<DoubleIntegrator, Eigen::Dynamic, 2> s2(o2);
    Solution<DoubleIntegrator> sol2 = make_start();
    check(s2.solve(p, sol2) == Status::kSolved, name + ": block(NB=2) kSolved");
    const int iters2 = s2.statistics().iter;
    if (failures > 0)
    {
        return;
    }

    compare_solutions(name + " NB=2 ", sol1, sol2, N, 1e-8);
    check(iters2 <= iters1 + 2,
          name + ": block(NB=2) iterations comparable (" +
              std::to_string(iters2) + " vs staged " +
              std::to_string(iters1) + ")");
    std::printf("  %s NB=2: staged %d it, block %d it\n", name.c_str(),
                iters1, iters2);

    // Mismatch: solver compiled with NB = 4 but qp_cond_N = 5 blocks
    // (block size 2): the driver falls back to the plain staged path.
    AcadosOptions o3;
    o3.qp_mode = QpMode::kPartialCondensed;
    o3.qp_cond_N = 5;
    BlockSolver<DoubleIntegrator, Eigen::Dynamic, 4> s3(o3);
    Solution<DoubleIntegrator> sol3 = make_start();
    check(s3.solve(p, sol3) == Status::kSolved,
          name + ": mismatched NB fallback kSolved");
    if (failures > 0)
    {
        return;
    }
    compare_solutions(name + " mmatch", sol1, sol3, N, 1e-8);
}

// ---------------------------------------------------------------------
// MassesChain N = 40: staged vs NB = 2 and NB = 4 (N % NB == 0 for
// both), 1e-8 (NLP tolerances 1e-6 as in sqp_masses_chain: the
// weakly-convex Hessian stalls at the 1e-8 default)
// ---------------------------------------------------------------------

template <int NB>
bool solve_block_mc(const std::string& name, int iters1, const MassesChain& p,
                    const AcadosOptions& o1, int N, Solution<MassesChain>& sol2)
{
    AcadosOptions o2 = o1;
    o2.qp_mode = QpMode::kPartialCondensed;
    o2.qp_cond_N = N / NB;
    BlockSolver<MassesChain, Eigen::Dynamic, NB> s2(o2);
    const Status st = s2.solve(p, sol2);
    const int it = s2.statistics().iter;
    std::printf("  %s NB=%d: staged %d it, block %d it\n",
                name.c_str(), NB, iters1, it);
    check(it <= iters1 + 2,
          name + ": block(NB=" + std::to_string(NB) +
          ") iterations comparable");
    return (st == Status::kSolved || st == Status::kMinStep);
}

void test_masses_chain()
{
    const std::string name = "e2e MC N=40";
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
    const int iters1 = s1.statistics().iter;
    if (!ok1)
    {
        return;
    }

    for (int NB : {2, 4})
    {
        const std::string tag =
            " NB=" + std::to_string(NB);
        Solution<MassesChain> sol2 = make_start();
        const bool ok2 = (NB == 2)
            ? solve_block_mc<2>(name, iters1, p, o1, N, sol2)
            : solve_block_mc<4>(name, iters1, p, o1, N, sol2);
        check(ok2, name + ": block" + tag + " converged");
        if (!ok2)
        {
            return;
        }
        compare_solutions(name + tag, sol1, sol2, N, 1e-8);
    }
}

}  // namespace

int run_e2e_2f_tests()
{
    failures = 0;
    test_di();
    test_masses_chain();

    if (failures == 0)
    {
        std::printf("All e2e (2f) checks passed.\n");
    }
    return failures;
}

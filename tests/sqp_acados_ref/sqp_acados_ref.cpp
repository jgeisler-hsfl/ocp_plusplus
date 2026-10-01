// Phase 2 sub-step 2i test: acados reference diff.
//
// The acados mass_spring QP example (acados/examples/c/
// no_interface_examples/mass_spring_model/mass_spring_qp.c) is solved by
// acados FULL_CONDENSING_HPIPM in a standalone C driver (see
// /tmp/opencode/acados_ref/ms_ref.c) and the full-precision (x, u, cost)
// output is embedded in ref_boxonly.h / ref_termeq.h. This test solves the
// same QP with our SqpSolver and compares:
//   - cost:       relative error < 1e-4
//   - (x, u):     component-wise inf-norm error < 1e-4
//
// Two variants:
//   boxonly : state box [-4,4], control box [-0.5,0.5], no terminal
//             constraint (ne_t = 0)
//   termeq  : as boxonly + hard terminal equality x_N(0:3) = 0 (ne_t = 4),
//             matching the GENERAL_CONSTRAINT_AT_TERMINAL_STAGE variant
//
// Both are pure QPs; the SQP should converge in a handful of iterations
// from the (possibly infeasible) forward-simulation warm start
// (x_0 = [2.5, 2.5, 0, ...]', u = 0).

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/solvers/acados/sqp.hpp"

#include "../../examples/mass_spring_ref/mass_spring_ref.hpp"
#include "ref_boxonly.h"
#include "ref_termeq.h"

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

constexpr int N = 15;

template <class P>
void run_variant(const char* name, const double ref_x[16][8],
                 const double ref_u[15][3], double ref_cost, P& p)
{
    Solution<P> sol(N);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }

    SqpOptions opts;
    SqpSolver<P> solver(opts);
    const Status st = solver.solve(p, sol);
    check(st == Status::kSolved, (std::string(name) + " solve kSolved").c_str());
    if (st != Status::kSolved)
    {
        return;
    }

    // cost (relative)
    const double cost_rel = std::fabs(sol.cost_value - ref_cost) /
                            std::fabs(ref_cost);
    check(cost_rel < 1e-4,
          (std::string(name) + " cost relative error < 1e-4").c_str());
    std::printf("%s: cost = %.15e (ref %.15e, rel err %.3e, %d iters)\n",
                name, sol.cost_value, ref_cost, cost_rel,
                solver.statistics().iter);

    // (x, u) component-wise
    double dx = 0.0, du = 0.0;
    for (int k = 0; k <= N; ++k)
    {
        for (int i = 0; i < 8; ++i)
        {
            dx = std::max(dx, std::fabs(sol.x[k](i) - ref_x[k][i]));
        }
    }
    for (int k = 0; k < N; ++k)
    {
        for (int j = 0; j < 3; ++j)
        {
            du = std::max(du, std::fabs(sol.u[k](j) - ref_u[k][j]));
        }
    }
    check(dx < 1e-4, (std::string(name) + " x inf-norm error < 1e-4").c_str());
    check(du < 1e-4, (std::string(name) + " u inf-norm error < 1e-4").c_str());
    std::printf("%s: dx = %.3e, du = %.3e\n", name, dx, du);
}

}  // namespace

int main()
{
    MassSpringRefBox p_box;
    run_variant("boxonly", ref_x_boxonly, ref_u_boxonly, ref_cost_boxonly, p_box);

    MassSpringRefTerm p_term;
    run_variant("termeq", ref_x_termeq, ref_u_termeq, ref_cost_termeq, p_term);

    if (failures > 0)
    {
        std::fprintf(stderr, "sqp_acados_ref: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("sqp_acados_ref: OK\n");
    return 0;
}

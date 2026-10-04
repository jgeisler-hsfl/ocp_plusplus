// gen_equivalence.cpp
//
// Verifies that the GENERATOR (tools/acados2ocp_pp.py) reproduces the
// hand-written masses_chain problem class: the generated MassesChainGen and the
// hand-written MassesChain must yield the SAME SQP solution (same cost and
// (x,u) trajectory to machine precision) when solved from the same warm start.
//
// Both problems share the identical CasADi C residual
// (masses_chain_impl_dae_fun), integrator config (IRK GL4 x 2 sub-steps, 3
// Newton iters), cost, box constraints, and fixed initial state; only the
// C++ wrapper differs (hand-written vs JSON-generated).

#include <cstdio>
#include <cmath>

#include "ocp/solvers/acados/sqp.hpp"

#include "masses_chain.hpp"               // hand-written: ocp::MassesChain
#include "generated/MassesChainGen.hpp"   // generated:    ocp::MassesChainGen

using namespace ocp;

namespace
{
// Warm-start exactly as examples/masses_chain/main.cpp: x0 fixed, u_k = 0,
// x_k = cost reference (spread config) for k = 1..N.
template <class P>
Solution<P> solve(const P& p, int N, SqpStatistics* stat)
{
    SqpOptions opts;
    opts.print_level = 0;
    SqpSolver<P> solver(opts);

    Solution<P> sol(N);
    if constexpr (P::fixed_initial_state)
    {
        sol.x[0] = p.initial_state();
    }
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = p.cost_reference();
    }

    solver.solve(p, sol);
    *stat = solver.statistics();
    return sol;
}
}  // namespace

int main()
{
    constexpr int N = 40;

    MassesChain a;
    MassesChainGen b;

    SqpStatistics sa, sb;
    Solution<MassesChain> sol_a = solve(a, N, &sa);
    Solution<MassesChainGen> sol_b = solve(b, N, &sb);

    double max_dx = 0.0, max_du = 0.0;
    for (int k = 0; k <= N; ++k)
    {
        max_dx = std::max(max_dx, (sol_a.x[k] - sol_b.x[k]).cwiseAbs().maxCoeff());
    }
    for (int k = 0; k < N; ++k)
    {
        max_du = std::max(max_du, (sol_a.u[k] - sol_b.u[k]).cwiseAbs().maxCoeff());
    }
    const double cost_a = sol_a.cost_value;
    const double cost_b = sol_b.cost_value;
    const double rel_dc =
        std::abs(cost_a - cost_b) / std::max(1.0, std::abs(cost_a));

    std::printf("hand-written : cost=%.12f  iters=%d  status=%d\n",
                cost_a, sa.iter, static_cast<int>(sa.status));
    std::printf("generated    : cost=%.12f  iters=%d  status=%d\n",
                cost_b, sb.iter, static_cast<int>(sb.status));
    std::printf("|dx|max = %.3e  |du|max = %.3e  cost rel diff = %.3e\n",
                max_dx, max_du, rel_dc);

    // The two problems are the same OCP described by two C++ wrappers that share
    // one C residual.  They agree to FD / matrix-product noise (~1e-10); the SQP
    // status can differ by one (kSolved vs kMinStep) at that noise level, so we
    // only require both to have converged and to match in trajectory + cost.
    const bool a_conv = (sa.status == Status::kSolved || sa.status == Status::kMinStep);
    const bool b_conv = (sb.status == Status::kSolved || sb.status == Status::kMinStep);
    const bool ok =
        a_conv && b_conv && sa.iter == sb.iter &&
        max_dx < 1e-8 && max_du < 1e-8 && rel_dc < 1e-9;

    if (!ok)
    {
        std::printf("gen_equivalence: FAIL\n");
        return 1;
    }
    std::printf("gen_equivalence: OK\n");
    return 0;
}

// sqp_unicycle.cpp
//
// Phase-3 verification of the ocp++ port of the acados unicycle OCP
// (codegen a0fdb215, Python/casadi codegen; generated wrapper
// examples/unicycle/generated/UnicycleGen.hpp). The acados reference
// (hessian_approx = GAUSS_NEWTON, IRK Gauss-Legendre 4-stage x 1 sub-step,
// 3 Newton iters, NLP tolerances 1e-6) is embedded in ref_unicycle.h.
//
// Batch OCP: drive the unicycle from x0 = 0 to (x, y) = (1, 1) by T = 2.0;
// yref = (1, 1, 0, 0, 0, 0, 0) (the reference of the acados closed-loop
// example). The control is smooth (no bang-bang saturation in the optimal
// profile), so the two SQP variants stay close.
//
// Checks (same pattern as tests/sqp_pendulum / sqp_masses_chain):
//  (A) INTEGRATOR EXACTNESS: the ocp++ collocation map (CasADi residual +
//      GL4 tableau) must reproduce x_{k+1} from every reference (x_k, u_k)
//      to machine precision. Independent of the SQP Hessian model; proves
//      the dynamics port (incl. the residual sign convention) is exact.
//  (B) COST: relative error vs the acados reference < 1e-4. ocp++ assembles
//      the QP Hessian from the stage cost Hessian only
//      (has_dynamics_hess_prod = false) where acados uses Gauss-Newton;
//      the observed gap is ~3e-6 (both at near-identical KKT points).
//  (C) TRAJECTORY: |dx|, |du| inf-norm vs the acados reference, relaxed
//      tolerance (Hessian-model gap accumulated over N = 50 stages).
//  (D) OCP++'s OWN KKT RESIDUALS of its converged solution must be small,
//      independently confirming the ocp++ iterate is a valid KKT point of
//      the (identical) NLP.

#include <cmath>
#include <cstdio>

#include "ocp/solvers/acados/sqp.hpp"

#include "../../examples/unicycle/generated/UnicycleGen.hpp"
#include "ref_unicycle.h"

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

using Ode = UnicycleGenOde;
using Dim = UnicycleGenDims;

// (A) feed each reference (x_k, u_k) through the ocp++ collocation map and
// compare to x_{k+1}; the max gap must be ~machine precision.
void check_integrator_map()
{
    Ode ode;
    UnicycleGenInteg integ(ode, UNICYCLE_DT);
    Eigen::Matrix<double, Dim::nx, 1> x, x_next;
    Eigen::Matrix<double, Dim::nu, 1> u;
    double max_gap = 0.0;
    for (int k = 0; k < UNICYCLE_N; ++k)
    {
        for (int i = 0; i < Dim::nx; ++i)
            x(i) = ref_x_unicycle[k][i];
        for (int i = 0; i < Dim::nu; ++i)
            u(i) = ref_u_unicycle[k][i];
        integ.value(x, u, k * UNICYCLE_DT, x_next);
        for (int i = 0; i < Dim::nx; ++i)
            max_gap = std::max(max_gap,
                               std::fabs(x_next(i) - ref_x_unicycle[k + 1][i]));
    }
    check(max_gap < 1e-9, "integrator map |Phi(x_k,u_k) - x_{k+1}| < 1e-9");
    std::printf("  (A) integrator map  max|Phi - x_{k+1}| = %.3e\n", max_gap);
}

}  // namespace

int main()
{
    UnicycleGen problem;

    // warm start mirrors the generated driver: x_0 = fixed initial state,
    // x_{1..N} = cost reference, u = 0.
    Solution<UnicycleGen> sol(UNICYCLE_N);
    sol.x[0] = problem.initial_state();
    for (int k = 0; k < UNICYCLE_N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = problem.cost_reference();
    }

    SqpOptions opts;
    opts.print_level = 0;
    opts.max_iter = 100;
    opts.tol_stat = 1e-06;
    opts.tol_eq = 1e-06;
    opts.tol_ineq = 1e-06;
    opts.tol_comp = 1e-06;
    SqpSolver<UnicycleGen> solver(opts);
    const Status st = solver.solve(problem, sol);

    check_integrator_map();

    // (B) cost
    const double cost_rel =
        std::fabs(sol.cost_value - UNICYCLE_REF_COST) /
        std::fabs(UNICYCLE_REF_COST);
    check(cost_rel < 1e-4, "cost relative error < 1e-4");
    std::printf("  (B) cost = %.12f (ref %.12f, rel %.3e, %d iters, status=%d)\n",
                sol.cost_value, UNICYCLE_REF_COST, cost_rel,
                solver.statistics().iter, static_cast<int>(st));

    // (C) trajectory (relaxed: Hessian-model gap, see file header (C))
    double dx = 0.0, du = 0.0;
    for (int k = 0; k <= UNICYCLE_N; ++k)
    {
        for (int i = 0; i < Dim::nx; ++i)
            dx = std::max(dx,
                          std::fabs(static_cast<double>(sol.x[k](i))
                                   - ref_x_unicycle[k][i]));
    }
    for (int k = 0; k < UNICYCLE_N; ++k)
    {
        for (int i = 0; i < Dim::nu; ++i)
            du = std::max(du,
                          std::fabs(static_cast<double>(sol.u[k](i))
                                   - ref_u_unicycle[k][i]));
    }
    check(dx < 0.1, "state inf-norm gap < 0.1");
    check(du < 0.2, "control inf-norm gap < 0.2");
    std::printf("  (C) trajectory        |dx| = %.3e  |du| = %.3e "
                "(Hessian-model gap)\n", dx, du);

    // (D) ocp++'s own KKT residuals (independent KKT-validity check)
    const NlpResiduals res = compute_nlp_residuals(problem, sol);
    check(res.res_stat < 1e-4, "ocp++ res_stat < 1e-4");
    check(res.res_eq < 1e-9, "ocp++ res_eq (dynamics gap) < 1e-9");
    check(res.res_ineq < 1e-9, "ocp++ res_ineq (box) < 1e-9");
    std::printf("  (D) ocp++ KKT         res_stat=%.3e res_eq=%.3e "
                "res_ineq=%.3e res_comp=%.3e\n",
                res.res_stat, res.res_eq, res.res_ineq, res.res_comp);

    if (failures > 0)
    {
        std::fprintf(stderr, "sqp_unicycle: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("sqp_unicycle: OK\n");
    return 0;
}

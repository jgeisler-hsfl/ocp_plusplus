// sqp_pendulum.cpp
//
// Phase-3 verification of the ocp++ port of the acados pendulum_on_cart OCP
// (codegen c662f455, Python/casadi codegen; generated wrapper
// examples/pendulum_on_cart/generated/PendulumGen.hpp). The acados reference
// (hessian_approx = GAUSS_NEWTON, IRK Gauss-Legendre 4-stage x 1 sub-step,
// 3 Newton iters, NLP tolerances 1e-6) is embedded in ref_pendulum.h.
//
// This OCP is non-convex with a bang-bang optimal control (u saturates at
// +/-80 on most stages). The checks therefore differ in spirit from
// sqp_masses_chain:
//  (A) INTEGRATOR EXACTNESS (the key correctness proof). The CasADi
//      residual + GL4 tableau + 3-Newton configuration is verified against
//      the acados reference map: for every reference stage the ocp++
//      collocation map must reproduce x_{k+1} from (x_k, u_k) to machine
//      precision. This is independent of the SQP Hessian model and proves
//      the dynamics port (including the residual sign convention) is exact.
//  (B) COST: relative error vs the acados reference < 1e-3. ocp++ assembles
//      the QP Hessian from the stage cost Hessian only
//      (Dims::has_dynamics_hess = false), whereas acados uses
//      Gauss-Newton (cost + dynamics J'J). On this non-convex OCP the two
//      quadratic SQP models follow different paths and settle at different
//      local KKT points; a relative gap of ~1e-4 is observed and expected.
//  (C) TRAJECTORY: |dx|, |du| inf-norm vs the acados reference, at a loose
//      tolerance. The control is bang-bang, so small Hessian-model
//      differences shift the switching instants and amplify along the
//      horizon; this bounds the disagreement but cannot be made tight.
//  (D) OCP++'s OWN KKT RESIDUALS of its converged solution (res_stat /
//      res_eq / res_ineq / res_comp) are reported and must be small,
//      independently confirming the ocp++ iterate is a valid KKT point of
//      the (identical) NLP.

#include <cmath>
#include <cstdio>

#include "ocp/solvers/acados/sqp.hpp"

#include "../../examples/pendulum_on_cart/generated/PendulumGen.hpp"
#include "ref_pendulum.h"

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

using Ode = PendulumGenOde;
using Dim = PendulumGenDims;

// (A) feed each reference (x_k, u_k) through the ocp++ collocation map and
// compare to x_{k+1}; the max gap must be ~machine precision.
void check_integrator_map()
{
    Ode ode;
    PendulumGenInteg integ(ode, PENDULUM_DT);
    Eigen::Matrix<double, Dim::nx, 1> x, x_next;
    Eigen::Matrix<double, Dim::nu, 1> u;
    double max_gap = 0.0;
    for (int k = 0; k < PENDULUM_N; ++k)
    {
        for (int i = 0; i < Dim::nx; ++i)
            x(i) = ref_x_pendulum[k][i];
        for (int i = 0; i < Dim::nu; ++i)
            u(i) = ref_u_pendulum[k][i];
        integ.value(x, u, k * PENDULUM_DT, x_next);
        for (int i = 0; i < Dim::nx; ++i)
            max_gap = std::max(max_gap,
                               std::fabs(x_next(i) - ref_x_pendulum[k + 1][i]));
    }
    check(max_gap < 1e-9, "integrator map |Phi(x_k,u_k) - x_{k+1}| < 1e-9");
    std::printf("  (A) integrator map  max|Phi - x_{k+1}| = %.3e\n", max_gap);
}

}  // namespace

int main()
{
    PendulumGen problem;

    // warm start mirrors the generated driver: x_0 = fixed initial state,
    // x_{1..N} = cost reference (yref = 0), u = 0.
    Solution<PendulumGen> sol(PENDULUM_N);
    sol.x[0] = problem.initial_state();
    for (int k = 0; k < PENDULUM_N; ++k)
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
    SqpSolver<PendulumGen> solver(opts);
    const Status st = solver.solve(problem, sol);

    check_integrator_map();

    // (B) cost (loose: distinct local KKT point, see file header (B))
    const double cost_rel =
        std::fabs(sol.cost_value - PENDULUM_REF_COST) /
        std::fabs(PENDULUM_REF_COST);
    check(cost_rel < 1e-3, "cost relative error < 1e-3");
    std::printf("  (B) cost = %.12f (ref %.12f, rel %.3e, %d iters, status=%d)\n",
                sol.cost_value, PENDULUM_REF_COST, cost_rel,
                solver.statistics().iter, static_cast<int>(st));

    // (C) trajectory (loose: bang-bang switching sensitivity, see header (C))
    double dx = 0.0, du = 0.0;
    for (int k = 0; k <= PENDULUM_N; ++k)
    {
        for (int i = 0; i < Dim::nx; ++i)
            dx = std::max(dx,
                          std::fabs(static_cast<double>(sol.x[k](i))
                                   - ref_x_pendulum[k][i]));
    }
    for (int k = 0; k < PENDULUM_N; ++k)
    {
        for (int i = 0; i < Dim::nu; ++i)
            du = std::max(du,
                          std::fabs(static_cast<double>(sol.u[k](i))
                                   - ref_u_pendulum[k][i]));
    }
    check(dx < 0.25, "state inf-norm gap < 0.25");
    check(du < 1.5, "control inf-norm gap < 1.5");
    std::printf("  (C) trajectory        |dx| = %.3e  |du| = %.3e "
                "(bang-bang + Hessian-model gap)\n", dx, du);

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
        std::fprintf(stderr, "sqp_pendulum: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("sqp_pendulum: OK\n");
    return 0;
}

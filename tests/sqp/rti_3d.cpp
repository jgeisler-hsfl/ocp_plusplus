// Phase 3 block 3d test: RTI parity evidence
// (include/ocp/solvers/acados/rti.hpp, SQP_HPIPM_PARITY_PHASE3_PLAN.md
// block 3d).
//
//   - DoubleIntegrator (N = 10, constant Hessian): one RtiSolver::solve()
//     from a deterministic warm start reproduces one AcadosSolver SQP
//     iteration (same QP, full step alpha = 1) in x/u to 1e-10; a second
//     RTI call from the (near-optimal) result takes a ~0 step; the
//     rti_log_residuals pre-step row matches compute_nlp_residuals.
//   - MassesChain (N = 40, GN Hessian): one RtiSolver::solve() reproduces
//     one AcadosSolver SQP iteration in x/u to 1e-10. The parity is
//     per-iteration: the GN step is not the converged NLP solution.
//   - MassSpring (N = 20, constant Hessian): one RtiSolver::solve()
//     reproduces the CONVERGED AcadosSolver solution (the QP subproblem
//     is exact: one Newton step lands) in x/u to 1e-8 and cost to 1e-10.
//   - QP-mode cross-check (DI N = 10, MC N = 40): the RTI iterate agrees
//     for kStaged, kFullCondensed, and kPartialCondensed (NB = 2) to 1e-8.

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/solvers/acados/rti.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"
#include "../../examples/masses_chain/masses_chain.hpp"
#include "../../examples/mass_spring/mass_spring.hpp"

namespace
{

using namespace ocp;

int failures = 0;

void check_true(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

template <class A, class B>
double max_abs_diff(const A& a, const B& b)
{
    double mx = 0.0;
    for (Eigen::Index i = 0; i < a.size(); ++i)
    {
        mx = std::max(mx, std::fabs(static_cast<double>(a.coeff(i))
                                    - static_cast<double>(b.coeff(i))));
    }
    return mx;
}

template <class P>
void check_iterate(const std::string& name, const Solution<P>& a,
                   const Solution<P>& b, double tol)
{
    for (int k = 0; k <= a.N; ++k)
    {
        check_true(max_abs_diff(a.x[k], b.x[k]) < tol,
                   name + ": x_" + std::to_string(k));
    }
    for (int k = 0; k < a.N; ++k)
    {
        check_true(max_abs_diff(a.u[k], b.u[k]) < tol,
                   name + ": u_" + std::to_string(k));
    }
}

/// RtiSolver with a compile-time block size NB for kPartialCondensed.
template <class P, int NB>
using RtiBlock = RtiSolver<P, Eigen::Dynamic,
                           HpipmQpSolver<P, Eigen::Dynamic>,
                           GlmRegularizer<P, Eigen::Dynamic>, NB>;

// Forward-simulation starts with every multiplier pinned to zero (the
// box multipliers must be explicit: vector::resize default-inserts, see
// the phase-3 plan open item 4).
double di_feasible_control(const DoubleIntegrator& p, int n_stages)
{
    const double Ts = p.Ts_;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * n_stages * (n_stages - 1) / 2.0
                       + Ts * n_stages;
    return (1.0 - (q0 + v0) - Ts * n_stages * v0) / den;
}

Solution<DoubleIntegrator> make_di(int N, const DoubleIntegrator& p)
{
    const double a = di_feasible_control(p, N);
    Solution<DoubleIntegrator> sol(N);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol.u[k](0) = a;
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }
    for (int k = 0; k <= N; ++k)
    {
        sol.lambda_box_state[k].setZero();
    }
    for (int k = 0; k < N; ++k)
    {
        sol.lambda_dyn[k].setZero();
        sol.lambda_ineq_stage[k].setZero();
        sol.lambda_eq_stage[k].setZero();
        sol.lambda_lin_stage[k].setZero();
        sol.lambda_box_control[k].setZero();
    }
    sol.lambda_ineq_term.setZero();
    sol.lambda_eq_term.setZero();
    sol.lambda_lin_term.setZero();
    return sol;
}

Solution<MassesChain> make_mc(int N, const MassesChain& p)
{
    Solution<MassesChain> sol(N);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = p.cost_reference();
        sol.lambda_dyn[k].setZero();
        sol.lambda_box_state[k + 1].setZero();
        sol.lambda_box_control[k].setZero();
    }
    sol.lambda_box_state[0].setZero();
    sol.lambda_ineq_term.setZero();
    sol.lambda_eq_term.setZero();
    sol.lambda_lin_term.setZero();
    return sol;
}

Solution<MassSpring> make_ms(int N, const MassSpring& p)
{
    Solution<MassSpring> sol(N);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
        sol.lambda_dyn[k].setZero();
        sol.lambda_box_state[k + 1].setZero();
        sol.lambda_box_control[k].setZero();
    }
    sol.lambda_box_state[0].setZero();
    sol.lambda_ineq_term.setZero();
    sol.lambda_eq_term.setZero();
    sol.lambda_lin_term.setZero();
    return sol;
}

// ---------------------------------------------------------------------
// DoubleIntegrator N = 10: one RTI call == one Acados iteration; second
// call ~0 step; pre-step residual row matches compute_nlp_residuals
// ---------------------------------------------------------------------

void check_di_single_iter()
{
    const std::string name = "3d DI N=10";
    const int N = 10;
    DoubleIntegrator p;

    // AcadosSolver: exactly one QP solve + step (iteration 0); the loop
    // index runs 0..max_iter, so max_iter = 1 and the second iteration
    // exits at the termination check (status kMaxIterations).
    AcadosOptions ao;
    ao.max_iter = 1;
    AcadosSolver<DoubleIntegrator> ac(ao);
    Solution<DoubleIntegrator> sol_ac = make_di(N, p);
    const Status st_ac = ac.solve(p, sol_ac);
    check_true(ac.statistics().rows.size() >= 1, name + ": acados row");
    check_true(ac.statistics().rows[0].alpha == 1.0,
               name + ": first SQP iteration took a full step");
    check_true(st_ac == Status::kMaxIterations, name + ": max_iter exit");

    RtiOptions ro;
    ro.rti_log_residuals = true;
    RtiSolver<DoubleIntegrator> rti(ro);
    Solution<DoubleIntegrator> sol_r = make_di(N, p);
    check_true(rti.solve(p, sol_r) == Status::kSolved, name + ": rti solve");
    if (failures > 0)
    {
        return;
    }
    check_iterate(name, sol_ac, sol_r, 1e-10);

    // second call from the (near-optimal) result: ~0 step
    const double step1 = rti.statistics().last.step_norm;
    check_true(step1 > 0.0, name + ": first step non-trivial");
    check_true(rti.solve(p, sol_r) == Status::kSolved,
               name + ": second rti call");
    const double step2 = rti.statistics().last.step_norm;
    check_true(step2 < 1e-4 && step2 < 1e-3 * step1,
               name + ": second step ~0 (" + std::to_string(step2) +
                   " vs " + std::to_string(step1) + ")");

    // pre-step residual row == compute_nlp_residuals at the start
    // iterate (first-call slacks are zero)
    SqpSlacks<DoubleIntegrator> sl0(N);
    sl0.setZero();
    const NlpResiduals r0 =
        compute_nlp_residuals(p, make_di(N, p), &sl0, ro.tau_min);
    // rti.statistics().last is the second call; re-solve once on a fresh
    // solver to read the first-call row
    RtiSolver<DoubleIntegrator> rti1(ro);
    Solution<DoubleIntegrator> sol_r1 = make_di(N, p);
    rti1.solve(p, sol_r1);
    const auto& s1 = rti1.statistics().last;
    check_true(s1.res_stat > 0.0, name + ": pre res_stat logged");
    check_true(std::fabs(s1.res_stat - r0.res_stat)
                   < 1e-9 * std::max(1.0, std::fabs(r0.res_stat)),
               name + ": pre res_stat matches");
    check_true(std::fabs(s1.res_comp - r0.res_comp)
                   < 1e-9 * std::max(1.0, std::fabs(r0.res_comp)),
               name + ": pre res_comp matches");
}

// ---------------------------------------------------------------------
// MassesChain N = 40 (GN Hessian): one RTI call == one Acados iteration
// ---------------------------------------------------------------------

void check_mc_single_iter()
{
    const std::string name = "3d MC N=40";
    const int N = 40;
    MassesChain p;

    AcadosOptions ao;
    ao.max_iter = 1;
    AcadosSolver<MassesChain> ac(ao);
    Solution<MassesChain> sol_ac = make_mc(N, p);
    ac.solve(p, sol_ac);
    check_true(ac.statistics().rows.size() >= 1, name + ": acados row");
    check_true(ac.statistics().rows[0].alpha == 1.0,
               name + ": first SQP iteration took a full step");

    RtiOptions ro;
    RtiSolver<MassesChain> rti(ro);
    Solution<MassesChain> sol_r = make_mc(N, p);
    check_true(rti.solve(p, sol_r) == Status::kSolved, name + ": rti solve");
    if (failures > 0)
    {
        return;
    }
    check_iterate(name, sol_ac, sol_r, 1e-10);
}

// ---------------------------------------------------------------------
// MassSpring N = 20 (constant Hessian): one RTI call == CONVERGED
// Acados solution (exact LQ subproblem)
// ---------------------------------------------------------------------

void check_ms_single_call()
{
    const std::string name = "3d MS N=20";
    const int N = 20;
    MassSpring p;

    AcadosOptions ao;
    AcadosSolver<MassSpring> ac(ao);
    Solution<MassSpring> sol_ac = make_ms(N, p);
    const Status st_ac = ac.solve(p, sol_ac);
    check_true(st_ac == Status::kSolved, name + ": acados converged");

    RtiOptions ro;
    RtiSolver<MassSpring> rti(ro);
    Solution<MassSpring> sol_r = make_ms(N, p);
    check_true(rti.solve(p, sol_r) == Status::kSolved, name + ": rti solve");
    if (failures > 0)
    {
        return;
    }
    check_iterate(name, sol_ac, sol_r, 1e-8);
    const double rel_cost =
        std::fabs(sol_ac.cost_value - sol_r.cost_value)
        / std::max(1e-30, std::fabs(sol_ac.cost_value));
    check_true(rel_cost < 1e-10, name + ": cost rel diff < 1e-10 ("
                                      + std::to_string(rel_cost) + ")");
}

// ---------------------------------------------------------------------
// QP-mode cross-check: staged / full-condensed / partial (NB = 2)
// ---------------------------------------------------------------------

template <class P>
void check_qp_modes(const std::string& name, int N, const P& p,
                    Solution<P> (*mk)(int, const P&))
{
    RtiOptions o_s;
    RtiSolver<P> rti_s(o_s);
    Solution<P> sol_s = mk(N, p);
    check_true(rti_s.solve(p, sol_s) == Status::kSolved,
               name + ": staged");

    RtiOptions o_f;
    o_f.qp_mode = QpMode::kFullCondensed;
    RtiSolver<P> rti_f(o_f);
    Solution<P> sol_f = mk(N, p);
    check_true(rti_f.solve(p, sol_f) == Status::kSolved,
               name + ": full-condensed");

    RtiOptions o_p;
    o_p.qp_mode = QpMode::kPartialCondensed;
    o_p.qp_cond_N = N / 2;  // NB = 2
    RtiBlock<P, 2> rti_p(o_p);
    Solution<P> sol_p = mk(N, p);
    check_true(rti_p.solve(p, sol_p) == Status::kSolved,
               name + ": partial (NB=2)");
    if (failures > 0)
    {
        return;
    }
    check_iterate(name + " staged vs full", sol_s, sol_f, 1e-8);
    check_iterate(name + " staged vs partial", sol_s, sol_p, 1e-8);
}

void check_modes()
{
    const int N = 10;
    DoubleIntegrator di;
    check_qp_modes("3d modes DI", N, di, make_di);

    const int NM = 40;
    MassesChain mc;
    check_qp_modes("3d modes MC", NM, mc, make_mc);
}

}  // namespace

int run_rti_3d_tests()
{
    failures = 0;
    check_di_single_iter();
    check_mc_single_iter();
    check_ms_single_call();
    check_modes();
    if (failures == 0)
    {
        std::printf("  3d rti parity (DI/MC one-iter, MS single-call, "
                    "qp modes) OK\n");
    }
    return failures;
}

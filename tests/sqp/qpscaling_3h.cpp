// Phase 3 sub-step 3h test: QP scaling + adaptive QP tolerances.
//
// Covers (SQP_PHASE3_PLAN.md sec. 3h):
//   - objective scaling factor (Gershgorin max-abs-eig + grad floor),
//   - constraint scaling (per-general-row s_j, DC/d scaling, slack-cost
//     rescale) and the inverse rescale of the solved duals/slacks,
//   - the per-iteration QP tolerance strategies (kFixedQpTol,
//     kAdaptiveCurrentResJoint, kAdaptiveQpScaling),
//   - end-to-end: solving with the scalings on reproduces the unscaled
//     primal / cost and converges to kSolved.

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/solvers/acados/sqp.hpp"
#include "ocp/solvers/acados/qpscaling.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

namespace
{

using namespace ocp;
using DI = DoubleIntegrator;

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

void check_close(double a, double b, double tol, const std::string& msg)
{
    if (std::fabs(a - b) > tol)
    {
        std::fprintf(stderr, "FAIL: %s (%.15e vs %.15e)\n", msg.c_str(), a,
                     b);
        ++failures;
    }
}

// ---------------------------------------------------------------------
// Objective scaling factor (acados ocp_nlp_qpscaling.c:483-540)
// ---------------------------------------------------------------------

void test_obj_factor_no_scaling()
{
    Qp<DI> qp(1);
    // (u;x) Hessian diagonal (1, 2, 3), Z diagonal 1 -> max-abs-eig 3.
    qp.first.hess << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0;
    qp.first.hess(3, 3) = 1.0;  // slack slot
    qp.first.grad << 10.0, 20.0, 0.5;

    // max_abs_eig = 3 < ub (1e5) -> factor 1.0; grad inf-norm 20 > 1e-4
    // (1.0 * 20 > lb) -> no upscale.
    const double f =
        QpScaler<DI>::compute_obj_scaling_factor(qp, 1e5, 1e-4);
    check_close(f, 1.0, 1e-15, "obj factor: no scaling -> 1.0");
}

void test_obj_factor_downscale()
{
    Qp<DI> qp(1);
    // (u;x) Hessian diagonal (100, 1, 2) -> Gershgorin max-abs-eig 100.
    qp.first.hess << 100.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0;
    qp.first.hess(3, 3) = 1.0;
    qp.first.grad << 0.1, 0.1, 0.1;

    // max_abs_eig = 100. ub = 1e2 -> factor = 1e2/100 = 1.0.
    check_close(QpScaler<DI>::compute_obj_scaling_factor(qp, 1e2, 1e-4),
                1.0, 1e-15, "obj factor: eig == ub -> 1.0");
    // ub = 10 -> factor = 10/100 = 0.1.
    check_close(QpScaler<DI>::compute_obj_scaling_factor(qp, 10.0, 1e-4),
                0.1, 1e-15, "obj factor: eig > ub -> ub/eig");
}

void test_obj_factor_grad_upscale()
{
    Qp<DI> qp(1);
    qp.first.hess << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0;
    qp.first.hess(3, 3) = 1.0;
    qp.first.grad << 1e-6, 1e-6, 1e-6;

    // max_abs_eig = 3 < ub -> factor 1.0, max_upscale = 1e5/3.
    // grad inf-norm 1e-6 -> 1.0 * 1e-6 <= lb (1e-4) -> upscale to
    // min(max_upscale, lb/grad) = min(3.33e4, 1e-4/1e-6) = 1e2.
    check_close(QpScaler<DI>::compute_obj_scaling_factor(qp, 1e5, 1e-4),
                1e2, 1e-9, "obj factor: small grad -> lb/grad");
}

// ---------------------------------------------------------------------
// Constraint scaling + rescale round trip (hand-computed s_j)
// ---------------------------------------------------------------------

// A minimal single-stage QP: a first-stage ineq row (the only soft row)
// with a large bound (forces s_j < 1) and a linear row (s_j = 1, a no-op).
Qp<DI> make_probe_qp()
{
    using D = QpDim<DI>;
    Qp<DI> qp(1);
    const auto& lay = D::lay_first;
    const int r_ineq = lay.row_off(detail::g_ineq);  // 2
    const int r_lin = lay.row_off(detail::g_lin);     // 4
    const int shi = lay.side_hi(r_ineq);             // 5
    const int llo = lay.side_lo(r_lin);              // 4
    const int lhi = lay.side_hi(r_lin);             // 7
    const int clo = D::idxs_hi_first[r_ineq];        // 3

    qp.first.hess.setZero();
    qp.first.grad.setZero();
    qp.first.DC.setZero();
    qp.first.d.setZero();
    qp.first.d_mask.setZero();
    // (u;x) diagonal Hessian + ineq-slack weight 2.0.
    for (int i = 0; i < 3; ++i)
    {
        qp.first.hess(i, i) = 1.0;
    }
    qp.first.hess(clo, clo) = 2.0;

    // ineq row: DC(u;x) = (1, 2, 3) (coeff inf-norm 3), d_hi = 10.
    qp.first.DC(r_ineq, 0) = 1.0;
    qp.first.DC(r_ineq, 1) = 2.0;
    qp.first.DC(r_ineq, 2) = 3.0;
    qp.first.d(shi) = 10.0;
    qp.first.d_mask(shi) = 1.0;

    // linear row: DC(u;x) = (1, 0, 0), d_lo = 3, d_hi = 4.
    qp.first.DC(r_lin, 0) = 1.0;
    qp.first.d(llo) = 3.0;
    qp.first.d(lhi) = 4.0;
    qp.first.d_mask(llo) = 1.0;
    qp.first.d_mask(lhi) = 1.0;
    return qp;
}

void test_constraint_scaling()
{
    using D = QpDim<DI>;
    const Qp<DI> qp = make_probe_qp();
    const auto& lay = D::lay_first;
    const int r_ineq = lay.row_off(detail::g_ineq);
    const int r_lin = lay.row_off(detail::g_lin);
    const int shi = lay.side_hi(r_ineq);
    const int llo = lay.side_lo(r_lin);
    const int lhi = lay.side_hi(r_lin);
    const int clo = D::idxs_hi_first[r_ineq];

    typename QpScaler<DI>::Opts o;
    o.scale_objective = false;
    o.scale_constraints = true;
    QpScaler<DI> sc(o);
    sc.resize(1);
    sc.scale_qp(qp);
    const Qp<DI>& s = sc.scaled_in();

    // ineq row: s_j = 1/max(1, max(bound 10, coeff 3)) = 0.1.
    check_close(sc.min_constr_scaling(), 0.1, 1e-15,
                "min constraint scaling == ineq s_j");
    check_close(static_cast<double>(s.first.DC(r_ineq, 0)), 0.1, 1e-15,
                "ineq DC[0] scaled x0.1");
    check_close(static_cast<double>(s.first.DC(r_ineq, 2)), 0.3, 1e-15,
                "ineq DC[2] scaled x0.1");
    check_close(static_cast<double>(s.first.d(shi)), 1.0, 1e-15,
                "ineq d_hi scaled x0.1");
    // slack cost rescale: H /= s^2, grad /= s.
    check_close(static_cast<double>(s.first.hess(clo, clo)), 200.0, 1e-9,
                "ineq slack hess /= s^2");

    // linear row: s_j = 1/max(1, max(bound max(|3|,|4|)=4, coeff 1)) = 0.25.
    check_close(static_cast<double>(s.first.DC(r_lin, 0)), 0.25, 1e-15,
                "lin DC scaled x0.25");
    check_close(static_cast<double>(s.first.d(llo)), 0.75, 1e-15,
                "lin d_lo scaled x0.25");
    check_close(static_cast<double>(s.first.d(lhi)), 1.0, 1e-15,
                "lin d_hi scaled x0.25");
}

void test_constraint_rescale()
{
    using D = QpDim<DI>;
    const Qp<DI> qp = make_probe_qp();
    const auto& lay = D::lay_first;
    const int r_ineq = lay.row_off(detail::g_ineq);
    const int shi = lay.side_hi(r_ineq);
    const int clo = D::idxs_hi_first[r_ineq];  // 3
    const int nv0 = D::nu + D::nx;             // 3
    const int ssbase = lay.lo_size() + lay.nrow();  // 5

    typename QpScaler<DI>::Opts o;
    o.scale_constraints = true;
    QpScaler<DI> sc(o);
    sc.resize(1);
    sc.scale_qp(qp);

    // Fabricate a solved scaled-space solution with a known ineq row.
    auto& sout = sc.scaled_out();
    sout.ux_first.setZero();
    sout.lam_first.setZero();
    sout.ux_first(clo) = 0.5;              // scaled-space ineq slack
    sout.lam_first(shi) = 2.0;             // scaled-space ineq hi multiplier
    sout.lam_first(ssbase + (clo - nv0)) = 3.0;  // scaled slack multiplier

    QpSol<DI> out;
    out.resize(1);
    sc.rescale_solution(out);

    // s_j = 0.1: slack var /= s, multipliers *= s.
    check_close(static_cast<double>(out.ux_first(clo)), 5.0, 1e-15,
                "rescale: ineq slack /= s");
    check_close(static_cast<double>(out.lam_first(shi)), 0.2, 1e-15,
                "rescale: ineq hi lam *= s");
    check_close(
        static_cast<double>(out.lam_first(ssbase + (clo - nv0))), 0.3, 1e-15,
        "rescale: ineq slack lam *= s");
    check_close(static_cast<double>(out.ux_first(0)), 0.0, 1e-15,
                "rescale: u step unchanged");
}

// ---------------------------------------------------------------------
// End-to-end: scaled vs unscaled give the same solution
// ---------------------------------------------------------------------

Solution<DI> make_warm_start(const DI& p, int n_stages)
{
    // Constant control that keeps the terminal equality q_N + v_N = 1
    // exactly feasible at the warm start (mirrors the 2h DI test).
    const double Ts = p.Ts_;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * n_stages * (n_stages - 1) / 2.0 +
                       Ts * n_stages;
    const double a = (1.0 - (q0 + v0) - Ts * n_stages * v0) / den;

    Solution<DI> sol(n_stages);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < n_stages; ++k)
    {
        sol.u[k](0) = a;
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }
    return sol;
}

void test_scaled_matches_unscaled()
{
    DI prob;
    const int N = 10;

    SqpOptions off;
    off.max_iter = 20;
    off.scale_qp_objective = false;
    off.scale_qp_constraints = false;
    SqpSolver<DI> solver_off(off);
    Solution<DI> sol_off = make_warm_start(prob, N);
    const Status st_off = solver_off.solve(prob, sol_off);
    check(st_off == Status::kSolved, "unscaled: kSolved");

    SqpOptions on;
    on = off;
    on.scale_qp_objective = true;
    on.scale_qp_constraints = true;
    SqpSolver<DI> solver_on(on);
    Solution<DI> sol_on = make_warm_start(prob, N);
    const Status st_on = solver_on.solve(prob, sol_on);
    check(st_on == Status::kSolved, "scaled: kSolved");

    check_close(sol_on.cost_value, sol_off.cost_value, 1e-6,
                "scaled vs unscaled: cost");
    double dx = 0.0, du = 0.0;
    for (int k = 0; k <= N; ++k)
    {
        dx = std::max(
            dx, static_cast<double>(
                    (sol_on.x[k] - sol_off.x[k]).cwiseAbs().maxCoeff()));
    }
    for (int k = 0; k < N; ++k)
    {
        du = std::max(
            du, static_cast<double>(
                    (sol_on.u[k] - sol_off.u[k]).cwiseAbs().maxCoeff()));
    }
    check(dx < 1e-8, "scaled vs unscaled: dx < 1e-8");
    check(du < 1e-8, "scaled vs unscaled: du < 1e-8");
}

// ---------------------------------------------------------------------
// End-to-end: adaptive QP tolerance strategies still converge
// ---------------------------------------------------------------------

void test_adaptive_tolerance_strategies()
{
    DI prob;
    const int N = 10;

    for (int strategy = 1; strategy <= 2; ++strategy)
    {
        SqpOptions opts;
        opts.max_iter = 30;
        opts.nlp_qp_tol_strategy = strategy;
        opts.scale_qp_constraints = true;  // gives strategy 2 a real factor
        SqpSolver<DI> solver(opts);
        Solution<DI> sol = make_warm_start(prob, N);
        const Status st = solver.solve(prob, sol);
        check(st == Status::kSolved,
              "adaptive tol strategy " + std::to_string(strategy) +
                  ": kSolved");
        check(solver.statistics().iter > 0,
              "adaptive tol strategy " + std::to_string(strategy) +
                  ": >= 1 iteration");
    }
}

void test_end_to_end_converges()
{
    DI prob;
    const int N = 10;

    // All features on at once.
    SqpOptions opts;
    opts.max_iter = 30;
    opts.scale_qp_objective = true;
    opts.scale_qp_constraints = true;
    opts.nlp_qp_tol_strategy = 2;
    SqpSolver<DI> solver(opts);
    Solution<DI> sol = make_warm_start(prob, N);
    const Status st = solver.solve(prob, sol);
    check(st == Status::kSolved, "all-scaling end-to-end: kSolved");
}

}  // namespace

int run_qpscaling_3h_tests()
{
    failures = 0;
    test_obj_factor_no_scaling();
    test_obj_factor_downscale();
    test_obj_factor_grad_upscale();
    test_constraint_scaling();
    test_constraint_rescale();
    test_scaled_matches_unscaled();
    test_adaptive_tolerance_strategies();
    test_end_to_end_converges();

    if (failures == 0)
    {
        std::printf("All QP scaling (3h) checks passed.\n");
    }
    return failures;
}

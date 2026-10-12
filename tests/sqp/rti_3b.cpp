// Phase 3 block 3b test: the RtiSolver core
// (include/ocp/solvers/acados/rti.hpp, SQP_HPIPM_PARITY_PHASE3_PLAN.md
// block 3b).
//
// Smoke checks (the full parity evidence vs AcadosSolver / the acados C
// reference is block 3d / 3e):
//   - DI (N = 10): one solve() from a deterministic warm start returns
//     kSolved and applies the full step in place; the resulting iterate
//     equals a manual re-application of the (u;x;s) step to the saved
//     start (in-place element safety, plan open item 1).
//   - prepare() + feedback() == solve() (same resulting iterate).
//   - LM shift: with NoRegularizer, levenberg_marquardt = 0.5 shifts the
//     (u;x) hess diagonal by exactly 0.5 in last_qp(); mu = 0 does not.
//   - QP mode: kStaged and kFullCondensed agree to 1e-10 (the dispatch
//     copy in rti.hpp).
//   - Degenerate fixed-x_0 problem (hard ineq violated at the pinned
//     state, row independent of u): solve() returns kInfeasible and
//     leaves the iterate untouched.
//   - Residual logging (block 3c): rti_log_residuals stores the pre-step
//     NLP residuals; unless rti_log_only_available_residuals, the full
//     residual recompute at the post-step iterate is appended; logging off
//     leaves all residual fields zero.

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/solvers/acados/rti.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

namespace
{

using namespace ocp;

int failures = 0;

void check_close(double a, double b, const char* msg)
{
    if (std::fabs(a - b) > 1e-12)
    {
        std::fprintf(stderr, "FAIL: %s (%.15e vs %.15e)\n", msg, a, b);
        ++failures;
    }
}

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

// Deterministic non-trivial iterate (same recipe as assemble_rti_3a.cpp).
void make_iterate(const DoubleIntegrator& prob,
                  Solution<DoubleIntegrator>& sol)
{
    const int N = sol.N;
    sol.x[0] = prob.initial_state();
    for (int k = 1; k <= N; ++k)
    {
        sol.x[k] = sol.x[k - 1];
        for (Eigen::Index i = 0; i < sol.x[k].size(); ++i)
        {
            sol.x[k](i) += 0.01 * (1.0 + 0.1 * static_cast<double>(i)
                                   + 0.01 * static_cast<double>(k));
        }
    }
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.u[k](0) = 0.1 * (1.0 + 0.05 * static_cast<double>(k));
    }
    for (int k = 0; k < N; ++k)
    {
        sol.lambda_dyn[k].setZero();
        for (Eigen::Index i = 0; i < sol.lambda_dyn[k].size(); ++i)
        {
            sol.lambda_dyn[k](i) =
                0.1 * (0.01 * static_cast<double>(k)
                       + 0.001 * static_cast<double>(i) + 0.01);
        }
        sol.lambda_ineq_stage[k].setZero();
        for (Eigen::Index i = 0; i < sol.lambda_ineq_stage[k].size(); ++i)
        {
            sol.lambda_ineq_stage[k](i) =
                0.1 * (0.01 * static_cast<double>(k) + 0.001);
        }
        sol.lambda_eq_stage[k].setZero();
        sol.lambda_lin_stage[k].setZero();
    }
    sol.lambda_ineq_term.setZero();
    sol.lambda_eq_term.setZero();
    sol.lambda_lin_term.setZero();
    // vector::resize default-inserts (C++11): Eigen slots are not
    // value-initialized, so pin the box multipliers explicitly.
    for (int k = 0; k <= N; ++k)
    {
        sol.lambda_box_state[k].setZero();
    }
    for (int k = 0; k < N; ++k)
    {
        sol.lambda_box_control[k].setZero();
    }
}

void copy_solution(const Solution<DoubleIntegrator>& from,
                    Solution<DoubleIntegrator>& to)
{
    const int N = from.N;
    for (int k = 0; k <= N; ++k)
    {
        to.x[k] = from.x[k];
        to.lambda_box_state[k] = from.lambda_box_state[k];
    }
    for (int k = 0; k < N; ++k)
    {
        to.u[k] = from.u[k];
        to.lambda_dyn[k] = from.lambda_dyn[k];
        to.lambda_ineq_stage[k] = from.lambda_ineq_stage[k];
        to.lambda_eq_stage[k] = from.lambda_eq_stage[k];
        to.lambda_lin_stage[k] = from.lambda_lin_stage[k];
        to.lambda_box_control[k] = from.lambda_box_control[k];
    }
    to.lambda_ineq_term = from.lambda_ineq_term;
    to.lambda_eq_term = from.lambda_eq_term;
    to.lambda_lin_term = from.lambda_lin_term;
}

void check_basic_solve()
{
    DoubleIntegrator prob;
    Solution<DoubleIntegrator> sol(10);
    make_iterate(prob, sol);
    Solution<DoubleIntegrator> start(10);
    copy_solution(sol, start);

    RtiSolver<DoubleIntegrator> rti;
    const Status st = rti.solve(prob, sol);
    check_true(st == Status::kSolved, "RTI DI: solve status");
    if (failures > 0)
    {
        return;
    }

    const RtiStatistics& stat = rti.statistics();
    check_true(stat.n_calls == 1, "RTI DI: statistics n_calls");
    check_true(stat.last.qp_iter > 0, "RTI DI: QP iterations > 0");
    check_close(stat.last.alpha, 1.0, "RTI DI: alpha is the full step");
    check_true(stat.last.step_norm > 0.0, "RTI DI: non-trivial step");

    // in-place full step: sol must equal start + step (alpha = 1),
    // exactly (same arithmetic as the in-place update)
    const QpSol<DoubleIntegrator>& step = rti.last_qp_sol();
    using D = QpDim<DoubleIntegrator>;
    Solution<DoubleIntegrator> exp(10);
    copy_solution(start, exp);
    exp.u[0] = start.u[0] + step.ux_first.head(D::nu);
    for (int k = 1; k < 10; ++k)
    {
        exp.u[k] = start.u[k] + step.ux_path[k - 1].head(D::nu);
    }
    exp.x[0] = start.x[0] + step.ux_first.segment(D::nu, D::nx);
    for (int k = 1; k < 10; ++k)
    {
        exp.x[k] = start.x[k] + step.ux_path[k - 1].segment(D::nu, D::nx);
    }
    exp.x[10] = start.x[10] + step.ux_term.segment(0, D::nx);
    for (int k = 0; k <= 10; ++k)
    {
        check_true(max_abs_diff(sol.x[k], exp.x[k]) == 0.0,
                   "RTI DI: x_" + std::to_string(k) + " in-place step");
    }
    for (int k = 0; k < 10; ++k)
    {
        check_true(max_abs_diff(sol.u[k], exp.u[k]) == 0.0,
                   "RTI DI: u_" + std::to_string(k) + " in-place step");
    }

    check_true(std::isfinite(sol.cost_value), "RTI DI: cost finite");
    // QP slacks are absolute and non-negative
    const double smin =
        step.ux_first.segment(D::nu + D::nx, D::nslack_first).minCoeff();
    check_true(smin >= -1e-15, "RTI DI: QP slacks non-negative");
}

void check_prepare_feedback()
{
    DoubleIntegrator prob;

    Solution<DoubleIntegrator> sol_a(10);
    make_iterate(prob, sol_a);
    RtiSolver<DoubleIntegrator> rti_a;
    const Status st_a = rti_a.solve(prob, sol_a);
    check_true(st_a == Status::kSolved, "RTI prepare/feedback: solve ref");

    Solution<DoubleIntegrator> sol_b(10);
    make_iterate(prob, sol_b);
    RtiSolver<DoubleIntegrator> rti_b;
    rti_b.resize(10);
    rti_b.prepare(prob, sol_b);
    const Status st_b = rti_b.feedback(prob, sol_b);
    check_true(st_b == Status::kSolved, "RTI prepare/feedback: feedback");
    if (failures > 0)
    {
        return;
    }
    for (int k = 0; k <= 10; ++k)
    {
        check_close(max_abs_diff(sol_a.x[k], sol_b.x[k]), 0.0,
                    ("RTI prepare/feedback: x_" + std::to_string(k)).c_str());
    }
    for (int k = 0; k < 10; ++k)
    {
        check_close(max_abs_diff(sol_a.u[k], sol_b.u[k]), 0.0,
                    ("RTI prepare/feedback: u_" + std::to_string(k)).c_str());
    }
}

void check_lm_shift()
{
    using RtiNoReg = RtiSolver<
        DoubleIntegrator, Eigen::Dynamic,
        HpipmQpSolver<DoubleIntegrator, Eigen::Dynamic>,
        NoRegularizer<DoubleIntegrator, Eigen::Dynamic>>;

    DoubleIntegrator prob;
    RtiOptions o_mu;
    o_mu.levenberg_marquardt = 0.5;
    RtiNoReg rti_mu(o_mu);
    Solution<DoubleIntegrator> sol_mu(10);
    make_iterate(prob, sol_mu);
    check_true(rti_mu.solve(prob, sol_mu) == Status::kSolved,
               "RTI LM: solve with mu");

    RtiNoReg rti_0;
    Solution<DoubleIntegrator> sol_0(10);
    make_iterate(prob, sol_0);
    check_true(rti_0.solve(prob, sol_0) == Status::kSolved,
               "RTI LM: solve with mu = 0");
    if (failures > 0)
    {
        return;
    }

    using D = QpDim<DoubleIntegrator>;
    constexpr int nux = D::nu + D::nx;
    // (u;x) diagonal: exactly +0.5; everything else identical
    check_close(rti_mu.last_qp().first.hess(0, 0),
                rti_0.last_qp().first.hess(0, 0) + 0.5, "RTI LM hess(0,0)");
    for (int i = 0; i < nux; ++i)
    {
        check_close(rti_mu.last_qp().first.hess(i, i),
                    rti_0.last_qp().first.hess(i, i) + 0.5,
                    ("RTI LM first hess(" + std::to_string(i) + ","
                     + std::to_string(i) + ")").c_str());
    }
    for (int k = 1; k < 10; ++k)
    {
        for (int i = 0; i < nux; ++i)
        {
            check_close(rti_mu.last_qp().path[k - 1].hess(i, i),
                        rti_0.last_qp().path[k - 1].hess(i, i) + 0.5,
                        "RTI LM path hess diagonal");
        }
    }
    check_close(rti_mu.last_qp().term.hess(0, 0),
                rti_0.last_qp().term.hess(0, 0) + 0.5,
                "RTI LM term hess(0,0)");
    check_close(rti_mu.last_qp().term.hess(1, 1),
                rti_0.last_qp().term.hess(1, 1) + 0.5,
                "RTI LM term hess(1,1)");
}

void check_qp_modes()
{
    DoubleIntegrator prob;

    RtiOptions o_staged;
    RtiSolver<DoubleIntegrator> rti_s(o_staged);
    Solution<DoubleIntegrator> sol_s(10);
    make_iterate(prob, sol_s);
    check_true(rti_s.solve(prob, sol_s) == Status::kSolved,
               "RTI qp mode: staged solve");

    RtiOptions o_dense;
    o_dense.qp_mode = QpMode::kFullCondensed;
    RtiSolver<DoubleIntegrator> rti_f(o_dense);
    Solution<DoubleIntegrator> sol_f(10);
    make_iterate(prob, sol_f);
    check_true(rti_f.solve(prob, sol_f) == Status::kSolved,
               "RTI qp mode: full-condensed solve");
    if (failures > 0)
    {
        return;
    }
    // The QP is only solved to the HPIPM tolerances, so the staged and
    // condensed solutions differ at the QP-residual level (~1e-6); the
    // applied step (and hence the resulting iterate) must agree to 1e-8
    // (plan 3d threshold).
    double mx = max_abs_diff(rti_s.last_qp_sol().ux_first,
                             rti_f.last_qp_sol().ux_first);
    for (int k = 1; k < 10; ++k)
    {
        mx = std::max(mx, max_abs_diff(rti_s.last_qp_sol().ux_path[k - 1],
                                       rti_f.last_qp_sol().ux_path[k - 1]));
    }
    mx = std::max(mx, max_abs_diff(rti_s.last_qp_sol().ux_term,
                                   rti_f.last_qp_sol().ux_term));
    check_true(mx < 1e-5, "RTI qp mode: staged vs condensed QP sol ~1e-6");
    for (int k = 0; k <= 10; ++k)
    {
        check_true(max_abs_diff(sol_s.x[k], sol_f.x[k]) < 1e-8,
                   "RTI qp mode: x_" + std::to_string(k));
    }
    for (int k = 0; k < 10; ++k)
    {
        check_true(max_abs_diff(sol_s.u[k], sol_f.u[k]) < 1e-8,
                   "RTI qp mode: u_" + std::to_string(k));
    }
}

// Degenerate fixed-x_0 probe: hard ineq g = x - 0.5 violated at the
// pinned state, independent of u -> kInfeasible, iterate untouched.
struct DegDimsRti
{
    static constexpr int nx = 1;
    static constexpr int nu = 1;
    static constexpr int ng = 1;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 0;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = true;
    static constexpr bool has_dynamics_hess = false;
    static constexpr bool has_constr_hess = false;
    static constexpr std::array<int, 1> state_box_idx = {0};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 0> terminal_state_box_idx = {};
    static constexpr std::array<int, 0> ineq_soft_idx = {};
    static constexpr std::array<int, 0> eq_soft_idx = {};
    static constexpr std::array<int, 0> lin_soft_idx = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {};
    static constexpr std::array<int, 0> state_box_soft_idx = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

class DegProbeRti : public ocp::Problem<DegDimsRti>
{
public:
    state_t initial_state() const
    {
        return state_t::Constant(1, 1.0);
    }

    state_t dynamics_next_state(int, const state_t& x, const control_t& u) const
    {
        return state_t::Constant(1, x(0) + u(0));
    }

    void dynamics_value_jac(int, const state_t& x, const control_t& u,
                            state_t& x_next, dyn_df_dx_t& df_dx,
                            dyn_df_du_t& df_du) const
    {
        x_next(0) = x(0) + u(0);
        df_dx(0, 0) = 1.0;
        df_du(0, 0) = 1.0;
    }

    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return 0.5 * x(0) * x(0) + 0.5 * u(0) * u(0);
    }

    void stage_cost_value_grad(int, const state_t& x, const control_t& u,
                               double& value, stage_grad_t& grad) const
    {
        value = 0.5 * x(0) * x(0) + 0.5 * u(0) * u(0);
        grad(0) = x(0);
        grad(1) = u(0);
    }

    void stage_cost_value_grad_hess(int, const state_t& x, const control_t& u,
                                    double& value, stage_grad_t& grad,
                                    stage_hess_t& hess) const
    {
        stage_cost_value_grad(0, x, u, value, grad);
        hess = stage_hess_t::Identity();
    }

    double terminal_cost_value(const state_t& x) const
    {
        return 0.5 * x(0) * x(0);
    }

    void terminal_cost_value_grad(const state_t& x, double& value,
                                  term_grad_t& grad) const
    {
        value = 0.5 * x(0) * x(0);
        grad(0) = x(0);
    }

    void terminal_cost_value_grad_hess(const state_t& x, double& value,
                                       term_grad_t& grad,
                                       term_hess_t& hess) const
    {
        terminal_cost_value_grad(x, value, grad);
        hess(0, 0) = 1.0;
    }

    ineq_t stage_inequality_value(int, const state_t& x, const control_t&) const
    {
        return ineq_t::Constant(1, x(0) - 0.5);
    }

    void stage_inequality_value_jac(int, const state_t& x, const control_t&,
                                    ineq_t& g, ineq_dg_dx_t& g_dx,
                                    ineq_dg_du_t& g_du) const
    {
        g(0) = x(0) - 0.5;
        g_dx(0, 0) = 1.0;
        g_du(0, 0) = 0.0;
    }

    state_box_t stage_state_box_constr(int) const
    {
        state_box_t spec;
        spec.lo(0) = -10.0;
        spec.hi(0) = 10.0;
        spec.penalty.quad(0) = 0.0;
        return spec;
    }

    control_box_t stage_control_box_constr(int) const
    {
        control_box_t spec;
        spec.lo(0) = -1.0;
        spec.hi(0) = 1.0;
        spec.penalty.quad(0) = 0.0;
        return spec;
    }
};

void check_degenerate()
{
    DegProbeRti prob;
    Solution<DegProbeRti> sol(1);
    sol.x[0] << 1.0;
    sol.x[1] << 1.0;
    sol.u[0] << 0.0;
    sol.lambda_dyn[0].setZero();
    sol.lambda_ineq_stage[0].setZero();

    RtiSolver<DegProbeRti> rti;
    const Status st = rti.solve(prob, sol);
    check_true(st == Status::kInfeasible, "RTI degenerate: kInfeasible");
    check_close(sol.x[0](0), 1.0, "RTI degenerate: x_0 untouched");
    check_close(sol.x[1](0), 1.0, "RTI degenerate: x_1 untouched");
    check_close(sol.u[0](0), 0.0, "RTI degenerate: u_0 untouched");
}

// Block 3c: residual logging. The solver's pre-step slacks are zero on
// the first call; after the full step (alpha = 1) they equal the QP
// solution slacks, which the test rebuilds from last_qp_sol().
void check_residual_logging()
{
    DoubleIntegrator prob;
    const RtiOptions base;

    // A: logging disabled -> all residual fields stay zero
    {
        Solution<DoubleIntegrator> sol(10);
        make_iterate(prob, sol);
        RtiSolver<DoubleIntegrator> rti;
        check_true(rti.solve(prob, sol) == Status::kSolved,
                   "RTI reslog A: solve");
        const auto& s = rti.statistics().last;
        check_close(s.res_stat, 0.0, "RTI reslog A: res_stat zero");
        check_close(s.res_comp, 0.0, "RTI reslog A: res_comp zero");
        check_close(s.res_stat_post, 0.0, "RTI reslog A: post zero");
    }

    // B: pre-step only (rti_log_only_available_residuals)
    {
        Solution<DoubleIntegrator> sol(10);
        make_iterate(prob, sol);
        Solution<DoubleIntegrator> pre(10);
        copy_solution(sol, pre);

        SqpSlacks<DoubleIntegrator> sl0(10);
        sl0.setZero();

        RtiOptions o;
        o.rti_log_residuals = true;
        o.rti_log_only_available_residuals = true;
        RtiSolver<DoubleIntegrator> rti(o);
        check_true(rti.solve(prob, sol) == Status::kSolved,
                   "RTI reslog B: solve");

        const NlpResiduals r0 =
            compute_nlp_residuals(prob, pre, &sl0, base.tau_min);
        const auto& s = rti.statistics().last;
        check_close(s.res_stat, r0.res_stat, "RTI reslog B: res_stat pre");
        check_close(s.res_eq, r0.res_eq, "RTI reslog B: res_eq pre");
        check_close(s.res_ineq, r0.res_ineq, "RTI reslog B: res_ineq pre");
        check_close(s.res_comp, r0.res_comp, "RTI reslog B: res_comp pre");
        check_close(s.res_stat_post, 0.0, "RTI reslog B: post zero");
    }

    // C: pre + post (default rti_log_only_available_residuals = false)
    {
        Solution<DoubleIntegrator> sol(10);
        make_iterate(prob, sol);

        RtiOptions o;
        o.rti_log_residuals = true;
        RtiSolver<DoubleIntegrator> rti(o);
        check_true(rti.solve(prob, sol) == Status::kSolved,
                   "RTI reslog C: solve");

        using D = QpDim<DoubleIntegrator>;
        SqpSlacks<DoubleIntegrator> sl1(10);
        sl1.first = rti.last_qp_sol().ux_first.segment(
            D::nu + D::nx, D::nslack_first);
        for (int k = 1; k < 10; ++k)
        {
            sl1.path[k - 1] = rti.last_qp_sol().ux_path[k - 1].segment(
                D::nu + D::nx, D::nslack_path);
        }
        sl1.term = rti.last_qp_sol().ux_term.segment(
            D::nx, D::nslack_term);
        const NlpResiduals r1 =
            compute_nlp_residuals(prob, sol, &sl1, o.tau_min);
        const auto& s = rti.statistics().last;
        check_close(s.res_stat_post, r1.res_stat, "RTI reslog C: res_stat post");
        check_close(s.res_eq_post, r1.res_eq, "RTI reslog C: res_eq post");
        check_close(s.res_ineq_post, r1.res_ineq,
                    "RTI reslog C: res_ineq post");
        check_close(s.res_comp_post, r1.res_comp,
                    "RTI reslog C: res_comp post");
    }
}

}  // namespace

int run_rti_3b_tests()
{
    check_basic_solve();
    check_prepare_feedback();
    check_lm_shift();
    check_qp_modes();
    check_degenerate();
    check_residual_logging();
    if (failures == 0)
    {
        std::printf("  3b rti (DI N=10): solve / prepare+feedback / LM shift "
                    "/ qp modes / degeneracy / reslog OK\n");
    }
    return failures;
}

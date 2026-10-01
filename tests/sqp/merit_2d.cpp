// Phase 2 sub-step 2d test: MeritBacktracking (merit function, weight
// update, backtracking line search) from
// include/ocp/solvers/acados/globalize.hpp.
//
// Checks (SQP_PHASE2_PLAN.md sec. 2d, acados
// ocp_nlp_globalization_merit_backtracking.c:290-408, :606-755):
//   - merit value: hand-computed cost + slack penalty + weighted L1
//     dynamics gap + weighted positive violations (N=1 and N=2)
//   - slack-relaxed ineq violation (g > 0, partial slack)
//   - box violation (x > hi)
//   - weight initialization and Leineweber update rule
//   - full-step accept (alpha=1, kSolved)
//   - backtracking (alpha=1 rejected, alpha=0.7 accepted; relaxed duals)
//   - kMinStep (all alpha rejected; cur still advanced)
//   - kNanDetected (NaN merit; cur untouched)
//   - detail::shift_slacks (trial slack computation)

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include "ocp/solvers/acados/globalize.hpp"

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

void check_close(double a, double b, const std::string& msg)
{
    if (std::fabs(a - b) > 1e-12)
    {
        std::fprintf(stderr, "FAIL: %s (%.12e vs %.12e)\n", msg.c_str(), a, b);
        ++failures;
    }
}

// ---------------------------------------------------------------------
// Test problem: nx=1, nu=1, one soft ineq (g = u-5 <= 0, penalty 4),
// hard state box (-10 <= x <= 10), no control/terminal/eq/lin rows.
//   dynamics:  x_{k+1} = 0.5*x + u
//   stage:     L = 0.5*x^2 + 0.5*u^2
//   terminal:  L_N = 10*x^2
//   fixed_initial_state = false
//
// QpDim: nvar_first = nvar_path = 3 (u;x;s), nvar_term = 1 (x)
//        nslack_first = nslack_path = 1 (ineq slack), nslack_term = 0
//        nrow_first = nrow_path = 2 (bx, ineq), nrow_term = 0
//        nside_first = nside_path = 4
// Side layout: [bx_lo=0, bx_hi=1, ineq_hi=2, slack=3]
// ---------------------------------------------------------------------
struct MeritDims
{
    static constexpr int nx = 1, nu = 1;
    static constexpr int ng = 1, ne = 0, nl = 0;
    static constexpr int ng_t = 0, ne_t = 0, nl_t = 0;
    static constexpr bool fixed_initial_state = false;
    static constexpr bool has_dynamics_hess_prod = false;
    static constexpr bool has_constr_hess_prod = false;
    static constexpr std::array<int, 1> state_box_idx = {0};
    static constexpr std::array<int, 0> control_box_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_idx = {};
    static constexpr std::array<int, 1> ineq_soft_idx = {0};
    static constexpr std::array<int, 0> eq_soft_idx = {};
    static constexpr std::array<int, 0> lin_soft_idx = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {};
    static constexpr std::array<int, 0> state_box_soft_idx = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

class MeritProb : public ocp::Problem<MeritDims>
{
public:
    state_t dynamics_next_state(int, const state_t& x, const control_t& u) const
    {
        state_t xn;
        xn(0) = 0.5 * x(0) + u(0);
        return xn;
    }

    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return 0.5 * x(0) * x(0) + 0.5 * u(0) * u(0);
    }

    double terminal_cost_value(const state_t& x) const
    {
        return 10.0 * x(0) * x(0);
    }

    ineq_t stage_inequality_constr(int, const state_t&,
                                   const control_t& u) const
    {
        ineq_t g;
        g(0) = u(0) - 5.0;
        return g;
    }

    ineq_pen_t stage_inequality_constr_soft_penalty(int) const
    {
        ineq_pen_t pen;
        pen(0) = 4.0;
        return pen;
    }

    state_box_t stage_state_box_constr(int) const
    {
        state_box_t spec;
        spec.lo(0) = -10.0;
        spec.hi(0) = 10.0;
        spec.soft_penalty.setZero();
        return spec;
    }

    control_box_t stage_control_box_constr(int) const
    {
        return control_box_t{};
    }

    term_state_box_t terminal_state_box_constr() const
    {
        return term_state_box_t{};
    }
};

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

template <class P, int NH>
void zero_solution(Solution<P, NH>& sol)
{
    const int N = sol.N;
    for (int k = 0; k < N; ++k)
    {
        sol.x[k].setZero();
        sol.u[k].setZero();
        sol.lambda_dyn[k].setZero();
        sol.lambda_ineq_stage[k].setZero();
        sol.lambda_eq_stage[k].setZero();
        sol.lambda_lin_stage[k].setZero();
        sol.lambda_box_state[k].setZero();
        sol.lambda_box_control[k].setZero();
    }
    sol.x[N].setZero();
    sol.lambda_box_state[N].setZero();
    sol.lambda_ineq_term.setZero();
    sol.lambda_eq_term.setZero();
    sol.lambda_lin_term.setZero();
}

template <class P, int NH>
void zero_qp_sol(QpSol<P, NH>& sol)
{
    const int N = sol.N;
    sol.ux_first.setZero();
    for (int k = 1; k < N; ++k)
    {
        sol.ux_path[k - 1].setZero();
    }
    sol.ux_term.setZero();
    for (int k = 0; k < N; ++k)
    {
        sol.pi[k].setZero();
    }
    sol.lam_first.setZero();
    for (int k = 1; k < N; ++k)
    {
        sol.lam_path[k - 1].setZero();
    }
    sol.lam_term.setZero();
    sol.t_first.setZero();
    for (int k = 1; k < N; ++k)
    {
        sol.t_path[k - 1].setZero();
    }
    sol.t_term.setZero();
}

// MeritProb N=1 QpSol: side layout [bx_lo, bx_hi, ineq_hi, slack]
QpSol<MeritProb> make_qp_sol_1(double pi0, double lam_bx_lo,
                                double lam_bx_hi, double lam_ineq_hi)
{
    constexpr int N = 1;
    QpSol<MeritProb> sol(N);
    zero_qp_sol<MeritProb, Eigen::Dynamic>(sol);
    sol.pi[0](0) = pi0;
    sol.lam_first(0) = lam_bx_lo;
    sol.lam_first(1) = lam_bx_hi;
    sol.lam_first(2) = lam_ineq_hi;
    return sol;
}

// ---------------------------------------------------------------------
// Merit value (N=1): hand-computed for three iterates that exercise
// different terms. Weights from QpSol: pi=3, lam=[1,2,4,0]
//   -> w_pi=3, w_box=[1,2], w_ineq=4
// ---------------------------------------------------------------------
void test_merit_value()
{
    const std::string p = "2d merit: ";
    constexpr int N = 1;
    MeritProb prob;
    MeritBacktracking<MeritProb> glob;

    auto qp = make_qp_sol_1(3.0, 1.0, 2.0, 4.0);

    // Case A: basic (no violations)
    {
        Solution<MeritProb> sol(N);
        zero_solution<MeritProb, Eigen::Dynamic>(sol);
        sol.x[0](0) = 1.0;
        sol.u[0](0) = 2.0;
        sol.x[1](0) = 1.5;
        SqpSlacks<MeritProb> sl(N);
    sl.setZero();
        sl.first(0) = 0.5;

        glob.initialize(prob, sol);
        glob.update_weights(qp);
        const double m = glob.merit(prob, sol, sl);
        // cost = 0.5+2+22.5 = 25; slack = 2*0.25 = 0.5
        // gap = |2.5-1.5| = 1, dyn = 3; ineq = 0; box = 0
        check_close(m, 28.5, p + "A basic");
    }

    // Case B: ineq violated (g=1, s=0.5 -> relaxed violation 0.5)
    {
        Solution<MeritProb> sol(N);
        zero_solution<MeritProb, Eigen::Dynamic>(sol);
        sol.x[0](0) = 0.0;
        sol.u[0](0) = 6.0;
        sol.x[1](0) = 3.0;
        SqpSlacks<MeritProb> sl(N);
    sl.setZero();
        sl.first(0) = 0.5;

        glob.initialize(prob, sol);
        glob.update_weights(qp);
        const double m = glob.merit(prob, sol, sl);
        // cost = 18+90 = 108; slack = 0.5; gap = 3, dyn = 9
        // ineq = 4*0.5 = 2; box = 0
        check_close(m, 119.5, p + "B ineq relaxed");
    }

    // Case C: box violation (x_0=11 > hi=10)
    {
        Solution<MeritProb> sol(N);
        zero_solution<MeritProb, Eigen::Dynamic>(sol);
        sol.x[0](0) = 11.0;
        sol.u[0](0) = 0.0;
        sol.x[1](0) = 0.0;
        SqpSlacks<MeritProb> sl(N);
    sl.setZero();
        sl.first(0) = 0.0;

        glob.initialize(prob, sol);
        glob.update_weights(qp);
        const double m = glob.merit(prob, sol, sl);
        // cost = 60.5; gap = 5.5, dyn = 16.5; box_hi = 2*1 = 2
        check_close(m, 79.0, p + "C box viol");
    }
}

// ---------------------------------------------------------------------
// Merit value (N=2): exercises the path stage.
// Weights: pi=[1,2], lam_first=[1,2,3,0], lam_path=[1,2,3,0]
//   -> w_pi=[1,2], w_box=[[1,2],[1,2]], w_ineq=[3,3]
// ---------------------------------------------------------------------
void test_merit_n2()
{
    const std::string p = "2d merit N=2: ";
    constexpr int N = 2;
    MeritProb prob;
    MeritBacktracking<MeritProb> glob;

    QpSol<MeritProb> qp(N);
    zero_qp_sol<MeritProb, Eigen::Dynamic>(qp);
    qp.pi[0](0) = 1.0;
    qp.pi[1](0) = 2.0;
    qp.lam_first << 1.0, 2.0, 3.0, 0.0;
    qp.lam_path[0] << 1.0, 2.0, 3.0, 0.0;

    Solution<MeritProb> sol(N);
    zero_solution<MeritProb, Eigen::Dynamic>(sol);
    sol.x[0](0) = 1.0;
    sol.u[0](0) = 1.0;
    sol.x[1](0) = 1.0;
    sol.u[1](0) = 2.0;
    sol.x[2](0) = 1.5;
    SqpSlacks<MeritProb> sl(N);
    sl.setZero();
    sl.first(0) = 0.5;
    sl.path[0](0) = 1.0;

    glob.initialize(prob, sol);
    glob.update_weights(qp);
    const double m = glob.merit(prob, sol, sl);
    // cost = 1 + 2.5 + 22.5 = 26; slack = 0.5+2 = 2.5
    // gap_0 = 0.5, dyn_0 = 0.5; gap_1 = 1, dyn_1 = 2
    check_close(m, 31.0, p + "path stage");
}

// ---------------------------------------------------------------------
// Weight update: first call seeds w = |dual|; second call applies
// w = max(|dual|, 0.5(|dual| + w_old)).
//
// First:  pi=3, lam=[1,2,4,0] -> w_pi=3, w_box=[1,2], w_ineq=4
// Second: pi=1, lam=[5,0.5,2,0]
//   -> w_pi = max(1, 0.5*(1+3)) = 2
//   -> w_box = [max(5,3), max(0.5,1.25)] = [5, 1.25]
//   -> w_ineq = max(2, 3) = 3
//
// Verify with a single merit evaluation that exercises all three.
// x_0=11, u_0=6, x_1=1, s_0=0.5:
//   cost = 60.5+18+10 = 88.5; slack = 0.5
//   gap = 10.5, dyn = 2*10.5 = 21
//   ineq = 3*0.5 = 1.5; box_hi = 1.25*1 = 1.25
//   merit = 112.75
// ---------------------------------------------------------------------
void test_weight_update()
{
    const std::string p = "2d weights: ";
    constexpr int N = 1;
    MeritProb prob;
    MeritBacktracking<MeritProb> glob;

    auto qp1 = make_qp_sol_1(3.0, 1.0, 2.0, 4.0);
    auto qp2 = make_qp_sol_1(1.0, 5.0, 0.5, 2.0);

    Solution<MeritProb> sol(N);
    zero_solution<MeritProb, Eigen::Dynamic>(sol);
    sol.x[0](0) = 11.0;
    sol.u[0](0) = 6.0;
    sol.x[1](0) = 1.0;
    SqpSlacks<MeritProb> sl(N);
    sl.setZero();
    sl.first(0) = 0.5;

    glob.initialize(prob, sol);
    glob.update_weights(qp1);
    glob.update_weights(qp2);
    const double m = glob.merit(prob, sol, sl);
    check_close(m, 112.75, p + "updated");
}

// ---------------------------------------------------------------------
// Full-step accept: merit(1) < merit(0) -> kSolved, alpha=1.
// Start: x_0=2, u_0=1, x_1=5, s_0=0 (large dynamics gap)
// Step:  du=-1, dx_0=0, ds=0, dx_1=-3
//   merit_0 = 252.5+9 = 261.5
//   merit(1) = 42+3 = 45 < 261.5 -> accept
// ---------------------------------------------------------------------
void test_full_step()
{
    const std::string p = "2d full-step: ";
    constexpr int N = 1;
    MeritProb prob;
    MeritBacktracking<MeritProb> glob;

    Solution<MeritProb> cur(N);
    zero_solution<MeritProb, Eigen::Dynamic>(cur);
    cur.x[0](0) = 2.0;
    cur.u[0](0) = 1.0;
    cur.x[1](0) = 5.0;

    QpSol<MeritProb> step(N);
    zero_qp_sol<MeritProb, Eigen::Dynamic>(step);
    step.ux_first(0) = -1.0;  // du_0
    step.ux_term(0) = -3.0;   // dx_1
    step.pi[0](0) = 3.0;
    step.lam_first(0) = 1.0;
    step.lam_first(1) = 2.0;
    step.lam_first(2) = 4.0;

    SqpSlacks<MeritProb> sl(N);
    sl.setZero();
    Solution<MeritProb> scratch(N);
    zero_solution<MeritProb, Eigen::Dynamic>(scratch);

    double alpha = 0.0;
    glob.initialize(prob, cur);
    const Status st = glob.find_acceptable_iterate(
        prob, cur, step, sl, scratch, alpha);

    check(st == Status::kSolved, p + "status");
    check_close(alpha, 1.0, p + "alpha");
    check_close(cur.x[0](0), 2.0, p + "x0");
    check_close(cur.u[0](0), 0.0, p + "u0");
    check_close(cur.x[1](0), 2.0, p + "x1");
    // alpha = 1: relaxed == full == mapped (start duals are zero)
    check_close(cur.lambda_dyn[0](0), -3.0, p + "dyn");
    check_close(cur.lambda_box_state[0](0), 1.0, p + "box lo");
    check_close(cur.lambda_box_state[0](1), 2.0, p + "box hi");
    check_close(cur.lambda_ineq_stage[0](0), 4.0, p + "ineq");
}

// ---------------------------------------------------------------------
// Backtracking: alpha=1 and alpha=0.7 rejected, alpha=0.49 accepted.
// The QP slack variable is the absolute new slack (2f convention), so
// the trial slack is s(a) = 1.5 + a * (-4 - 1.5) = 1.5 - 5.5a and
// merit(a) = 15 - 33a + 60.5a^2 (zero primal step: cost and dyn gap
// are constant).
//   merit(1) = 42.5 > 15      -> reject
//   merit(0.7) = 21.545 > 15  -> reject
//   merit(0.49) = 13.356 < 15 -> accept
//
// Start duals: dyn=0.9, box=[0.6,0.4], ineq=0.8
// Mapped step duals: dyn=-0.5, box=[1,2], ineq=3
// Relaxed (0.3*start + 0.7*mapped):
//   dyn = -0.08, box = [0.88, 1.52], ineq = 2.34
// ---------------------------------------------------------------------
void test_backtracking()
{
    const std::string p = "2d backtrack: ";
    constexpr int N = 1;
    MeritProb prob;
    MeritBacktracking<MeritProb> glob;

    Solution<MeritProb> cur(N);
    zero_solution<MeritProb, Eigen::Dynamic>(cur);
    cur.x[0](0) = 0.0;
    cur.u[0](0) = 0.0;
    cur.x[1](0) = 1.0;
    cur.lambda_dyn[0](0) = 0.9;
    cur.lambda_box_state[0](0) = 0.6;
    cur.lambda_box_state[0](1) = 0.4;
    cur.lambda_ineq_stage[0](0) = 0.8;

    QpSol<MeritProb> step(N);
    zero_qp_sol<MeritProb, Eigen::Dynamic>(step);
    step.ux_first(2) = -4.0;  // ds_0
    step.pi[0](0) = 0.5;
    step.lam_first(0) = 1.0;
    step.lam_first(1) = 2.0;
    step.lam_first(2) = 3.0;

    SqpSlacks<MeritProb> sl(N);
    sl.setZero();
    sl.first(0) = 1.5;
    Solution<MeritProb> scratch(N);
    zero_solution<MeritProb, Eigen::Dynamic>(scratch);

    double alpha = 0.0;
    glob.initialize(prob, cur);
    const Status st = glob.find_acceptable_iterate(
        prob, cur, step, sl, scratch, alpha);

    check(st == Status::kSolved, p + "status");
    check_close(alpha, 0.49, p + "alpha");
    // primal unchanged (du=0, dx=0)
    check_close(cur.x[0](0), 0.0, p + "x0");
    check_close(cur.u[0](0), 0.0, p + "u0");
    check_close(cur.x[1](0), 1.0, p + "x1");
    // relaxed duals (0.51 * start + 0.49 * mapped)
    check_close(cur.lambda_dyn[0](0), 0.214, p + "dyn");
    check_close(cur.lambda_box_state[0](0), 0.796, p + "box lo");
    check_close(cur.lambda_box_state[0](1), 1.184, p + "box hi");
    check_close(cur.lambda_ineq_stage[0](0), 1.878, p + "ineq");
}

// ---------------------------------------------------------------------
// kMinStep: zero QP duals -> all merit weights zero -> merit = cost +
// slack_pen only. The step increases the cost, so every trial is
// rejected. Final alpha = 0.7^8 (8 trials exhausted the range).
// cur is still advanced by the post-shrink step.
// ---------------------------------------------------------------------
void test_min_step()
{
    const std::string p = "2d min-step: ";
    constexpr int N = 1;
    MeritProb prob;
    MeritBacktracking<MeritProb> glob;

    auto qp = make_qp_sol_1(0.0, 0.0, 0.0, 0.0);

    Solution<MeritProb> cur(N);
    zero_solution<MeritProb, Eigen::Dynamic>(cur);

    QpSol<MeritProb> step(N);
    zero_qp_sol<MeritProb, Eigen::Dynamic>(step);
    step.ux_first(0) = 1.0;  // du_0 = 1

    SqpSlacks<MeritProb> sl(N);
    sl.setZero();
    Solution<MeritProb> scratch(N);
    zero_solution<MeritProb, Eigen::Dynamic>(scratch);

    double alpha = 0.0;
    glob.initialize(prob, cur);
    const Status st = glob.find_acceptable_iterate(
        prob, cur, step, sl, scratch, alpha);

    check(st == Status::kMinStep, p + "status");
    check_close(alpha, std::pow(0.7, 8), p + "alpha");
    check_close(cur.u[0](0), std::pow(0.7, 8), p + "u0 advanced");
    check_close(cur.x[0](0), 0.0, p + "x0 unchanged");
    check_close(cur.x[1](0), 0.0, p + "x1 unchanged");
}

// ---------------------------------------------------------------------
// kNanDetected: the step puts x_1 = NaN -> merit is NaN for every
// trial. cur is left untouched.
// ---------------------------------------------------------------------
void test_nan_detected()
{
    const std::string p = "2d nan: ";
    constexpr int N = 1;
    MeritProb prob;
    MeritBacktracking<MeritProb> glob;

    auto qp = make_qp_sol_1(1.0, 1.0, 1.0, 1.0);

    Solution<MeritProb> cur(N);
    zero_solution<MeritProb, Eigen::Dynamic>(cur);
    cur.x[0](0) = 0.0;
    cur.u[0](0) = 0.0;
    cur.x[1](0) = 0.0;

    QpSol<MeritProb> step(N);
    zero_qp_sol<MeritProb, Eigen::Dynamic>(step);
    step.ux_term(0) = std::numeric_limits<double>::quiet_NaN();

    SqpSlacks<MeritProb> sl(N);
    sl.setZero();
    Solution<MeritProb> scratch(N);
    zero_solution<MeritProb, Eigen::Dynamic>(scratch);

    double alpha = 0.0;
    glob.initialize(prob, cur);
    const Status st = glob.find_acceptable_iterate(
        prob, cur, step, sl, scratch, alpha);

    check(st == Status::kNanDetected, p + "status");
    check_close(cur.x[1](0), 0.0, p + "x1 untouched");
    check_close(cur.u[0](0), 0.0, p + "u0 untouched");
}

// ---------------------------------------------------------------------
// detail::shift_slacks: trial slacks = s + alpha * (s_qp - s)
// (the QP slack variable is the absolute new slack, 2f convention)
// ---------------------------------------------------------------------
void test_shift_slacks()
{
    const std::string p = "2d shift: ";
    constexpr int N = 2;

    SqpSlacks<MeritProb> s(N);
    s.setZero();
    s.first(0) = 1.0;
    s.path[0](0) = 3.0;

    QpSol<MeritProb> step(N);
    zero_qp_sol<MeritProb, Eigen::Dynamic>(step);
    step.ux_first(2) = 2.0;      // s_qp_0 = 2 (absolute)
    step.ux_path[0](2) = 1.0;    // s_qp_1 = 1 (absolute)

    SqpSlacks<MeritProb> dest(N);
    detail::shift_slacks<MeritProb, Eigen::Dynamic>(s, step, 0.5, dest);
    check_close(dest.first(0), 1.0 + 0.5 * (2.0 - 1.0), p + "first");
    check_close(dest.path[0](0), 3.0 + 0.5 * (1.0 - 3.0), p + "path");
}

}  // namespace

int run_merit_2d_tests()
{
    failures = 0;
    test_merit_value();
    test_merit_n2();
    test_weight_update();
    test_full_step();
    test_backtracking();
    test_min_step();
    test_nan_detected();
    test_shift_slacks();

    if (failures == 0)
    {
        std::printf("All merit backtracking (2d) checks passed.\n");
    }
    return failures;
}

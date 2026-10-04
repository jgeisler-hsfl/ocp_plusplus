// Phase 2 sub-step 2b test: the SQP globalization primitives
// (GlobOptions, SqpSlacks, detail::map_qp_duals_to_solution, apply_sqp_step)
// from include/ocp/solvers/acados/globalize.hpp.
//
// Checks (SQP_PHASE2_PLAN.md sec. 2b, acados ocp_nlp_common.c:3355-3414
// ocp_nlp_update_variables_sqp, ocp_nlp_globalization_common.h:82-90):
//   - GlobOptions defaults match the acados defaults
//   - SqpSlacks: sizing (dynamic + fixed horizon), resize, setZero
//   - detail::map_qp_duals_to_solution: every sec. 1.3 mapping row
//     (dynamics sign flip, ineq hi, eq/lin net, box lower/upper, terminal)
//   - apply_sqp_step: primal interpolation of u and x, relaxed dual update
//     (1-alpha)*start + alpha*mapped, and full-step dual update
//   - fixed x_0: the pinned initial state is left untouched

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/solvers/acados/globalize.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

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
// A problem type with every constraint group active (no soft rows), so the
// dual mapping can be exercised on all rows. fixed_initial_state = false so
// the first stage keeps its state-box rows (the fixed-x_0 case is covered
// separately with DoubleIntegrator).
// ---------------------------------------------------------------------
struct MapDims
{
    static constexpr int nx = 2, nu = 1;
    static constexpr int ng = 1, ne = 1, nl = 1;
    static constexpr int ng_t = 1, ne_t = 1, nl_t = 1;
    static constexpr bool fixed_initial_state = false;
    static constexpr bool has_dynamics_hess = false;
    static constexpr bool has_constr_hess = false;
    static constexpr std::array<int, 2> state_box_idx = {0, 1};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 2> terminal_state_box_idx = {0, 1};
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

struct MapProb : Problem<MapDims>
{
};

// Zero every field of a Solution (guards against uninitialized reads on
// dynamic-extent trajectories after resize).
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

// Zero every field of a QpSol.
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

// ---------------------------------------------------------------------
// GlobOptions defaults (ocp_nlp_globalization_common.c:83-86).
// ---------------------------------------------------------------------
void test_glob_options()
{
    const std::string p = "2b options: ";
    GlobOptions o;
    check(o.alpha_min == 0.05, p + "alpha_min = 0.05");
    check(o.alpha_reduction == 0.7, p + "alpha_reduction = 0.7");
    check(o.eps_sufficient_descent == 1e-4, p + "eps_sufficient_descent");
    check(o.full_step_dual == false, p + "full_step_dual = false");
}

// ---------------------------------------------------------------------
// SqpSlacks: sizing and setZero.
// ---------------------------------------------------------------------
void test_slacks()
{
    const std::string p = "2b slacks: ";

    // MapProb has zero slacks in every stage type.
    {
        SqpSlacks<MapProb> sl(2);
        check(sl.N == 2, p + "MapProb N = 2");
        check(static_cast<int>(sl.first.size()) == 0,
              p + "MapProb first size 0");
        check(sl.path.size() == 1, p + "MapProb path has N-1 = 1 slot");
        check(static_cast<int>(sl.term.size()) == 0,
              p + "MapProb term size 0");
        sl.setZero();  // must not crash on zero-extent vectors
        check(true, p + "MapProb setZero no crash");
    }

    // DoubleIntegrator: nslack_first = nslack_path = 1 (soft ineq),
    // nslack_term = 0.
    {
        SqpSlacks<DoubleIntegrator> sl(3);
        check(sl.N == 3, p + "DI N = 3");
        check(static_cast<int>(sl.first.size()) == 1,
              p + "DI first size 1");
        check(sl.path.size() == 2, p + "DI path has N-1 = 2 slots");
        check(static_cast<int>(sl.term.size()) == 0,
              p + "DI term size 0");
        sl.first(0) = 1.0;
        sl.path[0](0) = 2.0;
        sl.path[1](0) = 3.0;
        sl.setZero();
        check(sl.first(0) == 0.0, p + "DI setZero first");
        check(sl.path[0](0) == 0.0, p + "DI setZero path 0");
        check(sl.path[1](0) == 0.0, p + "DI setZero path 1");
    }

    // Fixed horizon: consistency check on resize, no allocation.
    {
        SqpSlacks<DoubleIntegrator, 2> sl;
        check(sl.N == 2, p + "DI NH=2 default N");
        sl.resize(2);  // consistent
        check(sl.path.size() == 1, p + "DI NH=2 path extent 1");
    }
}

// ---------------------------------------------------------------------
// detail::map_qp_duals_to_solution: hand-computed every-row mapping.
//
// MapProb layout (no soft rows, not fixed):
//   first/path rows [bx(0), bx(1), bu(2), ineq(3), eq(4), lin(5)]
//   first/path sides: lo [bx0=0, bx1=1, bu=2, eq=3, lin=4]
//                     hi [bx0=5, bx1=6, bu=7, ineq=8, eq=9, lin=10]
//   term rows [bx(0), bx(1), ineq(2), eq(3), lin(4)]
//   term sides:    lo [bx0=0, bx1=1, eq=2, lin=3]
//                  hi [bx0=4, bx1=5, ineq=6, eq=7, lin=8]
// ---------------------------------------------------------------------
template <int NH>
void test_dual_mapping(const char* name)
{
    using TP = MapProb;
    using D = QpDim<TP>;
    const std::string p = std::string("2b mapping ") + name + ": ";
    constexpr int N = 2;

    static_assert(D::nside_first == 11, "first-side count");
    static_assert(D::nside_path == 11, "path-side count");
    static_assert(D::nside_term == 9, "term-side count");

    QpSol<TP, NH> sol(N);
    zero_qp_sol(sol);
    sol.pi[0] << 1.0, 2.0;
    sol.pi[1] << 3.0, 4.0;
    sol.lam_first << 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20;
    sol.lam_path[0] << 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31;
    sol.lam_term << 40, 41, 42, 43, 44, 45, 46, 47, 48;

    Solution<TP, NH> dest(N);
    zero_solution(dest);
    detail::map_qp_duals_to_solution<TP, NH>(sol, dest);

    // dynamics: lambda_dyn[k] = -pi[k] (sec. 1.3 sign flip)
    check_close(dest.lambda_dyn[0](0), -1.0, p + "dyn 0");
    check_close(dest.lambda_dyn[0](1), -2.0, p + "dyn 1");
    check_close(dest.lambda_dyn[1](0), -3.0, p + "dyn 2");
    check_close(dest.lambda_dyn[1](1), -4.0, p + "dyn 3");

    // first stage (k = 0)
    check_close(dest.lambda_ineq_stage[0](0), 18.0, p + "ineq 0 (hi)");
    check_close(dest.lambda_eq_stage[0](0), 19.0 - 13.0, p + "eq 0 (net)");
    check_close(dest.lambda_lin_stage[0](0), 20.0 - 14.0, p + "lin 0 (net)");
    check_close(dest.lambda_box_state[0](0), 10.0, p + "boxx 0 lower");
    check_close(dest.lambda_box_state[0](1), 11.0, p + "boxx 0 upper-lower");
    check_close(dest.lambda_box_state[0](2), 15.0, p + "boxx 0 upper-lower2");
    check_close(dest.lambda_box_state[0](3), 16.0, p + "boxx 0 upper-upper");
    check_close(dest.lambda_box_control[0](0), 12.0, p + "boxu 0 lower");
    check_close(dest.lambda_box_control[0](1), 17.0, p + "boxu 0 upper");

    // path stage (k = 1)
    check_close(dest.lambda_ineq_stage[1](0), 29.0, p + "ineq 1 (hi)");
    check_close(dest.lambda_eq_stage[1](0), 30.0 - 24.0, p + "eq 1 (net)");
    check_close(dest.lambda_lin_stage[1](0), 31.0 - 25.0, p + "lin 1 (net)");
    check_close(dest.lambda_box_state[1](0), 21.0, p + "boxx 1 lower");
    check_close(dest.lambda_box_state[1](1), 22.0, p + "boxx 1 lower2");
    check_close(dest.lambda_box_state[1](2), 26.0, p + "boxx 1 upper");
    check_close(dest.lambda_box_state[1](3), 27.0, p + "boxx 1 upper2");
    check_close(dest.lambda_box_control[1](0), 23.0, p + "boxu 1 lower");
    check_close(dest.lambda_box_control[1](1), 28.0, p + "boxu 1 upper");

    // terminal stage (k = N)
    check_close(dest.lambda_ineq_term(0), 46.0, p + "term ineq (hi)");
    check_close(dest.lambda_eq_term(0), 47.0 - 42.0, p + "term eq (net)");
    check_close(dest.lambda_lin_term(0), 48.0 - 43.0, p + "term lin (net)");
    check_close(dest.lambda_box_state[N](0), 40.0, p + "term boxx lower");
    check_close(dest.lambda_box_state[N](1), 41.0, p + "term boxx lower2");
    check_close(dest.lambda_box_state[N](2), 44.0, p + "term boxx upper");
    check_close(dest.lambda_box_state[N](3), 45.0, p + "term boxx upper2");
}

// ---------------------------------------------------------------------
// apply_sqp_step: primal interpolation + dual updates.
// ---------------------------------------------------------------------
void test_apply_step()
{
    using TP = MapProb;
    const std::string p = "2b step: ";
    constexpr int N = 2;
    const double alpha = 0.5;

    // --- step (QP solution) ---
    QpSol<TP> step(N);
    zero_qp_sol(step);
    step.ux_first << 10.0, 20.0, 30.0;  // (u0; x0; s)
    step.ux_path[0] << 11.0, 21.0, 31.0;  // (u1; x1; s)
    step.ux_term << 12.0, 22.0;  // (xN; s)
    step.pi[0] << 1.0, 2.0;
    step.pi[1] << 3.0, 4.0;
    step.lam_first << 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20;
    step.lam_path[0] << 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31;
    step.lam_term << 40, 41, 42, 43, 44, 45, 46, 47, 48;

    // --- start iterate ---
    Solution<TP> start(N);
    zero_solution(start);
    start.x[0] << 1.0, 1.0;
    start.x[1] << 2.0, 2.0;
    start.x[N] << 3.0, 3.0;
    start.u[0](0) = 5.0;
    start.u[1](0) = 6.0;
    start.lambda_dyn[0] << 0.1, 0.2;
    start.lambda_dyn[1] << 0.3, 0.4;
    start.lambda_ineq_stage[0](0) = 1.0;
    start.lambda_ineq_stage[1](0) = 2.0;
    start.lambda_eq_stage[0](0) = 3.0;
    start.lambda_eq_stage[1](0) = 4.0;
    start.lambda_lin_stage[0](0) = 5.0;
    start.lambda_lin_stage[1](0) = 6.0;
    start.lambda_box_state[0] << 0.1, 0.2, 0.3, 0.4;
    start.lambda_box_state[1] << 0.1, 0.2, 0.3, 0.4;
    start.lambda_box_state[N] << 0.5, 0.6, 0.7, 0.8;
    start.lambda_box_control[0] << 0.1, 0.2;
    start.lambda_box_control[1] << 0.3, 0.4;
    start.lambda_ineq_term(0) = 10.0;
    start.lambda_eq_term(0) = 11.0;
    start.lambda_lin_term(0) = 12.0;

    // --- expected dual mapping (from the step) ---
    Solution<TP> mapped(N);
    zero_solution(mapped);
    detail::map_qp_duals_to_solution<TP>(step, mapped);

    // --- relaxed dual update (default) ---
    {
        Solution<TP> dest(N);
        zero_solution(dest);
        apply_sqp_step<TP>(start, step, alpha, /*full_step_dual=*/false, dest);

        // primal interpolation
        check_close(dest.u[0](0), 5.0 + alpha * 10.0, p + "u0");
        check_close(dest.u[1](0), 6.0 + alpha * 11.0, p + "u1");
        check_close(dest.x[0](0), 1.0 + alpha * 20.0, p + "x0 q");
        check_close(dest.x[0](1), 1.0 + alpha * 30.0, p + "x0 v");
        check_close(dest.x[1](0), 2.0 + alpha * 21.0, p + "x1 q");
        check_close(dest.x[1](1), 2.0 + alpha * 31.0, p + "x1 v");
        check_close(dest.x[N](0), 3.0 + alpha * 12.0, p + "xN q");
        check_close(dest.x[N](1), 3.0 + alpha * 22.0, p + "xN v");

        const double w = 1.0 - alpha;
        // dynamics (sign flip carried through the mapping)
        check_close(dest.lambda_dyn[0](0),
                    w * 0.1 + alpha * mapped.lambda_dyn[0](0), p + "dyn 0");
        check_close(dest.lambda_dyn[1](1),
                    w * 0.4 + alpha * mapped.lambda_dyn[1](1), p + "dyn 3");
        // ineq / eq / lin
        check_close(dest.lambda_ineq_stage[0](0),
                    w * 1.0 + alpha * 18.0, p + "ineq 0");
        check_close(dest.lambda_ineq_stage[1](0),
                    w * 2.0 + alpha * 29.0, p + "ineq 1");
        check_close(dest.lambda_eq_stage[0](0),
                    w * 3.0 + alpha * 6.0, p + "eq 0");
        check_close(dest.lambda_eq_stage[1](0),
                    w * 4.0 + alpha * 6.0, p + "eq 1");
        check_close(dest.lambda_lin_stage[0](0),
                    w * 5.0 + alpha * 6.0, p + "lin 0");
        check_close(dest.lambda_lin_stage[1](0),
                    w * 6.0 + alpha * 6.0, p + "lin 1");
        // box state / control
        check_close(dest.lambda_box_state[0](0),
                    w * 0.1 + alpha * 10.0, p + "boxx 0 lower");
        check_close(dest.lambda_box_state[0](3),
                    w * 0.4 + alpha * 16.0, p + "boxx 0 upper");
        check_close(dest.lambda_box_state[1](2),
                    w * 0.3 + alpha * 26.0, p + "boxx 1 upper");
        check_close(dest.lambda_box_state[N](0),
                    w * 0.5 + alpha * 40.0, p + "term boxx lower");
        check_close(dest.lambda_box_state[N](3),
                    w * 0.8 + alpha * 45.0, p + "term boxx upper");
        check_close(dest.lambda_box_control[0](0),
                    w * 0.1 + alpha * 12.0, p + "boxu 0 lower");
        check_close(dest.lambda_box_control[0](1),
                    w * 0.2 + alpha * 17.0, p + "boxu 0 upper");
        check_close(dest.lambda_box_control[1](1),
                    w * 0.4 + alpha * 28.0, p + "boxu 1 upper");
        // terminal
        check_close(dest.lambda_ineq_term(0),
                    w * 10.0 + alpha * 46.0, p + "term ineq");
        check_close(dest.lambda_eq_term(0),
                    w * 11.0 + alpha * 5.0, p + "term eq");
        check_close(dest.lambda_lin_term(0),
                    w * 12.0 + alpha * 5.0, p + "term lin");
    }

    // --- full-step dual update ---
    {
        Solution<TP> dest(N);
        zero_solution(dest);
        apply_sqp_step<TP>(start, step, alpha, /*full_step_dual=*/true, dest);

        // primal is identical regardless of the dual scheme
        check_close(dest.u[0](0), 5.0 + alpha * 10.0, p + "full u0");
        check_close(dest.x[1](1), 2.0 + alpha * 31.0, p + "full x1 v");

        // duals are exactly the mapped step (not blended)
        for (int k = 0; k < N; ++k)
        {
            check_close(dest.lambda_dyn[k](0), mapped.lambda_dyn[k](0),
                        p + "full dyn " + std::to_string(k));
            check_close(dest.lambda_dyn[k](1), mapped.lambda_dyn[k](1),
                        p + "full dyn " + std::to_string(k) + "b");
            check_close(dest.lambda_ineq_stage[k](0),
                        mapped.lambda_ineq_stage[k](0), p + "full ineq");
            check_close(dest.lambda_eq_stage[k](0),
                        mapped.lambda_eq_stage[k](0), p + "full eq");
            check_close(dest.lambda_lin_stage[k](0),
                        mapped.lambda_lin_stage[k](0), p + "full lin");
            for (int j = 0; j < 4; ++j)
            {
                check_close(dest.lambda_box_state[k](j),
                            mapped.lambda_box_state[k](j), p + "full boxx");
            }
            check_close(dest.lambda_box_control[k](0),
                        mapped.lambda_box_control[k](0), p + "full boxu lo");
            check_close(dest.lambda_box_control[k](1),
                        mapped.lambda_box_control[k](1), p + "full boxu hi");
        }
        for (int j = 0; j < 4; ++j)
        {
            check_close(dest.lambda_box_state[N](j),
                        mapped.lambda_box_state[N](j), p + "full term boxx");
        }
        check_close(dest.lambda_ineq_term(0), mapped.lambda_ineq_term(0),
                    p + "full term ineq");
        check_close(dest.lambda_eq_term(0), mapped.lambda_eq_term(0),
                    p + "full term eq");
        check_close(dest.lambda_lin_term(0), mapped.lambda_lin_term(0),
                    p + "full term lin");
    }
}

// ---------------------------------------------------------------------
// Fixed x_0 (DoubleIntegrator): with the pin rows the x_0 step is zero, so
// the update leaves x[0] exactly at the start value.
// ---------------------------------------------------------------------
void test_fixed_x0()
{
    using TP = DoubleIntegrator;
    const std::string p = "2b fixed-x0: ";
    constexpr int N = 1;
    const double alpha = 0.5;

    QpSol<TP> step(N);
    zero_qp_sol(step);
    // x_0 step part (cols 1..2 of ux_first) is zero, as the pin rows force.
    step.ux_first << 3.0, 0.0, 0.0, 0.0;  // (u0; x0; s)
    step.ux_term << 4.0, 6.0;  // (xN; s)

    Solution<TP> start(N);
    zero_solution(start);
    start.x[0] << 1.0, 0.5;  // initial_state
    start.x[N] << 2.0, 3.0;
    start.u[0](0) = 5.0;

    Solution<TP> dest(N);
    zero_solution(dest);
    apply_sqp_step<TP>(start, step, alpha, false, dest);

    check_close(dest.x[0](0), 1.0, p + "x0 q preserved");
    check_close(dest.x[0](1), 0.5, p + "x0 v preserved");
    check_close(dest.u[0](0), 5.0 + alpha * 3.0, p + "u0 updated");
    check_close(dest.x[N](0), 2.0 + alpha * 4.0, p + "xN q updated");
    check_close(dest.x[N](1), 3.0 + alpha * 6.0, p + "xN v updated");
}

}  // namespace

int run_apply_step_2b_tests()
{
    failures = 0;
    test_glob_options();
    test_slacks();
    test_dual_mapping<Eigen::Dynamic>("dyn");
    test_dual_mapping<2>("fixed-NH");
    test_apply_step();
    test_fixed_x0();

    if (failures == 0)
    {
        std::printf("All apply-step (2b) checks passed.\n");
    }
    return failures;
}

// Phase 2 sub-step 2f test: QP assembly
// (SqpSolver::assemble_qp, SqpSolver::add_lm_term) from
// include/ocp/solvers/acados/sqp.hpp.
//
// Checks (SQP_PHASE2_PLAN.md sec. 2f, acados ocp_nlp_common.c:3079-3203,
// :3034-3059):
//   - DoubleIntegrator (N = 2, fixed x_0, zero HVPs): every stage field
//     (hess, grad, BA, b, DC, d, d_mask) against hand-computed values,
//     incl. pin rows, soft ineq slack column, offset-form d.
//   - QuadTest (N = 1, x_0 free, non-zero HVPs): Hessian composition
//     H_cost - M_dyn + M_ineq, terminal cost Hessian + terminal eq HVP,
//     soft-terminal-eq slack diagonal.
//   - DegProbe (fixed x_0, hard ineq violated at the pinned state):
//     assemble_qp returns kInfeasible.
//   - add_lm_term: mu*I on the (u;x) diagonal of every stage, no-op when
//     compute_hess = false or mu <= 0.

#include <cmath>
#include <cstdio>

#include "ocp/solvers/acados/sqp.hpp"

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

// ---------------------------------------------------------------------
// QuadTest: nx = 1, nu = 1, x_0 free, one soft ineq (stage), one soft
// terminal equality; non-zero dynamics / ineq / terminal-eq HVPs.
//
//   x scalar p, u scalar a
//   dynamics:  p_{k+1} = p + Ts p a        (Ts = 0.2)
//   stage cost: p^2 + 0.5 p a + 0.25 a^2
//   terminal:   4 p^2
//   ineq:       p a + p - 0.5 <= 0        (soft, penalty 7)
//   term eq:    p^2 - 1 = 0                (soft, penalty 3)
//   box:        -5 <= p <= 5, -2 <= a <= 2 (hard)
//
// Iterate (N = 1): x_0 = 0.3, u_0 = 0.8, x_1 = 0.5
//   lambda_dyn[0] = 1.5, lambda_ineq[0] = 0.4, lambda_eq_term = -2.5
// ---------------------------------------------------------------------

struct QuadDims
{
    static constexpr int nx = 1;
    static constexpr int nu = 1;
    static constexpr int ng = 1;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 1;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = false;
    static constexpr bool has_dynamics_hess_prod = true;
    static constexpr bool has_constr_hess_prod = true;
    static constexpr std::array<int, 1> state_box_idx = {0};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 0> terminal_state_box_idx = {};
    static constexpr std::array<int, 1> ineq_soft_idx = {0};
    static constexpr std::array<int, 0> eq_soft_idx = {};
    static constexpr std::array<int, 0> lin_soft_idx = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 1> terminal_eq_soft_idx = {0};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {};
    static constexpr std::array<int, 0> state_box_soft_idx = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

class QuadTest : public ocp::Problem<QuadDims>
{
public:
    static constexpr double Ts = 0.2;

    state_t dynamics_next_state(int, const state_t& x, const control_t& u) const
    {
        return state_t::Constant(1, x(0) + Ts * x(0) * u(0));
    }

    void dynamics_jacobian(int, const state_t& x, const control_t& u,
                           dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
    {
        df_dx(0, 0) = 1.0 + Ts * u(0);
        df_du(0, 0) = Ts * x(0);
    }

    void dynamics_hess_prod(int, const state_t&, const control_t&,
                            const state_t& w, const state_t& v_x,
                            const control_t& v_u, state_t& hv_x,
                            control_t& hv_u) const
    {
        // d^2/dp da of (p + Ts p a) is Ts; HVP = w * Ts * (v_u; v_x)
        hv_x(0) = w(0) * Ts * v_u(0);
        hv_u(0) = w(0) * Ts * v_x(0);
    }

    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return x(0) * x(0) + 0.5 * x(0) * u(0) + 0.25 * u(0) * u(0);
    }

    stage_grad_t stage_cost_gradient(int, const state_t& x,
                                     const control_t& u) const
    {
        stage_grad_t g;
        g(0) = 2.0 * x(0) + 0.5 * u(0);
        g(1) = 0.5 * x(0) + 0.5 * u(0);
        return g;
    }

    stage_hess_t stage_cost_hessian(int, const state_t&, const control_t&) const
    {
        stage_hess_t H;
        H << 2.0, 0.5, 0.5, 0.5;  // [x; u] layout
        return H;
    }

    double terminal_cost_value(const state_t& x) const
    {
        return 4.0 * x(0) * x(0);
    }

    term_grad_t terminal_cost_gradient(const state_t& x) const
    {
        return term_grad_t::Constant(1, 8.0 * x(0));
    }

    term_hess_t terminal_cost_hessian(const state_t&) const
    {
        return term_hess_t::Constant(1, 1, 8.0);
    }

    ineq_t stage_inequality_constr(int, const state_t& x, const control_t& u) const
    {
        return ineq_t::Constant(1, x(0) * u(0) + x(0) - 0.5);
    }

    void stage_inequality_constr_jacobian(int, const state_t& x,
                                          const control_t& u,
                                          ineq_dg_dx_t& g_dx,
                                          ineq_dg_du_t& g_du) const
    {
        g_dx(0, 0) = u(0) + 1.0;
        g_du(0, 0) = x(0);
    }

    void stage_inequality_constr_hess_prod(int, const state_t&,
                                           const control_t&, const ineq_t& w,
                                           const state_t& v_x,
                                           const control_t& v_u,
                                           state_t& hv_x,
                                           control_t& hv_u) const
    {
        // mixed Hessian of p a + p is the off-diagonal 1
        hv_x(0) = w(0) * v_u(0);
        hv_u(0) = w(0) * v_x(0);
    }

    ineq_pen_t stage_inequality_constr_soft_penalty(int) const
    {
        return ineq_pen_t::Constant(1, 7.0);
    }

    state_box_t stage_state_box_constr(int) const
    {
        state_box_t spec;
        spec.lo(0) = -5.0;
        spec.hi(0) = 5.0;
        spec.soft_penalty(0) = 0.0;
        return spec;
    }

    control_box_t stage_control_box_constr(int) const
    {
        control_box_t spec;
        spec.lo(0) = -2.0;
        spec.hi(0) = 2.0;
        spec.soft_penalty(0) = 0.0;
        return spec;
    }

    eq_term_t terminal_equality_constr(const state_t& x) const
    {
        return eq_term_t::Constant(1, x(0) * x(0) - 1.0);
    }

    void terminal_equality_constr_jacobian(const state_t& x,
                                           eq_term_de_dx_t& e_dx) const
    {
        e_dx(0, 0) = 2.0 * x(0);
    }

    void terminal_equality_constr_hess_prod(const state_t&,
                                            const eq_term_t& w,
                                            const state_t& v,
                                            state_t& hv) const
    {
        hv(0) = 2.0 * w(0) * v(0);
    }

    eq_term_pen_t terminal_equality_constr_soft_penalty() const
    {
        return eq_term_pen_t::Constant(1, 3.0);
    }
};

// ---------------------------------------------------------------------
// DegProbe: fixed x_0, one HARD ineq g = x_0 - 0.5 that depends only on
// the pinned state -> first-stage degeneracy must yield kInfeasible.
// ---------------------------------------------------------------------

struct DegDims
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
    static constexpr bool has_dynamics_hess_prod = false;
    static constexpr bool has_constr_hess_prod = false;
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

class DegProbe : public ocp::Problem<DegDims>
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

    void dynamics_jacobian(int, const state_t&, const control_t&,
                           dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
    {
        df_dx(0, 0) = 1.0;
        df_du(0, 0) = 1.0;
    }

    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return 0.5 * x(0) * x(0) + 0.5 * u(0) * u(0);
    }

    stage_grad_t stage_cost_gradient(int, const state_t& x,
                                     const control_t& u) const
    {
        stage_grad_t g;
        g(0) = x(0);
        g(1) = u(0);
        return g;
    }

    stage_hess_t stage_cost_hessian(int, const state_t&, const control_t&) const
    {
        return stage_hess_t::Identity();
    }

    double terminal_cost_value(const state_t& x) const
    {
        return 0.5 * x(0) * x(0);
    }

    term_grad_t terminal_cost_gradient(const state_t& x) const
    {
        return term_grad_t::Constant(1, x(0));
    }

    term_hess_t terminal_cost_hessian(const state_t&) const
    {
        return term_hess_t::Constant(1, 1, 1.0);
    }

    ineq_t stage_inequality_constr(int, const state_t& x, const control_t&) const
    {
        return ineq_t::Constant(1, x(0) - 0.5);
    }

    void stage_inequality_constr_jacobian(int, const state_t&, const control_t&,
                                          ineq_dg_dx_t& g_dx,
                                          ineq_dg_du_t& g_du) const
    {
        g_dx(0, 0) = 1.0;
        g_du(0, 0) = 0.0;  // independent of u -> degenerate when x_0 fixed
    }

    state_box_t stage_state_box_constr(int) const
    {
        state_box_t spec;
        spec.lo(0) = -10.0;
        spec.hi(0) = 10.0;
        spec.soft_penalty(0) = 0.0;
        return spec;
    }

    control_box_t stage_control_box_constr(int) const
    {
        control_box_t spec;
        spec.lo(0) = -1.0;
        spec.hi(0) = 1.0;
        spec.soft_penalty(0) = 0.0;
        return spec;
    }
};

// ---------------------------------------------------------------------
// Check A: DoubleIntegrator, N = 2, fixed x_0.
// ---------------------------------------------------------------------

void check_double_integrator()
{
    DoubleIntegrator prob;
    static constexpr int N = 2;

    Solution<DoubleIntegrator> sol(N);
    sol.x[0] << 1.0, 0.5;
    sol.x[1] << 1.1, 0.7;
    sol.x[2] << 1.15, 0.6;
    sol.u[0] << 0.2;
    sol.u[1] << -0.1;
    sol.lambda_dyn[0] << 1.0, 2.0;  // HVPs are zero; values are immaterial
    sol.lambda_dyn[1] << 3.0, 4.0;
    sol.lambda_ineq_stage[0] << 0.1;
    sol.lambda_ineq_stage[1] << 0.2;
    sol.lambda_eq_term << 0.3;

    SqpOptions opts;
    SqpSolver<DoubleIntegrator> solver(opts);
    solver.resize(N);

    const Status st = solver.assemble_qp(prob, sol);
    if (st != Status::kSolved)
    {
        std::fprintf(stderr,
                     "FAIL: DI assemble_qp status %d (want kSolved)\n",
                     static_cast<int>(st));
        ++failures;
        return;
    }

    const auto& qp = solver.last_qp();

    // -- first stage --------------------------------------------------
    {
        const auto& s = qp.first;
        // hess (u; x; s): cost Hessian in [u;x] layout (HVPs are zero for
        // the linear double integrator), slack diag = 100.
        double h_exp[4][4] = {
            {0.01, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 0.1, 0.0},
            {0.0, 0.0, 0.0, 100.0},
        };
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                check_close(s.hess(i, j), h_exp[i][j], "DI first hess");
        double g_exp[4] = {0.002, 1.0, 0.05, 0.0};
        for (int i = 0; i < 4; ++i)
            check_close(s.grad(i), g_exp[i], "DI first grad");
        // BA = [B | A], b = f(x_0, u_0) - x_1
        // BA = [B | A]: B = [0; Ts], A = [[0, Ts], [0, 0]]
        check_close(s.BA(0, 0), 0.0, "DI first BA(0,0)");
        check_close(s.BA(0, 1), 0.0, "DI first BA(0,1)");
        check_close(s.BA(0, 2), 0.1, "DI first BA(0,2)");
        check_close(s.BA(1, 0), 0.1, "DI first BA(1,0)");
        check_close(s.BA(1, 1), 0.0, "DI first BA(1,1)");
        check_close(s.BA(1, 2), 0.0, "DI first BA(1,2)");
        check_close(s.b(0), -0.05, "DI first b(0)");
        check_close(s.b(1), -0.18, "DI first b(1)");
        // DC: rows [pin, pin, bu, ineq, lin], cols (u; x; s)
        check_close(s.DC(0, 1), 1.0, "DI first DC pin0");
        check_close(s.DC(1, 2), 1.0, "DI first DC pin1");
        check_close(s.DC(2, 0), 1.0, "DI first DC bu");
        check_close(s.DC(3, 1), 0.0, "DI first DC ineq u");
        check_close(s.DC(3, 2), 1.0, "DI first DC ineq x");
        check_close(s.DC(3, 3), -1.0, "DI first DC ineq slack");
        check_close(s.DC(4, 0), 0.0, "DI first DC lin u");
        check_close(s.DC(4, 1), 0.5, "DI first DC lin x0");
        check_close(s.DC(4, 2), 1.0, "DI first DC lin x1");
        // d (offset form): [lo pin, lo pin, lo bu, lo lin,
        //                   hi pin, hi pin, hi bu, hi ineq, hi lin, slack]
        double d_exp[10] = {0.0, 0.0, -1.2, -4.0,
                            0.0, 0.0, -0.8, -1.5, -2.0, 0.0};
        for (int i = 0; i < 10; ++i)
            check_close(s.d(i), d_exp[i], "DI first d");
        for (int i = 0; i < 10; ++i)
            check_close(s.d_mask(i), 1.0, "DI first d_mask");
    }

    // -- path stage (k = 1) -------------------------------------------
    {
        const auto& s = qp.path[0];
        double h_exp[4][4] = {
            {0.01, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 0.1, 0.0},
            {0.0, 0.0, 0.0, 100.0},
        };
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                check_close(s.hess(i, j), h_exp[i][j], "DI path hess");
        double g_exp[4] = {-0.001, 1.1, 0.07, 0.0};
        for (int i = 0; i < 4; ++i)
            check_close(s.grad(i), g_exp[i], "DI path grad");
        check_close(s.BA(0, 2), 0.1, "DI path BA(0,2)");
        check_close(s.BA(1, 0), 0.1, "DI path BA(1,0)");
        check_close(s.BA(1, 2), 0.0, "DI path BA(1,2)");
        check_close(s.b(0), 0.02, "DI path b(0)");
        check_close(s.b(1), 0.09, "DI path b(1)");
        // DC: rows [bx, bx, bu, ineq, lin]
        check_close(s.DC(0, 1), 1.0, "DI path DC bx0");
        check_close(s.DC(1, 2), 1.0, "DI path DC bx1");
        check_close(s.DC(2, 0), 1.0, "DI path DC bu");
        check_close(s.DC(3, 2), 1.0, "DI path DC ineq x");
        check_close(s.DC(3, 3), -1.0, "DI path DC ineq slack");
        check_close(s.DC(4, 1), 0.5, "DI path DC lin x0");
        check_close(s.DC(4, 2), 1.0, "DI path DC lin x1");
        // d: [lo bx, lo bx, lo bu, lo lin,
        //     hi bx, hi bx, hi bu, hi ineq, hi lin, slack]
        double d_exp[10] = {-11.1, -10.7, -0.9, -4.25,
                            -8.9, -9.3, -1.1, -1.3, -1.75, 0.0};
        for (int i = 0; i < 10; ++i)
            check_close(s.d(i), d_exp[i], "DI path d");
    }

    // -- terminal stage ------------------------------------------------
    {
        const auto& s = qp.term;
        check_close(s.hess(0, 0), 20.0, "DI term hess(0,0)");
        check_close(s.hess(1, 1), 20.0, "DI term hess(1,1)");
        check_close(s.grad(0), 23.0, "DI term grad(0)");
        check_close(s.grad(1), 12.0, "DI term grad(1)");
        check_close(s.DC(0, 0), 1.0, "DI term DC bx0");
        check_close(s.DC(1, 1), 1.0, "DI term DC bx1");
        check_close(s.DC(2, 0), 1.0, "DI term DC eq x0");
        check_close(s.DC(2, 1), 1.0, "DI term DC eq x1");
        // d: [lo bx, lo bx, lo eq, hi bx, hi bx, hi eq]
        double d_exp[6] = {-11.15, -10.6, -0.75, -8.85, -9.4, 0.75};
        for (int i = 0; i < 6; ++i)
            check_close(s.d(i), d_exp[i], "DI term d");
        for (int i = 0; i < 6; ++i)
            check_close(s.d_mask(i), 1.0, "DI term d_mask");
    }

    // -- LM term ---------------------------------------------------------
    {
        SqpOptions opts_lm;
        opts_lm.levenberg_marquardt = 0.0;  // not used here; mu is explicit
        SqpSolver<DoubleIntegrator> solver_lm(opts_lm);
        solver_lm.resize(N);
        const Status st2 = solver_lm.assemble_qp(prob, sol);
        if (st2 != Status::kSolved)
        {
            ++failures;
            return;
        }
        auto& q = const_cast<Qp<DoubleIntegrator, Eigen::Dynamic>&>(
            solver_lm.last_qp());
        solver_lm.add_lm_term(q, 5.0);
        // first / path: (u;x) diagonal += 5, slack untouched
        double exp_diag[4] = {5.01, 6.0, 5.1, 100.0};
        for (int i = 0; i < 4; ++i)
        {
            check_close(q.first.hess(i, i), exp_diag[i], "DI LM first");
            check_close(q.path[0].hess(i, i), exp_diag[i], "DI LM path");
        }
        check_close(q.term.hess(0, 0), 25.0, "DI LM term(0,0)");
        check_close(q.term.hess(1, 1), 25.0, "DI LM term(1,1)");
        // off-diagonals unchanged
        check_close(q.first.hess(0, 1), 0.0, "DI LM first offdiag");
        // no-op: mu <= 0 leaves the Hessian untouched
        SqpSolver<DoubleIntegrator> solver_no;
        solver_no.resize(N);
        solver_no.assemble_qp(prob, sol);
        auto& q2 = const_cast<Qp<DoubleIntegrator, Eigen::Dynamic>&>(
            solver_no.last_qp());
        solver_no.add_lm_term(q2, 0.0);
        check_close(q2.first.hess(0, 0), 0.01, "DI LM no-op (mu=0)");
        // no-op: compute_hess = false leaves the Hessian untouched
        SqpOptions opts_no;
        opts_no.compute_hess = false;
        SqpSolver<DoubleIntegrator> solver_off(opts_no);
        solver_off.resize(N);
        solver_off.assemble_qp(prob, sol);
        auto& q3 = const_cast<Qp<DoubleIntegrator, Eigen::Dynamic>&>(
            solver_off.last_qp());
        solver_off.add_lm_term(q3, 5.0);
        check_close(q3.first.hess(0, 0), 0.01, "DI LM no-op (no hess)");
    }
}

// ---------------------------------------------------------------------
// Check B: QuadTest, N = 1, x_0 free, non-zero HVPs.
// ---------------------------------------------------------------------

void check_quad()
{
    QuadTest prob;
    static constexpr int N = 1;

    Solution<QuadTest> sol(N);
    sol.x[0] << 0.3;
    sol.x[1] << 0.5;
    sol.u[0] << 0.8;
    sol.lambda_dyn[0] << 1.5;
    sol.lambda_ineq_stage[0] << 0.4;
    sol.lambda_eq_term << -2.5;

    SqpOptions opts;
    SqpSolver<QuadTest> solver(opts);
    solver.resize(N);

    const Status st = solver.assemble_qp(prob, sol);
    if (st != Status::kSolved)
    {
        std::fprintf(stderr,
                     "FAIL: QuadTest assemble_qp status %d (want kSolved)\n",
                     static_cast<int>(st));
        ++failures;
        return;
    }

    const auto& qp = solver.last_qp();

    // -- first stage (only stage; no path) ------------------------------
    // hess (u; x; s): H_cost(u;x) - M_dyn + M_ineq, slack diag 7.
    //   H_cost = [[0.5, 0.5], [0.5, 2]],  M_dyn = [[0, 0.3], [0.3, 0]],
    //   M_ineq = [[0, 0.4], [0.4, 0]]
    {
        const auto& s = qp.first;
        double h_exp[3][3] = {
            {0.5, 0.6, 0.0},
            {0.6, 2.0, 0.0},
            {0.0, 0.0, 7.0},
        };
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                check_close(s.hess(i, j), h_exp[i][j], "Quad first hess");
        // grad (u; x; s): [gamma x + 2 beta u, 2 alpha x + gamma u, 0]
        check_close(s.grad(0), 0.55, "Quad first grad u");
        check_close(s.grad(1), 1.0, "Quad first grad x");
        check_close(s.grad(2), 0.0, "Quad first grad s");
        // BA = [B | A]: B = Ts x = 0.06, A = 1 + Ts u = 1.16
        check_close(s.BA(0, 0), 0.06, "Quad first BA B");
        check_close(s.BA(0, 1), 1.16, "Quad first BA A");
        // b = x + Ts x u - x_1 = 0.3 + 0.048 - 0.5
        check_close(s.b(0), -0.152, "Quad first b");
        // DC rows [bx, bu, ineq], cols (u; x; s)
        check_close(s.DC(0, 1), 1.0, "Quad first DC bx");
        check_close(s.DC(1, 0), 1.0, "Quad first DC bu");
        check_close(s.DC(2, 0), 0.3, "Quad first DC ineq du");
        check_close(s.DC(2, 1), 1.8, "Quad first DC ineq dx");
        check_close(s.DC(2, 2), -1.0, "Quad first DC ineq slack");
        // d: [lo bx, lo bu, hi bx, hi bu, hi ineq, slack]
        double d_exp[6] = {-5.3, -2.8, -4.7, -1.2, 0.04, 0.0};
        for (int i = 0; i < 6; ++i)
            check_close(s.d(i), d_exp[i], "Quad first d");
        for (int i = 0; i < 6; ++i)
            check_close(s.d_mask(i), 1.0, "Quad first d_mask");
    }

    // -- terminal ---------------------------------------------------------
    // hess (x; s_lo; s_hi): 8 (cost) + 2 * (-2.5) (eq HVP) = 3; slack 3.
    {
        const auto& s = qp.term;
        double h_exp[3][3] = {
            {3.0, 0.0, 0.0},
            {0.0, 3.0, 0.0},
            {0.0, 0.0, 3.0},
        };
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                check_close(s.hess(i, j), h_exp[i][j], "Quad term hess");
        check_close(s.grad(0), 4.0, "Quad term grad");
        check_close(s.grad(1), 0.0, "Quad term grad s_lo");
        check_close(s.grad(2), 0.0, "Quad term grad s_hi");
        // DC row [eq], cols (x; s_lo; s_hi)
        check_close(s.DC(0, 0), 1.0, "Quad term DC eq dx");
        check_close(s.DC(0, 1), 1.0, "Quad term DC eq slack lo");
        check_close(s.DC(0, 2), -1.0, "Quad term DC eq slack hi");
        // d: [lo eq, hi eq, slack, slack]
        double d_exp[4] = {0.75, -0.75, 0.0, 0.0};
        for (int i = 0; i < 4; ++i)
            check_close(s.d(i), d_exp[i], "Quad term d");
        for (int i = 0; i < 4; ++i)
            check_close(s.d_mask(i), 1.0, "Quad term d_mask");
    }
}

// ---------------------------------------------------------------------
// Check C: DegProbe -> first-stage degeneracy => kInfeasible.
// ---------------------------------------------------------------------

void check_degeneracy()
{
    DegProbe prob;
    static constexpr int N = 1;

    Solution<DegProbe> sol(N);
    sol.x[0] << 1.0;
    sol.x[1] << 1.0;
    sol.u[0] << 0.0;

    SqpOptions opts;
    SqpSolver<DegProbe> solver(opts);
    solver.resize(N);

    const Status st = solver.assemble_qp(prob, sol);
    if (st != Status::kInfeasible)
    {
        std::fprintf(stderr,
                     "FAIL: DegProbe assemble_qp status %d (want "
                     "kInfeasible)\n",
                     static_cast<int>(st));
        ++failures;
    }
}

}  // namespace

int run_assemble_2f_tests()
{
    check_double_integrator();
    check_quad();
    check_degeneracy();
    return failures;
}

// Phase 3 block 3a test: the shared LHS / RHS QP assembly split
// (include/ocp/solvers/acados/assemble.hpp, SQP_HPIPM_PARITY_PHASE3_PLAN.md
// block 3a).
//
// Checks:
//   - Split A (DoubleIntegrator, N = 10) / B (MassesChain, N = 40): the
//     shared free functions (detail::eval_stage / eval_term +
//     detail::fill_stage_{first,path,term}_{lhs,rhs}) reproduce
//     AcadosSolver::assemble_qp bit-exactly, every stage field including
//     d_mask and m.
//   - LHS-only fill: all RHS fields (grad, b, d) remain exactly 0.
//   - RHS-only fill on a zero QP: all LHS fields (hess, BA, DC, d_mask)
//     remain exactly 0 (trivial-row masking can only clear entries).
//   - Hessian gate (HessGate, non-zero constraint HVP): the contracted
//     Hessian enters the stage hess only when compute_hess_constr is set.
//   - Degenerate first stage (DegProbe: fixed x_0, hard ineq violated at
//     the pinned state, row independent of u): check_first_degeneracy and
//     fill_stage_rhs_first return kInfeasible, and the public assemble_qp
//     reports kInfeasible as well.

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/solvers/acados/acados.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"
#include "../../examples/masses_chain/masses_chain.hpp"

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

/// max |m| over all entries (the matrix is exactly zero iff this is 0).
template <class M>
double max_abs(const M& m)
{
    double mx = 0.0;
    for (Eigen::Index i = 0; i < m.size(); ++i)
    {
        mx = std::max(mx, std::fabs(static_cast<double>(m.coeff(i))));
    }
    return mx;
}

/// max |a - b| over all entries (exact equality iff this is 0).
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

// ---------------------------------------------------------------
//  Per-stage field checks (exact 0 / exact equality)
// ---------------------------------------------------------------

template <class P>
void check_stage_rhs_zero(const QpStageFirst<P>& st, const std::string& w)
{
    check_true(max_abs(st.grad) == 0.0, w + ": grad not zero");
    check_true(max_abs(st.b) == 0.0, w + ": b not zero");
    check_true(max_abs(st.d) == 0.0, w + ": d not zero");
}

template <class P>
void check_stage_rhs_zero(const QpStagePath<P>& st, const std::string& w)
{
    check_true(max_abs(st.grad) == 0.0, w + ": grad not zero");
    check_true(max_abs(st.b) == 0.0, w + ": b not zero");
    check_true(max_abs(st.d) == 0.0, w + ": d not zero");
}

template <class P>
void check_term_rhs_zero(const QpStageTerm<P>& st, const std::string& w)
{
    check_true(max_abs(st.grad) == 0.0, w + ": grad not zero");
    check_true(max_abs(st.d) == 0.0, w + ": d not zero");
}

template <class P>
void check_stage_lhs_zero(const QpStageFirst<P>& st, const std::string& w)
{
    check_true(max_abs(st.hess) == 0.0, w + ": hess not zero");
    check_true(max_abs(st.BA) == 0.0, w + ": BA not zero");
    check_true(max_abs(st.DC) == 0.0, w + ": DC not zero");
    check_true(max_abs(st.d_mask) == 0.0, w + ": d_mask not zero");
}

template <class P>
void check_stage_lhs_zero(const QpStagePath<P>& st, const std::string& w)
{
    check_true(max_abs(st.hess) == 0.0, w + ": hess not zero");
    check_true(max_abs(st.BA) == 0.0, w + ": BA not zero");
    check_true(max_abs(st.DC) == 0.0, w + ": DC not zero");
    check_true(max_abs(st.d_mask) == 0.0, w + ": d_mask not zero");
}

template <class P>
void check_term_lhs_zero(const QpStageTerm<P>& st, const std::string& w)
{
    check_true(max_abs(st.hess) == 0.0, w + ": hess not zero");
    check_true(max_abs(st.DC) == 0.0, w + ": DC not zero");
    check_true(max_abs(st.d_mask) == 0.0, w + ": d_mask not zero");
}

template <class P>
void check_stage_lhs_equal(const QpStageFirst<P>& a, const QpStageFirst<P>& b,
                           const std::string& w)
{
    check_true(max_abs_diff(a.hess, b.hess) == 0.0, w + ": hess");
    check_true(max_abs_diff(a.BA, b.BA) == 0.0, w + ": BA");
    check_true(max_abs_diff(a.DC, b.DC) == 0.0, w + ": DC");
    // structural d_mask may only be cleared (by trivial-row masking in
    // the RHS fill), never set: a >= b element-wise
    for (Eigen::Index i = 0; i < a.d_mask.size(); ++i)
    {
        check_true(a.d_mask.coeff(i) >= b.d_mask.coeff(i) - 1e-15,
                   w + ": d_mask structural < masked");
    }
}

template <class P>
void check_stage_lhs_equal(const QpStagePath<P>& a, const QpStagePath<P>& b,
                           const std::string& w)
{
    check_true(max_abs_diff(a.hess, b.hess) == 0.0, w + ": hess");
    check_true(max_abs_diff(a.BA, b.BA) == 0.0, w + ": BA");
    check_true(max_abs_diff(a.DC, b.DC) == 0.0, w + ": DC");
    for (Eigen::Index i = 0; i < a.d_mask.size(); ++i)
    {
        check_true(a.d_mask.coeff(i) >= b.d_mask.coeff(i) - 1e-15,
                   w + ": d_mask structural < masked");
    }
}

template <class P>
void check_term_lhs_equal(const QpStageTerm<P>& a, const QpStageTerm<P>& b,
                          const std::string& w)
{
    check_true(max_abs_diff(a.hess, b.hess) == 0.0, w + ": hess");
    check_true(max_abs_diff(a.DC, b.DC) == 0.0, w + ": DC");
    for (Eigen::Index i = 0; i < a.d_mask.size(); ++i)
    {
        check_true(a.d_mask.coeff(i) >= b.d_mask.coeff(i) - 1e-15,
                   w + ": d_mask structural < masked");
    }
}

// Note: these compare only the fields the RHS fill owns (grad, b, d).
// d_mask is an LHS field: it is written structurally by the LHS fill and
// only *cleared* by trivial-row masking, which also needs the LHS DC to
// judge the Jacobian. On an RHS-only (zero-DC) QP the mask is not
// meaningful and is checked to stay 0 instead.
template <class P>
void check_stage_rhs_equal(const QpStageFirst<P>& a, const QpStageFirst<P>& b,
                           const std::string& w)
{
    check_true(max_abs_diff(a.grad, b.grad) == 0.0, w + ": grad");
    check_true(max_abs_diff(a.b, b.b) == 0.0, w + ": b");
    check_true(max_abs_diff(a.d, b.d) == 0.0, w + ": d");
}

template <class P>
void check_stage_rhs_equal(const QpStagePath<P>& a, const QpStagePath<P>& b,
                           const std::string& w)
{
    check_true(max_abs_diff(a.grad, b.grad) == 0.0, w + ": grad");
    check_true(max_abs_diff(a.b, b.b) == 0.0, w + ": b");
    check_true(max_abs_diff(a.d, b.d) == 0.0, w + ": d");
}

template <class P>
void check_term_rhs_equal(const QpStageTerm<P>& a, const QpStageTerm<P>& b,
                          const std::string& w)
{
    check_true(max_abs_diff(a.grad, b.grad) == 0.0, w + ": grad");
    check_true(max_abs_diff(a.d, b.d) == 0.0, w + ": d");
}

// Full (LHS+RHS) comparison: every field, incl. d_mask, must be exact.
template <class P>
void check_stage_full(const QpStageFirst<P>& a, const QpStageFirst<P>& b,
                      const std::string& w)
{
    check_stage_lhs_equal(a, b, w);
    check_stage_rhs_equal(a, b, w);
    check_true(max_abs_diff(a.d_mask, b.d_mask) == 0.0, w + ": d_mask");
    check_true(max_abs_diff(a.m, b.m) == 0.0, w + ": m");
}

template <class P>
void check_stage_full(const QpStagePath<P>& a, const QpStagePath<P>& b,
                      const std::string& w)
{
    check_stage_lhs_equal(a, b, w);
    check_stage_rhs_equal(a, b, w);
    check_true(max_abs_diff(a.d_mask, b.d_mask) == 0.0, w + ": d_mask");
    check_true(max_abs_diff(a.m, b.m) == 0.0, w + ": m");
}

template <class P>
void check_term_full(const QpStageTerm<P>& a, const QpStageTerm<P>& b,
                     const std::string& w)
{
    check_term_lhs_equal(a, b, w);
    check_term_rhs_equal(a, b, w);
    check_true(max_abs_diff(a.d_mask, b.d_mask) == 0.0, w + ": d_mask");
    check_true(max_abs_diff(a.m, b.m) == 0.0, w + ": m");
}

// ---------------------------------------------------------------
//  Split fill driver: the shared free functions, one side or both
// ---------------------------------------------------------------

template <class P>
Status fill_split(Qp<P>& qp, const P& prob, const Solution<P>& sol,
                  const detail::AssembleFlags& fl, bool do_lhs, bool do_rhs)
{
    const int N = sol.N;

    const auto ev0 = detail::eval_stage(prob, sol, 0, fl);
    if (do_lhs)
    {
        detail::fill_stage_lhs_first(prob, ev0, fl, qp.first);
    }
    Status st = Status::kSolved;
    if (do_rhs)
    {
        st = detail::fill_stage_rhs_first(prob, sol, ev0, qp.first);
    }
    if (st != Status::kSolved)
    {
        return st;
    }
    for (int k = 1; k < N; ++k)
    {
        const auto ev = detail::eval_stage(prob, sol, k, fl);
        if (do_lhs)
        {
            detail::fill_stage_lhs_path(prob, k, ev, fl, qp.path[k - 1]);
        }
        if (do_rhs)
        {
            detail::fill_stage_rhs_path(prob, sol, k, ev, qp.path[k - 1]);
        }
    }
    const auto tev = detail::eval_term(prob, sol, fl);
    if (do_lhs)
    {
        detail::fill_stage_lhs_term(prob, tev, fl, qp.term);
    }
    if (do_rhs)
    {
        detail::fill_stage_rhs_term(prob, sol, tev, qp.term);
    }
    return Status::kSolved;
}

// ---------------------------------------------------------------
//  Deterministic non-trivial iterate: off-set states, non-zero control,
//  non-zero multipliers (exercises the contracted-Hessian paths)
// ---------------------------------------------------------------

template <class P>
void init_iterate(const P& prob, Solution<P>& sol)
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
}

// ---------------------------------------------------------------
//  Split check: LHS / RHS / full vs the public assemble_qp
// ---------------------------------------------------------------

template <class P>
void check_split(const char* name, int N)
{
    P prob;
    Solution<P> sol(N);
    init_iterate(prob, sol);
    const std::string w(name);

    AcadosSolver<P> ref_solver;
    ref_solver.resize(N);
    const Status st_ref = ref_solver.assemble_qp(prob, sol);
    check_true(st_ref == Status::kSolved, w + ": assemble_qp status");
    if (failures > 0)
    {
        return;
    }
    const auto& ref = ref_solver.last_qp();

    const detail::AssembleFlags fl{true, true};

    // LHS only: RHS fields stay 0, LHS fields match the reference
    Qp<P> qp_l(N);
    const Status st_l = fill_split(qp_l, prob, sol, fl, true, false);
    check_true(st_l == Status::kSolved, w + ": split LHS status");
    check_stage_rhs_zero(qp_l.first, w + " LHS k=0");
    for (int k = 1; k < N; ++k)
    {
        check_stage_rhs_zero(qp_l.path[k - 1],
                             w + " LHS k=" + std::to_string(k));
    }
    check_term_rhs_zero(qp_l.term, w + " LHS term");
    check_stage_lhs_equal(qp_l.first, ref.first, w + " LHS k=0");
    for (int k = 1; k < N; ++k)
    {
        check_stage_lhs_equal(qp_l.path[k - 1], ref.path[k - 1],
                              w + " LHS k=" + std::to_string(k));
    }
    check_term_lhs_equal(qp_l.term, ref.term, w + " LHS term");

    // RHS only: LHS fields stay 0, RHS fields match the reference
    Qp<P> qp_r(N);
    const Status st_r = fill_split(qp_r, prob, sol, fl, false, true);
    check_true(st_r == Status::kSolved, w + ": split RHS status");
    check_stage_lhs_zero(qp_r.first, w + " RHS k=0");
    for (int k = 1; k < N; ++k)
    {
        check_stage_lhs_zero(qp_r.path[k - 1],
                             w + " RHS k=" + std::to_string(k));
    }
    check_term_lhs_zero(qp_r.term, w + " RHS term");
    check_stage_rhs_equal(qp_r.first, ref.first, w + " RHS k=0");
    for (int k = 1; k < N; ++k)
    {
        check_stage_rhs_equal(qp_r.path[k - 1], ref.path[k - 1],
                              w + " RHS k=" + std::to_string(k));
    }
    check_term_rhs_equal(qp_r.term, ref.term, w + " RHS term");

    // LHS + RHS: every field bit-exact against the reference
    Qp<P> qp_f(N);
    const Status st_f = fill_split(qp_f, prob, sol, fl, true, true);
    check_true(st_f == Status::kSolved, w + ": split full status");
    check_stage_full(qp_f.first, ref.first, w + " full k=0");
    for (int k = 1; k < N; ++k)
    {
        check_stage_full(qp_f.path[k - 1], ref.path[k - 1],
                         w + " full k=" + std::to_string(k));
    }
    check_term_full(qp_f.term, ref.term, w + " full term");
}

// ---------------------------------------------------------------
//  HessGate: nx = 1, nu = 1, x_0 free, one HARD ineq with a non-zero
//  contracted Hessian (quadratic in the state). No box, no dynamics
//  Hessian. Verifies the compute_hess_constr gate on the shared LHS fill.
//
//   dynamics:  x_{k+1} = x + 0.5 u        (df_dx = 1, df_du = 0.5)
//   cost:      0.5 x^2 + 0.25 u^2          (H = diag(1, 0.5) [x; u])
//   ineq:      0.5 x^2 + x u - 1 <= 0      (D2g = [[1,1],[1,0]] [x; u])
//   terminal:  0.5 x^2
//
//   Iterate (N = 1): x_0 = 0.3, u_0 = 0.8, x_1 = 0.5,
//   lambda_ineq_stage[0] = 0.4.
//   QP hess (u;x layout): on  = [[0.5, 0.4], [0.4, 1.4]],
//                         off = [[0.5, 0.0], [0.0, 1.0]].
// ---------------------------------------------------------------

struct HessGateDims
{
    static constexpr int nx = 1;
    static constexpr int nu = 1;
    static constexpr int ng = 1;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 0;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = false;
    static constexpr bool has_dynamics_hess = false;
    static constexpr bool has_constr_hess = true;
    static constexpr std::array<int, 0> state_box_idx = {};
    static constexpr std::array<int, 0> control_box_idx = {};
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

class HessGate : public ocp::Problem<HessGateDims>
{
public:
    state_t dynamics_next_state(int, const state_t& x, const control_t& u) const
    {
        return state_t::Constant(1, x(0) + 0.5 * u(0));
    }

    void dynamics_value_jac(int, const state_t& x, const control_t& u,
                            state_t& x_next, dyn_df_dx_t& df_dx,
                            dyn_df_du_t& df_du) const
    {
        x_next(0) = x(0) + 0.5 * u(0);
        df_dx(0, 0) = 1.0;
        df_du(0, 0) = 0.5;
    }

    void stage_cost_value_grad_hess(int, const state_t& x, const control_t& u,
                                    double& value, stage_grad_t& grad,
                                    stage_hess_t& hess) const
    {
        value = 0.5 * x(0) * x(0) + 0.25 * u(0) * u(0);
        grad(0) = x(0);
        grad(1) = 0.5 * u(0);
        hess.setZero();
        hess(0, 0) = 1.0;
        hess(1, 1) = 0.5;
    }

    void terminal_cost_value_grad_hess(const state_t& x, double& value,
                                       term_grad_t& grad,
                                       term_hess_t& hess) const
    {
        value = 0.5 * x(0) * x(0);
        grad(0) = x(0);
        hess(0, 0) = 1.0;
    }

    void stage_inequality_value_jac(int, const state_t& x, const control_t& u,
                                    ineq_t& g, ineq_dg_dx_t& g_dx,
                                    ineq_dg_du_t& g_du) const
    {
        g(0) = 0.5 * x(0) * x(0) + x(0) * u(0) - 1.0;
        g_dx(0, 0) = x(0) + u(0);
        g_du(0, 0) = x(0);
    }

    void stage_inequality_value_jac_hess(int, const state_t& x,
                                         const control_t& u,
                                         const ineq_t& lam, ineq_t& g,
                                         ineq_dg_dx_t& g_dx,
                                         ineq_dg_du_t& g_du,
                                         constr_hess_t& hess) const
    {
        stage_inequality_value_jac(0, x, u, g, g_dx, g_du);
        // contracted Hessian lam^T D2g, [x; u] layout: D2g = [[1,1],[1,0]]
        hess.setZero();
        hess(0, 0) = lam(0);
        hess(0, 1) = lam(0);
        hess(1, 0) = lam(0);
    }
};

void check_hess_gate()
{
    HessGate prob;
    Solution<HessGate> sol(1);
    sol.x[0] << 0.3;
    sol.x[1] << 0.5;
    sol.u[0] << 0.8;
    sol.lambda_dyn[0].setZero();
    sol.lambda_ineq_stage[0] << 0.4;

    AcadosSolver<HessGate> solver_on;
    solver_on.resize(1);
    check_true(solver_on.assemble_qp(prob, sol) == Status::kSolved,
               "HessGate: assemble (gate on)");
    if (failures > 0)
    {
        return;
    }
    const auto& q_on = solver_on.last_qp();
    check_close(q_on.first.hess(0, 0), 0.5, "HessGate on hess(u,u)");
    check_close(q_on.first.hess(0, 1), 0.4, "HessGate on hess(u,x)");
    check_close(q_on.first.hess(1, 0), 0.4, "HessGate on hess(x,u)");
    check_close(q_on.first.hess(1, 1), 1.4, "HessGate on hess(x,x)");

    AcadosOptions opts_off;
    opts_off.compute_hess_constr = false;
    AcadosSolver<HessGate> solver_off(opts_off);
    solver_off.resize(1);
    check_true(solver_off.assemble_qp(prob, sol) == Status::kSolved,
               "HessGate: assemble (gate off)");
    const auto& q_off = solver_off.last_qp();
    check_close(q_off.first.hess(0, 1), 0.0, "HessGate off hess(u,x)");
    check_close(q_off.first.hess(1, 0), 0.0, "HessGate off hess(x,u)");
    check_close(q_off.first.hess(1, 1), 1.0, "HessGate off hess(x,x)");
    // every other field identical regardless of the gate
    check_true(max_abs_diff(q_off.first.grad, q_on.first.grad) == 0.0,
               "HessGate off grad");
    check_true(max_abs_diff(q_off.first.BA, q_on.first.BA) == 0.0,
               "HessGate off BA");
    check_true(max_abs_diff(q_off.first.b, q_on.first.b) == 0.0,
               "HessGate off b");
    check_true(max_abs_diff(q_off.first.DC, q_on.first.DC) == 0.0,
               "HessGate off DC");
    check_true(max_abs_diff(q_off.first.d, q_on.first.d) == 0.0,
               "HessGate off d");
    check_close(q_off.first.hess(0, 0), 0.5, "HessGate off hess(u,u)");
}

// ---------------------------------------------------------------
//  DegProbe: fixed x_0 = 1, one HARD ineq g = x - 0.5 that depends only
//  on the pinned state -> first-stage degeneracy must yield kInfeasible
//  (plan sec. 2f.6; same problem as the phase-2 test, here exercised via
//  the shared free functions).
// ---------------------------------------------------------------

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

    void dynamics_value_jac(int, const state_t& x, const control_t& u,
                            state_t& x_next, dyn_df_dx_t& df_dx,
                            dyn_df_du_t& df_du) const
    {
        x_next(0) = x(0) + u(0);
        df_dx(0, 0) = 1.0;
        df_du(0, 0) = 1.0;
    }

    void stage_cost_value_grad_hess(int, const state_t& x, const control_t& u,
                                    double& value, stage_grad_t& grad,
                                    stage_hess_t& hess) const
    {
        value = 0.5 * x(0) * x(0) + 0.5 * u(0) * u(0);
        grad(0) = x(0);
        grad(1) = u(0);
        hess = stage_hess_t::Identity();
    }

    void terminal_cost_value_grad_hess(const state_t& x, double& value,
                                       term_grad_t& grad,
                                       term_hess_t& hess) const
    {
        value = 0.5 * x(0) * x(0);
        grad(0) = x(0);
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
        g_du(0, 0) = 0.0;  // independent of u -> degenerate when x_0 fixed
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

void check_degeneracy()
{
    DegProbe prob;
    Solution<DegProbe> sol(1);
    sol.x[0] << 1.0;
    sol.x[1] << 1.0;
    sol.u[0] << 0.0;
    sol.lambda_dyn[0].setZero();
    sol.lambda_ineq_stage[0].setZero();

    check_true(detail::check_first_degeneracy(prob, sol)
                   == Status::kInfeasible,
               "DegProbe: check_first_degeneracy");

    // the shared RHS first-stage fill must report it
    Qp<DegProbe> qp(1);
    const detail::AssembleFlags fl{true, true};
    const auto ev = detail::eval_stage(prob, sol, 0, fl);
    detail::fill_stage_lhs_first(prob, ev, fl, qp.first);
    const Status st_fill = detail::fill_stage_rhs_first(prob, sol, ev,
                                                        qp.first);
    check_true(st_fill == Status::kInfeasible,
               "DegProbe: fill_stage_rhs_first");

    // and the public path still does
    AcadosSolver<DegProbe> solver;
    solver.resize(1);
    const Status st = solver.assemble_qp(prob, sol);
    check_true(st == Status::kInfeasible, "DegProbe: assemble_qp");
}

}  // namespace

int run_assemble_rti_3a_tests()
{
    check_split<DoubleIntegrator>("DI", 10);
    check_split<MassesChain>("MC", 40);
    check_hess_gate();
    check_degeneracy();
    if (failures == 0)
    {
        std::printf("  3a split (DI N=10, MC N=40): LHS/RHS fills "
                    "bit-exact vs assemble_qp; hess gate + degeneracy OK\n");
    }
    return failures;
}

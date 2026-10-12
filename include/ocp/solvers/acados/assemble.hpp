// ocp/solvers/acados/assemble.hpp
//
// Shared per-stage QP assembly for the SQP and RTI drivers (phase 3,
// block 3a, docs/plans/SQP_HPIPM_PARITY_PHASE3_PLAN.md).  Extracted from
// AcadosSolver (acados.hpp).  Each stage is split into an LHS fill
// (hess, BA, DC, d_mask, slack weights) and an RHS fill (grad, b, bound
// offsets d, slack linear part, trivial-row masking), mirroring acados'
// approximate_qp_matrices / approximate_qp_vectors_sqp.  Both fills
// consume one model-evaluation bundle (eval_stage / eval_term), so every
// problem function is evaluated exactly once per stage per assembly, as
// in the pre-split AcadosSolver.
//
// Contents:
//   - AssembleFlags                  (Hessian gates)
//   - StageConstr / TermConstr       (constraint evaluation records)
//   - StageEval / TermEval           (LHS + RHS evaluation bundles)
//   - eval_stage / eval_term         (one fused problem call per module)
//   - fill_dc_{first,path,term}      (LHS constraint Jacobian)
//   - fill_d_mask_lhs_{first,path,term} / fill_d_rhs_{first,path,term}
//   - fill_slack_diag_{lhs,rhs} / fill_slack_diag_term_{lhs,rhs}
//   - fill_stage_{first,path,term}_{lhs,rhs}  (per-stage composite fills)
//   - mask_trivial_zero_rows, check_first_degeneracy

#pragma once

#include <cmath>

#include <Eigen/Dense>

#include "../../problem.hpp"
#include "../hpipm/qp.hpp"

namespace ocp
{

namespace detail
{

/// Hessian gates for QP assembly (mirrors the corresponding
/// AcadosOptions fields).  The flags pick the fused *_hess problem
/// variants and whether the contracted Hessians are added to the stage
/// Hessian.
struct AssembleFlags
{
    bool compute_hess_dyn = true;
    bool compute_hess_constr = true;
};

/// Stage nonlinear-constraint evaluations at one iterate (values +
/// Jacobians), filled by a single fused problem call.
template <class P>
struct StageConstr
{
    typename P::ineq_t g{};
    typename P::eq_t e{};
    typename P::ineq_dg_dx_t g_dx{};
    typename P::ineq_dg_du_t g_du{};
    typename P::eq_de_dx_t e_dx{};
    typename P::eq_de_du_t e_du{};
};

/// Terminal nonlinear-constraint evaluations (state only).
template <class P>
struct TermConstr
{
    typename P::ineq_term_t g{};
    typename P::eq_term_t e{};
    typename P::ineq_term_dg_dx_t g_dx{};
    typename P::eq_term_de_dx_t e_dx{};
};

/// Stage (first / path) model evaluations at one iterate: one fused
/// problem call per module (cost, dynamics, stage constraints).  Carries
/// the LHS quantities (Hessian, Jacobians, contracted Hessians) and the
/// RHS quantities (cost gradient, dynamics value, constraint values,
/// linear-constraint values / spec); the Hessian gates pick the problem
/// variants (contracted Hessians are only valid when the gate is set and
/// the problem provides them).
template <class P>
struct StageEval
{
    typename P::stage_grad_t g_cost{};
    typename P::stage_hess_t H{};
    typename P::state_t f{};
    typename P::dyn_df_dx_t A{};
    typename P::dyn_df_du_t B{};
    typename P::dyn_hess_t M_dyn{};
    StageConstr<P> c;
    typename P::constr_hess_t M_ineq{};
    typename P::constr_hess_t M_eq{};
    typename P::lin_t lv{};
    typename P::stage_linear_t lin_spec{};
};

/// Terminal model evaluations at x_N (state only).  See StageEval.
template <class P>
struct TermEval
{
    typename P::term_grad_t g_term{};
    typename P::term_hess_t Ht{};
    TermConstr<P> tc;
    typename P::term_constr_hess_t M_ineq{};
    typename P::term_constr_hess_t M_eq{};
    typename P::lin_t lv{};
    typename P::term_linear_t lin_spec{};
};

/// Evaluate stage k once into a StageEval bundle (fused problem calls).
template <class P, int NH>
StageEval<P> eval_stage(const P& problem, const Solution<P, NH>& sol, int k,
                        const AssembleFlags& fl)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    StageEval<P> ev;
    const auto& x = sol.x[k];
    const auto& u = sol.u[k];

    // cost: fused value + grad + hess at the working iterate
    double lcost;
    problem.stage_cost_value_grad_hess(k, x, u, lcost, ev.g_cost, ev.H);

    // dynamics: fused value + Jacobian (+ contracted Hessian)
    if constexpr (P::has_dynamics_hess)
    {
        if (fl.compute_hess_dyn)
            problem.dynamics_value_jac_hess(k, x, u, sol.lambda_dyn[k],
                                            ev.f, ev.A, ev.B, ev.M_dyn);
        else
            problem.dynamics_value_jac(k, x, u, ev.f, ev.A, ev.B);
    }
    else
    {
        problem.dynamics_value_jac(k, x, u, ev.f, ev.A, ev.B);
    }

    // stage nonlinear constraints: fused value + Jacobians
    // (+ contracted Hessians)
    if constexpr (D::ng > 0)
    {
        if constexpr (P::has_constr_hess)
        {
            if (fl.compute_hess_constr)
                problem.stage_inequality_value_jac_hess(
                    k, x, u, sol.lambda_ineq_stage[k],
                    ev.c.g, ev.c.g_dx, ev.c.g_du, ev.M_ineq);
            else
                problem.stage_inequality_value_jac(k, x, u, ev.c.g, ev.c.g_dx,
                                                   ev.c.g_du);
        }
        else
        {
            problem.stage_inequality_value_jac(k, x, u, ev.c.g, ev.c.g_dx,
                                               ev.c.g_du);
        }
    }
    if constexpr (D::ne > 0)
    {
        if constexpr (P::has_constr_hess)
        {
            if (fl.compute_hess_constr)
                problem.stage_equality_value_jac_hess(
                    k, x, u, sol.lambda_eq_stage[k],
                    ev.c.e, ev.c.e_dx, ev.c.e_du, ev.M_eq);
            else
                problem.stage_equality_value_jac(k, x, u, ev.c.e, ev.c.e_dx,
                                                 ev.c.e_du);
        }
        else
        {
            problem.stage_equality_value_jac(k, x, u, ev.c.e, ev.c.e_dx,
                                             ev.c.e_du);
        }
    }

    // linear constraint values at the iterate (for the d offsets)
    if constexpr (D::nl > 0)
    {
        ev.lin_spec = problem.stage_linear_constr(k);
        for (int i = 0; i < D::nl; ++i)
        {
            double val = 0.0;
            for (int j = 0; j < D::nx; ++j)
            {
                val += static_cast<double>(ev.lin_spec.A(i, j))
                     * static_cast<double>(x(j));
            }
            for (int j = 0; j < D::nu; ++j)
            {
                val += static_cast<double>(ev.lin_spec.B(i, j))
                     * static_cast<double>(u(j));
            }
            ev.lv(i) = static_cast<S>(val);
        }
    }
    return ev;
}

/// Evaluate the terminal stage once into a TermEval bundle.
template <class P, int NH>
TermEval<P> eval_term(const P& problem, const Solution<P, NH>& sol,
                      const AssembleFlags& fl)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    TermEval<P> ev;
    const int N = sol.N;
    const auto& x = sol.x[N];

    // terminal cost: fused value + grad + hess
    double tcost;
    problem.terminal_cost_value_grad_hess(x, tcost, ev.g_term, ev.Ht);

    // terminal nonlinear constraints: fused value + Jacobian
    // (+ contracted Hessian)
    if constexpr (D::ng_t > 0)
    {
        if constexpr (P::has_constr_hess)
        {
            if (fl.compute_hess_constr)
                problem.terminal_inequality_value_jac_hess(
                    x, sol.lambda_ineq_term, ev.tc.g, ev.tc.g_dx, ev.M_ineq);
            else
                problem.terminal_inequality_value_jac(x, ev.tc.g, ev.tc.g_dx);
        }
        else
        {
            problem.terminal_inequality_value_jac(x, ev.tc.g, ev.tc.g_dx);
        }
    }
    if constexpr (D::ne_t > 0)
    {
        if constexpr (P::has_constr_hess)
        {
            if (fl.compute_hess_constr)
                problem.terminal_equality_value_jac_hess(
                    x, sol.lambda_eq_term, ev.tc.e, ev.tc.e_dx, ev.M_eq);
            else
                problem.terminal_equality_value_jac(x, ev.tc.e, ev.tc.e_dx);
        }
        else
        {
            problem.terminal_equality_value_jac(x, ev.tc.e, ev.tc.e_dx);
        }
    }

    // terminal linear constraint values at the iterate
    if constexpr (D::nl_t > 0)
    {
        ev.lin_spec = problem.terminal_linear_constr();
        for (int i = 0; i < D::nl_t; ++i)
        {
            double val = 0.0;
            for (int j = 0; j < D::nx; ++j)
            {
                val += static_cast<double>(ev.lin_spec.A(i, j))
                     * static_cast<double>(x(j));
            }
            ev.lv(i) = static_cast<S>(val);
        }
    }
    return ev;
}

/// [x;u]-layout matrix into the QP (u;x) layout, element-wise.
template <class P, class HessT>
void add_permuted_hess(HessT& hess, const typename P::constr_hess_t& M,
                       double sign)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    for (int i = 0; i < D::nu + D::nx; ++i)
    {
        for (int j = 0; j < D::nu + D::nx; ++j)
        {
            const int pi = (i < D::nu) ? D::nx + i : i - D::nu;
            const int pj = (j < D::nu) ? D::nx + j : j - D::nu;
            hess(i, j) += sign * static_cast<S>(M(pi, pj));
        }
    }
}

// ---------------------------------------------------------------
//  DC fill (LHS: constraint Jacobian over the stage variable vector)
// ---------------------------------------------------------------

template <class P, class StT>
void fill_dc_first(StT& st, const StageConstr<P>& c,
                   const typename P::stage_linear_t& lin_spec)
{
    using D = QpDim<P>;
    const auto& lay = D::lay_first;

    // pin rows: unit vector at the pinned x_0 variable
    if constexpr (P::fixed_initial_state)
    {
        for (int j = 0; j < D::nx; ++j)
        {
            st.DC(j, D::idx_x0[j]) = 1;
        }
    }

    // state box (absent when x_0 is fixed)
    if constexpr (D::nbx_first > 0)
    {
        for (int j = 0; j < D::nbx_first; ++j)
        {
            const int r = lay.row_off(g_bx) + j;
            st.DC(r, D::idxb_first[j]) = 1;
            // soft box row: +1 (lo) / -1 (hi) slack columns
            const int cl = D::idxs_lo_first[r];
            if (cl >= 0)
            {
                st.DC(r, cl) = 1;
            }
            const int ch = D::idxs_hi_first[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }

    // control box
    if constexpr (D::nbu > 0)
    {
        for (int j = 0; j < D::nbu; ++j)
        {
            const int r = lay.row_off(g_bu) + j;
            st.DC(r, D::idxb_first[D::nbx_first + j]) = 1;
            // soft box row: +1 (lo) / -1 (hi) slack columns
            const int cl = D::idxs_lo_first[r];
            if (cl >= 0)
            {
                st.DC(r, cl) = 1;
            }
            const int ch = D::idxs_hi_first[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }

    // ineq (Jacobian [d/du | d/dx]; hi-side slack column -1)
    if constexpr (D::ng > 0)
    {
        const auto& g_dx = c.g_dx;
        const auto& g_du = c.g_du;
        for (int j = 0; j < D::ng; ++j)
        {
            const int r = lay.row_off(g_ineq) + j;
            for (int i = 0; i < D::nu; ++i)
            {
                st.DC(r, i) = g_du(j, i);
            }
            for (int i = 0; i < D::nx; ++i)
            {
                st.DC(r, D::nu + i) = g_dx(j, i);
            }
            const int ch = D::idxs_hi_first[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }

    // eq (lo slack column +1, hi slack column -1)
    if constexpr (D::ne > 0)
    {
        const auto& e_dx = c.e_dx;
        const auto& e_du = c.e_du;
        for (int j = 0; j < D::ne; ++j)
        {
            const int r = lay.row_off(g_eq) + j;
            for (int i = 0; i < D::nu; ++i)
            {
                st.DC(r, i) = e_du(j, i);
            }
            for (int i = 0; i < D::nx; ++i)
            {
                st.DC(r, D::nu + i) = e_dx(j, i);
            }
            const int cl = D::idxs_lo_first[r];
            if (cl >= 0)
            {
                st.DC(r, cl) = 1;
            }
            const int ch = D::idxs_hi_first[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }

    // lin
    if constexpr (D::nl > 0)
    {
        for (int j = 0; j < D::nl; ++j)
        {
            const int r = lay.row_off(g_lin) + j;
            for (int i = 0; i < D::nu; ++i)
            {
                st.DC(r, i) = lin_spec.B(j, i);
            }
            for (int i = 0; i < D::nx; ++i)
            {
                st.DC(r, D::nu + i) = lin_spec.A(j, i);
            }
            const int cl = D::idxs_lo_first[r];
            if (cl >= 0)
            {
                st.DC(r, cl) = 1;
            }
            const int ch = D::idxs_hi_first[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }
}

template <class P, class StT>
void fill_dc_path(StT& st, const StageConstr<P>& c,
                  const typename P::stage_linear_t& lin_spec)
{
    using D = QpDim<P>;
    const auto& lay = D::lay_path;

    if constexpr (D::nbx > 0)
    {
        for (int j = 0; j < D::nbx; ++j)
        {
            const int r = lay.row_off(g_bx) + j;
            st.DC(r, D::idxb_path[j]) = 1;
            // soft box row: +1 (lo) / -1 (hi) slack columns
            const int cl = D::idxs_lo_path[r];
            if (cl >= 0)
            {
                st.DC(r, cl) = 1;
            }
            const int ch = D::idxs_hi_path[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }
    if constexpr (D::nbu > 0)
    {
        for (int j = 0; j < D::nbu; ++j)
        {
            const int r = lay.row_off(g_bu) + j;
            st.DC(r, D::idxb_path[D::nbx + j]) = 1;
            // soft box row: +1 (lo) / -1 (hi) slack columns
            const int cl = D::idxs_lo_path[r];
            if (cl >= 0)
            {
                st.DC(r, cl) = 1;
            }
            const int ch = D::idxs_hi_path[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }
    if constexpr (D::ng > 0)
    {
        const auto& g_dx = c.g_dx;
        const auto& g_du = c.g_du;
        for (int j = 0; j < D::ng; ++j)
        {
            const int r = lay.row_off(g_ineq) + j;
            for (int i = 0; i < D::nu; ++i)
            {
                st.DC(r, i) = g_du(j, i);
            }
            for (int i = 0; i < D::nx; ++i)
            {
                st.DC(r, D::nu + i) = g_dx(j, i);
            }
            const int ch = D::idxs_hi_path[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }
    if constexpr (D::ne > 0)
    {
        const auto& e_dx = c.e_dx;
        const auto& e_du = c.e_du;
        for (int j = 0; j < D::ne; ++j)
        {
            const int r = lay.row_off(g_eq) + j;
            for (int i = 0; i < D::nu; ++i)
            {
                st.DC(r, i) = e_du(j, i);
            }
            for (int i = 0; i < D::nx; ++i)
            {
                st.DC(r, D::nu + i) = e_dx(j, i);
            }
            const int cl = D::idxs_lo_path[r];
            if (cl >= 0)
            {
                st.DC(r, cl) = 1;
            }
            const int ch = D::idxs_hi_path[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }
    if constexpr (D::nl > 0)
    {
        for (int j = 0; j < D::nl; ++j)
        {
            const int r = lay.row_off(g_lin) + j;
            for (int i = 0; i < D::nu; ++i)
            {
                st.DC(r, i) = lin_spec.B(j, i);
            }
            for (int i = 0; i < D::nx; ++i)
            {
                st.DC(r, D::nu + i) = lin_spec.A(j, i);
            }
            const int cl = D::idxs_lo_path[r];
            if (cl >= 0)
            {
                st.DC(r, cl) = 1;
            }
            const int ch = D::idxs_hi_path[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }
}

template <class P, class StT>
void fill_dc_term(StT& st, const TermConstr<P>& tc,
                  const typename P::term_linear_t& lin_spec)
{
    using D = QpDim<P>;
    const auto& lay = D::lay_term;

    if constexpr (D::nbx_t > 0)
    {
        for (int j = 0; j < D::nbx_t; ++j)
        {
            const int r = lay.row_off(g_bx) + j;
            st.DC(r, D::idxb_term[j]) = 1;
            // soft box row: +1 (lo) / -1 (hi) slack columns
            const int cl = D::idxs_lo_term[r];
            if (cl >= 0)
            {
                st.DC(r, cl) = 1;
            }
            const int ch = D::idxs_hi_term[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }
    if constexpr (D::ng_t > 0)
    {
        const auto& g_dx = tc.g_dx;
        for (int j = 0; j < D::ng_t; ++j)
        {
            const int r = lay.row_off(g_ineq) + j;
            for (int i = 0; i < D::nx; ++i)
            {
                st.DC(r, i) = g_dx(j, i);
            }
            const int ch = D::idxs_hi_term[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }
    if constexpr (D::ne_t > 0)
    {
        const auto& e_dx = tc.e_dx;
        for (int j = 0; j < D::ne_t; ++j)
        {
            const int r = lay.row_off(g_eq) + j;
            for (int i = 0; i < D::nx; ++i)
            {
                st.DC(r, i) = e_dx(j, i);
            }
            const int cl = D::idxs_lo_term[r];
            if (cl >= 0)
            {
                st.DC(r, cl) = 1;
            }
            const int ch = D::idxs_hi_term[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }
    if constexpr (D::nl_t > 0)
    {
        for (int j = 0; j < D::nl_t; ++j)
        {
            const int r = lay.row_off(g_lin) + j;
            for (int i = 0; i < D::nx; ++i)
            {
                st.DC(r, i) = lin_spec.A(j, i);
            }
            const int cl = D::idxs_lo_term[r];
            if (cl >= 0)
            {
                st.DC(r, cl) = 1;
            }
            const int ch = D::idxs_hi_term[r];
            if (ch >= 0)
            {
                st.DC(r, ch) = -1;
            }
        }
    }
}

// ---------------------------------------------------------------
//  d / d_mask fill (offset form, plan sec. 1.1-1.2, 2f.5)
//
//  LHS (structural): which sides are finite (d_mask).
//  RHS (per iterate): the offsets d_lo = lo - w_cur, d_hi = w_cur - hi.
// ---------------------------------------------------------------

template <class P, class StT>
void fill_d_mask_lhs_first(const P& problem, const StageEval<P>& ev, StT& st)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    const auto& lay = D::lay_first;
    constexpr int nrow = D::nrow_first;

    for (int r = 0; r < nrow; ++r)
    {
        const int g = lay.group_of(r);
        const int j = r - lay.row_off(g);
        double lo = 0.0, hi = 0.0;

        switch (g)
        {
        case g_pin:
            if constexpr (D::fixed_initial_state)
            {
                const auto x0 = problem.initial_state();
                lo = hi = static_cast<double>(x0(j));
            }
            break;
        case g_bx:
            if constexpr (D::nbx_first > 0)
            {
                const auto spec = problem.stage_state_box_constr(0);
                lo = static_cast<double>(spec.lo(j));
                hi = static_cast<double>(spec.hi(j));
            }
            break;
        case g_bu:
            if constexpr (D::nbu > 0)
            {
                const auto spec = problem.stage_control_box_constr(0);
                lo = static_cast<double>(spec.lo(j));
                hi = static_cast<double>(spec.hi(j));
            }
            break;
        case g_ineq:
            hi = 0.0;
            break;
        case g_eq:
            lo = hi = 0.0;
            break;
        default:  // g_lin
            if constexpr (D::nl > 0)
            {
                lo = static_cast<double>(ev.lin_spec.bounds.lo(j));
                hi = static_cast<double>(ev.lin_spec.bounds.hi(j));
            }
            break;
        }

        const int sl = lay.side_lo(r);
        if (sl >= 0)
        {
            st.d_mask(sl) =
                std::isfinite(lo) ? static_cast<S>(1) : static_cast<S>(0);
        }
        const int sh = lay.side_hi(r);
        st.d_mask(sh) =
            std::isfinite(hi) ? static_cast<S>(1) : static_cast<S>(0);
    }

    // slack sides: d_mask = 1
    const int lo_sz = lay.lo_size();
    for (int j = 0; j < D::nslack_first; ++j)
    {
        st.d_mask(lo_sz + nrow + j) = 1;
    }
}

template <class P, int NH, class StT>
void fill_d_rhs_first(const P& problem, const Solution<P, NH>& sol,
                      const StageEval<P>& ev, StT& st)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    const auto& x = sol.x[0];
    const auto& u = sol.u[0];
    const auto& lay = D::lay_first;
    constexpr int nrow = D::nrow_first;

    for (int r = 0; r < nrow; ++r)
    {
        const int g = lay.group_of(r);
        const int j = r - lay.row_off(g);
        double w_cur = 0.0, lo = 0.0, hi = 0.0;

        switch (g)
        {
        case g_pin:
            if constexpr (D::fixed_initial_state)
            {
                const auto x0 = problem.initial_state();
                w_cur = static_cast<double>(x(j));
                lo = hi = static_cast<double>(x0(j));
            }
            break;
        case g_bx:
            if constexpr (D::nbx_first > 0)
            {
                const auto spec = problem.stage_state_box_constr(0);
                w_cur = static_cast<double>(x(P::state_box_idx[j]));
                lo = static_cast<double>(spec.lo(j));
                hi = static_cast<double>(spec.hi(j));
            }
            break;
        case g_bu:
            if constexpr (D::nbu > 0)
            {
                const auto spec = problem.stage_control_box_constr(0);
                w_cur = static_cast<double>(u(P::control_box_idx[j]));
                lo = static_cast<double>(spec.lo(j));
                hi = static_cast<double>(spec.hi(j));
            }
            break;
        case g_ineq:
            w_cur = static_cast<double>(ev.c.g(j));
            hi = 0.0;
            break;
        case g_eq:
            w_cur = static_cast<double>(ev.c.e(j));
            lo = hi = 0.0;
            break;
        default:  // g_lin
            if constexpr (D::nl > 0)
            {
                w_cur = static_cast<double>(ev.lv(j));
                lo = static_cast<double>(ev.lin_spec.bounds.lo(j));
                hi = static_cast<double>(ev.lin_spec.bounds.hi(j));
            }
            break;
        }

        const int sl = lay.side_lo(r);
        if (sl >= 0)
        {
            // Inactive sides store d = 0 to keep all downstream
            // arithmetic finite (they are masked out anyway).
            st.d(sl) = std::isfinite(lo)
                ? static_cast<S>(lo - w_cur) : static_cast<S>(0.0);
        }
        const int sh = lay.side_hi(r);
        st.d(sh) = std::isfinite(hi)
            ? static_cast<S>(w_cur - hi) : static_cast<S>(0.0);
    }

    // slack sides: d = 0
    const int lo_sz = lay.lo_size();
    for (int j = 0; j < D::nslack_first; ++j)
    {
        st.d(lo_sz + nrow + j) = 0;
    }
}

template <class P, class StT>
void fill_d_mask_lhs_path(const P& problem, int k, const StageEval<P>& ev,
                          StT& st)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    const auto& lay = D::lay_path;
    constexpr int nrow = D::nrow_path;

    for (int r = 0; r < nrow; ++r)
    {
        const int g = lay.group_of(r);
        const int j = r - lay.row_off(g);
        double lo = 0.0, hi = 0.0;

        switch (g)
        {
        case g_bx:
            if constexpr (D::nbx > 0)
            {
                const auto spec = problem.stage_state_box_constr(k);
                lo = static_cast<double>(spec.lo(j));
                hi = static_cast<double>(spec.hi(j));
            }
            break;
        case g_bu:
            if constexpr (D::nbu > 0)
            {
                const auto spec = problem.stage_control_box_constr(k);
                lo = static_cast<double>(spec.lo(j));
                hi = static_cast<double>(spec.hi(j));
            }
            break;
        case g_ineq:
            hi = 0.0;
            break;
        case g_eq:
            lo = hi = 0.0;
            break;
        default:  // g_lin
            if constexpr (D::nl > 0)
            {
                lo = static_cast<double>(ev.lin_spec.bounds.lo(j));
                hi = static_cast<double>(ev.lin_spec.bounds.hi(j));
            }
            break;
        }

        const int sl = lay.side_lo(r);
        if (sl >= 0)
        {
            st.d_mask(sl) =
                std::isfinite(lo) ? static_cast<S>(1) : static_cast<S>(0);
        }
        const int sh = lay.side_hi(r);
        st.d_mask(sh) =
            std::isfinite(hi) ? static_cast<S>(1) : static_cast<S>(0);
    }

    // slack sides: d_mask = 1
    const int lo_sz = lay.lo_size();
    for (int j = 0; j < D::nslack_path; ++j)
    {
        st.d_mask(lo_sz + nrow + j) = 1;
    }
}

template <class P, int NH, class StT>
void fill_d_rhs_path(const P& problem, const Solution<P, NH>& sol, int k,
                     const StageEval<P>& ev, StT& st)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    const auto& x = sol.x[k];
    const auto& u = sol.u[k];
    const auto& lay = D::lay_path;
    constexpr int nrow = D::nrow_path;

    for (int r = 0; r < nrow; ++r)
    {
        const int g = lay.group_of(r);
        const int j = r - lay.row_off(g);
        double w_cur = 0.0, lo = 0.0, hi = 0.0;

        switch (g)
        {
        case g_bx:
            if constexpr (D::nbx > 0)
            {
                const auto spec = problem.stage_state_box_constr(k);
                w_cur = static_cast<double>(x(P::state_box_idx[j]));
                lo = static_cast<double>(spec.lo(j));
                hi = static_cast<double>(spec.hi(j));
            }
            break;
        case g_bu:
            if constexpr (D::nbu > 0)
            {
                const auto spec = problem.stage_control_box_constr(k);
                w_cur = static_cast<double>(u(P::control_box_idx[j]));
                lo = static_cast<double>(spec.lo(j));
                hi = static_cast<double>(spec.hi(j));
            }
            break;
        case g_ineq:
            w_cur = static_cast<double>(ev.c.g(j));
            hi = 0.0;
            break;
        case g_eq:
            w_cur = static_cast<double>(ev.c.e(j));
            lo = hi = 0.0;
            break;
        default:  // g_lin
            if constexpr (D::nl > 0)
            {
                w_cur = static_cast<double>(ev.lv(j));
                lo = static_cast<double>(ev.lin_spec.bounds.lo(j));
                hi = static_cast<double>(ev.lin_spec.bounds.hi(j));
            }
            break;
        }

        const int sl = lay.side_lo(r);
        if (sl >= 0)
        {
            // Inactive sides store d = 0 to keep all downstream
            // arithmetic finite (they are masked out anyway).
            st.d(sl) = std::isfinite(lo)
                ? static_cast<S>(lo - w_cur) : static_cast<S>(0.0);
        }
        const int sh = lay.side_hi(r);
        st.d(sh) = std::isfinite(hi)
            ? static_cast<S>(w_cur - hi) : static_cast<S>(0.0);
    }

    // slack sides: d = 0
    const int lo_sz = lay.lo_size();
    for (int j = 0; j < D::nslack_path; ++j)
    {
        st.d(lo_sz + nrow + j) = 0;
    }
}

template <class P, class StT>
void fill_d_mask_lhs_term(const P& problem, const TermEval<P>& ev, StT& st)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    const auto& lay = D::lay_term;
    constexpr int nrow = D::nrow_term;

    for (int r = 0; r < nrow; ++r)
    {
        const int g = lay.group_of(r);
        const int j = r - lay.row_off(g);
        double lo = 0.0, hi = 0.0;

        switch (g)
        {
        case g_bx:
            if constexpr (D::nbx_t > 0)
            {
                const auto spec = problem.terminal_state_box_constr();
                lo = static_cast<double>(spec.lo(j));
                hi = static_cast<double>(spec.hi(j));
            }
            break;
        case g_ineq:
            hi = 0.0;
            break;
        case g_eq:
            lo = hi = 0.0;
            break;
        default:  // g_lin
            if constexpr (D::nl_t > 0)
            {
                lo = static_cast<double>(ev.lin_spec.bounds.lo(j));
                hi = static_cast<double>(ev.lin_spec.bounds.hi(j));
            }
            break;
        }

        const int sl = lay.side_lo(r);
        if (sl >= 0)
        {
            st.d_mask(sl) =
                std::isfinite(lo) ? static_cast<S>(1) : static_cast<S>(0);
        }
        const int sh = lay.side_hi(r);
        st.d_mask(sh) =
            std::isfinite(hi) ? static_cast<S>(1) : static_cast<S>(0);
    }

    // slack sides: d_mask = 1
    const int lo_sz = lay.lo_size();
    for (int j = 0; j < D::nslack_term; ++j)
    {
        st.d_mask(lo_sz + nrow + j) = 1;
    }
}

template <class P, int NH, class StT>
void fill_d_rhs_term(const P& problem, const Solution<P, NH>& sol,
                     const TermEval<P>& ev, StT& st)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    const int N = sol.N;
    const auto& x = sol.x[N];
    const auto& lay = D::lay_term;
    constexpr int nrow = D::nrow_term;

    for (int r = 0; r < nrow; ++r)
    {
        const int g = lay.group_of(r);
        const int j = r - lay.row_off(g);
        double w_cur = 0.0, lo = 0.0, hi = 0.0;

        switch (g)
        {
        case g_bx:
            if constexpr (D::nbx_t > 0)
            {
                const auto spec = problem.terminal_state_box_constr();
                w_cur = static_cast<double>(
                    x(P::terminal_state_box_idx[j]));
                lo = static_cast<double>(spec.lo(j));
                hi = static_cast<double>(spec.hi(j));
            }
            break;
        case g_ineq:
            w_cur = static_cast<double>(ev.tc.g(j));
            hi = 0.0;
            break;
        case g_eq:
            w_cur = static_cast<double>(ev.tc.e(j));
            lo = hi = 0.0;
            break;
        default:  // g_lin
            if constexpr (D::nl_t > 0)
            {
                w_cur = static_cast<double>(ev.lv(j));
                lo = static_cast<double>(ev.lin_spec.bounds.lo(j));
                hi = static_cast<double>(ev.lin_spec.bounds.hi(j));
            }
            break;
        }

        const int sl = lay.side_lo(r);
        if (sl >= 0)
        {
            st.d(sl) = std::isfinite(lo)
                ? static_cast<S>(lo - w_cur) : static_cast<S>(0.0);
        }
        const int sh = lay.side_hi(r);
        st.d(sh) = std::isfinite(hi)
            ? static_cast<S>(w_cur - hi) : static_cast<S>(0.0);
    }

    // slack sides: d = 0
    const int lo_sz = lay.lo_size();
    for (int j = 0; j < D::nslack_term; ++j)
    {
        st.d(lo_sz + nrow + j) = 0;
    }
}

// ---------------------------------------------------------------
//  Slack Hessian diagonal / linear term (soft rows: weight w and
//  penalty r; unused slack slots: hess 1.0, grad 0)
// ---------------------------------------------------------------

template <class P, class HessT>
void fill_slack_diag_lhs(HessT& hess, const P& problem, bool is_first, int k)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    constexpr int nux = D::nu + D::nx;
    const int nslack = is_first ? D::nslack_first : D::nslack_path;
    if (nslack == 0)
    {
        return;
    }

    for (int j = 0; j < nslack; ++j)
    {
        hess(nux + j, nux + j) = 1.0;
    }

    // bx lo / bx hi
    if constexpr (P::nbx_soft > 0)
    {
        const int nbx_s =
            (P::fixed_initial_state && is_first) ? 0 : P::nbx_soft;
        const auto spec = problem.stage_state_box_constr(k);
        for (int i = 0; i < nbx_s; ++i)
        {
            const int row = P::state_box_soft_idx[i];
            const double w = static_cast<double>(spec.penalty.quad(row));
            hess(nux + i, nux + i) = static_cast<S>(w);
            const int col = nux + nbx_s + i;
            hess(col, col) = static_cast<S>(w);
        }
    }
    // bu lo / bu hi
    if constexpr (P::nbu_soft > 0)
    {
        const int off_bx =
            (is_first && P::fixed_initial_state) ? 0 : P::nbx_soft;
        const auto spec = problem.stage_control_box_constr(k);
        for (int i = 0; i < P::nbu_soft; ++i)
        {
            const int row = P::control_box_soft_idx[i];
            const double w = static_cast<double>(spec.penalty.quad(row));
            hess(nux + 2 * off_bx + i,
                 nux + 2 * off_bx + i) = static_cast<S>(w);
            hess(nux + 2 * off_bx + P::nbu_soft + i,
                 nux + 2 * off_bx + P::nbu_soft + i) = static_cast<S>(w);
        }
    }
    // ineq hi
    if constexpr (P::ng_soft > 0)
    {
        const int off = 2 * ((is_first && P::fixed_initial_state)
                                 ? 0
                                 : P::nbx_soft)
                       + 2 * P::nbu_soft;
        const auto pen = problem.stage_inequality_constr_soft_penalty(k);
        for (int i = 0; i < P::ng_soft; ++i)
        {
            const int col = nux + off + i;
            hess(col, col) = static_cast<S>(static_cast<double>(pen.quad(i)));
        }
    }
    // eq lo / eq hi
    if constexpr (P::ne_soft > 0)
    {
        const int off = 2 * ((is_first && P::fixed_initial_state)
                                 ? 0
                                 : P::nbx_soft)
                       + 2 * P::nbu_soft + P::ng_soft;
        const auto pen = problem.stage_equality_constr_soft_penalty(k);
        for (int i = 0; i < P::ne_soft; ++i)
        {
            const double w = static_cast<double>(pen.quad(i));
            hess(nux + off + i, nux + off + i) = static_cast<S>(w);
            hess(nux + off + P::ne_soft + i,
                 nux + off + P::ne_soft + i) = static_cast<S>(w);
        }
    }
    // lin lo / lin hi
    if constexpr (P::nl_soft > 0)
    {
        const int off = 2 * ((is_first && P::fixed_initial_state)
                                 ? 0
                                 : P::nbx_soft)
                       + 2 * P::nbu_soft + P::ng_soft
                       + 2 * P::ne_soft;
        const auto spec = problem.stage_linear_constr(k);
        for (int i = 0; i < P::nl_soft; ++i)
        {
            const int row = P::lin_soft_idx[i];
            const double w = static_cast<double>(spec.bounds.penalty.quad(row));
            hess(nux + off + i, nux + off + i) = static_cast<S>(w);
            hess(nux + off + P::nl_soft + i,
                 nux + off + P::nl_soft + i) = static_cast<S>(w);
        }
    }
}

template <class P, class GradT>
void fill_slack_diag_rhs(GradT& grad, const P& problem, bool is_first, int k)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    constexpr int nux = D::nu + D::nx;
    const int nslack = is_first ? D::nslack_first : D::nslack_path;
    if (nslack == 0)
    {
        return;
    }

    // bx lo / bx hi
    if constexpr (P::nbx_soft > 0)
    {
        const int nbx_s =
            (P::fixed_initial_state && is_first) ? 0 : P::nbx_soft;
        const auto spec = problem.stage_state_box_constr(k);
        for (int i = 0; i < nbx_s; ++i)
        {
            const int row = P::state_box_soft_idx[i];
            const double r = static_cast<double>(spec.penalty.lin(row));
            grad(nux + i) = static_cast<S>(r);
            const int col = nux + nbx_s + i;
            grad(col) = static_cast<S>(r);
        }
    }
    // bu lo / bu hi
    if constexpr (P::nbu_soft > 0)
    {
        const int off_bx =
            (is_first && P::fixed_initial_state) ? 0 : P::nbx_soft;
        const auto spec = problem.stage_control_box_constr(k);
        for (int i = 0; i < P::nbu_soft; ++i)
        {
            const int row = P::control_box_soft_idx[i];
            const double r = static_cast<double>(spec.penalty.lin(row));
            grad(nux + 2 * off_bx + i) = static_cast<S>(r);
            grad(nux + 2 * off_bx + P::nbu_soft + i) = static_cast<S>(r);
        }
    }
    // ineq hi
    if constexpr (P::ng_soft > 0)
    {
        const int off = 2 * ((is_first && P::fixed_initial_state)
                                 ? 0
                                 : P::nbx_soft)
                       + 2 * P::nbu_soft;
        const auto pen = problem.stage_inequality_constr_soft_penalty(k);
        for (int i = 0; i < P::ng_soft; ++i)
        {
            const int col = nux + off + i;
            grad(col) = static_cast<S>(static_cast<double>(pen.lin(i)));
        }
    }
    // eq lo / eq hi
    if constexpr (P::ne_soft > 0)
    {
        const int off = 2 * ((is_first && P::fixed_initial_state)
                                 ? 0
                                 : P::nbx_soft)
                       + 2 * P::nbu_soft + P::ng_soft;
        const auto pen = problem.stage_equality_constr_soft_penalty(k);
        for (int i = 0; i < P::ne_soft; ++i)
        {
            const double r = static_cast<double>(pen.lin(i));
            grad(nux + off + i) = static_cast<S>(r);
            grad(nux + off + P::ne_soft + i) = static_cast<S>(r);
        }
    }
    // lin lo / lin hi
    if constexpr (P::nl_soft > 0)
    {
        const int off = 2 * ((is_first && P::fixed_initial_state)
                                 ? 0
                                 : P::nbx_soft)
                       + 2 * P::nbu_soft + P::ng_soft
                       + 2 * P::ne_soft;
        const auto spec = problem.stage_linear_constr(k);
        for (int i = 0; i < P::nl_soft; ++i)
        {
            const int row = P::lin_soft_idx[i];
            const double r = static_cast<double>(spec.bounds.penalty.lin(row));
            grad(nux + off + i) = static_cast<S>(r);
            grad(nux + off + P::nl_soft + i) = static_cast<S>(r);
        }
    }
}

template <class P, class HessT>
void fill_slack_diag_term_lhs(HessT& hess, const P& problem)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    constexpr int nx = D::nx;
    const int nslack = D::nslack_term;
    if (nslack == 0)
    {
        return;
    }

    for (int j = 0; j < nslack; ++j)
    {
        hess(nx + j, nx + j) = 1.0;
    }

    if constexpr (P::nbx_t_soft > 0)
    {
        const auto spec = problem.terminal_state_box_constr();
        for (int i = 0; i < P::nbx_t_soft; ++i)
        {
            const int row = P::terminal_state_box_soft_idx[i];
            const double w = static_cast<double>(spec.penalty.quad(row));
            hess(nx + i, nx + i) = static_cast<S>(w);
            hess(nx + P::nbx_t_soft + i,
                 nx + P::nbx_t_soft + i) = static_cast<S>(w);
        }
    }
    if constexpr (P::ng_t_soft > 0)
    {
        const auto pen = problem.terminal_inequality_constr_soft_penalty();
        const int off = 2 * P::nbx_t_soft;
        for (int i = 0; i < P::ng_t_soft; ++i)
        {
            hess(nx + off + i, nx + off + i) =
                static_cast<S>(static_cast<double>(pen.quad(i)));
        }
    }
    if constexpr (P::ne_t_soft > 0)
    {
        const auto pen = problem.terminal_equality_constr_soft_penalty();
        const int off = 2 * P::nbx_t_soft + P::ng_t_soft;
        for (int i = 0; i < P::ne_t_soft; ++i)
        {
            const double w = static_cast<double>(pen.quad(i));
            hess(nx + off + i, nx + off + i) = static_cast<S>(w);
            hess(nx + off + P::ne_t_soft + i,
                 nx + off + P::ne_t_soft + i) = static_cast<S>(w);
        }
    }
    if constexpr (P::nl_t_soft > 0)
    {
        const auto spec = problem.terminal_linear_constr();
        const int off = 2 * P::nbx_t_soft + P::ng_t_soft
                       + 2 * P::ne_t_soft;
        for (int i = 0; i < P::nl_t_soft; ++i)
        {
            const int row = P::terminal_lin_soft_idx[i];
            const double w =
                static_cast<double>(spec.bounds.penalty.quad(row));
            hess(nx + off + i, nx + off + i) = static_cast<S>(w);
            hess(nx + off + P::nl_t_soft + i,
                 nx + off + P::nl_t_soft + i) = static_cast<S>(w);
        }
    }
}

template <class P, class GradT>
void fill_slack_diag_term_rhs(GradT& grad, const P& problem)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    constexpr int nx = D::nx;
    const int nslack = D::nslack_term;
    if (nslack == 0)
    {
        return;
    }

    if constexpr (P::nbx_t_soft > 0)
    {
        const auto spec = problem.terminal_state_box_constr();
        for (int i = 0; i < P::nbx_t_soft; ++i)
        {
            const int row = P::terminal_state_box_soft_idx[i];
            const double r = static_cast<double>(spec.penalty.lin(row));
            grad(nx + i) = static_cast<S>(r);
            grad(nx + P::nbx_t_soft + i) = static_cast<S>(r);
        }
    }
    if constexpr (P::ng_t_soft > 0)
    {
        const auto pen = problem.terminal_inequality_constr_soft_penalty();
        const int off = 2 * P::nbx_t_soft;
        for (int i = 0; i < P::ng_t_soft; ++i)
        {
            grad(nx + off + i) = static_cast<S>(static_cast<double>(pen.lin(i)));
        }
    }
    if constexpr (P::ne_t_soft > 0)
    {
        const auto pen = problem.terminal_equality_constr_soft_penalty();
        const int off = 2 * P::nbx_t_soft + P::ng_t_soft;
        for (int i = 0; i < P::ne_t_soft; ++i)
        {
            const double r = static_cast<double>(pen.lin(i));
            grad(nx + off + i) = static_cast<S>(r);
            grad(nx + off + P::ne_t_soft + i) = static_cast<S>(r);
        }
    }
    if constexpr (P::nl_t_soft > 0)
    {
        const auto spec = problem.terminal_linear_constr();
        const int off = 2 * P::nbx_t_soft + P::ng_t_soft
                       + 2 * P::ne_t_soft;
        for (int i = 0; i < P::nl_t_soft; ++i)
        {
            const int row = P::terminal_lin_soft_idx[i];
            const double r =
                static_cast<double>(spec.bounds.penalty.lin(row));
            grad(nx + off + i) = static_cast<S>(r);
            grad(nx + off + P::nl_t_soft + i) = static_cast<S>(r);
        }
    }
}

// ---------------------------------------------------------------
//  First-stage degeneracy check (plan sec. 2f.6)
// ---------------------------------------------------------------

/// A first-stage ineq/lin row that is violated at the iterate and whose
/// Jacobian w.r.t. the free variables (u_0; s_0) is identically zero
/// cannot be satisfied by any step (x_0 is pinned, the row depends only
/// on it): the problem is infeasible.
template <class P, int NH>
Status check_first_degeneracy(const P& problem, const Solution<P, NH>& sol)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    const auto& x = sol.x[0];
    const auto& u = sol.u[0];
    const auto& lay = D::lay_first;

    if constexpr (D::ng > 0)
    {
        typename P::ineq_t g;
        typename P::ineq_dg_dx_t g_dx;
        typename P::ineq_dg_du_t g_du;
        problem.stage_inequality_value_jac(0, x, u, g, g_dx, g_du);
        for (int j = 0; j < D::ng; ++j)
        {
            if (static_cast<double>(g(j)) > 0.0)
            {
                bool jac_zero = true;
                for (int i = 0; i < D::nu; ++i)
                {
                    if (g_du(j, i) != S(0))
                    {
                        jac_zero = false;
                    }
                }
                const int r = lay.row_off(g_ineq) + j;
                const int ch = D::idxs_hi_first[r];
                if (jac_zero && ch < 0)
                {
                    return Status::kInfeasible;
                }
            }
        }
    }

    if constexpr (D::nl > 0)
    {
        const auto spec = problem.stage_linear_constr(0);
        for (int j = 0; j < D::nl; ++j)
        {
            double val = 0.0;
            for (int i = 0; i < D::nx; ++i)
            {
                val += static_cast<double>(spec.A(j, i))
                     * static_cast<double>(x(i));
            }
            for (int i = 0; i < D::nu; ++i)
            {
                val += static_cast<double>(spec.B(j, i))
                     * static_cast<double>(u(i));
            }
            const double lo = static_cast<double>(spec.bounds.lo(j));
            const double hi = static_cast<double>(spec.bounds.hi(j));
            const int r = lay.row_off(g_lin) + j;
            const bool lo_hard = D::idxs_lo_first[r] < 0;
            const bool hi_hard = D::idxs_hi_first[r] < 0;
            if ((val > hi && hi_hard) || (val < lo && lo_hard))
            {
                // violated side is hard and the row is independent of
                // the free variables (u_0; s_0): no step can fix it
                bool jac_zero = true;
                for (int i = 0; i < D::nu; ++i)
                {
                    if (spec.B(j, i) != S(0))
                    {
                        jac_zero = false;
                    }
                }
                if (jac_zero)
                {
                    return Status::kInfeasible;
                }
            }
        }
    }

    return Status::kSolved;
}

// ---------------------------------------------------------------
//  Per-stage composite LHS / RHS fills
// ---------------------------------------------------------------

/// LHS of the first stage: hess (permuted cost Hessian + contracted
/// Hessians + slack diagonal), BA, DC, d_mask.
template <class P, class StT>
void fill_stage_lhs_first(const P& problem, const StageEval<P>& ev,
                          const AssembleFlags& fl, StT& st)
{
    using D = QpDim<P>;
    constexpr int nux = D::nu + D::nx;

    // hess: element-wise; block extraction from the small local H trips
    // a GCC 13 -Warray-bounds false positive on small matrices
    for (int i = 0; i < nux; ++i)
    {
        for (int j = 0; j < nux; ++j)
        {
            const int pi = (i < D::nu) ? D::nx + i : i - D::nu;
            const int pj = (j < D::nu) ? D::nx + j : j - D::nu;
            st.hess(i, j) = ev.H(pi, pj);
        }
    }

    if (fl.compute_hess_dyn)
    {
        if constexpr (P::has_dynamics_hess)
            add_permuted_hess<P>(st.hess, ev.M_dyn, -1.0);
    }
    if (fl.compute_hess_constr)
    {
        if constexpr (P::has_constr_hess)
        {
            if constexpr (D::ng > 0)
                add_permuted_hess<P>(st.hess, ev.M_ineq, +1.0);
            if constexpr (D::ne > 0)
                add_permuted_hess<P>(st.hess, ev.M_eq, +1.0);
        }
    }
    fill_slack_diag_lhs<P>(st.hess, problem, /*is_first=*/true, /*k=*/0);

    // BA = [B | A]
    for (int i = 0; i < D::nx; ++i)
    {
        for (int j = 0; j < D::nu; ++j)
        {
            st.BA(i, j) = ev.B(i, j);
        }
        for (int j = 0; j < D::nx; ++j)
        {
            st.BA(i, D::nu + j) = ev.A(i, j);
        }
    }

    fill_dc_first<P>(st, ev.c, ev.lin_spec);
    fill_d_mask_lhs_first<P>(problem, ev, st);
}

/// RHS of the first stage: grad (permuted cost gradient + slack linear
/// part), b, d offsets, trivial-row masking.  Returns kInfeasible when a
/// degenerate fixed-x_0 row is violated (see check_first_degeneracy).
template <class P, int NH, class StT>
Status fill_stage_rhs_first(const P& problem, const Solution<P, NH>& sol,
                            const StageEval<P>& ev, StT& st)
{
    using D = QpDim<P>;
    constexpr int nux = D::nu + D::nx;

    // grad: [g_u; g_x], slack part filled by fill_slack_diag_rhs
    for (int i = 0; i < nux; ++i)
    {
        st.grad(i) = ev.g_cost((i < D::nu) ? D::nx + i : i - D::nu);
    }
    fill_slack_diag_rhs<P>(st.grad, problem, /*is_first=*/true, /*k=*/0);

    // b = f(x_0, u_0) - x_1
    for (int i = 0; i < D::nx; ++i)
    {
        st.b(i) = ev.f(i) - sol.x[1](i);
    }

    fill_d_rhs_first<P, NH>(problem, sol, ev, st);
    mask_trivial_zero_rows(D::lay_first, D::nu + D::nx, st);

    // degenerate first-stage check (plan sec. 2f.6)
    if constexpr (P::fixed_initial_state)
    {
        return check_first_degeneracy(problem, sol);
    }
    return Status::kSolved;
}

/// LHS of a path stage (k >= 1): hess, BA, DC, d_mask.
template <class P, class StT>
void fill_stage_lhs_path(const P& problem, int k, const StageEval<P>& ev,
                         const AssembleFlags& fl, StT& st)
{
    using D = QpDim<P>;
    constexpr int nux = D::nu + D::nx;

    for (int i = 0; i < nux; ++i)
    {
        for (int j = 0; j < nux; ++j)
        {
            const int pi = (i < D::nu) ? D::nx + i : i - D::nu;
            const int pj = (j < D::nu) ? D::nx + j : j - D::nu;
            st.hess(i, j) = ev.H(pi, pj);
        }
    }

    if (fl.compute_hess_dyn)
    {
        if constexpr (P::has_dynamics_hess)
            add_permuted_hess<P>(st.hess, ev.M_dyn, -1.0);
    }
    if (fl.compute_hess_constr)
    {
        if constexpr (P::has_constr_hess)
        {
            if constexpr (D::ng > 0)
                add_permuted_hess<P>(st.hess, ev.M_ineq, +1.0);
            if constexpr (D::ne > 0)
                add_permuted_hess<P>(st.hess, ev.M_eq, +1.0);
        }
    }
    fill_slack_diag_lhs<P>(st.hess, problem, /*is_first=*/false, k);

    // BA = [B | A]
    for (int i = 0; i < D::nx; ++i)
    {
        for (int j = 0; j < D::nu; ++j)
        {
            st.BA(i, j) = ev.B(i, j);
        }
        for (int j = 0; j < D::nx; ++j)
        {
            st.BA(i, D::nu + j) = ev.A(i, j);
        }
    }

    fill_dc_path<P>(st, ev.c, ev.lin_spec);
    fill_d_mask_lhs_path<P>(problem, k, ev, st);
}

/// RHS of a path stage (k >= 1): grad, b = f(x_k, u_k) - x_{k+1}, d
/// offsets, trivial-row masking.
template <class P, int NH, class StT>
void fill_stage_rhs_path(const P& problem, const Solution<P, NH>& sol, int k,
                         const StageEval<P>& ev, StT& st)
{
    using D = QpDim<P>;
    constexpr int nux = D::nu + D::nx;

    for (int i = 0; i < nux; ++i)
    {
        st.grad(i) = ev.g_cost((i < D::nu) ? D::nx + i : i - D::nu);
    }
    fill_slack_diag_rhs<P>(st.grad, problem, /*is_first=*/false, k);

    for (int i = 0; i < D::nx; ++i)
    {
        st.b(i) = ev.f(i) - sol.x[k + 1](i);
    }

    fill_d_rhs_path<P, NH>(problem, sol, k, ev, st);
    mask_trivial_zero_rows(D::lay_path, D::nu + D::nx, st);
}

/// LHS of the terminal stage: hess (terminal cost Hessian + contracted
/// Hessians + slack diagonal), DC, d_mask.
template <class P, class StT>
void fill_stage_lhs_term(const P& problem, const TermEval<P>& ev,
                         const AssembleFlags& fl, StT& st)
{
    using D = QpDim<P>;
    constexpr int nx = D::nx;

    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < nx; ++j)
        {
            st.hess(i, j) = ev.Ht(i, j);
        }
    }
    if (fl.compute_hess_constr)
    {
        if constexpr (P::has_constr_hess)
        {
            // terminal constraint Hessians are state-only (nx x nx)
            // and match the QP terminal layout; no permutation
            if constexpr (D::ng_t > 0)
            {
                for (int i = 0; i < nx; ++i)
                {
                    for (int j = 0; j < nx; ++j)
                    {
                        st.hess(i, j) += ev.M_ineq(i, j);
                    }
                }
            }
            if constexpr (D::ne_t > 0)
            {
                for (int i = 0; i < nx; ++i)
                {
                    for (int j = 0; j < nx; ++j)
                    {
                        st.hess(i, j) += ev.M_eq(i, j);
                    }
                }
            }
        }
    }
    fill_slack_diag_term_lhs<P>(st.hess, problem);

    fill_dc_term<P>(st, ev.tc, ev.lin_spec);
    fill_d_mask_lhs_term<P>(problem, ev, st);
}

/// RHS of the terminal stage: grad, d offsets, trivial-row masking.
template <class P, int NH, class StT>
void fill_stage_rhs_term(const P& problem, const Solution<P, NH>& sol,
                         const TermEval<P>& ev, StT& st)
{
    using D = QpDim<P>;
    constexpr int nx = D::nx;

    for (int i = 0; i < nx; ++i)
    {
        st.grad(i) = ev.g_term(i);
    }
    fill_slack_diag_term_rhs<P>(st.grad, problem);

    fill_d_rhs_term<P, NH>(problem, sol, ev, st);
    mask_trivial_zero_rows(D::lay_term, D::nx, st);
}

// ---------------------------------------------------------------
//  Trivial-row masking (LHS post-processing, needs the RHS d offsets)
// ---------------------------------------------------------------

/// Mask general rows (ineq / lin) that are numerically invisible to the
/// QP: Jacobian row (u;x part) all ~0 and both sides trivially satisfied
/// (d_lo <= 0, d_hi <= 0 at the current iterate). Such a row contributes
/// nothing to the KKT matrix (C' diag(Gamma) C = 0), the right-hand side,
/// or the dual gap, so masking is QP-equivalent. It matters when the row
/// is active at the linearization point (d_lo == 0 with a zero Jacobian,
/// e.g. a nonlinear constraint whose gradient vanishes there): the IPM
/// then pins t at the floor while lam -> inf, stalls on that
/// complementarity pair, and leaves a garbage dual that pollutes the SQP
/// merit weights.
template <class StT>
void mask_trivial_zero_rows(QpLayout lay, int nux, StT& st)
{
    constexpr double kTol = 1e-12;
    const auto& dc = st.DC;
    const int nrow = static_cast<int>(dc.rows());
    for (int r = 0; r < nrow; ++r)
    {
        const int g = lay.group_of(r);
        if (g != g_ineq && g != g_lin)
        {
            continue;
        }
        double mx = 0.0;
        for (int j = 0; j < nux; ++j)
        {
            mx = std::max(mx,
                          std::fabs(static_cast<double>(dc(r, j))));
        }
        if (mx > kTol)
        {
            continue;
        }
        const int sl = lay.side_lo(r);
        const int sh = lay.side_hi(r);
        const bool lo_ok =
            (sl < 0) || (static_cast<double>(st.d(sl)) <= kTol);
        const bool hi_ok = static_cast<double>(st.d(sh)) <= kTol;
        if (lo_ok && hi_ok)
        {
            if (sl >= 0)
            {
                st.d_mask(sl) = 0;
            }
            st.d_mask(sh) = 0;
        }
    }
}

}  // namespace detail

}  // namespace ocp

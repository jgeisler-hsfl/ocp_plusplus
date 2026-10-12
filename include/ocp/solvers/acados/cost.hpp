// ocp/solvers/acados/cost.hpp
//
// NLP objective computation shared by the SQP drivers (phase 3 block 3b):
// the problem's stage + terminal costs plus the solver-internal slack
// penalty. Extracted from AcadosSolver (private raw_cost / slack_penalty /
// compute_cost / slack_pair) so AcadosSolver and RtiSolver cost an iterate
// identically.
//
// The problem's cost functions exclude the solver-internal slacks; the
// penalty 0.5 * w * s^2 + r * s is added here per soft side (plan
// sec. 1.6).

#pragma once

#include "../hpipm/qp.hpp"
#include "globalize.hpp"  // SqpSlacks

namespace ocp
{

namespace detail
{

/// (0.5 * w * s^2 + r * s) over one soft row's slack columns
/// (absent sides, col < 0, contribute nothing).
template <class SV>
double slack_pair(const SV& sv, int base, int col_lo, int col_hi,
                  double w, double r)
{
    double m = 0.0;
    if (col_lo >= 0)
    {
        const double s = static_cast<double>(sv[col_lo - base]);
        m += 0.5 * w * s * s + r * s;
    }
    if (col_hi >= 0)
    {
        const double s = static_cast<double>(sv[col_hi - base]);
        m += 0.5 * w * s * s + r * s;
    }
    return m;
}

}  // namespace detail

/// Raw NLP objective: the problem's stage + terminal costs only,
/// excluding the solver-internal slack penalty.
template <class P, int NH>
double raw_cost(const P& problem, const Solution<P, NH>& sol)
{
    double m = 0.0;
    for (int k = 0; k < sol.N; ++k)
    {
        m += static_cast<double>(
            problem.stage_cost_value(k, sol.x[k], sol.u[k]));
    }
    m += static_cast<double>(problem.terminal_cost_value(sol.x[sol.N]));
    return m;
}

/// Slack penalty 0.5 * w * s^2 + r * s over every soft side (same
/// weights and slack order as fill_slack_diag /
/// fill_slack_diag_term; the first
/// stage drops the state-box slacks when x_0 is fixed, so the bx lo
/// block is absent there — guarded via the stage-type layout).
template <class P, int NH>
double slack_penalty(const P& problem, const SqpSlacks<P, NH>& sl)
{
    using D = QpDim<P>;
    double m = 0.0;

    // first / path stages share the stage specs
    const auto stage_pen = [&](int k, const auto& sv, int base,
                               const auto& idxl, const auto& idxh,
                               const auto& lay)
    {
        if constexpr (P::nbx_soft > 0)
        {
            if (lay.rows[detail::g_bx] > 0)
            {
                const auto spec = problem.stage_state_box_constr(k);
                for (int i = 0; i < P::nbx_soft; ++i)
                {
                    const int row = P::state_box_soft_idx[i];
                    const int r = lay.row_off(detail::g_bx) + row;
                    m += detail::slack_pair(
                        sv, base, idxl[r], idxh[r],
                        static_cast<double>(spec.penalty.quad(row)),
                        static_cast<double>(spec.penalty.lin(row)));
                }
            }
        }
        if constexpr (P::nbu_soft > 0)
        {
            const auto spec = problem.stage_control_box_constr(k);
            for (int i = 0; i < P::nbu_soft; ++i)
            {
                const int row = P::control_box_soft_idx[i];
                const int r = lay.row_off(detail::g_bu) + row;
                m += detail::slack_pair(
                    sv, base, idxl[r], idxh[r],
                    static_cast<double>(spec.penalty.quad(row)),
                    static_cast<double>(spec.penalty.lin(row)));
            }
        }
        if constexpr (P::ng_soft > 0)
        {
            const auto pen =
                problem.stage_inequality_constr_soft_penalty(k);
            for (int i = 0; i < P::ng_soft; ++i)
            {
                const int r =
                    lay.row_off(detail::g_ineq) + P::ineq_soft_idx[i];
                m += detail::slack_pair(sv, base, idxl[r], idxh[r],
                                        static_cast<double>(pen.quad(i)),
                                        static_cast<double>(pen.lin(i)));
            }
        }
        if constexpr (P::ne_soft > 0)
        {
            const auto pen =
                problem.stage_equality_constr_soft_penalty(k);
            for (int i = 0; i < P::ne_soft; ++i)
            {
                const int r =
                    lay.row_off(detail::g_eq) + P::eq_soft_idx[i];
                m += detail::slack_pair(sv, base, idxl[r], idxh[r],
                                        static_cast<double>(pen.quad(i)),
                                        static_cast<double>(pen.lin(i)));
            }
        }
        if constexpr (P::nl_soft > 0)
        {
            const auto spec = problem.stage_linear_constr(k);
            for (int i = 0; i < P::nl_soft; ++i)
            {
                const int row = P::lin_soft_idx[i];
                const int r = lay.row_off(detail::g_lin) + row;
                m += detail::slack_pair(
                    sv, base, idxl[r], idxh[r],
                    static_cast<double>(spec.bounds.penalty.quad(row)),
                    static_cast<double>(spec.bounds.penalty.lin(row)));
            }
        }
    };

    stage_pen(0, sl.first, D::nvar_first - D::nslack_first,
              D::idxs_lo_first, D::idxs_hi_first, D::lay_first);
    for (int k = 1; k < sl.N; ++k)
    {
        stage_pen(k, sl.path[k - 1], D::nvar_path - D::nslack_path,
                  D::idxs_lo_path, D::idxs_hi_path, D::lay_path);
    }

    // terminal
    {
        const auto& sv = sl.term;
        const int base = D::nvar_term - D::nslack_term;
        const auto& lay = D::lay_term;
        if constexpr (P::nbx_t_soft > 0)
        {
            const auto spec = problem.terminal_state_box_constr();
            for (int i = 0; i < P::nbx_t_soft; ++i)
            {
                const int row = P::terminal_state_box_soft_idx[i];
                const int r = lay.row_off(detail::g_bx) + row;
                m += detail::slack_pair(
                    sv, base, D::idxs_lo_term[r], D::idxs_hi_term[r],
                    static_cast<double>(spec.penalty.quad(row)),
                    static_cast<double>(spec.penalty.lin(row)));
            }
        }
        if constexpr (P::ng_t_soft > 0)
        {
            const auto pen =
                problem.terminal_inequality_constr_soft_penalty();
            for (int i = 0; i < P::ng_t_soft; ++i)
            {
                const int r = lay.row_off(detail::g_ineq)
                             + P::terminal_ineq_soft_idx[i];
                m += detail::slack_pair(sv, base, D::idxs_lo_term[r],
                                        D::idxs_hi_term[r],
                                        static_cast<double>(pen.quad(i)),
                                        static_cast<double>(pen.lin(i)));
            }
        }
        if constexpr (P::ne_t_soft > 0)
        {
            const auto pen =
                problem.terminal_equality_constr_soft_penalty();
            for (int i = 0; i < P::ne_t_soft; ++i)
            {
                const int r = lay.row_off(detail::g_eq)
                             + P::terminal_eq_soft_idx[i];
                m += detail::slack_pair(sv, base, D::idxs_lo_term[r],
                                        D::idxs_hi_term[r],
                                        static_cast<double>(pen.quad(i)),
                                        static_cast<double>(pen.lin(i)));
            }
        }
        if constexpr (P::nl_t_soft > 0)
        {
            const auto spec = problem.terminal_linear_constr();
            for (int i = 0; i < P::nl_t_soft; ++i)
            {
                const int row = P::terminal_lin_soft_idx[i];
                const int r = lay.row_off(detail::g_lin) + row;
                m += detail::slack_pair(
                    sv, base, D::idxs_lo_term[r], D::idxs_hi_term[r],
                    static_cast<double>(spec.bounds.penalty.quad(row)),
                    static_cast<double>(spec.bounds.penalty.lin(row)));
            }
        }
    }
    return m;
}

/// NLP objective at the iterate: the problem's stage + terminal costs
/// plus the slack penalty (sec. 1.6; the problem cost functions exclude
/// the solver-internal slacks).
template <class P, int NH>
double compute_nlp_cost(const P& problem, const Solution<P, NH>& sol,
                        const SqpSlacks<P, NH>& sl)
{
    return raw_cost(problem, sol) + slack_penalty(problem, sl);
}

}  // namespace ocp

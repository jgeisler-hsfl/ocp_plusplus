// ocp/solvers/acados/acados.hpp
//
// SQP driver components (phase 2).
//
// This file is built up incrementally across sub-steps 2c-2g of
// SQP_PHASE2_PLAN.md.
//
// 2c: NlpResiduals, compute_nlp_residuals
//     Port of acados ocp_nlp_res_compute (ocp_nlp_common.c:3743-3846).
// 2e: AcadosOptions, SqpIteration, AcadosStatistics, print_sqp_iteration
//     Driver options (defaults per ocp_nlp_common.c:1215-1287,
//     ocp_nlp_sqp.c:103-119), per-iteration statistics (acados `stat`
//     matrix, ocp_nlp_sqp.c:266-271) and the stdout printing
//     (ocp_nlp_sqp.c:449-466).
// 2f: AcadosSolver::assemble_qp / add_lm_term / resize (QP assembly).
// 2g: AcadosSolver::solve / check_termination / compute_cost (driver).
//
// References (re-read at implementation time):
//   acados/ocp_nlp/ocp_nlp_common.c:3743-3846  (ocp_nlp_res_compute)
//   acados/ocp_nlp/ocp_nlp_sqp.c:351-430        (check_termination)
//   acados/ocp_nlp/ocp_nlp_sqp.c:449-466        (print_iteration)
//   acados/ocp_nlp/ocp_nlp_sqp.c:538-800        (main loop, 2g)
//   acados/ocp_nlp/ocp_nlp_constraints_bgp.c    (fun layout, dmask)
//   acados/ocp_qp/ocp_qp_common.c:263-279       (primal step norm, 2g)

#pragma once

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include "../hpipm/qp.hpp"
#include "../hpipm/hpipm.hpp"
#include "../hpipm/condense.hpp"
#include "../hpipm/block_condense.hpp"
#include "assemble.hpp"
#include "cost.hpp"
#include "globalize.hpp"
#include "qpscaling.hpp"
#include "regularize.hpp"

namespace ocp
{

// =========================================================================
//  NLP residuals (sub-step 2c)
// =========================================================================

/// Four NLP residual norms (acados ocp_nlp_res analogue).
///
/// res_stat  - stationarity: max over stages of the inf-norm of the
///             Lagrangian gradient w.r.t. (u_k, x_k).  When x_0 is fixed
///             (fixed_initial_state), the x_0 component is omitted (x_0 is
///             not a free variable; the QP pin rows handle it).
/// res_eq    - dynamics gap: max_k ||f_k(x_k, u_k) - x_{k+1}||_inf.
/// res_ineq  - max positive violation over all constraint rows (0 if
///             fully feasible).
/// res_comp  - complementarity: max over stages of the inf-norm of
///             lambda * violation + tau_min per active side; equality and
///             pin rows are zeroed (acados idxe masking).
struct NlpResiduals
{
    double res_stat = 0.0;
    double res_eq   = 0.0;
    double res_ineq = 0.0;
    double res_comp = 0.0;
};

/// Compute the four NLP residual norms at the iterate `sol`.
///
/// The Lagrangian used for res_stat (sec. 1.3 of SQP_PHASE2_PLAN.md):
///   L = sum_k L_k(x_k, u_k) + L_N(x_N)
///     + sum_k pi_k^T (x_{k+1} - f_k(x_k, u_k))
///     + sum_k lam_g^T g_k + sum_k mu^T e_k
///     + box/linear terms
///
/// @param problem  the concrete problem (provides model functions).
/// @param sol      the current NLP iterate (primal + multipliers).
/// @param sl       the solver-internal slacks (null for the raw residual;
///                 non-null soft rows are measured at their slack-relaxed
///                 values, acados counts the slacks as NLP variables).
/// @param tau_min  complementarity floor added to every active non-equality
///                 side (acados quirk: added, not subtracted; sec. 1.4).
template <class P, int NH = Eigen::Dynamic>
NlpResiduals compute_nlp_residuals(const P& problem,
                                   const Solution<P, NH>& sol,
                                   const SqpSlacks<P, NH>* sl = nullptr,
                                   double tau_min = 1e-16)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    const int N = sol.N;
    NlpResiduals res;

    // -----------------------------------------------------------------
    // res_stat: stationarity (Lagrangian gradient)
    // -----------------------------------------------------------------
    double res_stat_max = 0.0;

    for (int k = 0; k < N; ++k)
    {
        const auto& x = sol.x[k];
        const auto& u = sol.u[k];

        double lcost;
        typename P::stage_grad_t grad;
        problem.stage_cost_value_grad(k, x, u, lcost, grad);

        typename P::state_t x_next;
        typename P::dyn_df_dx_t df_dx;
        typename P::dyn_df_du_t df_du;
        problem.dynamics_value_jac(k, x, u, x_next, df_dx, df_du);

        // constraint Jacobians (computed once, reused for u and x parts)
        typename P::ineq_dg_dx_t g_dx;
        typename P::ineq_dg_du_t g_du;
        typename P::ineq_t g_val;
        if constexpr (D::ng > 0)
            problem.stage_inequality_value_jac(k, x, u, g_val, g_dx, g_du);

        typename P::eq_de_dx_t e_dx;
        typename P::eq_de_du_t e_du;
        typename P::eq_t e_val;
        if constexpr (D::ne > 0)
            problem.stage_equality_value_jac(k, x, u, e_val, e_dx, e_du);

        // ---- u-part: dL/du_k ----
        if constexpr (D::nu > 0)
        {
            Eigen::Matrix<S, D::nu, 1> adj_u = grad.tail(D::nu);
            adj_u -= df_du.transpose() * sol.lambda_dyn[k];

            if constexpr (D::ng > 0)
                adj_u += g_du.transpose() * sol.lambda_ineq_stage[k];
            if constexpr (D::ne > 0)
                adj_u += e_du.transpose() * sol.lambda_eq_stage[k];
            if constexpr (D::nl > 0)
            {
                auto lin_spec = problem.stage_linear_constr(k);
                adj_u += lin_spec.B.transpose() * sol.lambda_lin_stage[k];
            }
            if constexpr (D::nbu > 0)
            {
                for (int j = 0; j < D::nbu; ++j)
                {
                    const int vi = P::control_box_idx[j];
                    adj_u(vi) += -sol.lambda_box_control[k](j)
                                + sol.lambda_box_control[k](D::nbu + j);
                }
            }

            res_stat_max = std::max(
                res_stat_max, static_cast<double>(adj_u.cwiseAbs().maxCoeff()));
        }

        // ---- x-part: dL/dx_k (skipped for fixed x_0 at k = 0) ----
        if constexpr (D::nx > 0)
        {
            if (!(k == 0 && P::fixed_initial_state))
            {
                Eigen::Matrix<S, D::nx, 1> adj_x = grad.head(D::nx);
                adj_x -= df_dx.transpose() * sol.lambda_dyn[k];
                if (k > 0)
                    adj_x += sol.lambda_dyn[k - 1];

                if constexpr (D::ng > 0)
                    adj_x += g_dx.transpose() * sol.lambda_ineq_stage[k];
                if constexpr (D::ne > 0)
                    adj_x += e_dx.transpose() * sol.lambda_eq_stage[k];
                if constexpr (D::nl > 0)
                {
                    auto lin_spec = problem.stage_linear_constr(k);
                    adj_x += lin_spec.A.transpose() * sol.lambda_lin_stage[k];
                }
                if constexpr (D::nbx > 0)
                {
                    for (int j = 0; j < D::nbx; ++j)
                    {
                        const int vi = P::state_box_idx[j];
                        adj_x(vi) += -sol.lambda_box_state[k](j)
                                    + sol.lambda_box_state[k](D::nbx + j);
                    }
                }

                res_stat_max = std::max(
                    res_stat_max,
                    static_cast<double>(adj_x.cwiseAbs().maxCoeff()));
            }
        }
    }

    // terminal stage (k = N): dL/dx_N
    if constexpr (D::nx > 0)
    {
        const auto& xN = sol.x[N];
        double lcost_t;
        typename P::term_grad_t tgrad;
        problem.terminal_cost_value_grad(xN, lcost_t, tgrad);
        Eigen::Matrix<S, D::nx, 1> adj_x = tgrad;
        adj_x += sol.lambda_dyn[N - 1];

        if constexpr (D::ng_t > 0)
        {
            typename P::ineq_term_t g_val;
            typename P::ineq_term_dg_dx_t g_dx;
            problem.terminal_inequality_value_jac(xN, g_val, g_dx);
            adj_x += g_dx.transpose() * sol.lambda_ineq_term;
        }
        if constexpr (D::ne_t > 0)
        {
            typename P::eq_term_t e_val;
            typename P::eq_term_de_dx_t e_dx;
            problem.terminal_equality_value_jac(xN, e_val, e_dx);
            adj_x += e_dx.transpose() * sol.lambda_eq_term;
        }
        if constexpr (D::nl_t > 0)
        {
            auto lin_spec = problem.terminal_linear_constr();
            adj_x += lin_spec.A.transpose() * sol.lambda_lin_term;
        }
        if constexpr (D::nbx_t > 0)
        {
            for (int j = 0; j < D::nbx_t; ++j)
            {
                const int vi = P::terminal_state_box_idx[j];
                adj_x(vi) += -sol.lambda_box_state[N](j)
                            + sol.lambda_box_state[N](D::nbx + j);
            }
        }

        res_stat_max = std::max(
            res_stat_max, static_cast<double>(adj_x.cwiseAbs().maxCoeff()));
    }

    res.res_stat = res_stat_max;

    // -----------------------------------------------------------------
    // res_eq: dynamics gap  max_k ||f_k(x_k, u_k) - x_{k+1}||_inf
    // -----------------------------------------------------------------
    for (int k = 0; k < N; ++k)
    {
        const auto& x = sol.x[k];
        const auto& u = sol.u[k];
        const auto f = problem.dynamics_next_state(k, x, u);
        res.res_eq = std::max(
            res.res_eq,
            static_cast<double>((f - sol.x[k + 1]).cwiseAbs().maxCoeff()));
    }

    // -----------------------------------------------------------------
    // res_ineq: max positive violation over all constraint rows
    // -----------------------------------------------------------------
    for (int k = 0; k < N; ++k)
    {
        const auto& x = sol.x[k];
        const auto& u = sol.u[k];
        const auto& lay = (k == 0) ? D::lay_first : D::lay_path;
        // Current slack of a soft side (col = full variable-vector column
        // in this stage type); hard sides / missing slacks -> 0.
        const auto slv = [sl, k](int col) -> double
        {
            if (col < 0 || sl == nullptr)
                return 0.0;
            if (k == 0)
            {
                const int base = D::nvar_first - D::nslack_first;
                return static_cast<double>(sl->first[col - base]);
            }
            const int base = D::nvar_path - D::nslack_path;
            return static_cast<double>(sl->path[k - 1][col - base]);
        };

        if constexpr (D::ng > 0)
        {
            const auto g = problem.stage_inequality_value(k, x, u);
            for (int j = 0; j < D::ng; ++j)
            {
                const int r = lay.row_off(detail::g_ineq) + j;
                const int ch =
                    (k == 0) ? D::idxs_hi_first[r] : D::idxs_hi_path[r];
                const double v =
                    static_cast<double>(g(j)) - slv(ch);
                res.res_ineq =
                    std::max(res.res_ineq, std::max(0.0, v));
            }
        }
        if constexpr (D::ne > 0)
        {
            const auto e = problem.stage_equality_value(k, x, u);
            for (int j = 0; j < D::ne; ++j)
            {
                const int r = lay.row_off(detail::g_eq) + j;
                const int cl =
                    (k == 0) ? D::idxs_lo_first[r] : D::idxs_lo_path[r];
                const int ch =
                    (k == 0) ? D::idxs_hi_first[r] : D::idxs_hi_path[r];
                const double v = static_cast<double>(e(j));
                // Soft eq: relaxed interval [−s_lo, s_hi].
                const double viol =
                    (cl < 0 && ch < 0)
                        ? std::fabs(v)
                        : std::max({ 0.0, -v - slv(cl), v - slv(ch) });
                res.res_ineq = std::max(res.res_ineq, viol);
            }
        }
        if constexpr (D::nl > 0)
        {
            const auto lin_spec = problem.stage_linear_constr(k);
            const auto v = lin_spec.A * x + lin_spec.B * u;
            for (int j = 0; j < D::nl; ++j)
            {
                const int r = lay.row_off(detail::g_lin) + j;
                const int cl =
                    (k == 0) ? D::idxs_lo_first[r] : D::idxs_lo_path[r];
                const int ch =
                    (k == 0) ? D::idxs_hi_first[r] : D::idxs_hi_path[r];
                const double val = static_cast<double>(v(j));
                const double lo =
                    static_cast<double>(lin_spec.bounds.lo(j));
                const double hi =
                    static_cast<double>(lin_spec.bounds.hi(j));
                res.res_ineq = std::max(
                    res.res_ineq, std::max(0.0, val - hi - slv(ch)));
                res.res_ineq = std::max(
                    res.res_ineq, std::max(0.0, lo - val - slv(cl)));
            }
        }
        if constexpr (D::nbx > 0)
        {
            // State-box rows are absent from the first-stage layout when
            // x_0 is fixed (pin rows replace them); no residual there.
            if (lay.rows[detail::g_bx] > 0)
            {
                const auto spec = problem.stage_state_box_constr(k);
                for (int j = 0; j < D::nbx; ++j)
                {
                    const int r = lay.row_off(detail::g_bx) + j;
                    const int cl =
                        (k == 0) ? D::idxs_lo_first[r] : D::idxs_lo_path[r];
                    const int ch =
                        (k == 0) ? D::idxs_hi_first[r] : D::idxs_hi_path[r];
                    const double v = static_cast<double>(
                        x(P::state_box_idx[j]));
                    const double lo = static_cast<double>(spec.lo(j));
                    const double hi = static_cast<double>(spec.hi(j));
                    res.res_ineq = std::max(
                        res.res_ineq, std::max(0.0, v - hi - slv(ch)));
                    res.res_ineq = std::max(
                        res.res_ineq, std::max(0.0, lo - v - slv(cl)));
                }
            }
        }
        if constexpr (D::nbu > 0)
        {
            const auto spec = problem.stage_control_box_constr(k);
            for (int j = 0; j < D::nbu; ++j)
            {
                const int r = lay.row_off(detail::g_bu) + j;
                const int cl =
                    (k == 0) ? D::idxs_lo_first[r] : D::idxs_lo_path[r];
                const int ch =
                    (k == 0) ? D::idxs_hi_first[r] : D::idxs_hi_path[r];
                const double v = static_cast<double>(
                    u(P::control_box_idx[j]));
                const double lo = static_cast<double>(spec.lo(j));
                const double hi = static_cast<double>(spec.hi(j));
                res.res_ineq = std::max(
                    res.res_ineq, std::max(0.0, v - hi - slv(ch)));
                res.res_ineq = std::max(
                    res.res_ineq, std::max(0.0, lo - v - slv(cl)));
            }
        }
    }

    // terminal
    {
        const auto& xN = sol.x[N];
        const auto& lay = D::lay_term;
        const auto slv = [sl](int col) -> double
        {
            if (col < 0 || sl == nullptr)
                return 0.0;
            const int base = D::nvar_term - D::nslack_term;
            return static_cast<double>(sl->term[col - base]);
        };
        if constexpr (D::ng_t > 0)
        {
            const auto g = problem.terminal_inequality_value(xN);
            for (int j = 0; j < D::ng_t; ++j)
            {
                const int r = lay.row_off(detail::g_ineq) + j;
                const int ch = D::idxs_hi_term[r];
                const double v =
                    static_cast<double>(g(j)) - slv(ch);
                res.res_ineq =
                    std::max(res.res_ineq, std::max(0.0, v));
            }
        }
        if constexpr (D::ne_t > 0)
        {
            const auto e = problem.terminal_equality_value(xN);
            for (int j = 0; j < D::ne_t; ++j)
            {
                const int r = lay.row_off(detail::g_eq) + j;
                const int cl = D::idxs_lo_term[r];
                const int ch = D::idxs_hi_term[r];
                const double v = static_cast<double>(e(j));
                const double viol =
                    (cl < 0 && ch < 0)
                        ? std::fabs(v)
                        : std::max({ 0.0, -v - slv(cl), v - slv(ch) });
                res.res_ineq = std::max(res.res_ineq, viol);
            }
        }
        if constexpr (D::nl_t > 0)
        {
            const auto lin_spec = problem.terminal_linear_constr();
            const auto v = lin_spec.A * xN;
            for (int j = 0; j < D::nl_t; ++j)
            {
                const int r = lay.row_off(detail::g_lin) + j;
                const int cl = D::idxs_lo_term[r];
                const int ch = D::idxs_hi_term[r];
                const double val = static_cast<double>(v(j));
                const double lo =
                    static_cast<double>(lin_spec.bounds.lo(j));
                const double hi =
                    static_cast<double>(lin_spec.bounds.hi(j));
                res.res_ineq = std::max(
                    res.res_ineq, std::max(0.0, val - hi - slv(ch)));
                res.res_ineq = std::max(
                    res.res_ineq, std::max(0.0, lo - val - slv(cl)));
            }
        }
        if constexpr (D::nbx_t > 0)
        {
            const auto spec = problem.terminal_state_box_constr();
            for (int j = 0; j < D::nbx_t; ++j)
            {
                const int r = lay.row_off(detail::g_bx) + j;
                const int cl = D::idxs_lo_term[r];
                const int ch = D::idxs_hi_term[r];
                const double v = static_cast<double>(
                    xN(P::terminal_state_box_idx[j]));
                const double lo = static_cast<double>(spec.lo(j));
                const double hi = static_cast<double>(spec.hi(j));
                res.res_ineq = std::max(
                    res.res_ineq, std::max(0.0, v - hi - slv(ch)));
                res.res_ineq = std::max(
                    res.res_ineq, std::max(0.0, lo - v - slv(cl)));
            }
        }
    }

    // -----------------------------------------------------------------
    // res_comp: complementarity  (acados ocp_nlp_res_compute, res_comp)
    //
    // Per active side: entry = lam * max(0, violation) + tau_min
    // Equality rows and pin rows are zeroed (acados idxe masking).
    // For linear rows with a signed net multiplier, the active side is
    // determined by the sign of the net (standard KKT decomposition).
    // -----------------------------------------------------------------
    double res_comp_max = 0.0;

    for (int k = 0; k < N; ++k)
    {
        const auto& x = sol.x[k];
        const auto& u = sol.u[k];
        double stage_max = 0.0;
        const auto& lay = (k == 0) ? D::lay_first : D::lay_path;
        // Soft sides are measured at their slack-relaxed violation (the
        // slacks are NLP variables in the residual; see res_ineq above).
        const auto slv = [sl, k](int col) -> double
        {
            if (col < 0 || sl == nullptr)
                return 0.0;
            if (k == 0)
            {
                const int base = D::nvar_first - D::nslack_first;
                return static_cast<double>(sl->first[col - base]);
            }
            const int base = D::nvar_path - D::nslack_path;
            return static_cast<double>(sl->path[k - 1][col - base]);
        };

        // box state (skipped at stage 0 when x_0 is fixed: pin rows
        // replace the state box, so no box multipliers exist)
        if constexpr (D::nbx > 0)
        {
            if (!(k == 0 && P::fixed_initial_state))
            {
                const auto spec = problem.stage_state_box_constr(k);
                for (int j = 0; j < D::nbx; ++j)
                {
                    const int r = lay.row_off(detail::g_bx) + j;
                    const int cl =
                        (k == 0) ? D::idxs_lo_first[r] : D::idxs_lo_path[r];
                    const int ch =
                        (k == 0) ? D::idxs_hi_first[r] : D::idxs_hi_path[r];
                    const double v = static_cast<double>(
                        x(P::state_box_idx[j]));
                    const double lo =
                        static_cast<double>(spec.lo(j));
                    const double hi =
                        static_cast<double>(spec.hi(j));
                    const double lam_lo =
                        static_cast<double>(
                            sol.lambda_box_state[k](j));
                    const double lam_hi =
                        static_cast<double>(
                            sol.lambda_box_state[k](D::nbx + j));
                    stage_max = std::max(stage_max,
                        std::fabs(lam_lo * std::max(0.0, lo - v - slv(cl))
                                  + tau_min));
                    stage_max = std::max(stage_max,
                        std::fabs(lam_hi * std::max(0.0, v - hi - slv(ch))
                                  + tau_min));
                }
            }
        }

        // box control
        if constexpr (D::nbu > 0)
        {
            const auto spec = problem.stage_control_box_constr(k);
            for (int j = 0; j < D::nbu; ++j)
            {
                const int r = lay.row_off(detail::g_bu) + j;
                const int cl =
                    (k == 0) ? D::idxs_lo_first[r] : D::idxs_lo_path[r];
                const int ch =
                    (k == 0) ? D::idxs_hi_first[r] : D::idxs_hi_path[r];
                const double v = static_cast<double>(
                    u(P::control_box_idx[j]));
                const double lo =
                    static_cast<double>(spec.lo(j));
                const double hi =
                    static_cast<double>(spec.hi(j));
                const double lam_lo =
                    static_cast<double>(
                        sol.lambda_box_control[k](j));
                const double lam_hi =
                    static_cast<double>(
                        sol.lambda_box_control[k](D::nbu + j));
                stage_max = std::max(stage_max,
                    std::fabs(lam_lo * std::max(0.0, lo - v - slv(cl))
                              + tau_min));
                stage_max = std::max(stage_max,
                    std::fabs(lam_hi * std::max(0.0, v - hi - slv(ch))
                              + tau_min));
            }
        }

        // ineq (one-sided: hi side only, hi = 0)
        if constexpr (D::ng > 0)
        {
            const auto g = problem.stage_inequality_value(k, x, u);
            for (int j = 0; j < D::ng; ++j)
            {
                const int r = lay.row_off(detail::g_ineq) + j;
                const int ch =
                    (k == 0) ? D::idxs_hi_first[r] : D::idxs_hi_path[r];
                const double v =
                    static_cast<double>(g(j)) - slv(ch);
                const double lam_hi =
                    static_cast<double>(
                        sol.lambda_ineq_stage[k](j));
                stage_max = std::max(stage_max,
                    std::fabs(lam_hi * std::max(0.0, v) + tau_min));
            }
        }

        // eq rows: zeroed (acados idxe masking, sec. 1.4)

        // linear (two-sided, signed net multiplier)
        if constexpr (D::nl > 0)
        {
            const auto lin_spec = problem.stage_linear_constr(k);
            const auto v = lin_spec.A * x + lin_spec.B * u;
            for (int j = 0; j < D::nl; ++j)
            {
                const int r = lay.row_off(detail::g_lin) + j;
                const int cl =
                    (k == 0) ? D::idxs_lo_first[r] : D::idxs_lo_path[r];
                const int ch =
                    (k == 0) ? D::idxs_hi_first[r] : D::idxs_hi_path[r];
                const double val = static_cast<double>(v(j));
                const double lo =
                    static_cast<double>(lin_spec.bounds.lo(j));
                const double hi =
                    static_cast<double>(lin_spec.bounds.hi(j));
                const double net =
                    static_cast<double>(
                        sol.lambda_lin_stage[k](j));
                double entry;
                if (net >= 0.0)
                    entry = net * std::max(0.0, val - hi - slv(ch))
                            + tau_min;
                else
                    entry = (-net) * std::max(0.0, lo - val - slv(cl))
                            + tau_min;
                stage_max = std::max(stage_max, std::fabs(entry));
            }
        }

        // pin rows: zeroed (acados idxe masking)

        res_comp_max = std::max(res_comp_max, stage_max);
    }

    // terminal
    {
        const auto& xN = sol.x[N];
        double stage_max = 0.0;
        const auto& lay = D::lay_term;
        const auto slv = [sl](int col) -> double
        {
            if (col < 0 || sl == nullptr)
                return 0.0;
            const int base = D::nvar_term - D::nslack_term;
            return static_cast<double>(sl->term[col - base]);
        };

        if constexpr (D::nbx_t > 0)
        {
            const auto spec = problem.terminal_state_box_constr();
            for (int j = 0; j < D::nbx_t; ++j)
            {
                const int r = lay.row_off(detail::g_bx) + j;
                const int cl = D::idxs_lo_term[r];
                const int ch = D::idxs_hi_term[r];
                const double v = static_cast<double>(
                    xN(P::terminal_state_box_idx[j]));
                const double lo =
                    static_cast<double>(spec.lo(j));
                const double hi =
                    static_cast<double>(spec.hi(j));
                const double lam_lo =
                    static_cast<double>(
                        sol.lambda_box_state[N](j));
                const double lam_hi =
                    static_cast<double>(
                        sol.lambda_box_state[N](D::nbx + j));
                stage_max = std::max(stage_max,
                    std::fabs(lam_lo * std::max(0.0, lo - v - slv(cl))
                              + tau_min));
                stage_max = std::max(stage_max,
                    std::fabs(lam_hi * std::max(0.0, v - hi - slv(ch))
                              + tau_min));
            }
        }

        if constexpr (D::ng_t > 0)
        {
            const auto g = problem.terminal_inequality_value(xN);
            for (int j = 0; j < D::ng_t; ++j)
            {
                const int r = lay.row_off(detail::g_ineq) + j;
                const int ch = D::idxs_hi_term[r];
                const double v =
                    static_cast<double>(g(j)) - slv(ch);
                const double lam_hi =
                    static_cast<double>(sol.lambda_ineq_term(j));
                stage_max = std::max(stage_max,
                    std::fabs(lam_hi * std::max(0.0, v) + tau_min));
            }
        }

        // terminal eq: zeroed

        if constexpr (D::nl_t > 0)
        {
            const auto lin_spec = problem.terminal_linear_constr();
            const auto v = lin_spec.A * xN;
            for (int j = 0; j < D::nl_t; ++j)
            {
                const int r = lay.row_off(detail::g_lin) + j;
                const int cl = D::idxs_lo_term[r];
                const int ch = D::idxs_hi_term[r];
                const double val = static_cast<double>(v(j));
                const double lo =
                    static_cast<double>(lin_spec.bounds.lo(j));
                const double hi =
                    static_cast<double>(lin_spec.bounds.hi(j));
                const double net =
                    static_cast<double>(sol.lambda_lin_term(j));
                double entry;
                if (net >= 0.0)
                    entry = net * std::max(0.0, val - hi - slv(ch))
                            + tau_min;
                else
                    entry = (-net) * std::max(0.0, lo - val - slv(cl))
                            + tau_min;
                stage_max = std::max(stage_max, std::fabs(entry));
            }
        }

        res_comp_max = std::max(res_comp_max, stage_max);
    }

    res.res_comp = res_comp_max;

    return res;
}

// =========================================================================
//  Options, statistics, printing (sub-step 2e)
// =========================================================================

/// SQP driver options.
///
/// Defaults verified against acados:
///   ocp_nlp_common.c:1215,1276,1282-1287  (max_iter, eval_residual, tolerances)
///   ocp_nlp_sqp.c:114-117                  (max_iter override, timeout)
///   ocp_nlp_sqp.c:103-119                  (ocp_nlp_sqp_opts_initialize_default)
///
/// `tau_min` follows HPIPM's IPM convention (x_ocp_qp_ipm.c:95); the acados
/// NLP-level default is 0 (zero-initialized struct).
/// QP solve mode of the SQP driver (parity plan Phase 1 / Phase 2).
///
/// kStaged solves the staged OCP QP with the HPIPM staged solver (default;
/// mirrors acados `SQP_RTI_HPIPM`). kFullCondensed first condenses the staged
/// QP to a dense QP on (x_0, u_0..u_{N-1}, slacks), solves it with the dense
/// HPIPM IPM, then expands the solution back to the staged layout (mirrors
/// acados `FULL_CONDENSING_HPIPM`). kPartialCondensed condenses the N stages
/// into N2 = N / NB blocks of NB stages each (the internal states of each
/// block are eliminated), solves the N2-stage block QP with the staged
/// HPIPM IPM over BlockQpDim<P, NB>, and expands the solution back (mirrors
/// acados `PARTIAL_CONDENSING_HPIPM`). NB is the trailing template parameter
/// of AcadosSolver; when it is 0 this mode falls back to kStaged.
enum class QpMode
{
    kStaged,
    kFullCondensed,
    kPartialCondensed
};

struct AcadosOptions
{
    int max_iter = 20;
    double tol_stat = 1e-8, tol_eq = 1e-8, tol_ineq = 1e-8, tol_comp = 1e-8;
    double tol_min_step_norm = 1e-12;
    double tol_unbounded = -1e10;
    // Per-source runtime gates for the multiplier-weighted Hessian terms
    // in the QP Hessian; mirror acados' `exact_hess_dyn` /
    // `exact_hess_constr`. Each is additionally gated by the Dims
    // capability flag (has_dynamics_hess / has_constr_hess). When a gate
    // is false the corresponding term is dropped from the QP Hessian
    // (Gauss-Newton for that source), regardless of the Dims flag.
    bool compute_hess_dyn = true;
    bool compute_hess_constr = true;
    double levenberg_marquardt = 0.0;
    bool with_adaptive_lm = false;        // 3g: adaptive LM (gates the mu schedule)
    double adaptive_lm_mu0 = 1e-3;        // 3g: mu / mu_bar init (ocp_nlp_common.c:1265)
    double adaptive_lm_lam = 5.0;         // 3g: growth/shrink factor
    double adaptive_lm_mu_min = 1e-16;    // 3g: floor for the full-step decay
    double adaptive_lm_obj_scalar = 2.0;  // 3g: mu_eff = obj_scalar * raw_cost * mu
    int print_level = 0;                  // 0 = silent, 1 = per-iteration rows,
                                          // 2 = + QP IPM stat table per QP solve
                                          // and a globalization-failure message
                                          // (phase 5; acados forwards a QP
                                          // print_level of 1 for levels > 2)
    int qp_warm_start = 0;                // HPIPM warm_start: 0 cold / 2 primal+dual / 3 hot
    bool warm_start_first_qp = false;     // also warm-start the very first QP
    QpMode qp_mode = QpMode::kStaged;     // kStaged | kFullCondensed | kPartialCondensed
    /// Phase 2: number of condensation blocks N2 used by kPartialCondensed
    /// (acados `qp_cond_N`). 0 (default) selects the identity fast path
    /// (plain staged solve). When > 0 the block size is NB = N / qp_cond_N,
    /// which must equal the solver's compile-time NB (template parameter).
    int qp_cond_N = 0;
    bool eval_residual_at_max_iter = false;
    int timeout_heuristic = 0;            // 3f: 0 = ZERO, 1 = LAST, 2 = MAX,
                                          // 3 = AVERAGE (acados
                                          // ocp_nlp_timeout_heuristic_t)
    double timeout_max_time = 0.0;        // 3f: wall-clock budget; 0 = none
    bool scale_qp_objective = false;      // 3h: Gershgorin objective scaling
    bool scale_qp_constraints = false;    // 3h: inf-norm constraint scaling
    double ub_max_abs_eig = 1e5;          // 3h: objective-scaling upper bound
    double lb_norm_inf_grad_obj = 1e-4;   // 3h: objective-scaling grad floor
    // 3h: per-iteration QP tolerance strategy
    // (ocp_nlp_common.c:4522-4574; acados types.h:102-107).
    //   0 = kFixedQpTol              (HPIPM defaults, no per-iter update)
    //   1 = kAdaptiveCurrentResJoint (from the NLP residuals)
    //   2 = kAdaptiveQpScaling       (from the scaling factors)
    int nlp_qp_tol_strategy = 0;
    double nlp_qp_tol_reduction_factor = 1e-1;
    double nlp_qp_tol_safety_factor = 0.1;
    double nlp_qp_tol_min_stat = 1e-9;
    double nlp_qp_tol_min_eq = 1e-10;
    double nlp_qp_tol_min_ineq = 1e-10;
    double nlp_qp_tol_min_comp = 1e-11;
    double tau_min = 1e-16;               // res_comp floor (plan sec. 1.4)
    GlobOptions glob;
};

/// One row of the SQP statistics.
///
/// Mirrors the acados `stat` matrix columns (ocp_nlp_sqp.c:266-271):
/// res_stat / res_eq / res_ineq / res_comp (NLP residuals at the
/// iteration start), then qp_status / qp_iter (QP solver result),
/// step_norm (max inf-norm of the primal step including slacks,
/// ocp_qp_common.c:263), alpha (accepted step size).
struct SqpIteration
{
    double res_stat = 0.0, res_eq = 0.0, res_ineq = 0.0, res_comp = 0.0;
    int qp_status = 0, qp_iter = 0;
    double step_norm = 0.0, alpha = 0.0;
    // Funnel accept type (0='-', 1='f', 2='h', 3='b', 4='p'); -1 when the
    // active globalizer is not a funnel (SQP_PHASE3_PLAN sec. 3i).
    int funnel_iter_type = -1;
    double funnel_width = 0.0;
    /// Wall-clock seconds spent in condense + expand for this iteration's QP
    /// (kFullCondensed mode only; 0 in kStaged mode).
    double cond_time = 0.0;
};

/// Per-iteration records of a SQP solve (acados `stat` matrix analogue).
///
/// `iter` is the number of recorded iterations.  `status` is the final
/// exit status, set by the driver before returning.
struct AcadosStatistics
{
    int iter = 0;
    Status status = Status::kUnset;
    std::vector<SqpIteration> rows;

    /// Append an iteration record.
    void record(const SqpIteration& row)
    {
        rows.push_back(row);
        ++iter;
    }

    /// Write statistics to `path` as CSV (header + one row per iteration).
    /// No-op if the file cannot be opened.
    void write_csv(const char* path) const
    {
        std::FILE* f = std::fopen(path, "w");
        if (f == nullptr)
            return;
        std::fprintf(f, "iter,res_stat,res_eq,res_ineq,res_comp,"
                        "qp_status,qp_iter,step_norm,alpha\n");
        for (std::size_t i = 0; i < rows.size(); ++i)
        {
            const SqpIteration& r = rows[i];
            std::fprintf(f, "%zu,%.10e,%.10e,%.10e,%.10e,%d,%d,%.10e,%.10e\n",
                         i, r.res_stat, r.res_eq, r.res_ineq, r.res_comp,
                         r.qp_status, r.qp_iter, r.step_norm, r.alpha);
        }
        std::fclose(f);
    }
};

/// Print one iteration to stdout (acados `print_iteration` analogue,
/// ocp_nlp_sqp.c:449-466).  The header is re-emitted every 10 iterations.
inline void print_sqp_iteration(int iter, const SqpIteration& row)
{
    if (iter % 10 == 0)
    {
        std::printf("%6s   %10s   %10s   %10s   %10s   "
                    "%7s   %7s   %10s   %8s\n",
                    "# it", "res_stat", "res_eq", "res_ineq", "res_comp",
                    "qp_stat", "qp_iter", "step_norm", "alpha");
    }
    std::printf("%6d   %10.4e   %10.4e   %10.4e   %10.4e   "
                "%7d   %7d   %10.2e   %8.2e\n",
                iter, row.res_stat, row.res_eq, row.res_ineq, row.res_comp,
                row.qp_status, row.qp_iter, row.step_norm, row.alpha);
}

// =========================================================================
//  AcadosSolver (sub-step 2f: QP assembly + LM term)
// =========================================================================

namespace detail
{
/// Detects a `last_iter_type()` accessor (funnel accept-type logging, 3i).
template <class G, class = void>
struct has_iter_type : std::false_type
{
};
template <class G>
struct has_iter_type<
    G, std::void_t<decltype(std::declval<const G&>().last_iter_type())>>
    : std::true_type
{
};

/// Block-QP workspaces of a SQP driver (phase 2e).  For NB > 0 holds the
/// block-condensing solver plus its staged-layout data / solution (always
/// dynamic-horizon: N2 = N / NB is a runtime quantity).  The NB == 0
/// specialization is empty, so a solver compiled without partial
/// condensing carries no extra state (and its default-constructed size /
/// layout are unchanged).
template <class P, int NB>
struct BlockQpWorkspaces
{
    HpipmQpSolver<P, Eigen::Dynamic, BlockQpDim<P, NB>> solver;
    Qp<P, Eigen::Dynamic, BlockQpDim<P, NB>> in;
    QpSol<P, Eigen::Dynamic, BlockQpDim<P, NB>> out;
};
template <class P>
struct BlockQpWorkspaces<P, 0>
{
};
}  // namespace detail

/// SQP driver (phase 2).
///
/// Template parameter order: P (problem), NH (horizon), then the pluggable
/// components with defaults. NH precedes the component parameters so the
/// defaults can reference it (a default template argument may only
/// reference preceding parameters; the plan sketch's order <P, QpSolver,
/// Regularizer, Globalizer, NH> was not compilable).
template <class P, int NH = Eigen::Dynamic,
          class QpSolver = HpipmQpSolver<P, NH>,
          class Regularizer = GlmRegularizer<P, NH>,
          class Globalizer = MeritBacktracking<P, NH>, int NB = 0>
class AcadosSolver
{
public:
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    explicit AcadosSolver(AcadosOptions opts = AcadosOptions{})
        : opts_(opts)
        , scaler_(make_scaler_opts(opts))
    {
        if (NB == 0 && opts_.qp_mode == QpMode::kPartialCondensed)
        {
            std::fprintf(stderr,
                "AcadosSolver: kPartialCondensed requested but the solver "
                "was compiled with NB = 0; falling back to kStaged.\n");
        }
    }

    /// Assemble the staged QP at the current NLP iterate into qp_in_.
    ///
    /// Fills all stage data (hess, grad, BA, b, DC, d, d_mask; m stays 0
    /// in v1). The QP is in the step formulation (SQP_PHASE2_PLAN sec. 1.1):
    /// variables z_k = (du_k; dx_k; s_k), row values v = DC z, b =
    /// f(x, u) - x_{k+1}; d holds the bound OFFSETS d_lo = lo - w_cur,
    /// d_hi = w_cur - hi (w_cur = the row value at the current iterate).
    ///
    /// Returns kSolved on success, kInfeasible if a first-stage ineq/lin
    /// row with zero Jacobian w.r.t. (u_0; s_0) is violated (degenerate
    /// fixed-x_0 case, plan sec. 2f.6).
    Status assemble_qp(const P& problem, const Solution<P, NH>& sol)
    {
        const int N = sol.N;
        assert(qp_in_.N == N);

        qp_in_.first.setZero();
        for (int k = 1; k < N; ++k)
        {
            qp_in_.path[k - 1].setZero();
        }
        qp_in_.term.setZero();

        Status st = assemble_first(problem, sol);
        if (st != Status::kSolved)
        {
            return st;
        }
        for (int k = 1; k < N; ++k)
        {
            assemble_path(problem, sol, k);
        }
        assemble_term(problem, sol);
        return Status::kSolved;
    }

    /// Add Levenberg-Marquardt damping: mu*I to the (u;x) Hessian block of
    /// every stage (first/path: leading (nu+nx) diagonal; term: full nx
    /// diagonal). No-op when both compute_hess_dyn and compute_hess_constr
    /// are false or mu <= 0.
    /// (acados ocp_nlp_add_levenberg_marquardt_term, ocp_nlp_common.c:3034;
    /// deviation: the acados cost-module `scaling` factor has no
    /// counterpart in our interface, the examples use scaling = 1.)
    void add_lm_term(Qp<P, NH>& qp, double mu) const
    {
        if ((!opts_.compute_hess_dyn && !opts_.compute_hess_constr) || mu <= 0.0)
        {
            return;
        }
        constexpr int nux = D::nu + D::nx;
        for (int i = 0; i < nux; ++i)
        {
            qp.first.hess(i, i) += static_cast<S>(mu);
        }
        for (int k = 0; k < qp.N - 1; ++k)
        {
            for (int i = 0; i < nux; ++i)
            {
                qp.path[k].hess(i, i) += static_cast<S>(mu);
            }
        }
        for (int i = 0; i < D::nx; ++i)
        {
            qp.term.hess(i, i) += static_cast<S>(mu);
        }
    }

    /// Update the adaptive Levenberg-Marquardt parameters (3g).
    ///
    /// Port of acados `adaptive_levenberg_marquardt_update_mu`
    /// (ocp_nlp_common.c:3011-3032). `alpha` is the accepted step size of the
    /// *previous* iteration (the driver's `alpha_` at the top of the loop; 0.0
    /// at iter 0, where it is ignored). A full step (alpha == 1) shrinks mu
    /// by 1/lam toward mu_min, carrying mu_bar = the old mu; a truncated step
    /// grows mu by lam, capped at 1. Exposed (with `adaptive_lm_mu()` /
    /// `adaptive_lm_mu_bar()`) so the schedule can be unit-tested, mirroring
    /// `check_termination`.
    void update_adaptive_lm_mu(int iter, double alpha)
    {
        if (iter == 0)
        {
            lm_mu_ = opts_.adaptive_lm_mu0;
            lm_mu_bar_ = opts_.adaptive_lm_mu0;
        }
        else if (alpha == 1.0)
        {
            const double mu_old = lm_mu_;
            lm_mu_ = std::max(opts_.adaptive_lm_mu_min,
                               lm_mu_bar_ / opts_.adaptive_lm_lam);
            lm_mu_bar_ = mu_old;
        }
        else
        {
            lm_mu_ = std::min(opts_.adaptive_lm_lam * lm_mu_, 1.0);
        }
    }

    double adaptive_lm_mu() const { return lm_mu_; }
    double adaptive_lm_mu_bar() const { return lm_mu_bar_; }

    /// Size the internal workspaces for a horizon of `n_stages` stages
    /// (dynamic-extent mode allocates; fixed-extent mode only asserts).
    void resize(int n_stages)
    {
        qp_in_.resize(n_stages);
        qp_out_.resize(n_stages);
        scaler_.resize(n_stages);
        trial_.resize(n_stages);
        slacks_.resize(n_stages);
        dense_qp_in_.resize(n_stages);
        dense_qp_out_.resize(n_stages);
        if constexpr (NB > 0)
        {
            if (n_stages % NB == 0)
            {
                block_qp_.in.resize(n_stages / NB);
                block_qp_.out.resize(n_stages / NB);
            }
        }
    }

    const AcadosOptions& options() const { return opts_; }
    const AcadosStatistics& statistics() const { return stat_; }
    const Qp<P, NH>& last_qp() const { return qp_in_; }
    const QpSol<P, NH>& last_qp_sol() const { return qp_out_; }

    /// NLP objective at the iterate: the problem's stage + terminal costs
    /// plus the slack penalty 0.5 * quad * s^2 + lin * s per soft side
    /// (sec. 1.6; the problem cost functions exclude the solver-internal
    /// slacks). Delegates to the shared free functions in cost.hpp.
    double compute_cost(const P& problem, const Solution<P, NH>& sol,
                        const SqpSlacks<P, NH>& sl) const
    {
        return compute_nlp_cost(problem, sol, sl);
    }

    /// Number of second-order-correction re-solves performed (3j).
    int soc_count() const { return soc_count_; }

    /// Run the SQP iteration on `sol` (warm start in, solution out).
    ///
    /// Mirrors acados ocp_nlp_sqp.c:538-800 (sec. 2g): each iteration
    /// assembles the step-formulation QP at the current iterate,
    /// regularizes it (before the termination check, as in acados),
    /// checks termination, solves the QP, and runs the globalizer's line
    /// search (which advances `sol` in place on acceptance or kMinStep).
    /// The solver-internal slacks are warm-started across repeated
    /// solves on the same object when the horizon matches (acados
    /// nlp_out slack analogue). `sol.status` / `sol.cost_value` are set
    /// on all exit paths (the cost includes the 0.5*w*s^2 slack penalty,
    /// plan sec. 1.6).
    /// Termination decision at the current iterate (public so the branch
    /// order can be unit-tested; see tests/sqp/driver_2g.cpp). Pure
    /// function of `opts_`, the four residual norms, the last step norm
    /// and cost.
    ///
    /// Mirrors acados `check_termination` (ocp_nlp_sqp.c:351-430), in
    /// exactly this order:
    ///   1. any residual NaN            -> kNanDetected
    ///   2. iter >= max_iter and
    ///      !eval_residual_at_max_iter  -> kMaxIterations
    ///   3. all four norms < tols       -> kSolved
    ///   4. iter > 0 and step_norm <
    ///      tol_min_step_norm (and
    ///      tol_min_step_norm > 0)      -> kMinStep
    ///   5. cost <= tol_unbounded       -> kUnbounded
    ///   6. iter >= max_iter            -> kMaxIterations
    ///   7. timeout_max_time > 0 and
    ///      timeout_max_time <=
    ///      elapsed + estimate           -> kTimeout (3f)
    ///   otherwise                      -> kUnset (keep iterating)
    ///
    /// `elapsed` is the wall-clock seconds since `solve()` entry (0 unless a
    /// timeout budget is active); the estimate is the last per-iteration
    /// duration tracked by the driver (3f).
    Status check_termination(int iter, const NlpResiduals& res,
                             double step_norm, double cost,
                             double elapsed = 0.0) const
    {
        if (std::isnan(res.res_stat) || std::isnan(res.res_eq) ||
            std::isnan(res.res_ineq) || std::isnan(res.res_comp))
        {
            return Status::kNanDetected;
        }
        if (!opts_.eval_residual_at_max_iter && iter >= opts_.max_iter)
        {
            return Status::kMaxIterations;
        }
        if (res.res_stat < opts_.tol_stat && res.res_eq < opts_.tol_eq &&
            res.res_ineq < opts_.tol_ineq && res.res_comp < opts_.tol_comp)
        {
            return Status::kSolved;
        }
        if (opts_.tol_min_step_norm > 0.0 && iter > 0 &&
            step_norm < opts_.tol_min_step_norm)
        {
            return Status::kMinStep;
        }
        if (cost <= opts_.tol_unbounded)
        {
            return Status::kUnbounded;
        }
        if (iter >= opts_.max_iter)
        {
            return Status::kMaxIterations;
        }
        // Timeout (acados ocp_nlp_sqp.c:434-440): the budget is exceeded when
        // the wall-clock elapsed so far plus the predicted per-iteration
        // duration would pass timeout_max_time. Reached only if none of the
        // earlier branches fired (in particular a converged iterate returns
        // kSolved above, as in acados).
        if (opts_.timeout_max_time > 0.0 &&
            opts_.timeout_max_time <=
                elapsed + timeout_estimated_per_iter_)
        {
            return Status::kTimeout;
        }
        return Status::kUnset;
    }

    Status solve(const P& problem, Solution<P, NH>& sol)
    {
        const int N = sol.N;

        stat_ = AcadosStatistics{};
        qp_status_ = 0;
        qp_iter_ = 0;
        alpha_ = 0.0;
        step_norm_ = 0.0;
        timeout_estimated_per_iter_ = 0.0;
        lm_mu_ = 0.0;
        lm_mu_bar_ = 0.0;
        soc_count_ = 0;
        cond_time_ = 0.0;
        qp_ws_ = 0;

        // 3j: hand the globalization options to the active globalizer.
        glob_.opts = opts_.glob;

        resize(N);
        if (slack_N_ != N)
        {
            slacks_.setZero();
        }
        slack_N_ = N;

        cost_value_ = compute_cost(problem, sol, slacks_);
        sol.cost_value = cost_value_;
        sol.status = Status::kUnset;

        const auto t_start = std::chrono::steady_clock::now();
        double timeout_prev_time_tot = 0.0;

        for (int iter = 0; iter <= opts_.max_iter; ++iter)
        {
            Status st = assemble_qp(problem, sol);
            if (st != Status::kSolved)
            {
                sol.status = st;
                stat_.status = st;
                return st;
            }
            // 3g: the adaptive LM update (ocp_nlp_common.c:3011-3059) uses
            // the *previous* iteration's accepted step size (alpha_), which
            // is exactly what alpha_ holds at the top of the loop (0.0 at
            // iter 0).
            double lm = opts_.levenberg_marquardt;
            if (opts_.with_adaptive_lm)
            {
                update_adaptive_lm_mu(iter, alpha_);
                lm = opts_.adaptive_lm_obj_scalar *
                     raw_cost(problem, sol) * lm_mu_;
            }
            add_lm_term(qp_in_, lm);

            // 3h: scale the assembled QP into the solver's working copy
            // (acados ocp_nlp_sqp.c:594-597); the regularizer and the QP
            // solver then act on the scaled copy.
            scaler_.scale_qp(qp_in_);

            const NlpResiduals res =
                compute_nlp_residuals(problem, sol, &slacks_,
                                      opts_.tau_min);

            if (iter == 0)
            {
                glob_.initialize(problem, sol);
            }

            // Before the termination check (acados ocp_nlp_sqp.c:599-600):
            // the regularized (scaled-space) QP is the one that would be
            // solved at a stationary iterate.
            reg_.regularize(scaler_.scaled_in());

            // Phase 3f: update the per-iteration time estimate before the
            // termination check (acados ocp_nlp_sqp.c:606-644).
            double elapsed = 0.0;
            if (opts_.timeout_max_time > 0.0)
            {
                elapsed = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - t_start)
                              .count();
                if (iter > 0)
                {
                    const double prev_iter_time =
                        elapsed - timeout_prev_time_tot;
                    switch (opts_.timeout_heuristic)
                    {
                    case 1:  // LAST
                        timeout_estimated_per_iter_ = prev_iter_time;
                        break;
                    case 2:  // MAX
                        timeout_estimated_per_iter_ = std::max(
                            timeout_estimated_per_iter_, prev_iter_time);
                        break;
                    case 3:  // AVERAGE
                        timeout_estimated_per_iter_ =
                            0.5 * prev_iter_time +
                            0.5 * timeout_estimated_per_iter_;
                        break;
                    default:  // 0 = ZERO: stays 0
                        break;
                    }
                }
                timeout_prev_time_tot = elapsed;
            }

            const Status term =
                check_termination(iter, res, step_norm_, cost_value_,
                                  elapsed);
            if (term != Status::kUnset)
            {
                sol.status = term;
                sol.cost_value = cost_value_;
                stat_.status = term;
                const SqpIteration row = make_row(res);
                stat_.record(row);
                if (opts_.print_level > 0)
                {
                    print_sqp_iteration(iter, row);
                }
                return term;
            }

            // 3h: per-iteration QP tolerance strategy (acados
            // ocp_nlp_common.c:4522-4574), applied right before the QP solve.
            // No-op when nlp_qp_tol_strategy == 0 (kFixedQpTol).
            update_qp_tolerances(res);

            // QP warm start (phase 3c, acados ocp_qp_hpipm.c:327-336 +
            // ocp_nlp_sqp.c:660-691): the primal step is always cold; only
            // the duals (pi / lam / t) carried in the output from the
            // previous QP are reused (acados warm-starts the scaled QP with
            // the previous scaled_qp_out). The first QP is cold unless
            // warm_start_first_qp is set.
            int ws = opts_.qp_warm_start;
            if (iter == 0 && !opts_.warm_start_first_qp)
            {
                ws = 0;
            }
            qp_ws_ = ws;

            const Status qp_st =
                solve_qp(scaler_.scaled_in(), scaler_.scaled_out(), ws);
            qp_status_ = static_cast<int>(qp_st);
            if (qp_st != Status::kSolved &&
                qp_st != Status::kMaxIterations)
            {
                // acados: a QP MAXITER is acceptable, any other failure is
                // a hard stop (ocp_nlp_sqp.c:725-749)
                sol.status = Status::kQpFailure;
                sol.cost_value = cost_value_;
                stat_.status = Status::kQpFailure;
                const SqpIteration row = make_row(res);
                stat_.record(row);
                if (opts_.print_level > 0)
                {
                    print_sqp_iteration(iter, row);
                }
                return Status::kQpFailure;
            }

            // 3h: map the scaled-space solution back to original space
            // (acados ocp_nlp_qpscaling_rescale_solution, :327-375 /
            // :668-686). qp_out_ then carries the original-space step and
            // duals, as in v1.
            scaler_.rescale_solution(qp_out_);

            step_norm_ = primal_step_norm_inf(qp_out_);

            // Advances sol in place on kSolved / kMinStep (2d); leaves it
            // untouched on kNanDetected. 3j: the globalizer may re-solve the
            // QP (second-order correction) through this callable; it runs in
            // original space and rewrites qp_out_ in place (alpha = 1 step
            // is what it sees).
            const Status gstatus = glob_.find_acceptable_iterate(
                problem, sol, qp_in_, qp_out_, slacks_, trial_, alpha_,
                [this, &problem, &sol]() -> Status {
                    return perform_second_order_correction(problem, sol);
                });
            if (gstatus == Status::kSolved || gstatus == Status::kMinStep)
            {
                update_slacks(qp_out_, alpha_);
                cost_value_ = compute_cost(problem, sol, slacks_);
                sol.cost_value = cost_value_;
            }

            const SqpIteration row = make_row(res);
            stat_.record(row);

            if (gstatus != Status::kSolved)
            {
                sol.status = gstatus;
                stat_.status = gstatus;
                if (opts_.print_level > 1)
                {
                    std::printf("\nFailure in globalization, got status %d!\n",
                                static_cast<int>(gstatus));
                }
                if (opts_.print_level > 0)
                {
                    print_sqp_iteration(iter, row);
                }
                return gstatus;
            }
            if (opts_.print_level > 0)
            {
                print_sqp_iteration(iter, row);
            }
        }

        // Unreachable: check_termination returns kMaxIterations at
        // iter == max_iter (ocp_nlp_sqp.c:419-430).
        sol.status = Status::kMaxIterations;
        sol.cost_value = cost_value_;
        stat_.status = Status::kMaxIterations;
        return Status::kMaxIterations;
    }

private:
    /// Map AcadosOptions to QpScaler opts (internal glue; used only by the
    /// constructor).
    static typename QpScaler<P, NH>::Opts make_scaler_opts(
        const AcadosOptions& o)
    {
        typename QpScaler<P, NH>::Opts so;
        so.scale_objective = o.scale_qp_objective;
        so.scale_constraints = o.scale_qp_constraints;
        so.ub_max_abs_eig = o.ub_max_abs_eig;
        so.lb_norm_inf_grad_obj = o.lb_norm_inf_grad_obj;
        return so;
    }

    // ---------------------------------------------------------------
    //  Driver helpers (sub-step 2g)
    // ---------------------------------------------------------------

    /// One statistics row: the NLP residuals at this iteration plus the
    /// QP status / iteration / step norm / alpha of the most recent QP
    /// solve (zeros before the first QP solve; the row of an iteration
    /// that continues therefore carries that iteration's QP data).
    SqpIteration make_row(const NlpResiduals& res) const
    {
        SqpIteration row;
        row.res_stat = res.res_stat;
        row.res_eq = res.res_eq;
        row.res_ineq = res.res_ineq;
        row.res_comp = res.res_comp;
        row.qp_status = qp_status_;
        row.qp_iter = qp_iter_;
        row.step_norm = step_norm_;
        row.alpha = alpha_;
        row.cond_time = cond_time_;
        if constexpr (detail::has_iter_type<Globalizer>::value)
        {
            row.funnel_iter_type = glob_.last_iter_type();
            row.funnel_width = glob_.funnel_width();
        }
        return row;
    }

    /// Per-iteration QP tolerance update (acados
    /// ocp_nlp_common.c:4522-4574, called inside
    /// ocp_nlp_solve_qp_and_correct_dual). No-op when strategy is 0
    /// (kFixedQpTol).
    void update_qp_tolerances(const NlpResiduals& res)
    {
        if (opts_.nlp_qp_tol_strategy == 0)
        {
            return;
        }
        double ts = 0, te = 0, ti = 0, tc = 0;
        if (opts_.nlp_qp_tol_strategy == 1)
        {
            // ADAPTIVE_CURRENT_RES_JOINT
            const double rf = opts_.nlp_qp_tol_reduction_factor;
            ts = std::min(rf * res.res_stat, 1e-2);
            te = std::min(rf * res.res_eq, 1e-2);
            ti = std::min(rf * res.res_ineq, 1e-2);
            tc = std::min(rf * res.res_comp, 1e-2);
            const double joint = std::max(
                ts, std::max(te, std::max(ti, tc)));
            const double sf = opts_.nlp_qp_tol_safety_factor;
            ts = std::max(sf * opts_.tol_stat, joint);
            te = std::max(sf * opts_.tol_eq, joint);
            ti = std::max(sf * opts_.tol_ineq, joint);
            tc = std::max(sf * opts_.tol_comp, joint);
        }
        else
        {
            // ADAPTIVE_QPSCALING
            const double mcs = scaler_.min_constr_scaling();
            const double sf = opts_.nlp_qp_tol_safety_factor;
            const double stat_factor = std::min(
                scaler_.obj_factor(), mcs);
            ts = sf * opts_.tol_stat * stat_factor;
            te = sf * opts_.tol_eq * 1.0;
            ti = sf * opts_.tol_ineq * mcs;
            tc = sf * opts_.tol_comp * mcs;
            ts = std::max(ts, opts_.nlp_qp_tol_min_stat);
            te = std::max(te, opts_.nlp_qp_tol_min_eq);
            ti = std::max(ti, opts_.nlp_qp_tol_min_ineq);
            tc = std::max(tc, opts_.nlp_qp_tol_min_comp);
        }
        qp_.set_residual_tolerances(ts, te, ti, tc);
        if (opts_.qp_mode == QpMode::kFullCondensed)
        {
            qp_dense_.set_residual_tolerances(ts, te, ti, tc);
        }
        if constexpr (NB > 0)
        {
            if (opts_.qp_mode == QpMode::kPartialCondensed)
            {
                block_qp_.solver.set_residual_tolerances(ts, te, ti, tc);
            }
        }
    }

    /// Phase 5: when print_level >= 2, dump the QP solver's per-IPM-iteration
    /// table (HPIPM stat, acados ocp_qp_hpipm.c:367-373) after each QP solve.
    /// Acados instead forwards print_level 1 to the QP solver for NLP levels
    /// > 2; here it is driven directly by the SQP print_level.
    template <class StatsT>
    void print_qp_stats(const StatsT& st) const
    {
        if (opts_.print_level > 1)
        {
            st.print();
        }
    }

    /// Solve the QP of the current SQP iteration, dispatching on
    /// opts_.qp_mode (phase 1e).  kStaged: the plain staged HPIPM solve.
    /// kFullCondensed: condense the staged QP to the dense workspace, solve
    /// with the dense HPIPM IPM, expand the solution back into the staged
    /// `out` layout (acados FULL_CONDENSING_HPIPM).
    ///
    /// The primal step is always cold (zeroed before the solve); the duals
    /// warm-start through the persistent workspaces (same convention as the
    /// staged path, phase 3c).
    Status solve_qp(const Qp<P, NH>& in, QpSol<P, NH>& out, int ws)
    {
        if (opts_.qp_mode == QpMode::kFullCondensed)
        {
            const Status st = solve_qp_full_condensed(in, out, ws);
            qp_iter_ = qp_dense_.statistics().iter;
            return st;
        }
        if (opts_.qp_mode == QpMode::kPartialCondensed)
        {
            return solve_qp_partial_condensed(in, out, ws);
        }
        out.ux_first.setZero();
        for (int k = 1; k < out.N; ++k)
        {
            out.ux_path[k - 1].setZero();
        }
        out.ux_term.setZero();
        qp_.set_warm_start(ws);
        const Status st = qp_.solve(in, out);
        print_qp_stats(qp_.statistics());
        qp_iter_ = qp_.statistics().iter;
        return st;
    }

    /// Full-condensing QP solve (phase 1e).  `in` is the (regularized,
    /// scaled) staged QP; `out` receives the staged-layout solution.  The
    /// dense primal v is zeroed (cold primal); lam / t warm-start from the
    /// persistent dense_qp_out_.  cond_time_ accumulates the wall clock of
    /// condense + expand (the dense IPM iterations show up in qp_iter_).
    Status solve_qp_full_condensed(const Qp<P, NH>& in, QpSol<P, NH>& out,
                                   int ws)
    {
        const auto t0 = std::chrono::steady_clock::now();
        condense_full(in, dense_qp_in_);
        dense_qp_out_.v.setZero();
        qp_dense_.set_warm_start(ws);
        const Status st = qp_dense_.solve(dense_qp_in_, dense_qp_out_);
        print_qp_stats(qp_dense_.statistics());
        expand_full_sol(in, dense_qp_in_, dense_qp_out_, out);
        cond_time_ += std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
        return st;
    }

    /// Partial-condensing QP solve (phase 2e, acados
    /// PARTIAL_CONDENSING_HPIPM).  Condenses the staged QP `in` into
    /// N2 = N / NB blocks (internal states of each block eliminated),
    /// solves the N2-stage block QP with the staged HPIPM IPM over
    /// BlockQpDim<P, NB>, and expands the solution back into the staged
    /// `out` layout.  The primal step is cold (zeroed); the duals
    /// warm-start through the persistent block_qp_.out (same convention
    /// as the other modes).  cond_time_ accumulates the wall clock of
    /// condense + expand (the block IPM iterations count into qp_iter_).
    ///
    /// Falls back to the plain staged solve (identity fast path) when the
    /// block size does not match this solver's compile-time NB:
    /// NB == 0, qp_cond_N >= N (=> block size 1), or N / qp_cond_N != NB.
    Status solve_qp_partial_condensed(const Qp<P, NH>& in, QpSol<P, NH>& out,
                                      int ws)
    {
        if constexpr (NB > 0)
        {
            const int N = in.N;
            const int N2 = (opts_.qp_cond_N > 0) ? opts_.qp_cond_N : N;
            if (N2 >= N || N % N2 != 0 || N / N2 != NB)
            {
                // identity fast path / block size mismatch: plain staged
                // solve
                out.ux_first.setZero();
                for (int k = 1; k < out.N; ++k)
                {
                    out.ux_path[k - 1].setZero();
                }
                out.ux_term.setZero();
                qp_.set_warm_start(ws);
                const Status st = qp_.solve(in, out);
                print_qp_stats(qp_.statistics());
                qp_iter_ = qp_.statistics().iter;
                return st;
            }
            const auto t0 = std::chrono::steady_clock::now();
            condense_block<P, NH, NB>(in, block_qp_.in);
            // resize() is a no-op for an unchanged horizon and preserves
            // the persistent duals (warm start); only the primal is
            // zeroed.
            block_qp_.out.resize(N2);
            block_qp_.out.ux_first.setZero();
            for (int k = 1; k < N2; ++k)
            {
                block_qp_.out.ux_path[k - 1].setZero();
            }
            block_qp_.out.ux_term.setZero();
            block_qp_.solver.set_warm_start(ws);
            const Status st =
                block_qp_.solver.solve(block_qp_.in, block_qp_.out);
            print_qp_stats(block_qp_.solver.statistics());
            expand_block_sol<P, NH, Eigen::Dynamic, NB>(
                in, block_qp_.in, block_qp_.out, out);
            cond_time_ += std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
            qp_iter_ = block_qp_.solver.statistics().iter;
            return st;
        }
        else
        {
            // NB == 0: partial condensing disabled; plain staged solve.
            out.ux_first.setZero();
            for (int k = 1; k < out.N; ++k)
            {
                out.ux_path[k - 1].setZero();
            }
            out.ux_term.setZero();
            qp_.set_warm_start(ws);
            const Status st = qp_.solve(in, out);
            print_qp_stats(qp_.statistics());
            qp_iter_ = qp_.statistics().iter;
            return st;
        }
    }

    /// Inf-norm of the primal step, max over stages (acados
    /// ocp_qp_out_compute_primal_nrm_inf, ocp_qp_common.c:263-279;
    /// includes the slack variables: the whole stage variable vector).
    double primal_step_norm_inf(const QpSol<P, NH>& step) const
    {
        double m = static_cast<double>(step.ux_first.cwiseAbs().maxCoeff());
        for (int k = 1; k < step.N; ++k)
        {
            m = std::max(
                m,
                static_cast<double>(
                    step.ux_path[k - 1].cwiseAbs().maxCoeff()));
        }
        m = std::max(
            m,
            static_cast<double>(step.ux_term.cwiseAbs().maxCoeff()));
        return m;
    }

    /// Update the solver-internal slacks after an accepted step.
    ///
    /// The QP slack variables are ABSOLUTE (qp.hpp sec. "Side layout":
    /// the d offsets hold lo - w_cur / w_cur - hi with no s_cur), so the
    /// iterate slacks are interpolated, not accumulated:
    /// s <- s + alpha * (s_qp - s).
    void update_slacks(const QpSol<P, NH>& step, double alpha)
    {
        slacks_.first = slacks_.first +
            alpha * (step.ux_first.segment(D::nu + D::nx,
                                           D::nslack_first) -
                     slacks_.first);
        for (int k = 1; k < step.N; ++k)
        {
            slacks_.path[k - 1] =
                slacks_.path[k - 1] +
                alpha * (step.ux_path[k - 1].segment(D::nu + D::nx,
                                                     D::nslack_path) -
                         slacks_.path[k - 1]);
        }
        slacks_.term = slacks_.term +
            alpha * (step.ux_term.segment(D::nx, D::nslack_term) -
                     slacks_.term);
    }

    // ---------------------------------------------------------------
    //  3j: second-order-correction (SOC) pre-pass
    // ---------------------------------------------------------------

    /// Second-order-correction re-solve (acados
    /// ocp_nlp_common.c:4290-4442). Rewrites the QP RHS at the full step
    /// (dynamics `b`, general-row `d`), re-solves the regularized QP in
    /// place, and corrects the duals. `sol` is the current iterate (the
    /// step has NOT been applied yet); `qp_out_` holds the full step
    /// (alpha = 1). Returns kQpFailure if the re-solve fails.
    Status perform_second_order_correction(const P& problem,
                                           const Solution<P, NH>& sol)
    {
        soc_rewrite_rhs(problem, sol);
        reg_.regularize(qp_in_);
        const Status st = solve_qp(qp_in_, qp_out_, qp_ws_);
        if (st != Status::kSolved && st != Status::kMaxIterations)
        {
            return Status::kQpFailure;
        }
        reg_.correct_dual_sol(qp_in_, qp_out_);
        ++soc_count_;
        return Status::kSolved;
    }

    /// Rewrite the QP right-hand sides at the full step. The corrected QP
    /// keeps the current-point Jacobians (BA, DC) and Hessian; only the
    /// right-hand sides are re-evaluated at the trial iterate (x + dx,
    /// u + du) (Nocedal & Wright 18.51, the acados second-order
    /// correction):
    ///
    ///   b_k  <- f_k(x_k + dx_k, u_k + du_k) - x_{k+1} - dx_{k+1}
    ///   d_lo <- lo - c_t,  d_hi <- c_t - hi   (general rows)
    ///
    /// Box / pin rows are linear and need no second-order correction.
    void soc_rewrite_rhs(const P& problem, const Solution<P, NH>& sol)
    {
        const int N = qp_in_.N;

        // dynamics: b_k <- f(x+dx, u+du) - x_{k+1} - dx_{k+1}
        for (int k = 0; k < N; ++k)
        {
            Eigen::Matrix<S, D::nx, 1> dxk;
            Eigen::Matrix<S, D::nu, 1> duk;
            if (k == 0)
            {
                dxk = qp_out_.ux_first.segment(D::nu, D::nx);
                duk = qp_out_.ux_first.head(D::nu);
            }
            else
            {
                dxk = qp_out_.ux_path[k - 1].segment(D::nu, D::nx);
                duk = qp_out_.ux_path[k - 1].head(D::nu);
            }

            // state step at stage k + 1 (x part): path vs terminal extent
            Eigen::Matrix<S, D::nx, 1> dx_next;
            if (k + 1 == N)
            {
                dx_next = qp_out_.ux_term.head(D::nx);
            }
            else
            {
                dx_next = qp_out_.ux_path[k].segment(D::nu, D::nx);
            }

            const auto xk = sol.x[k] + dxk;
            const auto uk = sol.u[k] + duk;
            const auto ftrial = problem.dynamics_next_state(k, xk, uk);
            const auto gap = ftrial - sol.x[k + 1] - dx_next;

            if (k == 0)
            {
                qp_in_.first.b = gap;
            }
            else
            {
                qp_in_.path[k - 1].b = gap;
            }
        }

        // general rows (ineq / eq / lin): first, path, terminal
        soc_rewrite_d_stage(problem, sol, qp_in_.first, D::lay_first,
                            D::idxs_lo_first, D::idxs_hi_first,
                            qp_out_.ux_first, 0);
        for (int k = 1; k < N; ++k)
        {
            soc_rewrite_d_stage(problem, sol, qp_in_.path[k - 1],
                                D::lay_path, D::idxs_lo_path,
                                D::idxs_hi_path, qp_out_.ux_path[k - 1], k);
        }
        soc_rewrite_d_term(problem, sol, qp_in_.term, D::lay_term,
                           D::idxs_lo_term, D::idxs_hi_term,
                           qp_out_.ux_term);
    }

    /// SOC d-rewrite for one path / first stage (ineq / eq / lin rows).
    ///
    /// The assembly convention is d_lo = lo - w_cur, d_hi = w_cur - hi
    /// (offset from the row value at the current iterate). The HPIPM row
    /// value v already carries DC_ux . dz and the slack columns, so the
    /// second-order correction only re-bases the offset to the nonlinear
    /// row value at the trial iterate:
    ///
    ///   d_lo <- lo - c_t
    ///   d_hi <- c_t - hi
    ///
    /// Box / pin rows are skipped (linear, no second-order term).
    template <class St, class IdxLo, class IdxHi, class Step>
    void soc_rewrite_d_stage(const P& problem, const Solution<P, NH>& sol,
                              St& st, const detail::QpLayout& lay,
                              const IdxLo& /*idxl*/, const IdxHi& /*idxh*/,
                              const Step& step, int k)
    {
        const int R = lay.nrow();

        const auto x_trial = sol.x[k] + step.segment(D::nu, D::nx);
        const auto u_trial = sol.u[k] + step.head(D::nu);

        typename P::ineq_t gv{};
        typename P::eq_t ev{};
        typename P::lin_t lv{};
        typename P::stage_linear_t lin_spec{};
        if constexpr (D::ng > 0)
            gv = problem.stage_inequality_value(k, x_trial, u_trial);
        if constexpr (D::ne > 0)
            ev = problem.stage_equality_value(k, x_trial, u_trial);
        if constexpr (D::nl > 0)
        {
            lin_spec = problem.stage_linear_constr(k);
            lv = lin_spec.A * x_trial + lin_spec.B * u_trial;
        }

        for (int r = 0; r < R; ++r)
        {
            const int g = lay.group_of(r);
            if (g != detail::g_ineq && g != detail::g_eq
                && g != detail::g_lin)
                continue;
            const int j = r - lay.row_off(g);

            double lo = 0.0, hi = 0.0, c_trial = 0.0;
            if (g == detail::g_ineq)
                c_trial = static_cast<double>(gv(j));
            else if (g == detail::g_eq)
                c_trial = static_cast<double>(ev(j));
            else  // g_lin
            {
                lo = static_cast<double>(lin_spec.bounds.lo(j));
                hi = static_cast<double>(lin_spec.bounds.hi(j));
                c_trial = static_cast<double>(lv(j));
            }

            st.d(lay.side_hi(r)) = static_cast<S>(c_trial - hi);
            const int slo = lay.side_lo(r);
            if (slo >= 0)
                st.d(slo) = static_cast<S>(lo - c_trial);
        }
    }

    /// SOC d-rewrite for the terminal stage (ineq / eq / lin rows).
    /// Mirrors soc_rewrite_d_stage with the terminal layout (x-only).
    template <class St, class IdxLo, class IdxHi, class Step>
    void soc_rewrite_d_term(const P& problem, const Solution<P, NH>& sol,
                            St& st, const detail::QpLayout& lay,
                            const IdxLo& /*idxl*/, const IdxHi& /*idxh*/,
                            const Step& step)
    {
        const int R = lay.nrow();

        const auto x_trial = sol.x[sol.N] + step.head(D::nx);

        typename P::ineq_term_t gv{};
        typename P::eq_term_t ev{};
        typename P::lin_term_t lv{};
        typename P::term_linear_t lin_spec{};
        if constexpr (D::ng_t > 0)
            gv = problem.terminal_inequality_value(x_trial);
        if constexpr (D::ne_t > 0)
            ev = problem.terminal_equality_value(x_trial);
        if constexpr (D::nl_t > 0)
        {
            lin_spec = problem.terminal_linear_constr();
            lv = lin_spec.A * x_trial;
        }

        for (int r = 0; r < R; ++r)
        {
            const int g = lay.group_of(r);
            if (g != detail::g_ineq && g != detail::g_eq
                && g != detail::g_lin)
                continue;
            const int j = r - lay.row_off(g);

            double lo = 0.0, hi = 0.0, c_trial = 0.0;
            if (g == detail::g_ineq)
                c_trial = static_cast<double>(gv(j));
            else if (g == detail::g_eq)
                c_trial = static_cast<double>(ev(j));
            else  // g_lin
            {
                lo = static_cast<double>(lin_spec.bounds.lo(j));
                hi = static_cast<double>(lin_spec.bounds.hi(j));
                c_trial = static_cast<double>(lv(j));
            }

            st.d(lay.side_hi(r)) = static_cast<S>(c_trial - hi);
            const int slo = lay.side_lo(r);
            if (slo >= 0)
                st.d(slo) = static_cast<S>(lo - c_trial);
        }
    }

    Status assemble_first(const P& problem, const Solution<P, NH>& sol)
    {
        const detail::AssembleFlags fl{opts_.compute_hess_dyn,
                                       opts_.compute_hess_constr};
        auto ev = detail::eval_stage(problem, sol, 0, fl);
        detail::fill_stage_lhs_first(problem, ev, fl, qp_in_.first);
        return detail::fill_stage_rhs_first(problem, sol, ev, qp_in_.first);
    }

    void assemble_path(const P& problem, const Solution<P, NH>& sol, int k)
    {
        const detail::AssembleFlags fl{opts_.compute_hess_dyn,
                                       opts_.compute_hess_constr};
        auto ev = detail::eval_stage(problem, sol, k, fl);
        detail::fill_stage_lhs_path(problem, k, ev, fl,
                                    qp_in_.path[k - 1]);
        detail::fill_stage_rhs_path(problem, sol, k, ev,
                                    qp_in_.path[k - 1]);
    }

    void assemble_term(const P& problem, const Solution<P, NH>& sol)
    {
        const detail::AssembleFlags fl{opts_.compute_hess_dyn,
                                       opts_.compute_hess_constr};
        auto ev = detail::eval_term(problem, sol, fl);
        detail::fill_stage_lhs_term(problem, ev, fl, qp_in_.term);
        detail::fill_stage_rhs_term(problem, sol, ev, qp_in_.term);
    }

    // ---------------------------------------------------------------
    //  Members
    // ---------------------------------------------------------------

    AcadosOptions opts_;
    QpScaler<P, NH> scaler_;
    QpSolver qp_;
    Regularizer reg_;
    Globalizer glob_;
    // Phase 1 (full condensing): dense QP workspace + solver, used when
    // opts_.qp_mode == QpMode::kFullCondensed.
    HpipmDenseQpSolver<P, NH> qp_dense_;
    DenseQp<P, NH> dense_qp_in_;
    DenseQpSol<P, NH> dense_qp_out_;
    // Phase 2 (partial condensing): block QP workspace + solver, used when
    // opts_.qp_mode == QpMode::kPartialCondensed and NB > 0. Always
    // dynamic-horizon (N2 = N / NB is a runtime quantity); with NB == 0
    // these fall back to plain staged types and are never used.
    detail::BlockQpWorkspaces<P, NB> block_qp_;
    AcadosStatistics stat_;
    Qp<P, NH> qp_in_;
    QpSol<P, NH> qp_out_;
    Solution<P, NH> trial_;
    SqpSlacks<P, NH> slacks_;
    int slack_N_ = 0;  // horizon of the current slacks_ (warm-start check)
    double alpha_ = 0.0;
    double step_norm_ = 0.0;
    double cost_value_ = 0.0;
    int qp_status_ = 0;  // last QP status (0 = none yet)
    int qp_iter_ = 0;    // last QP iteration count
    double timeout_estimated_per_iter_ = 0.0;  // 3f: per-iter time estimate
    double lm_mu_ = 0.0;     // 3g: adaptive LM mu
    double lm_mu_bar_ = 0.0; // 3g: adaptive LM mu_bar (previous mu)
    int soc_count_ = 0;      // 3j: number of SOC re-solves performed
    double cond_time_ = 0.0; // 1e: total condense + expand wall clock
    int qp_ws_ = 0;          // 1e: warm-start mode of the last QP solve
};

}  // namespace ocp

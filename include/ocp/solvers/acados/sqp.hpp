// ocp/solvers/acados/sqp.hpp
//
// SQP driver components (phase 2).
//
// This file is built up incrementally across sub-steps 2c-2g of
// SQP_PHASE2_PLAN.md.
//
// 2c: NlpResiduals, compute_nlp_residuals
//     Port of acados ocp_nlp_res_compute (ocp_nlp_common.c:3743-3846).
// 2e: SqpOptions, SqpIteration, SqpStatistics, print_sqp_iteration
//     Driver options (defaults per ocp_nlp_common.c:1215-1287,
//     ocp_nlp_sqp.c:103-119), per-iteration statistics (acados `stat`
//     matrix, ocp_nlp_sqp.c:266-271) and the stdout printing
//     (ocp_nlp_sqp.c:449-466).
//
// References (re-read at implementation time):
//   acados/ocp_nlp/ocp_nlp_common.c:3743-3846  (ocp_nlp_res_compute)
//   acados/ocp_nlp/ocp_nlp_sqp.c:351-430        (check_termination)
//   acados/ocp_nlp/ocp_nlp_sqp.c:449-466        (print_iteration)
//   acados/ocp_nlp/ocp_nlp_constraints_bgp.c    (fun layout, dmask)

#pragma once

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "../hpipm/qp.hpp"
#include "globalize.hpp"

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
/// @param tau_min  complementarity floor added to every active non-equality
///                 side (acados quirk: added, not subtracted; sec. 1.4).
template <class P, int NH = Eigen::Dynamic>
NlpResiduals compute_nlp_residuals(const P& problem,
                                   const Solution<P, NH>& sol,
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

        auto grad = problem.stage_cost_gradient(k, x, u);

        typename P::dyn_df_dx_t df_dx;
        typename P::dyn_df_du_t df_du;
        problem.dynamics_jacobian(k, x, u, df_dx, df_du);

        // constraint Jacobians (computed once, reused for u and x parts)
        typename P::ineq_dg_dx_t g_dx;
        typename P::ineq_dg_du_t g_du;
        if constexpr (D::ng > 0)
            problem.stage_inequality_constr_jacobian(k, x, u, g_dx, g_du);

        typename P::eq_de_dx_t e_dx;
        typename P::eq_de_du_t e_du;
        if constexpr (D::ne > 0)
            problem.stage_equality_constr_jacobian(k, x, u, e_dx, e_du);

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
        Eigen::Matrix<S, D::nx, 1> adj_x =
            problem.terminal_cost_gradient(xN);
        adj_x += sol.lambda_dyn[N - 1];

        if constexpr (D::ng_t > 0)
        {
            typename P::ineq_term_dg_dx_t g_dx;
            problem.terminal_inequality_constr_jacobian(xN, g_dx);
            adj_x += g_dx.transpose() * sol.lambda_ineq_term;
        }
        if constexpr (D::ne_t > 0)
        {
            typename P::eq_term_de_dx_t e_dx;
            problem.terminal_equality_constr_jacobian(xN, e_dx);
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

        if constexpr (D::ng > 0)
        {
            const auto g = problem.stage_inequality_constr(k, x, u);
            for (int j = 0; j < D::ng; ++j)
                res.res_ineq = std::max(res.res_ineq,
                    static_cast<double>(std::max(0.0, g(j))));
        }
        if constexpr (D::ne > 0)
        {
            const auto e = problem.stage_equality_constr(k, x, u);
            for (int j = 0; j < D::ne; ++j)
                res.res_ineq = std::max(res.res_ineq,
                    static_cast<double>(std::fabs(e(j))));
        }
        if constexpr (D::nl > 0)
        {
            const auto lin_spec = problem.stage_linear_constr(k);
            const auto v = lin_spec.A * x + lin_spec.B * u;
            for (int j = 0; j < D::nl; ++j)
            {
                res.res_ineq = std::max(res.res_ineq,
                    static_cast<double>(
                        std::max(0.0, v(j) - lin_spec.bounds.hi(j))));
                res.res_ineq = std::max(res.res_ineq,
                    static_cast<double>(
                        std::max(0.0, lin_spec.bounds.lo(j) - v(j))));
            }
        }
        if constexpr (D::nbx > 0)
        {
            const auto spec = problem.stage_state_box_constr(k);
            for (int j = 0; j < D::nbx; ++j)
            {
                const double v = static_cast<double>(
                    x(P::state_box_idx[j]));
                res.res_ineq = std::max(res.res_ineq,
                    std::max(0.0,
                        v - static_cast<double>(spec.hi(j))));
                res.res_ineq = std::max(res.res_ineq,
                    std::max(0.0,
                        static_cast<double>(spec.lo(j)) - v));
            }
        }
        if constexpr (D::nbu > 0)
        {
            const auto spec = problem.stage_control_box_constr(k);
            for (int j = 0; j < D::nbu; ++j)
            {
                const double v = static_cast<double>(
                    u(P::control_box_idx[j]));
                res.res_ineq = std::max(res.res_ineq,
                    std::max(0.0,
                        v - static_cast<double>(spec.hi(j))));
                res.res_ineq = std::max(res.res_ineq,
                    std::max(0.0,
                        static_cast<double>(spec.lo(j)) - v));
            }
        }
    }

    // terminal
    {
        const auto& xN = sol.x[N];
        if constexpr (D::ng_t > 0)
        {
            const auto g = problem.terminal_inequality_constr(xN);
            for (int j = 0; j < D::ng_t; ++j)
                res.res_ineq = std::max(res.res_ineq,
                    static_cast<double>(std::max(0.0, g(j))));
        }
        if constexpr (D::ne_t > 0)
        {
            const auto e = problem.terminal_equality_constr(xN);
            for (int j = 0; j < D::ne_t; ++j)
                res.res_ineq = std::max(res.res_ineq,
                    static_cast<double>(std::fabs(e(j))));
        }
        if constexpr (D::nl_t > 0)
        {
            const auto lin_spec = problem.terminal_linear_constr();
            const auto v = lin_spec.A * xN;
            for (int j = 0; j < D::nl_t; ++j)
            {
                res.res_ineq = std::max(res.res_ineq,
                    static_cast<double>(
                        std::max(0.0, v(j) - lin_spec.bounds.hi(j))));
                res.res_ineq = std::max(res.res_ineq,
                    static_cast<double>(
                        std::max(0.0, lin_spec.bounds.lo(j) - v(j))));
            }
        }
        if constexpr (D::nbx_t > 0)
        {
            const auto spec = problem.terminal_state_box_constr();
            for (int j = 0; j < D::nbx_t; ++j)
            {
                const double v = static_cast<double>(
                    xN(P::terminal_state_box_idx[j]));
                res.res_ineq = std::max(res.res_ineq,
                    std::max(0.0,
                        v - static_cast<double>(spec.hi(j))));
                res.res_ineq = std::max(res.res_ineq,
                    std::max(0.0,
                        static_cast<double>(spec.lo(j)) - v));
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

        // box state (skipped at stage 0 when x_0 is fixed: pin rows
        // replace the state box, so no box multipliers exist)
        if constexpr (D::nbx > 0)
        {
            if (!(k == 0 && P::fixed_initial_state))
            {
                const auto spec = problem.stage_state_box_constr(k);
                for (int j = 0; j < D::nbx; ++j)
                {
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
                        std::fabs(lam_lo * std::max(0.0, lo - v)
                                  + tau_min));
                    stage_max = std::max(stage_max,
                        std::fabs(lam_hi * std::max(0.0, v - hi)
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
                    std::fabs(lam_lo * std::max(0.0, lo - v)
                              + tau_min));
                stage_max = std::max(stage_max,
                    std::fabs(lam_hi * std::max(0.0, v - hi)
                              + tau_min));
            }
        }

        // ineq (one-sided: hi side only, hi = 0)
        if constexpr (D::ng > 0)
        {
            const auto g = problem.stage_inequality_constr(k, x, u);
            for (int j = 0; j < D::ng; ++j)
            {
                const double v = static_cast<double>(g(j));
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
                    entry = net * std::max(0.0, val - hi) + tau_min;
                else
                    entry = (-net) * std::max(0.0, lo - val) + tau_min;
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

        if constexpr (D::nbx_t > 0)
        {
            const auto spec = problem.terminal_state_box_constr();
            for (int j = 0; j < D::nbx_t; ++j)
            {
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
                    std::fabs(lam_lo * std::max(0.0, lo - v)
                              + tau_min));
                stage_max = std::max(stage_max,
                    std::fabs(lam_hi * std::max(0.0, v - hi)
                              + tau_min));
            }
        }

        if constexpr (D::ng_t > 0)
        {
            const auto g = problem.terminal_inequality_constr(xN);
            for (int j = 0; j < D::ng_t; ++j)
            {
                const double v = static_cast<double>(g(j));
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
                const double val = static_cast<double>(v(j));
                const double lo =
                    static_cast<double>(lin_spec.bounds.lo(j));
                const double hi =
                    static_cast<double>(lin_spec.bounds.hi(j));
                const double net =
                    static_cast<double>(sol.lambda_lin_term(j));
                double entry;
                if (net >= 0.0)
                    entry = net * std::max(0.0, val - hi) + tau_min;
                else
                    entry = (-net) * std::max(0.0, lo - val) + tau_min;
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
struct SqpOptions
{
    int max_iter = 20;
    double tol_stat = 1e-8, tol_eq = 1e-8, tol_ineq = 1e-8, tol_comp = 1e-8;
    double tol_min_step_norm = 1e-12;
    double tol_unbounded = -1e10;
    bool compute_hess = true;             // gates dynamics/constraint HVP terms
    double levenberg_marquardt = 0.0;
    bool with_adaptive_lm = false;        // phase 3 (inert in v1)
    int print_level = 0;                  // 0 = silent, 1 = per-iteration rows
    int qp_warm_start = 0;                // v1: inert (HPIPM cold-starts each QP)
    bool warm_start_first_qp = false;     // v1: inert
    bool eval_residual_at_max_iter = false;
    double timeout_max_time = 0.0;        // phase 3 (inert in v1)
    bool scale_qp_objective = false;      // phase 3 (inert in v1)
    bool scale_qp_constraints = false;
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
};

/// Per-iteration records of a SQP solve (acados `stat` matrix analogue).
///
/// `iter` is the number of recorded iterations.  `status` is the final
/// exit status, set by the driver before returning.
struct SqpStatistics
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

}  // namespace ocp

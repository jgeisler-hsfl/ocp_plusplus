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
// 2f: SqpSolver::assemble_qp / add_lm_term / resize (QP assembly).
// 2g: SqpSolver::solve / check_termination / compute_cost (driver).
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
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "../hpipm/qp.hpp"
#include "../hpipm/hpipm.hpp"
#include "globalize.hpp"
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

// =========================================================================
//  SqpSolver (sub-step 2f: QP assembly + LM term)
// =========================================================================

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
          class Globalizer = MeritBacktracking<P, NH>>
class SqpSolver
{
public:
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    explicit SqpSolver(SqpOptions opts = SqpOptions{})
        : opts_(opts)
    {
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
    /// diagonal). No-op when compute_hess is false or mu <= 0.
    /// (acados ocp_nlp_add_levenberg_marquardt_term, ocp_nlp_common.c:3034;
    /// deviation: the acados cost-module `scaling` factor has no
    /// counterpart in our interface, the examples use scaling = 1.)
    void add_lm_term(Qp<P, NH>& qp, double mu) const
    {
        if (!opts_.compute_hess || mu <= 0.0)
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

    /// Size the internal workspaces for a horizon of `n_stages` stages
    /// (dynamic-extent mode allocates; fixed-extent mode only asserts).
    void resize(int n_stages)
    {
        qp_in_.resize(n_stages);
        qp_out_.resize(n_stages);
        trial_.resize(n_stages);
        slacks_.resize(n_stages);
    }

    const SqpOptions& options() const { return opts_; }
    const SqpStatistics& statistics() const { return stat_; }
    const Qp<P, NH>& last_qp() const { return qp_in_; }
    const QpSol<P, NH>& last_qp_sol() const { return qp_out_; }

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
    ///   otherwise                      -> kUnset (keep iterating)
    Status check_termination(int iter, const NlpResiduals& res,
                             double step_norm, double cost) const
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
        return Status::kUnset;
    }

    Status solve(const P& problem, Solution<P, NH>& sol)
    {
        const int N = sol.N;

        stat_ = SqpStatistics{};
        qp_status_ = 0;
        qp_iter_ = 0;
        alpha_ = 0.0;
        step_norm_ = 0.0;

        resize(N);
        if (slack_N_ != N)
        {
            slacks_.setZero();
        }
        slack_N_ = N;

        cost_value_ = compute_cost(problem, sol, slacks_);
        sol.cost_value = cost_value_;
        sol.status = Status::kUnset;

        for (int iter = 0; iter <= opts_.max_iter; ++iter)
        {
            Status st = assemble_qp(problem, sol);
            if (st != Status::kSolved)
            {
                sol.status = st;
                stat_.status = st;
                return st;
            }
            add_lm_term(qp_in_, opts_.levenberg_marquardt);

            const NlpResiduals res =
                compute_nlp_residuals(problem, sol, opts_.tau_min);

            if (iter == 0)
            {
                glob_.initialize(problem, sol);
            }

            // Before the termination check (acados ocp_nlp_sqp.c:599-600):
            // the returned QP is the one that would be solved at a
            // stationary iterate.
            reg_.regularize(qp_in_);

            const Status term =
                check_termination(iter, res, step_norm_, cost_value_);
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

            const Status qp_st = qp_.solve(qp_in_, qp_out_);
            qp_status_ = static_cast<int>(qp_st);
            qp_iter_ = qp_.statistics().iter;
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

            step_norm_ = primal_step_norm_inf(qp_out_);

            // Advances sol in place on kSolved / kMinStep (2d); leaves it
            // untouched on kNanDetected.
            const Status gstatus = glob_.find_acceptable_iterate(
                problem, sol, qp_out_, slacks_, trial_, alpha_);
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
        return row;
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

    /// NLP objective at the iterate: the problem's stage + terminal costs
    /// plus the slack penalty 0.5 * w * s^2 per soft side (sec. 1.6; the
    /// problem cost functions exclude the solver-internal slacks).
    double compute_cost(const P& problem, const Solution<P, NH>& sol,
                        const SqpSlacks<P, NH>& sl) const
    {
        double m = 0.0;
        for (int k = 0; k < sol.N; ++k)
        {
            m += static_cast<double>(
                problem.stage_cost_value(k, sol.x[k], sol.u[k]));
        }
        m += static_cast<double>(problem.terminal_cost_value(sol.x[sol.N]));
        return m + slack_penalty(problem, sl);
    }

    /// Slack penalty 0.5 * w * s^2 over every soft side (same weights and
    /// slack order as fill_slack_diag / fill_slack_diag_term; the first
    /// stage drops the state-box slacks when x_0 is fixed, so the bx lo
    /// block is absent there — guarded via the stage-type layout).
    double slack_penalty(const P& problem,
                         const SqpSlacks<P, NH>& sl) const
    {
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
                        const int r =
                            lay.row_off(detail::g_bx)
                            + P::state_box_soft_idx[i];
                        m += slack_pair(sv, base, idxl[r], idxh[r],
                                        static_cast<double>(
                                            spec.soft_penalty(
                                                P::state_box_soft_idx[i])));
                    }
                }
            }
            if constexpr (P::nbu_soft > 0)
            {
                const auto spec = problem.stage_control_box_constr(k);
                for (int i = 0; i < P::nbu_soft; ++i)
                {
                    const int r =
                        lay.row_off(detail::g_bu)
                        + P::control_box_soft_idx[i];
                    m += slack_pair(sv, base, idxl[r], idxh[r],
                                    static_cast<double>(
                                        spec.soft_penalty(
                                            P::control_box_soft_idx[i])));
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
                    m += slack_pair(sv, base, idxl[r], idxh[r],
                                    static_cast<double>(pen(i)));
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
                    m += slack_pair(sv, base, idxl[r], idxh[r],
                                    static_cast<double>(pen(i)));
                }
            }
            if constexpr (P::nl_soft > 0)
            {
                const auto spec = problem.stage_linear_constr(k);
                for (int i = 0; i < P::nl_soft; ++i)
                {
                    const int r =
                        lay.row_off(detail::g_lin) + P::lin_soft_idx[i];
                    m += slack_pair(sv, base, idxl[r], idxh[r],
                                    static_cast<double>(
                                        spec.bounds.soft_penalty(
                                            P::lin_soft_idx[i])));
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
                    const int r = lay.row_off(detail::g_bx)
                                 + P::terminal_state_box_soft_idx[i];
                    m += slack_pair(sv, base, D::idxs_lo_term[r],
                                    D::idxs_hi_term[r],
                                    static_cast<double>(
                                        spec.soft_penalty(
                                            P::terminal_state_box_soft_idx[i])));
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
                    m += slack_pair(sv, base, D::idxs_lo_term[r],
                                    D::idxs_hi_term[r],
                                    static_cast<double>(pen(i)));
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
                    m += slack_pair(sv, base, D::idxs_lo_term[r],
                                    D::idxs_hi_term[r],
                                    static_cast<double>(pen(i)));
                }
            }
            if constexpr (P::nl_t_soft > 0)
            {
                const auto spec = problem.terminal_linear_constr();
                for (int i = 0; i < P::nl_t_soft; ++i)
                {
                    const int r = lay.row_off(detail::g_lin)
                                 + P::terminal_lin_soft_idx[i];
                    m += slack_pair(sv, base, D::idxs_lo_term[r],
                                    D::idxs_hi_term[r],
                                    static_cast<double>(
                                        spec.bounds.soft_penalty(
                                            P::terminal_lin_soft_idx[i])));
                }
            }
        }
        return m;
    }

    /// 0.5 * w * (s_lo^2 + s_hi^2) for one soft row's slack columns
    /// (absent sides, col < 0, contribute nothing).
    template <class SV>
    static double slack_pair(const SV& sv, int base, int col_lo, int col_hi,
                             double w)
    {
        double m = 0.0;
        if (col_lo >= 0)
        {
            const double s = static_cast<double>(sv[col_lo - base]);
            m += 0.5 * w * s * s;
        }
        if (col_hi >= 0)
        {
            const double s = static_cast<double>(sv[col_hi - base]);
            m += 0.5 * w * s * s;
        }
        return m;
    }

    // ---------------------------------------------------------------
    //  HVP matrix builders (nx+nu unit-vector calls, plan sec. 1.5)
    // ---------------------------------------------------------------

    /// (u;x)-layout matrix M with M v = dynamics HVP(w, v_x, v_u).
    Eigen::Matrix<S, D::nu + D::nx, D::nu + D::nx>
    build_dyn_hvp(int k, const P& problem, const Solution<P, NH>& sol) const
    {
        using Mat = Eigen::Matrix<S, D::nu + D::nx, D::nu + D::nx>;
        Mat mat;
        mat.setZero();
        const auto& x = sol.x[k];
        const auto& u = sol.u[k];
        const auto& w = sol.lambda_dyn[k];
        for (int i = 0; i < D::nu + D::nx; ++i)
        {
            typename P::state_t v_x;
            typename P::control_t v_u;
            v_x.setZero();
            v_u.setZero();
            if (i < D::nu)
            {
                v_u(i) = 1;
            }
            else
            {
                v_x(i - D::nu) = 1;
            }
            typename P::state_t hv_x;
            typename P::control_t hv_u;
            problem.dynamics_hess_prod(k, x, u, w, v_x, v_u, hv_x, hv_u);
            mat.col(i).head(D::nu) = hv_u;
            mat.col(i).tail(D::nx) = hv_x;
        }
        return mat;
    }

    /// (u;x)-layout matrix M with M v = stage-inequality HVP.
    Eigen::Matrix<S, D::nu + D::nx, D::nu + D::nx>
    build_ineq_hvp(int k, const P& problem, const Solution<P, NH>& sol) const
    {
        using Mat = Eigen::Matrix<S, D::nu + D::nx, D::nu + D::nx>;
        Mat mat;
        mat.setZero();
        if constexpr (D::ng > 0)
        {
            const auto& x = sol.x[k];
            const auto& u = sol.u[k];
            const auto& w = sol.lambda_ineq_stage[k];
            for (int i = 0; i < D::nu + D::nx; ++i)
            {
                typename P::state_t v_x;
                typename P::control_t v_u;
                v_x.setZero();
                v_u.setZero();
                if (i < D::nu)
                {
                    v_u(i) = 1;
                }
                else
                {
                    v_x(i - D::nu) = 1;
                }
                typename P::state_t hv_x;
                typename P::control_t hv_u;
                problem.stage_inequality_constr_hess_prod(
                    k, x, u, w, v_x, v_u, hv_x, hv_u);
                mat.col(i).head(D::nu) = hv_u;
                mat.col(i).tail(D::nx) = hv_x;
            }
        }
        return mat;
    }

    /// (u;x)-layout matrix M with M v = stage-equality HVP.
    Eigen::Matrix<S, D::nu + D::nx, D::nu + D::nx>
    build_eq_hvp(int k, const P& problem, const Solution<P, NH>& sol) const
    {
        using Mat = Eigen::Matrix<S, D::nu + D::nx, D::nu + D::nx>;
        Mat mat;
        mat.setZero();
        if constexpr (D::ne > 0)
        {
            const auto& x = sol.x[k];
            const auto& u = sol.u[k];
            const auto& w = sol.lambda_eq_stage[k];
            for (int i = 0; i < D::nu + D::nx; ++i)
            {
                typename P::state_t v_x;
                typename P::control_t v_u;
                v_x.setZero();
                v_u.setZero();
                if (i < D::nu)
                {
                    v_u(i) = 1;
                }
                else
                {
                    v_x(i - D::nu) = 1;
                }
                typename P::state_t hv_x;
                typename P::control_t hv_u;
                problem.stage_equality_constr_hess_prod(
                    k, x, u, w, v_x, v_u, hv_x, hv_u);
                mat.col(i).head(D::nu) = hv_u;
                mat.col(i).tail(D::nx) = hv_x;
            }
        }
        return mat;
    }

    /// (nx)-layout matrix M with M v = terminal constraint HVP.
    template <class F>
    Eigen::Matrix<S, D::nx, D::nx> build_term_hvp(const F& call) const
    {
        using Mat = Eigen::Matrix<S, D::nx, D::nx>;
        Mat mat;
        mat.setZero();
        for (int i = 0; i < D::nx; ++i)
        {
            typename P::state_t v;
            v.setZero();
            v(i) = 1;
            typename P::state_t hv;
            call(v, hv);
            mat.col(i) = hv;
        }
        return mat;
    }

    // ---------------------------------------------------------------
    //  Per-stage assembly helpers
    // ---------------------------------------------------------------

    Status assemble_first(const P& problem, const Solution<P, NH>& sol)
    {
        const auto& x = sol.x[0];
        const auto& u = sol.u[0];
        auto& st = qp_in_.first;
        constexpr int nux = D::nu + D::nx;

        // hess: permuted cost Hessian + HVP terms + slack diagonal
        {
            const auto H = problem.stage_cost_hessian(0, x, u);
            // element-wise: block extraction from the small local H trips
            // a GCC 13 -Warray-bounds false positive on small matrices
            for (int i = 0; i < nux; ++i)
            {
                for (int j = 0; j < nux; ++j)
                {
                    const int pi = (i < D::nu) ? D::nx + i : i - D::nu;
                    const int pj = (j < D::nu) ? D::nx + j : j - D::nu;
                    st.hess(i, j) = H(pi, pj);
                }
            }

            if (opts_.compute_hess)
            {
                if constexpr (P::has_dynamics_hess_prod)
                {
                    const auto M = build_dyn_hvp(0, problem, sol);
                    for (int i = 0; i < nux; ++i)
                    {
                        for (int j = 0; j < nux; ++j)
                        {
                            st.hess(i, j) -= M(i, j);
                        }
                    }
                }
                if constexpr (P::has_constr_hess_prod)
                {
                    if constexpr (D::ng > 0)
                    {
                        const auto M = build_ineq_hvp(0, problem, sol);
                        for (int i = 0; i < nux; ++i)
                        {
                            for (int j = 0; j < nux; ++j)
                            {
                                st.hess(i, j) += M(i, j);
                            }
                        }
                    }
                    if constexpr (D::ne > 0)
                    {
                        const auto M = build_eq_hvp(0, problem, sol);
                        for (int i = 0; i < nux; ++i)
                        {
                            for (int j = 0; j < nux; ++j)
                            {
                                st.hess(i, j) += M(i, j);
                            }
                        }
                    }
                }
            }
            fill_slack_diag(st.hess, problem, /*is_first=*/true, /*k=*/0);
        }

        // grad: [g_u; g_x], slack part 0
        {
            const auto g = problem.stage_cost_gradient(0, x, u);
            for (int i = 0; i < nux; ++i)
            {
                st.grad(i) = g((i < D::nu) ? D::nx + i : i - D::nu);
            }
        }

        // BA = [B | A], b = f(x_0, u_0) - x_1
        {
            typename P::dyn_df_dx_t A;
            typename P::dyn_df_du_t B;
            problem.dynamics_jacobian(0, x, u, A, B);
            for (int i = 0; i < D::nx; ++i)
            {
                for (int j = 0; j < D::nu; ++j)
                {
                    st.BA(i, j) = B(i, j);
                }
                for (int j = 0; j < D::nx; ++j)
                {
                    st.BA(i, D::nu + j) = A(i, j);
                }
            }
            const auto f = problem.dynamics_next_state(0, x, u);
            for (int i = 0; i < D::nx; ++i)
            {
                st.b(i) = f(i) - sol.x[1](i);
            }
        }

        fill_dc_first(problem, sol, st);
        fill_d_mask_first(problem, sol, st);

        // degenerate first-stage check (plan sec. 2f.6)
        if constexpr (P::fixed_initial_state)
        {
            Status deg = check_first_degeneracy(problem, sol);
            if (deg != Status::kSolved)
            {
                return deg;
            }
        }
        return Status::kSolved;
    }

    void assemble_path(const P& problem, const Solution<P, NH>& sol, int k)
    {
        const auto& x = sol.x[k];
        const auto& u = sol.u[k];
        auto& st = qp_in_.path[k - 1];
        constexpr int nux = D::nu + D::nx;

        // hess
        {
            const auto H = problem.stage_cost_hessian(k, x, u);
            for (int i = 0; i < nux; ++i)
            {
                for (int j = 0; j < nux; ++j)
                {
                    const int pi = (i < D::nu) ? D::nx + i : i - D::nu;
                    const int pj = (j < D::nu) ? D::nx + j : j - D::nu;
                    st.hess(i, j) = H(pi, pj);
                }
            }

            if (opts_.compute_hess)
            {
                if constexpr (P::has_dynamics_hess_prod)
                {
                    const auto M = build_dyn_hvp(k, problem, sol);
                    for (int i = 0; i < nux; ++i)
                    {
                        for (int j = 0; j < nux; ++j)
                        {
                            st.hess(i, j) -= M(i, j);
                        }
                    }
                }
                if constexpr (P::has_constr_hess_prod)
                {
                    if constexpr (D::ng > 0)
                    {
                        const auto M = build_ineq_hvp(k, problem, sol);
                        for (int i = 0; i < nux; ++i)
                        {
                            for (int j = 0; j < nux; ++j)
                            {
                                st.hess(i, j) += M(i, j);
                            }
                        }
                    }
                    if constexpr (D::ne > 0)
                    {
                        const auto M = build_eq_hvp(k, problem, sol);
                        for (int i = 0; i < nux; ++i)
                        {
                            for (int j = 0; j < nux; ++j)
                            {
                                st.hess(i, j) += M(i, j);
                            }
                        }
                    }
                }
            }
            fill_slack_diag(st.hess, problem, /*is_first=*/false, k);
        }

        // grad: [g_u; g_x], slack part 0
        {
            const auto g = problem.stage_cost_gradient(k, x, u);
            for (int i = 0; i < nux; ++i)
            {
                st.grad(i) = g((i < D::nu) ? D::nx + i : i - D::nu);
            }
        }

        // BA = [B | A], b = f(x_k, u_k) - x_{k+1}
        {
            typename P::dyn_df_dx_t A;
            typename P::dyn_df_du_t B;
            problem.dynamics_jacobian(k, x, u, A, B);
            for (int i = 0; i < D::nx; ++i)
            {
                for (int j = 0; j < D::nu; ++j)
                {
                    st.BA(i, j) = B(i, j);
                }
                for (int j = 0; j < D::nx; ++j)
                {
                    st.BA(i, D::nu + j) = A(i, j);
                }
            }
            const auto f = problem.dynamics_next_state(k, x, u);
            for (int i = 0; i < D::nx; ++i)
            {
                st.b(i) = f(i) - sol.x[k + 1](i);
            }
        }

        fill_dc_path(problem, sol, k, st);
        fill_d_mask_path(problem, sol, k, st);
    }

    void assemble_term(const P& problem, const Solution<P, NH>& sol)
    {
        const int N = sol.N;
        const auto& x = sol.x[N];
        auto& st = qp_in_.term;
        constexpr int nx = D::nx;

        // hess: terminal cost Hessian + terminal constraint HVPs + slack
        // (state block only; the slack diagonal is filled separately)
        {
            const auto Ht = problem.terminal_cost_hessian(x);
            for (int i = 0; i < nx; ++i)
            {
                for (int j = 0; j < nx; ++j)
                {
                    st.hess(i, j) = Ht(i, j);
                }
            }
            if (opts_.compute_hess && P::has_constr_hess_prod)
            {
                if constexpr (D::ng_t > 0)
                {
                    const auto& w = sol.lambda_ineq_term;
                    auto mat = build_term_hvp([&](const auto& v, auto& hv) {
                        problem.terminal_inequality_constr_hess_prod(
                            x, w, v, hv);
                    });
                    for (int i = 0; i < nx; ++i)
                    {
                        for (int j = 0; j < nx; ++j)
                        {
                            st.hess(i, j) += mat(i, j);
                        }
                    }
                }
                if constexpr (D::ne_t > 0)
                {
                    const auto& w = sol.lambda_eq_term;
                    auto mat = build_term_hvp([&](const auto& v, auto& hv) {
                        problem.terminal_equality_constr_hess_prod(
                            x, w, v, hv);
                    });
                    for (int i = 0; i < nx; ++i)
                    {
                        for (int j = 0; j < nx; ++j)
                        {
                            st.hess(i, j) += mat(i, j);
                        }
                    }
                }
            }
            fill_slack_diag_term(st.hess, problem);
        }

        // grad: [g_x], slack part 0
        {
            const auto gt = problem.terminal_cost_gradient(x);
            for (int i = 0; i < nx; ++i)
            {
                st.grad(i) = gt(i);
            }
        }

        fill_dc_term(problem, sol, st);
        fill_d_mask_term(problem, sol, st);
    }

    // ---------------------------------------------------------------
    //  DC fill (constraint Jacobian over the stage variable vector)
    // ---------------------------------------------------------------

    template <class StT>
    void fill_dc_first(const P& problem, const Solution<P, NH>& sol, StT& st)
    {
        const auto& x = sol.x[0];
        const auto& u = sol.u[0];
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
                const int r = lay.row_off(detail::g_bx) + j;
                st.DC(r, D::idxb_first[j]) = 1;
                // soft box row: +1 (lo) / -1 (hi) slack columns (3a)
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
                const int r = lay.row_off(detail::g_bu) + j;
                st.DC(r, D::idxb_first[D::nbx_first + j]) = 1;
                // soft box row: +1 (lo) / -1 (hi) slack columns (3a)
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
            typename P::ineq_dg_dx_t g_dx;
            typename P::ineq_dg_du_t g_du;
            problem.stage_inequality_constr_jacobian(0, x, u, g_dx, g_du);
            for (int j = 0; j < D::ng; ++j)
            {
                const int r = lay.row_off(detail::g_ineq) + j;
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
            typename P::eq_de_dx_t e_dx;
            typename P::eq_de_du_t e_du;
            problem.stage_equality_constr_jacobian(0, x, u, e_dx, e_du);
            for (int j = 0; j < D::ne; ++j)
            {
                const int r = lay.row_off(detail::g_eq) + j;
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
            const auto spec = problem.stage_linear_constr(0);
            for (int j = 0; j < D::nl; ++j)
            {
                const int r = lay.row_off(detail::g_lin) + j;
                for (int i = 0; i < D::nu; ++i)
                {
                    st.DC(r, i) = spec.B(j, i);
                }
                for (int i = 0; i < D::nx; ++i)
                {
                    st.DC(r, D::nu + i) = spec.A(j, i);
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

    template <class StT>
    void fill_dc_path(const P& problem, const Solution<P, NH>& sol, int k,
                      StT& st)
    {
        const auto& x = sol.x[k];
        const auto& u = sol.u[k];
        const auto& lay = D::lay_path;

        if constexpr (D::nbx > 0)
        {
            for (int j = 0; j < D::nbx; ++j)
            {
                const int r = lay.row_off(detail::g_bx) + j;
                st.DC(r, D::idxb_path[j]) = 1;
                // soft box row: +1 (lo) / -1 (hi) slack columns (3a)
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
                const int r = lay.row_off(detail::g_bu) + j;
                st.DC(r, D::idxb_path[D::nbx + j]) = 1;
                // soft box row: +1 (lo) / -1 (hi) slack columns (3a)
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
            typename P::ineq_dg_dx_t g_dx;
            typename P::ineq_dg_du_t g_du;
            problem.stage_inequality_constr_jacobian(k, x, u, g_dx, g_du);
            for (int j = 0; j < D::ng; ++j)
            {
                const int r = lay.row_off(detail::g_ineq) + j;
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
            typename P::eq_de_dx_t e_dx;
            typename P::eq_de_du_t e_du;
            problem.stage_equality_constr_jacobian(k, x, u, e_dx, e_du);
            for (int j = 0; j < D::ne; ++j)
            {
                const int r = lay.row_off(detail::g_eq) + j;
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
            const auto spec = problem.stage_linear_constr(k);
            for (int j = 0; j < D::nl; ++j)
            {
                const int r = lay.row_off(detail::g_lin) + j;
                for (int i = 0; i < D::nu; ++i)
                {
                    st.DC(r, i) = spec.B(j, i);
                }
                for (int i = 0; i < D::nx; ++i)
                {
                    st.DC(r, D::nu + i) = spec.A(j, i);
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

    template <class StT>
    void fill_dc_term(const P& problem, const Solution<P, NH>& sol, StT& st)
    {
        const int N = sol.N;
        const auto& x = sol.x[N];
        const auto& lay = D::lay_term;

        if constexpr (D::nbx_t > 0)
        {
            for (int j = 0; j < D::nbx_t; ++j)
            {
                const int r = lay.row_off(detail::g_bx) + j;
                st.DC(r, D::idxb_term[j]) = 1;
                // soft box row: +1 (lo) / -1 (hi) slack columns (3a)
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
            typename P::ineq_term_dg_dx_t g_dx;
            problem.terminal_inequality_constr_jacobian(x, g_dx);
            for (int j = 0; j < D::ng_t; ++j)
            {
                const int r = lay.row_off(detail::g_ineq) + j;
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
            typename P::eq_term_de_dx_t e_dx;
            problem.terminal_equality_constr_jacobian(x, e_dx);
            for (int j = 0; j < D::ne_t; ++j)
            {
                const int r = lay.row_off(detail::g_eq) + j;
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
            const auto spec = problem.terminal_linear_constr();
            for (int j = 0; j < D::nl_t; ++j)
            {
                const int r = lay.row_off(detail::g_lin) + j;
                for (int i = 0; i < D::nx; ++i)
                {
                    st.DC(r, i) = spec.A(j, i);
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
    // ---------------------------------------------------------------

    template <class StageT>
    void fill_d_mask_first(const P& problem, const Solution<P, NH>& sol,
                           StageT& st)
    {
        const auto& x = sol.x[0];
        const auto& u = sol.u[0];
        const auto& lay = D::lay_first;
        constexpr int nrow = D::nrow_first;

        typename P::ineq_t gv{};
        typename P::eq_t ev{};
        typename P::lin_t lv{};
        typename P::stage_linear_t lin_spec{};
        if constexpr (D::ng > 0)
            gv = problem.stage_inequality_constr(0, x, u);
        if constexpr (D::ne > 0)
            ev = problem.stage_equality_constr(0, x, u);
        if constexpr (D::nl > 0)
        {
            lin_spec = problem.stage_linear_constr(0);
            for (int i = 0; i < D::nl; ++i)
            {
                double val = 0.0;
                for (int j = 0; j < D::nx; ++j)
                {
                    val += static_cast<double>(lin_spec.A(i, j))
                         * static_cast<double>(x(j));
                }
                for (int j = 0; j < D::nu; ++j)
                {
                    val += static_cast<double>(lin_spec.B(i, j))
                         * static_cast<double>(u(j));
                }
                lv(i) = static_cast<S>(val);
            }
        }

        for (int r = 0; r < nrow; ++r)
        {
            const int g = lay.group_of(r);
            const int j = r - lay.row_off(g);
            double w_cur = 0.0, lo = 0.0, hi = 0.0;

            switch (g)
            {
            case detail::g_pin:
                if constexpr (P::fixed_initial_state)
                {
                    const auto x0 = problem.initial_state();
                    w_cur = static_cast<double>(x(j));
                    lo = hi = static_cast<double>(x0(j));
                }
                break;
            case detail::g_bx:
                if constexpr (D::nbx_first > 0)
                {
                    const auto spec = problem.stage_state_box_constr(0);
                    w_cur = static_cast<double>(x(P::state_box_idx[j]));
                    lo = static_cast<double>(spec.lo(j));
                    hi = static_cast<double>(spec.hi(j));
                }
                break;
            case detail::g_bu:
                if constexpr (D::nbu > 0)
                {
                    const auto spec = problem.stage_control_box_constr(0);
                    w_cur = static_cast<double>(u(P::control_box_idx[j]));
                    lo = static_cast<double>(spec.lo(j));
                    hi = static_cast<double>(spec.hi(j));
                }
                break;
            case detail::g_ineq:
                w_cur = static_cast<double>(gv(j));
                hi = 0.0;
                break;
            case detail::g_eq:
                w_cur = static_cast<double>(ev(j));
                lo = hi = 0.0;
                break;
            default:  // g_lin
                if constexpr (D::nl > 0)
                {
                    w_cur = static_cast<double>(lv(j));
                    lo = static_cast<double>(lin_spec.bounds.lo(j));
                    hi = static_cast<double>(lin_spec.bounds.hi(j));
                }
                break;
            }

            const int sl = lay.side_lo(r);
            if (sl >= 0)
            {
                st.d(sl) = static_cast<S>(lo - w_cur);
                st.d_mask(sl) =
                    std::isfinite(lo) ? static_cast<S>(1) : static_cast<S>(0);
            }
            const int sh = lay.side_hi(r);
            st.d(sh) = static_cast<S>(w_cur - hi);
            st.d_mask(sh) =
                std::isfinite(hi) ? static_cast<S>(1) : static_cast<S>(0);
        }

        // slack sides: d = 0, d_mask = 1
        const int lo_sz = lay.lo_size();
        for (int j = 0; j < D::nslack_first; ++j)
        {
            const int si = lo_sz + nrow + j;
            st.d(si) = 0;
            st.d_mask(si) = 1;
        }
    }

    template <class StageT>
    void fill_d_mask_path(const P& problem, const Solution<P, NH>& sol,
                          int k, StageT& st)
    {
        const auto& x = sol.x[k];
        const auto& u = sol.u[k];
        const auto& lay = D::lay_path;
        constexpr int nrow = D::nrow_path;

        typename P::ineq_t gv{};
        typename P::eq_t ev{};
        typename P::lin_t lv{};
        typename P::stage_linear_t lin_spec{};
        if constexpr (D::ng > 0)
            gv = problem.stage_inequality_constr(k, x, u);
        if constexpr (D::ne > 0)
            ev = problem.stage_equality_constr(k, x, u);
        if constexpr (D::nl > 0)
        {
            lin_spec = problem.stage_linear_constr(k);
            for (int i = 0; i < D::nl; ++i)
            {
                double val = 0.0;
                for (int j = 0; j < D::nx; ++j)
                {
                    val += static_cast<double>(lin_spec.A(i, j))
                         * static_cast<double>(x(j));
                }
                for (int j = 0; j < D::nu; ++j)
                {
                    val += static_cast<double>(lin_spec.B(i, j))
                         * static_cast<double>(u(j));
                }
                lv(i) = static_cast<S>(val);
            }
        }

        for (int r = 0; r < nrow; ++r)
        {
            const int g = lay.group_of(r);
            const int j = r - lay.row_off(g);
            double w_cur = 0.0, lo = 0.0, hi = 0.0;

            switch (g)
            {
            case detail::g_bx:
                if constexpr (D::nbx > 0)
                {
                    const auto spec = problem.stage_state_box_constr(k);
                    w_cur = static_cast<double>(x(P::state_box_idx[j]));
                    lo = static_cast<double>(spec.lo(j));
                    hi = static_cast<double>(spec.hi(j));
                }
                break;
            case detail::g_bu:
                if constexpr (D::nbu > 0)
                {
                    const auto spec = problem.stage_control_box_constr(k);
                    w_cur = static_cast<double>(u(P::control_box_idx[j]));
                    lo = static_cast<double>(spec.lo(j));
                    hi = static_cast<double>(spec.hi(j));
                }
                break;
            case detail::g_ineq:
                w_cur = static_cast<double>(gv(j));
                hi = 0.0;
                break;
            case detail::g_eq:
                w_cur = static_cast<double>(ev(j));
                lo = hi = 0.0;
                break;
            default:  // g_lin
                if constexpr (D::nl > 0)
                {
                    w_cur = static_cast<double>(lv(j));
                    lo = static_cast<double>(lin_spec.bounds.lo(j));
                    hi = static_cast<double>(lin_spec.bounds.hi(j));
                }
                break;
            }

            const int sl = lay.side_lo(r);
            if (sl >= 0)
            {
                st.d(sl) = static_cast<S>(lo - w_cur);
                st.d_mask(sl) =
                    std::isfinite(lo) ? static_cast<S>(1) : static_cast<S>(0);
            }
            const int sh = lay.side_hi(r);
            st.d(sh) = static_cast<S>(w_cur - hi);
            st.d_mask(sh) =
                std::isfinite(hi) ? static_cast<S>(1) : static_cast<S>(0);
        }

        const int lo_sz = lay.lo_size();
        for (int j = 0; j < D::nslack_path; ++j)
        {
            const int si = lo_sz + nrow + j;
            st.d(si) = 0;
            st.d_mask(si) = 1;
        }
    }

    template <class StageT>
    void fill_d_mask_term(const P& problem, const Solution<P, NH>& sol,
                          StageT& st)
    {
        const int N = sol.N;
        const auto& x = sol.x[N];
        const auto& lay = D::lay_term;
        constexpr int nrow = D::nrow_term;

        typename P::ineq_term_t gv{};
        typename P::eq_term_t ev{};
        typename P::lin_term_t lv{};
        typename P::term_linear_t lin_spec{};
        if constexpr (D::ng_t > 0)
            gv = problem.terminal_inequality_constr(x);
        if constexpr (D::ne_t > 0)
            ev = problem.terminal_equality_constr(x);
        if constexpr (D::nl_t > 0)
        {
            lin_spec = problem.terminal_linear_constr();
            for (int i = 0; i < D::nl_t; ++i)
            {
                double val = 0.0;
                for (int j = 0; j < D::nx; ++j)
                {
                    val += static_cast<double>(lin_spec.A(i, j))
                         * static_cast<double>(x(j));
                }
                lv(i) = static_cast<S>(val);
            }
        }

        for (int r = 0; r < nrow; ++r)
        {
            const int g = lay.group_of(r);
            const int j = r - lay.row_off(g);
            double w_cur = 0.0, lo = 0.0, hi = 0.0;

            switch (g)
            {
            case detail::g_bx:
                if constexpr (D::nbx_t > 0)
                {
                    const auto spec = problem.terminal_state_box_constr();
                    w_cur = static_cast<double>(
                        x(P::terminal_state_box_idx[j]));
                    lo = static_cast<double>(spec.lo(j));
                    hi = static_cast<double>(spec.hi(j));
                }
                break;
            case detail::g_ineq:
                w_cur = static_cast<double>(gv(j));
                hi = 0.0;
                break;
            case detail::g_eq:
                w_cur = static_cast<double>(ev(j));
                lo = hi = 0.0;
                break;
            default:  // g_lin
                if constexpr (D::nl_t > 0)
                {
                    w_cur = static_cast<double>(lv(j));
                    lo = static_cast<double>(lin_spec.bounds.lo(j));
                    hi = static_cast<double>(lin_spec.bounds.hi(j));
                }
                break;
            }

            const int sl = lay.side_lo(r);
            if (sl >= 0)
            {
                st.d(sl) = static_cast<S>(lo - w_cur);
                st.d_mask(sl) =
                    std::isfinite(lo) ? static_cast<S>(1) : static_cast<S>(0);
            }
            const int sh = lay.side_hi(r);
            st.d(sh) = static_cast<S>(w_cur - hi);
            st.d_mask(sh) =
                std::isfinite(hi) ? static_cast<S>(1) : static_cast<S>(0);
        }

        const int lo_sz = lay.lo_size();
        for (int j = 0; j < D::nslack_term; ++j)
        {
            const int si = lo_sz + nrow + j;
            st.d(si) = 0;
            st.d_mask(si) = 1;
        }
    }

    // ---------------------------------------------------------------
    //  Slack Hessian diagonal (soft rows: weight w, unused: 1.0)
    // ---------------------------------------------------------------

    template <class HessT>
    void fill_slack_diag(HessT& hess, const P& problem, bool is_first,
                         int k) const
    {
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
                hess(nux + i, nux + i) = static_cast<S>(
                    static_cast<double>(
                        spec.soft_penalty(P::state_box_soft_idx[i])));
                const int col = nux + nbx_s + i;
                hess(col, col) = static_cast<S>(
                    static_cast<double>(
                        spec.soft_penalty(P::state_box_soft_idx[i])));
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
                const double w = static_cast<double>(
                    spec.soft_penalty(P::control_box_soft_idx[i]));
                hess(nux + 2 * off_bx + i,
                     nux + 2 * off_bx + i) = static_cast<S>(w);
                hess(nux + 2 * off_bx + P::nbu_soft + i,
                     nux + 2 * off_bx + P::nbu_soft + i) =
                    static_cast<S>(w);
            }
        }
        // ineq hi
        if constexpr (P::ng_soft > 0)
        {
            const int off = 2 * ((is_first && P::fixed_initial_state)
                                     ? 0
                                     : P::nbx_soft)
                           + 2 * P::nbu_soft;
            const auto pen =
                problem.stage_inequality_constr_soft_penalty(k);
            for (int i = 0; i < P::ng_soft; ++i)
            {
                const int col = nux + off + i;
                hess(col, col) =
                    static_cast<S>(static_cast<double>(pen(i)));
            }
        }
        // eq lo / eq hi
        if constexpr (P::ne_soft > 0)
        {
            const int off = 2 * ((is_first && P::fixed_initial_state)
                                     ? 0
                                     : P::nbx_soft)
                           + 2 * P::nbu_soft + P::ng_soft;
            const auto pen =
                problem.stage_equality_constr_soft_penalty(k);
            for (int i = 0; i < P::ne_soft; ++i)
            {
                const double w = static_cast<double>(pen(i));
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
                const double w = static_cast<double>(
                    spec.bounds.soft_penalty(P::lin_soft_idx[i]));
                hess(nux + off + i, nux + off + i) = static_cast<S>(w);
                hess(nux + off + P::nl_soft + i,
                     nux + off + P::nl_soft + i) = static_cast<S>(w);
            }
        }
    }

    template <class HessT>
    void fill_slack_diag_term(HessT& hess, const P& problem) const
    {
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
                const double w = static_cast<double>(
                    spec.soft_penalty(P::terminal_state_box_soft_idx[i]));
                hess(nx + i, nx + i) = static_cast<S>(w);
                hess(nx + P::nbx_t_soft + i,
                     nx + P::nbx_t_soft + i) = static_cast<S>(w);
            }
        }
        if constexpr (P::ng_t_soft > 0)
        {
            const auto pen =
                problem.terminal_inequality_constr_soft_penalty();
            const int off = 2 * P::nbx_t_soft;
            for (int i = 0; i < P::ng_t_soft; ++i)
            {
                hess(nx + off + i, nx + off + i) =
                    static_cast<S>(static_cast<double>(pen(i)));
            }
        }
        if constexpr (P::ne_t_soft > 0)
        {
            const auto pen =
                problem.terminal_equality_constr_soft_penalty();
            const int off = 2 * P::nbx_t_soft + P::ng_t_soft;
            for (int i = 0; i < P::ne_t_soft; ++i)
            {
                const double w = static_cast<double>(pen(i));
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
                const double w = static_cast<double>(
                    spec.bounds.soft_penalty(P::terminal_lin_soft_idx[i]));
                hess(nx + off + i, nx + off + i) = static_cast<S>(w);
                hess(nx + off + P::nl_t_soft + i,
                     nx + off + P::nl_t_soft + i) = static_cast<S>(w);
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
    Status check_first_degeneracy(const P& problem,
                                  const Solution<P, NH>& sol) const
    {
        const auto& x = sol.x[0];
        const auto& u = sol.u[0];
        const auto& lay = D::lay_first;

        if constexpr (D::ng > 0)
        {
            const auto g = problem.stage_inequality_constr(0, x, u);
            typename P::ineq_dg_dx_t g_dx;
            typename P::ineq_dg_du_t g_du;
            problem.stage_inequality_constr_jacobian(0, x, u, g_dx, g_du);
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
                    const int r = lay.row_off(detail::g_ineq) + j;
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
                const double lo =
                    static_cast<double>(spec.bounds.lo(j));
                const double hi =
                    static_cast<double>(spec.bounds.hi(j));
                const int r = lay.row_off(detail::g_lin) + j;
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
    //  Members
    // ---------------------------------------------------------------

    SqpOptions opts_;
    QpSolver qp_;
    Regularizer reg_;
    Globalizer glob_;
    SqpStatistics stat_;
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
};

}  // namespace ocp

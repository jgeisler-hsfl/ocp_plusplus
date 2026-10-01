// ocp/solvers/acados/globalize.hpp
//
// Globalization primitives for the SQP driver (phase 2, sub-step 2b).
// Port of acados globalization options and the SQP variable-update
// routine (ocp_nlp_common.c: ocp_nlp_update_variables_sqp).
//
// Contents:
//   - GlobOptions                          (ocp_nlp_globalization_opts)
//   - SqpSlacks<P, NH>                    solver-internal slack storage
//   - detail::map_qp_duals_to_solution    (NLP<->QP dual mapping, plan sec. 1.3)
//   - apply_sqp_step                      (ocp_nlp_update_variables_sqp)
//
// The line search (MeritBacktracking::find_acceptable_iterate) is in 2d.

#pragma once

#include <cassert>

#include <Eigen/Dense>

#include "../hpipm/qp.hpp"

namespace ocp
{

// =========================================================================
//  Options
// =========================================================================

/// Globalization options (ocp_nlp_globalization_common.h:82-90; defaults
/// verified against ocp_nlp_globalization_common.c:83-86). SOC and
/// sufficient-descent are phase 3 (SQP_PLAN sec. 8).
struct GlobOptions
{
    double alpha_min = 0.05;             // minimum accepted step size
    double alpha_reduction = 0.7;        // backtracking shrink factor
    double eps_sufficient_descent = 1e-4;  // (phase 3, Leineweber 1999)
    bool full_step_dual = false;        // full-step dual update vs. relaxed
};

// =========================================================================
//  Slack storage
// =========================================================================

/// Solver-internal slack storage (not part of ocp::Solution per AGENTS.md).
/// One vector per stage type; the SQP driver owns this and warm-starts it
/// across repeated solves on the same solver object (acados nlp_out slack
/// analogue).
template <class P, int NH = Eigen::Dynamic>
struct SqpSlacks
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    using first_t = Eigen::Matrix<S, D::nslack_first, 1>;
    using path_t  = Eigen::Matrix<S, D::nslack_path,  1>;
    using term_t  = Eigen::Matrix<S, D::nslack_term,  1>;

    int N = (NH == Eigen::Dynamic) ? 0 : NH;

    first_t first{};
    Trajectory<path_t, detail::traj_extent<NH, -1>()> path;  // k = 1..N-1
    term_t  term{};

    SqpSlacks() = default;
    explicit SqpSlacks(int n_stages) { resize(n_stages); }

    /// Dynamic horizon: allocate path storage; fixed horizon: consistency
    /// check only (no allocation).
    void resize(int n_stages)
    {
        if constexpr (NH == Eigen::Dynamic)
        {
            assert(n_stages >= 1);
            N = n_stages;
            path.resize(N - 1);
        }
        else
        {
            assert(n_stages == NH);
            N = NH;
        }
    }

    void setZero()
    {
        first.setZero();
        for (int k = 1; k < N; ++k)
        {
            path[k - 1].setZero();
        }
        term.setZero();
    }
};

// =========================================================================
//  QP -> NLP dual mapping (plan sec. 1.3)
// =========================================================================

namespace detail
{

/// Map QP per-side multipliers onto the NLP Solution multipliers.
///
///   lambda_dyn[k]        = - pi_qp[k]          (QP pi is the negative of
///                                          the NLP dynamics multiplier;
///                                          res_g carries +BA^T pi_qp)
///   lambda_ineq_stage[k] = lam[side_hi(r)]      (g <= 0, lambda >= 0)
///   lambda_eq_stage[k]   = lam[side_hi(r)] - lam[side_lo(r)]  (signed)
///   lambda_lin_stage[k]  = lam[side_hi(r)] - lam[side_lo(r)]  (signed)
///   lambda_box_state[k]  = [lower; upper], lower <-> side_lo,
///                          upper <-> side_hi
///   lambda_box_control[k] = same rule for control box rows
///   terminal variants    = same rules on terminal stage rows
///
/// All target fields are zeroed first, so rows absent from the QP (e.g.
/// the stage-0 state box when x_0 is fixed) map to zero.
template <class P, int NH = Eigen::Dynamic>
void map_qp_duals_to_solution(const QpSol<P, NH>& sol, Solution<P, NH>& dest)
{
    using D = QpDim<P>;
    const int N = sol.N;

    // zero all NLP multiplier fields
    for (int k = 0; k < N; ++k)
    {
        dest.lambda_dyn[k].setZero();
        dest.lambda_ineq_stage[k].setZero();
        dest.lambda_eq_stage[k].setZero();
        dest.lambda_lin_stage[k].setZero();
        dest.lambda_box_state[k].setZero();
        dest.lambda_box_control[k].setZero();
    }
    dest.lambda_box_state[N].setZero();
    dest.lambda_ineq_term.setZero();
    dest.lambda_eq_term.setZero();
    dest.lambda_lin_term.setZero();

    // dynamics multipliers (sign flip, sec. 1.3)
    for (int k = 0; k < N; ++k)
    {
        dest.lambda_dyn[k] = -sol.pi[k];
    }

    // ineq / eq / lin for a stage, parameterized by the group row counts
    // (the terminal stage uses the *_t counts, the first / path stages the
    // stage counts). Generic over the stage's lam vector and the dest vectors.
    auto map_group = [&](const auto& lam, const detail::QpLayout& lay,
                         auto& ineq, int n_ineq, auto& eq, int n_eq,
                         auto& lin, int n_lin)
    {
        for (int j = 0; j < n_ineq; ++j)
        {
            const int r = lay.row_off(detail::g_ineq) + j;
            ineq(j) = lam(lay.side_hi(r));
        }
        for (int j = 0; j < n_eq; ++j)
        {
            const int r = lay.row_off(detail::g_eq) + j;
            eq(j) = lam(lay.side_hi(r)) - lam(lay.side_lo(r));
        }
        for (int j = 0; j < n_lin; ++j)
        {
            const int r = lay.row_off(detail::g_lin) + j;
            lin(j) = lam(lay.side_hi(r)) - lam(lay.side_lo(r));
        }
    };

    // first stage (k = 0); state box rows absent when x_0 is fixed
    {
        const auto& lam = sol.lam_first;
        const auto& lay = D::lay_first;
        for (int j = 0; j < D::nbx_first; ++j)
        {
            const int r = lay.row_off(detail::g_bx) + j;
            dest.lambda_box_state[0](j) = lam(lay.side_lo(r));
            dest.lambda_box_state[0](D::nbx + j) = lam(lay.side_hi(r));
        }
        for (int j = 0; j < D::nbu; ++j)
        {
            const int r = lay.row_off(detail::g_bu) + j;
            dest.lambda_box_control[0](j) = lam(lay.side_lo(r));
            dest.lambda_box_control[0](D::nbu + j) = lam(lay.side_hi(r));
        }
        map_group(lam, lay, dest.lambda_ineq_stage[0], D::ng,
                  dest.lambda_eq_stage[0], D::ne, dest.lambda_lin_stage[0],
                  D::nl);
    }

    // path stages (k = 1..N-1)
    for (int k = 1; k < N; ++k)
    {
        const auto& lam = sol.lam_path[k - 1];
        const auto& lay = D::lay_path;
        for (int j = 0; j < D::nbx; ++j)
        {
            const int r = lay.row_off(detail::g_bx) + j;
            dest.lambda_box_state[k](j) = lam(lay.side_lo(r));
            dest.lambda_box_state[k](D::nbx + j) = lam(lay.side_hi(r));
        }
        for (int j = 0; j < D::nbu; ++j)
        {
            const int r = lay.row_off(detail::g_bu) + j;
            dest.lambda_box_control[k](j) = lam(lay.side_lo(r));
            dest.lambda_box_control[k](D::nbu + j) = lam(lay.side_hi(r));
        }
        map_group(lam, lay, dest.lambda_ineq_stage[k], D::ng,
                  dest.lambda_eq_stage[k], D::ne, dest.lambda_lin_stage[k],
                  D::nl);
    }

    // terminal stage (k = N): state box + terminal ineq / eq / lin. The NLP
    // box_state[N] vector is [lower; upper] of size 2*nbx (upper half at
    // index nbx), so the nbx_t terminal rows map into the first nbx_t of
    // each half (nbx_t == nbx in the current examples).
    {
        const auto& lam = sol.lam_term;
        const auto& lay = D::lay_term;
        for (int j = 0; j < D::nbx_t; ++j)
        {
            const int r = lay.row_off(detail::g_bx) + j;
            dest.lambda_box_state[N](j) = lam(lay.side_lo(r));
            dest.lambda_box_state[N](D::nbx + j) = lam(lay.side_hi(r));
        }
        map_group(lam, lay, dest.lambda_ineq_term, D::ng_t,
                  dest.lambda_eq_term, D::ne_t, dest.lambda_lin_term, D::nl_t);
    }
}

}  // namespace detail

// =========================================================================
//  SQP step application (ocp_nlp_update_variables_sqp)
// =========================================================================

/// Update the NLP iterate with a QP step
/// (ocp_nlp_common.c:3355-3414, ocp_nlp_update_variables_sqp).
///
/// Primal (acados daxpy over the whole per-stage (u;x;s) block; slacks are
/// kept out of ocp::Solution, so only the (u;x) part is applied here):
///   dest.u[k]   = start.u[k]   + alpha * (u part of step.ux_k)   k = 0..N-1
///   dest.x[k]   = start.x[k]   + alpha * (x part of step.ux_k)   k = 0..N
///   x_0 is untouched when x_0 is fixed: the pin rows force the x_0 step to
///   zero, so the formula leaves it exactly at start.x[0].
///
/// Duals: full_step_dual -> dest.lambda_* = mapped(step) (absolute,
/// acados dveccp); otherwise the relaxed update
///   dest.lambda_* = (1 - alpha) * start.lambda_* + alpha * mapped(step)
/// (acados daxpby 1.0-alpha / alpha).
///
/// Slacks are NOT updated here; the driver does s <- s + alpha * ds
/// (analogous to acados's algebraic-variable update, kept out of Solution
/// per AGENTS.md).
template <class P, int NH = Eigen::Dynamic>
void apply_sqp_step(const Solution<P, NH>& start,
                    const QpSol<P, NH>& step,
                    double alpha,
                    bool full_step_dual,
                    Solution<P, NH>& dest)
{
    using D = QpDim<P>;
    const int N = start.N;
    assert(step.N == N);
    assert(dest.N == N);

    // ---- primal: controls (k = 0..N-1) ----
    for (int k = 0; k < N; ++k)
    {
        if (k == 0)
        {
            dest.u[k] = start.u[k] + alpha * step.ux_first.head(D::nu);
        }
        else
        {
            dest.u[k] = start.u[k] + alpha * step.ux_path[k - 1].head(D::nu);
        }
    }

    // ---- primal: states (k = 0..N) ----
    for (int k = 0; k <= N; ++k)
    {
        if (k == 0)
        {
            dest.x[k] = start.x[k] + alpha * step.ux_first.segment(D::nu, D::nx);
        }
        else if (k == N)
        {
            dest.x[k] = start.x[k] + alpha * step.ux_term.segment(0, D::nx);
        }
        else
        {
            dest.x[k] = start.x[k] + alpha *
                step.ux_path[k - 1].segment(D::nu, D::nx);
        }
    }

    // ---- duals ----
    Solution<P, NH> mapped(N);
    detail::map_qp_duals_to_solution<P, NH>(step, mapped);

    if (full_step_dual)
    {
        for (int k = 0; k < N; ++k)
        {
            dest.lambda_dyn[k] = mapped.lambda_dyn[k];
            dest.lambda_ineq_stage[k] = mapped.lambda_ineq_stage[k];
            dest.lambda_eq_stage[k] = mapped.lambda_eq_stage[k];
            dest.lambda_lin_stage[k] = mapped.lambda_lin_stage[k];
            dest.lambda_box_state[k] = mapped.lambda_box_state[k];
            dest.lambda_box_control[k] = mapped.lambda_box_control[k];
        }
        dest.lambda_box_state[N] = mapped.lambda_box_state[N];
        dest.lambda_ineq_term = mapped.lambda_ineq_term;
        dest.lambda_eq_term = mapped.lambda_eq_term;
        dest.lambda_lin_term = mapped.lambda_lin_term;
    }
    else
    {
        const double w = 1.0 - alpha;
        for (int k = 0; k < N; ++k)
        {
            dest.lambda_dyn[k] =
                w * start.lambda_dyn[k] + alpha * mapped.lambda_dyn[k];
            dest.lambda_ineq_stage[k] =
                w * start.lambda_ineq_stage[k]
                + alpha * mapped.lambda_ineq_stage[k];
            dest.lambda_eq_stage[k] =
                w * start.lambda_eq_stage[k]
                + alpha * mapped.lambda_eq_stage[k];
            dest.lambda_lin_stage[k] =
                w * start.lambda_lin_stage[k]
                + alpha * mapped.lambda_lin_stage[k];
            dest.lambda_box_state[k] =
                w * start.lambda_box_state[k]
                + alpha * mapped.lambda_box_state[k];
            dest.lambda_box_control[k] =
                w * start.lambda_box_control[k]
                + alpha * mapped.lambda_box_control[k];
        }
        dest.lambda_box_state[N] =
            w * start.lambda_box_state[N] + alpha * mapped.lambda_box_state[N];
        dest.lambda_ineq_term =
            w * start.lambda_ineq_term + alpha * mapped.lambda_ineq_term;
        dest.lambda_eq_term =
            w * start.lambda_eq_term + alpha * mapped.lambda_eq_term;
        dest.lambda_lin_term =
            w * start.lambda_lin_term + alpha * mapped.lambda_lin_term;
    }
}

}  // namespace ocp

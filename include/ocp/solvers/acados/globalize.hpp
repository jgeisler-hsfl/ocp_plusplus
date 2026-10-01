// ocp/solvers/acados/globalize.hpp
//
// Globalization primitives for the SQP driver (phase 2, sub-steps 2b + 2d).
// Ports of the acados globalization options, the SQP variable-update
// routine (ocp_nlp_common.c: ocp_nlp_update_variables_sqp) and the merit
// backtracking line search (ocp_nlp_globalization_merit_backtracking.c).
//
// Contents:
//   - GlobOptions                          (ocp_nlp_globalization_opts)
//   - SqpSlacks<P, NH>                    solver-internal slack storage
//   - detail::map_qp_duals_to_solution    (NLP<->QP dual mapping, plan sec. 1.3)
//   - detail::apply_sqp_step_primal       (primal part of the step update)
//   - detail::shift_slacks                (trial slacks for the line search)
//   - apply_sqp_step                      (ocp_nlp_update_variables_sqp)
//   - MeritBacktracking<P, NH>            (merit function + line search)

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>

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
                  dest.lambda_eq_term, D::ne_t, dest.lambda_lin_term,
                  D::nl_t);
    }
}

/// Primal part only of apply_sqp_step (no dual mapping / update). The
/// merit line search uses it for trial iterates, which need no duals.
template <class P, int NH = Eigen::Dynamic>
void apply_sqp_step_primal(const Solution<P, NH>& start,
                           const QpSol<P, NH>& step,
                           double alpha,
                           Solution<P, NH>& dest)
{
    using D = QpDim<P>;
    const int N = start.N;
    assert(step.N == N);
    assert(dest.N == N);

    // controls (k = 0..N-1)
    for (int k = 0; k < N; ++k)
    {
        if (k == 0)
        {
            dest.u[k] = start.u[k] + alpha * step.ux_first.head(D::nu);
        }
        else
        {
            dest.u[k] =
                start.u[k] + alpha * step.ux_path[k - 1].head(D::nu);
        }
    }

    // states (k = 0..N)
    for (int k = 0; k <= N; ++k)
    {
        if (k == 0)
        {
            dest.x[k] =
                start.x[k] + alpha * step.ux_first.segment(D::nu, D::nx);
        }
        else if (k == N)
        {
            dest.x[k] =
                start.x[k] + alpha * step.ux_term.segment(0, D::nx);
        }
        else
        {
            dest.x[k] = start.x[k] +
                alpha * step.ux_path[k - 1].segment(D::nu, D::nx);
        }
    }
}

/// Trial slacks for the line search: s + alpha * (slack part of the QP
/// step). The slack part is the tail of each stage-type step vector (the
/// last nslack_<type> entries of (u;x;s) / (x;s), qp.hpp layout). `dest`
/// must already be sized to the step's horizon.
template <class P, int NH = Eigen::Dynamic>
void shift_slacks(const SqpSlacks<P, NH>& s, const QpSol<P, NH>& step,
                  double alpha, SqpSlacks<P, NH>& dest)
{
    using D = QpDim<P>;
    dest.first =
        s.first + alpha * step.ux_first.segment(D::nu + D::nx,
                                                D::nslack_first);
    for (int k = 1; k < s.N; ++k)
    {
        dest.path[k - 1] = s.path[k - 1] +
            alpha * step.ux_path[k - 1].segment(D::nu + D::nx,
                                                D::nslack_path);
    }
    dest.term =
        s.term + alpha * step.ux_term.segment(D::nx, D::nslack_term);
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
    const int N = start.N;

    // ---- primal ----
    detail::apply_sqp_step_primal<P, NH>(start, step, alpha, dest);

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

// =========================================================================
//  Merit backtracking line search (sub-step 2d)
// =========================================================================

/// Backtracking line search on a Leineweber merit function.
///
/// merit(x, u, s) = sum_k L_k(x_k, u_k) + L_N(x_N)
///               + 0.5 * sum w_soft * s^2          (slack penalty, 1.6)
///               + sum_k |pi_w| * |f_k - x_{k+1}|  (weighted L1 gap)
///               + sum_rows |lam_w| * viol^+       (weighted violations)
///
/// Soft rows use slack-relaxed violations (e.g. ineq: max(0, g - s));
/// hard equality rows enter as |e| (acados counts every two-sided row).
/// Pin rows (fixed x_0) are a QP-internal device and stay satisfied by
/// construction, so they are excluded.
///
/// Weights (Leineweber M5.1, acados :705-755), updated from the QP duals
/// at the start of every line search:
///   first call  :  w <- |dual|
///   later calls :  w <- max(|dual|, 0.5 (|dual| + w_old))
///
/// Line search (ocp_nlp_line_search :290-408):
///   alpha = 1; while alpha * alpha_reduction > alpha_min:
///     trial = cur + alpha * step;  if merit1 < merit0 and finite:
///     accept and return kSolved;  else alpha *= alpha_reduction.
///   Exhausted: apply the post-shrink step to cur (acados :894) and
///   return kMinStep, or kNanDetected if the last merit was non-finite
///   (cur left untouched, acados :886-890).
template <class P, int NH = Eigen::Dynamic>
struct MeritBacktracking
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    GlobOptions opts;

    // mutable per-solve merit weights (contract is const-callable)
    mutable int N_ = 0;
    mutable bool weights_set_ = false;
    mutable Trajectory<typename P::state_t,
                       detail::traj_extent<NH, 0>()>
        w_pi;
    mutable Trajectory<typename P::ineq_t,
                       detail::traj_extent<NH, 0>()>
        w_ineq_stage;
    mutable Trajectory<typename P::eq_t,
                       detail::traj_extent<NH, 0>()>
        w_eq_stage;
    mutable Trajectory<typename P::lin_t,
                       detail::traj_extent<NH, 0>()>
        w_lin_stage;
    mutable Trajectory<typename Solution<P, NH>::box_state_lam_t,
                       detail::traj_extent<NH, 0>()>
        w_box_state;
    mutable Trajectory<typename Solution<P, NH>::box_control_lam_t,
                       detail::traj_extent<NH, 0>()>
        w_box_control;
    mutable Eigen::Matrix<S, 2 * D::nbx_t, 1> w_box_term{};
    mutable typename P::ineq_term_t w_ineq_term{};
    mutable typename P::eq_term_t w_eq_term{};
    mutable typename P::lin_term_t w_lin_term{};

    /// Call once per solve() at iteration 0. Resets weight state.
    void initialize(const P& /*problem*/, const Solution<P, NH>& sol)
    {
        N_ = sol.N;
        weights_set_ = false;
        w_pi.resize(N_);
        w_ineq_stage.resize(N_);
        w_eq_stage.resize(N_);
        w_lin_stage.resize(N_);
        w_box_state.resize(N_);
        w_box_control.resize(N_);
        for (int k = 0; k < N_; ++k)
        {
            w_pi[k].setZero();
            w_ineq_stage[k].setZero();
            w_eq_stage[k].setZero();
            w_lin_stage[k].setZero();
            w_box_state[k].setZero();
            w_box_control[k].setZero();
        }
        w_box_term.setZero();
        w_ineq_term.setZero();
        w_eq_term.setZero();
        w_lin_term.setZero();
    }

    /// Update weights from the latest QP dual solution.
    /// First call after initialize(): w <- |dual|.
    /// Later calls: w <- max(|dual|, 0.5 (|dual| + w_old)).
    void update_weights(const QpSol<P, NH>& qp) const
    {
        Solution<P, NH> mapped(N_);
        detail::map_qp_duals_to_solution<P, NH>(qp, mapped);

        if (!weights_set_)
        {
            for (int k = 0; k < N_; ++k)
            {
                w_pi[k] = mapped.lambda_dyn[k].cwiseAbs();
                w_ineq_stage[k] =
                    mapped.lambda_ineq_stage[k].cwiseAbs();
                w_eq_stage[k] = mapped.lambda_eq_stage[k].cwiseAbs();
                w_lin_stage[k] =
                    mapped.lambda_lin_stage[k].cwiseAbs();
                w_box_state[k] =
                    mapped.lambda_box_state[k].cwiseAbs();
                w_box_control[k] =
                    mapped.lambda_box_control[k].cwiseAbs();
            }
            for (int j = 0; j < D::nbx_t; ++j)
            {
                w_box_term(j) =
                    std::fabs(static_cast<double>(
                        mapped.lambda_box_state[N_](j)));
                w_box_term(D::nbx_t + j) =
                    std::fabs(static_cast<double>(
                        mapped.lambda_box_state[N_](D::nbx + j)));
            }
            w_ineq_term = mapped.lambda_ineq_term.cwiseAbs();
            w_eq_term = mapped.lambda_eq_term.cwiseAbs();
            w_lin_term = mapped.lambda_lin_term.cwiseAbs();
            weights_set_ = true;
        }
        else
        {
            for (int k = 0; k < N_; ++k)
            {
                const auto pi = mapped.lambda_dyn[k].cwiseAbs();
                w_pi[k] = pi.cwiseMax(0.5 * (pi + w_pi[k]));
                const auto wi =
                    mapped.lambda_ineq_stage[k].cwiseAbs();
                w_ineq_stage[k] =
                    wi.cwiseMax(0.5 * (wi + w_ineq_stage[k]));
                const auto we =
                    mapped.lambda_eq_stage[k].cwiseAbs();
                w_eq_stage[k] =
                    we.cwiseMax(0.5 * (we + w_eq_stage[k]));
                const auto wl =
                    mapped.lambda_lin_stage[k].cwiseAbs();
                w_lin_stage[k] =
                    wl.cwiseMax(0.5 * (wl + w_lin_stage[k]));
                const auto wb =
                    mapped.lambda_box_state[k].cwiseAbs();
                w_box_state[k] =
                    wb.cwiseMax(0.5 * (wb + w_box_state[k]));
                const auto wc =
                    mapped.lambda_box_control[k].cwiseAbs();
                w_box_control[k] =
                    wc.cwiseMax(0.5 * (wc + w_box_control[k]));
            }
            for (int j = 0; j < D::nbx_t; ++j)
            {
                const double lo = std::fabs(static_cast<double>(
                    mapped.lambda_box_state[N_](j)));
                const double hi = std::fabs(static_cast<double>(
                    mapped.lambda_box_state[N_](D::nbx + j)));
                w_box_term(j) =
                    std::max(lo, 0.5 * (lo + w_box_term(j)));
                w_box_term(D::nbx_t + j) = std::max(
                    hi, 0.5 * (hi + w_box_term(D::nbx_t + j)));
            }
            const auto wti = mapped.lambda_ineq_term.cwiseAbs();
            w_ineq_term = wti.cwiseMax(0.5 * (wti + w_ineq_term));
            const auto wte = mapped.lambda_eq_term.cwiseAbs();
            w_eq_term = wte.cwiseMax(0.5 * (wte + w_eq_term));
            const auto wtl = mapped.lambda_lin_term.cwiseAbs();
            w_lin_term = wtl.cwiseMax(0.5 * (wtl + w_lin_term));
        }
    }

    /// Merit value at the iterate (sol, sl) using current weights.
    double merit(const P& problem, const Solution<P, NH>& sol,
                 const SqpSlacks<P, NH>& sl) const
    {
        const int N = sol.N;
        double m = 0.0;

        for (int k = 0; k < N; ++k)
            m += static_cast<double>(
                problem.stage_cost_value(k, sol.x[k], sol.u[k]));
        m += static_cast<double>(
            problem.terminal_cost_value(sol.x[N]));

        m += slack_penalty(problem, sl);

        for (int k = 0; k < N; ++k)
        {
            // .eval(): the LHS of the difference is a temporary; without
            // it the lazy expression would outlive it
            const auto gap = (problem.dynamics_next_state(
                                  k, sol.x[k], sol.u[k])
                              - sol.x[k + 1])
                                 .eval();
            for (int j = 0; j < D::nx; ++j)
                m += static_cast<double>(w_pi[k](j))
                   * std::fabs(static_cast<double>(gap(j)));
        }

        for (int k = 0; k < N; ++k)
            m += stage_violation(problem, sol, sl, k);
        m += term_violation(problem, sol, sl);

        return m;
    }

    /// Backtracking line search. Advances `cur` in place on success
    /// or kMinStep. Leaves `cur` untouched on kNanDetected.
    Status find_acceptable_iterate(const P& problem,
                                   Solution<P, NH>& cur,
                                   const QpSol<P, NH>& step,
                                   const SqpSlacks<P, NH>& slacks,
                                   Solution<P, NH>& scratch,
                                   double& alpha) const
    {
        const int N = cur.N;
        assert(N_ == N && step.N == N && slacks.N == N
               && scratch.N == N);

        update_weights(step);
        const double merit0 = merit(problem, cur, slacks);

        SqpSlacks<P, NH> trial_slacks(N);
        double a = 1.0;
        double merit1 = merit0;

        while (a * opts.alpha_reduction > opts.alpha_min)
        {
            detail::apply_sqp_step_primal<P, NH>(cur, step, a,
                                                  scratch);
            detail::shift_slacks(slacks, step, a, trial_slacks);
            merit1 = merit(problem, scratch, trial_slacks);
            if (merit1 < merit0 && std::isfinite(merit1))
            {
                alpha = a;
                apply_sqp_step<P, NH>(cur, step, a,
                                       opts.full_step_dual, cur);
                return Status::kSolved;
            }
            a *= opts.alpha_reduction;
        }

        alpha = a;
        if (!std::isfinite(merit1))
        {
            return Status::kNanDetected;
        }
        apply_sqp_step<P, NH>(cur, step, a,
                               opts.full_step_dual, cur);
        return Status::kMinStep;
    }

private:
    // -- slack penalty: 0.5 * w * s^2 per soft side ------------------

    template <class SV>
    static double add_slack_pair(const SV& sv, int base,
                                 int col_lo, int col_hi, double w)
    {
        double m = 0.0;
        if (col_lo >= 0)
        {
            const double s =
                static_cast<double>(sv[col_lo - base]);
            m += 0.5 * w * s * s;
        }
        if (col_hi >= 0)
        {
            const double s =
                static_cast<double>(sv[col_hi - base]);
            m += 0.5 * w * s * s;
        }
        return m;
    }

    template <class SV, class IdxLo, class IdxHi>
    double stage_slack_pen(const P& problem, const SV& sv, int base,
                           const IdxLo& idxl, const IdxHi& idxh,
                           const detail::QpLayout& lay,
                           int k) const
    {
        double m = 0.0;
        if constexpr (P::nbx_soft > 0)
        {
            if (lay.rows[detail::g_bx] > 0)
            {
                const auto spec =
                    problem.stage_state_box_constr(k);
                for (int i = 0; i < P::nbx_soft; ++i)
                {
                    const int r = lay.row_off(detail::g_bx)
                                 + P::state_box_soft_idx[i];
                    m += add_slack_pair(
                        sv, base, idxl[r], idxh[r],
                        static_cast<double>(
                            spec.soft_penalty(
                                P::state_box_soft_idx[i])));
                }
            }
        }
        if constexpr (P::nbu_soft > 0)
        {
            const auto spec =
                problem.stage_control_box_constr(k);
            for (int i = 0; i < P::nbu_soft; ++i)
            {
                const int r = lay.row_off(detail::g_bu)
                             + P::control_box_soft_idx[i];
                m += add_slack_pair(
                    sv, base, idxl[r], idxh[r],
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
                const int r = lay.row_off(detail::g_ineq)
                             + P::ineq_soft_idx[i];
                m += add_slack_pair(
                    sv, base, idxl[r], idxh[r],
                    static_cast<double>(pen(i)));
            }
        }
        if constexpr (P::ne_soft > 0)
        {
            const auto pen =
                problem.stage_equality_constr_soft_penalty(k);
            for (int i = 0; i < P::ne_soft; ++i)
            {
                const int r = lay.row_off(detail::g_eq)
                             + P::eq_soft_idx[i];
                m += add_slack_pair(
                    sv, base, idxl[r], idxh[r],
                    static_cast<double>(pen(i)));
            }
        }
        if constexpr (P::nl_soft > 0)
        {
            const auto spec = problem.stage_linear_constr(k);
            for (int i = 0; i < P::nl_soft; ++i)
            {
                const int r = lay.row_off(detail::g_lin)
                             + P::lin_soft_idx[i];
                m += add_slack_pair(
                    sv, base, idxl[r], idxh[r],
                    static_cast<double>(spec.bounds.soft_penalty(
                        P::lin_soft_idx[i])));
            }
        }
        return m;
    }

    double slack_penalty(const P& problem,
                         const SqpSlacks<P, NH>& sl) const
    {
        double m = 0.0;
        for (int k = 0; k < N_; ++k)
        {
            if (k == 0)
            {
                m += stage_slack_pen(
                    problem, sl.first,
                    D::nvar_first - D::nslack_first,
                    D::idxs_lo_first, D::idxs_hi_first,
                    D::lay_first, 0);
            }
            else
            {
                m += stage_slack_pen(
                    problem, sl.path[k - 1],
                    D::nvar_path - D::nslack_path,
                    D::idxs_lo_path, D::idxs_hi_path,
                    D::lay_path, k);
            }
        }
        // terminal
        {
            const auto& sv = sl.term;
            const int base = D::nvar_term - D::nslack_term;
            const auto& idxl = D::idxs_lo_term;
            const auto& idxh = D::idxs_hi_term;
            const auto& lay = D::lay_term;
            if constexpr (P::nbx_t_soft > 0)
            {
                const auto spec =
                    problem.terminal_state_box_constr();
                for (int i = 0; i < P::nbx_t_soft; ++i)
                {
                    const int r = lay.row_off(detail::g_bx)
                                 + P::terminal_state_box_soft_idx[i];
                    m += add_slack_pair(
                        sv, base, idxl[r], idxh[r],
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
                    m += add_slack_pair(
                        sv, base, idxl[r], idxh[r],
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
                    m += add_slack_pair(
                        sv, base, idxl[r], idxh[r],
                        static_cast<double>(pen(i)));
                }
            }
            if constexpr (P::nl_t_soft > 0)
            {
                const auto spec =
                    problem.terminal_linear_constr();
                for (int i = 0; i < P::nl_t_soft; ++i)
                {
                    const int r = lay.row_off(detail::g_lin)
                                 + P::terminal_lin_soft_idx[i];
                    m += add_slack_pair(
                        sv, base, idxl[r], idxh[r],
                        static_cast<double>(
                            spec.bounds.soft_penalty(
                                P::terminal_lin_soft_idx[i])));
                }
            }
        }
        return m;
    }

    // -- weighted positive constraint violations ---------------------

    double stage_violation(const P& problem,
                           const Solution<P, NH>& sol,
                           const SqpSlacks<P, NH>& sl,
                           int k) const
    {
        const auto& x = sol.x[k];
        const auto& u = sol.u[k];
        const bool first = (k == 0);
        const detail::QpLayout& lay =
            first ? D::lay_first : D::lay_path;
        const auto& sv = first ? sl.first : sl.path[k - 1];
        const auto& idxl =
            first ? D::idxs_lo_first : D::idxs_lo_path;
        const auto& idxh =
            first ? D::idxs_hi_first : D::idxs_hi_path;
        const int base = first
            ? D::nvar_first - D::nslack_first
            : D::nvar_path - D::nslack_path;
        const int R = first ? D::nrow_first : D::nrow_path;

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
            lv = lin_spec.A * x + lin_spec.B * u;
        }

        double m = 0.0;
        for (int r = 0; r < R; ++r)
        {
            const int g = lay.group_of(r);
            if (g == detail::g_pin)
                continue;
            const int j = r - lay.row_off(g);
            const double s_lo = (idxl[r] >= 0)
                ? static_cast<double>(sv[idxl[r] - base])
                : 0.0;
            const double s_hi = (idxh[r] >= 0)
                ? static_cast<double>(sv[idxh[r] - base])
                : 0.0;

            if (g == detail::g_bx)
            {
                const auto spec =
                    problem.stage_state_box_constr(k);
                const double v = static_cast<double>(
                    x(P::state_box_idx[j]));
                m += static_cast<double>(w_box_state[k](j))
                   * std::max(0.0,
                       static_cast<double>(spec.lo(j))
                       - s_lo - v);
                m += static_cast<double>(
                         w_box_state[k](D::nbx + j))
                   * std::max(0.0,
                       v - static_cast<double>(spec.hi(j))
                       - s_hi);
            }
            else if (g == detail::g_bu)
            {
                const auto spec =
                    problem.stage_control_box_constr(k);
                const double v = static_cast<double>(
                    u(P::control_box_idx[j]));
                m += static_cast<double>(w_box_control[k](j))
                   * std::max(0.0,
                       static_cast<double>(spec.lo(j))
                       - s_lo - v);
                m += static_cast<double>(
                         w_box_control[k](D::nbu + j))
                   * std::max(0.0,
                       v - static_cast<double>(spec.hi(j))
                       - s_hi);
            }
            else if (g == detail::g_ineq)
            {
                const double v =
                    static_cast<double>(gv(j));
                m += static_cast<double>(w_ineq_stage[k](j))
                   * std::max(0.0, v - s_hi);
            }
            else if (g == detail::g_eq)
            {
                const double e =
                    static_cast<double>(ev(j));
                m += static_cast<double>(w_eq_stage[k](j))
                   * (std::max(0.0, -s_lo - e)
                      + std::max(0.0, e - s_hi));
            }
            else  // g_lin
            {
                const double v =
                    static_cast<double>(lv(j));
                const double lo =
                    static_cast<double>(
                        lin_spec.bounds.lo(j));
                const double hi =
                    static_cast<double>(
                        lin_spec.bounds.hi(j));
                m += static_cast<double>(w_lin_stage[k](j))
                   * (std::max(0.0, lo - s_lo - v)
                      + std::max(0.0, v - s_hi - hi));
            }
        }
        return m;
    }

    double term_violation(const P& problem,
                          const Solution<P, NH>& sol,
                          const SqpSlacks<P, NH>& sl) const
    {
        const auto& x = sol.x[sol.N];
        const auto& sv = sl.term;
        const auto& idxl = D::idxs_lo_term;
        const auto& idxh = D::idxs_hi_term;
        const int base = D::nvar_term - D::nslack_term;
        const detail::QpLayout& lay = D::lay_term;

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
            lv = lin_spec.A * x;
        }

        double m = 0.0;
        for (int r = 0; r < D::nrow_term; ++r)
        {
            const int g = lay.group_of(r);
            const int j = r - lay.row_off(g);
            const double s_lo = (idxl[r] >= 0)
                ? static_cast<double>(sv[idxl[r] - base])
                : 0.0;
            const double s_hi = (idxh[r] >= 0)
                ? static_cast<double>(sv[idxh[r] - base])
                : 0.0;

            if (g == detail::g_bx)
            {
                const auto spec =
                    problem.terminal_state_box_constr();
                const double v = static_cast<double>(
                    x(P::terminal_state_box_idx[j]));
                m += static_cast<double>(w_box_term(j))
                   * std::max(0.0,
                       static_cast<double>(spec.lo(j))
                       - s_lo - v);
                m += static_cast<double>(
                         w_box_term(D::nbx_t + j))
                   * std::max(0.0,
                       v - static_cast<double>(spec.hi(j))
                       - s_hi);
            }
            else if (g == detail::g_ineq)
            {
                const double v =
                    static_cast<double>(gv(j));
                m += static_cast<double>(w_ineq_term(j))
                   * std::max(0.0, v - s_hi);
            }
            else if (g == detail::g_eq)
            {
                const double e =
                    static_cast<double>(ev(j));
                m += static_cast<double>(w_eq_term(j))
                   * (std::max(0.0, -s_lo - e)
                      + std::max(0.0, e - s_hi));
            }
            else  // g_lin
            {
                const double v =
                    static_cast<double>(lv(j));
                const double lo =
                    static_cast<double>(
                        lin_spec.bounds.lo(j));
                const double hi =
                    static_cast<double>(
                        lin_spec.bounds.hi(j));
                m += static_cast<double>(w_lin_term(j))
                   * (std::max(0.0, lo - s_lo - v)
                      + std::max(0.0, v - s_hi - hi));
            }
        }
        return m;
    }
};

}  // namespace ocp

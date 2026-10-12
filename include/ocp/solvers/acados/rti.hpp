// ocp/solvers/acados/rti.hpp
//
// RTI (real-time iteration) SQP solver, standard form (as_rti_iter = 0,
// as_rti_level = STANDARD_RTI). Phase 3 of the SQP+HPIPM parity plan
// (SQP_HPIPM_PARITY_PHASE3_PLAN.md), block 3b.
//
// One solve() call performs exactly one Newton step:
//   prepare  - LHS assembly at the current iterate (Hessians + Jacobians)
//              plus the fixed Levenberg-Marquardt shift
//              (acados ocp_nlp_sqp_rti_preparation_step)
//   feedback - RHS assembly (gradients + offsets), regularization, QP
//              solve in the active QpMode, dual correction, and the full
//              primal step (alpha = 1, no line search)
//              (acados ocp_nlp_sqp_rti_feedback_step)
//
// Design notes (see the phase-3 plan, open items):
// - The LHS/RHS split reuses the free functions of assemble.hpp (3a);
//   prepare() fills the LHS into qp_in_, feedback() the RHS.
// - QP scaling (phase 3h) is deliberately not applied: the acados
//   standard-RTI path does not scale the QP.
// - The QP-solve dispatch (staged / full / partial condensing) is a
//   parallel copy of AcadosSolver::solve_qp (plan open item 3): sharing
//   it would mean refactoring the tested phase 1/2 driver, which costs
//   more than the copy.
// - Cost / slack-penalty computation uses the shared free functions of
//   cost.hpp.

#pragma once

#include <cassert>
#include <chrono>
#include <cstdio>

#include "acados.hpp"    // QpMode, NlpResiduals, compute_nlp_residuals
#include "assemble.hpp"  // detail::eval_stage/eval_term, fill_stage_*
#include "cost.hpp"      // raw_cost, slack_penalty, compute_nlp_cost

namespace ocp
{

/// RTI SQP options - mirrors the acados ocp_nlp_sqp_rti_opts fields that
/// apply to the standard (non-as-RTI) path.
struct RtiOptions
{
    // LHS Hessian gates (mirror AcadosOptions / acados exact_hess_*)
    bool compute_hess_dyn = true;
    bool compute_hess_constr = true;

    // Fixed Levenberg-Marquardt shift on the QP Hessian. Standard RTI
    // passes alpha = 1.0, iter = 0 to the acados LM term, so the adaptive
    // scheme is not involved; mu is used as-is (acados default 0.0).
    double levenberg_marquardt = 0.0;

    // QP solve mode (phase 1e / 2e)
    QpMode qp_mode = QpMode::kStaged;
    int    qp_cond_N = 0;

    // HPIPM warm-start mode (0 cold / 2 primal+dual / 3 hot)
    int  qp_warm_start = 0;
    bool warm_start_first_qp = false;

    // QP tolerance strategy (acados nlp_qp_tol_strategy). 0 = fixed
    // solver tolerances; 1 = ADAPTIVE_CURRENT_RES_JOINT. Strategy 2
    // (ADAPTIVE_QPSCALING) requires QP scaling, which standard RTI does
    // not perform, so it is not supported.
    int    nlp_qp_tol_strategy = 0;
    double nlp_qp_tol_reduction_factor = 0.1;
    double nlp_qp_tol_safety_factor = 0.1;
    double nlp_qp_tol_min_stat = 1e-9;
    double nlp_qp_tol_min_eq = 1e-10;
    double nlp_qp_tol_min_ineq = 1e-10;
    double nlp_qp_tol_min_comp = 1e-11;

    // Base NLP tolerances (floor for the adaptive QP tolerances, strategy 1)
    double tol_stat = 1e-8;
    double tol_eq = 1e-8;
    double tol_ineq = 1e-8;
    double tol_comp = 1e-8;

    // Complementarity floor for the NLP residuals (compute_nlp_residuals)
    double tau_min = 1e-16;

    // Residual logging / printing (phase 3c)
    bool rti_log_residuals = false;
    bool rti_log_only_available_residuals = false;
    int  print_level = 0;
};

/// One statistics row of an RtiSolver::solve() call.
struct RtiIteration
{
    // pre-step NLP residuals (valid when rti_log_residuals; zero otherwise)
    double res_stat = 0.0;
    double res_eq = 0.0;
    double res_ineq = 0.0;
    double res_comp = 0.0;
    // post-step NLP residuals (valid when rti_log_residuals and not
    // rti_log_only_available_residuals; zero otherwise)
    double res_stat_post = 0.0;
    double res_eq_post = 0.0;
    double res_ineq_post = 0.0;
    double res_comp_post = 0.0;
    int qp_status = 0;
    int qp_iter = 0;
    double step_norm = 0.0;
    double alpha = 1.0;     // always 1.0 in standard RTI
    double cond_time = 0.0;
};

/// Statistics of the most recent RtiSolver::solve() call.
struct RtiStatistics
{
    RtiIteration last{};
    int n_calls = 0;
};

/// Standard real-time-iteration SQP solver (one Newton step per solve).
///
/// Template parameter order: P (problem), NH (horizon), then the
/// pluggable components with defaults (same order as AcadosSolver).
template <class P, int NH = Eigen::Dynamic,
          class QpSolver = HpipmQpSolver<P, NH>,
          class Regularizer = GlmRegularizer<P, NH>, int NB = 0>
class RtiSolver
{
public:
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    explicit RtiSolver(RtiOptions opts = RtiOptions{})
        : opts_(opts)
    {
        if (NB == 0 && opts_.qp_mode == QpMode::kPartialCondensed)
        {
            std::fprintf(stderr,
                "RtiSolver: kPartialCondensed requested but the solver "
                "was compiled with NB = 0; falling back to kStaged.\n");
        }
    }

    /// LHS assembly at the current iterate (acados RTI preparation
    /// step): zero the QP stages, fill the Hessians (contracted terms
    /// per the compute_hess_* gates), the Jacobians (BA, DC) and the
    /// structural d_mask, then apply the fixed Levenberg-Marquardt
    /// shift. No RHS field is touched.
    ///
    /// The caller is responsible for the horizon: `sol.N` must match the
    /// solver's workspace (solve() resizes; direct prepare/feedback
    /// calls require a prior resize()).
    void prepare(const P& problem, const Solution<P, NH>& sol)
    {
        const int N = sol.N;
        assert(qp_in_.N == N);
        const detail::AssembleFlags fl{opts_.compute_hess_dyn,
                                       opts_.compute_hess_constr};

        qp_in_.first.setZero();
        for (int k = 1; k < N; ++k)
        {
            qp_in_.path[k - 1].setZero();
        }
        qp_in_.term.setZero();

        auto ev0 = detail::eval_stage(problem, sol, 0, fl);
        detail::fill_stage_lhs_first(problem, ev0, fl, qp_in_.first);
        for (int k = 1; k < N; ++k)
        {
            auto ev = detail::eval_stage(problem, sol, k, fl);
            detail::fill_stage_lhs_path(problem, k, ev, fl,
                                        qp_in_.path[k - 1]);
        }
        auto tev = detail::eval_term(problem, sol, fl);
        detail::fill_stage_lhs_term(problem, tev, fl, qp_in_.term);

        add_lm_term();
    }

    /// RHS assembly at the current iterate + regularization + QP solve +
    /// full step (acados RTI feedback step). The LHS must already be
    /// filled (prepare() or solve()); returns the first-stage degeneracy
    /// status (kInfeasible) without touching `sol` in that case.
    ///
    /// On success `sol` is advanced by the full QP step (alpha = 1,
    /// primal and duals, incl. the solver-internal slacks) and
    /// sol.status / sol.cost_value are updated.
    Status feedback(const P& problem, Solution<P, NH>& sol)
    {
        const int N = sol.N;
        assert(qp_in_.N == N);
        const detail::AssembleFlags fl{opts_.compute_hess_dyn,
                                       opts_.compute_hess_constr};

        // RHS fill (runs the first-stage degeneracy check)
        auto ev0 = detail::eval_stage(problem, sol, 0, fl);
        const Status st_rhs =
            detail::fill_stage_rhs_first(problem, sol, ev0, qp_in_.first);
        if (st_rhs != Status::kSolved)
        {
            sol.status = st_rhs;
            return st_rhs;
        }
        for (int k = 1; k < N; ++k)
        {
            auto ev = detail::eval_stage(problem, sol, k, fl);
            detail::fill_stage_rhs_path(problem, sol, k, ev,
                                        qp_in_.path[k - 1]);
        }
        auto tev = detail::eval_term(problem, sol, fl);
        detail::fill_stage_rhs_term(problem, sol, tev, qp_in_.term);

        // pre-step NLP residuals (acados feedback_step: nlp_res is
        // computed before the step; always available for the stats row)
        const NlpResiduals res =
            compute_nlp_residuals(problem, sol, &slacks_, opts_.tau_min);

        // full regularization (acados PREPARATION_AND_FEEDBACK phase;
        // regularize_rhs is a no-op for GLM)
        reg_.regularize(qp_in_);

        // adaptive QP tolerances (acados ocp_nlp_common.c:4522-4574)
        update_qp_tolerances(res);

        // QP warm start: cold on the very first call unless requested
        bool first_call = (call_count_ == 0);
        int ws = opts_.qp_warm_start;
        if (first_call && !opts_.warm_start_first_qp)
        {
            ws = 0;
        }

        const Status qp_st = solve_qp(qp_in_, qp_out_, ws);
        const int qp_iter = qp_iter_;
        if (qp_st != Status::kSolved && qp_st != Status::kMaxIterations)
        {
            // acados: a QP MAXITER is acceptable, any other failure is a
            // hard stop (ocp_nlp_sqp_rti.c:613-620)
            sol.status = Status::kQpFailure;
            record_row(res, NlpResiduals(), opts_.rti_log_residuals, false,
                       qp_st, qp_iter, 0.0, 1.0);
            return Status::kQpFailure;
        }

        // dual correction (no-op for GLM / NoReg; kept for parity with
        // ocp_nlp_solve_qp_and_correct_dual)
        reg_.correct_dual_sol(qp_in_, qp_out_);

        // full primal + dual step, alpha = 1 (no globalization in
        // standard RTI; the RHS is element-safe in place because every
        // update writes from an Eigen temporary)
        const double step_norm = primal_step_norm_inf(qp_out_);
        apply_sqp_step(sol, qp_out_, 1.0, true, sol);
        update_slacks(qp_out_, 1.0);

        sol.cost_value = compute_nlp_cost(problem, sol, slacks_);
        sol.status = Status::kSolved;

        // full residual recompute at the post-step iterate (acados
        // prepare_full_residual_computation + second ocp_nlp_res_compute);
        // the pre-step pass is already available from the tolerance
        // update above
        const bool log_pre = opts_.rti_log_residuals;
        const bool log_post =
            log_pre && !opts_.rti_log_only_available_residuals;
        NlpResiduals res_post;
        if (log_post)
        {
            res_post = compute_nlp_residuals(problem, sol, &slacks_,
                                             opts_.tau_min);
        }
        record_row(res, res_post, log_pre, log_post, qp_st, qp_iter,
                   step_norm, 1.0);
        return Status::kSolved;
    }

    /// One RTI iteration: prepare (LHS + LM) + feedback (RHS + solve +
    /// full step). Resizes the workspaces to sol.N on the first call.
    Status solve(const P& problem, Solution<P, NH>& sol)
    {
        const int N = sol.N;
        if (qp_in_.N != N || slack_N_ != N)
        {
            resize(N);
        }
        prepare(problem, sol);
        return feedback(problem, sol);
    }

    /// Size the internal workspaces for a horizon of `n_stages` stages
    /// (dynamic-extent mode allocates; fixed-extent mode only asserts).
    void resize(int n_stages)
    {
        qp_in_.resize(n_stages);
        qp_out_.resize(n_stages);
        if (slack_N_ != n_stages)
        {
            slacks_.setZero();
        }
        slacks_.resize(n_stages);
        slack_N_ = n_stages;
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

    const RtiStatistics& statistics() const { return stat_; }
    const Qp<P, NH>& last_qp() const { return qp_in_; }
    const QpSol<P, NH>& last_qp_sol() const { return qp_out_; }

private:
    /// Fixed Levenberg-Marquardt shift: mu * I on the (u;x) diagonal of
    /// every stage (acados ocp_nlp_add_levenberg_marquardt_term with
    /// alpha = 1.0, iter = 0 and with_adaptive_levenberg_marquardt = 0;
    /// cost-module scaling factor is 1 in this interface).
    void add_lm_term()
    {
        if ((!opts_.compute_hess_dyn && !opts_.compute_hess_constr)
            || opts_.levenberg_marquardt <= 0.0)
        {
            return;
        }
        const double mu = opts_.levenberg_marquardt;
        constexpr int nux = D::nu + D::nx;
        for (int i = 0; i < nux; ++i)
        {
            qp_in_.first.hess(i, i) += static_cast<S>(mu);
        }
        for (int k = 1; k < qp_in_.N; ++k)
        {
            for (int i = 0; i < nux; ++i)
            {
                qp_in_.path[k - 1].hess(i, i) += static_cast<S>(mu);
            }
        }
        for (int i = 0; i < D::nx; ++i)
        {
            qp_in_.term.hess(i, i) += static_cast<S>(mu);
        }
    }

    /// QP-solve dispatch on opts_.qp_mode: a parallel copy of
    /// AcadosSolver::solve_qp (phase 3 plan, open item 3).
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
        qp_iter_ = qp_.statistics().iter;
        return st;
    }

    Status solve_qp_full_condensed(const Qp<P, NH>& in, QpSol<P, NH>& out,
                                   int ws)
    {
        const auto t0 = std::chrono::steady_clock::now();
        condense_full(in, dense_qp_in_);
        dense_qp_out_.v.setZero();
        qp_dense_.set_warm_start(ws);
        const Status st = qp_dense_.solve(dense_qp_in_, dense_qp_out_);
        expand_full_sol(in, dense_qp_in_, dense_qp_out_, out);
        cond_time_ += std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
        return st;
    }

    Status solve_qp_partial_condensed(const Qp<P, NH>& in,
                                      QpSol<P, NH>& out, int ws)
    {
        if constexpr (NB > 0)
        {
            const int N = in.N;
            const int N2 = (opts_.qp_cond_N > 0) ? opts_.qp_cond_N : N;
            if (N2 >= N || N % N2 != 0 || N / N2 != NB)
            {
                // identity fast path / block size mismatch: staged solve
                out.ux_first.setZero();
                for (int k = 1; k < out.N; ++k)
                {
                    out.ux_path[k - 1].setZero();
                }
                out.ux_term.setZero();
                qp_.set_warm_start(ws);
                const Status st = qp_.solve(in, out);
                qp_iter_ = qp_.statistics().iter;
                return st;
            }
            const auto t0 = std::chrono::steady_clock::now();
            condense_block<P, NH, NB>(in, block_qp_.in);
            // resize() keeps the persistent duals (warm start); only the
            // primal is zeroed.
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
            qp_iter_ = qp_.statistics().iter;
            return st;
        }
    }

    /// Per-call QP tolerance update (acados
    /// ocp_nlp_common.c:4522-4574). No-op when strategy is 0
    /// (fixed solver tolerances).
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
            // strategy 2 (ADAPTIVE_QPSCALING) needs QP scaling, which the
            // standard RTI path does not perform
            std::fprintf(stderr,
                "RtiSolver: nlp_qp_tol_strategy 2 requires QP scaling, "
                "which standard RTI does not perform; using fixed "
                "tolerances.\n");
            return;
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

    /// Inf-norm of the primal step, max over stages (incl. slacks).
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
    /// s <- s + alpha * (s_qp - s) (QP slacks are absolute; the full RTI
    /// step uses alpha = 1, i.e. s <- s_qp).
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

    void record_row(const NlpResiduals& res_pre, const NlpResiduals& res_post,
                    bool log_pre, bool log_post, Status qp_st, int qp_iter,
                    double step_norm, double alpha)
    {
        auto& r = stat_.last;
        r.res_stat = log_pre ? res_pre.res_stat : 0.0;
        r.res_eq = log_pre ? res_pre.res_eq : 0.0;
        r.res_ineq = log_pre ? res_pre.res_ineq : 0.0;
        r.res_comp = log_pre ? res_pre.res_comp : 0.0;
        r.res_stat_post = log_post ? res_post.res_stat : 0.0;
        r.res_eq_post = log_post ? res_post.res_eq : 0.0;
        r.res_ineq_post = log_post ? res_post.res_ineq : 0.0;
        r.res_comp_post = log_post ? res_post.res_comp : 0.0;
        r.qp_status = static_cast<int>(qp_st);
        r.qp_iter = qp_iter;
        r.step_norm = step_norm;
        r.alpha = alpha;
        r.cond_time = cond_time_;
        stat_.n_calls += 1;
        call_count_ += 1;
        if (opts_.print_level > 0)
        {
            if (log_post)
            {
                std::printf(
                    "  rti call %d: res %.3e %.3e %.3e %.3e (post "
                    "%.3e %.3e %.3e %.3e), qp_status %d, qp_iter %d, "
                    "step %.3e, alpha %.1f, cond %.3e s\n",
                    stat_.n_calls, r.res_stat, r.res_eq, r.res_ineq,
                    r.res_comp, r.res_stat_post, r.res_eq_post,
                    r.res_ineq_post, r.res_comp_post, r.qp_status, qp_iter,
                    step_norm, alpha, cond_time_);
            }
            else
            {
                std::printf(
                    "  rti call %d: res %.3e %.3e %.3e %.3e, qp_status %d, "
                    "qp_iter %d, step %.3e, alpha %.1f, cond %.3e s\n",
                    stat_.n_calls, r.res_stat, r.res_eq, r.res_ineq,
                    r.res_comp, r.qp_status, qp_iter, step_norm, alpha,
                    cond_time_);
            }
        }
    }

    RtiOptions opts_;
    Qp<P, NH> qp_in_;
    QpSol<P, NH> qp_out_;
    SqpSlacks<P, NH> slacks_;
    int slack_N_ = 0;
    QpSolver qp_;
    Regularizer reg_;
    // full condensing (phase 1e)
    HpipmDenseQpSolver<P, NH> qp_dense_;
    DenseQp<P, NH> dense_qp_in_;
    DenseQpSol<P, NH> dense_qp_out_;
    // partial condensing (phase 2e)
    detail::BlockQpWorkspaces<P, NB> block_qp_;
    RtiStatistics stat_{};
    int call_count_ = 0;
    int qp_iter_ = 0;
    double cond_time_ = 0.0;
};

}  // namespace ocp

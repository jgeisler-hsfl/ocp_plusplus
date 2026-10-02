// ocp/solvers/hpipm/hpipm.hpp
//
// HPIPM-style primal-dual interior-point solver for the staged QP data model
// of qp.hpp. Port of the HPIPM OCP-QP IPM (relative formulation, Mehrotra
// predictor-corrector) with a 1:1 method mapping to the HPIPM macros
// (acados/docs/algorithm/05-hpipm-qp-solver.md, sec. 3-4):
//
//   OCP_QP_INIT_VAR                    -> init_point           (worklog 1c)
//   OCP_QP_RES_COMPUTE                 -> compute_residuals    (worklog 1b)
//   OCP_QP_FACT_SOLVE_KKT_STEP         -> fact_solve_kkt       (worklog 1e)
//   OCP_QP_SOLVE_KKT_STEP              -> solve_kkt            (worklog 1e)
//   inequality-free KKT fast path      -> solve_kkt_unconstr   (worklog 1e)
//   COMPUTE_ALPHA_QP                   -> compute_alpha        (worklog 1f)
//   COMPUTE_MU_AFF_QP                  -> compute_mu_aff       (worklog 1g)
//   COMPUTE_CENTERING(_CORRECTION)_QP  -> apply_centering      (worklog 1g)
//   UPDATE_VAR_QP                      -> update_vars          (worklog 1f)
//
// HPIPM references (re-read at implementation time; deviations are recorded
// in the SQP_PHASE1_WORKLOG.md checkpoints):
//   acados/external/hpipm/include/hpipm_d_ocp_qp_ipm.h  (arg fields)
//   acados/external/hpipm/ocp_qp/x_ocp_qp_ipm.c         (presets, main loop,
//                                                        stat, exit/status)
//   acados/external/hpipm/core/x_core_qp_ipm_aux.c      (COMPUTE_ALPHA etc.)
//   acados/external/hpipm/ocp_qp/x_ocp_qp_res.c         (residual definitions)
//   acados/docs/algorithm/05-hpipm-qp-solver.md
//
// v1 scope (SQP_PLAN.md sec. 2): relative formulation only (abs_form = 0),
// Cholesky only (lq_fact = 0), no iterative refinement, no split step; the
// warm start is driven by the SQP driver. The horizon NH is a template
// parameter (as for Qp / QpSol / QpRes): path-wise workspace storage uses the
// Trajectory mechanism, runtime-extent (sized at solve()) for
// NH = Eigen::Dynamic, fixed-extent (std::array) for a compile-time NH.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "qp.hpp"

namespace ocp
{

// =========================================================================
//  Options
// =========================================================================

/// IPM arguments (the scalar fields of HPIPM `d_ocp_qp_ipm_arg`).
///
/// Defaults are the HPIPM BALANCE preset (`x_ocp_qp_ipm.c:145-179`) plus the
/// acados overrides (`ocp_qp_hpipm.c:101-113`): res_g_max 1e-6, res_b/d/m_max
/// 1e-8, iter_max = stat_max = 50, alpha_min 1e-8, mu0 1.0,
/// var_init_scheme 1; plus the v1 minimal-scope flags (SQP_PLAN.md sec. 2):
/// lq_fact = 0, itref_*_max = 0, split_step = 0, abs_form = 0.
struct HpipmOptions
{
    double mu0 = 1.0;             // initial barrier parameter (acados: 1e0)
    double alpha_min = 1e-8;     // exit cond on step length (acados; HPIPM 1e-12)
    double res_g_max = 1e-6;     // exit cond: inf norm of stationarity residual
    double res_b_max = 1e-8;     // exit cond: inf norm of dynamics residual
    double res_d_max = 1e-8;     // exit cond: inf norm of ineq feasibility
    double res_m_max = 1e-8;     // exit cond: inf norm of (tau-shifted) res_m
    double dual_gap_max = 1e15;  // exit cond on the duality gap
    double reg_prim = 1e-15;     // primal Hessian regularization
    double lam_min = 1e-16;      // min value in the lam vector (solution)
    double t_min = 1e-16;        // min value in the t vector (solution)
    double tau_min = 1e-16;      // min value of the barrier parameter
    double lam0_min = 1e-9;      // min lam at (warm-)start initialization
    double t0_min = 1e-9;        // min t at (warm-)start initialization
    double m_safe = 0.5;         // aggressiveness/robustness trade-off in [0,1]
    int iter_max = 50;           // IPM iteration cap (acados; HPIPM BALANCE 30)
    int stat_max = 50;           // iteration records kept (acados; HPIPM 30)
    bool pred_corr = true;      // Mehrotra predictor-corrector
    bool cond_pred_corr = true; // conditional predictor-corrector
    int itref_pred_max = 0;     // predictor refinement (v1: off; BALANCE 0)
    int itref_corr_max = 0;     // corrector refinement (v1: off; BALANCE 2)
    int warm_start = 0;         // 0 none / 1 primal / 2 primal+dual / 3 hot
    bool square_root_alg = true;  // 1: square-root Riccati (0: classical)
    int lq_fact = 0;            // v1: Cholesky only (HPIPM BALANCE: 1)
    int abs_form = 0;           // v1: relative formulation only
    int split_step = 0;         // v1: single step for primal and dual
    int var_init_scheme = 1;    // acados override (HPIPM: 0)
    int t_lam_min = 2;          // clip t and lam: 0 no / 1 in Gamma / 2 in sol
    int t0_init = 2;            // slack init: 0 sqrt(mu0) / 1 1.0 / 2 heuristic
    int update_fact_exit = 1;   // leave an updated factorization on exit
};

// =========================================================================
//  Statistics
// =========================================================================

/// One row of the HPIPM `stat` matrix: the 21 per-IPM-iteration quantities
/// in the column order printed by acados (`ocp_qp_hpipm.c:371`;
/// acados/docs/algorithm/05-hpipm-qp-solver.md sec. 5.3).
struct HpipmIteration
{
    double alpha_prim_aff = 0;  // affine (predictor) primal step length
    double alpha_dual_aff = 0;  // affine (predictor) dual step length
    double mu_aff = 0;          // barrier parameter of the affine step
    double sigma = 0;           // Mehrotra centering parameter
    double alpha_prim = 0;      // accepted primal step length
    double alpha_dual = 0;      // accepted dual step length
    double mu = 0;              // barrier parameter after the update
    double res_stat = 0;        // inf norm: stationarity residual
    double res_eq = 0;          // inf norm: dynamics (equality) residual
    double res_ineq = 0;        // inf norm: inequality feasibility residual
    double res_comp = 0;        // inf norm: (tau-shifted) complementarity
    double dual_gap = 0;        // primal - dual objective gap
    double obj = 0;             // primal objective at the iterate
    int lq_fact = 0;            // LQ mode actually used at this iteration
    int itref_pred = 0;         // predictor iterative-refinement steps
    int itref_corr = 0;         // corrector iterative-refinement steps
    double lin_res_stat = 0;   // linearized residual (iterative refinement)
    double lin_res_eq = 0;
    double lin_res_ineq = 0;
    double lin_res_comp = 0;
    int npd_reg_hess = 0;       // 1: Hessian found non-PSD, was regularized
};

/// Per-IPM-iteration records (HPIPM `stat` matrix analogue: 21 columns x
/// (stat_max + 1) rows; row 0 holds the pre-iteration residuals, row kk
/// (1..stat_max) the state after IPM iteration kk - 1).
struct HpipmStatistics
{
    int iter = 0;                   // IPM iterations performed
    int stat_max = 0;               // record capacity
    Status status = Status::kUnset; // solver status at return
    std::vector<HpipmIteration> rows;  // size stat_max + 1 after init()

    /// Allocate stat_max + 1 rows and reset the counters.
    void init(int n)
    {
        stat_max = n;
        iter = 0;
        status = Status::kUnset;
        rows.assign(n + 1, HpipmIteration{});
    }

    HpipmIteration& row(int i) { return rows[i]; }
    const HpipmIteration& row(int i) const { return rows[i]; }
};

// =========================================================================
//  Solver
// =========================================================================

/// HPIPM-style primal-dual interior-point solver for a staged QP.
///
/// Templated over the concrete problem type `P` and the horizon `NH`,
/// following the Qp / QpSol / QpRes pattern: `NH = Eigen::Dynamic`
/// (default) keeps N a runtime quantity; a non-negative `NH` fixes it at
/// compile time and switches the per-stage workspace storage to
/// fixed-extent (std::array) Trajectories.
///
/// The solve direction follows the HPIPM convention (see QpSol): `out`
/// carries the primal STEP over the stage-type variable vectors and the
/// ABSOLUTE multipliers (pi, lam, t).
template <class P, int NH = Eigen::Dynamic>
class HpipmQpSolver
{
    static_assert(NH == Eigen::Dynamic || NH >= 1,
        "ocp::HpipmQpSolver: the horizon must be at least 1 (one dynamics "
        "step) or Eigen::Dynamic.");
public:
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    // Per-stage-type matrix / vector aliases (compile-time extents).
    using M_first_t = Eigen::Matrix<S, D::nvar_first, D::nvar_first>;
    using M_path_t  = Eigen::Matrix<S, D::nvar_path,  D::nvar_path>;
    using M_term_t  = Eigen::Matrix<S, D::nvar_term,  D::nvar_term>;
    using v_first_t = Eigen::Matrix<S, D::nvar_first, 1>;
    using v_path_t  = Eigen::Matrix<S, D::nvar_path,  1>;
    using v_term_t  = Eigen::Matrix<S, D::nvar_term,  1>;
    using c_first_t = Eigen::Matrix<S, D::nside_first, 1>;
    using c_path_t  = Eigen::Matrix<S, D::nside_path,  1>;
    using c_term_t  = Eigen::Matrix<S, D::nside_term,  1>;
    using Pb_first_t = Eigen::Matrix<S, D::nx, D::nvar_first>;
    using Pb_path_t  = Eigen::Matrix<S, D::nx, D::nvar_path>;
    using Pb_term_t  = Eigen::Matrix<S, D::nx, D::nvar_term>;
    using Pi_t        = Eigen::Matrix<S, D::nx, 1>;
    using Ba_t        = Eigen::Matrix<S, D::nx, D::nu + D::nx>;
    // init_point bound tables (row, varidx) pairs; compile-time sized
    using BoundFirstArr = std::array<std::pair<int, int>, D::nbound_first>;
    using BoundPathArr  = std::array<std::pair<int, int>, D::nbound_path>;
    using BoundTermArr  = std::array<std::pair<int, int>, D::nbound_term>;
    // Riccati state Hessian P_k (x-space Schur complement of the augmented
    // stage Hessian) and the costate linear term q_k (worklog 1e).
    using P_first_t   = Eigen::Matrix<S, D::nx, D::nx>;
    using P_path_t    = Eigen::Matrix<S, D::nx, D::nx>;
    using P_term_t    = Eigen::Matrix<S, D::nx, D::nx>;

    /// Horizon-sized solver workspace (internal state; sized in solve()).
    ///
    /// first / term stages are single objects; the per-stage quantities over
    /// the path stages (k = 1..N-1) and over the dynamics steps
    /// (k = 0..N-1) are Trajectories with the same NH mechanism as
    /// Qp / QpSol: runtime-extent (std::vector, resized in solve()) for
    /// NH = Eigen::Dynamic, fixed-extent (std::array) for a compile-time
    /// horizon. Factors (M, L) and cached products (Pb) are filled by
    /// fact_solve_kkt (worklog 1e); the delta step vectors by solve_kkt;
    /// gamma / t_inv by the centering step.
    struct Workspace
    {
        int N = (NH == Eigen::Dynamic) ? 0 : NH;

        // stage matrices and their Cholesky factors (M_k = H_k + reg*I +
        // DC_k' diag(Gamma_k) DC_k; L_k the square-root Riccati factor)
        M_first_t M_first{};
        M_first_t L_first{};
        Trajectory<M_path_t, detail::traj_extent<NH, -1>()> M_path;
        Trajectory<M_path_t, detail::traj_extent<NH, -1>()> L_path;
        M_term_t M_term{};
        M_term_t L_term{};

        // LQ-factor Hessian blocks Lh_k = chol(H_k + reg*I_(u;x)) cached
        // across IPM iterations (HPIPM `use_hess_fact`); valid only when
        // `lh_reg` equals the reg in force (a reg-ladder growth invalidates
        // the whole set by changing `lh_reg`).
        M_first_t Lh_first{};
        Trajectory<M_path_t, detail::traj_extent<NH, -1>()> Lh_path;
        M_term_t Lh_term{};
        S lh_reg = S(-1);

        // cached forward-pass product (BAbt_k * L_{k+1,xx}'), doc 05 sec. 4
        Pb_first_t Pb_first{};
        Trajectory<Pb_path_t, detail::traj_extent<NH, -1>()> Pb_path;
        Pb_term_t Pb_term{};

        // Riccati state Hessian P_k (x-space Schur complement of the
        // augmented stage Hessian Mtilde_k), per stage k = 0..N
        P_first_t P_first{};
        Trajectory<P_path_t, detail::traj_extent<NH, -1>()> P_path;
        P_term_t P_term{};

        // Riccati costate linear term q_k, per stage k = 0..N
        Pi_t q_first{};
        Trajectory<Pi_t, detail::traj_extent<NH, -1>()> q_path;
        Pi_t q_term{};

        // reduced Newton RHS rtilde_k = rtilde_k - BAbt_k'(P_{k+1} res_b_k
        // + q_{k+1}), in the (u;s;x) ordering, per stage k = 0..N
        v_first_t rtilde_first{};
        Trajectory<v_path_t, detail::traj_extent<NH, -1>()> rtilde_path;
        v_term_t rtilde_term{};

        // KKT residuals (HPIPM d_ocp_qp_res analogue)
        QpRes<P, NH> res;
        // unshifted complementarity residual backup (HPIPM res_m_bkp,
        // BACKUP_RES_M): the centering / pure-centering targets are built
        // from this pre-tau-shift snapshot
        QpRes<P, NH> res_bkp;

        // per-side gamma / Gamma (HPIPM COMPUTE_GAMMA, x_core_qp_ipm_aux.c)
        c_first_t gamma_first{};
        Trajectory<c_path_t, detail::traj_extent<NH, -1>()> gamma_path;
        c_term_t gamma_term{};
        c_first_t Gamma_first{};
        Trajectory<c_path_t, detail::traj_extent<NH, -1>()> Gamma_path;
        c_term_t Gamma_term{};

        // per-side 1/t (line search and clipping)
        c_first_t t_inv_first{};
        Trajectory<c_path_t, detail::traj_extent<NH, -1>()> t_inv_path;
        c_term_t t_inv_term{};

        // Newton-system right-hand side: rhs_k = -res_g_k - DC_k'(sigma*gamma)
        v_first_t rhs_first{};
        Trajectory<v_path_t, detail::traj_extent<NH, -1>()> rhs_path;
        v_term_t rhs_term{};

        // Newton step (delta variables)
        v_first_t dv_first{};
        Trajectory<v_path_t, detail::traj_extent<NH, -1>()> dv_path;
        v_term_t dv_term{};
        Trajectory<Pi_t, detail::traj_extent<NH, 0>()> dpi;  // k = 0..N-1
        c_first_t dlam_first{};
        Trajectory<c_path_t, detail::traj_extent<NH, -1>()> dlam_path;
        c_term_t dlam_term{};
        c_first_t dt_first{};
        Trajectory<c_path_t, detail::traj_extent<NH, -1>()> dt_path;
        c_term_t dt_term{};

        // IPM scalars
        S mu = 0;
        S mu_aff = 0;
        S sigma = 0;
        S alpha_prim = 0;  // accepted primal step length (drives v, t)
        S alpha_dual = 0;  // accepted dual step length (drives pi, lam)
        S reg = 0;  // current reg_prim (grown on Cholesky failure)

        int iter = 0;
        bool npd_reg_hess = false;

        // reg-primal growth ladder (HPIPM compute_preg, x_ocp_qp_kkt.c:1166-
        // 1210): the last successfully-used reg, persistent across IPM
        // iterations within a solve (reset in solve()).
        S preg_last = S(0);
        // LQ auto-switch latch (HPIPM force_lq, x_ocp_qp_ipm.c:2259): once a
        // mode-1 Cholesky auto-switch fires, stay on LQ for the rest of the
        // solve. Reset in solve().
        bool force_lq = false;
        // LQ mode actually used at the current IPM iteration (stat col 13).
        int last_lq_fact = 0;
        // iterative-refinement scratch (3d): the linearized residual and the
        // correction solve (reused factors).
        QpRes<P, NH> res_itref;
        QpSol<P, NH> sol_itref;

        /// (Re)allocate the horizon-sized storage for `n` stages (k = 0..n).
        /// Dynamic horizon: resizes the path trajectories; fixed horizon:
        /// no allocation, only a consistency check. All members are written
        /// by the solver methods before they are read.
        void resize(int n)
        {
            if constexpr (NH == Eigen::Dynamic)
            {
                assert(n >= 1);
                N = n;
                const int np = n - 1;  // path stages
                M_path.resize(np);
                L_path.resize(np);
                Lh_path.resize(np);
                Pb_path.resize(np);
                P_path.resize(np);
                q_path.resize(np);
                rtilde_path.resize(np);
                gamma_path.resize(np);
                Gamma_path.resize(np);
                t_inv_path.resize(np);
                rhs_path.resize(np);
                dv_path.resize(np);
                dlam_path.resize(np);
                dt_path.resize(np);
                dpi.resize(n);
                res.resize(n);
                res_bkp.resize(n);
                res_itref.resize(n);
                sol_itref.resize(n);
            }
            else
            {
                assert(n == NH);
                N = NH;
            }
        }
    };

    explicit HpipmQpSolver(HpipmOptions opts = HpipmOptions{}) : opts_(opts)
    {
        stat_.init(opts_.stat_max);
    }

    /// Solve the staged QP (relative-formulation IPM, worklog 1h).
    ///
    /// `in`: the QP data (HPIPM `qp`); `out`: the primal STEP over the
    /// stage-type variable vectors (u; x; s) / (x; s) plus the ABSOLUTE
    /// multipliers pi, lam, t (the QpSol mixed convention).
    ///
    /// Fast path: when no side is active (all d_mask = 0) the QP is
    /// unconstrained, so a single `solve_kkt_unconstr` yields the solution
    /// in one shot (HPIPM `OCP_QP_IPM_SOLVE` unconstr branch; iter = 0).
    /// Otherwise the predictor-corrector IPM loop runs:
    ///   init_point -> [residuals -> (exit test) -> delta step (tau-shift,
    ///   fact_solve_kkt, alpha, mu_aff, centering, solve_kkt, update_vars)]*.
    ///
    /// Exit (worklog sec. 1, `x_ocp_qp_ipm.c:3119`): converged when
    /// alpha > alpha_min and res_g / res_b / res_d / (tau-shifted) res_m
    /// are within their tolerances and dual_gap <= dual_gap_max.
    /// @return kSolved, or kMaxIterations / kMinStep / kQpFailure /
    ///         kNanDetected on failure.
    Status solve(const Qp<P, NH>& in, QpSol<P, NH>& out)
    {
        const int N = in.N;
        ws_.resize(N);
        stat_.init(opts_.stat_max);

        if (!has_active_side(in))
        {
            return solve_unconstr(in, out);
        }

        // 3e: absolute formulation (HPIPM OCP_QP_IPM_SOLVE abs branch)
        if (opts_.abs_form)
        {
            return solve_abs(in, out);
        }

        // interior-point initialization (worklog 1c) + mask absolute lam
        init_point(in, out);
        mask_abs_lam(in, out);
        ws_.alpha_prim = S(1);
        ws_.alpha_dual = S(1);

        // pre-iteration residuals (stat row 0; x_ocp_qp_ipm.c:3085-3107)
        compute_residuals(in, out, ws_.res);
        S rm_tau = res_m_tau(in, ws_.res);
        fill_stat_residuals(stat_.row(0), ws_.res);

        // 3d: per-solve latches for the reg ladder + LQ auto-switch
        ws_.force_lq = false;
        ws_.preg_last = S(opts_.reg_prim);

        QpSol<P, NH> step(N);
        for (int kk = 0; kk < opts_.iter_max; ++kk)
        {
            // exit test (x_ocp_qp_ipm.c:3119-3128): both step lengths must
            // stay above alpha_min (single value when split_step = 0).
            if (ws_.alpha_prim <= opts_.alpha_min
                    or ws_.alpha_dual <= opts_.alpha_min)
            {
                break;
            }
            const bool converged =
                (ws_.res.res_g_max <= opts_.res_g_max &&
                 ws_.res.res_b_max <= opts_.res_b_max &&
                 ws_.res.res_d_max <= opts_.res_d_max &&
                 rm_tau <= opts_.res_m_max &&
                 ws_.res.dual_gap <= opts_.dual_gap_max);
            if (converged)
            {
                break;
            }

            // --- delta step (OCP_QP_IPM_DELTA_STEP) ---
            // 1) BACKUP_RES_M + COMPUTE_TAU_MIN (predictor shift)
            ws_.res_bkp = ws_.res;
            shift_res_m(in, ws_.res, opts_.tau_min);

            // 2) predictor: factorize + solve (3d: lq_fact dispatch +
            //    reg ladder; worklog 1e)
            const Status fs = delta_factor_solve(in, out, ws_.res, step);
            if (fs != Status::kSolved)
            {
                stat_.iter = kk;
                stat_.status = Status::kQpFailure;
                return Status::kQpFailure;
            }
            mask_step(in, step);

            // 2b) predictor iterative refinement (3d)
            int itref_pred = 0;
            int itref_corr = 0;
            itref_pred = refine_step(in, out, step, opts_.itref_pred_max);

            // 3) predictor step length (stat 0, 1)
            compute_alpha(in, out, step);
            HpipmIteration* row = stat_row(kk);
            if (row)
            {
                row->alpha_prim_aff = ws_.alpha_prim;
                row->alpha_dual_aff = ws_.alpha_dual;
            }

            // 4) Mehrotra predictor-corrector (worklog 1g)
            if (opts_.pred_corr)
            {
                const S mu_aff_pred = compute_mu_aff(in, out, step);
                if (row)
                {
                    row->mu_aff = mu_aff_pred;
                }
                // corrector: centering target, reuse the stored factors
                apply_centering(in, ws_.res_bkp, step, /*correction=*/true,
                                ws_.res);
                solve_kkt(in, out, ws_.res, step);
                mask_step(in, step);
                compute_alpha(in, out, step);
                if (row)
                {
                    row->sigma = ws_.sigma;
                    row->alpha_prim = ws_.alpha_prim;
                    row->alpha_dual = ws_.alpha_dual;
                }
                // conditional pure centering (x_ocp_qp_ipm.c:2564-2605)
                if (opts_.cond_pred_corr)
                {
                    const S mu_aff_corr = compute_mu_aff(in, out, step);
                    if (mu_aff_corr > S(2) * mu_aff_pred)
                    {
                        apply_centering(in, ws_.res_bkp, step,
                                        /*correction=*/false, ws_.res);
                        solve_kkt(in, out, ws_.res, step);
                        mask_step(in, step);
                        compute_alpha(in, out, step);
                        if (row)
                        {
                            row->alpha_prim = ws_.alpha_prim;
                            row->alpha_dual = ws_.alpha_dual;
                        }
                    }
                }
                // corrector iterative refinement (3d), then re-measure the
                // accepted step on the refined step
                itref_corr = refine_step(in, out, step,
                                         opts_.itref_corr_max);
                compute_alpha(in, out, step);
                if (row)
                {
                    row->alpha_prim = ws_.alpha_prim;
                    row->alpha_dual = ws_.alpha_dual;
                }
            }
            else
            {
                if (row)
                {
                    row->alpha_prim = ws_.alpha_prim;
                    row->alpha_dual = ws_.alpha_dual;
                }
            }

            // 5) update the iterate (worklog 1f)
            update_vars(out, step);

            // 6) fresh residuals + stat row kk+1
            compute_residuals(in, out, ws_.res);
            rm_tau = res_m_tau(in, ws_.res);
            if (row)
            {
                row->mu = ws_.res.res_mu;
                fill_stat_residuals(*row, ws_.res);
                row->lq_fact = ws_.last_lq_fact;
                row->itref_pred = itref_pred;
                row->itref_corr = itref_corr;
                row->lin_res_stat = ws_.res_itref.res_g_max;
                row->lin_res_eq = ws_.res_itref.res_b_max;
                row->lin_res_ineq = ws_.res_itref.res_d_max;
                row->lin_res_comp = ws_.res_itref.res_m_max;
                row->npd_reg_hess = ws_.npd_reg_hess ? 1 : 0;
            }
            stat_.iter = kk + 1;
        }

        // status mapping (x_ocp_qp_ipm.c:3182-3209)
        if (stat_.iter == opts_.iter_max)
        {
            stat_.status = Status::kMaxIterations;
            return Status::kMaxIterations;
        }
        if (ws_.alpha_prim <= opts_.alpha_min
                or ws_.alpha_dual <= opts_.alpha_min)
        {
            stat_.status = Status::kMinStep;
            return Status::kMinStep;
        }
        if (std::isnan(ws_.res.res_mu))
        {
            stat_.status = Status::kNanDetected;
            return Status::kNanDetected;
        }
        stat_.status = Status::kSolved;
        return Status::kSolved;
    }

    /// KKT residuals of a primal-dual iterate (HPIPM `OCP_QP_RES_COMPUTE`,
    /// x_ocp_qp_res.c:345-531). Pure function of the QP data and the
    /// solution (no workspace needed); used by the IPM loop each iteration
    /// (worklog 1h) and directly by the phase-1 tests.
    ///
    /// Residual conventions (worklog sec. 1; multiplier signs verified
    /// against x_ocp_qp_res.c in the 1b checkpoint). Per row r, v_r is the
    /// row value on the (u;x) part of z; a soft lo side sees v_r + s_lo, a
    /// soft hi side v_r − s_hi:
    ///   res_g_k = H_k z_k + g_k + DC_noxᵀ(λ_hi − λ_lo)
    ///             − Σ soft sides λ · e_{slack col} − Σ_j λ_{s_j} e_{s_j}
    ///             + BA_kᵀ π_k − pin(π_{k−1})
    ///   res_b_k = BA_k z_k + b_k − x_{k+1}
    ///   res_d_i = d_i + t_i − σ_i v_{row(i)}   (masked by d_mask)
    ///   res_m_i = λ_i t_i − m_i                (masked by d_mask)
    /// where σ_i = +1 (lo / slack sides), −1 (hi sides).
    ///
    /// @param in   staged QP data (HPIPM `qp`)
    /// @param sol  primal STEP (ux_*) + ABSOLUTE duals (pi, lam, t)
    /// @param[out] res  KKT residuals + obj, dual_gap, res_mu
    void compute_residuals(const Qp<P, NH>& in, const QpSol<P, NH>& sol,
                           QpRes<P, NH>& res) const
    {
        const int N = in.N;
        assert(sol.N == N && res.N == N);

        S obj = 0.0;
        S dual_gap = 0.0;

        // first stage (k = 0)
        {
            Pi_t x1;
            if (N > 1)
            {
                x1 = sol.ux_path[0].segment(D::nu, D::nx);
            }
            else
            {
                x1 = sol.ux_term.head(D::nx);
            }
            stage_residuals(in.first.hess, in.first.grad, in.first.DC,
                            in.first.d, in.first.d_mask, in.first.m,
                            sol.ux_first, sol.lam_first, sol.t_first,
                            /*pi_prev*/ nullptr, &sol.pi[0], &x1,
                            &in.first.BA, &in.first.b, D::lay_first,
                            D::idxs_lo_first, D::idxs_hi_first,
                            res.res_g_first, res.res_d_first,
                            res.res_m_first, &res.res_b[0], obj, dual_gap);
        }

        // path stages (k = 1..N-1)
        for (int k = 1; k < N; ++k)
        {
            const int i = k - 1;
            Pi_t x_next;
            if (k + 1 < N)
            {
                x_next = sol.ux_path[k].segment(D::nu, D::nx);
            }
            else
            {
                x_next = sol.ux_term.head(D::nx);
            }
            stage_residuals(in.path[i].hess, in.path[i].grad, in.path[i].DC,
                            in.path[i].d, in.path[i].d_mask, in.path[i].m,
                            sol.ux_path[i], sol.lam_path[i], sol.t_path[i],
                            &sol.pi[k - 1], &sol.pi[k], &x_next,
                            &in.path[i].BA, &in.path[i].b, D::lay_path,
                            D::idxs_lo_path, D::idxs_hi_path,
                            res.res_g_path[i], res.res_d_path[i],
                            res.res_m_path[i], &res.res_b[k], obj, dual_gap);
        }

        // terminal stage (k = N)
        stage_residuals(in.term.hess, in.term.grad, in.term.DC,
                        in.term.d, in.term.d_mask, in.term.m,
                        sol.ux_term, sol.lam_term, sol.t_term,
                        &sol.pi[N - 1], /*pi_cur*/ nullptr,
                        /*x_next*/ nullptr, /*ba*/ nullptr, /*b*/ nullptr,
                        D::lay_term, D::idxs_lo_term, D::idxs_hi_term,
                        res.res_g_term, res.res_d_term,
                        res.res_m_term, /*res_b*/ nullptr, obj, dual_gap);

        // inf-norm maxima across all stages (HPIPM
        // OCP_QP_RES_COMPUTE_INF_NORM, x_ocp_qp_res.c:689)
        res.res_g_max = inf_norm(res.res_g_first);
        res.res_d_max = inf_norm(res.res_d_first);
        res.res_m_max = inf_norm(res.res_m_first);
        res.res_b_max = inf_norm(res.res_b[0]);
        for (int k = 1; k < N; ++k)
        {
            res.res_g_max = std::max(res.res_g_max,
                inf_norm(res.res_g_path[k - 1]));
            res.res_d_max = std::max(res.res_d_max,
                inf_norm(res.res_d_path[k - 1]));
            res.res_m_max = std::max(res.res_m_max,
                inf_norm(res.res_m_path[k - 1]));
            res.res_b_max = std::max(res.res_b_max,
                inf_norm(res.res_b[k]));
        }
        res.res_g_max = std::max(res.res_g_max,
            inf_norm(res.res_g_term));
        res.res_d_max = std::max(res.res_d_max,
            inf_norm(res.res_d_term));
        res.res_m_max = std::max(res.res_m_max,
            inf_norm(res.res_m_term));

        // res_mu = masked mean of |res_m| (HPIPM x_ocp_qp_res.c:520-527)
        res.res_mu_sum = res.res_m_first.cwiseAbs().sum();
        int nc_mask = 0;
        for (int i = 0; i < D::nside_first; ++i)
        {
            nc_mask += (in.first.d_mask(i) > 0.5) ? 1 : 0;
        }
        for (int k = 1; k < N; ++k)
        {
            res.res_mu_sum += res.res_m_path[k - 1].cwiseAbs().sum();
            for (int i = 0; i < D::nside_path; ++i)
            {
                nc_mask += (in.path[k - 1].d_mask(i) > 0.5) ? 1 : 0;
            }
        }
        res.res_mu_sum += res.res_m_term.cwiseAbs().sum();
        for (int i = 0; i < D::nside_term; ++i)
        {
            nc_mask += (in.term.d_mask(i) > 0.5) ? 1 : 0;
        }
        res.res_mu = (nc_mask > 0) ? res.res_mu_sum / nc_mask : 0.0;
        res.obj = obj;
        res.dual_gap = dual_gap;
    }

    /// Linearized KKT residual (HPIPM `OCP_QP_RES_COMPUTE_LIN`,
    /// x_ocp_qp_res.c:535-685). The step `(step.ux, step.pi, step.lam,
    /// step.t)` is treated as a "solution" of the *linearized* KKT system:
    /// res_g / res_b / res_d are the stationarity / dynamics / feasibility
    /// residuals evaluated at the step (delta variables), while res_m uses
    /// the linearized complementarity
    ///   res_m = d_mask . (m + lam_bar . delta_t + delta_lam . t_bar)
    /// where lam_bar / t_bar are the iterate's multiplier / barrier. obj /
    /// dual_gap / res_mu are left untouched (the caller ignores them).
    void compute_linear_residual(const Qp<P, NH>& in,
                                 const QpSol<P, NH>& iter,
                                 const QpSol<P, NH>& step,
                                 QpRes<P, NH>& linres) const
    {
        const int N = in.N;
        assert(step.N == N && linres.N == N);

        // first stage (k = 0)
        {
            S dummy_obj = 0.0, dummy_gap = 0.0;
            Pi_t x1;
            if (N > 1)
            {
                x1 = step.ux_path[0].segment(D::nu, D::nx);
            }
            else
            {
                x1 = step.ux_term.head(D::nx);
            }
            stage_residuals(in.first.hess, in.first.grad, in.first.DC,
                            in.first.d, in.first.d_mask, in.first.m,
                            step.ux_first, step.lam_first, step.t_first,
                            /*pi_prev*/ nullptr, &step.pi[0], &x1,
                            &in.first.BA, &in.first.b, D::lay_first,
                            D::idxs_lo_first, D::idxs_hi_first,
                            linres.res_g_first, linres.res_d_first,
                            linres.res_m_first, &linres.res_b[0], dummy_obj,
                            dummy_gap);
            linres.res_m_first = in.first.d_mask.cwiseProduct(
                in.first.m + iter.lam_first.cwiseProduct(step.t_first)
                + step.lam_first.cwiseProduct(iter.t_first));
        }

        // path stages (k = 1..N-1)
        for (int k = 1; k < N; ++k)
        {
            const int i = k - 1;
            S dummy_obj = 0.0, dummy_gap = 0.0;
            Pi_t x_next;
            if (k + 1 < N)
            {
                x_next = step.ux_path[k].segment(D::nu, D::nx);
            }
            else
            {
                x_next = step.ux_term.head(D::nx);
            }
            stage_residuals(in.path[i].hess, in.path[i].grad, in.path[i].DC,
                            in.path[i].d, in.path[i].d_mask, in.path[i].m,
                            step.ux_path[i], step.lam_path[i], step.t_path[i],
                            &step.pi[k - 1], &step.pi[k], &x_next,
                            &in.path[i].BA, &in.path[i].b, D::lay_path,
                            D::idxs_lo_path, D::idxs_hi_path,
                            linres.res_g_path[i], linres.res_d_path[i],
                            linres.res_m_path[i], &linres.res_b[k], dummy_obj,
                            dummy_gap);
            linres.res_m_path[i] = in.path[i].d_mask.cwiseProduct(
                in.path[i].m + iter.lam_path[i].cwiseProduct(step.t_path[i])
                + step.lam_path[i].cwiseProduct(iter.t_path[i]));
        }

        // terminal stage (k = N)
        {
            S dummy_obj = 0.0, dummy_gap = 0.0;
            stage_residuals(in.term.hess, in.term.grad, in.term.DC,
                            in.term.d, in.term.d_mask, in.term.m,
                            step.ux_term, step.lam_term, step.t_term,
                            &step.pi[N - 1], /*pi_cur*/ nullptr,
                            /*x_next*/ nullptr, /*ba*/ nullptr, /*b*/ nullptr,
                            D::lay_term, D::idxs_lo_term, D::idxs_hi_term,
                            linres.res_g_term, linres.res_d_term,
                            linres.res_m_term, /*res_b*/ nullptr, dummy_obj,
                            dummy_gap);
            linres.res_m_term = in.term.d_mask.cwiseProduct(
                in.term.m + iter.lam_term.cwiseProduct(step.t_term)
                + step.lam_term.cwiseProduct(iter.t_term));
        }

        // inf-norm maxima across all stages
        linres.res_g_max = inf_norm(linres.res_g_first);
        linres.res_d_max = inf_norm(linres.res_d_first);
        linres.res_m_max = inf_norm(linres.res_m_first);
        linres.res_b_max = inf_norm(linres.res_b[0]);
        for (int k = 1; k < N; ++k)
        {
            linres.res_g_max = std::max(
                linres.res_g_max, inf_norm(linres.res_g_path[k - 1]));
            linres.res_d_max = std::max(
                linres.res_d_max, inf_norm(linres.res_d_path[k - 1]));
            linres.res_m_max = std::max(
                linres.res_m_max, inf_norm(linres.res_m_path[k - 1]));
            linres.res_b_max = std::max(
                linres.res_b_max, inf_norm(linres.res_b[k]));
        }
        linres.res_g_max = std::max(linres.res_g_max,
            inf_norm(linres.res_g_term));
        linres.res_d_max = std::max(linres.res_d_max,
            inf_norm(linres.res_d_term));
        linres.res_m_max = std::max(linres.res_m_max,
            inf_norm(linres.res_m_term));
    }

    /// Interior-point initialization (HPIPM `OCP_QP_INIT_VAR`,
    /// x_ocp_qp_ipm.c:1632-2049; var_init_scheme = 1, t0_init = 2).
    ///
    /// Branches on opts_.warm_start (phase 3c):
    ///   >= 3 (hot): keep the whole iterate (ux, pi, lam, t); clip lam/t to
    ///       lam0_min / t0_min; return.
    ///   == 2 (primal+dual): keep pi / lam / t; clip lam/t to thr0 = 1e-1;
    ///       leave ux as handed in (the SQP driver zeroes it); return.
    ///   <= 1 (cold): zero the primal (u;x;s) steps and the dynamics
    ///       multipliers, then fill an interior (t, lam) pair: slacks s_j
    ///       start at thr0 with t_j = thr0; every bound side gets t_j =
    ///       distance from the bound at the current iterate (clipped to
    ///       >= thr0, repairing the decision variable of a bound row when
    ///       both sides are violated); finally lam_j = mu0 / t_j, so that
    ///       lam_j * t_j = mu0 on every active side (res_mu = mu0 at the
    ///       initial iterate, m = 0 in v1).
    ///
    /// Made public for the phase-1 tests (plan sec. 6 lists it private).
    void init_point(const Qp<P, NH>& in, QpSol<P, NH>& out)
    {
        const int N = in.N;
        assert(out.N == N);

        // Per-stage iterate mutators (used by the warm-start branches).
        auto zero_primal = [&]()
        {
            out.ux_first.setZero();
            for (int k = 1; k < N; ++k)
            {
                out.ux_path[k - 1].setZero();
            }
            out.ux_term.setZero();
        };
        auto zero_dual = [&]()
        {
            for (int k = 0; k < N; ++k)
            {
                out.pi[k].setZero();
            }
            out.lam_first.setZero();
            for (int k = 1; k < N; ++k)
            {
                out.lam_path[k - 1].setZero();
            }
            out.lam_term.setZero();
            out.t_first.setZero();
            for (int k = 1; k < N; ++k)
            {
                out.t_path[k - 1].setZero();
            }
            out.t_term.setZero();
        };
        auto clip_lam_t = [&](S lam_flo, S t_flo)
        {
            auto clip = [&](auto& lam, auto& t)
            {
                const int n = static_cast<int>(lam.size());
                for (int i = 0; i < n; ++i)
                {
                    if (lam(i) < lam_flo)
                    {
                        lam(i) = lam_flo;
                    }
                    if (t(i) < t_flo)
                    {
                        t(i) = t_flo;
                    }
                }
            };
            clip(out.lam_first, out.t_first);
            for (int k = 1; k < N; ++k)
            {
                clip(out.lam_path[k - 1], out.t_path[k - 1]);
            }
            clip(out.lam_term, out.t_term);
        };

        // Hot start (warm_start >= 3, x_ocp_qp_ipm.c:1658-1675): keep the
        // whole iterate (ux, pi, lam, t); enforce only the warm-start floors.
        if (opts_.warm_start >= 3)
        {
            clip_lam_t(opts_.lam0_min, opts_.t0_min);
            return;
        }

        // Primal + dual warm start (warm_start == 2, x_ocp_qp_ipm.c:
        // 1678-1694): keep pi / lam / t, clip to thr0 = 1e-1. The primal is
        // left as handed in; the SQP driver zeroes ux before each solve, so
        // the driven case is a cold primal + warm dual (matching acados).
        if (opts_.warm_start == 2)
        {
            const S thr0 = S(1e-1);
            clip_lam_t(thr0, thr0);
            return;
        }

        // Cold start (warm_start <= 1, x_ocp_qp_ipm.c:1697-2042): zero the
        // primal steps, pi, lam, t and build a heuristic interior (lam, t).
        zero_primal();
        zero_dual();

        const S thr0 = 1e-1;  // x_ocp_qp_ipm.c:1655
        const S mu0 = opts_.mu0;

        // first stage (k = 0): pin rows (g_pin) + state/control box rows
        {
            const auto& lay = D::lay_first;
            BoundFirstArr bound;
            int bi = 0;
            if (P::fixed_initial_state)
            {
                for (int j = 0; j < D::nx; ++j)
                {
                    bound[bi++] = {j, D::idx_x0[j]};
                }
            }
            const int nbxf = D::nbx_first;
            for (int j = 0; j < nbxf; ++j)
            {
                bound[bi++] = {lay.row_off(detail::g_bx) + j,
                               D::idxb_first[j]};
            }
            for (int j = 0; j < D::nbu; ++j)
            {
                bound[bi++] = {lay.row_off(detail::g_bu) + j,
                               D::idxb_first[nbxf + j]};
            }
            init_stage(out.ux_first, out.lam_first, out.t_first, in.first.d,
                       in.first.DC, lay, D::idxs_lo_first, D::idxs_hi_first,
                       bound, D::nu + D::nx, D::nslack_first, thr0, mu0);
        }

        // path stages (k = 1..N-1)
        {
            const auto& lay = D::lay_path;
            BoundPathArr bound;
            for (int j = 0; j < D::nbx; ++j)
            {
                bound[j] = {lay.row_off(detail::g_bx) + j, D::idxb_path[j]};
            }
            for (int j = 0; j < D::nbu; ++j)
            {
                bound[D::nbx + j] = {lay.row_off(detail::g_bu) + j,
                                     D::idxb_path[D::nbx + j]};
            }
            for (int k = 1; k < N; ++k)
            {
                const int i = k - 1;
                init_stage(out.ux_path[i], out.lam_path[i], out.t_path[i],
                           in.path[i].d, in.path[i].DC, lay, D::idxs_lo_path,
                           D::idxs_hi_path, bound, D::nu + D::nx,
                           D::nslack_path, thr0, mu0);
            }
        }

        // terminal stage (k = N): state box rows only
        {
            const auto& lay = D::lay_term;
            BoundTermArr bound;
            for (int j = 0; j < D::nbx_t; ++j)
            {
                bound[j] = {lay.row_off(detail::g_bx) + j, D::idxb_term[j]};
            }
            init_stage(out.ux_term, out.lam_term, out.t_term, in.term.d,
                       in.term.DC, lay, D::idxs_lo_term, D::idxs_hi_term,
                       bound, D::nx, D::nslack_term, thr0, mu0);
        }
    }

    // ==================================================================
    //  1e: KKT solve (two-pass Riccati; worklog 1e)
    // ==================================================================

    /// Factorize the KKT system (backward pass) and solve for the Newton
    /// step (forward pass) plus the closed-form dual / slack update
    /// (HPIPM `OCP_QP_FACT_SOLVE_KKT_STEP`, x_ocp_qp_kkt.c:1117).
    ///
    /// `in`: QP data; `iter`: the current primal-dual iterate (absolute
    /// lam / t, used for Gamma / gamma); `res`: the KKT residuals; `step`:
    /// [out] the Newton step — `step.ux_*` = δz, `step.pi` = δπ,
    /// `step.lam_*` = δλ, `step.t_*` = δt (the HPIPM `sol_step`).
    ///
    /// @return kSolved on success, kQpFailure if the Cholesky stays
    ///         singular after the reg-primal growth schedule.
    Status fact_solve_kkt(const Qp<P, NH>& in, const QpSol<P, NH>& iter,
                          const QpRes<P, NH>& res, QpSol<P, NH>& step)
    {
        return fact_solve_kkt_reg(in, iter, res, step, S(opts_.reg_prim));
    }

    /// Same as fact_solve_kkt but with an explicit reg-primal value (the
    /// reg-ladder path in the lq_fact dispatch, 3d).
    Status fact_solve_kkt_reg(const Qp<P, NH>& in, const QpSol<P, NH>& iter,
                              const QpRes<P, NH>& res, QpSol<P, NH>& step,
                              S reg)
    {
        const int N = in.N;
        ws_.resize(N);
        compute_gamma(in, iter, res);
        for (int k = N; k >= 0; --k)
        {
            if (!backward_factor_stage(k, in, reg))
            {
                return Status::kQpFailure;
            }
        }
        for (int k = N; k >= 0; --k)
        {
            backward_reduced_stage(k, in, res);
        }
        for (int k = 0; k <= N; ++k)
        {
            forward_stage(k, in, res, step);
        }
        for (int k = 0; k <= N; ++k)
        {
            closed_form_stage(k, in, res, iter, step);
        }
        return Status::kSolved;
    }

    /// Re-solve the KKT system reusing the last factorization (the L_k and
    /// P_k from fact_solve_kkt) — the Mehrotra corrector / refinement path
    /// (HPIPM `OCP_QP_SOLVE_KKT_STEP`, x_ocp_qp_kkt.c:2306). Gamma / gamma
    /// are recomputed from `res` (the centering-shifted res_m) before the
    /// forward pass, as in HPIPM's leading COMPUTE_GAMMA_QP
    /// (x_ocp_qp_kkt.c:2346).
    void solve_kkt(const Qp<P, NH>& in, const QpSol<P, NH>& iter,
                   const QpRes<P, NH>& res, QpSol<P, NH>& step)
    {
        const int N = in.N;
        compute_gamma(in, iter, res);
        for (int k = N; k >= 0; --k)
        {
            backward_reduced_stage(k, in, res);
        }
        for (int k = 0; k <= N; ++k)
        {
            forward_stage(k, in, res, step);
        }
        for (int k = 0; k <= N; ++k)
        {
            closed_form_stage(k, in, res, iter, step);
        }
    }

    /// Inequality-free fast path (HPIPM `OCP_QP_FACT_SOLVE_KKT_UNCONSTR`,
    /// x_ocp_qp_kkt.c:316): the same two-pass with Gamma = gamma = 0 (no
    /// constraint rows; `res` carries res_g / res_b only).
    Status solve_kkt_unconstr(const Qp<P, NH>& in, const QpRes<P, NH>& res,
                              QpSol<P, NH>& step)
    {
        const int N = in.N;
        ws_.gamma_first.setZero();
        for (int k = 1; k < N; ++k)
        {
            ws_.gamma_path[k - 1].setZero();
        }
        ws_.gamma_term.setZero();
        ws_.Gamma_first.setZero();
        for (int k = 1; k < N; ++k)
        {
            ws_.Gamma_path[k - 1].setZero();
        }
        ws_.Gamma_term.setZero();
        const S reg = opts_.reg_prim;
        for (int k = N; k >= 0; --k)
        {
            if (!backward_factor_stage(k, in, reg))
            {
                return Status::kQpFailure;
            }
        }
        for (int k = N; k >= 0; --k)
        {
            backward_reduced_stage(k, in, res);
        }
        for (int k = 0; k <= N; ++k)
        {
            forward_stage(k, in, res, step);
        }
        // no inequality sides: δt = δλ = 0
        step.lam_first.setZero();
        for (int k = 1; k < N; ++k)
        {
            step.lam_path[k - 1].setZero();
        }
        step.lam_term.setZero();
        step.t_first.setZero();
        for (int k = 1; k < N; ++k)
        {
            step.t_path[k - 1].setZero();
        }
        step.t_term.setZero();
        return Status::kSolved;
    }

    /// LQ KKT factorization + solve (3d): the same two-pass Riccati as
    /// fact_solve_kkt, but the backward factorization uses the wide-matrix
    /// LQ (factor_one_stage_lq) instead of the dense Cholesky.  The forward,
    /// backward-reduced, and closed-form stages are shared.  A singular
    /// Hessian block (any stage's Lh) returns kQpFailure (the reg-primal
    /// growth ladder is applied by the caller in the lq_fact dispatch).
    Status fact_solve_kkt_lq(const Qp<P, NH>& in, const QpSol<P, NH>& iter,
                             const QpRes<P, NH>& res, QpSol<P, NH>& step)
    {
        return fact_solve_kkt_lq_reg(in, iter, res, step, S(opts_.reg_prim));
    }

    /// Same as fact_solve_kkt_lq but with an explicit reg-primal value (the
    /// reg-ladder path in the lq_fact dispatch, 3d).
    Status fact_solve_kkt_lq_reg(const Qp<P, NH>& in,
                                 const QpSol<P, NH>& iter,
                                 const QpRes<P, NH>& res, QpSol<P, NH>& step,
                                 S reg)
    {
        const int N = in.N;
        compute_gamma(in, iter, res);
        const bool lh_valid = (ws_.lh_reg == reg);
        for (int k = N; k >= 0; --k)
        {
            if (!backward_factor_stage_lq(k, in, reg, lh_valid))
            {
                return Status::kQpFailure;
            }
        }
        ws_.lh_reg = reg;
        for (int k = N; k >= 0; --k)
        {
            backward_reduced_stage(k, in, res);
        }
        for (int k = 0; k <= N; ++k)
        {
            forward_stage(k, in, res, step);
        }
        for (int k = 0; k <= N; ++k)
        {
            closed_form_stage(k, in, res, iter, step);
        }
        return Status::kSolved;
    }

    // ==================================================================
    //  3d: LQ fallback + iterative refinement (SQP_PHASE3_PLAN sec. 3d)
    // ==================================================================

    /// reg-primal growth ladder (HPIPM `compute_preg`,
    /// x_ocp_qp_kkt.c:1166-1210): run `lq ? fact_solve_kkt_lq_reg :
    /// fact_solve_kkt_reg` and, on a singular factorization, grow the
    /// reg-primal (`preg`) and retry.  Growth schedule (first growth is
    /// additive, subsequent are multiplicative; `ws_.preg_last` holds the
    /// last successfully-used reg and persists across IPM iterations):
    ///   first growth:  +1e-4 if preg_last == base, else /3
    ///   next  growths: x100 if preg_last == base, else x8
    /// Sets `ws_.npd_reg_hess` on any failure and persists the reg used in
    /// `ws_.preg_last` on success.
    Status fact_solve_ladder(const Qp<P, NH>& in, const QpSol<P, NH>& iter,
                             const QpRes<P, NH>& res, QpSol<P, NH>& step,
                             bool lq)
    {
        const S base = S(opts_.reg_prim);
        S preg = ws_.preg_last;
        bool preg_first = true;
        ws_.npd_reg_hess = false;
        for (int attempt = 0; attempt < 15; ++attempt)
        {
            const Status st = lq
                ? fact_solve_kkt_lq_reg(in, iter, res, step, preg)
                : fact_solve_kkt_reg(in, iter, res, step, preg);
            if (st == Status::kSolved)
            {
                ws_.preg_last = preg;
                return Status::kSolved;
            }
            ws_.npd_reg_hess = true;
            if (preg_first)
            {
                preg = (ws_.preg_last == base) ? preg + S(1e-4) : preg / S(3.0);
            }
            else
            {
                preg = (ws_.preg_last == base) ? preg * S(100.0) : preg * S(8.0);
            }
            preg_first = false;
        }
        return Status::kQpFailure;
    }

    /// Delta-step KKT factor + solve with the `lq_fact` mode dispatch
    /// (HPIPM x_ocp_qp_ipm.c:2274-2369):
    ///   0 -> Cholesky (fact_solve_ladder, lq = false);
    ///   1 -> Cholesky; on a factorization failure (non-PSD / singular) set
    ///        `ws_.force_lq` and re-factor with LQ; the latch persists for
    ///        the rest of the solve (HPIPM `force_lq`, x_ocp_qp_ipm.c:2259);
    ///   2 -> LQ (fact_solve_ladder, lq = true) always.
    /// Sets `ws_.last_lq_fact` (stat col 13) to whether LQ was used.
    Status delta_factor_solve(const Qp<P, NH>& in, const QpSol<P, NH>& iter,
                              const QpRes<P, NH>& res, QpSol<P, NH>& step)
    {
        ws_.last_lq_fact = 0;
        const bool use_lq = (opts_.lq_fact == 2) || ws_.force_lq;
        Status st;
        if (!use_lq && opts_.lq_fact == 1)
        {
            st = fact_solve_ladder(in, iter, res, step, /*lq=*/false);
            if (st != Status::kSolved)
            {
                ws_.force_lq = true;
                st = fact_solve_ladder(in, iter, res, step, /*lq=*/true);
            }
        }
        else
        {
            st = fact_solve_ladder(in, iter, res, step, use_lq);
        }
        ws_.last_lq_fact = use_lq || ws_.force_lq ? 1 : 0;
        return st;
    }

    /// Re-solve the KKT system reusing the stored L/P factors and gamma
    /// (from the most recent fact_solve_kkt / fact_solve_kkt_lq), with the
    /// linearized residual `rhs` as the forcing term.  This is the
    /// iterative-refinement correction solve (HPIPM `OCP_QP_SOLVE_KKT_STEP`
    /// with `use_Pb = 0`, x_ocp_qp_ipm.c:2424-2425).  Note: the δπ update
    /// uses the stored P matrix (the P-based forward pass) rather than
    /// HPIPM's L-factor form; equivalent to working precision.
    void solve_kkt_reuse_gamma(const Qp<P, NH>& in, const QpSol<P, NH>& iter,
                               const QpRes<P, NH>& rhs, QpSol<P, NH>& step)
    {
        const int N = in.N;
        for (int k = N; k >= 0; --k)
        {
            backward_reduced_stage(k, in, rhs);
        }
        for (int k = 0; k <= N; ++k)
        {
            forward_stage(k, in, rhs, step);
        }
        for (int k = 0; k <= N; ++k)
        {
            closed_form_stage(k, in, rhs, iter, step);
        }
    }

    /// Add (Add = true) or subtract (Add = false) the step `b` into the
    /// step `a`, in place (used to apply / undo an iterative-refinement
    /// correction).
    template <bool Add>
    void combine_sol(QpSol<P, NH>& a, const QpSol<P, NH>& b)
    {
        const int N = a.N;
        auto op = [](auto& x, const auto& y)
        {
            if constexpr (Add)
            {
                x = x + y;
            }
            else
            {
                x = x - y;
            }
        };
        op(a.ux_first, b.ux_first);
        for (int k = 1; k < N; ++k)
        {
            op(a.ux_path[k - 1], b.ux_path[k - 1]);
        }
        op(a.ux_term, b.ux_term);
        for (int k = 0; k < N; ++k)
        {
            op(a.pi[k], b.pi[k]);
        }
        op(a.lam_first, b.lam_first);
        for (int k = 1; k < N; ++k)
        {
            op(a.lam_path[k - 1], b.lam_path[k - 1]);
        }
        op(a.lam_term, b.lam_term);
        op(a.t_first, b.t_first);
        for (int k = 1; k < N; ++k)
        {
            op(a.t_path[k - 1], b.t_path[k - 1]);
        }
        op(a.t_term, b.t_term);
    }

    /// One iterative-refinement pass (HPIPM x_ocp_qp_ipm.c:2373-2505
    /// predictor, 2609-2732 corrector).  Computes the linearized KKT
    /// residual of the step; while it is not small enough, solves the
    /// KKT system with the residual as RHS (reusing the stored factors),
    /// adds the correction to the step, and recomputes; undoes the
    /// correction and stops if the residual did not shrink.  `max_iter` is
    /// `itref_pred_max` (predictor) or `itref_corr_max` (corrector); when
    /// it is 0 the pass is a no-op.  Returns the number of corrections
    /// applied (0 if `max_iter <= 0`).  The final linearized residual is
    /// left in `ws_.res_itref`.
    int refine_step(const Qp<P, NH>& in, const QpSol<P, NH>& iter,
                    QpSol<P, NH>& step, int max_iter)
    {
        if (max_iter <= 0)
        {
            return 0;
        }
        (void)in;
        compute_linear_residual(in, iter, step, ws_.res_itref);
        S norm[4] = { ws_.res_itref.res_g_max, ws_.res_itref.res_b_max,
                      ws_.res_itref.res_d_max, ws_.res_itref.res_m_max };
        S maxnorm = norm[0];
        for (int i = 1; i < 4; ++i)
        {
            maxnorm = std::max(maxnorm, norm[i]);
        }
        int nref = 0;
        for (int it = 0; it < max_iter; ++it)
        {
            const bool good =
                (norm[0] < S(opts_.res_g_max) ||
                 norm[0] < S(1e-3) * ws_.res.res_g_max) &&
                (norm[1] < S(opts_.res_b_max) ||
                 norm[1] < S(1e-3) * ws_.res.res_b_max) &&
                (norm[2] < S(opts_.res_d_max) ||
                 norm[2] < S(1e-3) * ws_.res.res_d_max) &&
                (norm[3] < S(opts_.res_m_max) ||
                 norm[3] < S(1e-3) * ws_.res.res_m_max);
            if (good)
            {
                break;
            }
            solve_kkt_reuse_gamma(in, iter, ws_.res_itref, ws_.sol_itref);
            mask_step(in, ws_.sol_itref);
            combine_sol<true>(step, ws_.sol_itref);
            compute_linear_residual(in, iter, step, ws_.res_itref);
            S newnorm[4] = { ws_.res_itref.res_g_max,
                             ws_.res_itref.res_b_max,
                             ws_.res_itref.res_d_max,
                             ws_.res_itref.res_m_max };
            S newmax = newnorm[0];
            for (int i = 1; i < 4; ++i)
            {
                newmax = std::max(newmax, newnorm[i]);
            }
            ++nref;
            if (newmax >= maxnorm)
            {
                combine_sol<false>(step, ws_.sol_itref);
                compute_linear_residual(in, iter, step, ws_.res_itref);
                break;
            }
            norm[0] = newnorm[0];
            norm[1] = newnorm[1];
            norm[2] = newnorm[2];
            norm[3] = newnorm[3];
            maxnorm = newmax;
        }
        return nref;
    }

    // ==================================================================
    //  1f: step length + iterate update (worklog 1f)
    // ==================================================================

    /// Mask the step's lam and t by d_mask (HPIPM masks sol_step after
    /// the KKT solve, before alpha computation;
    /// x_ocp_qp_ipm.c:2284-2285).
    void mask_step(const Qp<P, NH>& in, QpSol<P, NH>& step) const
    {
        step.lam_first = step.lam_first.cwiseProduct(in.first.d_mask);
        for (int k = 1; k < in.N; ++k)
        {
            step.lam_path[k - 1] =
                step.lam_path[k - 1].cwiseProduct(in.path[k - 1].d_mask);
        }
        step.lam_term = step.lam_term.cwiseProduct(in.term.d_mask);
        step.t_first = step.t_first.cwiseProduct(in.first.d_mask);
        for (int k = 1; k < in.N; ++k)
        {
            step.t_path[k - 1] =
                step.t_path[k - 1].cwiseProduct(in.path[k - 1].d_mask);
        }
        step.t_term = step.t_term.cwiseProduct(in.term.d_mask);
    }

    /// Maximum step length keeping λ ≥ 0, t ≥ 0, and (when m ≠ 0)
    /// λ·t ≥ m_safe·m (HPIPM COMPUTE_ALPHA_QP, x_core_qp_ipm_aux.c:193).
    ///
    /// split_step = 0: a single α drives λ and t (α_prim = α_dual = α).
    /// split_step = 1: separate α_prim (drives t, the primal side) and
    /// α_dual (drives λ, the dual side), computed by a three-pass line
    /// search (x_core_qp_ipm_aux.c:219-374):
    ///   pass 1: feasibility + one-sided complementarity (the dlam < 0 ∧
    ///           dt < 0 case is deferred),
    ///   pass 2: quadratic complementarity correction for the remaining
    ///           dlam < 0 ∧ dt < 0 sides (r = (−b − √disc)/(2a) scaling of
    ///           both alphas; 0 if the root is non-positive or c ≤ 0),
    ///   pass 3: re-apply the complementarity adjustment any pass-2
    ///           scaling may have left violated (no feasibility check in
    ///           the source).
    ///
    /// Call `mask_step` before this so that dlam / dt are zero on absent
    /// sides.  Stores the result in ws_.alpha_prim / ws_.alpha_dual.
    void compute_alpha(const Qp<P, NH>& in, const QpSol<P, NH>& iter,
                       const QpSol<P, NH>& step)
    {
        // m_zero: true when all m entries are zero
        // (x_ocp_qp_ipm.c:2966-2973)
        S m_norm = inf_norm(in.first.m);
        for (int k = 1; k < in.N; ++k)
        {
            m_norm = std::max(m_norm, inf_norm(in.path[k - 1].m));
        }
        m_norm = std::max(m_norm, inf_norm(in.term.m));
        const bool m_zero = (m_norm == 0.0);

        S m_safe = std::min(std::max(S(opts_.m_safe), S(0)), S(1));

        if (opts_.split_step == 1)
        {
            S a_p = S(1);
            S a_d = S(1);
            if (m_zero)
            {
                // Separate feasibility pass
                // (x_core_qp_ipm_aux.c:222-239).
                auto feas =
                    [&](const auto& lam, const auto& t, const auto& dlam,
                        const auto& dt)
                {
                    const int n = static_cast<int>(lam.size());
                    for (int i = 0; i < n; ++i)
                    {
                        if (lam(i) + a_d * dlam(i) < S(0))
                        {
                            a_d = -lam(i) / dlam(i);
                        }
                        if (t(i) + a_p * dt(i) < S(0))
                        {
                            a_p = -t(i) / dt(i);
                        }
                    }
                };
                feas(iter.lam_first, iter.t_first, step.lam_first,
                     step.t_first);
                for (int k = 1; k < in.N; ++k)
                {
                    feas(iter.lam_path[k - 1], iter.t_path[k - 1],
                         step.lam_path[k - 1], step.t_path[k - 1]);
                }
                feas(iter.lam_term, iter.t_term, step.lam_term,
                     step.t_term);
            }
            else
            {
                // Pass 1 (x_core_qp_ipm_aux.c:244-287).
                auto pass1 =
                    [&](const auto& lam, const auto& t, const auto& dlam,
                        const auto& dt, const auto& m, const auto& dmask)
                {
                    const int n = static_cast<int>(lam.size());
                    for (int i = 0; i < n; ++i)
                    {
                        S lam1 = lam(i) + a_d * dlam(i);
                        S t1 = t(i) + a_p * dt(i);
                        if (lam1 < S(0))
                        {
                            a_d = -lam(i) / dlam(i);
                            lam1 = lam(i) + a_d * dlam(i);
                        }
                        if (t1 < S(0))
                        {
                            a_p = -t(i) / dt(i);
                            t1 = t(i) + a_p * dt(i);
                        }
                        const S m1 = m_safe * (m(i) * dmask(i));
                        if (lam1 * t1 - m1 < S(-1e-12))
                        {
                            if (dlam(i) < S(0))
                            {
                                if (dt(i) >= S(0))
                                {
                                    a_d = (m1 - lam(i) * t1) / (dlam(i) * t1);
                                }
                                // dt < 0: deferred to pass 2.
                            }
                            else if (dt(i) < S(0))
                            {
                                a_p = (m1 - t(i) * lam1) / (dt(i) * lam1);
                            }
                        }
                    }
                };
                pass1(iter.lam_first, iter.t_first, step.lam_first,
                      step.t_first, in.first.m, in.first.d_mask);
                for (int k = 1; k < in.N; ++k)
                {
                    pass1(iter.lam_path[k - 1], iter.t_path[k - 1],
                          step.lam_path[k - 1], step.t_path[k - 1],
                          in.path[k - 1].m, in.path[k - 1].d_mask);
                }
                pass1(iter.lam_term, iter.t_term, step.lam_term,
                      step.t_term, in.term.m, in.term.d_mask);

                // Pass 2 (x_core_qp_ipm_aux.c:292-340).
                auto pass2 =
                    [&](const auto& lam, const auto& t, const auto& dlam,
                        const auto& dt, const auto& m, const auto& dmask)
                {
                    const int n = static_cast<int>(lam.size());
                    for (int i = 0; i < n; ++i)
                    {
                        if (!(dlam(i) < S(0) and dt(i) < S(0)))
                        {
                            continue;
                        }
                        S lam1 = lam(i) + a_d * dlam(i);
                        S t1 = t(i) + a_p * dt(i);
                        const S m1 = m_safe * (m(i) * dmask(i));
                        if (lam1 * t1 - m1 < S(-1e-12))
                        {
                            const S c = lam(i) * t(i) - m1;
                            if (c > S(0))
                            {
                                const S adlam = a_d * dlam(i);
                                const S adt = a_p * dt(i);
                                const S aa = adlam * adt;
                                const S b = adlam * t(i) + lam(i) * adt;
                                const S disc = b * b - S(4) * aa * c;
                                if (disc >= S(0))
                                {
                                    const S r =
                                        (-b - std::sqrt(disc)) *
                                        (S(0.5) / aa);
                                    if (r > S(0))
                                    {
                                        a_d *= r;
                                        a_p *= r;
                                    }
                                    else
                                    {
                                        a_d = S(0);
                                        a_p = S(0);
                                    }
                                }
                                else
                                {
                                    a_d = S(0);
                                    a_p = S(0);
                                }
                            }
                            else
                            {
                                a_d = S(0);
                                a_p = S(0);
                            }
                        }
                    }
                };
                pass2(iter.lam_first, iter.t_first, step.lam_first,
                      step.t_first, in.first.m, in.first.d_mask);
                for (int k = 1; k < in.N; ++k)
                {
                    pass2(iter.lam_path[k - 1], iter.t_path[k - 1],
                          step.lam_path[k - 1], step.t_path[k - 1],
                          in.path[k - 1].m, in.path[k - 1].d_mask);
                }
                pass2(iter.lam_term, iter.t_term, step.lam_term,
                      step.t_term, in.term.m, in.term.d_mask);

                // Pass 3 (x_core_qp_ipm_aux.c:343-369).
                auto pass3 =
                    [&](const auto& lam, const auto& t, const auto& dlam,
                        const auto& dt, const auto& m, const auto& dmask)
                {
                    const int n = static_cast<int>(lam.size());
                    for (int i = 0; i < n; ++i)
                    {
                        const S lam1 = lam(i) + a_d * dlam(i);
                        const S t1 = t(i) + a_p * dt(i);
                        const S m1 = m_safe * (m(i) * dmask(i));
                        if (lam1 * t1 - m1 < S(-1e-12))
                        {
                            if (dlam(i) < S(0))
                            {
                                if (dt(i) >= S(0))
                                {
                                    a_d = (m1 - lam(i) * t1) / (dlam(i) * t1);
                                }
                            }
                            else if (dt(i) < S(0))
                            {
                                a_p = (m1 - t(i) * lam1) / (dt(i) * lam1);
                            }
                        }
                    }
                };
                pass3(iter.lam_first, iter.t_first, step.lam_first,
                      step.t_first, in.first.m, in.first.d_mask);
                for (int k = 1; k < in.N; ++k)
                {
                    pass3(iter.lam_path[k - 1], iter.t_path[k - 1],
                          step.lam_path[k - 1], step.t_path[k - 1],
                          in.path[k - 1].m, in.path[k - 1].d_mask);
                }
                pass3(iter.lam_term, iter.t_term, step.lam_term,
                      step.t_term, in.term.m, in.term.d_mask);
            }
            ws_.alpha_prim = a_p;
            ws_.alpha_dual = a_d;
            return;
        }

        // split_step = 0: a single α drives both λ and t.
        S alpha = S(1);
        auto process =
            [&](const auto& lam, const auto& t, const auto& dlam,
                const auto& dt, const auto& m, const auto& dmask)
        {
            const int n = static_cast<int>(lam.size());
            for (int i = 0; i < n; ++i)
            {
                S lam1 = lam(i) + alpha * dlam(i);
                S t1 = t(i) + alpha * dt(i);
                if (lam1 < S(0))
                {
                    alpha = -lam(i) / dlam(i);
                    lam1 = lam(i) + alpha * dlam(i);
                }
                if (t1 < S(0))
                {
                    alpha = -t(i) / dt(i);
                    t1 = t(i) + alpha * dt(i);
                }
                if (!m_zero)
                {
                    const S m1 = m_safe * (m(i) * dmask(i));
                    if (lam1 * t1 - m1 < S(-1e-12))
                    {
                        const S c = lam(i) * t(i) - m1;
                        if (c > S(0))
                        {
                            const S a = dlam(i) * dt(i);
                            const S b = dlam(i) * t(i) + lam(i) * dt(i);
                            const S d = b * b - S(4) * a * c;
                            const S sd = std::sqrt(d);
                            alpha = (-b - sd) * (S(0.5) / a);
                        }
                        else
                        {
                            alpha = S(0);
                        }
                    }
                }
            }
        };

        process(iter.lam_first, iter.t_first,
                step.lam_first, step.t_first,
                in.first.m, in.first.d_mask);
        for (int k = 1; k < in.N; ++k)
        {
            process(iter.lam_path[k - 1], iter.t_path[k - 1],
                    step.lam_path[k - 1], step.t_path[k - 1],
                    in.path[k - 1].m, in.path[k - 1].d_mask);
        }
        process(iter.lam_term, iter.t_term,
                step.lam_term, step.t_term,
                in.term.m, in.term.d_mask);

        ws_.alpha_prim = alpha;
        ws_.alpha_dual = alpha;
    }

    /// Accepted step lengths from the most recent `compute_alpha` call
    /// (equal when split_step = 0).
    S alpha_prim() const { return ws_.alpha_prim; }
    S alpha_dual() const { return ws_.alpha_dual; }

    /// Apply the Newton step to the iterate (HPIPM UPDATE_VAR_QP,
    /// x_core_qp_ipm_aux.c:472; t_lam_min = 2).
    ///
    /// Damping: when min(α_p, α_d) < 1, each is damped independently
    /// via α' = α·(0.99(1−α) + 0.9999999·α).
    /// Applied: ux += α_p·δz, π += α_d·δπ,
    /// λ += α_d·δλ (clip ≥ lam_min), t += α_p·δt (clip ≥ t_min).
    /// With split_step = 0, α_p = α_d so behaviour is identical.
    void update_vars(QpSol<P, NH>& iter, const QpSol<P, NH>& step) const
    {
        const int N = iter.N;
        S alpha_p = ws_.alpha_prim;
        S alpha_d = ws_.alpha_dual;
        if (std::min(alpha_p, alpha_d) < S(1))
        {
            alpha_p = alpha_p * (S(0.99) * (S(1) - alpha_p)
                                 + S(0.9999999) * alpha_p);
            alpha_d = alpha_d * (S(0.99) * (S(1) - alpha_d)
                                 + S(0.9999999) * alpha_d);
        }

        iter.ux_first += alpha_p * step.ux_first;
        for (int k = 1; k < N; ++k)
        {
            iter.ux_path[k - 1] += alpha_p * step.ux_path[k - 1];
        }
        iter.ux_term += alpha_p * step.ux_term;

        for (int k = 0; k < N; ++k)
        {
            iter.pi[k] += alpha_d * step.pi[k];
        }

        update_lam_t(iter.lam_first, iter.t_first,
                     step.lam_first, step.t_first, alpha_d, alpha_p);
        for (int k = 1; k < N; ++k)
        {
            update_lam_t(iter.lam_path[k - 1], iter.t_path[k - 1],
                         step.lam_path[k - 1], step.t_path[k - 1],
                         alpha_d, alpha_p);
        }
        update_lam_t(iter.lam_term, iter.t_term,
                     step.lam_term, step.t_term, alpha_d, alpha_p);
    }

    // ==================================================================
    //  1g: Mehrotra predictor-corrector (worklog 1g)
    // ==================================================================

    /// Barrier parameter of the affine (predictor) iterate (HPIPM
    /// COMPUTE_MU_AFF_QP, x_core_qp_ipm_aux.c:636):
    ///   mu_aff = (1 / nc_mask) * sum_sides | -m_i + (lam_i + a_d dlam_i)
    ///                                       * (t_i + a_p dt_i) |
    /// over the active sides (d_mask = 1); a_p / a_d from
    /// ws_.alpha_prim / ws_.alpha_dual.
    /// `m` is the QP's complementarity RHS (in.*, zero in v1), NOT the
    /// workspace res_m. Must be called after compute_alpha so that
    /// ws_.alpha_prim / ws_.alpha_dual hold the accepted step lengths.
    /// Stores the result in ws_.mu_aff and returns it.
    S compute_mu_aff(const Qp<P, NH>& in, const QpSol<P, NH>& iter,
                     const QpSol<P, NH>& step)
    {
        const S a_d = ws_.alpha_dual;  // drives λ
        const S a_p = ws_.alpha_prim;  // drives t
        S sum = 0.0;
        int nc_mask = 0;
        auto add_stage = [&](const auto& m, const auto& lam, const auto& t,
                             const auto& dlam, const auto& dt,
                             const auto& dmask)
        {
            const int n = static_cast<int>(lam.size());
            for (int i = 0; i < n; ++i)
            {
                if (dmask(i) > 0.5)
                {
                    const S lam1 = lam(i) + a_d * dlam(i);
                    const S t1 = t(i) + a_p * dt(i);
                    sum += std::fabs(-m(i) + lam1 * t1);
                    ++nc_mask;
                }
            }
        };
        add_stage(in.first.m, iter.lam_first, iter.t_first,
                  step.lam_first, step.t_first, in.first.d_mask);
        for (int k = 1; k < in.N; ++k)
        {
            add_stage(in.path[k - 1].m, iter.lam_path[k - 1],
                      iter.t_path[k - 1], step.lam_path[k - 1],
                      step.t_path[k - 1], in.path[k - 1].d_mask);
        }
        add_stage(in.term.m, iter.lam_term, iter.t_term, step.lam_term,
                  step.t_term, in.term.d_mask);
        const S mu_aff = (nc_mask > 0) ? sum / nc_mask : 0.0;
        ws_.mu_aff = mu_aff;
        return mu_aff;
    }

    /// Set the complementarity target res_m for the corrector (or pure-
    /// centering) KKT solve (HPIPM COMPUTE_CENTERING_CORRECTION_QP /
    /// COMPUTE_CENTERING_QP, x_core_qp_ipm_aux.c:695/729):
    ///   sigma = (mu_aff / mu)^3,  sigma_mu = max(sigma * mu, tau_min)
    ///   correction: res_m = d_mask ∘ (res_m_bkp + dt ∘ dlam - sigma_mu)
    ///   centering:  res_m = d_mask ∘ (res_m_bkp - sigma_mu)
    /// where mu = res_bkp.res_mu (the barrier parameter of the current
    /// iterate) and res_m_bkp is the UNshifted complementarity residual
    /// (before the tau_min predictor shift). Must be called after
    /// compute_mu_aff (reads ws_.mu_aff). Writes only the res_m_* fields of
    /// res_out; res_out.res_g / res_b / res_d must already hold the (fixed)
    /// stationarity / dynamics / feasibility residuals of the current
    /// iterate. Stores the (unfloored) sigma in ws_.sigma.
    void apply_centering(const Qp<P, NH>& in, const QpRes<P, NH>& res_bkp,
                         const QpSol<P, NH>& step, bool correction,
                         QpRes<P, NH>& res_out)
    {
        const S mu = res_bkp.res_mu;
        const S ratio = ws_.mu_aff / mu;
        const S sigma = ratio * ratio * ratio;
        ws_.sigma = sigma;
        S sigma_mu = sigma * mu;
        if (sigma_mu < opts_.tau_min)
        {
            sigma_mu = opts_.tau_min;
        }
        auto center = [&](const auto& bkp, const auto& dlam, const auto& dt,
                          const auto& dmask, auto& out)
        {
            const int n = static_cast<int>(out.size());
            for (int i = 0; i < n; ++i)
            {
                S v = bkp(i) - sigma_mu;
                if (correction)
                {
                    v += dt(i) * dlam(i);
                }
                out(i) = (dmask(i) > 0.5) ? v : 0.0;
            }
        };
        center(res_bkp.res_m_first, step.lam_first, step.t_first,
               in.first.d_mask, res_out.res_m_first);
        for (int k = 1; k < in.N; ++k)
        {
            center(res_bkp.res_m_path[k - 1], step.lam_path[k - 1],
                   step.t_path[k - 1], in.path[k - 1].d_mask,
                   res_out.res_m_path[k - 1]);
        }
        center(res_bkp.res_m_term, step.lam_term, step.t_term,
               in.term.d_mask, res_out.res_m_term);
    }

    const HpipmOptions& options() const { return opts_; }
    const HpipmStatistics& statistics() const { return stat_; }

    /// Override the warm-start mode for subsequent solve() calls (phase 3c;
    /// the SQP driver sets this per QP solve). 0: cold, 2: primal+dual, 3:
    /// hot (acados mapping).
    void set_warm_start(int ws) { opts_.warm_start = ws; }

private:
    HpipmOptions opts_;
    HpipmStatistics stat_;
    Workspace ws_;

    /// Infinity norm of a vector (0 for an empty vector).
    template <class V>
    static S inf_norm(const V& v)
    {
        return v.size() > 0 ? v.cwiseAbs().maxCoeff() : S(0);
    }

    /// Update one stage's λ / t with the (already damped) step, applying the
    /// t_lam_min = 2 floors (x_core_qp_ipm_aux.c:545-578). `dlam` / `dt` are
    /// assumed already masked by d_mask (see mask_step).
    template <class V>
    void update_lam_t(V& lam, V& t, const V& dlam, const V& dt, S alpha_d,
                      S alpha_p) const
    {
        lam += alpha_d * dlam;
        t += alpha_p * dt;
        if (opts_.t_lam_min == 2)
        {
            for (int i = 0; i < static_cast<int>(lam.size()); ++i)
            {
                if (lam(i) <= S(opts_.lam_min))
                {
                    lam(i) = S(opts_.lam_min);
                }
                if (t(i) <= S(opts_.t_min))
                {
                    t(i) = S(opts_.t_min);
                }
            }
        }
    }

    /// True when at least one side has d_mask > 0.5 (any active
    /// constraint). Drives the fast-path branch in solve().
    bool has_active_side(const Qp<P, NH>& in) const
    {
        auto any = [](const auto& dm)
        {
            for (int i = 0; i < static_cast<int>(dm.size()); ++i)
            {
                if (dm(i) > 0.5)
                {
                    return true;
                }
            }
            return false;
        };
        if (any(in.first.d_mask))
        {
            return true;
        }
        for (int k = 1; k < in.N; ++k)
        {
            if (any(in.path[k - 1].d_mask))
            {
                return true;
            }
        }
        return any(in.term.d_mask);
    }

    /// ||res_m − tau_min ∘ d_mask||_∞ across all stages (HPIPM's
    /// res_m_tau exit-test quantity, x_ocp_qp_ipm.c:3114-3116).
    S res_m_tau(const Qp<P, NH>& in, const QpRes<P, NH>& res) const
    {
        S m = 0;
        auto acc = [&](const auto& rm, const auto& dm)
        {
            for (int i = 0; i < static_cast<int>(rm.size()); ++i)
            {
                m = std::max(m, std::fabs(rm(i) - opts_.tau_min * dm(i)));
            }
        };
        acc(res.res_m_first, in.first.d_mask);
        for (int k = 1; k < in.N; ++k)
        {
            acc(res.res_m_path[k - 1], in.path[k - 1].d_mask);
        }
        acc(res.res_m_term, in.term.d_mask);
        return m;
    }

    /// Mask the absolute multipliers by d_mask (HPIPM masks qp_sol->lam
    /// after init; t is kept for all sides to stay interior).
    void mask_abs_lam(const Qp<P, NH>& in, QpSol<P, NH>& sol) const
    {
        sol.lam_first = sol.lam_first.cwiseProduct(in.first.d_mask);
        for (int k = 1; k < in.N; ++k)
        {
            sol.lam_path[k - 1] =
                sol.lam_path[k - 1].cwiseProduct(in.path[k - 1].d_mask);
        }
        sol.lam_term = sol.lam_term.cwiseProduct(in.term.d_mask);
    }

    /// res_m -= tau * d_mask on every stage (COMPUTE_TAU_MIN_QP,
    /// x_core_qp_ipm_aux.c:756-778).
    void shift_res_m(const Qp<P, NH>& in, QpRes<P, NH>& res, S tau) const
    {
        res.res_m_first.array() -= tau * in.first.d_mask.array();
        for (int k = 1; k < in.N; ++k)
        {
            res.res_m_path[k - 1].array() -=
                tau * in.path[k - 1].d_mask.array();
        }
        res.res_m_term.array() -= tau * in.term.d_mask.array();
    }

    /// Fill the residual / objective fields of a stat row (HPIPM cols 7-12).
    static void fill_stat_residuals(HpipmIteration& row,
                                    const QpRes<P, NH>& res)
    {
        row.res_stat = res.res_g_max;
        row.res_eq = res.res_b_max;
        row.res_ineq = res.res_d_max;
        row.res_comp = res.res_m_max;
        row.dual_gap = res.dual_gap;
        row.obj = res.obj;
    }

    /// Pointer to the stat row for iteration kk (row index kk+1), or null
    /// when the row is beyond the stat_max storage.
    HpipmIteration* stat_row(int kk)
    {
        return (kk + 1 < stat_.stat_max) ? &stat_.row(kk + 1) : nullptr;
    }

    /// Unconstrained fast path (HPIPM OCP_QP_IPM_SOLVE unconstr branch,
    /// x_ocp_qp_ipm.c:2908-2950). Zero the iterate, solve the KKT system
    /// once, return iter = 0.
    Status solve_unconstr(const Qp<P, NH>& in, QpSol<P, NH>& out)
    {
        const int N = in.N;
        out.ux_first.setZero();
        for (int k = 1; k < N; ++k)
        {
            out.ux_path[k - 1].setZero();
        }
        out.ux_term.setZero();
        for (int k = 0; k < N; ++k)
        {
            out.pi[k].setZero();
        }
        out.lam_first.setZero();
        for (int k = 1; k < N; ++k)
        {
            out.lam_path[k - 1].setZero();
        }
        out.lam_term.setZero();
        out.t_first.setZero();
        for (int k = 1; k < N; ++k)
        {
            out.t_path[k - 1].setZero();
        }
        out.t_term.setZero();

        compute_residuals(in, out, ws_.res);
        QpSol<P, NH> step(N);
        const Status st = solve_kkt_unconstr(in, ws_.res, step);
        if (st != Status::kSolved)
        {
            stat_.iter = 0;
            stat_.status = Status::kQpFailure;
            return Status::kQpFailure;
        }
        out.ux_first = step.ux_first;
        for (int k = 1; k < N; ++k)
        {
            out.ux_path[k - 1] = step.ux_path[k - 1];
        }
        out.ux_term = step.ux_term;
        for (int k = 0; k < N; ++k)
        {
            out.pi[k] = step.pi[k];
        }

        stat_.iter = 0;
        compute_residuals(in, out, ws_.res);
        fill_stat_residuals(stat_.row(0), ws_.res);
        if (!out.ux_first.allFinite() || !out.ux_term.allFinite())
        {
            stat_.status = Status::kNanDetected;
            return Status::kNanDetected;
        }
        stat_.status = Status::kSolved;
        return Status::kSolved;
    }

    // ==================================================================
    //  3e: absolute formulation (SQP_PHASE3_PLAN 3e)
    // ==================================================================
    //
    // HPIPM OCP_QP_IPM_SOLVE abs branch + OCP_QP_IPM_ABS_STEP
    // (x_ocp_qp_ipm.c:2777-3051, 2053-2228). Orthogonal to split_step
    // (shares compute_alpha / update_vars). The KKT right-hand side is the
    // ORIGINAL QP vectors (rqz = grad, b, d) rather than the KKT residuals;
    // the complementarity RHS is transformed each iteration
    // (res_m <- -(lam*t - m) - 2*m, then tau-shifted); each KKT solve
    // returns the NEW iterate, from which the current iterate is subtracted
    // to recover the delta; the exit test is the scalar |mu - tau_min|
    // (mu = mean |lam*t - m| over active sides) plus the alpha guards, with
    // no per-iteration OCP_QP_RES_COMPUTE. No iterative refinement in the
    // abs path.

    /// Subtract the current iterate `cur` from `step` in place: the
    /// absolute-form KKT solve returns the new iterate; the Newton delta is
    /// new - current (HPIPM AXPY -1.0, x_ocp_qp_ipm.c:2096-2099).
    void subtract_iterate(const QpSol<P, NH>& cur, QpSol<P, NH>& step) const
    {
        const int N = step.N;
        step.ux_first -= cur.ux_first;
        for (int k = 1; k < N; ++k)
        {
            step.ux_path[k - 1] -= cur.ux_path[k - 1];
        }
        step.ux_term -= cur.ux_term;
        for (int k = 0; k < N; ++k)
        {
            step.pi[k] -= cur.pi[k];
        }
        step.lam_first -= cur.lam_first;
        for (int k = 1; k < N; ++k)
        {
            step.lam_path[k - 1] -= cur.lam_path[k - 1];
        }
        step.lam_term -= cur.lam_term;
        step.t_first -= cur.t_first;
        for (int k = 1; k < N; ++k)
        {
            step.t_path[k - 1] -= cur.t_path[k - 1];
        }
        step.t_term -= cur.t_term;
    }

    /// Scalar barrier parameter of the absolute iterate: mu = mean
    /// |lam*t - m| over the active sides (HPIPM inline mu in the abs loop,
    /// x_ocp_qp_ipm.c:3034-3044).
    S inline_mu(const Qp<P, NH>& in, const QpSol<P, NH>& out) const
    {
        S sum = S(0);
        int nc = 0;
        auto acc = [&](const auto& lam, const auto& t, const auto& m,
                       const auto& dmask)
        {
            const int n = static_cast<int>(lam.size());
            for (int i = 0; i < n; ++i)
            {
                if (dmask(i) > S(0.5))
                {
                    sum += std::fabs(lam(i) * t(i) - m(i));
                    ++nc;
                }
            }
        };
        acc(out.lam_first, out.t_first, in.first.m, in.first.d_mask);
        for (int k = 1; k < in.N; ++k)
        {
            acc(out.lam_path[k - 1], out.t_path[k - 1], in.path[k - 1].m,
                in.path[k - 1].d_mask);
        }
        acc(out.lam_term, out.t_term, in.term.m, in.term.d_mask);
        return (nc > 0) ? sum / nc : S(0);
    }

    /// Absolute-form KKT right-hand side: the ORIGINAL QP vectors
    /// (res_g = grad, res_b = b, res_d = d) rather than the KKT residuals
    /// (HPIPM sets qp_step->rqz/b/d to the original QP when abs_form = 1,
    /// x_ocp_qp_ipm.c:2990-2992). res_m is filled separately each iteration
    /// by transform_abs_res_m + shift_res_m.
    void fill_abs_rhs(const Qp<P, NH>& in, QpRes<P, NH>& rhs) const
    {
        rhs.res_g_first = in.first.grad;
        for (int k = 1; k < in.N; ++k)
        {
            rhs.res_g_path[k - 1] = in.path[k - 1].grad;
        }
        rhs.res_g_term = in.term.grad;
        rhs.res_b[0] = in.first.b;
        for (int k = 1; k < in.N; ++k)
        {
            rhs.res_b[k] = in.path[k - 1].b;
        }
        rhs.res_d_first = in.first.d;
        for (int k = 1; k < in.N; ++k)
        {
            rhs.res_d_path[k - 1] = in.path[k - 1].d;
        }
        rhs.res_d_term = in.term.d;
    }

    /// Absolute-form m-RHS transform (HPIPM AXPBY, x_ocp_qp_ipm.c:2075):
    /// res_m <- -(lam*t - m) - 2*m = -lam*t - m (element-wise; v1 m = 0).
    void transform_abs_res_m(const Qp<P, NH>& in, const QpSol<P, NH>& out,
                             QpRes<P, NH>& res) const
    {
        res.res_m_first =
            -(out.lam_first.cwiseProduct(out.t_first)) - in.first.m;
        for (int k = 1; k < in.N; ++k)
        {
            res.res_m_path[k - 1] =
                -(out.lam_path[k - 1].cwiseProduct(out.t_path[k - 1]))
                - in.path[k - 1].m;
        }
        res.res_m_term =
            -(out.lam_term.cwiseProduct(out.t_term)) - in.term.m;
    }

    /// Absolute-formulation IPM (see section banner). Mirrors the relative
    /// solve() loop with the absolute KKT right-hand side, the per-iteration
    /// current-iterate subtraction, and the scalar |mu - tau_min| exit test.
    Status solve_abs(const Qp<P, NH>& in, QpSol<P, NH>& out)
    {
        const int N = in.N;
        ws_.resize(N);
        stat_.init(opts_.stat_max);

        QpRes<P, NH> abs_rhs(N);
        QpRes<P, NH> abs_bkp(N);
        fill_abs_rhs(in, abs_rhs);

        init_point(in, out);
        mask_abs_lam(in, out);
        ws_.alpha_prim = S(1);
        ws_.alpha_dual = S(1);
        ws_.force_lq = false;
        ws_.preg_last = S(opts_.reg_prim);

        // pre-iteration residuals (stat row 0) + initial mu
        compute_residuals(in, out, ws_.res);
        S mu = inline_mu(in, out);
        fill_stat_residuals(stat_.row(0), ws_.res);
        stat_.row(0).mu = mu;

        QpSol<P, NH> step(N);
        for (int kk = 0; kk < opts_.iter_max; ++kk)
        {
            // exit test (x_ocp_qp_ipm.c:3022-3027)
            if (ws_.alpha_prim <= opts_.alpha_min
                    or ws_.alpha_dual <= opts_.alpha_min)
            {
                break;
            }
            if (std::fabs(mu - opts_.tau_min) <= opts_.res_m_max)
            {
                break;
            }

            // --- predictor (OCP_QP_IPM_ABS_STEP) ---
            transform_abs_res_m(in, out, abs_rhs);
            abs_bkp = abs_rhs;
            abs_bkp.res_mu = mu;
            shift_res_m(in, abs_rhs, opts_.tau_min);

            const Status fs = delta_factor_solve(in, out, abs_rhs, step);
            if (fs != Status::kSolved)
            {
                stat_.iter = kk;
                stat_.status = Status::kQpFailure;
                return Status::kQpFailure;
            }
            subtract_iterate(out, step);
            mask_step(in, step);

            compute_alpha(in, out, step);
            HpipmIteration* row = stat_row(kk);
            if (row)
            {
                row->alpha_prim_aff = ws_.alpha_prim;
                row->alpha_dual_aff = ws_.alpha_dual;
            }

            // Mehrotra predictor-corrector (abs: same structure, abs RHS)
            if (opts_.pred_corr)
            {
                const S mu_aff_pred = compute_mu_aff(in, out, step);
                if (row)
                {
                    row->mu_aff = mu_aff_pred;
                }
                apply_centering(in, abs_bkp, step, /*correction=*/true,
                                abs_rhs);
                solve_kkt(in, out, abs_rhs, step);
                subtract_iterate(out, step);
                mask_step(in, step);
                compute_alpha(in, out, step);
                if (row)
                {
                    row->sigma = ws_.sigma;
                    row->alpha_prim = ws_.alpha_prim;
                    row->alpha_dual = ws_.alpha_dual;
                }
                if (opts_.cond_pred_corr)
                {
                    const S mu_aff_corr = compute_mu_aff(in, out, step);
                    if (mu_aff_corr > S(2) * mu_aff_pred)
                    {
                        apply_centering(in, abs_bkp, step,
                                        /*correction=*/false, abs_rhs);
                        solve_kkt(in, out, abs_rhs, step);
                        subtract_iterate(out, step);
                        mask_step(in, step);
                        compute_alpha(in, out, step);
                        if (row)
                        {
                            row->alpha_prim = ws_.alpha_prim;
                            row->alpha_dual = ws_.alpha_dual;
                        }
                    }
                }
            }
            else
            {
                if (row)
                {
                    row->alpha_prim = ws_.alpha_prim;
                    row->alpha_dual = ws_.alpha_dual;
                }
            }

            update_vars(out, step);
            mask_abs_lam(in, out);

            mu = inline_mu(in, out);
            if (row)
            {
                row->mu = mu;
            }
            stat_.iter = kk + 1;
        }

        // final residuals (HPIPM comp_res_exit for the abs branch) + status
        compute_residuals(in, out, ws_.res);
        if (stat_.iter > 0 && stat_.iter < stat_.stat_max)
        {
            fill_stat_residuals(stat_.row(stat_.iter), ws_.res);
        }

        if (stat_.iter == opts_.iter_max)
        {
            stat_.status = Status::kMaxIterations;
            return Status::kMaxIterations;
        }
        if (ws_.alpha_prim <= opts_.alpha_min
                or ws_.alpha_dual <= opts_.alpha_min)
        {
            stat_.status = Status::kMinStep;
            return Status::kMinStep;
        }
        if (std::isnan(mu))
        {
            stat_.status = Status::kNanDetected;
            return Status::kNanDetected;
        }
        stat_.status = Status::kSolved;
        return Status::kSolved;
    }

    /// One stage of compute_residuals (x_ocp_qp_res.c:424-523).
    ///
    /// Multiplier / constraint-value convention (x_ocp_qp_res.c:442-492):
    /// the constraint value v of a row is evaluated on the (u;x) part of z
    /// only (HPIPM's DCt covers (u;x); the stored slack columns of DC are
    /// NOT used here). Per-side slack coupling is explicit:
    ///   v_lo = v + s_lo,  v_hi = v − s_hi   (soft sides via idxs_lo / idxs_hi)
    /// and res_g picks up −λ_lo·e_{s_lo} − λ_hi·e_{s_hi} per soft side, plus
    /// −λ_slack at each slack column (x_ocp_qp_res.c:477, 480-488).
    ///
    /// @param pi_prev  π_{k−1} (nullptr at k = 0)
    /// @param pi_cur   π_k (nullptr at the terminal stage)
    /// @param x_next   x-part of z_{k+1} (nullptr at the terminal stage)
    /// @param ba       BA_k (nullptr at the terminal stage)
    /// @param b        b_k (nullptr at the terminal stage)
    /// @param idxs_lo / idxs_hi  per-row slack column (−1 if the side is not
    ///                           soft), matching the DC row layout
    /// @param res_b    res_b_k (nullptr at the terminal stage)
    template <class H, class G, class DC, class DV, class ZV, class LV,
              class Rg, class Rd, class Rm>
    void stage_residuals(const H& hess, const G& grad, const DC& dc,
                         const DV& d, const DV& d_mask, const DV& m,
                         const ZV& z, const LV& lam, const LV& t,
                         const Pi_t* pi_prev, const Pi_t* pi_cur,
                         const Pi_t* x_next, const Ba_t* ba, const Pi_t* b,
                         detail::QpLayout lay,
                         const std::array<int, DC::RowsAtCompileTime>& idxs_lo,
                         const std::array<int, DC::RowsAtCompileTime>& idxs_hi,
                         Rg& res_g, Rd& res_d, Rm& res_m, Pi_t* res_b,
                         S& obj, S& dual_gap) const
    {
        const int nvar = static_cast<int>(z.size());
        const int nrow = static_cast<int>(dc.rows());
        const int lo = lay.lo_size();
        const int nslack = static_cast<int>(d.size()) - lo - nrow;
        const int s0 = nvar - nslack;  // first slack column in z
        const int xoff = nvar - nslack - D::nx;  // x-part column offset
        const int nux = nvar - nslack;  // (u;x)-part width

        const auto lam_m = d_mask.cwiseProduct(lam);

        res_g = hess * z + grad;
        const auto Hz = hess * z;
        obj += 0.5 * z.dot(Hz) + z.dot(grad);
        dual_gap += z.dot(Hz) + z.dot(grad);
        dual_gap -= d.transpose() * lam_m;  // x_ocp_qp_res.c:497

        // Multiplier part of the stationarity (x_ocp_qp_res.c:442-462):
        // res_g += (λ_hi − λ_lo)ᵀ DC_nox_row (masked sides contribute 0).
        if (nrow > 0)
        {
            Eigen::Matrix<S, DC::RowsAtCompileTime, 1> w;
            for (int r = 0; r < nrow; ++r)
            {
                const int sh = lay.side_hi(r);
                const int sl = lay.side_lo(r);
                w(r) = lam_m(sh) - (sl >= 0 ? lam_m(sl) : 0.0);
            }
            res_g.head(nux) += dc.leftCols(nux).transpose() * w;
        }
        // Slack coupling of the row multipliers (x_ocp_qp_res.c:480-488):
        // res_g[s_lo col] −= λ_lo, res_g[s_hi col] −= λ_hi.
        for (int r = 0; r < nrow; ++r)
        {
            const int cl = idxs_lo[r];
            if (cl >= 0)
            {
                res_g(cl) -= lam_m(lay.side_lo(r));
            }
            const int ch = idxs_hi[r];
            if (ch >= 0)
            {
                res_g(ch) -= lam_m(lay.side_hi(r));
            }
        }
        // Slack-side multipliers (x_ocp_qp_res.c:477): res_g[s_j] −= λ_sj.
        for (int j = 0; j < nslack; ++j)
        {
            res_g(s0 + j) -= lam_m(lo + nrow + j);
        }

        // res_d = d + t − σ·v, masked (x_ocp_qp_res.c:452-466, 491-492);
        // v evaluated on (u;x), per-side slack added / removed.
        auto v = dc.leftCols(nux) * z.head(nux);
        for (int r = 0; r < nrow; ++r)
        {
            const int sh = lay.side_hi(r);
            const int sl = lay.side_lo(r);
            S v_hi = v(r);
            const int ch = idxs_hi[r];
            if (ch >= 0)
            {
                v_hi -= z(ch);
            }
            res_d(sh) = d(sh) + t(sh) + v_hi;
            if (sl >= 0)
            {
                S v_lo = v(r);
                const int cl = idxs_lo[r];
                if (cl >= 0)
                {
                    v_lo += z(cl);
                }
                res_d(sl) = d(sl) + t(sl) - v_lo;
            }
        }
        for (int j = 0; j < nslack; ++j)
        {
            const int si = lo + nrow + j;
            res_d(si) = d(si) + t(si) - z(s0 + j);
        }
        res_d = d_mask.cwiseProduct(res_d);

        // res_m = d_mask ∘ (λ ∘ t − m) (x_ocp_qp_res.c:513-516)
        res_m = d_mask.cwiseProduct(lam.cwiseProduct(t) - m);

        // Dynamics coupling (x_ocp_qp_res.c:440, 505-510)
        if (ba != nullptr)
        {
            res_g.head(D::nu + D::nx) += ba->transpose() * (*pi_cur);
            *res_b = (*ba) * z.head(D::nu + D::nx) + (*b) - (*x_next);
            dual_gap -= (*b).transpose() * (*pi_cur);  // x_ocp_qp_res.c:508
        }
        if (pi_prev != nullptr)
        {
            for (int i = 0; i < D::nx; ++i)
            {
                res_g(xoff + i) -= (*pi_prev)(i);
            }
        }
    }

    /// One stage of init_point (HPIPM OCP_QP_INIT_VAR, var_init_scheme = 1,
    /// t0_init = 2; x_ocp_qp_ipm.c:1905-2040).
    ///
    /// Order (as in HPIPM): (1) slack variables / their sides, (2) box-type
    /// rows (pin / bx / bu) from the variable values, with bound repair of
    /// the decision variable, (3) general rows (ineq / eq / lin) from the
    /// row value v = DC z_(u;x), (4) lam = mu0 / t on every side. Absent
    /// sides (d_mask = 0) are computed like present ones; they are masked
    /// out in the residuals.
    ///
    /// @param z       variable vector (u; x; s) / (x; s), mutated in place
    ///                 (slack values, bound repair)
    /// @param lam     [out] multipliers, one per side
    /// @param t       [out] IPM slacks, one per side
    /// @param d       bound offsets (lo: lo, hi: -hi, slack: 0)
    /// @param dc      constraint Jacobian over z (natural orientation)
    /// @param lay     side/row layout of this stage type
    /// @param idxs_lo / idxs_hi  per-row slack column in z (-1 if not soft)
    /// @param bound   (row, varidx) pairs of the box-type rows (compile-time
    ///                 sized std::array; alloc-free)
    /// @param nux     width of the (u;x) part of z
    /// @param nslack  number of slack variables
    template <class Z, class Lam, class T, class DV, class DC, class IdxLo,
              class IdxHi, class BoundArr>
    void init_stage(Z& z, Lam& lam, T& t, const DV& d, const DC& dc,
                    detail::QpLayout lay, const IdxLo& idxs_lo,
                    const IdxHi& idxs_hi, const BoundArr& bound, int nux,
                    int nslack, S thr0, S mu0) const
    {
        const int nrow = static_cast<int>(dc.rows());
        const int lo = lay.lo_size();
        const int nside = static_cast<int>(t.size());
        const int s0 = static_cast<int>(z.size()) - nslack;

        // (1) slacks: t_s = -d_s + s (d_s = 0), repair s
        for (int j = 0; j < nslack; ++j)
        {
            const int side = lo + nrow + j;
            S ts = -d(side) + z(s0 + j);
            if (ts < thr0)
            {
                ts = thr0;
                z(s0 + j) = d(side) + thr0;
            }
            t(side) = ts;
        }

        // (2) box-type rows (pin / bx / bu)
        for (const auto& br : bound)
        {
            const int row = br.first;
            const int varidx = br.second;
            const int sl = lay.side_lo(row);
            const int sh = lay.side_hi(row);
            S val = z(varidx);
            const S slo = (idxs_lo[row] >= 0) ? z(idxs_lo[row]) : S(0);
            const S shi = (idxs_hi[row] >= 0) ? z(idxs_hi[row]) : S(0);
            S tl = val + slo - d(sl);
            S th = -val + shi - d(sh);
            if (tl < thr0)
            {
                if (th < thr0)
                {
                    z(varidx) = 0.5 * (d(sl) - d(sh));
                    tl = thr0;
                    th = thr0;
                }
                else
                {
                    tl = thr0;
                    z(varidx) = d(sl) + thr0;
                }
            }
            else if (th < thr0)
            {
                th = thr0;
                z(varidx) = -d(sh) - thr0;
            }
            t(sl) = tl;
            t(sh) = th;
        }

        // (3) general rows (ineq / eq / lin)
        for (int r = 0; r < nrow; ++r)
        {
            const int g = lay.group_of(r);
            if (g != detail::g_ineq && g != detail::g_eq && g != detail::g_lin)
            {
                continue;
            }
            S v = 0.0;
            for (int j = 0; j < nux; ++j)
            {
                v += dc(r, j) * z(j);
            }
            const S slo = (idxs_lo[r] >= 0) ? z(idxs_lo[r]) : S(0);
            const S shi = (idxs_hi[r] >= 0) ? z(idxs_hi[r]) : S(0);
            const int sl = lay.side_lo(r);
            const int sh = lay.side_hi(r);
            S th = -v + shi - d(sh);
            t(sh) = (th < thr0) ? thr0 : th;
            if (sl >= 0)
            {
                S tl = v + slo - d(sl);
                t(sl) = (tl < thr0) ? thr0 : tl;
            }
        }

        // (4) multipliers: lam = mu0 / t (every side; masked sides are
        // ignored by the residuals)
        for (int i = 0; i < nside; ++i)
        {
            lam(i) = mu0 / t(i);
        }
    }

    // ==================================================================
    //  1e: KKT solve (two-pass Riccati, slacks kept explicit in M_k)
    // ==================================================================
    //
    // Reduced Newton system per stage k (worklog sec. 2):
    //   M_k dz_k + BA_k' dpi_k - pin(dpi_{k-1}) = rtilde0_k,  rtilde0_k =
    //       -res_g_k - C_k'(gamma_k)
    //   dx_{k+1} = D_k dz_k + res_b_k
    // with M_k = H_k + reg*I_(u;x) + C_k' diag(Gamma_k) C_k (Decision D1:
    // slacks explicit, full dense Cholesky over (u;x;s)). The two-pass
    // Riccati is done in a (u;s;x) ordering (x last) so that the x-block
    // Schur complement of Mtilde_k = M_k + D_k' P_{k+1} D_k is read
    // directly off the Cholesky factor:
    //   backward (k = N..0): Mtilde_k, L_k = chol(Mtilde_k),
    //       P_k = L_k(x,x)' L_k(x,x);  rtilde_k = rtilde0_k - D_k'(P_{k+1}
    //       res_b_k + q_{k+1});  q_k = -rtilde_kx + L_k(x,w) z, z =
    //       L_k(w,w)^{-1} rtilde_kw.
    //   forward (k = 0..N): dz_k = Mtilde_k^{-1}(rtilde_k + e_x dpi_{k-1});
    //       dpi_k = P_{k+1}(D_k dz_k + res_b_k) + q_{k+1}.
    //   closed form: dt_i = C_i dz_k - res_d_i,
    //       dlam_i = d_mask_i (-(res_m_i + lam_i dt_i)/t_i).
    // P_{N+1} = q_{N+1} = 0, dpi_{-1} = 0; D_N / BA_N absent (terminal).

    /// Stored (u;x;s) index for a (u;s;x) position i (x last).
    static int usx_old_idx(int i, int nu, int nx, int nsk)
    {
        if (i < nu)
        {
            return i;
        }
        if (i < nu + nsk)
        {
            return nu + nx + (i - nu);
        }
        return nu + (i - nu - nsk);
    }

    /// Permute a column vector from (u;x;s) to (u;s;x) order (x last).
    template <class V>
    static V usx_permute_vec(const V& v, int nu, int nx, int nsk)
    {
        V out;
        const int nv = static_cast<int>(v.size());
        for (int i = 0; i < nv; ++i)
        {
            out(i) = v(usx_old_idx(i, nu, nx, nsk));
        }
        return out;
    }

    /// Permute a column vector from (u;s;x) back to (u;x;s) order.
    template <class V>
    static V usx_unpermute_vec(const V& v, int nu, int nx, int nsk)
    {
        V out;
        const int nv = static_cast<int>(v.size());
        for (int i = 0; i < nv; ++i)
        {
            out(usx_old_idx(i, nu, nx, nsk)) = v(i);
        }
        return out;
    }

    /// Permute a square matrix from (u;x;s) to (u;s;x) order (rows + cols).
    template <class M>
    static M usx_permute_mat(const M& m, int nu, int nx, int nsk)
    {
        M out;
        const int nv = static_cast<int>(m.rows());
        for (int i = 0; i < nv; ++i)
        {
            const int oi = usx_old_idx(i, nu, nx, nsk);
            for (int j = 0; j < nv; ++j)
            {
                out(i, j) = m(oi, usx_old_idx(j, nu, nx, nsk));
            }
        }
        return out;
    }

    /// Permute only the COLUMNS of a matrix from (u;x;s) to (u;s;x) order
    /// (rows kept in place).  For a rectangular side-DC (nside x nvar) whose
    /// columns run over the natural (u;x;s) variable order, this yields the
    /// same block expressed in the (u;s;x) order used by the LQ wide matrix.
    template <class M>
    static M usx_permute_cols(const M& m, int nu, int nx, int nsk)
    {
        M out;
        const int nr = static_cast<int>(m.rows());
        const int nc = static_cast<int>(m.cols());
        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nc; ++j)
            {
                out(i, j) = m(i, usx_old_idx(j, nu, nx, nsk));
            }
        }
        return out;
    }

    /// Effective side-DC C (nside x nvar, stored (u;x;s) order): the gradient
    /// of each side's constraint function w.r.t. z. `nside_ref` is unused
    /// except to fix the row extent.
    template <class Dc, class Dv, class IdxLo, class IdxHi>
    static auto usx_build_side_dc(const Dc& dc, const Dv& nside_ref,
                                  detail::QpLayout lay,
                                  const IdxLo& idxs_lo,
                                  const IdxHi& idxs_hi, int nsk)
    {
        (void)nside_ref;
        using C = Eigen::Matrix<typename Dc::Scalar, Dv::RowsAtCompileTime,
                                Dc::ColsAtCompileTime>;
        C c;
        c.setZero();
        const int nrow = static_cast<int>(dc.rows());
        const int lo = lay.lo_size();
        for (int r = 0; r < nrow; ++r)
        {
            const int sl = lay.side_lo(r);
            const int sh = lay.side_hi(r);
            if (sl >= 0)
            {
                c.row(sl) = dc.row(r);
                const int ch = idxs_hi[r];
                if (ch >= 0)
                {
                    c(sl, ch) = 0.0;
                }
            }
            c.row(sh) = -dc.row(r);
            const int cl = idxs_lo[r];
            if (cl >= 0)
            {
                c(sh, cl) = 0.0;
            }
        }
        const int s0 = Dc::ColsAtCompileTime - nsk;
        for (int j = 0; j < nsk; ++j)
        {
            c(lo + nrow + j, s0 + j) = 1.0;
        }
        return c;
    }

    /// Per-side gamma / Gamma / t_inv (HPIPM COMPUTE_GAMMA_GAMMA_QP,
    /// x_core_qp_ipm_aux.c:38): gamma_i = (res_m_i - lam_i res_d_i)/t_i,
    /// Gamma_i = lam_i/t_i, both masked by d_mask (absent sides -> 0).
    void compute_gamma(const Qp<P, NH>& in, const QpSol<P, NH>& iter,
                       const QpRes<P, NH>& res)
    {
        auto fill = [](auto& gamma, auto& Gamma, auto& t_inv, const auto& dmask,
                       const auto& lam, const auto& t, const auto& res_d,
                       const auto& res_m)
        {
            const int nside = static_cast<int>(gamma.size());
            for (int i = 0; i < nside; ++i)
            {
                t_inv(i) = 1.0 / t(i);
                if (dmask(i) > 0.5)
                {
                    Gamma(i) = lam(i) * t_inv(i);
                    gamma(i) = t_inv(i) * (res_m(i) - lam(i) * res_d(i));
                }
                else
                {
                    Gamma(i) = 0.0;
                    gamma(i) = 0.0;
                }
            }
        };
        fill(ws_.gamma_first, ws_.Gamma_first, ws_.t_inv_first,
             in.first.d_mask, iter.lam_first, iter.t_first, res.res_d_first,
             res.res_m_first);
        for (int k = 1; k < in.N; ++k)
        {
            fill(ws_.gamma_path[k - 1], ws_.Gamma_path[k - 1],
                 ws_.t_inv_path[k - 1], in.path[k - 1].d_mask,
                 iter.lam_path[k - 1], iter.t_path[k - 1],
                 res.res_d_path[k - 1], res.res_m_path[k - 1]);
        }
        fill(ws_.gamma_term, ws_.Gamma_term, ws_.t_inv_term, in.term.d_mask,
             iter.lam_term, iter.t_term, res.res_d_term, res.res_m_term);
    }

    /// Factor one stage (backward): form Mtilde_k = M_k + D_k' P_{k+1} D_k
    /// in (u;s;x) order, Cholesky it, and read off P_k = L_k(x,x)' L_k(x,x).
    /// Returns false if the Cholesky is singular. NU / NX / NSK are the
    /// stage's (u; x; s) extents (NSK may be 0; the term stage has NU = 0).
    template <int NU, int NX, int NSK, class H, class Dc, class Dv, class Ba,
              class IdxLo, class IdxHi>
    bool factor_one_stage(bool has_dyn, const H& hess, const Dc& dc,
                          const Dv& gamma, const Dv& Gamma, const Ba* ba,
                          detail::QpLayout lay, const IdxLo& idxs_lo,
                          const IdxHi& idxs_hi, S reg,
                          const P_first_t& p_next, H& m_usx_out,
                          H& l_usx_out, P_first_t& p_out)
    {
        (void)gamma;
        using Ss = typename H::Scalar;
        constexpr int nvar = H::RowsAtCompileTime;
        constexpr int nw = NU + NSK;
        const auto c = usx_build_side_dc(dc, Gamma, lay, idxs_lo, idxs_hi,
                                         NSK);
        H m_nat = hess + c.transpose() * (Gamma.asDiagonal() * c);
        H m_usx = usx_permute_mat(m_nat, NU, NX, NSK);
        for (int i = 0; i < NU; ++i)
        {
            m_usx(i, i) += reg;
        }
        for (int i = 0; i < NX; ++i)
        {
            m_usx(nw + i, nw + i) += reg;
        }
        if (has_dyn)
        {
            Eigen::Matrix<Ss, NX, nvar> dk;
            dk.setZero();
            if (NU > 0)
            {
                dk.block(0, 0, NX, NU) = ba->block(0, 0, NX, NU);
            }
            dk.block(0, nw, NX, NX) = ba->block(0, NU, NX, NX);
            m_usx += dk.transpose() * p_next * dk;
        }
        m_usx_out = m_usx;
        Eigen::LLT<H> llt(m_usx);
        if (llt.info() != Eigen::Success)
        {
            return false;
        }
        l_usx_out = llt.matrixL();
        const auto lxx = l_usx_out.block(nw, nw, NX, NX);
        p_out = lxx * lxx.transpose();
        return true;
    }

    /// LQ-factor one stage (backward, 3d): build the wide matrix
    ///   W_k = [ Lh_k | (sqrt(Gamma) C_k)^T | D_k^T L_{k+1,xx} ]
    /// in the (u;s;x) ordering so that W_k W_k^T = Mtilde_k (the same
    /// augmented Hessian the Cholesky path forms, worklog 1e).  LQ-factor
    /// W_k (Eigen HouseholderQR of W_k^T, top-left block of R transposed)
    /// and sign-normalize the diagonal so L_k L_k^T = Mtilde_k with
    /// L_k(i,i) >= 0 (HPIPM GELQF + COLSC, x_ocp_qp_kkt.c:2032-2047).  The
    /// forward / backward-reduced / closed-form stages then reuse L_k and
    /// P_k exactly as the Cholesky path does (only the factor source
    /// differs).  `lh_valid` / `lh_stored` carry the cached Lh_k (HPIPM
    /// `use_hess_fact`); a singular Hessian block returns false (the caller
    /// runs the reg-primal growth ladder).
    template <int NU, int NX, int NSK, class H, class Dc, class Dv, class Ba,
              class IdxLo, class IdxHi, class Lxx>
    bool factor_one_stage_lq(bool has_dyn, const H& hess, const Dc& dc,
                             const Dv& gamma, const Dv& Gamma, const Ba* ba,
                             detail::QpLayout lay, const IdxLo& idxs_lo,
                             const IdxHi& idxs_hi, S reg, bool lh_valid,
                             const H& lh_stored, const Lxx& l_next_xx,
                             H& lh_out, H& l_usx_out, P_first_t& p_out)
    {
        (void)gamma;
        using Ss = typename H::Scalar;
        constexpr int nvar = H::RowsAtCompileTime;
        constexpr int nside = Dv::RowsAtCompileTime;
        constexpr int nw = NU + NSK;

        // 1) Lh_k = chol(hess_usx + reg*I_(u;x)) (or reuse the cache)
        H lh;
        if (!lh_valid)
        {
            H h_usx = usx_permute_mat(hess, NU, NX, NSK);
            for (int i = 0; i < NU; ++i)
            {
                h_usx(i, i) += reg;
            }
            for (int i = 0; i < NX; ++i)
            {
                h_usx(nw + i, nw + i) += reg;
            }
            Eigen::LLT<H> llt(h_usx);
            if (llt.info() != Eigen::Success)
            {
                return false;
            }
            lh = llt.matrixL();
        }
        else
        {
            lh = lh_stored;
        }
        lh_out = lh;

        // 2) wide matrix W_k = [ Lh | (sqrt(Gamma) C_k)^T | D_k^T L_{k+1,xx} ]
        //    in (u;s;x) column order.  Lh is already usx (h_usx above); the
        //    side-DC from usx_build_side_dc is in NATURAL (u;x;s) column order
        //    (the Cholesky path combines it with the natural Hessian then
        //    permutes), so permute its columns to usx before embedding.
        using WideT = Eigen::Matrix<Ss, nvar, nvar + nside + NX>;
        WideT w;
        w.setZero();
        w.leftCols(nvar) = lh;
        const auto c_nat =
            usx_build_side_dc(dc, Gamma, lay, idxs_lo, idxs_hi, NSK);
        using Cs = Eigen::Matrix<Ss, nside, nvar>;
        const Cs c_usx = usx_permute_cols(c_nat, NU, NX, NSK);
        const Eigen::Matrix<Ss, nside, 1> sq =
            Eigen::Matrix<Ss, nside, 1>(Gamma.cwiseSqrt());
        // (sqrt(Gamma) C_k)^T (nvar x nside); inactive sides have Gamma = 0
        w.block(0, nvar, nvar, nside) = (sq.asDiagonal() * c_usx).transpose();
        if (has_dyn)
        {
            // D_k (nx x nvar, (u;s;x) order), the same layout as in
            // factor_one_stage (D_k z_k -> x_{k+1}).
            Eigen::Matrix<Ss, NX, nvar> dk;
            dk.setZero();
            if (NU > 0)
            {
                dk.block(0, 0, NX, NU) = ba->block(0, 0, NX, NU);
            }
            dk.block(0, nw, NX, NX) = ba->block(0, NU, NX, NX);
            // Pb_k = D_k^T L_{k+1,xx} (nvar x NX)
            w.block(0, nvar + nside, nvar, NX) = dk.transpose() * l_next_xx;
        }

        // 3) L_k from the LQ of W_k: L_k L_k^T = W_k W_k^T = Mtilde_k.
        //    QR-factor the tall matrix W_k^T (Eigen 3.3 HouseholderQR has
        //    no matrixR()); its top nvar x nvar upper triangle holds the top
        //    block of R (W_k^T = Q R), so its transpose is L_k (lower-tri).
        // L_k = R^T from the QR of W_k^T (nwide x nvar, tall): its top
        // nvar x nvar block is the upper-triangular R, so its transpose is
        // the lower-triangular L_k with L_k L_k^T = W_k W_k^T = Mtilde_k.
        // This is the LQ (HPIPM GELQF); unlike a Cholesky of Mtilde it stays
        // full rank when the augmented Hessian is ill-conditioned, which is
        // the whole point of the lq_fact fallback.
        using TallT = Eigen::Matrix<Ss, nvar + nside + NX, nvar>;
        const TallT wt = w.transpose();
        Eigen::HouseholderQR<TallT> qr(wt);
        l_usx_out =
            qr.matrixQR().topLeftCorner(nvar, nvar)
                .template triangularView<Eigen::Upper>()
                .transpose();
        // sign-normalize the diagonal (HPIPM COLSC, x_ocp_qp_kkt.c:2044-2046)
        for (int i = 0; i < nvar; ++i)
        {
            if (l_usx_out(i, i) < 0)
            {
                l_usx_out.col(i).tail(nvar - i) *= -1.0;
            }
        }

        // 4) P_k = L_k(x,x) L_k(x,x)^T
        const auto lxx = l_usx_out.block(nw, nw, NX, NX);
        p_out = lxx * lxx.transpose();
        return true;
    }

    /// Reduced RHS + costate offset for one stage (backward): rtilde_k =
    /// rtilde0_k - D_k'(P_{k+1} res_b_k + q_{k+1}) and q_k = -rtilde_kx +
    /// L_k(x,w) z with z = L_k(w,w)^{-1} rtilde_kw. Reuses L_k / P_{k+1}
    /// from the factorization; only gamma changes (corrector reuse).
    template <int NU, int NX, int NSK, class H, class Dc, class Dv, class Rg,
              class Rb, class Ba, class IdxLo, class IdxHi>
    void reduce_one_stage(bool has_dyn, const H& hess, const Dc& dc,
                          const Dv& gamma, const Rg& res_g, const Rb* res_b,
                          const Ba* ba, detail::QpLayout lay,
                          const IdxLo& idxs_lo, const IdxHi& idxs_hi,
                          const P_first_t& p_next, const Pi_t& q_next,
                          const H& l_usx, Rg& rtilde_out, Pi_t& q_out)
    {
        (void)hess;
        using Ss = typename H::Scalar;
        constexpr int nvar = H::RowsAtCompileTime;
        constexpr int nw = NU + NSK;
        const auto c = usx_build_side_dc(dc, gamma, lay, idxs_lo, idxs_hi,
                                         NSK);
        Rg r0_nat = -res_g - c.transpose() * gamma;
        Rg rtilde = usx_permute_vec(r0_nat, NU, NX, NSK);
        if (has_dyn)
        {
            Eigen::Matrix<Ss, NX, nvar> dk;
            dk.setZero();
            if (NU > 0)
            {
                dk.block(0, 0, NX, NU) = ba->block(0, 0, NX, NU);
            }
            dk.block(0, nw, NX, NX) = ba->block(0, NU, NX, NX);
            const Pi_t term = p_next * (*res_b) + q_next;
            rtilde -= dk.transpose() * term;
        }
        rtilde_out = rtilde;
        using Wm = Eigen::Matrix<Ss, nw, nw>;
        using Wv = Eigen::Matrix<Ss, nw, 1>;
        using Xv = Eigen::Matrix<Ss, NX, 1>;
        const Wv rw = rtilde.head(nw);
        const Xv rx = rtilde.tail(NX);
        const Wm lww = l_usx.topLeftCorner(nw, nw);
        // explicit forward substitution (L z = rw); avoids the GCC 13 -O2
        // maybe-uninitialized false positive in Eigen's triangular solver
        Wv z;
        for (int i = 0; i < nw; ++i)
        {
            Ss acc = rw(i);
            for (int j = 0; j < i; ++j)
            {
                acc -= lww(i, j) * z(j);
            }
            z(i) = acc / lww(i, i);
        }
        const Eigen::Matrix<Ss, NX, nw> lxm = l_usx.block(nw, 0, NX, nw);
        q_out = -rx + lxm * z;
    }

    /// Forward solve for one stage: dz_k = Mtilde_k^{-1}(rtilde_k + e_x
    /// dpi_{k-1}) and (when a costate exists) dpi_k = P_{k+1}(D_k dz_k +
    /// res_b_k) + q_{k+1}.
    template <class H, class Rg, class Rb, class Ba>
    void forward_one_stage(bool has_dyn, const H& l_usx, const Rg& rtilde,
                           const Pi_t& dpi_prev, const P_first_t& p_next,
                           const Pi_t& q_next, const Rb* res_b, const Ba* ba,
                           int nsk, int nu, int nx, Rg& dz_usx_out,
                           Pi_t* dpi_out)
    {
        (void)has_dyn;
        using Ss = typename H::Scalar;
        Rg b = rtilde;
        for (int i = 0; i < nx; ++i)
        {
            b(nu + nsk + i) += dpi_prev(i);
        }
        // Solve Mtilde dz = b with Mtilde = L L^T, by explicit forward /
        // backward substitution on L (numerically identical to the Eigen
        // triangular solves; avoids a GCC 13 -O2 maybe-uninitialized false
        // positive in Eigen's unrolled triangular solver on tiny stages).
        const int nn = static_cast<int>(b.size());
        Rg y;
        for (int i = 0; i < nn; ++i)
        {
            Ss acc = b(i);
            for (int j = 0; j < i; ++j)
            {
                acc -= l_usx(i, j) * y(j);
            }
            y(i) = acc / l_usx(i, i);
        }
        for (int i = nn - 1; i >= 0; --i)
        {
            Ss acc = y(i);
            for (int j = i + 1; j < nn; ++j)
            {
                acc -= l_usx(j, i) * dz_usx_out(j);
            }
            dz_usx_out(i) = acc / l_usx(i, i);
        }
        if (dpi_out != nullptr)
        {
            Pi_t dnext;
            for (int i = 0; i < nx; ++i)
            {
                Ss acc = (*res_b)(i);
                for (int j = 0; j < nu; ++j)
                {
                    acc += (*ba)(i, j) * dz_usx_out(j);
                }
                for (int j = 0; j < nx; ++j)
                {
                    acc += (*ba)(i, nu + j) * dz_usx_out(nu + nsk + j);
                }
                dnext(i) = acc;
            }
            *dpi_out = p_next * dnext + q_next;
        }
    }

    /// Closed-form slack / multiplier step for one stage:
    /// dt_i = C_i dz_k - res_d_i and
    /// dlam_i = d_mask_i (-(res_m_i + lam_i dt_i)/t_i).
    ///
    /// HPIPM (x_core_qp_ipm_aux.c COMPUTE_LAM_T_QP) writes the same thing
    /// as `dlam = -t^-1 (res_m + lam dt_aff - lam res_d); dt -= res_d`;
    /// the two forms are identical (dt_aff - res_d = dt).
    template <class Dc, class Dv, class Lam, class T, class Rd, class Rm,
              class Rg, class IdxLo, class IdxHi>
    void closed_form_one_stage(const Dc& dc, const Dv& dmask, const Lam& lam,
                               const T& t, const Rd& res_d, const Rm& res_m,
                               detail::QpLayout lay, const IdxLo& idxs_lo,
                               const IdxHi& idxs_hi, int nsk, const Rg& dz,
                               Rd& dt_out, Rm& dlam_out)
    {
        const auto c = usx_build_side_dc(dc, res_d, lay, idxs_lo, idxs_hi,
                                         nsk);
        dt_out = c * dz - res_d;
        const auto lam_m = dmask.cwiseProduct(lam);
        dlam_out =
            dmask.cwiseProduct((-res_m - lam_m.cwiseProduct(dt_out))
                                   .cwiseQuotient(t));
    }

    // Stage dispatch wrappers (first / path / term).

    bool backward_factor_stage(int k, const Qp<P, NH>& in, S reg)
    {
        const int N = in.N;
        if (k == 0)
        {
            const auto& st = in.first;
            const P_first_t& p_next = (N == 1) ? ws_.P_term : ws_.P_path[0];
            return factor_one_stage<D::nu, D::nx, D::nslack_first>(
                true, st.hess, st.DC, ws_.gamma_first, ws_.Gamma_first,
                &st.BA, D::lay_first, D::idxs_lo_first, D::idxs_hi_first, reg,
                p_next, ws_.M_first, ws_.L_first, ws_.P_first);
        }
        if (k == N)
        {
            const auto& st = in.term;
            const P_first_t p_next = P_first_t::Zero();
            return factor_one_stage<0, D::nx, D::nslack_term>(
                false, st.hess, st.DC, ws_.gamma_term, ws_.Gamma_term,
                static_cast<const Ba_t*>(nullptr), D::lay_term,
                D::idxs_lo_term, D::idxs_hi_term, reg, p_next, ws_.M_term,
                ws_.L_term, ws_.P_term);
        }
        const auto& st = in.path[k - 1];
        const P_first_t& p_next =
            (k + 1 == N) ? ws_.P_term : ws_.P_path[k];
        return factor_one_stage<D::nu, D::nx, D::nslack_path>(
            true, st.hess, st.DC, ws_.gamma_path[k - 1],
            ws_.Gamma_path[k - 1], &st.BA, D::lay_path, D::idxs_lo_path,
            D::idxs_hi_path, reg, p_next, ws_.M_path[k - 1],
            ws_.L_path[k - 1], ws_.P_path[k - 1]);
    }

    /// LQ variant of backward_factor_stage (3d): fills L_k / P_k via the
    /// wide-matrix LQ factor (factor_one_stage_lq) instead of the dense
    /// Cholesky of Mtilde_k.  Reads the x-block of L_{k+1} (already factored,
    /// we walk k = N..0) to build the dynamics column of W_k; zero at k = N.
    bool backward_factor_stage_lq(int k, const Qp<P, NH>& in, S reg,
                                  bool lh_valid)
    {
        const int N = in.N;
        Eigen::Matrix<S, D::nx, D::nx> lnext_xx;
        lnext_xx.setZero();
        if (k < N)
        {
            if (k + 1 == N)
            {
                const int nw = D::nslack_term;
                lnext_xx = ws_.L_term.block(nw, nw, D::nx, D::nx);
            }
            else
            {
                const int nw = D::nu + D::nslack_path;
                lnext_xx = ws_.L_path[k].block(nw, nw, D::nx, D::nx);
            }
        }
        if (k == 0)
        {
            const auto& st = in.first;
            return factor_one_stage_lq<D::nu, D::nx, D::nslack_first>(
                true, st.hess, st.DC, ws_.gamma_first, ws_.Gamma_first,
                &st.BA, D::lay_first, D::idxs_lo_first, D::idxs_hi_first,
                reg, lh_valid, ws_.Lh_first, lnext_xx, ws_.Lh_first,
                ws_.L_first, ws_.P_first);
        }
        if (k == N)
        {
            const auto& st = in.term;
            return factor_one_stage_lq<0, D::nx, D::nslack_term>(
                false, st.hess, st.DC, ws_.gamma_term, ws_.Gamma_term,
                static_cast<const Ba_t*>(nullptr), D::lay_term,
                D::idxs_lo_term, D::idxs_hi_term, reg, lh_valid, ws_.Lh_term,
                lnext_xx, ws_.Lh_term, ws_.L_term, ws_.P_term);
        }
        const auto& st = in.path[k - 1];
        return factor_one_stage_lq<D::nu, D::nx, D::nslack_path>(
            true, st.hess, st.DC, ws_.gamma_path[k - 1],
            ws_.Gamma_path[k - 1], &st.BA, D::lay_path, D::idxs_lo_path,
            D::idxs_hi_path, reg, lh_valid, ws_.Lh_path[k - 1], lnext_xx,
            ws_.Lh_path[k - 1], ws_.L_path[k - 1], ws_.P_path[k - 1]);
    }

    void backward_reduced_stage(int k, const Qp<P, NH>& in,
                                const QpRes<P, NH>& res)
    {
        const int N = in.N;
        if (k == 0)
        {
            const auto& st = in.first;
            const P_first_t& p_next = (N == 1) ? ws_.P_term : ws_.P_path[0];
            const Pi_t& q_next = (N == 1) ? ws_.q_term : ws_.q_path[0];
            reduce_one_stage<D::nu, D::nx, D::nslack_first>(
                true, st.hess, st.DC, ws_.gamma_first, res.res_g_first,
                &res.res_b[0], &st.BA, D::lay_first, D::idxs_lo_first,
                D::idxs_hi_first, p_next, q_next, ws_.L_first,
                ws_.rtilde_first, ws_.q_first);
        }
        else if (k == N)
        {
            const auto& st = in.term;
            const P_first_t p_next = P_first_t::Zero();
            const Pi_t q_next = Pi_t::Zero();
            reduce_one_stage<0, D::nx, D::nslack_term>(
                false, st.hess, st.DC, ws_.gamma_term, res.res_g_term,
                static_cast<const Pi_t*>(nullptr),
                static_cast<const Ba_t*>(nullptr), D::lay_term,
                D::idxs_lo_term, D::idxs_hi_term, p_next, q_next, ws_.L_term,
                ws_.rtilde_term, ws_.q_term);
        }
        else
        {
            const auto& st = in.path[k - 1];
            const P_first_t& p_next =
                (k + 1 == N) ? ws_.P_term : ws_.P_path[k];
            const Pi_t& q_next = (k + 1 == N) ? ws_.q_term : ws_.q_path[k];
            reduce_one_stage<D::nu, D::nx, D::nslack_path>(
                true, st.hess, st.DC, ws_.gamma_path[k - 1],
                res.res_g_path[k - 1], &res.res_b[k], &st.BA, D::lay_path,
                D::idxs_lo_path, D::idxs_hi_path, p_next, q_next,
                ws_.L_path[k - 1], ws_.rtilde_path[k - 1],
                ws_.q_path[k - 1]);
        }
    }

    void forward_stage(int k, const Qp<P, NH>& in, const QpRes<P, NH>& res,
                       QpSol<P, NH>& step)
    {
        const int N = in.N;
        Pi_t dpi_prev = Pi_t::Zero();
        if (k >= 1)
        {
            dpi_prev = step.pi[k - 1];
        }
        if (k == 0)
        {
            const auto& st = in.first;
            const P_first_t& p_next = (N == 1) ? ws_.P_term : ws_.P_path[0];
            const Pi_t& q_next = (N == 1) ? ws_.q_term : ws_.q_path[0];
            v_first_t dz_usx;
            forward_one_stage(true, ws_.L_first, ws_.rtilde_first, dpi_prev,
                              p_next, q_next, &res.res_b[0], &st.BA,
                              D::nslack_first, D::nu, D::nx, dz_usx,
                              &step.pi[0]);
            step.ux_first = usx_unpermute_vec(dz_usx, D::nu, D::nx,
                                              D::nslack_first);
        }
        else if (k == N)
        {
            v_term_t dz_usx;
            forward_one_stage(false, ws_.L_term, ws_.rtilde_term, dpi_prev,
                              P_first_t::Zero(), Pi_t::Zero(),
                              static_cast<const Pi_t*>(nullptr),
                              static_cast<const Ba_t*>(nullptr),
                              D::nslack_term, 0, D::nx, dz_usx, nullptr);
            step.ux_term = usx_unpermute_vec(dz_usx, 0, D::nx,
                                             D::nslack_term);
        }
        else
        {
            const auto& st = in.path[k - 1];
            const P_first_t& p_next =
                (k + 1 == N) ? ws_.P_term : ws_.P_path[k];
            const Pi_t& q_next = (k + 1 == N) ? ws_.q_term : ws_.q_path[k];
            v_path_t dz_usx;
            forward_one_stage(true, ws_.L_path[k - 1],
                              ws_.rtilde_path[k - 1], dpi_prev, p_next,
                              q_next, &res.res_b[k], &st.BA, D::nslack_path,
                              D::nu, D::nx, dz_usx, &step.pi[k]);
            step.ux_path[k - 1] =
                usx_unpermute_vec(dz_usx, D::nu, D::nx, D::nslack_path);
        }
    }

    void closed_form_stage(int k, const Qp<P, NH>& in, const QpRes<P, NH>& res,
                           const QpSol<P, NH>& iter, QpSol<P, NH>& step)
    {
        const int N = in.N;
        if (k == 0)
        {
            const auto& st = in.first;
            c_first_t dt, dlam;
            closed_form_one_stage(st.DC, st.d_mask, iter.lam_first,
                                  iter.t_first, res.res_d_first,
                                  res.res_m_first, D::lay_first,
                                  D::idxs_lo_first, D::idxs_hi_first,
                                  D::nslack_first, step.ux_first, dt, dlam);
            step.t_first = dt;
            step.lam_first = dlam;
        }
        else if (k == N)
        {
            const auto& st = in.term;
            c_term_t dt, dlam;
            closed_form_one_stage(st.DC, st.d_mask, iter.lam_term,
                                  iter.t_term, res.res_d_term,
                                  res.res_m_term, D::lay_term, D::idxs_lo_term,
                                  D::idxs_hi_term, D::nslack_term,
                                  step.ux_term, dt, dlam);
            step.t_term = dt;
            step.lam_term = dlam;
        }
        else
        {
            const auto& st = in.path[k - 1];
            c_path_t dt, dlam;
            closed_form_one_stage(st.DC, st.d_mask, iter.lam_path[k - 1],
                                  iter.t_path[k - 1], res.res_d_path[k - 1],
                                  res.res_m_path[k - 1], D::lay_path,
                                  D::idxs_lo_path, D::idxs_hi_path,
                                  D::nslack_path, step.ux_path[k - 1], dt,
                                  dlam);
            step.t_path[k - 1] = dt;
            step.lam_path[k - 1] = dlam;
        }
    }

    // Remaining HPIPM flow, owned by later sub-steps (worklog sec. 4):
    //   OCP_QP_IPM_DELTA_STEP + main IPM loop   -> 1h (solve())
};

}  // namespace ocp

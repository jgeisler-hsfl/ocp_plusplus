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
#include <cassert>
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

        // cached forward-pass product (BAbt_k * L_{k+1,xx}'), doc 05 sec. 4
        Pb_first_t Pb_first{};
        Trajectory<Pb_path_t, detail::traj_extent<NH, -1>()> Pb_path;
        Pb_term_t Pb_term{};

        // KKT residuals (HPIPM d_ocp_qp_res analogue)
        QpRes<P, NH> res;

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
        S alpha = 0;
        S reg = 0;  // current reg_prim (grown on Cholesky failure)

        int iter = 0;
        bool npd_reg_hess = false;

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
                Pb_path.resize(np);
                gamma_path.resize(np);
                Gamma_path.resize(np);
                t_inv_path.resize(np);
                rhs_path.resize(np);
                dv_path.resize(np);
                dlam_path.resize(np);
                dt_path.resize(np);
                dpi.resize(n);
                res.resize(n);
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

    /// Solve the staged QP (relative-formulation IPM).
    ///
    /// `in`: the QP data (HPIPM `qp`); `out`: the primal STEP over the
    /// stage-type variable vectors (u; x; s) / (x; s) plus the ABSOLUTE
    /// multipliers pi, lam, t (the QpSol mixed convention).
    ///
    /// Exit (worklog sec. 1, `x_ocp_qp_ipm.c:3119`): converged when
    /// alpha > alpha_min and res_g / res_b / res_d / (tau-shifted) res_m
    /// are within their tolerances and dual_gap <= dual_gap_max.
    /// @return kSolved, or kMaxIterations / kMinStep / kQpFailure /
    ///         kNanDetected on failure.
    ///
    /// @note shell (worklog 1a): returns kAborted until the IPM loop is
    ///         implemented (worklog 1h).
    Status solve(const Qp<P, NH>& in, QpSol<P, NH>& out)
    {
        ws_.resize(in.N);
        (void)out;
        return Status::kAborted;
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

    const HpipmOptions& options() const { return opts_; }
    const HpipmStatistics& statistics() const { return stat_; }

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

    // 1:1 translations of the HPIPM macros, implemented in the owning
    // sub-step (SQP_PHASE1_WORKLOG.md sec. 4):
    //   void init_point(const Qp<P, NH>&, QpSol<P, NH>&);       // 1c
    //   void fact_solve_kkt(const Qp<P, NH>&, QpSol<P, NH>&,
    //                       QpRes<P, NH>&);                    // 1e
    //   void solve_kkt(const Qp<P, NH>&, QpSol<P, NH>&,
    //                  QpRes<P, NH>&);                        // 1e
    //   void solve_kkt_unconstr(const Qp<P, NH>&, QpSol<P, NH>&);  // 1e
    //   double compute_alpha(const Qp<P, NH>&, const QpSol<P, NH>&);  // 1f
    //   double compute_mu_aff(const Qp<P, NH>&, const QpSol<P, NH>&); // 1g
    //   void apply_centering(const Qp<P, NH>&, const QpSol<P, NH>&); // 1g
    //   void update_vars(const Qp<P, NH>&, const QpSol<P, NH>&,
    //                    double alpha);                          // 1f
};

}  // namespace ocp

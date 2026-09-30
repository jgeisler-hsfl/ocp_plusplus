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

    const HpipmOptions& options() const { return opts_; }
    const HpipmStatistics& statistics() const { return stat_; }

private:
    HpipmOptions opts_;
    HpipmStatistics stat_;
    Workspace ws_;

    // 1:1 translations of the HPIPM macros, implemented in the owning
    // sub-step (SQP_PHASE1_WORKLOG.md sec. 4):
    //   void init_point(const Qp<P, NH>&, QpSol<P, NH>&);       // 1c
    //   void compute_residuals(const Qp<P, NH>&, const QpSol<P, NH>&,
    //                          QpRes<P, NH>&);                 // 1b
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

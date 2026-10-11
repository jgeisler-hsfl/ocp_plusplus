// ocp/solvers/hpipm/dense_qp.hpp
//
// Dense QP data model for full condensing (parity plan Phase 1,
// docs/plans/SQP_HPIPM_PARITY_PHASE1_PLAN.md).
//
// A staged Qp<P, NH> is condensed into a single dense QP on
//
//   v = [x_0 ; u_0 ; ... ; u_{N-1} ; s_0 ; ... ; s_N]
//
// x_0 is always a decision variable (when fixed_initial_state it is pinned by
// a lo == hi box row, parity plan decision Q4).  The slacks s_k (one per soft
// side, as in the staged model) are ordinary dense variables: the s-s block
// of H is diagonal (the soft-penalty weights), the +/-1 slack columns live in
// C, and each slack carries a non-negativity row (lo side active, hi side
// masked).  The dynamics are eliminated by substitution, so the condensed
// problem has NO equality rows:
//
//   min   0.5 v' H v + g' v
//   s.t.  box rows (variable v[idxb[r]]):
//           lo side:  v_b - d <= t       (d holds the lower bound `lo`)
//           hi side: -v_b - d <= t      (d holds `-hi`)
//          general rows: lo side C_lo v, hi side C_hi v, same side
//          convention.  C_lo and C_hi carry the same (x_0; u) coefficients
//          but DIFFERENT slack columns: the staged model couples each side
//          to its own slack only (lo: v + s_lo >= d_lo, hi: v - s_hi <= -d_hi,
//          HPIPM OCP x_ocp_qp_res.c), so a single shared C row with both the
//          +1 s_lo and -1 s_hi columns would cross-couple the two sides and
//          define a different problem.
//
// Side layout: [lo sides (nb + ng); hi sides (nb + ng)], i.e. d / d_mask / m
// / lam / t have 2 * (nb + ng) entries, lo block first.  Dense row order:
//   box rows:  stage-0 rows that stay box (pin rows when fixed, then the
//              HARD state box when free, then the hard control box), then
//              the hard control-box rows of stages k = 1..N-1, then the
//              slack non-negativity rows (in v order)
//   gen rows:  stage-0 soft state / control-box rows followed by ineq/eq/lin,
//              then for k = 1..N-1 the (converted) state-box rows, soft
//              control-box rows and ineq/eq/lin, then the terminal rows
//              (state box + ineq/eq/lin)
//
// Soft box rows (a *_soft_idx entry) become general rows: their staged DC
// row carries the +1 / -1 slack columns, which a dense box row (one idxb
// variable) cannot express.
//
// Extents (N = horizon, N >= 1):
//   nv = nx + N*nu + nslack_first + (N-1)*nslack_path + nslack_term
//   nb = nbox_first + (N-1)*nbox_path + nslack_total(N)
//   ng = ngen_first + (N-1)*ngen_path + ngen_term
// where (from QpDim<P>):
//   nbox_first = (fixed ? nx : 0) + (fixed ? 0 : nbx - nbx_soft) + (nbu - nbu_soft)
//   ngen_first = ng + ne + nl + nbx_soft + nbu_soft
//   ngen_path  = nbx + ng + ne + nl + nbu_soft
//   nbox_path  = nbu - nbu_soft
//   ngen_term  = nbx_t + ng_t + ne_t + nl_t
//
// Naming translation (ocp++ dense -> HPIPM d_dense_qp):
//   v, H, g  ->  v (hard vars), Hv, gz  (slacks folded into v; HPIPM keeps
//                  them separate with Z and idxs/idxs_rev - we do not)
//   C, d, d_mask, m -> C (t), d, d_mask, m (two-sided rows; HPIPM masks
//                  one-sided rows the same way via d_mask)
//   idxb -> idxb (box-row -> variable index)
//
// NH = Eigen::Dynamic (default): runtime horizon, heap-backed storage,
// extents set by resize(int).  NH >= 1: compile-time horizon; fixed-extent
// storage, resize(int) only asserts the argument matches NH.  Same contract
// as Qp / QpSol / QpRes.
//
// HPIPM reference:
//   acados/external/hpipm/include/hpipm_d_dense_qp*.h
//   acados/external/hpipm/cond/x_cond.c  (COND_QP_COMPUTE_DIM)

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "../../problem.hpp"
#include "qp.hpp"

namespace ocp
{

// =========================================================================
//  DenseQpDim
// =========================================================================

/// Compile-time structural dimensions of the fully condensed dense QP,
/// derived from the staged QpDim<P>.  See the header comment for the extent
/// formulas and the dense row / side layouts.
template <class P>
struct DenseQpDim
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    static constexpr int nx = D::nx;
    static constexpr int nu = D::nu;
    static constexpr bool kFixed = P::fixed_initial_state;

    /// Soft state-box rows at stage 0 (emitted as general rows: the soft
    /// slack column must stay in the constraint, which a box row cannot
    /// carry).  Absent when x_0 is fixed (state box replaced by pin rows).
    static constexpr int nbx_soft_first = kFixed ? 0 : P::nbx_soft;
    /// Soft control-box rows (emitted as general rows at every stage that
    /// carries a control box; same reason as nbx_soft_first).
    static constexpr int nbu_soft = P::nbu_soft;

    /// Stage-0 rows that stay box rows in the dense QP (pin rows when
    /// x_0 is fixed, then the hard state box when free, then the hard
    /// control box).
    static constexpr int nbox_first = (kFixed ? nx : 0)
                                      + (kFixed ? 0 : D::nbx - nbx_soft_first)
                                      + (D::nbu - nbu_soft);
    /// Stage-0 rows that become general rows (soft state / control box,
    /// then ineq / eq / lin; the hard state box and pin rows are box rows).
    static constexpr int ngen_first = D::ng + D::ne + D::nl
                                     + nbx_soft_first + nbu_soft;
    /// Path-stage rows that become general rows (the state box converts
    /// from box to general; soft control box converts too; ineq / eq / lin
    /// already are).
    static constexpr int ngen_path = D::nbx + D::ng + D::ne + D::nl + nbu_soft;
    /// Path-stage rows that stay box rows (the hard control box).
    static constexpr int nbox_path = D::nbu - nbu_soft;
    /// Terminal rows that become general rows (all of them).
    static constexpr int ngen_term = D::nbx_t + D::ng_t + D::ne_t + D::nl_t;

    /// Total slack variables for a horizon of N stages.
    static constexpr int nslack_total(int N)
    {
        return D::nslack_first + (N - 1) * D::nslack_path + D::nslack_term;
    }

    /// Dense variable count for a horizon of N stages.
    static constexpr int nv_total(int N)
    {
        return nx + N * nu + nslack_total(N);
    }

    /// Dense box-row count for a horizon of N stages.
    static constexpr int nb_total(int N)
    {
        return nbox_first + (N - 1) * nbox_path + nslack_total(N);
    }

    /// Dense general-row count for a horizon of N stages.
    static constexpr int ng_total(int N)
    {
        return ngen_first + (N - 1) * ngen_path + ngen_term;
    }

    /// Dense side count for a horizon of N stages (length of d / d_mask /
    /// m / lam / t).
    static constexpr int nside_total(int N)
    {
        return 2 * (nb_total(N) + ng_total(N));
    }

    // ------------------------------------------------------------------
    // Variable offsets in v = [x_0; u_0; ...; u_{N-1}; s_0; ...; s_N]
    // ------------------------------------------------------------------

    /// First column of the x_0 block.
    static constexpr int x0_off()
    {
        return 0;
    }

    /// First column of the u_k block (k = 0..N-1).
    static constexpr int u_off(int k)
    {
        return nx + k * nu;
    }

    /// First column of the slack block for a horizon of N stages.
    static constexpr int s_base(int N)
    {
        return nx + N * nu;
    }

    /// First column of the stage-k slack block (k = 0..N) for a horizon of
    /// N stages; slacks are ordered [s_0; s_1; ...; s_N].
    static constexpr int s_off(int k, int N)
    {
        return (k >= 1)
            ? s_base(N) + D::nslack_first + (k - 1) * D::nslack_path
            : s_base(N);
    }
};

namespace detail
{

/// Matrix type with compile-time extents when NH is fixed, dynamic
/// otherwise (R / C must be positive constants in the fixed case).
template <class S, bool Dyn, int R, int C>
struct DenseMat
{
    using type = std::conditional_t<
        Dyn,
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>,
        Eigen::Matrix<S, R, C>>;
};

/// Vector type with compile-time extent when NH is fixed, dynamic
/// otherwise.
template <class S, bool Dyn, int R>
struct DenseVec
{
    using type = std::conditional_t<
        Dyn, Eigen::Matrix<S, Eigen::Dynamic, 1>, Eigen::Matrix<S, R, 1>>;
};

/// Index vector type: std::array when NH is fixed, std::vector otherwise.
template <bool Dyn, int N>
struct DenseIdx
{
    using type = std::conditional_t<Dyn, std::vector<int>, std::array<int, N>>;
};

}  // namespace detail

// =========================================================================
//  DenseQp
// =========================================================================

/// A fully condensed dense QP instance (horizon N, see the header comment).
///
/// NH = Eigen::Dynamic (default): runtime horizon, `resize(int)` allocates.
/// NH >= 1: compile-time horizon; fixed-extent storage, `resize(int)` only
/// asserts that the argument matches NH.
template <class P, int NH = Eigen::Dynamic>
struct DenseQp
{
    static_assert(NH == Eigen::Dynamic || NH >= 1,
        "ocp::DenseQp: the horizon must be at least 1 (one dynamics step) "
        "or Eigen::Dynamic.");

    using D = DenseQpDim<P>;
    using S = typename P::scalar_t;
    static constexpr bool kDyn = (NH == Eigen::Dynamic);

    // Extents for fixed-NH storage (0 in dynamic mode; the storage types
    // only use them in the fixed branch).
    static constexpr int nv_ext = (NH >= 1) ? D::nv_total(NH) : 0;
    static constexpr int nb_ext = (NH >= 1) ? D::nb_total(NH) : 0;
    static constexpr int ng_ext = (NH >= 1) ? D::ng_total(NH) : 0;
    static constexpr int nside_ext = (NH >= 1) ? D::nside_total(NH) : 0;

    using H_t = typename detail::DenseMat<S, kDyn, nv_ext, nv_ext>::type;
    using G_t = typename detail::DenseVec<S, kDyn, nv_ext>::type;
    using C_t = typename detail::DenseMat<S, kDyn, ng_ext, nv_ext>::type;
    using D_t = typename detail::DenseVec<S, kDyn, nside_ext>::type;
    using Idx_t = typename detail::DenseIdx<kDyn, nb_ext>::type;

    int N = kDyn ? 0 : NH;
    int nv = 0;  // dense variables (runtime in dynamic mode)
    int nb = 0;  // dense box rows
    int ng = 0;  // dense general rows
    int nside = 0;  // 2 * (nb + ng)

    /// Dense Hessian (symmetric, full storage) over v.
    H_t H{};
    /// Dense linear cost term.
    G_t g{};
    /// Lo-side general-row Jacobian (ng x nv), dense general-row order.
    /// Carries the (x_0; u) flow-map coefficients plus the +1 s_lo slack
    /// column of each soft row; the s_hi column is zero (the staged model
    /// couples the lo side to its own slack only, see header comment).
    C_t C_lo{};
    /// Hi-side general-row Jacobian (ng x nv), same (x_0; u) coefficients as
    /// C_lo; carries the -1 s_hi slack column of each soft row; the s_lo
    /// column is zero.
    C_t C_hi{};
    /// Box-row -> variable index (dense box-row order).
    Idx_t idxb{};
    /// Bound offsets (side layout [lo; hi]; lo sides hold `lo`, hi sides
    /// hold `-hi`; general rows carry the flow-map constant shift, see
    /// condense.hpp).
    D_t d{};
    /// 1 = finite / active side, 0 = infinite / absent side (the masked hi
    /// sides of slack non-negativity rows).
    D_t d_mask{};
    /// Complementarity RHS: lam * t = m (zero in v1).
    D_t m{};

    DenseQp() { setZero(); }

    /// Dynamic horizon: allocate for `n_stages` stages; fixed horizon:
    /// assert that `n_stages` matches NH.
    void resize(int n_stages)
    {
        if constexpr (kDyn)
        {
            assert(n_stages >= 1);
            N = n_stages;
        }
        else
        {
            assert(n_stages == NH);
            N = NH;
        }
        set_extents(N);
        if constexpr (kDyn)
        {
            H.resize(nv, nv);
            H.setZero();
            g.resize(nv);
            g.setZero();
            C_lo.resize(ng, nv);
            C_lo.setZero();
            C_hi.resize(ng, nv);
            C_hi.setZero();
            idxb.resize(nb);
            d.resize(nside);
            d.setZero();
            d_mask.resize(nside);
            d_mask.setZero();
            m.resize(nside);
            m.setZero();
        }
    }

    /// Zero all dense QP data (container contract for fixed-extent
    /// storage).
    void setZero()
    {
        set_extents(N);
        H.setZero();
        g.setZero();
        C_lo.setZero();
        C_hi.setZero();
        d.setZero();
        d_mask.setZero();
        m.setZero();
    }

private:
    void set_extents(int n_stages)
    {
        nv = D::nv_total(n_stages);
        nb = D::nb_total(n_stages);
        ng = D::ng_total(n_stages);
        nside = D::nside_total(n_stages);
    }
};

// =========================================================================
//  DenseQpSol
// =========================================================================

/// Solution of a condensed dense QP.
///
/// Mixed convention (matching the staged QpSol):
///   - v   : primal STEP over the dense variable vector
///   - lam : ABSOLUTE constraint multipliers (side layout [lo; hi])
///   - t   : ABSOLUTE IPM slacks, same layout as lam
template <class P, int NH = Eigen::Dynamic>
struct DenseQpSol
{
    static_assert(NH == Eigen::Dynamic || NH >= 1,
        "ocp::DenseQpSol: the horizon must be at least 1 (one dynamics "
        "step) or Eigen::Dynamic.");

    using D = DenseQpDim<P>;
    using S = typename P::scalar_t;
    static constexpr bool kDyn = (NH == Eigen::Dynamic);

    static constexpr int nv_ext = (NH >= 1) ? D::nv_total(NH) : 0;
    static constexpr int nside_ext = (NH >= 1) ? D::nside_total(NH) : 0;

    using V_t = typename detail::DenseVec<S, kDyn, nv_ext>::type;
    using Side_t = typename detail::DenseVec<S, kDyn, nside_ext>::type;

    int N = kDyn ? 0 : NH;
    int nv = 0;
    int nside = 0;

    V_t v{};
    Side_t lam{};
    Side_t t{};

    /// Primal objective value (set by the solver).
    S obj = 0.0;

    DenseQpSol() { setZero(); }

    /// Dynamic horizon: allocate for `n_stages` stages; fixed horizon:
    /// assert that `n_stages` matches NH.
    void resize(int n_stages)
    {
        if constexpr (kDyn)
        {
            assert(n_stages >= 1);
            N = n_stages;
        }
        else
        {
            assert(n_stages == NH);
            N = NH;
        }
        nv = D::nv_total(N);
        nside = D::nside_total(N);
        if constexpr (kDyn)
        {
            v.resize(nv);
            lam.resize(nside);
            t.resize(nside);
        }
        setZero();
    }

    /// Zero all solution data.
    void setZero()
    {
        v.setZero();
        lam.setZero();
        t.setZero();
        obj = 0.0;
    }
};

// =========================================================================
//  DenseQpRes
// =========================================================================

/// KKT residuals of a condensed dense QP (HPIPM `d_dense_qp_res` analogue,
/// without the equality residual).
///
/// Residual conventions (same sign convention as the staged solver,
/// hpipm.hpp compute_residuals):
///   res_g = H v + g + J_b'(lam_hi - lam_lo) + C'(lam_hi - lam_lo)
///           (masked sides contribute 0)
///   res_d = d + t -/+ v_row   (lo / slack sides: -v_row; hi sides: +v_row)
///           (masked by d_mask)
///   res_m = lam * t - m       (masked by d_mask)
template <class P, int NH = Eigen::Dynamic>
struct DenseQpRes
{
    static_assert(NH == Eigen::Dynamic || NH >= 1,
        "ocp::DenseQpRes: the horizon must be at least 1 (one dynamics "
        "step) or Eigen::Dynamic.");

    using D = DenseQpDim<P>;
    using S = typename P::scalar_t;
    static constexpr bool kDyn = (NH == Eigen::Dynamic);

    static constexpr int nv_ext = (NH >= 1) ? D::nv_total(NH) : 0;
    static constexpr int nside_ext = (NH >= 1) ? D::nside_total(NH) : 0;

    using V_t = typename detail::DenseVec<S, kDyn, nv_ext>::type;
    using Side_t = typename detail::DenseVec<S, kDyn, nside_ext>::type;

    int N = kDyn ? 0 : NH;
    int nv = 0;
    int nside = 0;

    /// Stationarity residual (nv).
    V_t res_g{};
    /// Inequality feasibility residual (side layout [lo; hi]).
    Side_t res_d{};
    /// Complementarity residual (side layout [lo; hi]).
    Side_t res_m{};

    /// Inf-norm maxima.
    S res_g_max = 0;
    S res_d_max = 0;
    S res_m_max = 0;

    /// res_mu = ||res_m||_1 / nc_mask  (masked mean of complementarity)
    S res_mu_sum = 0;
    S res_mu = 0;

    /// Primal objective value / dual gap (accumulated during residual
    /// computation).
    S obj = 0;
    S dual_gap = 0;

    DenseQpRes() { setZero(); }

    /// Dynamic horizon: allocate for `n_stages` stages; fixed horizon:
    /// assert that `n_stages` matches NH.
    void resize(int n_stages)
    {
        if constexpr (kDyn)
        {
            assert(n_stages >= 1);
            N = n_stages;
        }
        else
        {
            assert(n_stages == NH);
            N = NH;
        }
        nv = D::nv_total(N);
        nside = D::nside_total(N);
        if constexpr (kDyn)
        {
            res_g.resize(nv);
            res_d.resize(nside);
            res_m.resize(nside);
        }
        setZero();
    }

    /// Zero all residual data.
    void setZero()
    {
        res_g.setZero();
        res_d.setZero();
        res_m.setZero();
        res_g_max = 0;
        res_d_max = 0;
        res_m_max = 0;
        res_mu_sum = 0;
        res_mu = 0;
        obj = 0;
        dual_gap = 0;
    }
};

}  // namespace ocp

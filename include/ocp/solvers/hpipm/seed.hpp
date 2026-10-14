// ocp/solvers/hpipm/seed.hpp
//
// QP seed data model (HPIPM `d_ocp_qp_seed` analogue, x_ocp_qp_seed.c).
// A seed carries only the RIGHT-hand-side fields of the staged QP:
//
//   seed_g  -> linear cost term     (this: grad_*)
//   seed_b  -> dynamics offset      (this: b)
//   seed_d  -> constraint offsets   (this: d_*, same side layout as Qp::d)
//   seed_m  -> complementarity RHS  (this: m_*, same side layout as Qp::m)
//
// The left-hand-side data (Hessian, BA, DC, d_mask) comes from the QP that
// was last solved; HpipmQpSolver::sens_forward / sens_adj re-solve the KKT
// system with the same LHS and the seed vectors as forcing (HPIPM
// OCP_QP_IPM_SENS_FRW / OCP_QP_IPM_SENS_ADJ, x_ocp_qp_ipm.c:3383/3455).
//
// Side layout: ours (lo block, hi block, slack sides per QpLayout), NOT
// HPIPM's [lb; lg; ub; ug; ls; us]. The hi-side sign convention is the same
// as Qp::d (w_cur - hi): a unit seed on an UPPER bound value enters d with
// a minus sign, mirroring HPIPM's set_seed_ub/ug/us -1 scaling.

#pragma once

#include "qp.hpp"
#include "dense_qp.hpp"

namespace ocp
{

/// Right-hand-side seed of a staged QP (HPIPM `d_ocp_qp_seed` analogue).
///
/// NH = Eigen::Dynamic (default): runtime horizon, `resize(int)`
/// allocates. NH >= 1: compile-time horizon, path storage is fixed-extent
/// (std::array) and `resize(int)` only asserts that the argument matches
/// NH. Same mechanism as Qp / QpSol / QpRes.
template <class P, int NH = Eigen::Dynamic, class Dim = QpDim<P>>
struct QpSeed
{
    static_assert(NH == Eigen::Dynamic || NH >= 1,
        "ocp::QpSeed: the horizon must be at least 1 (one dynamics step) "
        "or Eigen::Dynamic.");

    using D = Dim;
    using S = typename P::scalar_t;

    using grad_first_t = Eigen::Matrix<S, D::nvar_first, 1>;
    using grad_path_t  = Eigen::Matrix<S, D::nvar_path, 1>;
    using grad_term_t  = Eigen::Matrix<S, D::nvar_term, 1>;
    using b_t          = Eigen::Matrix<S, D::nx, 1>;
    using d_first_t    = Eigen::Matrix<S, D::nside_first, 1>;
    using d_path_t     = Eigen::Matrix<S, D::nside_path, 1>;
    using d_term_t     = Eigen::Matrix<S, D::nside_term, 1>;

    int N = (NH == Eigen::Dynamic) ? 0 : NH;

    /// Linear cost term seed (HPIPM `seed_g`), over (u; x; s) / (x; s).
    grad_first_t grad_first;
    Trajectory<grad_path_t, detail::traj_extent<NH, -1>()> grad_path;
    grad_term_t grad_term;

    /// Dynamics offset seed (HPIPM `seed_b`), k = 0..N-1.
    Trajectory<b_t, detail::traj_extent<NH, 0>()> b;

    /// Constraint offset seed (HPIPM `seed_d`), same side layout as Qp::d.
    d_first_t d_first;
    Trajectory<d_path_t, detail::traj_extent<NH, -1>()> d_path;
    d_term_t d_term;

    /// Complementarity RHS seed (HPIPM `seed_m`), same side layout as
    /// Qp::m (zero in v1; carried for HPIPM parity and adjoint scaling).
    d_first_t m_first;
    Trajectory<d_path_t, detail::traj_extent<NH, -1>()> m_path;
    d_term_t m_term;

    /// All seed vectors are zero-initialized (an unset seed row contributes
    /// nothing to the KKT right-hand side).
    QpSeed()
    {
        setZero();
    }
    /// Dynamic horizon: allocate for `n_stages` stages; fixed horizon:
    /// assert that `n_stages` matches NH.
    explicit QpSeed(int n_stages)
    {
        resize(n_stages);
    }

    void resize(int n_stages)
    {
        if constexpr (NH == Eigen::Dynamic)
        {
            assert(n_stages >= 1);
            N = n_stages;
            grad_path.resize(N - 1);
            b.resize(N);
            d_path.resize(N - 1);
            m_path.resize(N - 1);
        }
        else
        {
            assert(n_stages == NH);
            N = NH;
        }
        setZero();
    }

    /// Zero all seed vectors.
    void setZero()
    {
        grad_first.setZero();
        for (int i = 0; i < N - 1; ++i)
        {
            grad_path[i].setZero();
        }
        grad_term.setZero();
        for (int k = 0; k < N; ++k)
        {
            b[k].setZero();
        }
        d_first.setZero();
        for (int i = 0; i < N - 1; ++i)
        {
            d_path[i].setZero();
        }
        d_term.setZero();
        m_first.setZero();
        for (int i = 0; i < N - 1; ++i)
        {
            m_path[i].setZero();
        }
        m_term.setZero();
    }
};

/// Right-hand-side seed of a condensed dense QP (HPIPM `d_dense_qp_seed`
/// analogue, x_dense_qp_seed.c).  Carries only the dense QP's RHS fields:
/// the linear cost term (`seed_g`), the bound offsets (`seed_d`, same side
/// layout as DenseQp::d: [lo; hi]), and the complementarity RHS (`seed_m`,
/// zero in v1; used by the adjoint scaling).  There is NO `seed_b`: the
/// condensed model has eliminated the dynamics by substitution, so there is
/// no dynamics-offset field (HPIPM's `seed_b` slot stays unused).
///
/// NH = Eigen::Dynamic (default): runtime horizon, `resize(int)`
/// allocates. NH >= 1: compile-time horizon, fixed-extent storage and
/// `resize(int)` only asserts that the argument matches NH.  Same
/// mechanism as DenseQp / DenseQpSol.
template <class P, int NH = Eigen::Dynamic>
struct DenseQpSeed
{
    static_assert(NH == Eigen::Dynamic || NH >= 1,
        "ocp::DenseQpSeed: the horizon must be at least 1 (one dynamics "
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

    /// Linear cost term seed (HPIPM `seed_g`), over the dense variable v.
    V_t seed_g{};
    /// Bound-offset seed (HPIPM `seed_d`), side layout [lo; hi].
    Side_t seed_d{};
    /// Complementarity RHS seed (HPIPM `seed_m`), side layout [lo; hi].
    Side_t seed_m{};

    /// All seed vectors are zero-initialized (an unset seed contributes
    /// nothing to the KKT right-hand side).
    DenseQpSeed()
    {
        setZero();
    }
    /// Dynamic horizon: allocate for `n_stages` stages; fixed horizon:
    /// assert that `n_stages` matches NH.
    explicit DenseQpSeed(int n_stages)
    {
        resize(n_stages);
    }

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
            seed_g.resize(nv);
            seed_d.resize(nside);
            seed_m.resize(nside);
        }
        setZero();
    }

    /// Zero all seed vectors.
    void setZero()
    {
        seed_g.setZero();
        seed_d.setZero();
        seed_m.setZero();
    }
};

}  // namespace ocp

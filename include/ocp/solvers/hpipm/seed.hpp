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

}  // namespace ocp

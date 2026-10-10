// ocp/solvers/acados/regularize.hpp
//
// Hessian regularization for the SQP driver.
//
// A regularizer is a duck-typed solver component with the two-method
// contract
//     void regularize(Qp<P, NH>&) const;
//     void correct_dual_sol(Qp<P, NH>&, QpSol<P, NH>&) const;
// invoked by the SQP driver between QP assembly and the QP solve.
//
// GlmRegularizer (GLM = Gershgorin-based): for every stage (first, each
// path, terminal) it takes the (u;x) block of the stage Hessian
// (first/path: the leading (nu+nx) x (nu+nx); terminal: the leading
// nx x nx, the x part) and computes a Gershgorin lower bound on its
// minimum eigenvalue
//     tmp = min_i ( a_ii - sum_{j != i} |a_ij| )
// If tmp < epsilon, the block diagonal is shifted by
//     alpha = (tmp < 0) ? |tmp| + epsilon : epsilon
// The slack diagonal and slack off-diagonals are not touched: only the
// (u;x) Hessian is regularized, and the slack penalty weights are
// positive definite by construction.
//
// NoRegularizer: both operations are no-ops. For both strategies,
// correct_dual_sol is a no-op: a pure Hessian shift leaves the dual
// (multiplier) solution valid.
//
// Supplementary - correspondence with the HPIPM/acados
// implementation:
//   GlmRegularizer     acados/ocp_nlp/ocp_nlp_reg_glm.c
//   NoRegularizer      acados/ocp_nlp/ocp_nlp_reg_noreg.c
//   gershgorin_min_eig acados/utils/math.c (compute_gershgorin_min_eig_estimate)
//
// HPIPM reference:
//   acados/ocp_nlp/ocp_nlp_reg_glm.c
//   acados/ocp_nlp/ocp_nlp_reg_noreg.c
//   acados/utils/math.c

#pragma once

#include <cmath>
#include <limits>

#include <Eigen/Dense>

#include "../hpipm/qp.hpp"

namespace ocp
{

/// Gershgorin lower bound on the minimum eigenvalue of a (symmetric)
/// matrix: min_i (a_ii - sum_{j != i} |a_ij|).
///
/// @param  A square symmetric matrix (or block expression of one)
/// @return lower bound on the minimum eigenvalue
template <class M>
double gershgorin_min_eig(const M& A)
{
    const int n = static_cast<int>(A.rows());
    double lam = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i)
    {
        double r_i = 0.0;
        for (int j = 0; j < n; ++j)
        {
            if (i != j)
            {
                r_i += std::fabs(A(i, j));
            }
        }
        lam = std::min(lam, static_cast<double>(A(i, i)) - r_i);
    }
    return lam;
}

/// GLM (Gershgorin-based) Hessian regularizer (duck-typed SQP component).
///
/// Templated over the concrete problem type `P` and the horizon `NH`,
/// mirroring HpipmQpSolver<P, NH> (the QP data model is NH-templated).
template <class P, int NH = Eigen::Dynamic>
class GlmRegularizer
{
public:
    /// Gershgorin threshold: a stage (u;x) block whose min-eig estimate is
    /// below epsilon gets shifted.
    double epsilon = 1e-6;

    /// Shift the (u;x) block of every stage Hessian so that its Gershgorin
    /// min-eig estimate reaches at least epsilon. In place.
    /// @param[out] qp staged QP, stage Hessians modified in place
    void regularize(Qp<P, NH>& qp) const
    {
        // first stage (k = 0) and path stages (k = 1..N-1): (u;x) block
        const int nux = QpDim<P>::nu + QpDim<P>::nx;
        regularize_block(qp.first.hess, nux);
        for (int k = 1; k < qp.N; ++k)
        {
            regularize_block(qp.path[k - 1].hess, nux);
        }
        // terminal stage (k = N): x block only (no control)
        regularize_block(qp.term.hess, QpDim<P>::nx);
    }

    /// No-op: a pure Hessian shift leaves the dual (multiplier) solution
    /// valid, so the solved duals need no correction.
    void correct_dual_sol(Qp<P, NH>&, QpSol<P, NH>&) const
    {
    }

private:
    /// Shift the leading nux x nux block of `hess` by alpha * I when its
    /// Gershgorin min-eig estimate tmp is below epsilon, with
    /// alpha = |tmp| + epsilon if tmp < 0, else epsilon.
    /// @param[out] hess stage Hessian, modified in place
    /// @param nux size of the leading (u;x) block to regularize
    template <class H>
    void regularize_block(H& hess, int nux) const
    {
        const double tmp =
            gershgorin_min_eig(hess.template topLeftCorner(nux, nux));
        if (tmp < epsilon)
        {
            const double alpha = (tmp < 0.0)
                ? std::fabs(tmp) + epsilon
                : epsilon;
            for (int i = 0; i < nux; ++i)
            {
                hess(i, i) += alpha;
            }
        }
    }
};

/// No-regularization strategy: both operations are no-ops.
///
/// Templated over the concrete problem type `P` and the horizon `NH`,
/// mirroring GlmRegularizer.
template <class P, int NH = Eigen::Dynamic>
class NoRegularizer
{
public:
    /// No-op: no regularization is applied.
    void regularize(Qp<P, NH>&) const
    {
    }

    /// No-op: no dual correction is needed.
    void correct_dual_sol(Qp<P, NH>&, QpSol<P, NH>&) const
    {
    }
};

}  // namespace ocp

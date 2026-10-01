// ocp/solvers/acados/regularize.hpp
//
// Hessian regularization for the SQP driver (phase 2, sub-step 2a).
// Port of the acados GLM (Gershgorin-based) Hessian regularizer
// (acados/ocp_nlp/ocp_nlp_reg_glm.c) and the no-regularization strategy
// (acados/ocp_nlp/ocp_nlp_reg_noreg.c).
//
// The regularizer is a duck-typed component of the SqpSolver (SQP_PLAN.md
// sec. 7; the two-method contract of SQP_PHASE2_PLAN.md sec. 2a):
//     void regularize(Qp<P, NH>&) const;
//     void correct_dual_sol(Qp<P, NH>&, QpSol<P, NH>&) const;
//
// GLM: for every stage (first, each path, term) take the (u;x) block of the
// stage Hessian (first/path: the leading (nu+nx) x (nu+nx); term: the
// leading nx x nx, the x part) and compute a Gershgorin lower bound on its
// minimum eigenvalue
//     tmp = min_i ( a_ii - sum_{j != i} |a_ij| )
// (acados/utils/math.c:1145-1170). If tmp < epsilon, shift the block
// diagonal by alpha with
//     alpha = (tmp < 0) ? |tmp| + epsilon : epsilon
// (ocp_nlp_reg_glm.c:227-239). The slack diagonal and slack off-diagonals
// are not touched: acados regularizes the (u;x) Hessian only (slacks are
// part of the QP, not the NLP Hessian), and the QP Hessian of the slacks
// (their penalty weights) is already PD by construction.
//
// correct_dual_sol is a no-op for both GLM and NoReg: a pure Hessian shift
// leaves the dual (multiplier) solution valid (ocp_nlp_reg_glm.c:255-258,
// ocp_nlp_reg_noreg.c; acados/docs/algorithm/04-regularization-and-qpscaling.md
// Part A).
//
// References (re-read at implementation time):
//   acados/ocp_nlp/ocp_nlp_reg_glm.c:69   (epsilon default 1e-6)
//   acados/ocp_nlp/ocp_nlp_reg_glm.c:216-258 (regularize, correct_dual_sol)
//   acados/utils/math.c:1145-1170         (compute_gershgorin_min_eig_estimate)

#pragma once

#include <cmath>
#include <limits>

#include <Eigen/Dense>

#include "../hpipm/qp.hpp"

namespace ocp
{

/// Gershgorin lower bound on the minimum eigenvalue of the (symmetric)
/// matrix `A`: min_i (a_ii - sum_{j != i} |a_ij|)
/// (acados/utils/math.c:1145-1170, compute_gershgorin_min_eig_estimate).
///
/// @param A square symmetric matrix (or block expression of one)
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
    /// below epsilon gets shifted (acados default, ocp_nlp_reg_glm.c:69).
    double epsilon = 1e-6;

    /// Shift the (u;x) block of every stage Hessian of `qp` so that its
    /// Gershgorin min-eig estimate reaches at least epsilon. In place.
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

    /// No-op: a pure Hessian shift does not move the dual solution
    /// (ocp_nlp_reg_glm.c:255-258).
    void correct_dual_sol(Qp<P, NH>&, QpSol<P, NH>&) const
    {
    }

private:
    /// Shift the leading nux x nux block of `hess` by alpha * I when its
    /// Gershgorin min-eig estimate tmp is below epsilon
    /// (ocp_nlp_reg_glm.c:230-238).
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

/// No-regularization strategy (acados/ocp_nlp/ocp_nlp_reg_noreg.c): both
/// operations are no-ops.
template <class P, int NH = Eigen::Dynamic>
class NoRegularizer
{
public:
    void regularize(Qp<P, NH>&) const
    {
    }

    void correct_dual_sol(Qp<P, NH>&, QpSol<P, NH>&) const
    {
    }
};

}  // namespace ocp

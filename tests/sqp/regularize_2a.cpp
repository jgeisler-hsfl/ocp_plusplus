// Phase 2 sub-step 2a test: the Hessian regularizers
// (GlmRegularizer / NoRegularizer) from
// include/ocp/solvers/acados/regularize.hpp.
//
// Checks (SQP_PHASE2_PLAN.md sec. 2a, ocp_nlp_reg_glm.c:216-258,
// math.c:1145-1170):
//   - an indefinite (u;x) block is shifted by |tmp| + epsilon, where tmp is
//     the (hand-computed) Gershgorin min-eig estimate
//   - a PSD block with min-eig estimate > epsilon is left untouched
//   - a PSD block with 0 <= tmp < epsilon is shifted by epsilon only
//   - the slack diagonal and slack off-diagonals are never touched
//   - first, path AND terminal stages are all regularized
//   - NoRegularizer is a no-op; correct_dual_sol is a no-op (both)

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include "ocp/solvers/acados/regularize.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"
#include "../../examples/mass_spring/mass_spring.hpp"

namespace
{

using namespace ocp;

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

void check_close(double a, double b, const std::string& msg)
{
    if (std::fabs(a - b) > 1e-12)
    {
        std::fprintf(stderr, "FAIL: %s (%.12e vs %.12e)\n", msg.c_str(), a, b);
        ++failures;
    }
}

// Independent Gershgorin min-eig estimate of the leading n x n block
// (re-implemented in the test; cross-checks the hand-computed constants).
template <class H>
double expected_gershgorin(const H& hess, int n)
{
    double tmp = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i)
    {
        double r_i = 0.0;
        for (int j = 0; j < n; ++j)
        {
            if (i != j)
            {
                r_i += std::fabs(hess(i, j));
            }
        }
        tmp = std::min(tmp, hess(i, i) - r_i);
    }
    return tmp;
}

// Max abs difference between two square matrices of the same extent.
template <class M1, class M2>
double max_diff(const M1& a, const M2& b)
{
    return (a - b).cwiseAbs().maxCoeff();
}

// ---------------------------------------------------------------------
// Case A: indefinite (u;x) block on the first stage (DI, N = 2).
// nux = 3, nvar_first = 4 (one ineq slack). Hand-computed Gershgorin:
// row 0: 1 - 2 = -1; row 1: 1 - 2 = -1; row 2: 1 - 0 = 1  =>  tmp = -1,
// alpha = 1 + epsilon. Also covers: the zero path / term blocks of the
// same Qp get shifted by epsilon only (tmp = 0).
// ---------------------------------------------------------------------
void case_a_indefinite_first()
{
    using DI = DoubleIntegrator;
    using D = QpDim<DI>;
    const std::string p = "2a A: ";
    constexpr int nux = D::nu + D::nx;      // 3
    constexpr int nvar = D::nvar_first;     // 4 (3 (u;x) + 1 slack)
    constexpr int s = nvar - 1;             // slack column

    Qp<DI> qp(2);
    auto& h = qp.first.hess;
    h.setZero();
    h(0, 0) = 1.0;
    h(1, 1) = 1.0;
    h(2, 2) = 1.0;
    h(0, 1) = 2.0;
    h(1, 0) = 2.0;
    h(s, s) = 7.0;         // slack diagonal sentinel
    h(0, s) = 3.3;         // (u;x)-slack off-diagonal sentinels
    h(s, 0) = 3.3;

    check_close(expected_gershgorin(h, nux), -1.0, p + "hand-computed tmp");
    check(qp.path[0].hess.isZero() && qp.term.hess.isZero(),
          p + "zero path / term blocks before");

    GlmRegularizer<DI> reg;
    reg.regularize(qp);

    const double alpha = 1.0 + reg.epsilon;
    check_close(h(0, 0), 1.0 + alpha, p + "diag(0) shifted by |tmp|+eps");
    check_close(h(1, 1), 1.0 + alpha, p + "diag(1)");
    check_close(h(2, 2), 1.0 + alpha, p + "diag(2)");
    check_close(h(0, 1), 2.0, p + "off-diagonal untouched");
    check_close(h(s, s), 7.0, p + "slack diagonal untouched");
    check_close(h(0, s), 3.3, p + "slack off-diagonal untouched");

    // zero path / term blocks: tmp = 0 < epsilon -> shifted by epsilon
    for (int i = 0; i < nux; ++i)
    {
        check_close(qp.path[0].hess(i, i), reg.epsilon,
                    p + "zero path block shifted by eps ("
                        + std::to_string(i) + ")");
    }
    for (int i = 0; i < D::nx; ++i)
    {
        check_close(qp.term.hess(i, i), reg.epsilon,
                    p + "zero term block shifted by eps ("
                        + std::to_string(i) + ")");
    }
    for (int i = 0; i < nux; ++i)
    {
        for (int j = 0; j < nux; ++j)
        {
            if (i != j)
            {
                check_close(qp.path[0].hess(i, j), 0.0,
                            p + "path off-diagonal untouched");
            }
        }
    }
}

// ---------------------------------------------------------------------
// Case B: PSD blocks with min-eig estimate > epsilon are untouched
// (first, path and term), bit-exact.
// ---------------------------------------------------------------------
void case_b_psd_untouched()
{
    using DI = DoubleIntegrator;
    using D = QpDim<DI>;
    const std::string p = "2a B: ";
    constexpr int nux = D::nu + D::nx;

    Qp<DI> qp(2);
    auto fill = [](auto& h, int n)
    {
        h.setZero();
        for (int i = 0; i < n; ++i)
        {
            h(i, i) = 5.0;
        }
        h(0, 1) = 0.1;
        h(1, 0) = 0.1;
    };
    fill(qp.first.hess, nux);
    fill(qp.path[0].hess, nux);
    fill(qp.term.hess, D::nx);

    // Gershgorin: 5 - 0.1 = 4.9 > epsilon on every row of every stage
    auto before_first = qp.first.hess;
    auto before_path = qp.path[0].hess;
    auto before_term = qp.term.hess;

    GlmRegularizer<DI> reg;
    reg.regularize(qp);

    check(max_diff(qp.first.hess, before_first) == 0.0,
          p + "PSD first block untouched");
    check(max_diff(qp.path[0].hess, before_path) == 0.0,
          p + "PSD path block untouched");
    check(max_diff(qp.term.hess, before_term) == 0.0,
          p + "PSD term block untouched");
}

// ---------------------------------------------------------------------
// Case C: terminal (x) block, indefinite (MS, N = 2). Hand-computed
// Gershgorin: rows 0,1 -> 1 - 3 = -2; rows 2..7 -> 1  =>  tmp = -2,
// alpha = 2 + epsilon; the WHOLE block diagonal is shifted.
// ---------------------------------------------------------------------
void case_c_indefinite_term()
{
    using MS = MassSpring;
    using D = QpDim<MS>;
    const std::string p = "2a C: ";
    constexpr int nux = D::nx;  // 8

    Qp<MS> qp(2);
    auto& h = qp.term.hess;
    h.setZero();
    for (int i = 0; i < nux; ++i)
    {
        h(i, i) = 1.0;
    }
    h(0, 1) = 3.0;
    h(1, 0) = 3.0;

    check_close(expected_gershgorin(h, nux), -2.0, p + "hand-computed tmp");

    GlmRegularizer<MS> reg;
    reg.regularize(qp);

    const double alpha = 2.0 + reg.epsilon;
    for (int i = 0; i < nux; ++i)
    {
        check_close(h(i, i), 1.0 + alpha, p + "term diag("
                                             + std::to_string(i) + ")");
    }
    check_close(h(0, 1), 3.0, p + "term off-diagonal untouched");
}

// ---------------------------------------------------------------------
// Case D: 0 <= tmp < epsilon -> shifted by epsilon only (DI terminal,
// block [[1, 1], [1, 1]]: tmp = 0).
// ---------------------------------------------------------------------
void case_d_small_psd_shift()
{
    using DI = DoubleIntegrator;
    using D = QpDim<DI>;
    const std::string p = "2a D: ";

    Qp<DI> qp(2);
    auto& h = qp.term.hess;
    h.setZero();
    h(0, 0) = 1.0;
    h(1, 1) = 1.0;
    h(0, 1) = 1.0;
    h(1, 0) = 1.0;
    check_close(expected_gershgorin(h, D::nx), 0.0, p + "hand-computed tmp");

    GlmRegularizer<DI> reg;
    reg.regularize(qp);

    check_close(h(0, 0), 1.0 + reg.epsilon, p + "tmp >= 0: alpha = epsilon");
    check_close(h(1, 1), 1.0 + reg.epsilon, p + "diag(1)");
    check_close(h(0, 1), 1.0, p + "off-diagonal untouched");
}

// ---------------------------------------------------------------------
// Case E: NoRegularizer is a no-op; correct_dual_sol is a no-op on both
// regularizers.
// ---------------------------------------------------------------------
void case_e_noreg_and_correct_dual()
{
    using DI = DoubleIntegrator;
    using D = QpDim<DI>;
    const std::string p = "2a E: ";
    constexpr int nux = D::nu + D::nx;

    Qp<DI> qp(2);
    auto& h = qp.first.hess;
    h.setZero();
    h(0, 0) = 1.0;
    h(1, 1) = 1.0;
    h(2, 2) = 1.0;
    h(0, 1) = 2.0;
    h(1, 0) = 2.0;
    auto before = h;

    NoRegularizer<DI> noreg;
    noreg.regularize(qp);
    check(max_diff(h, before) == 0.0, p + "NoRegularizer leaves hess alone");

    // fill the QP solution with a recognizable pattern
    QpSol<DI> sol(2);
    sol.ux_first.setConstant(0.5);
    sol.ux_path[0].setConstant(0.5);
    sol.ux_term.setConstant(0.5);
    for (int k = 0; k < 2; ++k)
    {
        sol.pi[k].setConstant(0.5);
    }
    sol.lam_first.setConstant(0.5);
    sol.lam_path[0].setConstant(0.5);
    sol.lam_term.setConstant(0.5);
    sol.t_first.setConstant(0.5);
    sol.t_path[0].setConstant(0.5);
    sol.t_term.setConstant(0.5);

    GlmRegularizer<DI> glm;
    glm.correct_dual_sol(qp, sol);
    noreg.correct_dual_sol(qp, sol);

    check(sol.ux_first.isConstant(0.5) && sol.ux_path[0].isConstant(0.5)
              && sol.ux_term.isConstant(0.5),
          p + "correct_dual_sol: ux untouched");
    check(sol.pi[0].isConstant(0.5) && sol.pi[1].isConstant(0.5),
          p + "correct_dual_sol: pi untouched");
    check(sol.lam_first.isConstant(0.5) && sol.lam_path[0].isConstant(0.5)
              && sol.lam_term.isConstant(0.5)
              && sol.t_first.isConstant(0.5) && sol.t_path[0].isConstant(0.5)
              && sol.t_term.isConstant(0.5),
          p + "correct_dual_sol: lam / t untouched");

    // GLM default epsilon
    check(GlmRegularizer<DI>().epsilon == 1e-6, p + "default epsilon = 1e-6");
    (void)nux;
}

}  // namespace

int run_regularize_2a_tests()
{
    failures = 0;
    case_a_indefinite_first();
    case_b_psd_untouched();
    case_c_indefinite_term();
    case_d_small_psd_shift();
    case_e_noreg_and_correct_dual();

    if (failures == 0)
    {
        std::printf("All regularize (2a) checks passed.\n");
    }
    return failures;
}

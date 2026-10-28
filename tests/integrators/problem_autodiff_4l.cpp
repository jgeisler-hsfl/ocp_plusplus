// Phase 4l: verify the AutoDiffProblem CRTP mix-in (autodiff_problem.hpp) --
// the problem-level cost / nonlinear-constraint derivative entries derived
// by forward-mode AD from the scalar-templated value functions alone.
//
//   FullValues    shared scalar-templated value set (stage cost L, stage
//                  inequalities g, stage equality e, terminal cost h,
//                  terminal inequality gt, terminal equality et); the
//                  hand-derived analytic references below and central
//                  finite differences cross-check the AD results.
//   ProbFull      FullValues + AutoDiffProblem (WithConstrHess = true);
//                  all 12 derivative entries come from AD.
//   ProbMix       FullValues + AutoDiffProblem with two entries
//                  hand-written (marker values) to prove per-function
//                  override precedence.
//   ProbGn        FullValues + AutoDiffProblem (WithConstrHess = false):
//                  the constraint *_value_jac_hess entries are suppressed,
//                  the cost entries are unaffected.
//   SqpAd / SqpHand
//                 continuous double-integrator OCP solved end-to-end by
//                 AcadosSolver; the same problem with the derivatives from
//                 AD (SqpAd) vs hand-written analytic (SqpHand) must yield
//                 the same solution.
//
// Checks:
//   * trait detection: the scalar-templated value functions are
//     dual-evaluable; a double-only one is not (guards the return-type
//     probe against Eigen's converting constructor);
//   * unit level: every AD entry vs the hand-derived analytic reference
//     (value + gradient/Jacobian + multiplier-contracted Hessian); the
//     stage/terminal cost gradients + Hessians and the stage/terminal
//     inequality contracted Hessians additionally vs central finite
//     differences;
//   * override precedence: hand-written entries beat the AD fallback, the
//     AD entries for the remaining functions still work;
//   * WithConstrHess = false: provides_constr_hess trait, cost AD intact;
//   * end-to-end: SqpAd == SqpHand (status, NLP residuals, cost,
//     trajectories).

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/autodiff_problem.hpp"
#include "ocp/integrators/continuous_problem.hpp"
#include "ocp/integrators/rk_explicit.hpp"
#include "ocp/solvers/acados/acados.hpp"

namespace
{

using namespace ocp;

int failures = 0;

void check(bool ok, const char* msg)
{
    if (!ok)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

template <typename Vec>
bool vec_close(const Vec& a, const Vec& b, double tol)
{
    for (int i = 0; i < a.size(); ++i)
    {
        if (std::abs(a(i) - b(i)) > tol)
            return false;
    }
    return true;
}

template <typename Mat>
bool mat_close(const Mat& a, const Mat& b, double tol)
{
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j)
            if (std::abs(a(i, j) - b(i, j)) > tol)
                return false;
    return true;
}

// ---------------------------------------------------------------------------
//  FullDims: all six value functions active, no boxes, no soft rows
// ---------------------------------------------------------------------------

struct FullDims
{
    static constexpr int nx = 2, nu = 1;
    static constexpr int ng = 2, ne = 1, nl = 0;
    static constexpr int ng_t = 1, ne_t = 1, nl_t = 0;
    static constexpr bool fixed_initial_state = true;
    static constexpr bool has_dynamics_hess = true;
    static constexpr bool has_constr_hess = true;
    static constexpr std::array<int, 0> state_box_idx = {};
    static constexpr std::array<int, 0> control_box_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_idx = {};
    static constexpr std::array<int, 0> ineq_soft_idx = {};
    static constexpr std::array<int, 0> eq_soft_idx = {};
    static constexpr std::array<int, 0> lin_soft_idx = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {};
    static constexpr std::array<int, 0> state_box_soft_idx = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

using PF = Problem<FullDims>;

// ---------------------------------------------------------------------------
//  Value set (polynomial, so dual arithmetic stays exact)
//
//    L(x,u)  = 0.5 x0^3 + 0.5 x0 x1 + 0.5 u^2 x1 + 0.1 u
//    g(x,u)  = [ x0^2 + x1^2 - 2 ;  u x1 - 0.5 ]   (g <= 0)
//    e(x,u)  = x0 u + x0 x1^2                                (e == 0)
//    h(x)    = 0.5 (x0 + x1^2)^2
//    gt(x)   = x0 x1 - 0.5                                    (gt <= 0)
//    et(x)   = x1^2 - x0 + 0.3                               (et == 0)
// ---------------------------------------------------------------------------

struct FullValues
{
    template <class XExpr, class UExpr>
    auto stage_cost_value(int, const XExpr& x, const UExpr& u) const
        -> typename XExpr::Scalar
    {
        using D = typename XExpr::Scalar;
        return D(0.5) * x(0) * x(0) * x(0)
             + D(0.5) * x(0) * x(1)
             + D(0.5) * u(0) * u(0) * x(1)
             + D(0.1) * u(0);
    }

    template <class XExpr, class UExpr>
    auto stage_inequality_value(int, const XExpr& x,
                                const UExpr& u) const
        -> Eigen::Matrix<typename XExpr::Scalar, 2, 1>
    {
        using D = typename XExpr::Scalar;
        Eigen::Matrix<D, 2, 1> g;
        g(0) = x(0) * x(0) + x(1) * x(1) - D(2.0);
        g(1) = u(0) * x(1) - D(0.5);
        return g;
    }

    template <class XExpr, class UExpr>
    auto stage_equality_value(int, const XExpr& x,
                              const UExpr& u) const
        -> Eigen::Matrix<typename XExpr::Scalar, 1, 1>
    {
        using D = typename XExpr::Scalar;
        Eigen::Matrix<D, 1, 1> e;
        e(0) = x(0) * u(0) + x(0) * x(1) * x(1);
        return e;
    }

    template <class XExpr>
    auto terminal_cost_value(const XExpr& x) const
        -> typename XExpr::Scalar
    {
        using D = typename XExpr::Scalar;
        const D s = x(0) + x(1) * x(1);
        return D(0.5) * s * s;
    }

    template <class XExpr>
    auto terminal_inequality_value(const XExpr& x) const
        -> Eigen::Matrix<typename XExpr::Scalar, 1, 1>
    {
        using D = typename XExpr::Scalar;
        Eigen::Matrix<D, 1, 1> g;
        g(0) = x(0) * x(1) - D(0.5);
        return g;
    }

    template <class XExpr>
    auto terminal_equality_value(const XExpr& x) const
        -> Eigen::Matrix<typename XExpr::Scalar, 1, 1>
    {
        using D = typename XExpr::Scalar;
        Eigen::Matrix<D, 1, 1> e;
        e(0) = x(1) * x(1) - x(0) + D(0.3);
        return e;
    }
};

// ---------------------------------------------------------------------------
//  Hand-derived analytic references (double)
// ---------------------------------------------------------------------------

double ref_L(double x0, double x1, double u)
{
    return 0.5 * x0 * x0 * x0 + 0.5 * x0 * x1
         + 0.5 * u * u * x1 + 0.1 * u;
}

Eigen::Matrix<double, 3, 1> ref_L_grad(double x0, double x1, double u)
{
    Eigen::Matrix<double, 3, 1> g;
    g(0) = 1.5 * x0 * x0 + 0.5 * x1;
    g(1) = 0.5 * x0 + 0.5 * u * u;
    g(2) = u * x1 + 0.1;
    return g;
}

Eigen::Matrix<double, 3, 3> ref_L_hess(double x0, double x1, double u)
{
    Eigen::Matrix<double, 3, 3> H;
    H(0, 0) = 3.0 * x0;  H(0, 1) = 0.5;    H(0, 2) = 0.0;
    H(1, 0) = 0.5;       H(1, 1) = 0.0;    H(1, 2) = u;
    H(2, 0) = 0.0;       H(2, 1) = u;      H(2, 2) = x1;
    return H;
}

Eigen::Matrix<double, 2, 1> ref_g(double x0, double x1, double u)
{
    return Eigen::Matrix<double, 2, 1>(
        x0 * x0 + x1 * x1 - 2.0, u * x1 - 0.5);
}

Eigen::Matrix<double, 2, 2> ref_g_dx(double x0, double x1, double u)
{
    Eigen::Matrix<double, 2, 2> J;
    J(0, 0) = 2.0 * x0;  J(0, 1) = 2.0 * x1;
    J(1, 0) = 0.0;       J(1, 1) = u;
    return J;
}

Eigen::Matrix<double, 2, 1> ref_g_du(double x1)
{
    return Eigen::Matrix<double, 2, 1>(0.0, x1);
}

// H = lam1 * Hess(g1) + lam2 * Hess(g2),  [x; u] layout.
//   Hess(g1) = diag(2, 2, 0),  Hess(g2): (x1,u) cross = 1
Eigen::Matrix<double, 3, 3> ref_g_hess_c(double l1, double l2)
{
    Eigen::Matrix<double, 3, 3> H;
    H.setZero();
    H(0, 0) = 2.0 * l1;
    H(1, 1) = 2.0 * l1;
    H(1, 2) = l2;
    H(2, 1) = l2;
    return H;
}

double ref_e(double x0, double x1, double u)
{
    return x0 * u + x0 * x1 * x1;
}

Eigen::Matrix<double, 1, 2> ref_e_dxd(double x0, double x1, double u)
{
    return Eigen::Matrix<double, 1, 2>(u + x1 * x1, 2.0 * x0 * x1);
}

// l * Hess(e):  d2e/dx0du = 1,  d2e/dx1^2 = 2 x0
Eigen::Matrix<double, 3, 3> ref_e_hess_c(double x0, double x1, double l)
{
    Eigen::Matrix<double, 3, 3> H;
    H.setZero();
    H(0, 1) = 2.0 * x1 * l;
    H(1, 0) = 2.0 * x1 * l;
    H(0, 2) = l;
    H(2, 0) = l;
    H(1, 1) = 2.0 * x0 * l;
    return H;
}

double ref_h(double x0, double x1)
{
    const double s = x0 + x1 * x1;
    return 0.5 * s * s;
}

Eigen::Matrix<double, 2, 1> ref_h_grad(double x0, double x1)
{
    const double s = x0 + x1 * x1;
    return Eigen::Matrix<double, 2, 1>(s, 2.0 * x1 * s);
}

Eigen::Matrix<double, 2, 2> ref_h_hess(double x0, double x1)
{
    const double s = x0 + x1 * x1;
    Eigen::Matrix<double, 2, 2> H;
    H(0, 0) = 1.0;
    H(0, 1) = 2.0 * x1;
    H(1, 0) = 2.0 * x1;
    H(1, 1) = 2.0 * s + 4.0 * x1 * x1;
    return H;
}

double ref_gt(double x0, double x1)
{
    return x0 * x1 - 0.5;
}

Eigen::Matrix<double, 1, 2> ref_gt_dx(double x0, double x1)
{
    return Eigen::Matrix<double, 1, 2>(x1, x0);
}

// l * Hess(gt):  mixed d2/dx0dx1 = 1
Eigen::Matrix<double, 2, 2> ref_gt_hess_c(double l)
{
    Eigen::Matrix<double, 2, 2> H;
    H.setZero();
    H(0, 1) = l;
    H(1, 0) = l;
    return H;
}

double ref_et(double x0, double x1)
{
    return x1 * x1 - x0 + 0.3;
}

Eigen::Matrix<double, 1, 2> ref_et_dx(double, double x1)
{
    return Eigen::Matrix<double, 1, 2>(-1.0, 2.0 * x1);
}

// l * Hess(et):  d2/dx1^2 = 2
Eigen::Matrix<double, 2, 2> ref_et_hess_c(double l)
{
    Eigen::Matrix<double, 2, 2> H;
    H.setZero();
    H(1, 1) = 2.0 * l;
    return H;
}

// ---------------------------------------------------------------------------
//  Central finite differences (independent cross-check)
// ---------------------------------------------------------------------------

template <int N, class F>
Eigen::Matrix<double, N, 1> fd_grad(const Eigen::Matrix<double, N, 1>& z0,
                                    const F& f, double h = 1e-5)
{
    Eigen::Matrix<double, N, 1> g;
    for (int j = 0; j < N; ++j)
    {
        Eigen::Matrix<double, N, 1> zp = z0;
        Eigen::Matrix<double, N, 1> zm = z0;
        zp(j) += h;
        zm(j) -= h;
        g(j) = (f(zp) - f(zm)) / (2.0 * h);
    }
    return g;
}

template <int N, class F>
Eigen::Matrix<double, N, N> fd_hess(const Eigen::Matrix<double, N, 1>& z0,
                                    const F& f, double h = 1e-4)
{
    Eigen::Matrix<double, N, N> H;
    for (int a = 0; a < N; ++a)
    {
        for (int b = a; b < N; ++b)
        {
            if (a == b)
            {
                Eigen::Matrix<double, N, 1> zp = z0;
                Eigen::Matrix<double, N, 1> zm = z0;
                zp(a) += h;
                zm(a) -= h;
                H(a, a) = (f(zp) - 2.0 * f(z0) + f(zm)) / (h * h);
            }
            else
            {
                Eigen::Matrix<double, N, 1> p11 = z0, p1m = z0;
                Eigen::Matrix<double, N, 1> pm1 = z0, pMM = z0;
                p11(a) += h; p11(b) += h;
                p1m(a) += h; p1m(b) -= h;
                pm1(a) -= h; pm1(b) += h;
                pMM(a) -= h; pMM(b) -= h;
                H(a, b) = (f(p11) - f(p1m) - f(pm1) + f(pMM))
                          / (4.0 * h * h);
            }
            H(b, a) = H(a, b);
        }
    }
    return H;
}

// ---------------------------------------------------------------------------
//  Problem classes (standalone, no dynamics)
// ---------------------------------------------------------------------------

/// All 12 derivative entries from AD.
struct ProbFull : FullValues, AutoDiffProblem<ProbFull, FullDims>
{
    using AutoDiffProblem<ProbFull, FullDims>::stage_cost_value_grad;
    using AutoDiffProblem<ProbFull, FullDims>::stage_cost_value_grad_hess;
    using AutoDiffProblem<ProbFull, FullDims>::stage_inequality_value_jac;
    using AutoDiffProblem<ProbFull, FullDims>::stage_inequality_value_jac_hess;
    using AutoDiffProblem<ProbFull, FullDims>::stage_equality_value_jac;
    using AutoDiffProblem<ProbFull, FullDims>::stage_equality_value_jac_hess;
    using AutoDiffProblem<ProbFull, FullDims>::terminal_cost_value_grad;
    using AutoDiffProblem<ProbFull, FullDims>::terminal_cost_value_grad_hess;
    using AutoDiffProblem<ProbFull, FullDims>::terminal_inequality_value_jac;
    using AutoDiffProblem<ProbFull, FullDims>::
        terminal_inequality_value_jac_hess;
    using AutoDiffProblem<ProbFull, FullDims>::terminal_equality_value_jac;
    using AutoDiffProblem<ProbFull, FullDims>::
        terminal_equality_value_jac_hess;
};

/// Two hand-written entries (marker values) + AD for the rest.
struct ProbMix : FullValues, AutoDiffProblem<ProbMix, FullDims>
{
    using PFmix = Problem<FullDims>;

    void stage_inequality_value_jac(int k, const PFmix::state_t& x,
                                    const PFmix::control_t& u,
                                    PFmix::ineq_t& g,
                                    PFmix::ineq_dg_dx_t& g_dx,
                                    PFmix::ineq_dg_du_t& g_du) const
    {
        g = stage_inequality_value(k, x, u);
        g_dx.setConstant(7.0);
        g_du.setConstant(7.0);
    }

    void terminal_cost_value_grad_hess(const PFmix::state_t& x,
                                       double& value,
                                       PFmix::term_grad_t& grad,
                                       PFmix::term_hess_t& hess) const
    {
        value = terminal_cost_value(x);
        grad.setZero();
        hess.setConstant(3.0);
    }

    using AutoDiffProblem<ProbMix, FullDims>::stage_cost_value_grad;
    using AutoDiffProblem<ProbMix, FullDims>::stage_cost_value_grad_hess;
    using AutoDiffProblem<ProbMix, FullDims>::stage_equality_value_jac;
    using AutoDiffProblem<ProbMix, FullDims>::stage_equality_value_jac_hess;
    using AutoDiffProblem<ProbMix, FullDims>::terminal_cost_value_grad;
    using AutoDiffProblem<ProbMix, FullDims>::terminal_inequality_value_jac;
    using AutoDiffProblem<ProbMix, FullDims>::
        terminal_inequality_value_jac_hess;
    using AutoDiffProblem<ProbMix, FullDims>::terminal_equality_value_jac;
    using AutoDiffProblem<ProbMix, FullDims>::
        terminal_equality_value_jac_hess;
};

/// Constraint Hessian entries suppressed (WithConstrHess = false).
struct ProbGn : FullValues, AutoDiffProblem<ProbGn, FullDims, false>
{
    using AutoDiffProblem<ProbGn, FullDims, false>::stage_cost_value_grad;
    using AutoDiffProblem<ProbGn, FullDims, false>::
        stage_cost_value_grad_hess;
    using AutoDiffProblem<ProbGn, FullDims, false>::
        stage_inequality_value_jac;
    using AutoDiffProblem<ProbGn, FullDims, false>::stage_equality_value_jac;
    using AutoDiffProblem<ProbGn, FullDims, false>::terminal_cost_value_grad;
    using AutoDiffProblem<ProbGn, FullDims, false>::
        terminal_cost_value_grad_hess;
    using AutoDiffProblem<ProbGn, FullDims, false>::
        terminal_inequality_value_jac;
    using AutoDiffProblem<ProbGn, FullDims, false>::
        terminal_equality_value_jac;
};

/// Double-only value function: must NOT look dual-evaluable.
struct ProbConcrete
{
    double stage_cost_value(int, const Eigen::Matrix<double, 2, 1>& x,
                            const Eigen::Matrix<double, 1, 1>& u) const
    {
        return 0.5 * x(0) * x(0) + 0.1 * u(0);
    }
};

// ---------------------------------------------------------------------------
//  Trait static_asserts
// ---------------------------------------------------------------------------

using Dual1S = duals::dual<double, 3>;   // nInS = 2 + 1
using Dual3S = duals::dual<Dual1S, 3>;
using Dual1T = duals::dual<double, 2>;   // nInT = 2
using Dual3T = duals::dual<Dual1T, 2>;

static_assert(stage_cost_value_evaluable_at<ProbFull, FullDims, Dual1S>,
              "templated stage_cost_value must be Dual1-evaluable");
static_assert(stage_cost_value_evaluable_at<ProbFull, FullDims, Dual3S>,
              "templated stage_cost_value must be Dual3-evaluable");
static_assert(
    stage_inequality_value_evaluable_at<ProbFull, FullDims, Dual3S>,
    "templated stage_inequality_value must be Dual3-evaluable");
static_assert(
    stage_equality_value_evaluable_at<ProbFull, FullDims, Dual1S>,
    "templated stage_equality_value must be Dual1-evaluable");
static_assert(
    terminal_cost_value_evaluable_at<ProbFull, FullDims, Dual3T>,
    "templated terminal_cost_value must be Dual3-evaluable");
static_assert(
    terminal_inequality_value_evaluable_at<ProbFull, FullDims, Dual1T>,
    "templated terminal_inequality_value must be Dual1-evaluable");
static_assert(
    terminal_equality_value_evaluable_at<ProbFull, FullDims, Dual3T>,
    "templated terminal_equality_value must be Dual3-evaluable");
static_assert(
    !stage_cost_value_evaluable_at<ProbConcrete, FullDims, Dual1S>,
    "double-only stage_cost_value must not look Dual1-evaluable");
static_assert(
    stage_cost_value_evaluable_at<ProbConcrete, FullDims, double>,
    "double-only stage_cost_value is double-evaluable");
static_assert(AutoDiffProblem<ProbFull, FullDims>::provides_constr_hess,
              "WithConstrHess = true must provide the constraint Hessians");
static_assert(
    !AutoDiffProblem<ProbGn, FullDims, false>::provides_constr_hess,
    "WithConstrHess = false must suppress the constraint Hessians");

// ---------------------------------------------------------------------------
//  Unit checks: ProbFull -- all 12 AD entries
// ---------------------------------------------------------------------------

void check_prob_full()
{
    ProbFull p;
    constexpr double TOL = 1e-9;

    struct Pt { double x0, x1, u; };
    const Pt pts[] = { {0.6, -0.4, 0.9}, {-0.3, 1.1, -0.7} };

    for (int pi = 0; pi < 2; ++pi)
    {
        const auto& pt = pts[pi];
        PF::state_t x;
        x << pt.x0, pt.x1;
        PF::control_t u;
        u(0) = pt.u;

        // -- stage cost ------------------------------------------------
        double v;
        PF::stage_grad_t g;
        PF::stage_hess_t H;
        p.stage_cost_value_grad_hess(2, x, u, v, g, H);
        check(std::abs(v - ref_L(pt.x0, pt.x1, pt.u)) < 1e-12,
              "4l full: stage cost value");
        check(vec_close(g, ref_L_grad(pt.x0, pt.x1, pt.u), TOL),
              "4l full: stage cost grad");
        check(mat_close(H, ref_L_hess(pt.x0, pt.x1, pt.u), TOL),
              "4l full: stage cost hess");
        check(std::abs(v - p.stage_cost_value(2, x, u)) < 1e-14,
              "4l full: fused value == value");

        double v1;
        PF::stage_grad_t g1;
        p.stage_cost_value_grad(2, x, u, v1, g1);
        check(vec_close(g1, g, 1e-12),
              "4l full: Dual1 grad == fused grad");

        if (pi == 0)
        {
            Eigen::Matrix<double, 3, 1> z;
            z << pt.x0, pt.x1, pt.u;
            auto Lz = [](const Eigen::Matrix<double, 3, 1>& zz)
            { return ref_L(zz(0), zz(1), zz(2)); };
            check(vec_close(g, fd_grad<3>(z, Lz), 1e-7),
                  "4l full: stage cost grad vs FD");
            check(mat_close(H, fd_hess<3>(z, Lz), 1e-5),
                  "4l full: stage cost hess vs FD");
        }

        // -- stage inequality -------------------------------------------
        PF::ineq_t gv;
        PF::ineq_dg_dx_t gdx;
        PF::ineq_dg_du_t gdu;
        p.stage_inequality_value_jac(1, x, u, gv, gdx, gdu);
        check(vec_close(gv, ref_g(pt.x0, pt.x1, pt.u), TOL),
              "4l full: stage ineq value");
        check(mat_close(gdx, ref_g_dx(pt.x0, pt.x1, pt.u), TOL),
              "4l full: stage ineq dg/dx");
        check(vec_close(gdu, ref_g_du(pt.x1), TOL),
              "4l full: stage ineq dg/du");

        const PF::ineq_t lam(0.7, -0.3);
        PF::ineq_t gv2;
        PF::ineq_dg_dx_t gdx2;
        PF::ineq_dg_du_t gdu2;
        PF::constr_hess_t CH;
        p.stage_inequality_value_jac_hess(1, x, u, lam, gv2, gdx2, gdu2, CH);
        check(vec_close(gv2, ref_g(pt.x0, pt.x1, pt.u), TOL),
              "4l full: stage ineq jac_hess value");
        check(mat_close(gdx2, ref_g_dx(pt.x0, pt.x1, pt.u), TOL),
              "4l full: stage ineq jac_hess dg/dx");
        check(vec_close(gdu2, ref_g_du(pt.x1), TOL),
              "4l full: stage ineq jac_hess dg/du");
        check(mat_close(CH, ref_g_hess_c(0.7, -0.3), TOL),
              "4l full: stage ineq contracted hess");
        if (pi == 0)
        {
            Eigen::Matrix<double, 3, 1> z;
            z << pt.x0, pt.x1, pt.u;
            auto phi = [](const Eigen::Matrix<double, 3, 1>& zz)
            {
                return 0.7 * (zz(0) * zz(0) + zz(1) * zz(1) - 2.0)
                     - 0.3 * (zz(2) * zz(1) - 0.5);
            };
            check(mat_close(CH, fd_hess<3>(z, phi), 1e-5),
                  "4l full: stage ineq contracted hess vs FD");
        }

        // -- stage equality ---------------------------------------------
        PF::eq_t ev;
        PF::eq_de_dx_t edx;
        PF::eq_de_du_t edu;
        p.stage_equality_value_jac(1, x, u, ev, edx, edu);
        check(std::abs(ev(0) - ref_e(pt.x0, pt.x1, pt.u)) < TOL,
              "4l full: stage eq value");
        check(vec_close(edx, ref_e_dxd(pt.x0, pt.x1, pt.u), TOL),
              "4l full: stage eq de/dx");
        check(std::abs(edu(0) - pt.x0) < TOL,
              "4l full: stage eq de/du");

        const PF::eq_t elam(0.4);
        PF::eq_t ev2;
        PF::eq_de_dx_t edx2;
        PF::eq_de_du_t edu2;
        PF::constr_hess_t CH2;
        p.stage_equality_value_jac_hess(1, x, u, elam, ev2, edx2, edu2,
                                        CH2);
        check(std::abs(ev2(0) - ref_e(pt.x0, pt.x1, pt.u)) < TOL,
              "4l full: stage eq jac_hess value");
        check(vec_close(edx2, ref_e_dxd(pt.x0, pt.x1, pt.u), TOL),
              "4l full: stage eq jac_hess de/dx");
        check(mat_close(CH2, ref_e_hess_c(pt.x0, pt.x1, 0.4), TOL),
              "4l full: stage eq contracted hess");
    }

    // -- terminal (two points) -----------------------------------------
    struct TPt { double x0, x1; };
    const TPt tpts[] = { {0.6, -0.4}, {-0.3, 1.1} };

    for (int ti = 0; ti < 2; ++ti)
    {
        const auto& t = tpts[ti];
        PF::state_t x;
        x << t.x0, t.x1;

        // terminal cost
        double hv;
        PF::term_grad_t hg;
        PF::term_hess_t HH;
        p.terminal_cost_value_grad_hess(x, hv, hg, HH);
        check(std::abs(hv - ref_h(t.x0, t.x1)) < 1e-12,
              "4l full: terminal cost value");
        check(vec_close(hg, ref_h_grad(t.x0, t.x1), TOL),
              "4l full: terminal cost grad");
        check(mat_close(HH, ref_h_hess(t.x0, t.x1), TOL),
              "4l full: terminal cost hess");
        check(std::abs(hv - p.terminal_cost_value(x)) < 1e-14,
              "4l full: fused terminal value == value");

        double hv1;
        PF::term_grad_t hg1;
        p.terminal_cost_value_grad(x, hv1, hg1);
        check(vec_close(hg1, hg, 1e-12),
              "4l full: Dual1 terminal grad == fused grad");

        if (ti == 0)
        {
            Eigen::Matrix<double, 2, 1> z;
            z << t.x0, t.x1;
            auto hz = [](const Eigen::Matrix<double, 2, 1>& zz)
            { return ref_h(zz(0), zz(1)); };
            check(vec_close(hg, fd_grad<2>(z, hz), 1e-7),
                  "4l full: terminal cost grad vs FD");
            check(mat_close(HH, fd_hess<2>(z, hz), 1e-5),
                  "4l full: terminal cost hess vs FD");
        }

        // terminal ineq
        PF::ineq_term_t gtv;
        PF::ineq_term_dg_dx_t gtdx;
        p.terminal_inequality_value_jac(x, gtv, gtdx);
        check(std::abs(gtv(0) - ref_gt(t.x0, t.x1)) < TOL,
              "4l full: terminal ineq value");
        check(vec_close(gtdx, ref_gt_dx(t.x0, t.x1), TOL),
              "4l full: terminal ineq dg/dx");

        const PF::ineq_term_t gtlam(0.9);
        PF::ineq_term_t gtv2;
        PF::ineq_term_dg_dx_t gtdx2;
        PF::term_constr_hess_t gth;
        p.terminal_inequality_value_jac_hess(x, gtlam, gtv2, gtdx2, gth);
        check(std::abs(gtv2(0) - ref_gt(t.x0, t.x1)) < TOL,
              "4l full: terminal ineq jac_hess value");
        check(mat_close(gth, ref_gt_hess_c(0.9), TOL),
              "4l full: terminal ineq contracted hess");
        if (ti == 0)
        {
            Eigen::Matrix<double, 2, 1> z;
            z << t.x0, t.x1;
            auto phi = [](const Eigen::Matrix<double, 2, 1>& zz)
            { return 0.9 * (zz(0) * zz(1) - 0.5); };
            check(mat_close(gth, fd_hess<2>(z, phi), 1e-5),
                  "4l full: terminal ineq contracted hess vs FD");
        }

        // terminal eq
        PF::eq_term_t etv;
        PF::eq_term_de_dx_t etdx;
        p.terminal_equality_value_jac(x, etv, etdx);
        check(std::abs(etv(0) - ref_et(t.x0, t.x1)) < TOL,
              "4l full: terminal eq value");
        check(vec_close(etdx, ref_et_dx(t.x0, t.x1), TOL),
              "4l full: terminal eq de/dx");

        const PF::eq_term_t etlam(0.2);
        PF::eq_term_t etv2;
        PF::eq_term_de_dx_t etdx2;
        PF::term_constr_hess_t eth;
        p.terminal_equality_value_jac_hess(x, etlam, etv2, etdx2, eth);
        check(std::abs(etv2(0) - ref_et(t.x0, t.x1)) < TOL,
              "4l full: terminal eq jac_hess value");
        check(mat_close(eth, ref_et_hess_c(0.2), TOL),
              "4l full: terminal eq contracted hess");
    }

    std::printf("  4l full        ProbFull  OK  (12 AD entries, "
                "analytic + FD)\n");
}

// ---------------------------------------------------------------------------
//  Override precedence: ProbMix
// ---------------------------------------------------------------------------

void check_prob_mix()
{
    ProbMix p;
    constexpr double TOL = 1e-9;

    const double x0 = 0.6, x1 = -0.4, u = 0.9;
    PF::state_t x;
    x << x0, x1;
    PF::control_t uk;
    uk(0) = u;

    // Hand-written marker must beat the AD fallback.
    PF::ineq_t g;
    PF::ineq_dg_dx_t gdx;
    PF::ineq_dg_du_t gdu;
    p.stage_inequality_value_jac(2, x, uk, g, gdx, gdu);
    check(vec_close(g, ref_g(x0, x1, u), 1e-12),
          "4l mix: marker jac value correct");
    check(gdx.cwiseAbs().isConstant(7.0) && gdu.cwiseAbs().isConstant(7.0),
          "4l mix: hand-written ineq jac beats AD fallback");

    double hv;
    PF::term_grad_t hg;
    PF::term_hess_t HH;
    p.terminal_cost_value_grad_hess(x, hv, hg, HH);
    check(std::abs(hv - ref_h(x0, x1)) < 1e-12,
          "4l mix: marker hess value correct");
    check(hg.cwiseAbs().maxCoeff() == 0.0, "4l mix: marker grad zero");
    const PF::term_hess_t hess_marker = PF::term_hess_t::Constant(3.0);
    check(mat_close(HH, hess_marker, 0.0),
          "4l mix: hand-written terminal hess beats AD fallback");

    // The AD fallbacks for the remaining entries still work.
    PF::eq_t ev;
    PF::eq_de_dx_t edx;
    PF::eq_de_du_t edu;
    p.stage_equality_value_jac(2, x, uk, ev, edx, edu);
    check(vec_close(edx, ref_e_dxd(x0, x1, u), TOL),
          "4l mix: AD eq jac still active");

    PF::stage_grad_t sg;
    PF::stage_hess_t sh;
    p.stage_cost_value_grad_hess(2, x, uk, hv, sg, sh);
    check(vec_close(sg, ref_L_grad(x0, x1, u), TOL),
          "4l mix: AD stage cost grad still active");
    check(mat_close(sh, ref_L_hess(x0, x1, u), TOL),
          "4l mix: AD stage cost hess still active");

    std::printf("  4l mix         ProbMix   OK  (hand-written beats AD, "
                "rest AD)\n");
}

// ---------------------------------------------------------------------------
//  WithConstrHess = false: ProbGn
// ---------------------------------------------------------------------------

void check_prob_gn()
{
    ProbGn p;
    constexpr double TOL = 1e-9;

    const double x0 = 0.6, x1 = -0.4, u = 0.9;
    PF::state_t x;
    x << x0, x1;
    PF::control_t uk;
    uk(0) = u;

    // Cost entries are unaffected by WithConstrHess.
    double v;
    PF::stage_grad_t g;
    PF::stage_hess_t H;
    p.stage_cost_value_grad_hess(0, x, uk, v, g, H);
    check(vec_close(g, ref_L_grad(x0, x1, u), TOL),
          "4l gn: stage cost grad with WithConstrHess=false");
    check(mat_close(H, ref_L_hess(x0, x1, u), TOL),
          "4l gn: stage cost hess with WithConstrHess=false");

    // Constraint Jacobians are still AD-provided.
    PF::ineq_t gv;
    PF::ineq_dg_dx_t gdx;
    PF::ineq_dg_du_t gdu;
    p.stage_inequality_value_jac(0, x, uk, gv, gdx, gdu);
    check(mat_close(gdx, ref_g_dx(x0, x1, u), TOL),
          "4l gn: stage ineq jac with WithConstrHess=false");

    std::printf("  4l gn          ProbGn    OK  (no constr hess, cost/jac "
                "AD)\n");
}

// ---------------------------------------------------------------------------
//  End-to-end SQP: SqpAd vs SqpHand
//
//  Double integrator (qdot = v, vdot = u), N = 5, h = 0.2,
//  box-bounded control.
//    L(x,u) = 0.5 (x0-1)^2 + 0.5 (x1-1)^2 + 0.1 u^2
//    g1     = 0.5 (x0^2 + x1^2) - 3  <= 0
//    g2     = x1^2 - 1.21            <= 0
//    h(x)   = (x0-1)^2 + 0.5 (x1-1)^2
//    gt(x)  = x0 x1 - 1.0            <= 0
//    et(x)  = x1^2 - x0 - 0.7        = 0
// ---------------------------------------------------------------------------

struct SqpDims
{
    static constexpr int nx = 2, nu = 1;
    static constexpr int ng = 2, ne = 0, nl = 0;
    static constexpr int ng_t = 1, ne_t = 1, nl_t = 0;
    static constexpr bool fixed_initial_state = true;
    static constexpr bool has_dynamics_hess = false;
    static constexpr bool has_constr_hess = true;
    static constexpr std::array<int, 2> state_box_idx = {0, 1};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 2> terminal_state_box_idx = {0, 1};
    static constexpr std::array<int, 0> ineq_soft_idx = {};
    static constexpr std::array<int, 0> eq_soft_idx = {};
    static constexpr std::array<int, 0> lin_soft_idx = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {};
    static constexpr std::array<int, 0> state_box_soft_idx = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

using SP = Problem<SqpDims>;

struct SqpOde
{
    SP::state_t f(const SP::state_t& x, const SP::control_t& u, double) const
    {
        SP::state_t r;
        r(0) = x(1);
        r(1) = u(0);
        return r;
    }

    void jacobian(const SP::state_t&, const SP::control_t&, double,
                  SP::dyn_df_dx_t& df_dx, SP::dyn_df_du_t& df_du) const
    {
        df_dx.setZero();
        df_dx(0, 1) = 1.0;
        df_du.setZero();
        df_du(1, 0) = 1.0;
    }
};

using SqpInteg = ExplicitRkIntegrator<SqpDims, SqpOde, K4Tag>;

/// AD variant: value functions templated, all derivatives from the mix-in.
struct SqpAd : ContinuousProblem<SqpDims, SqpOde, SqpInteg>,
               AutoDiffProblem<SqpAd, SqpDims>
{
    SqpAd(double h)
        : ContinuousProblem<SqpDims, SqpOde, SqpInteg>(SqpOde{}, h)
    {
    }

    SP::state_t initial_state() const
    {
        SP::state_t x0;
        x0 << 1.0, 0.5;
        return x0;
    }

    template <class XExpr, class UExpr>
    auto stage_cost_value(int, const XExpr& x, const UExpr& u) const
        -> typename XExpr::Scalar
    {
        using D = typename XExpr::Scalar;
        return D(0.5) * (x(0) - D(1.0)) * (x(0) - D(1.0))
             + D(0.5) * (x(1) - D(1.0)) * (x(1) - D(1.0))
             + D(0.1) * u(0) * u(0);
    }

    template <class XExpr, class UExpr>
    auto stage_inequality_value(int, const XExpr& x,
                                const UExpr&) const
        -> Eigen::Matrix<typename XExpr::Scalar, 2, 1>
    {
        using D = typename XExpr::Scalar;
        Eigen::Matrix<D, 2, 1> g;
        g(0) = D(0.5) * (x(0) * x(0) + x(1) * x(1)) - D(3.0);
        g(1) = x(1) * x(1) - D(1.21);
        return g;
    }

    template <class XExpr>
    auto terminal_cost_value(const XExpr& x) const
        -> typename XExpr::Scalar
    {
        using D = typename XExpr::Scalar;
        return (x(0) - D(1.0)) * (x(0) - D(1.0))
             + D(0.5) * (x(1) - D(1.0)) * (x(1) - D(1.0));
    }

    template <class XExpr>
    auto terminal_inequality_value(const XExpr& x) const
        -> Eigen::Matrix<typename XExpr::Scalar, 1, 1>
    {
        using D = typename XExpr::Scalar;
        Eigen::Matrix<D, 1, 1> g;
        g(0) = x(0) * x(1) - D(1.0);
        return g;
    }

    template <class XExpr>
    auto terminal_equality_value(const XExpr& x) const
        -> Eigen::Matrix<typename XExpr::Scalar, 1, 1>
    {
        using D = typename XExpr::Scalar;
        Eigen::Matrix<D, 1, 1> e;
        e(0) = x(1) * x(1) - x(0) - D(0.7);
        return e;
    }

    SP::state_box_t stage_state_box_constr(int) const
    {
        SP::state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.penalty = {};
        return spec;
    }

    SP::control_box_t stage_control_box_constr(int) const
    {
        SP::control_box_t spec;
        spec.lo(0) = -10.0;
        spec.hi(0) = 10.0;
        spec.penalty = {};
        return spec;
    }

    SP::term_state_box_t terminal_state_box_constr() const
    {
        SP::term_state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.penalty = {};
        return spec;
    }

    using AutoDiffProblem<SqpAd, SqpDims>::stage_cost_value_grad;
    using AutoDiffProblem<SqpAd, SqpDims>::stage_cost_value_grad_hess;
    using AutoDiffProblem<SqpAd, SqpDims>::stage_inequality_value_jac;
    using AutoDiffProblem<SqpAd, SqpDims>::
        stage_inequality_value_jac_hess;
    using AutoDiffProblem<SqpAd, SqpDims>::terminal_cost_value_grad;
    using AutoDiffProblem<SqpAd, SqpDims>::terminal_cost_value_grad_hess;
    using AutoDiffProblem<SqpAd, SqpDims>::terminal_inequality_value_jac;
    using AutoDiffProblem<SqpAd, SqpDims>::
        terminal_inequality_value_jac_hess;
    using AutoDiffProblem<SqpAd, SqpDims>::terminal_equality_value_jac;
    using AutoDiffProblem<SqpAd, SqpDims>::
        terminal_equality_value_jac_hess;
};

/// Hand-written variant: concrete value functions + analytic derivatives.
struct SqpHand : ContinuousProblem<SqpDims, SqpOde, SqpInteg>
{
    SqpHand(double h)
        : ContinuousProblem<SqpDims, SqpOde, SqpInteg>(SqpOde{}, h)
    {
    }

    SP::state_t initial_state() const
    {
        SP::state_t x0;
        x0 << 1.0, 0.5;
        return x0;
    }

    double stage_cost_value(int, const SP::state_t& x,
                            const SP::control_t& u) const
    {
        return 0.5 * (x(0) - 1.0) * (x(0) - 1.0)
             + 0.5 * (x(1) - 1.0) * (x(1) - 1.0)
             + 0.1 * u(0) * u(0);
    }

    void stage_cost_value_grad(int, const SP::state_t& x,
                               const SP::control_t& u, double& value,
                               SP::stage_grad_t& grad) const
    {
        value = stage_cost_value(0, x, u);
        grad(0) = x(0) - 1.0;
        grad(1) = x(1) - 1.0;
        grad(2) = 0.2 * u(0);
    }

    void stage_cost_value_grad_hess(int, const SP::state_t& x,
                                    const SP::control_t& u, double& value,
                                    SP::stage_grad_t& grad,
                                    SP::stage_hess_t& hess) const
    {
        stage_cost_value_grad(0, x, u, value, grad);
        hess.setZero();
        hess(0, 0) = 1.0;
        hess(1, 1) = 1.0;
        hess(2, 2) = 0.2;
    }

    SP::ineq_t stage_inequality_value(int, const SP::state_t& x,
                                      const SP::control_t&) const
    {
        SP::ineq_t g;
        g(0) = 0.5 * (x(0) * x(0) + x(1) * x(1)) - 3.0;
        g(1) = x(1) * x(1) - 1.21;
        return g;
    }

    void stage_inequality_value_jac(int, const SP::state_t& x,
                                    const SP::control_t&, SP::ineq_t& g,
                                    SP::ineq_dg_dx_t& g_dx,
                                    SP::ineq_dg_du_t& g_du) const
    {
        g(0) = 0.5 * (x(0) * x(0) + x(1) * x(1)) - 3.0;
        g(1) = x(1) * x(1) - 1.21;
        g_dx(0, 0) = x(0);
        g_dx(0, 1) = x(1);
        g_dx(1, 0) = 0.0;
        g_dx(1, 1) = 2.0 * x(1);
        g_du.setZero();
    }

    void stage_inequality_value_jac_hess(
        int, const SP::state_t& x, const SP::control_t&,
        const SP::ineq_t& lam, SP::ineq_t& g, SP::ineq_dg_dx_t& g_dx,
        SP::ineq_dg_du_t& g_du, SP::constr_hess_t& hess) const
    {
        SP::control_t uz;
        uz.setZero();
        stage_inequality_value_jac(0, x, uz, g, g_dx, g_du);
        hess.setZero();
        hess(0, 0) = lam(0);
        hess(1, 1) = 0.5 * lam(0) + 2.0 * lam(1);
    }

    double terminal_cost_value(const SP::state_t& x) const
    {
        return (x(0) - 1.0) * (x(0) - 1.0)
             + 0.5 * (x(1) - 1.0) * (x(1) - 1.0);
    }

    void terminal_cost_value_grad(const SP::state_t& x, double& value,
                                  SP::term_grad_t& grad) const
    {
        value = terminal_cost_value(x);
        grad(0) = 2.0 * (x(0) - 1.0);
        grad(1) = (x(1) - 1.0);
    }

    void terminal_cost_value_grad_hess(const SP::state_t& x, double& value,
                                       SP::term_grad_t& grad,
                                       SP::term_hess_t& hess) const
    {
        terminal_cost_value_grad(x, value, grad);
        hess.setZero();
        hess(0, 0) = 2.0;
        hess(1, 1) = 1.0;
    }

    SP::ineq_term_t terminal_inequality_value(const SP::state_t& x) const
    {
        SP::ineq_term_t g;
        g(0) = x(0) * x(1) - 1.0;
        return g;
    }

    void terminal_inequality_value_jac(const SP::state_t& x,
                                       SP::ineq_term_t& g,
                                       SP::ineq_term_dg_dx_t& g_dx) const
    {
        g(0) = x(0) * x(1) - 1.0;
        g_dx(0, 0) = x(1);
        g_dx(0, 1) = x(0);
    }

    void terminal_inequality_value_jac_hess(
        const SP::state_t& x, const SP::ineq_term_t& lam,
        SP::ineq_term_t& g, SP::ineq_term_dg_dx_t& g_dx,
        SP::term_constr_hess_t& hess) const
    {
        terminal_inequality_value_jac(x, g, g_dx);
        hess.setZero();
        hess(0, 1) = lam(0);
        hess(1, 0) = lam(0);
    }

    SP::eq_term_t terminal_equality_value(const SP::state_t& x) const
    {
        SP::eq_term_t e;
        e(0) = x(1) * x(1) - x(0) - 0.7;
        return e;
    }

    void terminal_equality_value_jac(const SP::state_t& x,
                                     SP::eq_term_t& e,
                                     SP::eq_term_de_dx_t& e_dx) const
    {
        e(0) = x(1) * x(1) - x(0) - 0.7;
        e_dx(0, 0) = -1.0;
        e_dx(0, 1) = 2.0 * x(1);
    }

    void terminal_equality_value_jac_hess(
        const SP::state_t& x, const SP::eq_term_t& lam, SP::eq_term_t& e,
        SP::eq_term_de_dx_t& e_dx, SP::term_constr_hess_t& hess) const
    {
        terminal_equality_value_jac(x, e, e_dx);
        hess.setZero();
        hess(1, 1) = 2.0 * lam(0);
    }

    SP::state_box_t stage_state_box_constr(int) const
    {
        SP::state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.penalty = {};
        return spec;
    }

    SP::control_box_t stage_control_box_constr(int) const
    {
        SP::control_box_t spec;
        spec.lo(0) = -10.0;
        spec.hi(0) = 10.0;
        spec.penalty = {};
        return spec;
    }

    SP::term_state_box_t terminal_state_box_constr() const
    {
        SP::term_state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.penalty = {};
        return spec;
    }
};

template <class Prob>
void warm_start_sqp(const Prob& p, Solution<Prob>& sol)
{
    sol.x[0] = p.initial_state();
    for (int k = 0; k < sol.N; ++k)
    {
        SP::control_t uk;
        uk(0) = 0.5;
        sol.u[k] = uk;
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }
}

template <class Prob>
void solve_sqp(const char* label, const Prob& p, Solution<Prob>& sol)
{
    AcadosOptions opts;
    AcadosSolver<Prob> solver(opts);
    const Status st = solver.solve(p, sol);
    check(st == Status::kSolved,
          (std::string(label) + ": kSolved").c_str());
    const NlpResiduals res = compute_nlp_residuals(p, sol);
    check(res.res_stat < 1e-6,
          (std::string(label) + ": res_stat").c_str());
    check(res.res_eq < 1e-6,
          (std::string(label) + ": res_eq").c_str());
    check(res.res_ineq < 1e-6,
          (std::string(label) + ": res_ineq").c_str());
    check(res.res_comp < 1e-6,
          (std::string(label) + ": res_comp").c_str());
}

void check_sqp()
{
    constexpr int N = 5;
    const double h = 0.2;

    SqpAd p_ad(h);
    SqpHand p_hand(h);

    Solution<SqpAd> sol_ad(N);
    Solution<SqpHand> sol_hand(N);
    warm_start_sqp(p_ad, sol_ad);
    warm_start_sqp(p_hand, sol_hand);

    solve_sqp("4l sqp AD  ", p_ad, sol_ad);
    solve_sqp("4l sqp hand", p_hand, sol_hand);

    double dx = 0.0;
    double du = 0.0;
    for (int k = 0; k <= N; ++k)
    {
        dx = std::max(dx,
                      (sol_ad.x[k] - sol_hand.x[k]).cwiseAbs().maxCoeff());
    }
    for (int k = 0; k < N; ++k)
    {
        du = std::max(du,
                      (sol_ad.u[k] - sol_hand.u[k]).cwiseAbs().maxCoeff());
    }
    check(std::abs(sol_ad.cost_value - sol_hand.cost_value) < 1e-8,
          "4l sqp: cost AD == hand");
    check(dx < 1e-6, "4l sqp: states AD == hand");
    check(du < 1e-6, "4l sqp: controls AD == hand");

    std::printf("  4l sqp         SqpAd/SqpHand OK  (cost=%.6f, maxdx=%.1e, "
                "maxdu=%.1e)\n",
                sol_ad.cost_value, dx, du);
}

}  // namespace

int run_problem_autodiff_4l_tests()
{
    check_prob_full();
    check_prob_mix();
    check_prob_gn();
    check_sqp();

    if (failures != 0)
    {
        std::fprintf(stderr, "[4l] %d check(s) failed.\n", failures);
    }
    return failures;
}

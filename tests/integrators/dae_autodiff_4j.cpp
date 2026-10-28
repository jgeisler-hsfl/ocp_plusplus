// Phase 4j: verify the AutoDiffDae CRTP base (dae_autodiff.hpp) through
// ImplicitRkIntegrator, the WithHess Hessian-model switch, and the explicit
// integrator's AD path from a plain scalar-templated ODE `f`.
//
//   OdeXsqURef   hand-written f / jacobian / hess_prod for
//                xdot = x.*x * u + x + [sin t; cos t] — exact reference and
//                the finite-difference ground truth.
//   OdeXsqUAd    plain ODE struct with ONLY the scalar-templated `f` —
//                drives the explicit integrator's dual-evaluability path.
//   OdeXsqUAdNoHess  same ODE with `ad_hess_prod = false` — the
//                compile-time opt-out of the integrator's nested-dual (Dual2)
//                HVP pass (paired with a has_dynamics_hess = false Dims).
//   DaeXsqUAd    AutoDiffDae (WithHess = true) with the residual
//                F = xdot - f — the AD-under-test (full-Newton Hessian).
//   DaeXsqUGn    AutoDiffDae (WithHess = false) — same residual, F_hess_prod
//                suppressed, so the integrator (and the SQP) fall back to
//                the Gauss-Newton Hessian model.
//
// Checks:
//   * trait detection: the AD DAE exposes F_hess_prod iff WithHess; the
//     templated ODE f is dual-evaluable while the hand-written one is not
//     (guards the return-type probe against Eigen's converting constructor);
//   * DAE level: F_jac / F_hess_prod of AutoDiffDae vs the hand-derived
//     reference and central FD over z = [x; xdot; u];
//   * integrator level (ImplicitRk Radau IIA 2): value_jac / hess_prod of
//     the composed map vs central FD; the GN variant returns zero HVPs with
//     an unchanged Jacobian; cross-backend: the hand-written ODE through
//     OdeToDae matches the AD DAE (guards the AD glue end-to-end);
//   * problem level (ContinuousProblem): the dynamics Hessian assembly is
//     FD-correct for WithHess = true and identically zero for WithHess =
//     false (the SQP's Gauss-Newton fallback);
//   * explicit integrator (K4, K4 x2) over the templated-f ODE: fused
//     value_jac / hess_prod / value_jac_hess_prod vs central FD and the
//     forward-cache behavior.

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/integrators/dae_autodiff.hpp"
#include "ocp/integrators/continuous_problem.hpp"
#include "ocp/integrators/rk_explicit.hpp"
#include "ocp/integrators/rk_implicit.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

namespace
{

using Dims = DoubleIntegratorDims;  // nx = 2, nu = 1
using P = ocp::Problem<Dims>;
using state_t = P::state_t;
using control_t = P::control_t;
using df_dx_t = P::dyn_df_dx_t;
using df_du_t = P::dyn_df_du_t;

constexpr int nx = Dims::nx;
constexpr int nu = Dims::nu;
constexpr int nIn = nx + nu;
constexpr int nInDae = 2 * nx + nu;  // z = [x; xdot; u]
constexpr double t_k = 0.37;  // OCP-interval start used by all checks

using z_t = Eigen::Matrix<double, nIn, 1>;
using zdae_t = Eigen::Matrix<double, nInDae, 1>;
using H_t = Eigen::Matrix<double, nIn, nIn>;
using Hdae_t = Eigen::Matrix<double, nInDae, nInDae>;
using Dual1 = duals::dual<double, nIn>;
using Dual2 = duals::dual<Dual1, 1>;

int failures = 0;

void check(bool cond, const char* msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "  FAIL: %s\n", msg);
        ++failures;
    }
}

// ---------------------------------------------------------------------------
//  ODE models:  xdot = x.*x * u + x + [sin t; cos t]
// ---------------------------------------------------------------------------

/// Hand-written reference (f, jacobian, hess_prod) — exact ground truth.
/// The additive [sin t; cos t] term makes the value time-varying while
/// leaving the (x, u)-derivative references time-invariant.
struct OdeXsqURef
{
    state_t f(const state_t& x, const control_t& u, double t) const
    {
        return x.cwiseProduct(x) * u(0) + x + state_t(std::sin(t), std::cos(t));
    }

    void jacobian(const state_t& x, const control_t& u, double, df_dx_t& df_dx,
                  df_du_t& df_du) const
    {
        // df/dx:  d(x_i^2 u)/dx_i = 2 x_i u (diagonal);  d(x)/dx = I.
        df_dx = df_dx_t::Identity();
        for (int i = 0; i < nx; ++i)
            df_dx(i, i) += 2.0 * x(i) * u(0);
        // df/du:  d(x^2 u)/du = x^2.
        df_du.col(0) = x.cwiseProduct(x);
    }

    void hess_prod(const state_t& x, const control_t& u, double,
                   const state_t& w, const state_t& v_x, const control_t& v_u,
                   state_t& hv_x, control_t& hv_u) const
    {
        // g = w' f,  f_i = x_i^2 u + x_i + c_i(t).
        //   d2f_i/dx_j^2   = 2 u delta_ij    -> (hv_x)_j += 2 u w_j v_x(j)
        //   d2f_i/(dx_j du) = 2 x_i delta_ij  -> (hv_x)_j += 2 w_j x_j v_u(0)
        //   d2f_i/du^2      = 0              (f_i is linear in u)
        //   d2f_i/(du dx_i) = 2 x_i          -> hv_u += 2 sum_i w_i x_i v_x_i
        for (int j = 0; j < nx; ++j)
            hv_x(j) = 2.0 * u(0) * w(j) * v_x(j) + 2.0 * w(j) * x(j) * v_u(0);
        double s = 0.0;
        for (int i = 0; i < nx; ++i)
            s += w(i) * x(i) * v_x(i);
        hv_u(0) = 2.0 * s;
    }
};

/// The explicit integrator's AD-under-test: ONLY `f` is written; the AD path
/// is selected by the dual-evaluability probe (no adapter, no derivatives).
struct OdeXsqUAd
{
    template <class D>
    Eigen::Matrix<D, nx, 1> f(const Eigen::Matrix<D, nx, 1>& x,
                              const Eigen::Matrix<D, nu, 1>& u, double t) const
    {
        return x.cwiseProduct(x) * u(0) + x
             + Eigen::Matrix<D, nx, 1>(D(std::sin(t)), D(std::cos(t)));
    }
};

/// Dims variant with the dynamics Hessian model disabled (Gauss-Newton),
/// for the ad_hess_prod opt-out check.
struct GnDims : Dims
{
    static constexpr bool has_dynamics_hess = false;
};

/// Same ODE as OdeXsqUAd but with the integrator's nested-dual (Dual2) HVP
/// pass opted out at compile time; must be paired with
/// Dims::has_dynamics_hess = false.
struct OdeXsqUAdNoHess
{
    static constexpr bool ad_hess_prod = false;

    template <class D>
    Eigen::Matrix<D, nx, 1> f(const Eigen::Matrix<D, nx, 1>& x,
                              const Eigen::Matrix<D, nu, 1>& u, double t) const
    {
        return x.cwiseProduct(x) * u(0) + x
             + Eigen::Matrix<D, nx, 1>(D(std::sin(t)), D(std::cos(t)));
    }
};

/// The DAE AD-under-test (WithHess = true): ONLY the scalar-templated
/// residual is written; F_jac / F_hess_prod come from the AutoDiffDae base.
struct DaeXsqUAd : ocp::AutoDiffDae<DaeXsqUAd, Dims>
{
    template <class D>
    Eigen::Matrix<D, nx, 1> F(const Eigen::Matrix<D, nx, 1>& x,
                              const Eigen::Matrix<D, nx, 1>& xdot,
                              const Eigen::Matrix<D, nu, 1>& u,
                              double t) const
    {
        OdeXsqUAd ode;
        return xdot - ode.f(x, u, t);
    }
};

/// Gauss-Newton variant (WithHess = false): F_hess_prod is suppressed, so
/// dae_supports_hess_prod is false and the integrator / SQP run the
/// Gauss-Newton Hessian model.
struct DaeXsqUGn : ocp::AutoDiffDae<DaeXsqUGn, Dims, false>
{
    template <class D>
    Eigen::Matrix<D, nx, 1> F(const Eigen::Matrix<D, nx, 1>& x,
                              const Eigen::Matrix<D, nx, 1>& xdot,
                              const Eigen::Matrix<D, nu, 1>& u,
                              double t) const
    {
        OdeXsqUAd ode;
        return xdot - ode.f(x, u, t);
    }
};

// ---------------------------------------------------------------------------
//  Traits
// ---------------------------------------------------------------------------

static_assert(ocp::dae_supports_hess_prod<DaeXsqUAd, Dims>,
              "AutoDiffDae WithHess=true must expose F_hess_prod");
static_assert(!ocp::dae_supports_hess_prod<DaeXsqUGn, Dims>,
              "AutoDiffDae WithHess=false must suppress F_hess_prod");
static_assert(ocp::ode_f_evaluable_at<OdeXsqUAd, Dims, Dual1>,
              "the templated f must be evaluable at Dual1");
static_assert(ocp::ode_f_evaluable_at<OdeXsqUAd, Dims, Dual2>,
              "the templated f must be evaluable at Dual2");
static_assert(!ocp::ode_f_evaluable_at<OdeXsqURef, Dims, Dual1>,
              "hand-written double-f must not appear dual-evaluable");
static_assert(!ocp::ode_f_evaluable_at<OdeXsqURef, Dims, Dual2>,
              "hand-written double-f must not appear dual-evaluable");
static_assert(ocp::ode_supports_hess_prod<OdeXsqURef, Dims>,
              "the hand-written reference must expose hess_prod");

static_assert(
    ocp::ExplicitRkIntegrator<Dims, OdeXsqUAd, ocp::K4Tag>::
        supports_value_jac_hess_prod,
    "explicit integrator over a dual-evaluable f must flag the fused HVP");
static_assert(
    !ocp::ExplicitRkIntegrator<Dims, OdeXsqURef, ocp::K4Tag>::
        supports_value_jac_hess_prod,
    "explicit integrator over a hand-written ODE must not");
static_assert(
    !ocp::ImplicitRkIntegrator<Dims, DaeXsqUAd, ocp::RadauIia2Tag>::
        supports_value_jac_hess_prod,
    "implicit integrator has no dual-Newton fused entry");
static_assert(ocp::ode_ad_hess_prod_enabled<OdeXsqUAd>,
              "AD ODE without an ad_hess_prod member keeps the HVP pass");
static_assert(!ocp::ode_ad_hess_prod_enabled<OdeXsqUAdNoHess>,
              "ad_hess_prod = false must disable the AD HVP pass");
static_assert(
    !ocp::ExplicitRkIntegrator<Dims, OdeXsqUAdNoHess, ocp::K4Tag>::
        supports_value_jac_hess_prod,
    "opted-out AD ODE must not flag the fused HVP");

// ---------------------------------------------------------------------------
//  Finite-difference reference: HVP of g(z) = w^T map(z)
// ---------------------------------------------------------------------------

template <class Map>  // Map: (const state_t&, const control_t&) -> state_t
z_t fd_hvp(const state_t& x0, const control_t& u0, const state_t& w,
           const state_t& v_x, const control_t& v_u, Map&& map)
{
    z_t z0;
    z0.head(nx) = x0;
    z0.tail(nu) = u0;
    H_t H;
    const double d = 1e-4;
    for (int a = 0; a < nIn; ++a)
    {
        for (int b = a; b < nIn; ++b)
        {
            if (a == b)
            {
                z_t zp = z0;
                z_t zm = z0;
                zp(a) += d;
                zm(a) -= d;
                H(a, a) = (w.dot(map(zp.head(nx), zp.tail(nu)))
                           - 2.0 * w.dot(map(z0.head(nx), z0.tail(nu)))
                           + w.dot(map(zm.head(nx), zm.tail(nu))))
                          / (d * d);
            }
            else
            {
                z_t p11 = z0;
                z_t p1m = z0;
                z_t pm1 = z0;
                z_t pMM = z0;
                p11(a) += d;
                p11(b) += d;
                p1m(a) += d;
                p1m(b) -= d;
                pm1(a) -= d;
                pm1(b) += d;
                pMM(a) -= d;
                pMM(b) -= d;
                H(a, b) =
                    (w.dot(map(p11.head(nx), p11.tail(nu)))
                     - w.dot(map(p1m.head(nx), p1m.tail(nu)))
                     - w.dot(map(pm1.head(nx), pm1.tail(nu)))
                     + w.dot(map(pMM.head(nx), pMM.tail(nu))))
                    / (4.0 * d * d);
            }
            H(b, a) = H(a, b);
        }
    }
    z_t v;
    v.head(nx) = v_x;
    v.tail(nu) = v_u;
    return H * v;
}

/// HVP of g(z) = w^T F(z) over the DAE argument z = [x; xdot; u].
template <class Map>  // Map: (const state_t&, const state_t&,
                      //       const control_t&) -> state_t
zdae_t fd_dae_hvp(const state_t& x0, const state_t& xdot0, const control_t& u0,
                  const state_t& w, const state_t& v_x, const state_t& v_xdot,
                  const control_t& v_u, Map&& map)
{
    zdae_t z0;
    z0.segment(0, nx) = x0;
    z0.segment(nx, nx) = xdot0;
    z0.tail(nu) = u0;
    Hdae_t H;
    const double d = 1e-4;
    for (int a = 0; a < nInDae; ++a)
    {
        for (int b = a; b < nInDae; ++b)
        {
            auto eval = [&](const zdae_t& z)
            {
                return w.dot(map(z.segment(0, nx), z.segment(nx, nx),
                                 z.tail(nu)));
            };
            if (a == b)
            {
                zdae_t zp = z0;
                zdae_t zm = z0;
                zp(a) += d;
                zm(a) -= d;
                H(a, a) = (eval(zp) - 2.0 * eval(z0) + eval(zm)) / (d * d);
            }
            else
            {
                zdae_t p11 = z0;
                zdae_t p1m = z0;
                zdae_t pm1 = z0;
                zdae_t pMM = z0;
                p11(a) += d;
                p11(b) += d;
                p1m(a) += d;
                p1m(b) -= d;
                pm1(a) -= d;
                pm1(b) += d;
                pMM(a) -= d;
                pMM(b) -= d;
                H(a, b) = (eval(p11) - eval(p1m) - eval(pm1) + eval(pMM))
                          / (4.0 * d * d);
            }
            H(b, a) = H(a, b);
        }
    }
    zdae_t v;
    v.segment(0, nx) = v_x;
    v.segment(nx, nx) = v_xdot;
    v.tail(nu) = v_u;
    return H * v;
}

struct Case
{
    state_t   w;
    state_t   v_x;
    state_t   v_xdot;
    control_t v_u;
};
const Case cases[] = {
    {state_t(0.7, -0.3), state_t(0.4, 0.9), state_t(0.2, 0.1),
     control_t(-0.5)},
    {state_t(1.0, 0.4), state_t(0.1, -0.6), state_t(-0.4, 0.8),
     control_t(0.8)},
    {state_t(0.2, 0.9), state_t(-0.3, 0.5), state_t(0.6, -0.2),
     control_t(0.2)},
};

// ---------------------------------------------------------------------------
//  DAE level: AutoDiffDae entries vs the hand-derived reference and FD
// ---------------------------------------------------------------------------

void check_dae_jacobian()
{
    DaeXsqUAd dae;
    OdeXsqURef ref;
    const state_t x(0.3, -0.7);
    const state_t xdot(1.1, 0.2);
    const control_t u(0.4);

    df_dx_t dFdx_ad, dFxdot_ad;
    df_du_t dFdu_ad;
    dae.F_jac(x, xdot, u, t_k, dFdx_ad, dFxdot_ad, dFdu_ad);

    // F = xdot - f:  dFdx = -dfdx,  dFxdot = I,  dFdu = -dfdu.
    df_dx_t dfdx_ref;
    df_du_t dfdu_ref;
    ref.jacobian(x, u, t_k, dfdx_ref, dfdu_ref);
    check(((dFdx_ad + dfdx_ref)).cwiseAbs().maxCoeff() < 1e-13,
          "DAE F_jac AD vs hand-derived (dFdx)");
    check((dFxdot_ad - df_dx_t::Identity()).cwiseAbs().maxCoeff() < 1e-13,
          "DAE F_jac AD vs hand-derived (dFxdot)");
    check((dFdu_ad + dfdu_ref).cwiseAbs().maxCoeff() < 1e-13,
          "DAE F_jac AD vs hand-derived (dFdu)");

    // Central FD against the residual value pass.
    const double d = 1e-6;
    for (int j = 0; j < nx; ++j)
    {
        state_t e = state_t::Zero();
        e(j) = d;
        const state_t xp = x + e;
        const state_t xm = x - e;
        const state_t fp = dae.F(xp, xdot, u, t_k);
        const state_t fm = dae.F(xm, xdot, u, t_k);
        for (int i = 0; i < nx; ++i)
            check(std::abs(dFdx_ad(i, j) - (fp(i) - fm(i)) / (2.0 * d)) < 1e-6,
                  "DAE F_jac AD vs FD (dFdx)");
    }
    for (int j = 0; j < nx; ++j)
    {
        state_t e = state_t::Zero();
        e(j) = d;
        const state_t xdotp = xdot + e;
        const state_t xdotm = xdot - e;
        const state_t fp = dae.F(x, xdotp, u, t_k);
        const state_t fm = dae.F(x, xdotm, u, t_k);
        for (int i = 0; i < nx; ++i)
            check(
                std::abs(dFxdot_ad(i, j) - (fp(i) - fm(i)) / (2.0 * d)) < 1e-6,
                "DAE F_jac AD vs FD (dFxdot)");
    }
    for (int j = 0; j < nu; ++j)
    {
        control_t e = control_t::Zero();
        e(j) = d;
        const control_t up = u + e;
        const control_t um = u - e;
        const state_t fp = dae.F(x, xdot, up, t_k);
        const state_t fm = dae.F(x, xdot, um, t_k);
        for (int i = 0; i < nx; ++i)
            check(std::abs(dFdu_ad(i, j) - (fp(i) - fm(i)) / (2.0 * d)) < 1e-6,
                  "DAE F_jac AD vs FD (dFdu)");
    }
    std::printf("  4j dae jac    DaeXsqUAd  OK  (exact vs hand-derived, "
                "FD<1e-6)\n");
}

void check_dae_hess_prod()
{
    DaeXsqUAd dae;
    OdeXsqURef ref;
    const state_t x(0.3, -0.7);
    const state_t xdot(1.1, 0.2);
    const control_t u(0.4);

    for (const auto& c : cases)
    {
        state_t hvx_ad, hvxd_ad;
        control_t hvu_ad;
        dae.F_hess_prod(x, xdot, u, t_k, c.w, c.v_x, c.v_xdot, c.v_u, hvx_ad,
                        hvxd_ad, hvu_ad);

        // D2F = -D2f in the (x, u) blocks; F is affine in xdot (hv_xdot = 0).
        state_t hvx_ref, hvxd_ref;
        control_t hvu_ref;
        ref.hess_prod(x, u, t_k, c.w, c.v_x, c.v_u, hvx_ref, hvu_ref);
        check((hvx_ad + hvx_ref).cwiseAbs().maxCoeff() < 1e-12,
              "DAE F_hess_prod AD vs hand-derived (hv_x)");
        check(hvxd_ad.cwiseAbs().maxCoeff() < 1e-12,
              "DAE F_hess_prod AD vs hand-derived (hv_xdot)");
        check((hvu_ad + hvu_ref).cwiseAbs().maxCoeff() < 1e-12,
              "DAE F_hess_prod AD vs hand-derived (hv_u)");

        const zdae_t fd =
            fd_dae_hvp(x, xdot, u, c.w, c.v_x, c.v_xdot, c.v_u,
                       [&](const state_t& xx, const state_t& xxd,
                           const control_t& uu)
                       { return dae.F(xx, xxd, uu, t_k); });
        zdae_t actual;
        actual.segment(0, nx) = hvx_ad;
        actual.segment(nx, nx) = hvxd_ad;
        actual.tail(nu) = hvu_ad;
        const double err = (actual - fd).cwiseAbs().maxCoeff();
        check(err < 1e-5, "DAE F_hess_prod AD vs FD Hessian");
    }
    std::printf("  4j dae hess   DaeXsqUAd  OK  (exact vs hand-derived, "
                "FD<1e-5, 3 cases)\n");
}

// ---------------------------------------------------------------------------
//  Integrator level: composed-map entries vs central FD
// ---------------------------------------------------------------------------

template <class Integ>
void check_integ_value_jac(Integ& integ, const char* name)
{
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    state_t x_next;
    df_dx_t Jx;
    df_du_t Ju;
    integ.value_jac(x, u, t_k, 0.1, x_next, Jx, Ju);

    // The fused value must match the plain value pass.
    state_t x_val;
    integ.value(x, u, t_k, 0.1, x_val);
    check((x_next - x_val).cwiseAbs().maxCoeff() < 1e-12,
          (std::string(name) + " value_jac value == value").c_str());

    const double d = 1e-6;
    for (int j = 0; j < nx; ++j)
    {
        state_t e = state_t::Zero();
        e(j) = d;
        state_t fp, fm;
        integ.value(x + e, u, t_k, 0.1, fp);
        integ.value(x - e, u, t_k, 0.1, fm);
        for (int i = 0; i < nx; ++i)
            check(std::abs(Jx(i, j) - (fp(i) - fm(i)) / (2.0 * d)) < 1e-6,
                  (std::string(name) + " value_jac df/dx vs FD").c_str());
    }
    for (int j = 0; j < nu; ++j)
    {
        control_t e = control_t::Zero();
        e(j) = d;
        state_t fp, fm;
        integ.value(x, u + e, t_k, 0.1, fp);
        integ.value(x, u - e, t_k, 0.1, fm);
        for (int i = 0; i < nx; ++i)
            check(std::abs(Ju(i, j) - (fp(i) - fm(i)) / (2.0 * d)) < 1e-6,
                  (std::string(name) + " value_jac df/du vs FD").c_str());
    }
    std::printf("  4j integ vjac  %s  OK  (value + Jacobian, FD<1e-6)\n", name);
}

template <class Integ>
void check_integ_hess(Integ& integ, const char* name)
{
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    for (const auto& c : cases)
    {
        state_t hvx;
        control_t hvu;
        integ.hess_prod(x, u, t_k, 0.1, c.w, c.v_x, c.v_u, hvx, hvu);
        const z_t fd = fd_hvp(
            x, u, c.w, c.v_x, c.v_u,
            [&](const state_t& xx, const control_t& uu)
            {
                state_t out;
                integ.value(xx, uu, t_k, 0.1, out);
                return out;
            });
        z_t actual;
        actual.head(nx) = hvx;
        actual.tail(nu) = hvu;
        const double err = (actual - fd).cwiseAbs().maxCoeff();
        check(err < 1e-5,
              (std::string(name) + " composed-map hess_prod vs FD").c_str());
    }
    std::printf("  4j integ hess  %s  OK  (FD<1e-5, 3 cases)\n", name);
}

template <class Integ>
void check_integ_value_jac_hess_prod(Integ& integ, const char* name)
{
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    state_t x_val;
    integ.value(x, u, t_k, 0.1, x_val);
    for (const auto& c : cases)
    {
        state_t x_next;
        df_dx_t Jx;
        df_du_t Ju;
        state_t hvx;
        control_t hvu;
        integ.value_jac_hess_prod(x, u, t_k, 0.1, x_next, Jx, Ju, c.w, c.v_x,
                                  c.v_u,
                                  hvx, hvu);
        check((x_next - x_val).cwiseAbs().maxCoeff() < 1e-12,
              (std::string(name) + " value_jac_hess_prod value == value")
                  .c_str());
        const z_t fd = fd_hvp(
            x, u, c.w, c.v_x, c.v_u,
            [&](const state_t& xx, const control_t& uu)
            {
                state_t out;
                integ.value(xx, uu, t_k, 0.1, out);
                return out;
            });
        z_t actual;
        actual.head(nx) = hvx;
        actual.tail(nu) = hvu;
        check((actual - fd).cwiseAbs().maxCoeff() < 1e-5,
              (std::string(name) + " value_jac_hess_prod HVP vs FD").c_str());
        (void)Jx;
        (void)Ju;
    }
    std::printf("  4j integ vjh   %s  OK  (fused triplet, 3 cases)\n", name);
}

/// Gauss-Newton switch: WithHess = false gives zero HVPs with an unchanged
/// Jacobian (the SQP's GN Hessian model).
void check_gn_switch()
{
    DaeXsqUGn dae_gn;
    ocp::ImplicitRkIntegrator<Dims, DaeXsqUGn, ocp::RadauIia2Tag> irk_gn(
        dae_gn, 8);

    const state_t x(0.3, -0.7);
    const control_t u(0.4);

    // value_jac must still be exact (the Newton path is unaffected).
    const double d = 1e-6;
    state_t x_next;
    df_dx_t Jx;
    df_du_t Ju;
    irk_gn.value_jac(x, u, t_k, 0.1, x_next, Jx, Ju);
    for (int j = 0; j < nx; ++j)
    {
        state_t e = state_t::Zero();
        e(j) = d;
        state_t fp, fm;
        irk_gn.value(x + e, u, t_k, 0.1, fp);
        irk_gn.value(x - e, u, t_k, 0.1, fm);
        for (int i = 0; i < nx; ++i)
            check(std::abs(Jx(i, j) - (fp(i) - fm(i)) / (2.0 * d)) < 1e-6,
                  "GN switch value_jac df/dx vs FD");
    }

    // Every HVP is zero: the composed dynamics Hessian is dropped.
    for (const auto& c : cases)
    {
        state_t hvx;
        control_t hvu;
        irk_gn.hess_prod(x, u, t_k, 0.1, c.w, c.v_x, c.v_u, hvx, hvu);
        check(hvx.cwiseAbs().maxCoeff() == 0.0 && hvu.cwiseAbs().maxCoeff() == 0.0,
              "GN switch: hess_prod returns zeros");
    }
    std::printf("  4j gn switch   Rad2-GN   OK  (zero HVPs, jac FD-correct)\n");
}

// ---------------------------------------------------------------------------
//  Forward cache: cold == warm, value_jac primes the cache
// ---------------------------------------------------------------------------

template <class Integ>
void check_cache(Integ& integ, const char* name)
{
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    const state_t w(0.7, -0.3);
    const state_t v1(0.4, 0.9);
    const state_t v2(-0.1, 0.6);
    const control_t vu1(-0.5);
    const control_t vu2(0.8);

    // Cold and warm HVP at the same (x, u, t_k) must agree bit-for-bit.
    state_t cold_x;
    control_t cold_u;
    state_t warm_x;
    control_t warm_u;
    integ.hess_prod(x, u, t_k, 0.1, w, v1, vu1, cold_x, cold_u);
    integ.hess_prod(x, u, t_k, 0.1, w, v1, vu1, warm_x, warm_u);
    check((cold_x - warm_x).cwiseAbs().maxCoeff() == 0.0 &&
              (cold_u - warm_u).cwiseAbs().maxCoeff() == 0.0,
          (std::string(name) + " hess_prod cold == warm (same key)").c_str());

    // A different direction must still be correct (re-threaded JVP on the
    // cached forward).
    integ.hess_prod(x, u, t_k, 0.1, w, v2, vu2, warm_x, warm_u);
    const z_t fd2 = fd_hvp(
        x, u, w, v2, vu2,
        [&](const state_t& xx, const control_t& uu)
        {
            state_t out;
            integ.value(xx, uu, t_k, 0.1, out);
            return out;
        });
    z_t actual;
    actual.head(nx) = warm_x;
    actual.tail(nu) = warm_u;
    check((actual - fd2).cwiseAbs().maxCoeff() < 1e-5,
          (std::string(name) + " hess_prod warm direction vs FD").c_str());

    // value_jac primes the cache: a following hess_prod with the same key
    // must be correct (it reuses the value_jac forward on the non-AD path).
    state_t x_next;
    df_dx_t Jx;
    df_du_t Ju;
    integ.value_jac(x + state_t(0.01, -0.02), u, t_k, 0.1, x_next, Jx, Ju);
    integ.hess_prod(x + state_t(0.01, -0.02), u, t_k, 0.1, w, v1, vu1,
                    warm_x,
                    warm_u);
    const state_t xn2 = x + state_t(0.01, -0.02);
    const z_t fd3 = fd_hvp(
        xn2, u, w, v1, vu1,
        [&](const state_t& xx, const control_t& uu)
        {
            state_t out;
            integ.value(xx, uu, t_k, 0.1, out);
            return out;
        });
    actual.head(nx) = warm_x;
    actual.tail(nu) = warm_u;
    check((actual - fd3).cwiseAbs().maxCoeff() < 1e-5,
          (std::string(name) + " hess_prod after value_jac vs FD").c_str());
    std::printf("  4j cache       %s  OK  (cold==warm, direction, primed)\n",
                name);
}

// ---------------------------------------------------------------------------
//  Cross-backend: hand-written ODE (via OdeToDae) == AD DAE, same integrator
// ---------------------------------------------------------------------------

template <class IntegAd, class IntegRef>
void check_cross_backend(IntegAd& ig_ad, IntegRef& ig_ref, const char* name)
{
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    state_t xn_ad;
    state_t xn_ref;
    df_dx_t Jx_ad;
    df_du_t Ju_ad;
    df_dx_t Jx_ref;
    df_du_t Ju_ref;
    ig_ad.value_jac(x, u, t_k, 0.1, xn_ad, Jx_ad, Ju_ad);
    ig_ref.value_jac(x, u, t_k, 0.1, xn_ref, Jx_ref, Ju_ref);
    check((xn_ad - xn_ref).cwiseAbs().maxCoeff() < 1e-12,
          (std::string(name) + " cross-backend value").c_str());
    check((Jx_ad - Jx_ref).cwiseAbs().maxCoeff() < 1e-12,
          (std::string(name) + " cross-backend df/dx").c_str());
    check((Ju_ad - Ju_ref).cwiseAbs().maxCoeff() < 1e-12,
          (std::string(name) + " cross-backend df/du").c_str());

    const state_t w(1.0, 0.4);
    const state_t v_x(0.1, -0.6);
    const control_t v_u(0.8);
    state_t hvx_ad;
    control_t hvu_ad;
    state_t hvx_ref;
    control_t hvu_ref;
    ig_ad.hess_prod(x, u, t_k, 0.1, w, v_x, v_u, hvx_ad, hvu_ad);
    ig_ref.hess_prod(x, u, t_k, 0.1, w, v_x, v_u, hvx_ref, hvu_ref);
    check((hvx_ad - hvx_ref).cwiseAbs().maxCoeff() < 1e-12,
          (std::string(name) + " cross-backend hv_x").c_str());
    check((hvu_ad - hvu_ref).cwiseAbs().maxCoeff() < 1e-12,
          (std::string(name) + " cross-backend hv_u").c_str());
    std::printf("  4j x-backend  %s  OK  (AD == hand-written, <1e-12)\n",
                name);
}

// ---------------------------------------------------------------------------
//  AD HVP opt-out: ad_hess_prod = false
// ---------------------------------------------------------------------------

void check_ad_hess_opt_out()
{
    // The Dual1 value_jac path is unaffected by the opt-out.
    ocp::ExplicitRkIntegrator<Dims, OdeXsqUAdNoHess, ocp::K4Tag> erk_no(
        OdeXsqUAdNoHess{});
    check_integ_value_jac(erk_no, "K4-AD-NoHess");

    // The consistent pair (opt-out + has_dynamics_hess = false) goes through
    // the problem adapter with the Dual1 Jacobian only; value + Jacobian must
    // match the enabled ODE at the same point.
    static_assert(!GnDims::has_dynamics_hess, "GN dims variant");
    using ErkNoHess =
        ocp::ExplicitRkIntegrator<GnDims, OdeXsqUAdNoHess, ocp::K4Tag>;
    ocp::ContinuousProblem<ErkNoHess> prob(OdeXsqUAdNoHess{}, 0.1);
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    state_t x_next;
    df_dx_t Jx;
    df_du_t Ju;
    prob.dynamics_value_jac(0, x, u, x_next, Jx, Ju);

    state_t x_next_ad;
    df_dx_t Jx_ad;
    df_du_t Ju_ad;
    OdeXsqUAd ad;
    ocp::ExplicitRkIntegrator<Dims, OdeXsqUAd, ocp::K4Tag> erk_ad(ad);
    erk_ad.value_jac(x, u, 0.0, 0.1, x_next_ad, Jx_ad, Ju_ad);
    check((x_next - x_next_ad).cwiseAbs().maxCoeff() < 1e-12,
          "ad opt-out: dynamics value == enabled ODE");
    check((Jx - Jx_ad).cwiseAbs().maxCoeff() < 1e-12,
          "ad opt-out: dynamics df/dx == enabled ODE");
    check((Ju - Ju_ad).cwiseAbs().maxCoeff() < 1e-12,
          "ad opt-out: dynamics df/du == enabled ODE");
    std::printf("  4j ad opt-out  K4-NoHess OK  (value_jac Dual1, GN dims)\n");
}

// ---------------------------------------------------------------------------
//  Problem level: ContinuousProblem dynamics Hessian assembly vs the switch
// ---------------------------------------------------------------------------

void check_problem_hess()
{
    DaeXsqUAd dae;
    using Irk = ocp::ImplicitRkIntegrator<Dims, DaeXsqUAd,
                                          ocp::RadauIia2Tag>;
    ocp::ContinuousProblem<Irk> prob(dae, 0.1, 8);
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    const auto& c = cases[0];

    state_t x_next;
    df_dx_t Jx;
    df_du_t Ju;
    P::dyn_hess_t hess;
    prob.dynamics_value_jac_hess(0, x, u, c.w, x_next, Jx, Ju, hess);

    // Each column j of the assembled Hessian is the composed-map HVP along
    // the unit direction e_j (z = [x; u] layout).
    for (int j = 0; j < nIn; ++j)
    {
        state_t v_x = state_t::Zero();
        control_t v_u = control_t::Zero();
        if (j < nx)
        {
            v_x(j) = 1.0;
        }
        else
        {
            v_u(j - nx) = 1.0;
        }
        const z_t fd = fd_hvp(
            x, u, c.w, v_x, v_u,
            [&](const state_t& xx, const control_t& uu)
            { return prob.dynamics_next_state(0, xx, uu); });
        check((hess.col(j) - fd).cwiseAbs().maxCoeff() < 1e-5,
              "problem dynamics Hessian column vs FD");
    }
    std::printf("  4j prob hess   Rad2-AD   OK  (columns FD<1e-5)\n");

    // GN variant: WithHess = false assembles an identically zero dynamics
    // Hessian with an unchanged Jacobian (the SQP then runs the GN model).
    DaeXsqUGn dae_gn;
    using IrkGn = ocp::ImplicitRkIntegrator<Dims, DaeXsqUGn,
                                            ocp::RadauIia2Tag>;
    ocp::ContinuousProblem<IrkGn> prob_gn(dae_gn, 0.1, 8);
    state_t xn_gn;
    df_dx_t Jx_gn;
    df_du_t Ju_gn;
    P::dyn_hess_t hess_gn;
    prob_gn.dynamics_value_jac_hess(0, x, u, c.w, xn_gn, Jx_gn, Ju_gn,
                                    hess_gn);
    check(hess_gn.cwiseAbs().maxCoeff() == 0.0,
          "GN switch: dynamics Hessian identically zero");
    check((Jx_gn - Jx).cwiseAbs().maxCoeff() < 1e-12,
          "GN switch: Jacobian unchanged");
    check((xn_gn - x_next).cwiseAbs().maxCoeff() < 1e-12,
          "GN switch: value unchanged");
    std::printf("  4j prob gn     Rad2-GN   OK  (zero Hessian, jac==value)\n");
}

}  // namespace

int run_autodiff_4j_tests()
{
    check_dae_jacobian();
    check_dae_hess_prod();

    OdeXsqUAd ad;
    OdeXsqURef ref;
    DaeXsqUAd dae;
    DaeXsqUGn dae_gn;

    ocp::ExplicitRkIntegrator<Dims, OdeXsqUAd, ocp::K4Tag> erk(ad);
    ocp::ExplicitRkIntegrator<Dims, OdeXsqUAd, ocp::K4Tag, 2> erk2(ad);
    ocp::ExplicitRkIntegrator<Dims, OdeXsqURef, ocp::K4Tag> erk_ref(ref);
    ocp::ImplicitRkIntegrator<Dims, DaeXsqUAd, ocp::RadauIia2Tag> irk(
        dae, 8);
    ocp::ImplicitRkIntegrator<Dims, DaeXsqUGn, ocp::RadauIia2Tag> irk_gn(
        dae_gn, 8);
    ocp::ImplicitRkIntegrator<Dims, ocp::OdeToDae<OdeXsqURef, Dims>,
                              ocp::RadauIia2Tag>
        irk_ref(ref, 8);

    check_integ_value_jac(erk, "K4-AD");
    check_integ_value_jac(erk2, "K4x2-AD");
    check_integ_hess(erk, "K4-AD");
    check_integ_value_jac_hess_prod(erk, "K4-AD");

    check_integ_value_jac(irk, "Rad2-AD");
    check_integ_hess(irk, "Rad2-AD");
    check_integ_value_jac_hess_prod(irk, "Rad2-AD");
    check_gn_switch();

    check_cache(erk_ref, "K4-ref");
    check_cache(irk_ref, "Rad2-ref");

    check_cross_backend(erk, erk_ref, "K4");
    check_cross_backend(irk, irk_ref, "Rad2");

    check_ad_hess_opt_out();

    check_problem_hess();

    if (failures != 0)
    {
        std::fprintf(stderr, "[4j] %d check(s) failed.\n", failures);
    }
    return failures;
}

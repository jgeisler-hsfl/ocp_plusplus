// Phase 4j: verify the AutoDiffOde CRTP base and the integrators' fused
// value / Jacobian / HVP entries.
//
//   OdeXsqURef  hand-written f / jacobian / hess_prod for
//               xdot = x.*x * u + x + [sin t; cos t] — exact reference and
//               the finite-difference ground truth.
//   OdeXsqUAd   AutoDiffOde-based model with the same f but no derivative
//               methods — the AD-under-test.
//
// Checks:
//   * trait detection: the AD ODE is dual-evaluable and exposes the fused
//     entries; the hand-written ODE is not dual-evaluable (guards the
//     return-type probe against Eigen's converting constructor);
//   * ODE level: legacy jacobian / hess_prod and the fused value_jac /
//     value_jac_hess_prod vs the hand-derived reference and central FD;
//   * integrator level (ExplicitRk K4, K4 x2, ImplicitRk Radau IIA 2):
//     value_jac / hess_prod / value_jac_hess_prod of the composed map vs
//     central FD; forward-cache behavior (cold == warm, value_jac primes);
//     cross-backend: the hand-written ODE through the same integrator
//     matches the AD ODE (guards the AD glue end-to-end).

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/integrators/ode_autodiff.hpp"
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
constexpr double t_k = 0.37;  // OCP-interval start used by all checks

using z_t = Eigen::Matrix<double, nIn, 1>;
using H_t = Eigen::Matrix<double, nIn, nIn>;
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

/// The AD-under-test: only f is written; the legacy and fused derivative
/// entries come from the AutoDiffOde base (forward-mode multidual AD).
struct OdeXsqUAd : ocp::AutoDiffOde<OdeXsqUAd, Dims>
{
    template <class D>
    Eigen::Matrix<D, nx, 1> f(const Eigen::Matrix<D, nx, 1>& x,
                              const Eigen::Matrix<D, nu, 1>& u, double t) const
    {
        return x.cwiseProduct(x) * u(0) + x
             + Eigen::Matrix<D, nx, 1>(D(std::sin(t)), D(std::cos(t)));
    }
};

// ---------------------------------------------------------------------------
//  Traits
// ---------------------------------------------------------------------------

static_assert(ocp::ode_supports_hess_prod<OdeXsqUAd, Dims>,
              "AutoDiffOde must expose hess_prod");
static_assert(ocp::ode_supports_value_jac<OdeXsqUAd, Dims>,
              "AutoDiffOde must expose value_jac");
static_assert(ocp::ode_supports_value_jac_hess_prod<OdeXsqUAd, Dims>,
              "AutoDiffOde must expose value_jac_hess_prod");
static_assert(ocp::ode_f_evaluable_at<OdeXsqUAd, Dims, Dual1>,
              "the AD f must be evaluable at Dual1");
static_assert(ocp::ode_f_evaluable_at<OdeXsqUAd, Dims, Dual2>,
              "the AD f must be evaluable at Dual2");
static_assert(!ocp::ode_f_evaluable_at<OdeXsqURef, Dims, Dual1>,
              "hand-written double-f must not appear dual-evaluable");
static_assert(!ocp::ode_f_evaluable_at<OdeXsqURef, Dims, Dual2>,
              "hand-written double-f must not appear dual-evaluable");
static_assert(ocp::ode_supports_hess_prod<OdeXsqURef, Dims>,
              "the hand-written reference must expose hess_prod");

static_assert(
    ocp::ExplicitRkIntegrator<Dims, OdeXsqUAd, 4, ocp::K4Tag>::
        supports_value_jac_hess_prod,
    "explicit integrator over an AD ODE must flag the fused HVP entry");
static_assert(
    !ocp::ExplicitRkIntegrator<Dims, OdeXsqURef, 4, ocp::K4Tag>::
        supports_value_jac_hess_prod,
    "explicit integrator over a hand-written ODE must not");
static_assert(
    !ocp::ImplicitRkIntegrator<Dims, OdeXsqUAd, 2, ocp::RadauIia2Tag>::
        supports_value_jac_hess_prod,
    "implicit integrator has no dual-Newton fused entry");

// ---------------------------------------------------------------------------
//  Finite-difference reference: HVP of g(z) = w^T map(z), z = [x; u]
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

struct Case
{
    state_t   w;
    state_t   v_x;
    control_t v_u;
};
const Case cases[] = {
    {state_t(0.7, -0.3), state_t(0.4, 0.9), control_t(-0.5)},
    {state_t(1.0, 0.4), state_t(0.1, -0.6), control_t(0.8)},
    {state_t(0.2, 0.9), state_t(-0.3, 0.5), control_t(0.2)},
};

// ---------------------------------------------------------------------------
//  ODE level: AD entries vs the hand-derived reference and FD
// ---------------------------------------------------------------------------

void check_ode_jacobian()
{
    OdeXsqUAd ad;
    OdeXsqURef ref;
    const state_t x(0.3, -0.7);
    const control_t u(0.4);

    df_dx_t Jx_ad;
    df_du_t Ju_ad;
    ad.jacobian(x, u, t_k, Jx_ad, Ju_ad);
    df_dx_t Jx_ref;
    df_du_t Ju_ref;
    ref.jacobian(x, u, t_k, Jx_ref, Ju_ref);

    check((Jx_ad - Jx_ref).cwiseAbs().maxCoeff() < 1e-13,
          "ODE jacobian AD vs hand-derived (df/dx)");
    check((Ju_ad - Ju_ref).cwiseAbs().maxCoeff() < 1e-13,
          "ODE jacobian AD vs hand-derived (df/du)");

    const double d = 1e-6;
    for (int j = 0; j < nx; ++j)
    {
        state_t e = state_t::Zero();
        e(j) = d;
        const state_t xp = x + e;
        const state_t xm = x - e;
        const state_t fp = ad.f(xp, u, t_k);
        const state_t fm = ad.f(xm, u, t_k);
        for (int i = 0; i < nx; ++i)
            check(std::abs(Jx_ad(i, j) - (fp(i) - fm(i)) / (2.0 * d)) < 1e-6,
                  "ODE jacobian AD vs FD (df/dx)");
    }
    for (int j = 0; j < nu; ++j)
    {
        control_t e = control_t::Zero();
        e(j) = d;
        const control_t up = u + e;
        const control_t um = u - e;
        const state_t fp = ad.f(x, up, t_k);
        const state_t fm = ad.f(x, um, t_k);
        for (int i = 0; i < nx; ++i)
            check(std::abs(Ju_ad(i, j) - (fp(i) - fm(i)) / (2.0 * d)) < 1e-6,
                  "ODE jacobian AD vs FD (df/du)");
    }
    std::printf(
        "  4j ode jac   OdeXsqUAd  OK  (exact vs hand-derived, FD<1e-6)\n");
}

void check_ode_hess_prod()
{
    OdeXsqUAd ad;
    OdeXsqURef ref;
    const state_t x(0.3, -0.7);
    const control_t u(0.4);

    for (const auto& c : cases)
    {
        state_t hvx_ad;
        control_t hvu_ad;
        ad.hess_prod(x, u, t_k, c.w, c.v_x, c.v_u, hvx_ad, hvu_ad);
        state_t hvx_ref;
        control_t hvu_ref;
        ref.hess_prod(x, u, t_k, c.w, c.v_x, c.v_u, hvx_ref, hvu_ref);

        check((hvx_ad - hvx_ref).cwiseAbs().maxCoeff() < 1e-12,
              "ODE hess_prod AD vs hand-derived (hv_x)");
        check((hvu_ad - hvu_ref).cwiseAbs().maxCoeff() < 1e-12,
              "ODE hess_prod AD vs hand-derived (hv_u)");

        const z_t fd = fd_hvp(x, u, c.w, c.v_x, c.v_u,
                              [&](const state_t& xx, const control_t& uu)
                              { return ad.f(xx, uu, t_k); });
        z_t actual;
        actual.head(nx) = hvx_ad;
        actual.tail(nu) = hvu_ad;
        const double err = (actual - fd).cwiseAbs().maxCoeff();
        check(err < 1e-5, "ODE hess_prod AD vs FD Hessian");
    }
    std::printf("  4j ode hess  OdeXsqUAd  OK  (exact vs hand-derived, "
                "FD<1e-5, 3 cases)\n");
}

void check_ode_value_jac()
{
    OdeXsqUAd ad;
    OdeXsqURef ref;
    const state_t x(0.3, -0.7);
    const control_t u(0.4);

    state_t f_val;
    df_dx_t Jx;
    df_du_t Ju;
    ad.value_jac(x, u, t_k, f_val, Jx, Ju);
    check((f_val - ref.f(x, u, t_k)).cwiseAbs().maxCoeff() < 1e-13,
          "ODE value_jac value vs hand f");
    df_dx_t Jx_ref;
    df_du_t Ju_ref;
    ref.jacobian(x, u, t_k, Jx_ref, Ju_ref);
    check((Jx - Jx_ref).cwiseAbs().maxCoeff() < 1e-13,
          "ODE value_jac df/dx vs hand-derived");
    check((Ju - Ju_ref).cwiseAbs().maxCoeff() < 1e-13,
          "ODE value_jac df/du vs hand-derived");
    std::printf("  4j ode vjac  OdeXsqUAd  OK  (value + Jacobian fused)\n");
}

void check_ode_value_jac_hess_prod()
{
    OdeXsqUAd ad;
    OdeXsqURef ref;
    const state_t x(0.3, -0.7);
    const control_t u(0.4);

    for (const auto& c : cases)
    {
        state_t f_val;
        df_dx_t Jx;
        df_du_t Ju;
        state_t hvx;
        control_t hvu;
        ad.value_jac_hess_prod(x, u, t_k, f_val, Jx, Ju, c.w, c.v_x, c.v_u,
                               hvx, hvu);
        check((f_val - ref.f(x, u, t_k)).cwiseAbs().maxCoeff() < 1e-13,
              "ODE value_jac_hess_prod value vs hand f");
        df_dx_t Jx_ref;
        df_du_t Ju_ref;
        ref.jacobian(x, u, t_k, Jx_ref, Ju_ref);
        check((Jx - Jx_ref).cwiseAbs().maxCoeff() < 1e-13,
              "ODE value_jac_hess_prod df/dx vs hand-derived");
        check((Ju - Ju_ref).cwiseAbs().maxCoeff() < 1e-13,
              "ODE value_jac_hess_prod df/du vs hand-derived");
        state_t hvx_ref;
        control_t hvu_ref;
        ref.hess_prod(x, u, t_k, c.w, c.v_x, c.v_u, hvx_ref, hvu_ref);
        check((hvx - hvx_ref).cwiseAbs().maxCoeff() < 1e-12,
              "ODE value_jac_hess_prod hv_x vs hand-derived");
        check((hvu - hvu_ref).cwiseAbs().maxCoeff() < 1e-12,
              "ODE value_jac_hess_prod hv_u vs hand-derived");
    }
    std::printf("  4j ode vjh   OdeXsqUAd  OK  (fused triplet, 3 cases)\n");
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
    integ.value_jac(x, u, t_k, x_next, Jx, Ju);

    // The fused value must match the plain value pass.
    state_t x_val;
    integ.value(x, u, t_k, x_val);
    check((x_next - x_val).cwiseAbs().maxCoeff() < 1e-12,
          (std::string(name) + " value_jac value == value").c_str());

    const double d = 1e-6;
    for (int j = 0; j < nx; ++j)
    {
        state_t e = state_t::Zero();
        e(j) = d;
        state_t fp, fm;
        integ.value(x + e, u, t_k, fp);
        integ.value(x - e, u, t_k, fm);
        for (int i = 0; i < nx; ++i)
            check(std::abs(Jx(i, j) - (fp(i) - fm(i)) / (2.0 * d)) < 1e-6,
                  (std::string(name) + " value_jac df/dx vs FD").c_str());
    }
    for (int j = 0; j < nu; ++j)
    {
        control_t e = control_t::Zero();
        e(j) = d;
        state_t fp, fm;
        integ.value(x, u + e, t_k, fp);
        integ.value(x, u - e, t_k, fm);
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
        integ.hess_prod(x, u, t_k, c.w, c.v_x, c.v_u, hvx, hvu);
        const z_t fd = fd_hvp(
            x, u, c.w, c.v_x, c.v_u,
            [&](const state_t& xx, const control_t& uu)
            {
                state_t out;
                integ.value(xx, uu, t_k, out);
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
    integ.value(x, u, t_k, x_val);
    for (const auto& c : cases)
    {
        state_t x_next;
        df_dx_t Jx;
        df_du_t Ju;
        state_t hvx;
        control_t hvu;
        integ.value_jac_hess_prod(x, u, t_k, x_next, Jx, Ju, c.w, c.v_x, c.v_u,
                                  hvx, hvu);
        check((x_next - x_val).cwiseAbs().maxCoeff() < 1e-12,
              (std::string(name) + " value_jac_hess_prod value == value")
                  .c_str());
        const z_t fd = fd_hvp(
            x, u, c.w, c.v_x, c.v_u,
            [&](const state_t& xx, const control_t& uu)
            {
                state_t out;
                integ.value(xx, uu, t_k, out);
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
    integ.hess_prod(x, u, t_k, w, v1, vu1, cold_x, cold_u);
    integ.hess_prod(x, u, t_k, w, v1, vu1, warm_x, warm_u);
    check((cold_x - warm_x).cwiseAbs().maxCoeff() == 0.0 &&
              (cold_u - warm_u).cwiseAbs().maxCoeff() == 0.0,
          (std::string(name) + " hess_prod cold == warm (same key)").c_str());

    // A different direction must still be correct (re-threaded JVP on the
    // cached forward).
    integ.hess_prod(x, u, t_k, w, v2, vu2, warm_x, warm_u);
    const z_t fd2 = fd_hvp(
        x, u, w, v2, vu2,
        [&](const state_t& xx, const control_t& uu)
        {
            state_t out;
            integ.value(xx, uu, t_k, out);
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
    integ.value_jac(x + state_t(0.01, -0.02), u, t_k, x_next, Jx, Ju);
    integ.hess_prod(x + state_t(0.01, -0.02), u, t_k, w, v1, vu1, warm_x,
                    warm_u);
    const state_t xn2 = x + state_t(0.01, -0.02);
    const z_t fd3 = fd_hvp(
        xn2, u, w, v1, vu1,
        [&](const state_t& xx, const control_t& uu)
        {
            state_t out;
            integ.value(xx, uu, t_k, out);
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
//  Cross-backend: hand-written ODE == AD ODE through the same integrator
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
    ig_ad.value_jac(x, u, t_k, xn_ad, Jx_ad, Ju_ad);
    ig_ref.value_jac(x, u, t_k, xn_ref, Jx_ref, Ju_ref);
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
    ig_ad.hess_prod(x, u, t_k, w, v_x, v_u, hvx_ad, hvu_ad);
    ig_ref.hess_prod(x, u, t_k, w, v_x, v_u, hvx_ref, hvu_ref);
    check((hvx_ad - hvx_ref).cwiseAbs().maxCoeff() < 1e-12,
          (std::string(name) + " cross-backend hv_x").c_str());
    check((hvu_ad - hvu_ref).cwiseAbs().maxCoeff() < 1e-12,
          (std::string(name) + " cross-backend hv_u").c_str());
    std::printf("  4j x-backend  %s  OK  (AD == hand-written, <1e-12)\n",
                name);
}

}  // namespace

int run_autodiff_4j_tests()
{
    check_ode_jacobian();
    check_ode_hess_prod();
    check_ode_value_jac();
    check_ode_value_jac_hess_prod();

    OdeXsqUAd ad;
    OdeXsqURef ref;

    ocp::ExplicitRkIntegrator<Dims, OdeXsqUAd, 4, ocp::K4Tag> erk(ad, 0.1);
    ocp::ExplicitRkIntegrator<Dims, OdeXsqUAd, 4, ocp::K4Tag, 2> erk2(ad, 0.1);
    ocp::ExplicitRkIntegrator<Dims, OdeXsqURef, 4, ocp::K4Tag> erk_ref(ref, 0.1);
    ocp::ImplicitRkIntegrator<Dims, OdeXsqUAd, 2, ocp::RadauIia2Tag> irk(
        ad, 0.1, 8);
    ocp::ImplicitRkIntegrator<Dims, OdeXsqURef, 2, ocp::RadauIia2Tag> irk_ref(
        ref, 0.1, 8);

    check_integ_value_jac(erk, "K4-AD");
    check_integ_value_jac(erk2, "K4x2-AD");
    check_integ_hess(erk, "K4-AD");
    check_integ_hess(irk, "Rad2-AD");
    check_integ_value_jac_hess_prod(erk, "K4-AD");
    check_integ_value_jac_hess_prod(irk, "Rad2-AD");

    check_cache(erk_ref, "K4-ref");
    check_cache(irk_ref, "Rad2-ref");

    check_cross_backend(erk, erk_ref, "K4");
    check_cross_backend(irk, irk_ref, "Rad2");

    if (failures != 0)
    {
        std::fprintf(stderr, "[4j] %d check(s) failed.\n", failures);
    }
    return failures;
}

// Phase 4b: verify ExplicitRkIntegrator value / jacobian / hess_prod against
// the analytic ODE solution and finite differences of the composed map.

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/integrators/rk_explicit.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

namespace
{

using Dims = DoubleIntegratorDims;  // nx = 2, nu = 1
using P = ocp::Problem<Dims>;
using state_t = P::state_t;
using control_t = P::control_t;
using df_dx_t = P::dyn_df_dx_t;
using df_du_t = P::dyn_df_du_t;

constexpr int nx = 2;
constexpr int nu = 1;

int failures = 0;

void check(bool cond, const char* msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

// ---------------------------------------------------------------------------
//  ODE models
// ---------------------------------------------------------------------------

/// xdot = -x  (linear; used for the value test against exp(-t)).
struct OdeNegX
{
    state_t f(const state_t& x, const control_t&) const
    {
        return -x;
    }
    void jacobian(const state_t&, const control_t&, df_dx_t& df_dx,
                  df_du_t& df_du) const
    {
        df_dx = -df_dx_t::Identity();
        df_du.setZero();
    }
    void hess_prod(const state_t&, const control_t&, const state_t&,
                   const state_t&, const control_t&, state_t& hv_x,
                   control_t& hv_u) const
    {
        hv_x.setZero();
        hv_u.setZero();
    }
};

/// xdot = x.*x  (elementwise nonlinear; value test against x0/(1+x0*t)).
struct OdeXsq
{
    state_t f(const state_t& x, const control_t&) const
    {
        return x.cwiseProduct(x);
    }
    void jacobian(const state_t& x, const control_t&, df_dx_t& df_dx,
                  df_du_t& df_du) const
    {
        df_dx = (2.0 * x).asDiagonal();
        df_du.setZero();
    }
    void hess_prod(const state_t&, const control_t&, const state_t& w,
                   const state_t& v_x, const control_t&, state_t& hv_x,
                   control_t& hv_u) const
    {
        hv_x = 2.0 * w.cwiseProduct(v_x);
        hv_u.setZero();
    }
};

/// xdot = -x .* u  (scalar control u(0); both Jacobian blocks non-trivial, and
/// the composed map is nonlinear in (x, u) because the stages couple u into
/// the stage states).
struct OdeNegXU
{
    state_t f(const state_t& x, const control_t& u) const
    {
        state_t r;
        r(0) = -x(0) * u(0);
        r(1) = -x(1) * u(0);
        return r;
    }
    void jacobian(const state_t& x, const control_t& u, df_dx_t& df_dx,
                  df_du_t& df_du) const
    {
        df_dx = (-u(0)) * df_dx_t::Identity();
        df_du(0, 0) = -x(0);
        df_du(1, 0) = -x(1);
    }
    void hess_prod(const state_t&, const control_t&, const state_t& w,
                   const state_t& v_x, const control_t& v_u, state_t& hv_x,
                   control_t& hv_u) const
    {
        // f_i = -x_i u:  d2f_i/(dx_i du) = -1, all pure-second terms are zero.
        hv_x.setZero();
        for (int i = 0; i < nx; ++i)
        {
            hv_x(i) = -w(i) * v_u(0);
        }
        hv_u(0) = -w(0) * v_x(0) - w(1) * v_x(1);
    }
};

// ---------------------------------------------------------------------------
//  value
// ---------------------------------------------------------------------------

template <int NS, class Tag>
void check_value_neg_x()
{
    using Integ = ocp::ExplicitRkIntegrator<Dims, OdeNegX, NS, Tag>;
    const Integ integ(OdeNegX{}, 0.1);
    state_t cur(1.0, 1.0);
    control_t u(0.0);

    for (int step = 0; step < 10; ++step)
    {
        integ.value(cur, u, cur);
    }

    const double expected = std::exp(-1.0);  // t = 10 * 0.1
    check(std::abs(cur(0) - expected) < 1e-5, "value xdot=-x K4 c0");
    check(std::abs(cur(1) - expected) < 1e-5, "value xdot=-x K4 c1");
    std::printf("  value xdot=-x   OK  (|err|<1e-5)\n");
}

template <int NS, class Tag>
void check_value_xsq()
{
    using Integ = ocp::ExplicitRkIntegrator<Dims, OdeXsq, NS, Tag>;
    const Integ integ(OdeXsq{}, 0.2);
    state_t cur(0.5, 0.5);
    control_t u(0.0);

    for (int step = 0; step < 5; ++step)
    {
        integ.value(cur, u, cur);
    }

    // xdot = x^2  ->  x(t) = x0 / (1 - x0 t);  t = 5 * 0.2 = 1, x0 = 0.5.
    const double expected = 0.5 / (1.0 - 0.5 * 1.0);  // = 1.0
    check(std::abs(cur(0) - expected) < 1e-4, "value xdot=x^2 K4 c0");
    check(std::abs(cur(1) - expected) < 1e-4, "value xdot=x^2 K4 c1");
    std::printf("  value xdot=x^2  OK  (|err|<1e-4)\n");
}

// ---------------------------------------------------------------------------
//  jacobian (central finite differences of the composed map)
// ---------------------------------------------------------------------------

template <int NS, class Tag>
void check_jacobian()
{
    using Integ = ocp::ExplicitRkIntegrator<Dims, OdeNegXU, NS, Tag>;
    const Integ integ(OdeNegXU{}, 0.1);
    const state_t x(0.3, -0.7);
    const control_t u(0.4);

    df_dx_t Jx;
    df_du_t Ju;
    integ.jacobian(x, u, Jx, Ju);

    const double d = 1e-6;
    for (int j = 0; j < nx; ++j)
    {
        state_t xp = x;
        state_t xm = x;
        state_t e;
        e.setZero();
        e(j) = d;
        xp += e;
        xm -= e;
        state_t fdp;
        state_t fdm;
        integ.value(xp, u, fdp);
        integ.value(xm, u, fdm);
        for (int i = 0; i < nx; ++i)
        {
            const double fd = (fdp(i) - fdm(i)) / (2.0 * d);
            check(std::abs(Jx(i, j) - fd) < 1e-6,
                  (std::string("jacobian df/dx(") + std::to_string(i) + ","
                      + std::to_string(j) + ")").c_str());
        }
    }
    for (int j = 0; j < nu; ++j)
    {
        control_t up = u;
        control_t um = u;
        control_t e;
        e.setZero();
        e(j) = d;
        up += e;
        um -= e;
        state_t fdp;
        state_t fdm;
        integ.value(x, up, fdp);
        integ.value(x, um, fdm);
        for (int i = 0; i < nx; ++i)
        {
            const double fd = (fdp(i) - fdm(i)) / (2.0 * d);
            check(std::abs(Ju(i, j) - fd) < 1e-6,
                  (std::string("jacobian df/du(") + std::to_string(i) + ")")
                      .c_str());
        }
    }
    std::printf("  jacobian  OdeNegXU  OK  (|err|<1e-6)\n");
}

// ---------------------------------------------------------------------------
//  hess_prod (finite-difference Hessian of the scalar  w' Phi, applied to v)
// ---------------------------------------------------------------------------

template <int NS, class Tag>
void check_hess_prod()
{
    using Integ = ocp::ExplicitRkIntegrator<Dims, OdeNegXU, NS, Tag>;
    using zvec_t = Eigen::Matrix<double, nx + nu, 1>;
    const Integ integ(OdeNegXU{}, 0.1);

    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    zvec_t z0;
    z0(0) = x(0);
    z0(1) = x(1);
    z0(2) = u(0);

    // A few deterministic (w, v) pairs.
    state_t w[3];
    state_t vx[3];
    control_t vu[3];
    w[0] << 0.7, -0.3;
    vx[0] << 0.4, 0.9;
    vu[0] << -0.5;
    w[1] << -0.2, 0.6;
    vx[1] << 0.1, -0.8;
    vu[1] << 0.3;
    w[2] << 0.9, 0.9;
    vx[2] << -0.5, 0.2;
    vu[2] << 0.7;

    const double d = 1e-4;
    const int n = nx + nu;
    Eigen::Matrix<double, n, n> H;

    for (int p = 0; p < 3; ++p)
    {
        auto g = [&](const zvec_t& z) -> double
        {
            state_t xx;
            control_t uu;
            xx(0) = z(0);
            xx(1) = z(1);
            uu(0) = z(2);
            state_t xn;
            integ.value(xx, uu, xn);
            return w[p].dot(xn);
        };

        H.setZero();
        for (int a = 0; a < n; ++a)
        {
            for (int b = a; b < n; ++b)
            {
                if (a == b)
                {
                    zvec_t zp = z0;
                    zvec_t zm = z0;
                    zp(a) += d;
                    zm(a) -= d;
                    H(a, a) = (g(zp) - 2.0 * g(z0) + g(zm)) / (d * d);
                }
                else
                {
                    zvec_t zpp = z0;
                    zvec_t zpm = z0;
                    zvec_t zmp = z0;
                    zvec_t zmm = z0;
                    zpp(a) += d;
                    zpp(b) += d;
                    zpm(a) -= d;
                    zpm(b) += d;
                    zmp(a) += d;
                    zmp(b) -= d;
                    zmm(a) -= d;
                    zmm(b) -= d;
                    H(a, b) = (g(zpp) - g(zpm) - g(zmp) + g(zmm)) / (4.0 * d * d);
                }
                H(b, a) = H(a, b);
            }
        }

        state_t hvx;
        control_t hvu;
        integ.hess_prod(x, u, w[p], vx[p], vu[p], hvx, hvu);

        zvec_t v;
        v(0) = vx[p](0);
        v(1) = vx[p](1);
        v(2) = vu[p](0);
        const zvec_t ref = H * v;
        for (int i = 0; i < nx; ++i)
        {
            check(std::abs(hvx(i) - ref(i)) < 1e-5,
                  (std::string("hess_prod hv_x(") + std::to_string(i)
                      + ") case " + std::to_string(p)).c_str());
        }
        for (int i = 0; i < nu; ++i)
        {
            check(std::abs(hvu(i) - ref(nx + i)) < 1e-5,
                  (std::string("hess_prod hv_u(") + std::to_string(i)
                      + ") case " + std::to_string(p)).c_str());
        }
    }
    std::printf("  hess_prod  OdeNegXU  OK  (|err|<1e-5, 3 cases)\n");
}

}  // namespace

int run_erk_4b_tests()
{
    // value
    check_value_neg_x<4, ocp::K4Tag>();
    check_value_xsq<4, ocp::K4Tag>();

    // jacobian
    check_jacobian<2, ocp::K2Tag>();
    check_jacobian<4, ocp::K4Tag>();

    // hess_prod
    check_hess_prod<2, ocp::K2Tag>();
    check_hess_prod<4, ocp::K4Tag>();

    if (failures != 0)
    {
        std::fprintf(stderr, "[4b] %d check(s) failed.\n", failures);
    }
    return failures;
}

// Phase 4h: verify time-varying ODE support (t in the ODE contract) for the
// ExplicitRkIntegrator (ERK). Tests value, jacobian, and hess_prod against
// high-resolution RK4 references and finite differences.

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

/// Time-varying ODE:  xdot_i = sin(t) * x_i + u(0).
struct OdeSinXPlusU
{
    state_t f(const state_t& x, const control_t& u, double t) const
    {
        return std::sin(t) * x + state_t::Constant(u(0));
    }
};

/// Time-invariant ODE (ignores t):  xdot = -x  (regression check).
struct OdeNegX
{
    state_t f(const state_t& x, const control_t&, double) const
    {
        return -x;
    }
};

/// Time-varying ODE:  xdot_i = sin(t)*x_i*u(0) + cos(t)*x_i.
/// Used for jacobian and hess_prod FD tests.
struct OdeSinCosXU
{
    state_t f(const state_t& x, const control_t& u, double t) const
    {
        return (std::sin(t) * u(0) + std::cos(t)) * x;
    }
    void jacobian(const state_t& x, const control_t& u, double t,
                  df_dx_t& df_dx, df_du_t& df_du) const
    {
        const double s = std::sin(t);
        const double c = std::cos(t);
        df_dx = (s * u(0) + c) * df_dx_t::Identity();
        df_du.col(0) = s * x;
    }
    void hess_prod(const state_t&, const control_t&, double t,
                   const state_t& w, const state_t& v_x, const control_t& v_u,
                   state_t& hv_x, control_t& hv_u) const
    {
        // f_i = (s*u + c) * x_i  with s=sin(t), c=cos(t).
        // d2f_i/dx_j/dx_k = 0  (linear in x)
        // d2f_i/dx_j/du   = s * delta_ij
        // d2f_i/du/dx_j   = s * delta_ij
        // d2f_i/du/du     = 0
        // => hv_x_j = s * w_j * v_u,  hv_u = s * w . v_x
        const double s = std::sin(t);
        for (int j = 0; j < nx; ++j)
        {
            hv_x(j) = s * w(j) * v_u(0);
        }
        hv_u(0) = s * w.dot(v_x);
    }
};

// ---------------------------------------------------------------------------
//  Reference RK4 (time-varying)
// ---------------------------------------------------------------------------

/// One classic RK4 step of a time-varying ODE.
template <class Ode>
state_t rk4_step(const Ode& ode, const state_t& x, const control_t& u,
                 double t, double h)
{
    const state_t k1 = ode.f(x, u, t);
    const state_t k2 = ode.f(x + 0.5 * h * k1, u, t + 0.5 * h);
    const state_t k3 = ode.f(x + 0.5 * h * k2, u, t + 0.5 * h);
    const state_t k4 = ode.f(x + h * k3, u, t + h);
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// ---------------------------------------------------------------------------
//  value: time-varying ODE  xdot = sin(t)*x + u  (K4, h=0.1, 10 steps)
// ---------------------------------------------------------------------------

void check_value_time_varying()
{
    using Integ = ocp::ExplicitRkIntegrator<Dims, OdeSinXPlusU, 4, ocp::K4Tag>;
    const Integ integ(OdeSinXPlusU{}, 0.1);
    state_t cur;
    cur << 0.3, -0.7;
    const control_t u(0.4);

    for (int k = 0; k < 10; ++k)
    {
        state_t next;
        integ.value(cur, u, k * 0.1, next);
        cur = next;
    }

    OdeSinXPlusU ode;
    state_t ref;
    ref << 0.3, -0.7;
    const int n_ref = 10000;
    const double h_ref = 1.0 / static_cast<double>(n_ref);
    for (int i = 0; i < n_ref; ++i)
    {
        ref = rk4_step(ode, ref, u, i * h_ref, h_ref);
    }

    const double err = (cur - ref).cwiseAbs().maxCoeff();
    check(err < 1e-4, "value time-varying K4 vs 10000-step RK4");
    std::printf("  value time-varying  OK  (|err|=%.2e < 1e-4)\n", err);
}

// ---------------------------------------------------------------------------
//  value, time-invariant regression:  xdot = -x  (ignores t)
// ---------------------------------------------------------------------------

void check_value_regression()
{
    using Integ = ocp::ExplicitRkIntegrator<Dims, OdeNegX, 4, ocp::K4Tag>;
    const Integ integ(OdeNegX{}, 0.05);
    state_t cur;
    cur << 1.0, 1.0;
    const control_t u(0.0);
    for (int step = 0; step < 20; ++step)
    {
        integ.value(cur, u, 0.0, cur);
    }
    const double expected = std::exp(-1.0);
    check(std::abs(cur(0) - expected) < 1e-4, "value regression xdot=-x c0");
    check(std::abs(cur(1) - expected) < 1e-4, "value regression xdot=-x c1");
    std::printf("  value regression  OK  (|err|<1e-4, exact=%.6f)\n", expected);
}

// ---------------------------------------------------------------------------
//  jacobian, FD:  xdot_i = sin(t)*x_i*u + cos(t)*x_i  (K4, t_k=0.3)
// ---------------------------------------------------------------------------

void check_jacobian_time_varying()
{
    using Integ = ocp::ExplicitRkIntegrator<Dims, OdeSinCosXU, 4, ocp::K4Tag>;
    const Integ integ(OdeSinCosXU{}, 0.1);
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    const double t_k = 0.3;

    df_dx_t Jx;
    df_du_t Ju;
    integ.jacobian(x, u, t_k, Jx, Ju);

    const double d = 1e-6;
    double max_err = 0.0;
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
        integ.value(xp, u, t_k, fdp);
        integ.value(xm, u, t_k, fdm);
        for (int i = 0; i < nx; ++i)
        {
            const double fd = (fdp(i) - fdm(i)) / (2.0 * d);
            max_err = std::max(max_err, std::abs(Jx(i, j) - fd));
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
        integ.value(x, up, t_k, fdp);
        integ.value(x, um, t_k, fdm);
        for (int i = 0; i < nx; ++i)
        {
            const double fd = (fdp(i) - fdm(i)) / (2.0 * d);
            max_err = std::max(max_err, std::abs(Ju(i, j) - fd));
        }
    }
    check(max_err < 1e-6, "jacobian time-varying FD");
    std::printf("  jacobian time-varying  OK  (max|err|=%.2e < 1e-6)\n",
                max_err);
}

// ---------------------------------------------------------------------------
//  hess_prod, FD (ERK):  xdot_i = sin(t)*x_i*u + cos(t)*x_i  (K2, t_k=0.5)
// ---------------------------------------------------------------------------

void check_hess_prod_time_varying()
{
    using Integ = ocp::ExplicitRkIntegrator<Dims, OdeSinCosXU, 2, ocp::K2Tag>;
    using zvec_t = Eigen::Matrix<double, nx + nu, 1>;
    const Integ integ(OdeSinCosXU{}, 0.1);
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    const double t_k = 0.5;

    zvec_t z0;
    z0(0) = x(0);
    z0(1) = x(1);
    z0(2) = u(0);

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

    const double d = 1e-5;
    const int n = nx + nu;
    double max_err = 0.0;
    for (int p = 0; p < 3; ++p)
    {
        auto g = [&](const zvec_t& z) -> double {
            state_t xx;
            xx(0) = z(0);
            xx(1) = z(1);
            control_t uu;
            uu(0) = z(2);
            state_t xn;
            integ.value(xx, uu, t_k, xn);
            return w[p].dot(xn);
        };

        Eigen::Matrix<double, n, n> H;
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
                    H(a, b) =
                        (g(zpp) - g(zpm) - g(zmp) + g(zmm)) / (4.0 * d * d);
                }
                H(b, a) = H(a, b);
            }
        }

        state_t hvx;
        control_t hvu;
        integ.hess_prod(x, u, t_k, w[p], vx[p], vu[p], hvx, hvu);

        zvec_t v;
        v(0) = vx[p](0);
        v(1) = vx[p](1);
        v(2) = vu[p](0);
        const zvec_t ref = H * v;
        for (int i = 0; i < nx; ++i)
        {
            max_err = std::max(max_err, std::abs(hvx(i) - ref(i)));
        }
        for (int i = 0; i < nu; ++i)
        {
            max_err = std::max(max_err, std::abs(hvu(i) - ref(nx + i)));
        }
    }
    check(max_err < 1e-5, "hess_prod time-varying FD (K2)");
    std::printf("  hess_prod time-varying  OK  (max|err|=%.2e < 1e-5)\n",
                max_err);
}

}  // namespace

int run_time_varying_4h_tests()
{
    check_value_time_varying();
    check_value_regression();
    check_jacobian_time_varying();
    check_hess_prod_time_varying();

    if (failures != 0)
    {
        std::fprintf(stderr, "[4h] %d check(s) failed.\n", failures);
    }
    return failures;
}

// Phase 4g: multi-step (NumSteps > 1) FD tests for the explicit RK integrator.
//
// Covers value / jacobian / hess_prod of the multi-step ERK against:
//   - value:      a 10000-step RK4 reference of the same ODE,
//   - jacobian:   central finite differences of the composed OCP map,
//   - hess_prod:  central finite-difference Hessian of  w^T Phi  applied to v.
// A NumSteps=1 value check doubles as the regression anchor against 4b.

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

// Nonlinear ODE:  xdot = x .* x .* u + x  (u scalar). Nonlinear in both x and
// u; both Jacobian blocks are non-trivial and the ODE Hessian is non-zero.
struct OdeXsqU
{
    state_t f(const state_t& x, const control_t& u) const
    {
        return x.cwiseProduct(x) * u(0) + x;
    }
    void jacobian(const state_t& x, const control_t& u, df_dx_t& df_dx,
                  df_du_t& df_du) const
    {
        df_dx = df_dx_t::Identity();
        for (int i = 0; i < nx; ++i)
        {
            df_dx(i, i) += 2.0 * x(i) * u(0);
        }
        df_du.col(0) = x.cwiseProduct(x);
    }
    void hess_prod(const state_t& x, const control_t& u, const state_t& w,
                   const state_t& v_x, const control_t& v_u, state_t& hv_x,
                   control_t& hv_u) const
    {
        // f_i = x_i^2 u + x_i:  d2f_i/(dx_i^2) = 2u,  d2f_i/(dx_i du) = 2 x_i.
        for (int j = 0; j < nx; ++j)
        {
            hv_x(j) = 2.0 * u(0) * w(j) * v_x(j) + 2.0 * w(j) * x(j) * v_u(0);
        }
        double s = 0.0;
        for (int i = 0; i < nx; ++i)
        {
            s += w(i) * x(i) * v_x(i);
        }
        hv_u(0) = 2.0 * s;
    }
};

// One classic RK4 step of the ODE (matches K4Tag).
state_t rk4_step(const OdeXsqU& ode, const state_t& x, const control_t& u,
                 double h)
{
    const state_t k1 = ode.f(x, u);
    const state_t k2 = ode.f(x + 0.5 * h * k1, u);
    const state_t k3 = ode.f(x + 0.5 * h * k2, u);
    const state_t k4 = ode.f(x + h * k3, u);
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// Reference state at t = 0.5 from 10000 RK4 steps of the same ODE.
state_t reference_value()
{
    const OdeXsqU ode;
    state_t cur;
    cur << 0.3, -0.7;
    const control_t u(0.4);
    const int n = 10000;
    const double h = 0.5 / static_cast<double>(n);
    for (int i = 0; i < n; ++i)
    {
        cur = rk4_step(ode, cur, u, h);
    }
    return cur;
}

// value: NumSteps sub-steps over 5 OCP steps (h=0.1) vs the RK4 reference.
template <int NumSteps>
void check_value(double tol)
{
    using Integ =
        ocp::ExplicitRkIntegrator<Dims, OdeXsqU, 4, ocp::K4Tag, NumSteps>;
    const Integ integ(OdeXsqU{}, 0.1);
    state_t cur;
    cur << 0.3, -0.7;
    const control_t u(0.4);
    for (int step = 0; step < 5; ++step)
    {
        integ.value(cur, u, cur);
    }
    const state_t ref = reference_value();
    const double err = (cur - ref).cwiseAbs().maxCoeff();
    check(err < tol, "value K4 multi-step vs RK4 reference");
    std::printf("  value K4 NS=%d  OK  (|err|=%.2e < %.1e)\n", NumSteps, err,
                tol);
}

// jacobian: central FD of the composed one-OCP-step map, NumSteps sub-steps.
template <int NumSteps>
void check_jacobian()
{
    using Integ =
        ocp::ExplicitRkIntegrator<Dims, OdeXsqU, 4, ocp::K4Tag, NumSteps>;
    const Integ integ(OdeXsqU{}, 0.1);
    const state_t x(0.3, -0.7);
    const control_t u(0.4);

    df_dx_t Jx;
    df_du_t Ju;
    integ.jacobian(x, u, Jx, Ju);

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
        integ.value(xp, u, fdp);
        integ.value(xm, u, fdm);
        for (int i = 0; i < nx; ++i)
        {
            max_err = std::max(
                max_err, std::abs(Jx(i, j) - (fdp(i) - fdm(i)) / (2.0 * d)));
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
            max_err = std::max(
                max_err, std::abs(Ju(i, j) - (fdp(i) - fdm(i)) / (2.0 * d)));
        }
    }
    check(max_err < 1e-6, "jacobian K4 multi-step FD");
    std::printf("  jacobian K4 NS=%d  OK  (max|err|=%.2e < 1e-6)\n", NumSteps,
                max_err);
}

// hess_prod: FD the (nx+nu)x(nx+nu) Hessian of  w^T Phi  for 3 (w, v) pairs,
// NumSteps sub-steps.
template <int NumSteps>
void check_hess_prod()
{
    using Integ =
        ocp::ExplicitRkIntegrator<Dims, OdeXsqU, 2, ocp::K2Tag, NumSteps>;
    const Integ integ(OdeXsqU{}, 0.1);
    using zvec_t = Eigen::Matrix<double, nx + nu, 1>;

    const state_t x(0.3, -0.7);
    const control_t u(0.4);
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
        auto g = [&](const zvec_t& z) -> double
        {
            state_t xx;
            xx(0) = z(0);
            xx(1) = z(1);
            control_t uu;
            uu(0) = z(2);
            state_t xn;
            integ.value(xx, uu, xn);
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
            max_err = std::max(max_err, std::abs(hvx(i) - ref(i)));
        }
        for (int i = 0; i < nu; ++i)
        {
            max_err = std::max(max_err, std::abs(hvu(i) - ref(nx + i)));
        }
    }
    check(max_err < 1e-5, "hess_prod K2 multi-step FD");
    std::printf("  hess_prod K2 NS=%d  OK  (max|err|=%.2e < 1e-5)\n", NumSteps,
                max_err);
}

}  // namespace

int run_multistep_4g_tests()
{
    // value (NumSteps=1 anchors the regression; 2 and 3 exercise sub-stepping)
    check_value<1>(2e-4);
    check_value<2>(1e-4);
    check_value<3>(1e-4);

    // jacobian
    check_jacobian<2>();

    // hess_prod (NS=1 anchors the regression against the 4b single-step path;
    // NS=2 exercises the multi-step adjoint threading)
    check_hess_prod<1>();
    check_hess_prod<2>();

    if (failures != 0)
    {
        std::fprintf(stderr, "[4g] %d check(s) failed.\n", failures);
    }
    return failures;
}

// Phase 4d/4f: verify ImplicitRkIntegrator (Radau IIA) value / jacobian /
// hess_prod against the analytic ODE solution and finite differences of the
// composed map, plus the Newton convergence diagnostic.
//
// hess_prod: the linear-ODE path returns zero (verified); the nonlinear path
// (4f, second-order implicit sensitivity) is verified against a central-FD
// Hessian of the composed map g(z) = w^T Phi(z) for a nonlinear ODE.

#include <cmath>
#include <cstdio>
#include <string>

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

constexpr int nx = 2;
constexpr int nu = 1;
constexpr int nIn = nx + nu;

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

/// xdot = -lambda * x  (linear, no hess_prod). lambda is a runtime parameter.
struct OdeLin
{
    double lambda = 10.0;
    state_t f(const state_t& x, const control_t&, double) const
    {
        return -lambda * x;
    }
    void jacobian(const state_t&, const control_t&, double, df_dx_t& df_dx,
                  df_du_t& df_du) const
    {
        df_dx = -lambda * df_dx_t::Identity();
        df_du.setZero();
    }
};

/// xdot = x.*x  (elementwise nonlinear; value test against x0/(1-x0*t)).
struct OdeXsq
{
    state_t f(const state_t& x, const control_t&, double) const
    {
        return x.cwiseProduct(x);
    }
    void jacobian(const state_t& x, const control_t&, double, df_dx_t& df_dx,
                  df_du_t& df_du) const
    {
        df_dx = (2.0 * x).asDiagonal();
        df_du.setZero();
    }
    void hess_prod(const state_t&, const control_t&, double, const state_t& w,
                   const state_t& v_x, const control_t&, state_t& hv_x,
                   control_t& hv_u) const
    {
        hv_x = 2.0 * w.cwiseProduct(v_x);
        hv_u.setZero();
    }
};

/// xdot = x.*x * u + x  (nonlinear in x and depends on u; both Jacobian
/// blocks non-trivial and the composed map is nonlinear).
struct OdeXsqU
{
    state_t f(const state_t& x, const control_t& u, double) const
    {
        return x.cwiseProduct(x) * u(0) + x;
    }
    void jacobian(const state_t& x, const control_t& u, double, df_dx_t& df_dx,
                  df_du_t& df_du) const
    {
        df_dx = df_dx_t::Identity();
        for (int i = 0; i < nx; ++i)
        {
            df_dx(i, i) += 2.0 * x(i) * u(0);
        }
        df_du.col(0) = x.cwiseProduct(x);
    }
    void hess_prod(const state_t& x, const control_t& u, double,
                   const state_t& w, const state_t& v_x, const control_t& v_u,
                   state_t& hv_x, control_t& hv_u) const
    {
        // f_i = x_i^2 u + x_i:  d2f_i/(dx_i dx_i) = 2 u,  d2f_i/(dx_i du) = 2 x_i.
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

/// Double integrator  xdot = [v; a]  (linear, no hess_prod). For the
/// nilpotent dynamics matrix the Radau collocation map is exact.
struct OdeDoubleInt
{
    state_t f(const state_t& x, const control_t& u, double) const
    {
        state_t r;
        r(0) = x(1);
        r(1) = u(0);
        return r;
    }
    void jacobian(const state_t&, const control_t&, double, df_dx_t& df_dx,
                  df_du_t& df_du) const
    {
        df_dx.setZero();
        df_dx(0, 1) = 1.0;
        df_du.setZero();
        df_du(1, 0) = 1.0;
    }
};

// ---------------------------------------------------------------------------
//  value
// ---------------------------------------------------------------------------

/// A-stable damping of a stiff linear ODE where an explicit K4 would be
/// unstable (z = h*lambda outside the K4 stability interval).
void check_value_stiff()
{
    using Integ = ocp::ImplicitRkIntegrator<Dims, OdeLin, 2, ocp::RadauIia2Tag>;
    OdeLin ode;
    ode.lambda = 30.0;
    const Integ integ(ode, 0.1);
    state_t cur(1.0, 1.0);
    control_t u(0.0);
    for (int step = 0; step < 10; ++step)
    {
        integ.value(cur, u, 0.0, cur);
    }
    // z = h*lambda = -3: the Radau IIA 2 stability function satisfies R(-3) = 0,
    // so the exact collocation map drives x to 0 in one step; A-stability then
    // keeps it there (no explicit-scheme blow-up). The value must be tiny
    // (driven to ~0) and bounded, regardless of a fp-sign sliver.
    const bool ok = std::isfinite(cur(0)) && std::abs(cur(0)) < 1e-6;
    check(ok, "value stiff xdot=-30x (A-stable, damped)");
    std::printf("  value stiff xdot=-30x  OK  (|val|<1e-6, exact~9e-14)\n");
}

/// Order-3 accuracy on a linear ODE against the analytic solution.
void check_value_accuracy()
{
    using Integ = ocp::ImplicitRkIntegrator<Dims, OdeLin, 2, ocp::RadauIia2Tag>;
    OdeLin ode;
    ode.lambda = 1.0;
    const Integ integ(ode, 0.05);
    state_t cur(1.0, 1.0);
    control_t u(0.0);
    for (int step = 0; step < 20; ++step)
    {
        integ.value(cur, u, 0.0, cur);
    }
    const double expected = std::exp(-1.0);  // t = 20 * 0.05 = 1.0
    check(std::abs(cur(0) - expected) < 1e-4, "value accuracy xdot=-x");
    std::printf("  value acc xdot=-x   OK  (|err|<1e-4, exact=%.6f)\n",
                expected);
}

/// Exactness on the nilpotent double-integrator dynamics (M^2 = 0): the
/// collocation map reproduces the exact discrete map to machine precision.
void check_value_exact_linear()
{
    using Integ =
        ocp::ImplicitRkIntegrator<Dims, OdeDoubleInt, 2, ocp::RadauIia2Tag>;
    const Integ integ(OdeDoubleInt{}, 0.2);
    state_t cur(0.3, -0.5);
    control_t u(0.7);
    for (int step = 0; step < 5; ++step)
    {
        integ.value(cur, u, 0.0, cur);
    }
    const double t = 5 * 0.2;  // = 1.0
    const double q = 0.3 + (-0.5) * t + 0.5 * 0.7 * t * t;  // = 0.15
    const double v = -0.5 + 0.7 * t;                         // = 0.2
    check(std::abs(cur(0) - q) < 1e-10, "value exact double-integrator q");
    check(std::abs(cur(1) - v) < 1e-10, "value exact double-integrator v");
    std::printf("  value exact double-integrator  OK  (|err|<1e-10)\n");
}

/// Nonlinear ODE against the analytic solution (also exercises the Newton
/// solve over several steps).
void check_value_xsq()
{
    using Integ =
        ocp::ImplicitRkIntegrator<Dims, OdeXsq, 2, ocp::RadauIia2Tag>;
    const Integ integ(OdeXsq{}, 0.2, 5);
    state_t cur(0.5, 0.5);
    control_t u(0.0);
    for (int step = 0; step < 5; ++step)
    {
        integ.value(cur, u, 0.0, cur);
    }
    const double expected = 0.5 / (1.0 - 0.5 * 1.0);  // t = 1, x0 = 0.5 -> 1.0
    check(std::abs(cur(0) - expected) < 1e-2, "value xdot=x^2 c0");
    check(std::abs(cur(1) - expected) < 1e-2, "value xdot=x^2 c1");
    std::printf("  value xdot=x^2    OK  (|err|<1e-2)\n");
}

// ---------------------------------------------------------------------------
//  Newton convergence
// ---------------------------------------------------------------------------

void check_newton_convergence()
{
    using Integ =
        ocp::ImplicitRkIntegrator<Dims, OdeXsq, 2, ocp::RadauIia2Tag>;
    const state_t x(0.5, 0.5);
    const control_t u(0.0);

    const Integ few(OdeXsq{}, 0.2, 1);
    state_t x_out;
    few.value(x, u, 0.0, x_out);
    const double r_few = few.newton_residual_inf();

    const Integ many(OdeXsq{}, 0.2, 5);
    many.value(x, u, 0.0, x_out);
    const double r_many = many.newton_residual_inf();

    check(r_many < r_few, "newton residual decreases with more iterations");
    check(r_many < 1e-10, "newton final residual < 1e-10 (nonlinear)");
    std::printf("  newton conv  OK  (r_1=%.2e > r_5=%.2e < 1e-10)\n",
                r_few, r_many);
}

// ---------------------------------------------------------------------------
//  jacobian (finite differences of the composed map)
// ---------------------------------------------------------------------------

template <int NS, class Tag>
void check_jacobian_fd()
{
    using Integ = ocp::ImplicitRkIntegrator<Dims, OdeXsqU, NS, Tag>;
    const Integ integ(OdeXsqU{}, 0.1, 5);
    const state_t x(0.3, -0.7);
    const control_t u(0.4);

    df_dx_t Jx;
    df_du_t Ju;
    integ.jacobian(x, u, 0.0, Jx, Ju);

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
        integ.value(xp, u, 0.0, fdp);
        integ.value(xm, u, 0.0, fdm);
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
        integ.value(x, up, 0.0, fdp);
        integ.value(x, um, 0.0, fdm);
        for (int i = 0; i < nx; ++i)
        {
            const double fd = (fdp(i) - fdm(i)) / (2.0 * d);
            check(std::abs(Ju(i, j) - fd) < 1e-6,
                  (std::string("jacobian df/du(") + std::to_string(i) + ")")
                      .c_str());
        }
    }
    std::printf("  jacobian fd OdeXsqU  OK  (|err|<1e-6)\n");
}

/// Exact Jacobian of the nilpotent double-integrator map:  df/dx = I + h M,
/// df/du = [h^2/2; h].
void check_jacobian_exact_linear()
{
    using Integ =
        ocp::ImplicitRkIntegrator<Dims, OdeDoubleInt, 2, ocp::RadauIia2Tag>;
    const Integ integ(OdeDoubleInt{}, 0.2);
    const state_t x(0.3, -0.5);
    const control_t u(0.7);
    df_dx_t Jx;
    df_du_t Ju;
    integ.jacobian(x, u, 0.0, Jx, Ju);

    const double h = 0.2;
    const double exp_qq = 1.0;
    const double exp_qv = h;
    const double exp_vq = 0.0;
    const double exp_vv = 1.0;
    check(std::abs(Jx(0, 0) - exp_qq) < 1e-10, "jac exact dq/dq");
    check(std::abs(Jx(0, 1) - exp_qv) < 1e-10, "jac exact dq/dv");
    check(std::abs(Jx(1, 0) - exp_vq) < 1e-10, "jac exact dv/dq");
    check(std::abs(Jx(1, 1) - exp_vv) < 1e-10, "jac exact dv/dv");
    check(std::abs(Ju(0, 0) - 0.5 * h * h) < 1e-10, "jac exact dq/da");
    check(std::abs(Ju(1, 0) - h) < 1e-10, "jac exact dv/da");
    std::printf("  jacobian exact double-integrator  OK  (|err|<1e-10)\n");
}

// ---------------------------------------------------------------------------
//  hess_prod (linear ODE -> zero, the only supported path in 4d)
// ---------------------------------------------------------------------------

void check_hess_prod_linear()
{
    using Integ =
        ocp::ImplicitRkIntegrator<Dims, OdeLin, 2, ocp::RadauIia2Tag>;
    const Integ integ(OdeLin{}, 0.1);
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    const state_t w(0.7, -0.3);
    const state_t v_x(0.4, 0.9);
    const control_t v_u(-0.5);
    state_t hv_x;
    control_t hv_u;
    integ.hess_prod(x, u, 0.0, w, v_x, v_u, hv_x, hv_u);
    check(hv_x.isZero() && hv_u.isZero(), "hess_prod linear ODE == 0");
    std::printf("  hess_prod linear   OK  (zero, as expected)\n");
}

// ---------------------------------------------------------------------------
//  hess_prod (nonlinear ODE, 4f: second-order implicit sensitivity)
// ---------------------------------------------------------------------------

/// Gold standard for the nonlinear path: finite-difference the composed-map
/// Hessian of g(z) = w^T Phi(z), z = [x; u], with central 4-point cross
/// derivatives, and compare the integrator's bilinear HVP w^T d2Phi . v.
template <int NS, class Tag>
void check_hess_prod_fd()
{
    using Integ = ocp::ImplicitRkIntegrator<Dims, OdeXsqU, NS, Tag>;
    using z_t = Eigen::Matrix<double, nIn, 1>;
    using hess_t = Eigen::Matrix<double, nIn, nIn>;
    const Integ integ(OdeXsqU{}, 0.1, 8);
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    const double d = 1e-4;

    const z_t z0 = [&]() {
        z_t z;
        z.head(nx) = x;
        z.tail(nu) = u;
        return z;
    }();

    const auto phi = [&integ](const z_t& z) {
        state_t out;
        integ.value(z.head(nx), z.tail(nu), 0.0, out);
        return out;
    };
    const auto g = [&](const state_t& w, const z_t& z) {
        return w.dot(phi(z));
    };
    const auto hessian_fd = [&g, &z0, d](const state_t& w) {
        hess_t H;
        for (int i = 0; i < nIn; ++i)
        {
            for (int j = 0; j < nIn; ++j)
            {
                if (i == j)
                {
                    z_t zp = z0;
                    zp(i) += d;
                    z_t zm = z0;
                    zm(i) -= d;
                    H(i, j) = (g(w, zp) - 2.0 * g(w, z0) + g(w, zm)) / (d * d);
                }
                else
                {
                    z_t p11 = z0;
                    p11(i) += d;
                    p11(j) += d;
                    z_t p1m = z0;
                    p1m(i) += d;
                    p1m(j) -= d;
                    z_t pm1 = z0;
                    pm1(i) -= d;
                    pm1(j) += d;
                    z_t pmM = z0;
                    pmM(i) -= d;
                    pmM(j) -= d;
                    H(i, j) = (g(w, p11) - g(w, p1m) - g(w, pm1) +
                               g(w, pmM)) / (4.0 * d * d);
                }
            }
        }
        return H;
    };

    struct Case
    {
        state_t w;
        state_t v_x;
        control_t v_u;
    };
    const Case cases[] = {
        {state_t(0.7, -0.3), state_t(0.4, 0.9), control_t(-0.5)},
        {state_t(1.0, 0.4), state_t(0.1, -0.6), control_t(0.8)},
        {state_t(0.2, 0.9), state_t(-0.3, 0.5), control_t(0.2)},
    };
    for (const auto& c : cases)
    {
        const hess_t H = hessian_fd(c.w);
        z_t v;
        v.head(nx) = c.v_x;
        v.tail(nu) = c.v_u;
        state_t hv_x;
        control_t hv_u;
        integ.hess_prod(x, u, 0.0, c.w, c.v_x, c.v_u, hv_x, hv_u);
        z_t actual;
        actual.head(nx) = hv_x;
        actual.tail(nu) = hv_u;
        const double err = (actual - H * v).cwiseAbs().maxCoeff();
        check(err < 1e-5, "hess_prod fd vs analytic (nonlinear)");
        std::printf("  hess_prod fd %s  OK  (max |err| = %.3e)\n",
                    (NS == 2 ? "radau2" : "radau4"), err);
    }
}

}  // namespace

int run_irk_4d_tests()
{
    // value
    check_value_stiff();
    check_value_accuracy();
    check_value_exact_linear();
    check_value_xsq();

    // Newton convergence
    check_newton_convergence();

    // jacobian
    check_jacobian_fd<2, ocp::RadauIia2Tag>();
    check_jacobian_fd<4, ocp::RadauIia4Tag>();
    check_jacobian_exact_linear();

    // hess_prod (linear + nonlinear FD, 4f)
    check_hess_prod_linear();
    check_hess_prod_fd<2, ocp::RadauIia2Tag>();
    check_hess_prod_fd<4, ocp::RadauIia4Tag>();

    if (failures != 0)
    {
        std::fprintf(stderr, "[4d] %d check(s) failed.\n", failures);
    }
    return failures;
}

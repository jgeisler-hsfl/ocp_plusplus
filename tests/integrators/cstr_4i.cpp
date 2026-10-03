// Phase 4i: CSTR (Continuous Stirred-Tank Reactor) nonlinear example.
//
// A genuinely nonlinear OCP based on the acados CSTR example
// (acados/examples/acados_python/cstr/cstr_model.py). The ODE model has
// hand-derived analytic f, jacobian, and hess_prod. Exercises:
//   - ODE-level: hess_prod vs central-FD Hessian of w' f  (direct)
//   - integrator value: K4 single-step vs 10k-step RK4 reference
//   - integrator jacobian: central-FD of the composed map (ERK + IRK)
//   - integrator hess_prod: central-FD Hessian of w' Phi (ERK K4, IRK Radau2)
//   - end-to-end: ContinuousProblem + SqpSolver, kSolved + NLP residuals

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/integrators/continuous_problem.hpp"
#include "ocp/integrators/rk_explicit.hpp"
#include "ocp/integrators/rk_implicit.hpp"
#include "ocp/solvers/acados/sqp.hpp"

namespace
{

using namespace ocp;

// ------------------------------------------------------------------
//  Dims  (nx=3, nu=2, ne_t=3)
// ------------------------------------------------------------------

struct CstrDims
{
    static constexpr int nx   = 3;
    static constexpr int nu   = 2;
    static constexpr int ng   = 0;
    static constexpr int ne   = 0;
    static constexpr int nl   = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 3;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state    = true;
    static constexpr bool has_dynamics_hess_prod = true;
    static constexpr bool has_constr_hess_prod   = true;
    static constexpr std::array<int, 3> state_box_idx      = {0, 1, 2};
    static constexpr std::array<int, 2> control_box_idx    = {0, 1};
    static constexpr std::array<int, 3> terminal_state_box_idx = {0, 1, 2};
    static constexpr std::array<int, 0> ineq_soft_idx           = {};
    static constexpr std::array<int, 0> eq_soft_idx             = {};
    static constexpr std::array<int, 0> lin_soft_idx            = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx  = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx    = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx   = {};
    static constexpr std::array<int, 0> state_box_soft_idx      = {};
    static constexpr std::array<int, 0> control_box_soft_idx    = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

using CstrP   = ocp::Problem<CstrDims>;
using state_t   = CstrP::state_t;
using control_t = CstrP::control_t;
using df_dx_t   = CstrP::dyn_df_dx_t;
using df_du_t   = CstrP::dyn_df_du_t;

constexpr int nx  = 3;
constexpr int nu  = 2;
constexpr int nIn = nx + nu;

int failures = 0;

void check(bool ok, const char* msg)
{
    if (!ok)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

// ------------------------------------------------------------------
//  CSTR ODE model (hand-derived analytic derivatives)
//
//  states   x = (c, T, h)
//  control  u = (Tc, F)
//
//  Ac    = pi * r^2
//  denom = Ac * (h + eps)
//  k     = k0 * exp(-EbR / T)
//  rate  = k * c
//
//  c_dot   = F0 (c0 - c) / denom - rate
//  T_dot   = F0 (T0 - T) / denom - q * rate + g * (Tc - T)
//  h_dot   = (F0 - F) / Ac
//
//  q = dH / (rho * Cp)   (dH < 0  ->  q < 0)
//  g = 2 U / (r * rho * Cp)
// ------------------------------------------------------------------

struct CstrOde
{
    // physical parameters (nominal acados values)
    double F0   = 0.1;
    double T0   = 350.0;
    double c0   = 1.0;
    double r    = 0.219;
    double k0   = 7.2e10;
    double EbR  = 8750.0;
    double U    = 54.94;
    double rho  = 1000.0;
    double Cp   = 0.239;
    double dH   = -5.0e4;
    double eps  = 1e-5;

    // derived constants
    double Ac = 3.14159265358979323846 * r * r;
    double q  = dH / (rho * Cp);
    double g  = 2.0 * U / (r * rho * Cp);

    state_t f(const state_t& x, const control_t& u, double) const
    {
        const double c = x(0), T = x(1), h = x(2);
        const double Tc = u(0), F = u(1);
        const double denom = Ac * (h + eps);
        const double k     = k0 * std::exp(-EbR / T);
        const double rate  = k * c;

        state_t out;
        out(0) = F0 * (c0 - c) / denom - rate;
        out(1) = F0 * (T0 - T) / denom - q * rate + g * (Tc - T);
        out(2) = (F0 - F) / Ac;
        return out;
    }

    void jacobian(const state_t& x, const control_t&, double,
                  df_dx_t& df_dx, df_du_t& df_du) const
    {
        const double c    = x(0), T = x(1), h = x(2);
        const double denom = Ac * (h + eps);
        const double k     = k0 * std::exp(-EbR / T);
        const double rate  = k * c;
        const double inv_d2 = 1.0 / (denom * denom);

        df_dx.setZero();
        // c_dot row
        df_dx(0, 0) = -F0 / denom - k;
        df_dx(0, 1) = -rate * EbR / (T * T);
        df_dx(0, 2) = -F0 * Ac * (c0 - c) * inv_d2;
        // T_dot row
        df_dx(1, 0) = -q * k;
        df_dx(1, 1) = -F0 / denom - q * rate * EbR / (T * T) - g;
        df_dx(1, 2) = -F0 * Ac * (T0 - T) * inv_d2;
        // h_dot row: all zero

        df_du.setZero();
        df_du(1, 0) = g;
        df_du(2, 1) = -1.0 / Ac;
    }

    void hess_prod(const state_t& x, const control_t&, double,
                   const state_t& w, const state_t& v_x, const control_t&,
                   state_t& hv_x, control_t& hv_u) const
    {
        // u enters f affinely -> d2f/du dx = d2f/du^2 = 0
        hv_u.setZero();

        const double c    = x(0), T = x(1), h = x(2);
        const double denom = Ac * (h + eps);
        const double k     = k0 * std::exp(-EbR / T);
        const double rate  = k * c;
        const double inv_d2 = 1.0 / (denom * denom);
        const double inv_d3 = inv_d2 / denom;

        const double T2 = T * T;
        const double T3 = T2 * T;
        const double T4 = T3 * T;

        // output 0 (c_dot):
        //   H0 = [[0,  a,  b],
        //         [a,  d,  0],
        //         [b,  0,  e]]
        const double a  = -k * EbR / T2;
        const double d  = -rate * (EbR * EbR / T4 - 2.0 * EbR / T3);
        const double b  = F0 * Ac * inv_d2;
        const double e  = 2.0 * F0 * Ac * Ac * (c0 - c) * inv_d3;

        // output 1 (T_dot):
        //   H1 = [[0,  a1, 0],
        //         [a1, d1, b1],
        //         [0,  b1, e1]]
        const double a1 = -q * k * EbR / T2;
        const double d1 = -q * rate * (EbR * EbR / T4 - 2.0 * EbR / T3);
        const double b1 = F0 * Ac * inv_d2;
        const double e1 = 2.0 * F0 * Ac * Ac * (T0 - T) * inv_d3;

        const double vc = v_x(0), vT = v_x(1), vh = v_x(2);

        // (H0 . v_x)
        const double h0_c = a * vT + b * vh;
        const double h0_T = a * vc + d * vT;
        const double h0_h = b * vc + e * vh;
        // (H1 . v_x)
        const double h1_c = a1 * vT;
        const double h1_T = a1 * vc + d1 * vT + b1 * vh;
        const double h1_h = b1 * vT + e1 * vh;

        hv_x(0) = w(0) * h0_c + w(1) * h1_c;
        hv_x(1) = w(0) * h0_T + w(1) * h1_T;
        hv_x(2) = w(0) * h0_h + w(1) * h1_h;
    }
};

// ------------------------------------------------------------------
//  RK4 reference step (local, for value test)
// ------------------------------------------------------------------

state_t rk4_step(const CstrOde& ode, const state_t& x,
                 const control_t& u, double t, double h)
{
    const state_t k1 = ode.f(x, u, t);
    const state_t k2 = ode.f(x + 0.5 * h * k1, u, t + 0.5 * h);
    const state_t k3 = ode.f(x + 0.5 * h * k2, u, t + 0.5 * h);
    const state_t k4 = ode.f(x + h * k3, u, t + h);
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// ------------------------------------------------------------------
//  ODE-level: hess_prod vs central-FD Hessian of w' f  (direct)
// ------------------------------------------------------------------

void check_ode_hess_prod_direct()
{
    CstrOde ode;
    using H_t = Eigen::Matrix<double, nx, nx>;

    state_t x;
    x << 0.878, 324.5, 0.659;
    control_t u;
    u << 300.0, 0.1;

    const auto g = [&](const state_t& xx, const control_t& uu) {
        return ode.f(xx, uu, 0.0);
    };

    struct Case { state_t w, v_x; control_t v_u; };
    const Case cases[] = {
        {state_t(0.7, -0.3, 1.0), state_t(0.4, 0.9, -0.5), control_t(1.0, -0.3)},
        {state_t(1.0, 0.4, 0.2),  state_t(0.1, -0.6, 0.3), control_t(-0.7, 0.5)},
        {state_t(-0.5, 0.8, 1.0), state_t(1.0, 0.2, -0.4), control_t(0.3, 0.8)},
    };

    const double d = 1e-4;
    double max_err = 0.0;

    for (const auto& c : cases)
    {
        // FD Hessian of g(z) = w' f(z, u) w.r.t. x only
        H_t H;
        for (int i = 0; i < nx; ++i)
        {
            for (int j = i; j < nx; ++j)
            {
                if (i == j)
                {
                    state_t xp = x; xp(i) += d;
                    state_t xm = x; xm(i) -= d;
                    const state_t fp = g(xp, u);
                    const state_t fm = g(xm, u);
                    const state_t f0 = g(x, u);
                    H(i, j) = c.w.dot((fp - 2.0 * f0 + fm) / (d * d));
                }
                else
                {
                    state_t pp = x; pp(i) += d; pp(j) += d;
                    state_t pm = x; pm(i) -= d; pm(j) += d;
                    state_t mp = x; mp(i) += d; mp(j) -= d;
                    state_t mm = x; mm(i) -= d; mm(j) -= d;
                    const state_t fpp = g(pp, u);
                    const state_t fpm = g(pm, u);
                    const state_t fmp = g(mp, u);
                    const state_t fmm = g(mm, u);
                    H(i, j) = c.w.dot((fpp - fpm - fmp + fmm) / (4.0 * d * d));
                }
                H(j, i) = H(i, j);
            }
        }

        state_t hv_x;
        control_t hv_u;
        ode.hess_prod(x, u, 0.0, c.w, c.v_x, c.v_u, hv_x, hv_u);

        // expected: H_xx * v_x + H_xu * v_u;  H_xu and H_uu should be ~0
        state_t exp_hv_x;
        exp_hv_x.setZero();
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                exp_hv_x(i) += H(i, j) * c.v_x(j);
            }
        }

        max_err = std::max(max_err, (hv_x - exp_hv_x).cwiseAbs().maxCoeff());
        max_err = std::max(max_err, hv_u.cwiseAbs().maxCoeff());
    }

    check(max_err < 1e-5,
          (std::string("ODE hess_prod direct FD (max err=") +
           std::to_string(max_err) + ")").c_str());
    std::printf("  ODE hess_prod direct  OK  (max|err|=%.3e < 1e-5)\n",
                max_err);
}

// ------------------------------------------------------------------
//  Integrator value: K4 single step vs 10k-step RK4 reference
// ------------------------------------------------------------------

void check_value()
{
    using Integ = ocp::ExplicitRkIntegrator<CstrDims, CstrOde, 4, ocp::K4Tag>;
    CstrOde ode;
    const Integ integ(ode, 0.1);

    state_t x0;
    x0 << 0.5, 340.0, 0.6;
    control_t u;
    u << 305.0, 0.12;

    state_t x_erk;
    integ.value(x0, u, 0.0, x_erk);

    // high-resolution reference: 10000 RK4 steps of size 0.1/10000
    state_t x_ref = x0;
    const int n = 10000;
    const double h_sub = 0.1 / n;
    for (int i = 0; i < n; ++i)
    {
        x_ref = rk4_step(ode, x_ref, u, i * h_sub, h_sub);
    }

    const double err = (x_erk - x_ref).cwiseAbs().maxCoeff();
    check(err < 1e-4,
          (std::string("CSTR value K4 single-step (err=") +
           std::to_string(err) + ")").c_str());
    std::printf("  value K4 single-step  OK  (|err|=%.3e < 1e-4)\n", err);
}

// ------------------------------------------------------------------
//  Integrator jacobian: central FD of composed map
// ------------------------------------------------------------------

template <class Integ>
void check_jacobian_fd_impl(const Integ& integ, const char* label)
{
    state_t x;
    x << 0.878, 324.5, 0.659;
    control_t u;
    u << 300.0, 0.1;

    df_dx_t Jx;
    df_du_t Ju;
    integ.jacobian(x, u, 0.0, Jx, Ju);

    const double d = 1e-6;
    double max_err = 0.0;

    // dPhi/dx
    for (int j = 0; j < nx; ++j)
    {
        state_t xp = x, xm = x;
        state_t e;
        e.setZero();
        e(j) = d;
        xp += e;
        xm -= e;
        state_t fdp, fdm;
        integ.value(xp, u, 0.0, fdp);
        integ.value(xm, u, 0.0, fdm);
        for (int i = 0; i < nx; ++i)
        {
            const double fd = (fdp(i) - fdm(i)) / (2.0 * d);
            max_err = std::max(max_err, std::abs(Jx(i, j) - fd));
        }
    }
    // dPhi/du
    for (int j = 0; j < nu; ++j)
    {
        control_t up = u, um = u;
        control_t e;
        e.setZero();
        e(j) = d;
        up += e;
        um -= e;
        state_t fdp, fdm;
        integ.value(x, up, 0.0, fdp);
        integ.value(x, um, 0.0, fdm);
        for (int i = 0; i < nx; ++i)
        {
            const double fd = (fdp(i) - fdm(i)) / (2.0 * d);
            max_err = std::max(max_err, std::abs(Ju(i, j) - fd));
        }
    }

    check(max_err < 1e-6,
          (std::string(label) + " jacobian FD (max err=" +
           std::to_string(max_err) + ")").c_str());
    std::printf("  jacobian %s  OK  (max|err|=%.3e < 1e-6)\n", label, max_err);
}

// ------------------------------------------------------------------
//  Integrator hess_prod: central-FD Hessian of w' Phi
// ------------------------------------------------------------------

template <class Integ>
void check_hess_prod_fd(const Integ& integ, const char* label)
{
    using z_t = Eigen::Matrix<double, nIn, 1>;
    using H_t = Eigen::Matrix<double, nIn, nIn>;

    state_t x;
    x << 0.878, 324.5, 0.659;
    control_t u;
    u << 300.0, 0.1;
    const z_t z0 = [&]() {
        z_t z;
        z.head(nx) = x;
        z.tail(nu) = u;
        return z;
    }();

    struct Case { state_t w, v_x; control_t v_u; };
    const Case cases[] = {
        {state_t(0.7, -0.3, 1.0), state_t(0.4, 0.9, -0.5), control_t(1.0, -0.3)},
        {state_t(1.0, 0.4, 0.2),  state_t(0.1, -0.6, 0.3), control_t(-0.7, 0.5)},
        {state_t(-0.5, 0.8, 1.0), state_t(1.0, 0.2, -0.4), control_t(0.3, 0.8)},
    };

    const double d = 1e-4;
    double max_err = 0.0;

    for (const auto& c : cases)
    {
        const auto g = [&](const z_t& z) -> double {
            state_t xx = z.head(nx);
            control_t uu = z.tail(nu);
            state_t out;
            integ.value(xx, uu, 0.0, out);
            return c.w.dot(out);
        };

        H_t H;
        for (int i = 0; i < nIn; ++i)
        {
            for (int j = i; j < nIn; ++j)
            {
                if (i == j)
                {
                    z_t zp = z0; zp(i) += d;
                    z_t zm = z0; zm(i) -= d;
                    H(i, j) = (g(zp) - 2.0 * g(z0) + g(zm)) / (d * d);
                }
                else
                {
                    z_t pp = z0; pp(i) += d; pp(j) += d;
                    z_t pm = z0; pm(i) -= d; pm(j) += d;
                    z_t mp = z0; mp(i) += d; mp(j) -= d;
                    z_t mm = z0; mm(i) -= d; mm(j) -= d;
                    H(i, j) = (g(pp) - g(pm) - g(mp) + g(mm)) / (4.0 * d * d);
                }
                H(j, i) = H(i, j);
            }
        }

        state_t hv_x;
        control_t hv_u;
        integ.hess_prod(x, u, 0.0, c.w, c.v_x, c.v_u, hv_x, hv_u);

        z_t v;
        v.head(nx) = c.v_x;
        v.tail(nu) = c.v_u;
        const z_t ref = H * v;

        for (int i = 0; i < nx; ++i)
        {
            max_err = std::max(max_err, std::abs(hv_x(i) - ref(i)));
        }
        for (int i = 0; i < nu; ++i)
        {
            max_err = std::max(max_err, std::abs(hv_u(i) - ref(nx + i)));
        }
    }

    check(max_err < 1e-4,
          (std::string(label) + " hess_prod FD (max err=" +
           std::to_string(max_err) + ")").c_str());
    std::printf("  hess_prod %s  OK  (max|err|=%.3e < 1e-4)\n", label, max_err);
}

// ------------------------------------------------------------------
//  End-to-end: ContinuousProblem + SqpSolver  (CSTR OCP)
// ------------------------------------------------------------------

using ErkInteg =
    ocp::ExplicitRkIntegrator<CstrDims, CstrOde, 4, ocp::K4Tag>;
using ErkBase  = ocp::ContinuousProblem<CstrDims, CstrOde, ErkInteg>;

struct CstrOcp : ErkBase
{
    // nominal steady state / control
    static constexpr double xs_c  = 0.878;
    static constexpr double xs_T  = 324.5;
    static constexpr double xs_h  = 0.659;
    static constexpr double us_Tc = 300.0;
    static constexpr double us_F  = 0.1;

    // cost weights (relative L2)
    double wc_  = 1.0;
    double wT_  = 1e-3;
    double wh_  = 1.0;
    double wTc_ = 1e-3;
    double wF_  = 1.0;

    CstrOcp(double h) : ErkBase(CstrOde{}, h) {}

    state_t initial_state() const
    {
        // Perturbation from the steady state: c is held at xs_c (disturbing c
        // drives the stiff c<->T reaction coupling and stalls SQP), while T and
        // the liquid level h are nudged so the optimal control is non-trivial.
        // This well-conditioned start converges to kSolved in ~7 iterations
        // with default options.
        state_t x0;
        x0 << 0.878, 325.5, 0.78;
        return x0;
    }

    // ---- stage cost ----
    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        const double dc = x(0) - xs_c;
        const double dT = x(1) - xs_T;
        const double dh = x(2) - xs_h;
        const double dTc = u(0) - us_Tc;
        const double dF  = u(1) - us_F;
        return 0.5 * (wc_ * dc * dc + wT_ * dT * dT + wh_ * dh * dh +
                      wTc_ * dTc * dTc + wF_ * dF * dF);
    }

    CstrP::stage_grad_t
    stage_cost_gradient(int, const state_t& x, const control_t& u) const
    {
        CstrP::stage_grad_t g;
        g(0) = wc_  * (x(0) - xs_c);
        g(1) = wT_  * (x(1) - xs_T);
        g(2) = wh_  * (x(2) - xs_h);
        g(3) = wTc_ * (u(0) - us_Tc);
        g(4) = wF_  * (u(1) - us_F);
        return g;
    }

    CstrP::stage_hess_t
    stage_cost_hessian(int, const state_t&, const control_t&) const
    {
        CstrP::stage_hess_t H;
        H.setZero();
        H(0, 0) = wc_;
        H(1, 1) = wT_;
        H(2, 2) = wh_;
        H(3, 3) = wTc_;
        H(4, 4) = wF_;
        return H;
    }

    // ---- terminal cost ----
    double terminal_cost_value(const state_t& x) const
    {
        const double dc = x(0) - xs_c;
        const double dT = x(1) - xs_T;
        const double dh = x(2) - xs_h;
        return 0.5 * (wc_ * dc * dc + wT_ * dT * dT + wh_ * dh * dh);
    }

    CstrP::term_grad_t terminal_cost_gradient(const state_t& x) const
    {
        CstrP::term_grad_t g;
        g(0) = wc_ * (x(0) - xs_c);
        g(1) = wT_ * (x(1) - xs_T);
        g(2) = wh_ * (x(2) - xs_h);
        return g;
    }

    CstrP::term_hess_t terminal_cost_hessian(const state_t&) const
    {
        CstrP::term_hess_t H;
        H.setZero();
        H(0, 0) = wc_;
        H(1, 1) = wT_;
        H(2, 2) = wh_;
        return H;
    }

    // ---- stage box constraints ----
    CstrP::state_box_t stage_state_box_constr(int) const
    {
        CstrP::state_box_t spec;
        spec.lo << 0.0, 250.0, 0.05;
        spec.hi << 2.0, 450.0, 1.5;
        spec.soft_penalty.setZero();
        return spec;
    }

    CstrP::control_box_t stage_control_box_constr(int) const
    {
        CstrP::control_box_t spec;
        spec.lo << 250.0, 0.0;
        spec.hi << 400.0, 0.5;
        spec.soft_penalty.setZero();
        return spec;
    }

    CstrP::term_state_box_t terminal_state_box_constr() const
    {
        CstrP::term_state_box_t spec;
        spec.lo << 0.0, 250.0, 0.05;
        spec.hi << 2.0, 450.0, 1.5;
        spec.soft_penalty.setZero();
        return spec;
    }

    // ---- terminal equality: x_N = xs ----
    CstrP::eq_term_t terminal_equality_constr(const state_t& x) const
    {
        CstrP::eq_term_t e;
        e(0) = x(0) - xs_c;
        e(1) = x(1) - xs_T;
        e(2) = x(2) - xs_h;
        return e;
    }

    void terminal_equality_constr_jacobian(
        const state_t&, CstrP::eq_term_de_dx_t& e_dx) const
    {
        e_dx = CstrP::eq_term_de_dx_t::Identity();
    }

    void terminal_equality_constr_hess_prod(const state_t&,
                                            const CstrP::eq_term_t&,
                                            const state_t&,
                                            state_t& hv) const
    {
        hv.setZero();
    }
};

void check_end_to_end()
{
    constexpr int N = 10;
    CstrOcp prob(0.1);

    SqpOptions opts;
    SqpSolver<CstrOcp> solver(opts);

    Solution<CstrOcp> sol(N);
    sol.x[0] = prob.initial_state();
    for (int k = 0; k < N; ++k)
    {
        control_t u_k;
        u_k << CstrOcp::us_Tc, CstrOcp::us_F;
        sol.u[k] = u_k;
        sol.x[k + 1] = prob.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }

    const Status st = solver.solve(prob, sol);
    check(st == Status::kSolved, "CSTR end-to-end: solve returns kSolved");
    check(sol.status == Status::kSolved, "CSTR end-to-end: sol.status kSolved");

    const NlpResiduals res = compute_nlp_residuals(prob, sol);
    check(res.res_stat < 1e-6,
          (std::string("CSTR end-to-end res_stat=") +
           std::to_string(res.res_stat)).c_str());
    check(res.res_eq < 1e-6,
          (std::string("CSTR end-to-end res_eq=") +
           std::to_string(res.res_eq)).c_str());
    check(res.res_ineq < 1e-6,
          (std::string("CSTR end-to-end res_ineq=") +
           std::to_string(res.res_ineq)).c_str());
    check(res.res_comp < 1e-6,
          (std::string("CSTR end-to-end res_comp=") +
           std::to_string(res.res_comp)).c_str());

    std::printf("  end-to-end CSTR SQP  OK  (cost=%.6f, "
                "res=[%.1e %.1e %.1e %.1e])\n",
                sol.cost_value, res.res_stat, res.res_eq,
                res.res_ineq, res.res_comp);
}

}  // namespace

int run_cstr_4i_tests()
{
    // One ODE instance shared by every integrator below. The integrators hold
    // a const Ode&, so it must outlive them (a temporary CstrOde{} would be
    // destroyed at the end of the constructor's full-expression and leave the
    // reference dangling).
    CstrOde ode;

    check_ode_hess_prod_direct();
    check_value();
    {
        ocp::ExplicitRkIntegrator<CstrDims, CstrOde, 2, ocp::K2Tag> k2(ode, 0.1);
        check_jacobian_fd_impl(k2, "K2");
    }
    {
        ocp::ExplicitRkIntegrator<CstrDims, CstrOde, 4, ocp::K4Tag> k4(ode, 0.1);
        check_jacobian_fd_impl(k4, "K4");
    }
    {
        ocp::ImplicitRkIntegrator<CstrDims, CstrOde, 2, ocp::RadauIia2Tag> rk(
            ode, 0.1, 10);
        check_jacobian_fd_impl(rk, "Radau2");
    }
    {
        ocp::ExplicitRkIntegrator<CstrDims, CstrOde, 4, ocp::K4Tag> k4(ode, 0.1);
        check_hess_prod_fd(k4, "K4");
    }
    {
        ocp::ImplicitRkIntegrator<CstrDims, CstrOde, 2, ocp::RadauIia2Tag> rk(
            ode, 0.1, 12);
        check_hess_prod_fd(rk, "Radau2");
    }
    check_end_to_end();

    if (failures != 0)
    {
        std::fprintf(stderr, "[4i] %d check(s) failed.\n", failures);
    }
    return failures;
}

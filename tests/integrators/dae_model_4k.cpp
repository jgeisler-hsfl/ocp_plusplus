// Phase 4k: verify the DAE model contract (dae_model.hpp).
//
// (1) OdeToDae: the wrapped residual / Jacobian / HVP must agree exactly
//     with the underlying ODE model (F = xdot - f, dFxdot = I, no
//     xdot-Hessian), and the dae_supports_hess_prod trait must track the
//     ODE's hess_prod presence (compile-time).
// (2) A directly written DAE that is nonlinear in xdot (xdot-xdot and
//     xdot-u cross terms) checked against 5-point finite differences:
//     the full F_jac block layout and the full F_hess_prod bilinear form
//     (reconstructed from FD of the individual residual components along
//     the unit z = (x, xdot, u) directions, then contracted with random
//     (w, v)).

#include <cmath>
#include <cstdio>

#include "ocp/integrators/dae_model.hpp"

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
constexpr int nZ = 2 * nx + nu;  // z = (x, xdot, u)

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
//  ODE models for the OdeToDae wrap
// ---------------------------------------------------------------------------

/// xdot = x.*x * u + x + t  (nonlinear in x, depends on u and t; has
/// hess_prod).  Mirrors rk_implicit_4d OdeXsqU plus a time shift.
struct OdeQuadU
{
    state_t f(const state_t& x, const control_t& u, double t) const
    {
        return x.cwiseProduct(x) * u(0) + x + t * state_t::Ones();
    }
    void jacobian(const state_t& x, const control_t& u, double,
                  df_dx_t& df_dx, df_du_t& df_du) const
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
        // f_i = x_i^2 u + x_i + t:  d2f_i/(dx_i dx_i) = 2 u,
        // d2f_i/(dx_i du) = 2 x_i.
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

/// xdot = -lambda * x  (linear, no hess_prod).
struct OdeLin
{
    double lambda = 3.0;
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

// ---------------------------------------------------------------------------
//  Directly written DAE (nonlinear in xdot)
//
//     F0 = xdot0 - x0*sin(x1) + u * xdot0^2 + x0 * xdot1
//     F1 = xdot1 - x1*u       + xdot0 * xdot1
//
// ---------------------------------------------------------------------------

struct DaeSin
{
    state_t F(const state_t& x, const state_t& xdot, const control_t& u,
              double) const
    {
        state_t r;
        r(0) = xdot(0) - x(0) * std::sin(x(1)) + u(0) * xdot(0) * xdot(0)
            + x(0) * xdot(1);
        r(1) = xdot(1) - x(1) * u(0) + xdot(0) * xdot(1);
        return r;
    }

    void F_jac(const state_t& x, const state_t& xdot, const control_t& u,
               double, df_dx_t& dFdx, df_dx_t& dFxdot, df_du_t& dFdu) const
    {
        dFdx(0, 0) = -std::sin(x(1)) + xdot(1);
        dFdx(0, 1) = -x(0) * std::cos(x(1));
        dFdx(1, 0) = 0.0;
        dFdx(1, 1) = -u(0);
        dFxdot(0, 0) = 1.0 + 2.0 * u(0) * xdot(0);
        dFxdot(0, 1) = x(0);
        dFxdot(1, 0) = xdot(1);
        dFxdot(1, 1) = 1.0 + xdot(0);
        dFdu(0, 0) = xdot(0) * xdot(0);
        dFdu(1, 0) = -x(1);
    }

    // Bilinear HVP  w^T d2F/dz^2 (v_x, v_xdot, v_u),  z = (x, xdot, u),
    // hand-derived from the two residual components above.
    void F_hess_prod(const state_t& x, const state_t& xdot,
                     const control_t& u, double, const state_t& w,
                     const state_t& v_x, const state_t& v_xdot,
                     const control_t& v_u, state_t& hv_x, state_t& hv_xdot,
                     control_t& hv_u) const
    {
        const double c1 = std::cos(x(1));
        const double s1 = std::sin(x(1));
        hv_x(0) = w(0) * (-c1 * v_x(1) + v_xdot(1));
        hv_x(1) = w(0) * (-c1 * v_x(0) + x(0) * s1 * v_x(1))
            - w(1) * v_u(0);
        hv_xdot(0) =
            w(0) * (2.0 * u(0) * v_xdot(0) + 2.0 * xdot(0) * v_u(0))
            + w(1) * v_xdot(1);
        hv_xdot(1) = w(0) * v_x(0) + w(1) * v_xdot(0);
        hv_u(0) = 2.0 * w(0) * xdot(0) * v_xdot(0) - w(1) * v_x(1);
    }
};

struct DaePoint
{
    state_t   x;
    state_t   xdot;
    control_t u;
    double    t;
};

// 5-point central difference of F_i w.r.t. the k-th scalar z-input
// (k = 0..1: x, k = 2..3: xdot, k = 4: u).
template <class Dae>
double fd_f(const Dae& d, int i, int k, const DaePoint& p, double h)
{
    auto fp = [&](double s) {
        state_t dx = state_t::Zero();
        state_t dxdot = state_t::Zero();
        control_t du = control_t::Zero();
        if (k < 2)
        {
            dx(k) = s;
        }
        else if (k < 4)
        {
            dxdot(k - 2) = s;
        }
        else
        {
            du(0) = s;
        }
        return d.F(p.x + dx, p.xdot + dxdot, p.u + du, p.t)(i);
    };
    return (fp(-2.0 * h) - 8.0 * fp(-h) + 8.0 * fp(h) - fp(2.0 * h))
        / (12.0 * h);
}

// ---------------------------------------------------------------------------
//  OdeToDae wrap checks
// ---------------------------------------------------------------------------

void check_wrap_traits()
{
    static_assert(ocp::dae_supports_hess_prod<
                      ocp::OdeToDae<OdeQuadU, Dims>, Dims>,
                  "OdeToDae of a nonlinear ODE must expose F_hess_prod");
    static_assert(!ocp::dae_supports_hess_prod<
                      ocp::OdeToDae<OdeLin, Dims>, Dims>,
                  "OdeToDae of a linear ODE must not expose F_hess_prod");
    std::printf("  4k dae trait   OK\n");
}

void check_wrap_exact()
{
    using Wrap = ocp::OdeToDae<OdeQuadU, Dims>;
    OdeQuadU ode;
    Wrap dae{ode};

    const DaePoint pts[3] = {
        {state_t(0.4, -0.7), state_t(1.1, -0.3), control_t(0.6), 0.2},
        {state_t(-0.9, 1.3), state_t(0.2, 0.8), control_t(-1.1), 1.0},
        {state_t(0.0, 0.0), state_t(0.5, -0.5), control_t(0.0), 0.0},
    };
    double err_f = 0.0;
    double err_j = 0.0;
    for (const auto& p : pts)
    {
        const state_t f = ode.f(p.x, p.u, p.t);
        err_f = std::max(err_f,
                         (dae.F(p.x, p.xdot, p.u, p.t) - (p.xdot - f))
                             .cwiseAbs()
                             .maxCoeff());
        df_dx_t df_dx, dFdx, dFxdot;
        df_du_t df_du, dFdu;
        ode.jacobian(p.x, p.u, p.t, df_dx, df_du);
        dae.F_jac(p.x, p.xdot, p.u, p.t, dFdx, dFxdot, dFdu);
        err_j = std::max(err_j, (dFdx + df_dx).cwiseAbs().maxCoeff());
        err_j = std::max(err_j,
                         (dFxdot - df_dx_t::Identity()).cwiseAbs().maxCoeff());
        err_j = std::max(err_j, (dFdu + df_du).cwiseAbs().maxCoeff());
    }
    check(err_f < 1e-13, "OdeToDae F == xdot - f");
    check(err_j < 1e-13, "OdeToDae F_jac blocks");
    std::printf("  4k wrap F      OK  (max |err| = %.3e)\n", err_f);
    std::printf("  4k wrap F_jac  OK  (max |err| = %.3e)\n", err_j);

    // HVP: x/u parts must match the ODE's, xdot part must be zero
    // (F is affine in xdot), for a nonzero v_xdot.
    const struct
    {
        state_t   x;
        control_t u;
        double    t;
        state_t   w;
        state_t   v_x;
        state_t   v_xdot;
        control_t v_u;
    } cases[3] = {
        {state_t(0.3, -0.5), control_t(0.8), 0.1,
         state_t(0.9, -0.2), state_t(0.4, 0.7), state_t(0.6, -0.9),
         control_t(0.3)},
        {state_t(-1.1, 0.6), control_t(-0.4), 0.7,
         state_t(-0.7, 0.5), state_t(0.1, 0.1), state_t(0.2, 0.2),
         control_t(-0.6)},
        {state_t(0.0, 0.0), control_t(0.0), 0.0,
         state_t(1.0, -1.0), state_t(-1.0, 1.0), state_t(1.0, 0.0),
         control_t(1.0)},
    };
    double err_h = 0.0;
    double err_xd = 0.0;
    for (const auto& c : cases)
    {
        state_t hx;
        control_t hu;
        ode.hess_prod(c.x, c.u, c.t, c.w, c.v_x, c.v_u, hx, hu);
        state_t d_hx;
        state_t d_hxd;
        control_t d_hu;
        const state_t xdot = state_t::Zero();
        dae.F_hess_prod(c.x, xdot, c.u, c.t, c.w, c.v_x, c.v_xdot, c.v_u,
                        d_hx, d_hxd, d_hu);
        // F = xdot - f  =>  D2F = -D2f (affine in xdot, xdot Hessian 0).
        err_h = std::max(err_h, (d_hx + hx).cwiseAbs().maxCoeff());
        err_h = std::max(err_h, (d_hu + hu).cwiseAbs().maxCoeff());
        err_xd = std::max(err_xd, d_hxd.cwiseAbs().maxCoeff());
    }
    check(err_h < 1e-13, "OdeToDae F_hess_prod x/u parts == -ODE HVP");
    check(err_xd == 0.0, "OdeToDae F_hess_prod xdot part == 0");
    std::printf("  4k wrap F_hp   OK  (x/u %.3e, xdot %.3e)\n", err_h,
                err_xd);
}

// ---------------------------------------------------------------------------
//  Direct DAE checks against 5-point FD
// ---------------------------------------------------------------------------

void check_dae_jac_fd()
{
    DaeSin dae;
    const DaePoint pts[3] = {
        {state_t(0.3, -0.5), state_t(0.7, 0.2), control_t(0.4), 0.1},
        {state_t(-1.2, 0.9), state_t(0.1, -0.8), control_t(-0.3), 0.7},
        {state_t(0.0, 0.0), state_t(0.0, 0.0), control_t(0.0), 0.0},
    };
    const double h = 1.0e-5;
    double err = 0.0;
    for (const auto& p : pts)
    {
        df_dx_t dFdx, dFxdot;
        df_du_t dFdu;
        dae.F_jac(p.x, p.xdot, p.u, p.t, dFdx, dFxdot, dFdu);
        for (int k = 0; k < nZ; ++k)
        {
            for (int i = 0; i < nx; ++i)
            {
                const double fd = fd_f<DaeSin>(dae, i, k, p, h);
                const double an =
                    (k < 2) ? dFdx(i, k)
                    : (k < 4) ? dFxdot(i, k - 2)
                              : dFdu(i, 0);
                err = std::max(err, std::fabs(fd - an));
            }
        }
    }
    check(err < 1e-8, "DAE F_jac vs 5-point FD");
    std::printf("  4k dae F_jac   OK  (max |err| = %.3e)\n", err);
}

// 4th-order central second derivative of g(s) at s = 0:
//   g''(0) = (-g(2h) + 16 g(h) - 30 g(0) + 16 g(-h) - g(-2h)) / (12 h^2)
template <class F>
double fd_d2(const F& g, double h)
{
    return (-g(2.0 * h) + 16.0 * g(h) - 30.0 * g(0.0) + 16.0 * g(-h)
            - g(-2.0 * h))
        / (12.0 * h * h);
}

void check_dae_hess_fd()
{
    DaeSin dae;
    const DaePoint pts[3] = {
        {state_t(0.3, -0.5), state_t(0.7, 0.2), control_t(0.4), 0.1},
        {state_t(-1.2, 0.9), state_t(0.1, -0.8), control_t(-0.3), 0.7},
        {state_t(0.0, 0.0), state_t(0.0, 0.0), control_t(0.0), 0.0},
    };
    // For a direction v in z-space and output weight w,  g''(0) of
    //  g(s) = w^T F(z + s v)  equals  v^T (w^T d2F/dz^2 v), i.e. the
    // full-z contraction of the F_hess_prod output.
    const struct
    {
        state_t   w;
        state_t   v_x;
        state_t   v_xdot;
        control_t v_u;
    } cases[3] = {
        {state_t(0.8, -0.6), state_t(0.5, -0.3), state_t(0.9, 0.1),
         control_t(0.4)},
        {state_t(-0.4, 1.1), state_t(-0.2, 0.7), state_t(-0.6, 0.4),
         control_t(-0.8)},
        {state_t(1.0, 1.0), state_t(1.0, 1.0), state_t(0.0, 0.0),
         control_t(0.0)},
    };
    const double h = 1.0e-3;
    double err = 0.0;
    for (const auto& p : pts)
    {
        for (const auto& c : cases)
        {
            auto g = [&](double s) {
                const state_t fx = dae.F(p.x + s * c.v_x,
                                         p.xdot + s * c.v_xdot,
                                         p.u + s * c.v_u, p.t);
                return c.w.dot(fx);
            };
            const double fd = fd_d2(g, h);
            state_t hv_x;
            state_t hv_xdot;
            control_t hv_u;
            dae.F_hess_prod(p.x, p.xdot, p.u, p.t, c.w, c.v_x, c.v_xdot,
                            c.v_u, hv_x, hv_xdot, hv_u);
            const double an = c.v_x.dot(hv_x) + c.v_xdot.dot(hv_xdot)
                + c.v_u(0) * hv_u(0);
            err = std::max(err, std::fabs(fd - an));
        }
    }
    check(err < 1e-8, "DAE F_hess_prod vs 4th-order FD (full z)");
    std::printf("  4k dae F_hp    OK  (max |err| = %.3e)\n", err);
}

}  // namespace

int run_dae_4k_tests()
{
    check_wrap_traits();
    check_wrap_exact();
    check_dae_jac_fd();
    check_dae_hess_fd();

    if (failures != 0)
    {
        std::fprintf(stderr, "[4k] %d check(s) failed.\n", failures);
    }
    return failures;
}

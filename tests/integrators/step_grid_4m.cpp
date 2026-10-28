// Phase 4m: verify the per-stage step-size hook (StepSpec / get_step_size
// CRTP hook of ContinuousProblem) end to end.
//
// A time-varying, nonlinear-in-x ODE is wrapped in a problem whose per-stage
// grid (h_k, t_k) is read from a duck-typed params() struct with inspect
// leaves; the grid is non-uniform and starts at t0 != 0. Covered:
//   (a) ERK (K4, NumSteps = 2) dynamics_next_state chained over the grid
//       vs. a high-resolution RK4 reference on the same (t_k, h_k)
//   (b) ERK dynamics_value_jac vs. central FD of the per-stage map
//   (c) ERK dynamics_value_jac_hess vs. central FD of w^T D2 Phi
//   (d) IRK (Radau IIA 2 via OdeToDae) value vs. the RK4 reference and
//       jacobian vs. FD of its own value
//   (e) fallback regression: default Derived (ConstantStepSize) == the
//       standalone integrator on the uniform grid {h, k * h}
//   (f) a constant hook override ({h0, k * h0}) == the fallback
//   (g) live mutation of the grid leaf through inspect::set_* is picked up
//       by the dynamics (params() is the SSOT)

#include <array>
#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/integrators/continuous_problem.hpp"
#include "ocp/integrators/rk_explicit.hpp"
#include "ocp/integrators/rk_implicit.hpp"
#include "ocp/utils/inspect.hpp"

namespace
{

// --- dims (nx = 2, nu = 1, no path/terminal constraints) ----------------

struct SGDims
{
    static constexpr int nx = 2;
    static constexpr int nu = 1;
    static constexpr int ng = 0;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 0;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = true;
    static constexpr bool has_dynamics_hess = true;  // nonlinear ODE
    static constexpr bool has_constr_hess = false;
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

using P = ocp::Problem<SGDims>;
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

void check_close(double a, double b, double tol, const char* msg)
{
    if (std::fabs(a - b) > tol)
    {
        std::fprintf(stderr, "FAIL: %s (%.15e vs %.15e)\n", msg, a, b);
        ++failures;
    }
}

// ---------------------------------------------------------------------------
//  ODE:  f_i = sin(t) * x_i^2 * u(0) + cos(t) * x_i
//  (time-varying, nonlinear in x, linear in u; hand-derived jacobian and
//  hess_prod)
// ---------------------------------------------------------------------------

struct OdeTvar
{
    state_t f(const state_t& x, const control_t& u, double t) const
    {
        return std::sin(t) * x.cwiseAbs2() * u(0)
            + std::cos(t) * x;
    }

    void jacobian(const state_t& x, const control_t& u, double t,
                  df_dx_t& df_dx, df_du_t& df_du) const
    {
        const double s = std::sin(t);
        const double c = std::cos(t);
        df_dx.setZero();
        for (int i = 0; i < nx; ++i)
        {
            df_dx(i, i) = 2.0 * s * u(0) * x(i) + c;
        }
        df_du.col(0) = s * x.cwiseAbs2();
    }

    void hess_prod(const state_t& x, const control_t& u, double t,
                   const state_t& w, const state_t& v_x, const control_t& v_u,
                   state_t& hv_x, control_t& hv_u) const
    {
        // f_i = s * x_i^2 * u + c * x_i  (s = sin t, c = cos t).
        //   d2f_i/dx_j/dx_k = 2 s u * delta_ik * delta_jk
        //   d2f_i/dx_j/du   = 2 s * x_j * delta_ij
        //   d2f_i/du/du     = 0
        // => hv_x(j) = 2 s w(j) (u v_x(j) + x(j) v_u(0))
        //    hv_u     = 2 s * sum_i w(i) x(i) v_x(i)
        const double s = std::sin(t);
        for (int j = 0; j < nx; ++j)
        {
            hv_x(j) = 2.0 * s * w(j) * (u(0) * v_x(j) + x(j) * v_u(0));
        }
        hv_u(0) = 0.0;
        for (int i = 0; i < nx; ++i)
        {
            hv_u(0) += 2.0 * s * w(i) * x(i) * v_x(i);
        }
    }
};

// ---------------------------------------------------------------------------
//  Grid schedule: duck-typed params() SSOT with inspect leaves.
//  Non-uniform widths h_k and a non-zero start time t0.
// ---------------------------------------------------------------------------

struct GridParams
{
    static constexpr int N = 4;
    Eigen::Matrix<double, N, 1> h_grid;
    double t0 = 0.37;

    GridParams()
    {
        h_grid << 0.1, 0.2, 0.15, 0.05;
    }

    template <class Visitor>
    static constexpr void inspect(Visitor&& vis)
    {
        vis("h_grid", &GridParams::h_grid);
        vis("t0", &GridParams::t0);
    }
};

// ---------------------------------------------------------------------------
//  Problems
// ---------------------------------------------------------------------------

using Erk = ocp::ExplicitRkIntegrator<SGDims, OdeTvar, ocp::K4Tag, 2>;

/// Problem whose per-stage grid comes from params() (CRTP hook, non-uniform
/// grid, t0 != 0):  t_k = t0 + sum_{i < k} h_i.
struct GridProblem : ocp::ContinuousProblem<SGDims, OdeTvar, Erk,
                                            GridProblem>
{
    using Base = ocp::ContinuousProblem<SGDims, OdeTvar, Erk, GridProblem>;

    GridParams params_;

    GridProblem() : Base(OdeTvar{}, 0.1) {}

    GridParams& params() { return params_; }
    const GridParams& params() const { return params_; }

    ocp::StepSpec get_step_size(int k) const
    {
        double t = params_.t0;
        for (int i = 0; i < k; ++i)
        {
            t += params_.h_grid(i);
        }
        return ocp::StepSpec{params_.h_grid(k), t};
    }
};

/// Default Derived (ConstantStepSize): the uniform grid {h, k * h}.
using FallbackProblem = ocp::ContinuousProblem<SGDims, OdeTvar, Erk>;

/// Named Derived whose hook returns the same uniform grid as the fallback;
/// must stay bit-identical to it.
struct ConstHookProblem
    : ocp::ContinuousProblem<SGDims, OdeTvar, Erk, ConstHookProblem>
{
    using Base = ocp::ContinuousProblem<SGDims, OdeTvar, Erk,
                                        ConstHookProblem>;

    static constexpr double h0 = 0.1;

    ConstHookProblem() : Base(OdeTvar{}, h0) {}

    ocp::StepSpec get_step_size(int k) const
    {
        return ocp::StepSpec{h0, (double)k * h0};
    }
};

using Irk = ocp::ImplicitRkIntegrator<SGDims, ocp::OdeToDae<OdeTvar, SGDims>,
                                      ocp::RadauIia2Tag>;

// ---------------------------------------------------------------------------
//  Reference: high-resolution classic RK4 over a per-stage (t, h) interval
// ---------------------------------------------------------------------------

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

const state_t x0_global(0.3, -0.7);
const control_t u_global(0.4);

// ---------------------------------------------------------------------------
//  (a) ERK value: chained dynamics_next_state vs. RK4 reference
// ---------------------------------------------------------------------------

void check_erk_value_grid()
{
    const GridProblem prob;
    const OdeTvar ode;

    state_t cur = x0_global;
    for (int k = 0; k < GridParams::N; ++k)
    {
        cur = prob.dynamics_next_state(k, cur, u_global);
    }

    state_t ref = x0_global;
    double t = prob.params_.t0;
    for (int k = 0; k < GridParams::N; ++k)
    {
        const double h = prob.params_.h_grid(k);
        const int n_sub = 2000;
        const double h_ref = h / n_sub;
        for (int i = 0; i < n_sub; ++i)
        {
            // time-varying ODE: t advances across the sub-steps.
            ref = rk4_step(ode, ref, u_global, t, h_ref);
            t += h_ref;
        }
    }

    const double err = (cur - ref).cwiseAbs().maxCoeff();
    // A wrong grid (e.g. the uniform fallback) shows ~3e-2 here, so this
    // tolerance is far below any grid error while allowing ~10x headroom
    // over the RK4 sub-step truncation error (~7e-8) for portability.
    check(err < 1e-6, "4m ERK value on non-uniform grid vs RK4 reference");
    std::printf("  4m ERK value  OK  (|err|=%.2e < 1e-6, non-uniform h, "
                "t0 != 0)\n", err);
}

// ---------------------------------------------------------------------------
//  (b) ERK jacobian: dynamics_value_jac vs. central FD of the per-stage map
// ---------------------------------------------------------------------------

void check_erk_jac_grid()
{
    const GridProblem prob;
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    const double d = 1e-6;
    double max_err = 0.0;

    for (int k = 0; k < GridParams::N; ++k)
    {
        df_dx_t Jx;
        df_du_t Ju;
        state_t xn;
        prob.dynamics_value_jac(k, x, u, xn, Jx, Ju);

        for (int j = 0; j < nx; ++j)
        {
            state_t xp = x;
            state_t xm = x;
            xp(j) += d;
            xm(j) -= d;
            const state_t fdp = prob.dynamics_next_state(k, xp, u);
            const state_t fdm = prob.dynamics_next_state(k, xm, u);
            for (int i = 0; i < nx; ++i)
            {
                max_err = std::max(max_err,
                                   std::abs(Jx(i, j)
                                           - (fdp(i) - fdm(i)) / (2.0 * d)));
            }
        }
        for (int j = 0; j < nu; ++j)
        {
            control_t up = u;
            control_t um = u;
            up(j) += d;
            um(j) -= d;
            const state_t fdp = prob.dynamics_next_state(k, x, up);
            const state_t fdm = prob.dynamics_next_state(k, x, um);
            for (int i = 0; i < nx; ++i)
            {
                max_err = std::max(max_err,
                                   std::abs(Ju(i, j)
                                           - (fdp(i) - fdm(i)) / (2.0 * d)));
            }
        }
    }
    check(max_err < 1e-6, "4m ERK jacobian non-uniform grid FD");
    std::printf("  4m ERK jac    OK  (max|err|=%.2e < 1e-6, 4 stages)\n",
                max_err);
}

// ---------------------------------------------------------------------------
//  (c) ERK contracted Hessian: dynamics_value_jac_hess vs. central FD
//      of g(z) = w^T Phi_k(z)  (stage k = 1: h = 0.2, t = 0.47)
// ---------------------------------------------------------------------------

void check_erk_hess_grid()
{
    const GridProblem prob;
    const int k = 1;
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    const state_t w(0.7, -0.3);

    using zvec_t = Eigen::Matrix<double, nIn, 1>;
    zvec_t z0;
    z0(0) = x(0);
    z0(1) = x(1);
    z0(2) = u(0);

    auto g = [&](const zvec_t& z) -> double
    {
        state_t xx;
        xx(0) = z(0);
        xx(1) = z(1);
        control_t uu;
        uu(0) = z(2);
        return w.dot(prob.dynamics_next_state(k, xx, uu));
    };

    const double d = 1e-5;
    Eigen::Matrix<double, nIn, nIn> H;
    for (int a = 0; a < nIn; ++a)
    {
        for (int b = a; b < nIn; ++b)
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

    state_t xn;
    df_dx_t Jx;
    df_du_t Ju;
    P::dyn_hess_t hess;
    prob.dynamics_value_jac_hess(k, x, u, w, xn, Jx, Ju, hess);

    double max_err = 0.0;
    for (int j = 0; j < nIn; ++j)
    {
        max_err = std::max(max_err,
                           (hess.col(j) - H.col(j)).cwiseAbs().maxCoeff());
    }
    check(max_err < 1e-5, "4m ERK hess non-uniform grid FD");
    std::printf("  4m ERK hess   OK  (max|err|=%.2e < 1e-5, stage k=1)\n",
                max_err);
}

// ---------------------------------------------------------------------------
//  (d) IRK (Radau IIA 2): value vs. RK4 reference, jacobian vs. FD
// ---------------------------------------------------------------------------

void check_irk_grid()
{
    const OdeTvar ode;
    Irk irk(ode, 8);
    const double h_k[GridParams::N] = {0.1, 0.2, 0.15, 0.05};
    const double t0 = 0.37;

    // value: chain the standalone integrator over the non-uniform grid.
    state_t cur = x0_global;
    state_t ref = x0_global;
    double t = t0;
    double max_err = 0.0;
    for (int k = 0; k < GridParams::N; ++k)
    {
        state_t next;
        irk.value(cur, u_global, t, h_k[k], next);
        const int n_sub = 2000;
        const double h_ref = h_k[k] / n_sub;
        for (int i = 0; i < n_sub; ++i)
        {
            // time-varying ODE: t advances across the sub-steps.
            ref = rk4_step(ode, ref, u_global, t, h_ref);
            t += h_ref;
        }
        max_err = std::max(max_err, (next - ref).cwiseAbs().maxCoeff());
        cur = next;
    }
    // Radau IIA 2 is 3rd order; ~5e-5 is the expected truncation error for
    // the 0.2 s stage. 5e-4 keeps a 10x margin (a wrong grid shows ~3e-2).
    check(max_err < 5e-4, "4m IRK value on non-uniform grid vs RK4");
    std::printf("  4m IRK value  OK  (|err|=%.2e < 5e-4)\n", max_err);

    // jacobian at stage k = 2 (t = 0.67, h = 0.15) vs. central FD of the
    // IRK's own value.
    const int k = 2;
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    const double d = 1e-6;
    df_dx_t Jx;
    df_du_t Ju;
    state_t xn;
    irk.value_jac(x, u, t0 + h_k[0] + h_k[1], h_k[k], xn, Jx, Ju);

    max_err = 0.0;
    for (int j = 0; j < nx; ++j)
    {
        state_t xp = x;
        state_t xm = x;
        xp(j) += d;
        xm(j) -= d;
        state_t fdp;
        state_t fdm;
        irk.value(xp, u, t0 + h_k[0] + h_k[1], h_k[k], fdp);
        irk.value(xm, u, t0 + h_k[0] + h_k[1], h_k[k], fdm);
        for (int i = 0; i < nx; ++i)
        {
            max_err = std::max(max_err,
                               std::abs(Jx(i, j)
                                       - (fdp(i) - fdm(i)) / (2.0 * d)));
        }
    }
    for (int j = 0; j < nu; ++j)
    {
        control_t up = u;
        control_t um = u;
        up(j) += d;
        um(j) -= d;
        state_t fdp;
        state_t fdm;
        irk.value(x, up, t0 + h_k[0] + h_k[1], h_k[k], fdp);
        irk.value(x, um, t0 + h_k[0] + h_k[1], h_k[k], fdm);
        for (int i = 0; i < nx; ++i)
        {
            max_err = std::max(max_err,
                               std::abs(Ju(i, j)
                                       - (fdp(i) - fdm(i)) / (2.0 * d)));
        }
    }
    check(max_err < 1e-6, "4m IRK jacobian FD");
    std::printf("  4m IRK jac    OK  (max|err|=%.2e < 1e-6)\n", max_err);
}

// ---------------------------------------------------------------------------
//  (e) fallback regression: default Derived == standalone integrator on
//      the uniform grid {h, k * h}
// ---------------------------------------------------------------------------

void check_fallback_regression()
{
    FallbackProblem fb(OdeTvar{}, 0.1);
    Erk integ(OdeTvar{});
    const state_t x(0.3, -0.7);
    const control_t u(0.4);

    for (int k = 0; k < 3; ++k)
    {
        const state_t xn = fb.dynamics_next_state(k, x, u);
        state_t ref;
        integ.value(x, u, k * 0.1, 0.1, ref);
        check_close((xn - ref).cwiseAbs().maxCoeff(), 0.0, 1e-15,
                    "4m fallback == standalone uniform-grid integrator");
    }
    std::printf("  4m fallback   OK  (constant-h grid {h, k*h} unchanged)\n");
}

// ---------------------------------------------------------------------------
//  (f) constant hook override == fallback
// ---------------------------------------------------------------------------

void check_const_hook_equals_fallback()
{
    FallbackProblem fb(OdeTvar{}, 0.1);
    const ConstHookProblem ch;
    const state_t x(0.3, -0.7);
    const control_t u(0.4);

    for (int k = 0; k < 3; ++k)
    {
        const state_t a = fb.dynamics_next_state(k, x, u);
        const state_t b = ch.dynamics_next_state(k, x, u);
        check_close((a - b).cwiseAbs().maxCoeff(), 0.0, 1e-15,
                    "4m constant hook == fallback");
    }
    std::printf("  4m const hook OK  ({h0, k*h0} == fallback)\n");
}

// ---------------------------------------------------------------------------
//  (g) live params: mutate the grid leaf by name (inspect::set_*) and check
//      the dynamics follow the new schedule
// ---------------------------------------------------------------------------

void check_live_params_mutation()
{
    GridProblem prob;
    const state_t x(0.3, -0.7);
    const control_t u(0.4);
    const state_t before = prob.dynamics_next_state(1, x, u);

    double h_new[GridParams::N] = {0.2, 0.1, 0.25, 0.05};
    check(ocp::inspect::set_matrix_double(prob.params(), "h_grid", h_new,
                                          GridParams::N),
          "4m set_matrix_double on the h_grid leaf");
    check(ocp::inspect::set_double(prob.params(), "t0", 1.25).ok(),
          "4m set_double on the t0 leaf");

    const state_t after = prob.dynamics_next_state(1, x, u);

    // standalone reference at the new stage-1 spec:
    // t_1 = 1.25 + 0.2 = 1.45,  h_1 = 0.1
    Erk integ(OdeTvar{});
    state_t ref;
    integ.value(x, u, 1.45, 0.1, ref);
    check_close((after - ref).cwiseAbs().maxCoeff(), 0.0, 1e-15,
                "4m dynamics follow the mutated grid");
    check((after - before).cwiseAbs().maxCoeff() > 1e-3,
          "4m grid mutation is visible in the dynamics");
    std::printf("  4m live params OK  (inspect::set_* changes the grid)\n");
}

}  // namespace

int run_step_grid_4m_tests()
{
    check_erk_value_grid();
    check_erk_jac_grid();
    check_erk_hess_grid();
    check_irk_grid();
    check_fallback_regression();
    check_const_hook_equals_fallback();
    check_live_params_mutation();

    if (failures != 0)
    {
        std::fprintf(stderr, "[4m] %d check(s) failed.\n", failures);
    }
    return failures;
}

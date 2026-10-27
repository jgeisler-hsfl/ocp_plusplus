// ScaledProblem wrapper test: verifies that the diagonal state/input
// scaling produces an exactly equivalent problem (same cost, correctly
// transformed derivatives, and identical solution after undoing scaling).

#include <cmath>
#include <cstdio>

#include "ocp/scaling.hpp"
#include "ocp/solvers/acados/acados.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

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

void check_close(double a, double b, double tol, const char* msg)
{
    if (std::fabs(a - b) > tol)
    {
        std::fprintf(stderr, "FAIL: %s (%.15e vs %.15e)\n", msg, a, b);
        ++failures;
    }
}

template <class A, class B>
void check_mat_close(const A& a, const B& b, double tol, const char* msg)
{
    const double diff = (a - b).cwiseAbs().maxCoeff();
    if (diff > tol)
    {
        std::fprintf(stderr, "FAIL: %s (max diff = %.15e)\n", msg, diff);
        ++failures;
    }
}

constexpr int N = 10;

// Test 1: value invariance
void test_value_invariance()
{
    DoubleIntegrator base;
    ScaledProblem<DoubleIntegrator> scaled(base);
    scaled.sx_ << 2.0, 0.5;
    scaled.su_ << 3.0;

    DoubleIntegrator::state_t x;
    x << 1.5, -0.3;
    DoubleIntegrator::control_t u;
    u << 0.4;

    auto x_s = x.cwiseProduct(scaled.sx_);
    auto u_s = u.cwiseProduct(scaled.su_);

    check_close(base.stage_cost_value(0, x, u),
                scaled.stage_cost_value(0, x_s, u_s),
                1e-12, "stage cost value invariant");

    check_close(base.terminal_cost_value(x),
                scaled.terminal_cost_value(x_s),
                1e-12, "terminal cost value invariant");

    auto x_next_b = base.dynamics_next_state(0, x, u);
    auto x_next_s = scaled.dynamics_next_state(0, x_s, u_s);
    check_mat_close(x_next_b.cwiseProduct(scaled.sx_), x_next_s,
                    1e-12, "dynamics_next_state consistent");

    auto g_b = base.stage_inequality_value(0, x, u);
    auto g_s = scaled.stage_inequality_value(0, x_s, u_s);
    check_close(g_b(0), g_s(0), 1e-12, "inequality value invariant");

    auto lin_b = base.stage_linear_constr(0);
    auto lin_s = scaled.stage_linear_constr(0);
    const double val_b = (lin_b.A * x + lin_b.B * u)(0);
    const double val_s = (lin_s.A * x_s + lin_s.B * u_s)(0);
    check_close(val_b, val_s, 1e-12, "linear form value invariant");

    auto eq_b = base.terminal_equality_value(x);
    auto eq_s = scaled.terminal_equality_value(x_s);
    check_close(eq_b(0), eq_s(0), 1e-12, "terminal eq value invariant");
}

// Test 2: gradient transform
void test_gradient_transform()
{
    DoubleIntegrator base;
    ScaledProblem<DoubleIntegrator> scaled(base);
    scaled.sx_ << 2.0, 0.5;
    scaled.su_ << 3.0;

    DoubleIntegrator::state_t x;
    x << 1.5, -0.3;
    DoubleIntegrator::control_t u;
    u << 0.4;
    auto x_s = x.cwiseProduct(scaled.sx_);
    auto u_s = u.cwiseProduct(scaled.su_);

    double val;
    DoubleIntegrator::stage_grad_t g_b;
    base.stage_cost_value_grad(0, x, u, val, g_b);

    double val_s;
    ScaledProblem<DoubleIntegrator>::stage_grad_t gs;
    scaled.stage_cost_value_grad(0, x_s, u_s, val_s, gs);
    check_close(val, val_s, 1e-12, "grad: value invariant");

    DoubleIntegrator::stage_grad_t g_expected;
    g_expected(0) = g_b(0) / 2.0;
    g_expected(1) = g_b(1) / 0.5;
    g_expected(2) = g_b(2) / 3.0;
    check_mat_close(g_expected, gs, 1e-12, "stage grad transform");

    double val_t;
    DoubleIntegrator::term_grad_t tg_b;
    base.terminal_cost_value_grad(x, val_t, tg_b);

    double val_t_s;
    ScaledProblem<DoubleIntegrator>::term_grad_t tg_s;
    scaled.terminal_cost_value_grad(x_s, val_t_s, tg_s);

    DoubleIntegrator::term_grad_t tg_expected;
    tg_expected(0) = tg_b(0) / 2.0;
    tg_expected(1) = tg_b(1) / 0.5;
    check_mat_close(tg_expected, tg_s, 1e-12, "terminal grad transform");
}

// Test 3: Hessian transform
void test_hessian_transform()
{
    DoubleIntegrator base;
    ScaledProblem<DoubleIntegrator> scaled(base);
    scaled.sx_ << 2.0, 0.5;
    scaled.su_ << 3.0;

    DoubleIntegrator::state_t x;
    x << 1.5, -0.3;
    DoubleIntegrator::control_t u;
    u << 0.4;
    auto x_s = x.cwiseProduct(scaled.sx_);
    auto u_s = u.cwiseProduct(scaled.su_);

    double val;
    DoubleIntegrator::stage_grad_t g;
    DoubleIntegrator::stage_hess_t H_b;
    base.stage_cost_value_grad_hess(0, x, u, val, g, H_b);

    double val_s;
    ScaledProblem<DoubleIntegrator>::stage_grad_t gs;
    ScaledProblem<DoubleIntegrator>::stage_hess_t H_s;
    scaled.stage_cost_value_grad_hess(0, x_s, u_s, val_s, gs, H_s);

    ScaledProblem<DoubleIntegrator>::stage_hess_t H_expected;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            const double di = (i < 2) ? scaled.sx_(i) : scaled.su_(i - 2);
            const double dj = (j < 2) ? scaled.sx_(j) : scaled.su_(j - 2);
            H_expected(i, j) = H_b(i, j) / (di * dj);
        }
    }
    check_mat_close(H_expected, H_s, 1e-12, "stage Hessian transform");

    DoubleIntegrator::term_grad_t tg;
    DoubleIntegrator::term_hess_t TH_b;
    base.terminal_cost_value_grad_hess(x, val, tg, TH_b);

    double val_s2;
    ScaledProblem<DoubleIntegrator>::term_grad_t tgs;
    ScaledProblem<DoubleIntegrator>::term_hess_t TH_s;
    scaled.terminal_cost_value_grad_hess(x_s, val_s2, tgs, TH_s);

    ScaledProblem<DoubleIntegrator>::term_hess_t TH_expected;
    TH_expected(0, 0) = TH_b(0, 0) / (2.0 * 2.0);
    TH_expected(0, 1) = TH_b(0, 1) / (2.0 * 0.5);
    TH_expected(1, 0) = TH_b(1, 0) / (0.5 * 2.0);
    TH_expected(1, 1) = TH_b(1, 1) / (0.5 * 0.5);
    check_mat_close(TH_expected, TH_s, 1e-12, "terminal Hessian transform");
}

// Test 4: Jacobian transform
void test_jacobian_transform()
{
    DoubleIntegrator base;
    ScaledProblem<DoubleIntegrator> scaled(base);
    scaled.sx_ << 2.0, 0.5;
    scaled.su_ << 3.0;

    DoubleIntegrator::state_t x;
    x << 1.5, -0.3;
    DoubleIntegrator::control_t u;
    u << 0.4;
    auto x_s = x.cwiseProduct(scaled.sx_);
    auto u_s = u.cwiseProduct(scaled.su_);

    DoubleIntegrator::state_t xn_b;
    DoubleIntegrator::dyn_df_dx_t A_b;
    DoubleIntegrator::dyn_df_du_t B_b;
    base.dynamics_value_jac(0, x, u, xn_b, A_b, B_b);

    ScaledProblem<DoubleIntegrator>::state_t xn_s;
    ScaledProblem<DoubleIntegrator>::dyn_df_dx_t As;
    ScaledProblem<DoubleIntegrator>::dyn_df_du_t Bs;
    scaled.dynamics_value_jac(0, x_s, u_s, xn_s, As, Bs);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            const double expected = A_b(i, j) * scaled.sx_(i) / scaled.sx_(j);
            check_close(As(i, j), expected, 1e-12, "dyn Jacobian A transform");
        }
    }
    for (int i = 0; i < 2; ++i)
    {
        const double expected = B_b(i, 0) * (i == 0 ? 2.0 : 0.5) / 3.0;
        check_close(Bs(i, 0), expected, 1e-12, "dyn Jacobian B transform");
    }
}

// Test 5: box bounds transform
void test_box_transform()
{
    DoubleIntegrator base;
    ScaledProblem<DoubleIntegrator> scaled(base);
    scaled.sx_ << 2.0, 0.5;
    scaled.su_ << 3.0;

    auto sb = scaled.stage_state_box_constr(0);
    check_close(sb.lo(0), -20.0, 1e-12, "state box lo(0) scaled");
    check_close(sb.lo(1), -5.0, 1e-12, "state box lo(1) scaled");
    check_close(sb.hi(0), 20.0, 1e-12, "state box hi(0) scaled");
    check_close(sb.hi(1), 5.0, 1e-12, "state box hi(1) scaled");

    auto cb = scaled.stage_control_box_constr(0);
    check_close(cb.lo(0), -3.0, 1e-12, "control box lo(0) scaled");
    check_close(cb.hi(0), 3.0, 1e-12, "control box hi(0) scaled");

    auto tb = scaled.terminal_state_box_constr();
    check_close(tb.lo(0), -20.0, 1e-12, "term state box lo(0)");
    check_close(tb.lo(1), -5.0, 1e-12, "term state box lo(1)");
    check_close(tb.hi(0), 20.0, 1e-12, "term state box hi(0)");
    check_close(tb.hi(1), 5.0, 1e-12, "term state box hi(1)");
}

// Test 6: initial state transform
void test_initial_state()
{
    DoubleIntegrator base;
    ScaledProblem<DoubleIntegrator> scaled(base);
    scaled.sx_ << 2.0, 0.5;
    scaled.su_ << 3.0;

    auto x0_b = base.initial_state();
    auto x0_s = scaled.initial_state();
    check_close(x0_s(0), x0_b(0) * 2.0, 1e-12, "initial state q scaled");
    check_close(x0_s(1), x0_b(1) * 0.5, 1e-12, "initial state v scaled");
}

// Constant control that makes the terminal equality q_N + v_N = 1 hold
// exactly (same warm-start strategy as the sqp_double_integrator test).
double terminal_eq_feasible_control(const DoubleIntegrator& p)
{
    const double Ts = p.Ts_;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * N * (N - 1) / 2.0 + Ts * N;
    return (1.0 - (q0 + v0) - Ts * N * v0) / den;
}

// Test 7: end-to-end — solve unscaled and scaled, verify equivalence.
void test_e2e_equivalence()
{
    DoubleIntegrator base;
    ScaledProblem<DoubleIntegrator> scaled(base);
    scaled.sx_ << 2.0, 0.5;
    scaled.su_ << 3.0;

    AcadosOptions opts;
    opts.print_level = 0;

    // build a warm start with the feasible constant control
    const double a = terminal_eq_feasible_control(base);
    Solution<DoubleIntegrator> warm(N);
    warm.x[0] = base.initial_state();
    for (int k = 0; k < N; ++k)
    {
        warm.u[k](0) = a;
        warm.x[k + 1] = base.dynamics_next_state(k, warm.x[k], warm.u[k]);
    }

    // unscaled solve
    Solution<DoubleIntegrator> sol_b = warm;
    AcadosSolver<DoubleIntegrator> solver_b(opts);
    const Status st_b = solver_b.solve(base, sol_b);
    check(st_b == Status::kSolved, "unscaled solve kSolved");

    // scaled warm start (same trajectory, scaled)
    Solution<ScaledProblem<DoubleIntegrator>> sol_s(N);
    for (int k = 0; k <= N; ++k)
    {
        sol_s.x[k].array() = warm.x[k].array() * scaled.sx_.array();
    }
    for (int k = 0; k < N; ++k)
    {
        sol_s.u[k].array() = warm.u[k].array() * scaled.su_.array();
    }

    // scaled solve
    AcadosSolver<ScaledProblem<DoubleIntegrator>> solver_s(opts);
    const Status st_s = solver_s.solve(scaled, sol_s);
    check(st_s == Status::kSolved, "scaled solve kSolved");

    auto sol_s_unscaled = scaled.unscale(sol_s);

    check_close(sol_b.cost_value, sol_s_unscaled.cost_value,
                1e-6, "cost equivalent");

    for (int k = 0; k <= N; ++k)
    {
        check_mat_close(sol_b.x[k], sol_s_unscaled.x[k], 1e-5,
                        "x trajectory equivalent");
    }
    for (int k = 0; k < N; ++k)
    {
        check_mat_close(sol_b.u[k], sol_s_unscaled.u[k], 1e-5,
                        "u trajectory equivalent");
    }

    for (int k = 0; k < N; ++k)
    {
        check_mat_close(sol_b.lambda_dyn[k], sol_s_unscaled.lambda_dyn[k],
                        1e-4, "lambda_dyn equivalent");
    }

    for (int k = 0; k < N; ++k)
    {
        check_mat_close(sol_b.lambda_ineq_stage[k],
                        sol_s_unscaled.lambda_ineq_stage[k],
                        1e-4, "lambda_ineq equivalent");
    }
    check_mat_close(sol_b.lambda_eq_term, sol_s_unscaled.lambda_eq_term,
                    1e-4, "lambda_eq_term equivalent");
}

// Test 8: identity scaling — wrapper must reproduce the base bit-exactly.
void test_identity_scaling()
{
    DoubleIntegrator base;
    ScaledProblem<DoubleIntegrator> scaled(base);
    // sx_ / su_ default to Ones()

    DoubleIntegrator::state_t x;
    x << 0.7, -1.2;
    DoubleIntegrator::control_t u;
    u << 0.9;

    // initial_state
    check((base.initial_state() - scaled.initial_state()).isZero(),
          "identity: initial_state");

    // dynamics_next_state
    check((base.dynamics_next_state(0, x, u) -
           scaled.dynamics_next_state(0, x, u)).isZero(),
          "identity: dynamics_next_state");

    // dynamics_value_jac
    DoubleIntegrator::state_t xn_b, xn_s;
    DoubleIntegrator::dyn_df_dx_t A_b, A_s;
    DoubleIntegrator::dyn_df_du_t B_b, B_s;
    base.dynamics_value_jac(0, x, u, xn_b, A_b, B_b);
    scaled.dynamics_value_jac(0, x, u, xn_s, A_s, B_s);
    check((A_b - A_s).isZero() && (B_b - B_s).isZero(),
          "identity: dynamics jac");

    // dynamics_value_jac_hess
    DoubleIntegrator::state_t lam;
    lam << 0.3, -0.5;
    DoubleIntegrator::dyn_hess_t H_b, H_s;
    base.dynamics_value_jac_hess(0, x, u, lam, xn_b, A_b, B_b, H_b);
    scaled.dynamics_value_jac_hess(0, x, u, lam, xn_s, A_s, B_s, H_s);
    check((H_b - H_s).isZero(), "identity: dynamics hess");

    // stage cost value / grad / hess
    double vb, vs;
    DoubleIntegrator::stage_grad_t gb, gs;
    DoubleIntegrator::stage_hess_t hb, hs;
    base.stage_cost_value_grad_hess(0, x, u, vb, gb, hb);
    scaled.stage_cost_value_grad_hess(0, x, u, vs, gs, hs);
    check(std::fabs(vb - vs) < 1e-15, "identity: stage cost value");
    check((gb - gs).isZero() && (hb - hs).isZero(),
          "identity: stage cost grad+hess");

    // terminal cost value / grad / hess
    DoubleIntegrator::term_grad_t tb, ts;
    DoubleIntegrator::term_hess_t THb, THs;
    base.terminal_cost_value_grad_hess(x, vb, tb, THb);
    scaled.terminal_cost_value_grad_hess(x, vs, ts, THs);
    check((tb - ts).isZero() && (THb - THs).isZero(),
          "identity: terminal cost grad+hess");

    // stage inequality value / jac / hess
    DoubleIntegrator::ineq_t ilam;
    ilam(0) = 1.5;
    DoubleIntegrator::ineq_t gib, gis;
    DoubleIntegrator::ineq_dg_dx_t gdx_b, gdx_s;
    DoubleIntegrator::ineq_dg_du_t gdu_b, gdu_s;
    DoubleIntegrator::constr_hess_t CHb, CHs;
    base.stage_inequality_value_jac_hess(0, x, u, ilam, gib, gdx_b, gdu_b, CHb);
    scaled.stage_inequality_value_jac_hess(0, x, u, ilam, gis, gdx_s, gdu_s, CHs);
    check((gdx_b - gdx_s).isZero() && (gdu_b - gdu_s).isZero() &&
          (CHb - CHs).isZero(),
          "identity: stage ineq jac+hess");

    // linear constraint
    auto lin_b = base.stage_linear_constr(0);
    auto lin_s = scaled.stage_linear_constr(0);
    check((lin_b.A - lin_s.A).isZero() && (lin_b.B - lin_s.B).isZero(),
          "identity: linear A/B");

    // box constraints
    auto sb_b = base.stage_state_box_constr(0);
    auto sb_s = scaled.stage_state_box_constr(0);
    check((sb_b.lo - sb_s.lo).isZero() && (sb_b.hi - sb_s.hi).isZero(),
          "identity: state box lo/hi");
    auto cb_b = base.stage_control_box_constr(0);
    auto cb_s = scaled.stage_control_box_constr(0);
    check((cb_b.lo - cb_s.lo).isZero() && (cb_b.hi - cb_s.hi).isZero(),
          "identity: control box lo/hi");

    // terminal equality value / jac
    DoubleIntegrator::eq_term_t teb, tes;
    DoubleIntegrator::eq_term_de_dx_t tedx_b, tedx_s;
    base.terminal_equality_value_jac(x, teb, tedx_b);
    scaled.terminal_equality_value_jac(x, tes, tedx_s);
    check((tedx_b - tedx_s).isZero(), "identity: term eq jac");

    // soft penalty pass-through
    auto pen_b = base.stage_inequality_constr_soft_penalty(0);
    auto pen_s = scaled.stage_inequality_constr_soft_penalty(0);
    check((pen_b.quad - pen_s.quad).isZero(),
          "identity: ineq soft penalty");

    // sens_p: dynamics
    DoubleIntegrator::dyn_df_dp_t fp_b, fp_s;
    DoubleIntegrator::dyn_dA_dp_t Ap_b, Ap_s;
    DoubleIntegrator::dyn_dB_dp_t Bp_b, Bp_s;
    base.dynamics_sens_p(0, x, u, fp_b, Ap_b, Bp_b);
    scaled.dynamics_sens_p(0, x, u, fp_s, Ap_s, Bp_s);
    check((fp_b - fp_s).isZero() && (Ap_b - Ap_s).isZero() &&
          (Bp_b - Bp_s).isZero(),
          "identity: dynamics sens_p");

    // sens_p: stage cost
    DoubleIntegrator::param_t Lp_b, Lp_s;
    DoubleIntegrator::stage_grad_dp_t gp_b, gp_s;
    base.stage_cost_sens_p(0, x, u, Lp_b, gp_b);
    scaled.stage_cost_sens_p(0, x, u, Lp_s, gp_s);
    check((Lp_b - Lp_s).isZero() && (gp_b - gp_s).isZero(),
          "identity: stage cost sens_p");

    // sens_p: terminal cost
    DoubleIntegrator::term_grad_dp_t tgp_b, tgp_s;
    base.terminal_cost_sens_p(x, Lp_b, tgp_b);
    scaled.terminal_cost_sens_p(x, Lp_s, tgp_s);
    check((tgp_b - tgp_s).isZero(), "identity: term cost sens_p");

    // sens_p: stage inequality
    DoubleIntegrator::ineq_dg_dp_t igp_b, igp_s;
    DoubleIntegrator::ineq_dg_dx_dp_t igdx_b, igdx_s;
    DoubleIntegrator::ineq_dg_du_dp_t igdu_b, igdu_s;
    base.stage_inequality_sens_p(0, x, u, igp_b, igdx_b, igdu_b);
    scaled.stage_inequality_sens_p(0, x, u, igp_s, igdx_s, igdu_s);
    check((igp_b - igp_s).isZero() && (igdx_b - igdx_s).isZero() &&
          (igdu_b - igdu_s).isZero(),
          "identity: stage ineq sens_p");

    // sens_p: stage linear
    auto sll_b = base.stage_linear_sens_p(0);
    auto sll_s = scaled.stage_linear_sens_p(0);
    check((sll_b.dA_dp - sll_s.dA_dp).isZero() &&
          (sll_b.dB_dp - sll_s.dB_dp).isZero(),
          "identity: stage linear sens_p");

    // sens_p: state box
    auto sbb_b = base.stage_state_box_sens_p(0);
    auto sbb_s = scaled.stage_state_box_sens_p(0);
    check((sbb_b.dlo_dp - sbb_s.dlo_dp).isZero() &&
          (sbb_b.dhi_dp - sbb_s.dhi_dp).isZero(),
          "identity: state box sens_p");

    // sens_p: control box
    auto cbb_b = base.stage_control_box_sens_p(0);
    auto cbb_s = scaled.stage_control_box_sens_p(0);
    check((cbb_b.dlo_dp - cbb_s.dlo_dp).isZero() &&
          (cbb_b.dhi_dp - cbb_s.dhi_dp).isZero(),
          "identity: control box sens_p");

    // sens_p: terminal equality
    DoubleIntegrator::eq_term_de_dp_t telp_b, telp_s;
    DoubleIntegrator::eq_term_de_dx_dp_t teldx_b, teldx_s;
    base.terminal_equality_sens_p(x, telp_b, teldx_b);
    scaled.terminal_equality_sens_p(x, telp_s, teldx_s);
    check((telp_b - telp_s).isZero() && (teldx_b - teldx_s).isZero(),
          "identity: term eq sens_p");

    // sens_p: initial state
    auto x0p_b = base.initial_state_sens_p();
    auto x0p_s = scaled.initial_state_sens_p();
    check((x0p_b - x0p_s).isZero(), "identity: x0 sens_p");
}

// Test 9: synthetic unscale() — known values, verify every field.
void test_unscale_recovery()
{
    DoubleIntegrator base;
    ScaledProblem<DoubleIntegrator> scaled(base);
    scaled.sx_ << 2.0, 4.0;
    scaled.su_ << 0.5;

    Solution<ScaledProblem<DoubleIntegrator>> sol_s(N);
    sol_s.status = Status::kSolved;
    sol_s.cost_value = 42.5;

    for (int k = 0; k <= N; ++k)
    {
        sol_s.x[k](0) = 2.0 * (k + 1);   // q_s = 2*(k+1)  -> q = k+1
        sol_s.x[k](1) = 4.0 * (k + 2);   // v_s = 4*(k+2)  -> v = k+2
    }
    for (int k = 0; k < N; ++k)
    {
        sol_s.u[k](0) = 0.5 * k;         // u_s = 0.5*k    -> u = k
    }

    // dynamics multipliers
    for (int k = 0; k < N; ++k)
    {
        sol_s.lambda_dyn[k](0) = 2.0 * k + 1;
        sol_s.lambda_dyn[k](1) = 4.0 * k + 1;
    }

    // box state multipliers (2*nbx = 4 rows: [lo x0, lo x1, hi x0, hi x1])
    for (int k = 0; k <= N; ++k)
    {
        sol_s.lambda_box_state[k](0) = 10.0;
        sol_s.lambda_box_state[k](1) = 20.0;
        sol_s.lambda_box_state[k](2) = 30.0;
        sol_s.lambda_box_state[k](3) = 40.0;
    }

    // box control multipliers (2*nbu = 2 rows)
    for (int k = 0; k < N; ++k)
    {
        sol_s.lambda_box_control[k](0) = 5.0;
        sol_s.lambda_box_control[k](1) = 7.0;
    }

    // inequality / linear multipliers
    for (int k = 0; k < N; ++k)
    {
        sol_s.lambda_ineq_stage[k](0) = 3.0;
        sol_s.lambda_lin_stage[k](0)  = 1.5;
    }
    sol_s.lambda_eq_term(0) = 9.0;

    auto sol = scaled.unscale(sol_s);

    // verify primal
    for (int k = 0; k <= N; ++k)
    {
        check_close(sol.x[k](0), static_cast<double>(k + 1), 1e-12,
                    "unscale x q");
        check_close(sol.x[k](1), static_cast<double>(k + 2), 1e-12,
                    "unscale x v");
    }
    for (int k = 0; k < N; ++k)
    {
        check_close(sol.u[k](0), static_cast<double>(k), 1e-12,
                    "unscale u");
    }

    // verify dynamics multipliers: lam = sx * lam_s
    for (int k = 0; k < N; ++k)
    {
        check_close(sol.lambda_dyn[k](0),
                    2.0 * (2.0 * k + 1), 1e-12, "unscale lam_dyn q");
        check_close(sol.lambda_dyn[k](1),
                    4.0 * (4.0 * k + 1), 1e-12, "unscale lam_dyn v");
    }

    // verify box state multipliers
    for (int k = 0; k <= N; ++k)
    {
        check_close(sol.lambda_box_state[k](0), 10.0 * 2.0, 1e-12,
                    "unscale box lo x0");
        check_close(sol.lambda_box_state[k](1), 20.0 * 4.0, 1e-12,
                    "unscale box lo x1");
        check_close(sol.lambda_box_state[k](2), 30.0 * 2.0, 1e-12,
                    "unscale box hi x0");
        check_close(sol.lambda_box_state[k](3), 40.0 * 4.0, 1e-12,
                    "unscale box hi x1");
    }

    // verify box control multipliers
    for (int k = 0; k < N; ++k)
    {
        check_close(sol.lambda_box_control[k](0), 5.0 * 0.5, 1e-12,
                    "unscale box lo u");
        check_close(sol.lambda_box_control[k](1), 7.0 * 0.5, 1e-12,
                    "unscale box hi u");
    }

    // invariant multipliers
    for (int k = 0; k < N; ++k)
    {
        check_close(sol.lambda_ineq_stage[k](0), 3.0, 1e-12,
                    "unscale ineq lam");
        check_close(sol.lambda_lin_stage[k](0), 1.5, 1e-12,
                    "unscale lin lam");
    }
    check_close(sol.lambda_eq_term(0), 9.0, 1e-12, "unscale eq term lam");

    // cost and status
    check_close(sol.cost_value, 42.5, 0.0, "unscale cost");
    check(sol.status == Status::kSolved, "unscale status");
}

}  // namespace

int main()
{
    test_value_invariance();
    test_gradient_transform();
    test_hessian_transform();
    test_jacobian_transform();
    test_box_transform();
    test_initial_state();
    test_e2e_equivalence();
    test_identity_scaling();
    test_unscale_recovery();

    if (failures > 0)
    {
        std::fprintf(stderr, "scaling_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("scaling_test: OK\n");
    return 0;
}

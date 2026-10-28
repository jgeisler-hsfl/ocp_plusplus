// Tutorial chapter 6 companion example:
// a continuous-time OCP solved two ways with the same model, cost and
// constraints — explicit RK4 (automatic differentiation path) and implicit
// RK / Gauss-Legendre collocation (DAE residual path) — demonstrating that
// nothing downstream of the problem interface changes.
//
// Model: a mass on a nonlinear (Duffing) spring with force input,
//        x = [q; v],  u = [F]
//
//        q_dot = v
//        v_dot = -q - 0.5*q^3 + F
//
// The ODE provides a dual-evaluable (scalar-templated) f for the explicit
// integrator's AD path, plus hand-written jacobian/hess_prod which the
// implicit integrator consumes through the OdeToDae adapter.
//
//   cost:   0.5*(q1*q^2 + q2*v^2 + r*F^2),  terminal 0.5*(qf*q^2 + qf*v^2)
//   box:    -1 <= F <= 1 (control box)
//   fixed:  x_0 = [1; 0]

#include <cmath>
#include <cstdio>

#include "ocp/integrators/continuous_problem.hpp"
#include "ocp/integrators/dae_model.hpp"
#include "ocp/integrators/rk_explicit.hpp"
#include "ocp/integrators/rk_implicit.hpp"
#include "ocp/solvers/acados/acados.hpp"

// ---------------------------------------------------------------------------
// Structural dimensions (same for both integrator variants).
// ---------------------------------------------------------------------------
struct DuffingDims
{
    static constexpr int nx = 2;   // [q; v]
    static constexpr int nu = 1;   // [F]

    static constexpr int ng   = 0;
    static constexpr int ne   = 0;
    static constexpr int nl   = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 0;
    static constexpr int nl_t = 0;

    static constexpr bool fixed_initial_state = true;
    // exact Hessian model: dynamics_value_jac_hess is used (the base class
    // builds it from the integrator; no dynamics code of our own)
    static constexpr bool has_dynamics_hess = true;
    static constexpr bool has_constr_hess   = false;

    static constexpr std::array<int, 0> state_box_idx   = {};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 0> terminal_state_box_idx = {};

    static constexpr std::array<int, 0> ineq_soft_idx   = {};
    static constexpr std::array<int, 0> eq_soft_idx     = {};
    static constexpr std::array<int, 0> lin_soft_idx    = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx   = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx  = {};
    static constexpr std::array<int, 0> state_box_soft_idx   = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

// ---------------------------------------------------------------------------
// The ODE model (duck-typed contract: f + jacobian + hess_prod).
//
// f is dual-evaluable (a template in the scalar type): the explicit RK
// integrator then computes value + Jacobian in one dual pass and the
// Hessian via a nested-dual pass, and never calls our hand-written
// jacobian/hess_prod. The implicit integrator has no AD path: it consumes
// them through the OdeToDae adapter below.
// ---------------------------------------------------------------------------
struct DuffingOde
{
    using P = ocp::Problem<DuffingDims>;
    using state_t   = typename P::state_t;
    using control_t = typename P::control_t;
    using dyn_df_dx_t = typename P::dyn_df_dx_t;
    using dyn_df_du_t = typename P::dyn_df_du_t;

    // dual-evaluable right-hand side (plain arithmetic only)
    template <class D>
    Eigen::Matrix<D, 2, 1> f(const Eigen::Matrix<D, 2, 1>& x,
                             const Eigen::Matrix<D, 1, 1>& u,
                             double /*t*/) const
    {
        Eigen::Matrix<D, 2, 1> xd;
        xd(0) = x(1);
        xd(1) = -x(0) - 0.5 * x(0) * x(0) * x(0) + u(0);
        return xd;
    }

    // hand-written Jacobian blocks (used by the implicit integrator)
    void jacobian(const state_t& x, const control_t& /*u*/, double /*t*/,
                  dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
    {
        df_dx << 0.0, 1.0,
                 -(1.0 + 1.5 * x(0) * x(0)), 0.0;
        df_du << 0.0, 1.0;
    }

    // hand-written bilinear Hessian-vector product of the ODE map:
    //   hv = sum_i w_i * D^2 f_i(x, u) * v
    // f is linear in v except d^2 f_2/dq^2 = -3*q
    void hess_prod(const state_t& x, const control_t& /*u*/, double /*t*/,
                   const state_t& w, const state_t& v_x,
                   const control_t& /*v_u*/,
                   state_t& hv_x, control_t& hv_u) const
    {
        hv_x(0) = 0.0;
        hv_x(1) = -3.0 * x(0) * v_x(0) * w(1);
        hv_u.setZero();
    }
};

// ---------------------------------------------------------------------------
// The problem: cost + box + initial state, dynamics provided by
// ContinuousProblem + the integrator. Templated on the base so the same
// code serves both integrator variants.
// ---------------------------------------------------------------------------
template <class Base>
class DuffingProblem : public Base
{
public:
    using typename Base::state_t;
    using typename Base::control_t;
    using typename Base::dyn_df_dx_t;
    using typename Base::dyn_df_du_t;
    using typename Base::stage_grad_t;
    using typename Base::stage_hess_t;
    using typename Base::term_grad_t;
    using typename Base::term_hess_t;
    using typename Base::control_box_t;

    template <class... Args>
    DuffingProblem(Args&&... args) : Base(std::forward<Args>(args)...)
    {
    }

    // user-managed cost parameters (not part of the interface)
    double q1_ = 1.0;
    double q2_ = 0.1;
    double r_  = 0.01;
    double qf_ = 5.0;

    state_t initial_state() const
    {
        state_t x0;
        x0 << 1.0, 0.0;
        return x0;
    }

    // --- stage cost ---------------------------------------------------
    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return 0.5 * (q1_ * x(0) * x(0) + q2_ * x(1) * x(1)
                      + r_ * u(0) * u(0));
    }

    void stage_cost_value_grad(int, const state_t& x, const control_t& u,
                               double& value, stage_grad_t& g) const
    {
        value = stage_cost_value(0, x, u);
        g(0) = q1_ * x(0);
        g(1) = q2_ * x(1);
        g(2) = r_ * u(0);
    }

    void stage_cost_value_grad_hess(int, const state_t& x, const control_t& u,
                                    double& value, stage_grad_t& g,
                                    stage_hess_t& H) const
    {
        stage_cost_value_grad(0, x, u, value, g);
        H.setZero();
        H(0, 0) = q1_;
        H(1, 1) = q2_;
        H(2, 2) = r_;
    }

    // --- terminal cost --------------------------------------------------
    double terminal_cost_value(const state_t& x) const
    {
        return 0.5 * qf_ * (x(0) * x(0) + x(1) * x(1));
    }

    void terminal_cost_value_grad(const state_t& x, double& value,
                                  term_grad_t& g) const
    {
        value = terminal_cost_value(x);
        g(0) = qf_ * x(0);
        g(1) = qf_ * x(1);
    }

    void terminal_cost_value_grad_hess(const state_t& x, double& value,
                                       term_grad_t& g, term_hess_t& H) const
    {
        terminal_cost_value_grad(x, value, g);
        H.setZero();
        H(0, 0) = qf_;
        H(1, 1) = qf_;
    }

    // --- box (control box only; see the index sets) ---------------------
    control_box_t stage_control_box_constr(int) const
    {
        control_box_t spec;
        spec.lo(0) = -1.0;
        spec.hi(0) =  1.0;
        spec.penalty = {};
        return spec;
    }
};

// ---------------------------------------------------------------------------
// Two problems: identical model, cost, constraints — different integrators.
// ---------------------------------------------------------------------------
template <class Prob>
static void solve_and_print(const char* name, const Prob& problem)
{
    const int N = 20;
    ocp::Solution<Prob> sol(N);
    sol.x[0] = problem.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = problem.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }

    ocp::AcadosOptions opts;
    // tolerances matched to the discretization: the default 1e-8 asks more
    // of the IRK second-order sensitivities than they deliver (the run then
    // stalls just above tol_stat and exits kMinStep)
    opts.tol_stat = 1e-6;
    opts.tol_eq = 1e-6;
    opts.tol_ineq = 1e-6;
    opts.tol_comp = 1e-6;
    ocp::AcadosSolver<Prob> solver(opts);
    solver.solve(problem, sol);

    std::printf("%-8s status=%d cost=%.9f iters=%d  q_N=%.6f v_N=%.6f\n",
                name, static_cast<int>(sol.status), sol.cost_value,
                solver.statistics().iter, sol.x[N](0), sol.x[N](1));
}

int main()
{
    DuffingOde ode;

    // variant 1: explicit RK4 (AD path: value/jacobian from duals)
    using Erk4 = ocp::ExplicitRkIntegrator<DuffingDims, DuffingOde,
                                           ocp::K4Tag>;
    using ErkBase = ocp::ContinuousProblem<Erk4>;
    DuffingProblem<ErkBase> erk(ode, 0.05);
    solve_and_print("ERK4", erk);

    // variant 2: implicit RK, Gauss-Legendre 4-stage collocation (the
    // acados IRK default), via the ODE -> DAE residual adapter
    using Dae = ocp::OdeToDae<DuffingOde, DuffingDims>;
    using IrkGl4 = ocp::ImplicitRkIntegrator<DuffingDims, Dae,
                                             ocp::GaussLegendre4Tag>;
    using IrkBase = ocp::ContinuousProblem<IrkGl4>;
    Dae dae{ode};
    DuffingProblem<IrkBase> irk(dae, 0.05);
    solve_and_print("IRK-GL4", irk);

    return 0;
}

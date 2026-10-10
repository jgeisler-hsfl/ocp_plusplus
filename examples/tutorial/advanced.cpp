// Tutorial chapter 7 companion example: advanced features.
//
// 1. compile-time horizon: Solution<P, NH> with stack-allocated, zero-
//    initialized trajectories (no heap allocation, no resize)
// 2. free initial state (MHE-style): fixed_initial_state = false makes x_0
//    a decision variable, bounded by the stage-0 state box
// 3. warm-started re-solve: shifting the solution and re-solving the same
//    solver object (receding-horizon pattern)
// 4. per-iteration statistics export to CSV
//
// Problem: the double integrator of chapters 2-3.

#include <cmath>
#include <cstdio>

#include "ocp/problem.hpp"
#include "ocp/solvers/acados/acados.hpp"

// ---------------------------------------------------------------------------
// Structural dimensions. Both variants share everything except
// fixed_initial_state: the MHE variant frees x_0 and bounds it with the
// stage-0 state box.
// ---------------------------------------------------------------------------
struct AdvDimsFixed
{
    static constexpr int nx = 2, nu = 1;
    static constexpr int ng = 0, ne = 0, nl = 0;
    static constexpr int ng_t = 0, ne_t = 0, nl_t = 0;
    static constexpr bool fixed_initial_state = true;
    static constexpr bool has_dynamics_hess = false;  // Gauss-Newton
    static constexpr bool has_constr_hess   = false;
    static constexpr std::array<int, 2> state_box_idx   = {0, 1};
    static constexpr std::array<int, 0> control_box_idx = {};
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

struct AdvDimsMhe
{
    static constexpr int nx = 2, nu = 1;
    static constexpr int ng = 0, ne = 0, nl = 0;
    static constexpr int ng_t = 0, ne_t = 0, nl_t = 0;
    static constexpr bool fixed_initial_state = false;  // x_0 is a decision
    static constexpr bool has_dynamics_hess = false;    // variable (MHE)
    static constexpr bool has_constr_hess   = false;
    static constexpr std::array<int, 2> state_box_idx   = {0, 1};  // also x_0
    static constexpr std::array<int, 0> control_box_idx = {};
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
// Shared cost + box code (mixin over the problem base).
// ---------------------------------------------------------------------------
template <class Base>
class AdvCost : public Base
{
public:
    using typename Base::state_t;
    using typename Base::control_t;
    using typename Base::stage_grad_t;
    using typename Base::stage_hess_t;
    using typename Base::term_grad_t;
    using typename Base::term_hess_t;
    using typename Base::state_box_t;

    double Ts_ = 0.1;
    double wq_ = 1.0;
    double wv_ = 0.1;
    double wa_ = 0.01;
    double wf_ = 10.0;

    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return 0.5 * (wq_ * x(0) * x(0) + wv_ * x(1) * x(1)
                      + wa_ * u(0) * u(0));
    }

    void stage_cost_value_grad(int, const state_t& x, const control_t& u,
                               double& value, stage_grad_t& g) const
    {
        value = stage_cost_value(0, x, u);
        g(0) = wq_ * x(0);
        g(1) = wv_ * x(1);
        g(2) = wa_ * u(0);
    }

    void stage_cost_value_grad_hess(int, const state_t& x, const control_t& u,
                                    double& value, stage_grad_t& g,
                                    stage_hess_t& H) const
    {
        stage_cost_value_grad(0, x, u, value, g);
        H.setZero();
        H(0, 0) = wq_;
        H(1, 1) = wv_;
        H(2, 2) = wa_;
    }

    double terminal_cost_value(const state_t& x) const
    {
        return wf_ * (x(0) * x(0) + x(1) * x(1));
    }

    void terminal_cost_value_grad(const state_t& x, double& value,
                                  term_grad_t& g) const
    {
        value = terminal_cost_value(x);
        g(0) = 2.0 * wf_ * x(0);
        g(1) = 2.0 * wf_ * x(1);
    }

    void terminal_cost_value_grad_hess(const state_t& x, double& value,
                                       term_grad_t& g, term_hess_t& H) const
    {
        terminal_cost_value_grad(x, value, g);
        H.setZero();
        H(0, 0) = 2.0 * wf_;
        H(1, 1) = 2.0 * wf_;
    }

    state_box_t stage_state_box_constr(int k) const
    {
        state_box_t spec;
        // MHE-style prior: a narrow stage-0 box bounds the estimated x_0.
        // Ignored when x_0 is fixed (the solver pins x_0 with internal pin
        // rows instead of the stage-0 state box).
        if (!Base::fixed_initial_state && k == 0)
        {
            spec.lo << 0.8, 0.3;
            spec.hi << 1.2, 0.7;
        }
        else
        {
            spec.lo << -5.0, -5.0;
            spec.hi <<  5.0,  5.0;
        }
        spec.penalty = {};
        return spec;
    }
};

// dynamics (shared); the fixed-x0 variant adds initial_state()
template <class Base>
class AdvDynamics : public AdvCost<Base>
{
public:
    using typename AdvCost<Base>::state_t;
    using typename AdvCost<Base>::control_t;
    using typename AdvCost<Base>::dyn_df_dx_t;
    using typename AdvCost<Base>::dyn_df_du_t;

    state_t dynamics_next_state(int, const state_t& x,
                                const control_t& u) const
    {
        state_t xn;
        xn(0) = x(0) + this->Ts_ * x(1);
        xn(1) = x(1) + this->Ts_ * u(0);
        return xn;
    }

    void dynamics_value_jac(int, const state_t& x, const control_t& u,
                            state_t& x_next, dyn_df_dx_t& df_dx,
                            dyn_df_du_t& df_du) const
    {
        x_next = this->dynamics_next_state(0, x, u);
        df_dx.setZero();
        df_dx(0, 0) = 1.0;
        df_dx(0, 1) = this->Ts_;
        df_dx(1, 1) = 1.0;
        df_du.setZero();
        df_du(1, 0) = this->Ts_;
    }
};

template <class Base>
class AdvFixedProblem : public AdvDynamics<Base>
{
public:
    using state_t = typename AdvDynamics<Base>::state_t;

    state_t initial_state() const
    {
        state_t x0;
        x0 << 1.0, 0.5;
        return x0;
    }
};

using AdvFixedBase = ocp::Problem<AdvDimsFixed>;
using AdvMheBase   = ocp::Problem<AdvDimsMhe>;
using AdvFixed     = AdvFixedProblem<AdvFixedBase>;
using AdvMhe       = AdvDynamics<AdvMheBase>;

// ---------------------------------------------------------------------------
// Part 1 + 3 + 4: compile-time horizon, warm-started re-solve, CSV stats.
// ---------------------------------------------------------------------------
static void fixed_horizon_and_warm_start(const AdvFixed& problem)
{
    constexpr int N = 10;

    // compile-time horizon: fixed-extent (stack) trajectories, zero-
    // initialized, no heap allocation and no resize call
    ocp::Solution<AdvFixed, N> sol;
    if (sol.N != N || sol.x.size() != std::size_t(N + 1) ||
        !sol.lambda_dyn[0].isZero())
    {
        std::printf("FAIL: fixed-horizon solution container\n");
        return;
    }

    // cold solve (iteration 1)
    sol.x[0] = problem.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = problem.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }
    ocp::AcadosSolver<AdvFixed, N> solver;
    solver.solve(problem, sol);
    std::printf("cold  solve: status=%d iters=%d cost=%.9f\n",
                static_cast<int>(sol.status), solver.statistics().iter,
                sol.cost_value);

    // warm-started re-solve (receding-horizon pattern): shift the
    // trajectory AND the multipliers one stage up, re-solve the same
    // solver object
    ocp::Solution<AdvFixed, N> warm(sol);  // copy = warm start
    for (int k = 0; k < N - 1; ++k)
    {
        warm.x[k] = sol.x[k + 1];
        warm.u[k] = sol.u[k + 1];
        warm.lambda_dyn[k] = sol.lambda_dyn[k + 1];
        warm.lambda_box_state[k] = sol.lambda_box_state[k + 1];
    }
    warm.lambda_dyn[N - 1].setZero();
    warm.lambda_box_state[N - 1].setZero();
    warm.x[0] = problem.initial_state();   // new initial state
    warm.u[N - 1].setZero();               // last control: hold guess
    solver.solve(problem, warm);
    std::printf("warm  solve: status=%d iters=%d cost=%.9f\n",
                static_cast<int>(warm.status), solver.statistics().iter,
                warm.cost_value);

    // per-iteration statistics to CSV
    solver.statistics().write_csv("tutorial_advanced_stat.csv");
    std::printf("wrote tutorial_advanced_stat.csv (%d rows)\n",
                solver.statistics().iter);
}

// ---------------------------------------------------------------------------
// Part 2: free initial state (MHE-style).
// ---------------------------------------------------------------------------
static void free_initial_state(const AdvMhe& problem)
{
    constexpr int N = 10;

    // x_0 is a decision variable now: initial_state() does not exist on
    // the problem; the stage-0 state box bounds the estimate
    ocp::Solution<AdvMhe, N> sol(N);
    sol.x[0] << 0.0, 0.0;  // guess for the initial state
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = problem.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }
    ocp::AcadosSolver<AdvMhe, N> solver;
    solver.solve(problem, sol);
    std::printf("MHE   solve: status=%d iters=%d cost=%.9f  "
                "x0=[%.4f; %.4f]\n",
                static_cast<int>(sol.status), solver.statistics().iter,
                sol.cost_value, sol.x[0](0), sol.x[0](1));
}

int main()
{
    AdvFixed fixed_problem;
    AdvMhe mhe_problem;

    std::printf("== compile-time horizon + warm start ==\n");
    fixed_horizon_and_warm_start(fixed_problem);

    std::printf("\n== free initial state (MHE-style) ==\n");
    free_initial_state(mhe_problem);

    return 0;
}

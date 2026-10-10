// Tutorial chapter 2 + 3 companion example:
// a minimal discrete-time optimal control problem, solved with the SQP
// solver. No constraints (all constraint dimensions zero), fixed initial
// state, runtime horizon.
//
//   x = [q; v]  (position, velocity),  u = [a]  (acceleration)
//
//   dynamics:    q_{k+1} = q_k + Ts * v_k
//                v_{k+1} = v_k + Ts * a_k
//
//   stage cost:  0.5 * (wq*q^2 + wv*v^2 + wa*a^2)
//   terminal:    wf * (q_N^2 + v_N^2)
//
// The program first verifies the hand-written derivatives against central
// finite differences (chapter 2), then solves the OCP and prints the
// solution and statistics (chapter 3).

#include <cmath>
#include <cstdio>

#include "ocp/problem.hpp"
#include "ocp/solvers/acados/acados.hpp"

// ---------------------------------------------------------------------------
// Step 1: the structural dimensions (a compile-time tag struct).
// ---------------------------------------------------------------------------
struct MinimalDims
{
    // state / control dimensions
    static constexpr int nx = 2;   // [q; v]
    static constexpr int nu = 1;   // [a]

    // constraint dimensions (all zero here; see tutorial chapter 4)
    static constexpr int ng   = 0;  // stage nonlinear inequalities g <= 0
    static constexpr int ne   = 0;  // stage nonlinear equalities   e == 0
    static constexpr int nl   = 0;  // stage linear two-sided rows
    static constexpr int ng_t = 0;  // terminal nonlinear inequalities
    static constexpr int ne_t = 0;  // terminal nonlinear equalities
    static constexpr int nl_t = 0;  // terminal linear two-sided rows

    // true: x_0 is a fixed given via initial_state(); false: x_0 is a
    // decision variable (e.g. moving-horizon estimation)
    static constexpr bool fixed_initial_state = true;

    // Hessian capabilities (tutorial chapter 5):
    // false: the solver assembles a Gauss-Newton QP Hessian (the dynamics
    //        map is linear here, so the dynamics Hessian is zero anyway)
    // true:  dynamics_value_jac_hess must be implemented and is used
    static constexpr bool has_dynamics_hess = false;
    static constexpr bool has_constr_hess   = false;

    // active box rows: index sets selecting the bounded components (empty
    // here: no box constraints; see tutorial chapter 4)
    static constexpr std::array<int, 0> state_box_idx   = {};
    static constexpr std::array<int, 0> control_box_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_idx = {};

    // soft constraint rows (all empty here; see tutorial chapter 4)
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
// Step 2: the concrete problem. Inherit ocp::Problem<MyDims> and implement
// the model functions. Which ones are required depends on the Dims above.
// ---------------------------------------------------------------------------
class MinimalProblem : public ocp::Problem<MinimalDims>
{
public:
    // user-managed problem parameters (not part of the interface)
    double Ts_ = 0.1;
    double wq_ = 1.0;
    double wv_ = 0.1;
    double wa_ = 0.01;
    double wf_ = 10.0;

    // --- initial state (required: fixed_initial_state = true) ---------
    state_t initial_state() const
    {
        state_t x0;
        x0 << 1.0, 0.5;
        return x0;
    }

    // --- dynamics (k = 0..N-1) ----------------------------------------
    state_t dynamics_next_state(int, const state_t& x,
                                const control_t& u) const
    {
        state_t xn;
        xn(0) = x(0) + Ts_ * x(1);
        xn(1) = x(1) + Ts_ * u(0);
        return xn;
    }

    // value + Jacobian in one call (the fused evaluation entry)
    void dynamics_value_jac(int, const state_t& x, const control_t& u,
                            state_t& x_next,
                            dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
    {
        x_next = dynamics_next_state(0, x, u);
        df_dx.setZero();
        df_dx(0, 0) = 1.0;
        df_dx(0, 1) = Ts_;
        df_dx(1, 1) = 1.0;
        df_du.setZero();
        df_du(1, 0) = Ts_;
    }

    // dynamics_value_jac_hess is NOT implemented: has_dynamics_hess = false
    // means the solver never calls it (Gauss-Newton QP Hessian).

    // --- stage cost (k = 0..N-1) --------------------------------------
    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return 0.5 * (wq_ * x(0) * x(0) + wv_ * x(1) * x(1)
                      + wa_ * u(0) * u(0));
    }

    void stage_cost_value_grad(int, const state_t& x, const control_t& u,
                               double& value, stage_grad_t& g) const
    {
        value = stage_cost_value(0, x, u);
        g(0) = wq_ * x(0);   // d/dq
        g(1) = wv_ * x(1);   // d/dv
        g(2) = wa_ * u(0);   // d/da   (layout [g_x; g_u])
    }

    // value + gradient + Hessian in one call; the Hessian is of the cost
    // only (exact here; the solver adds the dynamics/constraint parts)
    void stage_cost_value_grad_hess(int, const state_t& x,
                                    const control_t& u, double& value,
                                    stage_grad_t& g, stage_hess_t& H) const
    {
        stage_cost_value_grad(0, x, u, value, g);
        H.setZero();
        H(0, 0) = wq_;
        H(1, 1) = wv_;
        H(2, 2) = wa_;
    }

    // --- terminal cost (stage N; no control) --------------------------
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

    // All constraint dimensions are zero, so no constraint interface
    // function is required: the solver never calls them.
};

// ---------------------------------------------------------------------------
// Step 3: verify the derivatives against central finite differences.
// ---------------------------------------------------------------------------
static int nfail = 0;

static void check_close(const char* name, double a, double b, double tol)
{
    if (std::fabs(a - b) > tol * (1.0 + std::fabs(b)))
    {
        std::printf("FAIL %s: got %g, expected %g\n", name, a, b);
        ++nfail;
    }
    else
    {
        std::printf("ok   %s: %g\n", name, a);
    }
}

static void verify_derivatives(const MinimalProblem& problem)
{
    const int k = 3;
    MinimalProblem::state_t x;
    MinimalProblem::control_t u;
    x << 0.7, -0.2;
    u << 0.3;

    const double h = 1e-6;

    // stage cost gradient vs central finite differences
    MinimalProblem::stage_grad_t g;
    double lcost = 0.0;
    MinimalProblem::stage_hess_t H;
    problem.stage_cost_value_grad_hess(k, x, u, lcost, g, H);
    for (int i = 0; i < 3; ++i)
    {
        MinimalProblem::state_t xp = x, xm = x;
        MinimalProblem::control_t up = u, um = u;
        if (i < 2) { xp(i) += h; xm(i) -= h; }
        else       { up(0) += h; um(0) -= h; }
        const double fd = (problem.stage_cost_value(k, xp, up) -
                           problem.stage_cost_value(k, xm, um)) / (2.0 * h);
        check_close("stage grad", g(i), fd, 1e-4);
    }

    // dynamics Jacobian vs central finite differences
    MinimalProblem::dyn_df_dx_t A;
    MinimalProblem::dyn_df_du_t B;
    MinimalProblem::state_t x_next;
    problem.dynamics_value_jac(k, x, u, x_next, A, B);
    for (int j = 0; j < 2; ++j)
    {
        MinimalProblem::state_t xf = x;
        xf(j) += h;
        const auto fp = problem.dynamics_next_state(k, xf, u);
        xf(j) -= 2.0 * h;
        const auto fm = problem.dynamics_next_state(k, xf, u);
        for (int i = 0; i < 2; ++i)
        {
            const double fd = (fp(i) - fm(i)) / (2.0 * h);
            check_close("dyn df/dx", A(i, j), fd, 1e-4);
        }
    }
    for (int j = 0; j < 1; ++j)
    {
        MinimalProblem::control_t uf = u;
        uf(j) += h;
        const auto fp = problem.dynamics_next_state(k, x, uf);
        uf(j) -= 2.0 * h;
        const auto fm = problem.dynamics_next_state(k, x, uf);
        for (int i = 0; i < 2; ++i)
        {
            const double fd = (fp(i) - fm(i)) / (2.0 * h);
            check_close("dyn df/du", B(i, j), fd, 1e-4);
        }
    }
}

// ---------------------------------------------------------------------------
// Step 4: solve the OCP with the SQP solver.
// ---------------------------------------------------------------------------
static void solve_ocp(const MinimalProblem& problem)
{
    const int N = 10;  // horizon (runtime quantity)

    // the solution container doubles as warm start and result; the
    // constructor allocates all trajectories for the horizon
    ocp::Solution<MinimalProblem> sol(N);

    // initialize the iterate: x_0 from the problem, a zero-hold start
    sol.x[0] = problem.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = problem.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }

    ocp::AcadosOptions opts;
    opts.print_level = 1;  // print one row per SQP iteration
    ocp::AcadosSolver<MinimalProblem> solver(opts);

    const ocp::Status st = solver.solve(problem, sol);
    std::printf("\nstatus=%d (kSolved=%d) cost=%.12f SQP iters=%d\n",
                static_cast<int>(st), static_cast<int>(ocp::Status::kSolved),
                sol.cost_value, solver.statistics().iter);

    // the solution trajectories
    for (int k = 0; k <= N; ++k)
    {
        if (k < N)
        {
            std::printf("k=%2d  q=%9.5f  v=%9.5f  a=%9.5f\n",
                        k, sol.x[k](0), sol.x[k](1), sol.u[k](0));
        }
        else
        {
            std::printf("k=%2d  q=%9.5f  v=%9.5f\n",
                        k, sol.x[k](0), sol.x[k](1));
        }
    }
}

int main()
{
    MinimalProblem problem;

    std::printf("== derivative checks (chapter 2) ==\n");
    verify_derivatives(problem);

    std::printf("\n== SQP solve (chapter 3) ==\n");
    solve_ocp(problem);

    return nfail == 0 ? 0 : 1;
}

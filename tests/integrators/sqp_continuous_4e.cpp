// Phase 4e: end-to-end continuous SQP + regression.
//
// A continuous double-integrator OCP (qdot = v, vdot = a) with terminal
// equality q_N + v_N = 1 and box-bounded control is solved by SqpSolver.
// The ODE is linear so both ERK (K4) and IRK (Radau IIA 2) give the exact
// discrete map; the achieved cost must match the discrete DoubleIntegrator
// reference (same horizon, same step size, same cost weights).

#include <cmath>
#include <cstdio>

#include "ocp/integrators/continuous_problem.hpp"
#include "ocp/integrators/rk_implicit.hpp"
#include "ocp/solvers/acados/sqp.hpp"

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



constexpr int N = 10;

// --- Dims for the continuous OCP ----------------------------------------

struct ContOcpDims
{
    static constexpr int nx = 2;
    static constexpr int nu = 1;
    static constexpr int ng = 0;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 1;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = true;
    static constexpr bool has_dynamics_hess_prod = false;
    static constexpr bool has_constr_hess_prod = false;
    static constexpr std::array<int, 2> state_box_idx = {0, 1};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 2> terminal_state_box_idx = {0, 1};
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

using OCP = ocp::Problem<ContOcpDims>;
using state_t = OCP::state_t;
using control_t = OCP::control_t;

// --- ODE: qdot = v, vdot = a (linear) ------------------------------------

struct DoubleIntegratorOde
{
    state_t f(const state_t& x, const control_t& u) const
    {
        state_t r;
        r(0) = x(1);
        r(1) = u(0);
        return r;
    }
    void jacobian(const state_t&, const control_t&,
                  OCP::dyn_df_dx_t& df_dx, OCP::dyn_df_du_t& df_du) const
    {
        df_dx.setZero();
        df_dx(0, 1) = 1.0;
        df_du.setZero();
        df_du(1, 0) = 1.0;
    }
};

// --- ERK variant --------------------------------------------------------

using ErkInteg = ocp::ExplicitRkIntegrator<ContOcpDims, DoubleIntegratorOde, 4,
                                              ocp::K4Tag>;
using ErkBase = ocp::ContinuousProblem<ContOcpDims, DoubleIntegratorOde, ErkInteg>;

struct ErkOcp : ErkBase
{
    double wq_ = 1.0;
    double wv_ = 0.1;
    double wa_ = 0.01;
    double wf_ = 10.0;

    ErkOcp(double h) : ErkBase(DoubleIntegratorOde{}, h) {}

    state_t initial_state() const
    {
        state_t x0;
        x0 << 1.0, 0.5;
        return x0;
    }

    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return 0.5 * (wq_ * x(0) * x(0) + wv_ * x(1) * x(1) +
                      wa_ * u(0) * u(0));
    }

    OCP::stage_grad_t
    stage_cost_gradient(int, const state_t& x, const control_t& u) const
    {
        OCP::stage_grad_t g;
        g(0) = wq_ * x(0);
        g(1) = wv_ * x(1);
        g(2) = wa_ * u(0);
        return g;
    }

    OCP::stage_hess_t
    stage_cost_hessian(int, const state_t&, const control_t&) const
    {
        OCP::stage_hess_t H;
        H.setZero();
        H(0, 0) = wq_;
        H(1, 1) = wv_;
        H(2, 2) = wa_;
        return H;
    }

    double terminal_cost_value(const state_t& x) const
    {
        return wf_ * (x(0) * x(0) + x(1) * x(1));
    }

    OCP::term_grad_t terminal_cost_gradient(const state_t& x) const
    {
        OCP::term_grad_t g;
        g(0) = 2.0 * wf_ * x(0);
        g(1) = 2.0 * wf_ * x(1);
        return g;
    }

    OCP::term_hess_t terminal_cost_hessian(const state_t&) const
    {
        OCP::term_hess_t H;
        H.setZero();
        H(0, 0) = 2.0 * wf_;
        H(1, 1) = 2.0 * wf_;
        return H;
    }

    OCP::state_box_t stage_state_box_constr(int) const
    {
        OCP::state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.soft_penalty.setZero();
        return spec;
    }

    OCP::control_box_t stage_control_box_constr(int) const
    {
        OCP::control_box_t spec;
        spec.lo(0) = -1.0;
        spec.hi(0) = 1.0;
        spec.soft_penalty.setZero();
        return spec;
    }

    OCP::term_state_box_t terminal_state_box_constr() const
    {
        OCP::term_state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.soft_penalty.setZero();
        return spec;
    }

    OCP::eq_term_t terminal_equality_constr(const state_t& x) const
    {
        OCP::eq_term_t e;
        e(0) = x(0) + x(1) - 1.0;
        return e;
    }

    void terminal_equality_constr_jacobian(const state_t&, OCP::eq_term_de_dx_t& e_dx) const
    {
        e_dx.setZero();
        e_dx(0, 0) = 1.0;
        e_dx(0, 1) = 1.0;
    }

    void terminal_equality_constr_hess_prod(const state_t&, const OCP::eq_term_t&,
                                            const state_t&, state_t& hv) const
    {
        hv.setZero();
    }
};

// --- IRK variant --------------------------------------------------------

using IrkInteg = ocp::ImplicitRkIntegrator<ContOcpDims, DoubleIntegratorOde, 2,
                                             ocp::RadauIia2Tag>;
using IrkBase = ocp::ContinuousProblem<ContOcpDims, DoubleIntegratorOde, IrkInteg>;

struct IrkOcp : IrkBase
{
    double wq_ = 1.0;
    double wv_ = 0.1;
    double wa_ = 0.01;
    double wf_ = 10.0;

    IrkOcp(double h) : IrkBase(DoubleIntegratorOde{}, h, 5) {}

    state_t initial_state() const
    {
        state_t x0;
        x0 << 1.0, 0.5;
        return x0;
    }

    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return 0.5 * (wq_ * x(0) * x(0) + wv_ * x(1) * x(1) +
                      wa_ * u(0) * u(0));
    }

    OCP::stage_grad_t
    stage_cost_gradient(int, const state_t& x, const control_t& u) const
    {
        OCP::stage_grad_t g;
        g(0) = wq_ * x(0);
        g(1) = wv_ * x(1);
        g(2) = wa_ * u(0);
        return g;
    }

    OCP::stage_hess_t
    stage_cost_hessian(int, const state_t&, const control_t&) const
    {
        OCP::stage_hess_t H;
        H.setZero();
        H(0, 0) = wq_;
        H(1, 1) = wv_;
        H(2, 2) = wa_;
        return H;
    }

    double terminal_cost_value(const state_t& x) const
    {
        return wf_ * (x(0) * x(0) + x(1) * x(1));
    }

    OCP::term_grad_t terminal_cost_gradient(const state_t& x) const
    {
        OCP::term_grad_t g;
        g(0) = 2.0 * wf_ * x(0);
        g(1) = 2.0 * wf_ * x(1);
        return g;
    }

    OCP::term_hess_t terminal_cost_hessian(const state_t&) const
    {
        OCP::term_hess_t H;
        H.setZero();
        H(0, 0) = 2.0 * wf_;
        H(1, 1) = 2.0 * wf_;
        return H;
    }

    OCP::state_box_t stage_state_box_constr(int) const
    {
        OCP::state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.soft_penalty.setZero();
        return spec;
    }

    OCP::control_box_t stage_control_box_constr(int) const
    {
        OCP::control_box_t spec;
        spec.lo(0) = -1.0;
        spec.hi(0) = 1.0;
        spec.soft_penalty.setZero();
        return spec;
    }

    OCP::term_state_box_t terminal_state_box_constr() const
    {
        OCP::term_state_box_t spec;
        spec.lo << -10.0, -10.0;
        spec.hi << 10.0, 10.0;
        spec.soft_penalty.setZero();
        return spec;
    }

    OCP::eq_term_t terminal_equality_constr(const state_t& x) const
    {
        OCP::eq_term_t e;
        e(0) = x(0) + x(1) - 1.0;
        return e;
    }

    void terminal_equality_constr_jacobian(const state_t&, OCP::eq_term_de_dx_t& e_dx) const
    {
        e_dx.setZero();
        e_dx(0, 0) = 1.0;
        e_dx(0, 1) = 1.0;
    }

    void terminal_equality_constr_hess_prod(const state_t&, const OCP::eq_term_t&,
                                            const state_t&, state_t& hv) const
    {
        hv.setZero();
    }
};

// --- helpers -----------------------------------------------------------

// Solve the discrete DoubleIntegrator reference (N=10, Ts=0.1).
double solve_discrete_reference()
{
    DoubleIntegrator p;
    SqpOptions opts;
    SqpSolver<DoubleIntegrator> solver(opts);

    Solution<DoubleIntegrator> sol(N);
    // Warm start: constant control that satisfies terminal equality
    const double Ts = p.Ts_;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * N * (N - 1) / 2.0 + Ts * N;
    const double a = (1.0 - (q0 + v0) - Ts * N * v0) / den;
    sol.x[0] = p.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol.u[k](0) = a;
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }

    const Status st = solver.solve(p, sol);
    check(st == Status::kSolved, "discrete ref: solve kSolved");
    return sol.cost_value;
}

template <class Problem>
double solve_continuous(const Problem& p, const char* label)
{
    SqpOptions opts;
    SqpSolver<Problem> solver(opts);

    Solution<Problem> sol(N);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < N; ++k)
    {
        control_t u_k;
        u_k.setZero();
        sol.u[k] = u_k;
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }

    const Status st = solver.solve(p, sol);
    check(st == Status::kSolved, (std::string(label) + ": kSolved").c_str());
    check(sol.status == Status::kSolved, (std::string(label) + ": sol.status").c_str());

    const NlpResiduals res = compute_nlp_residuals(p, sol);
    check(res.res_stat < 1e-6, (std::string(label) + ": res_stat").c_str());
    check(res.res_eq < 1e-6, (std::string(label) + ": res_eq").c_str());
    check(res.res_ineq < 1e-6, (std::string(label) + ": res_ineq").c_str());
    check(res.res_comp < 1e-6, (std::string(label) + ": res_comp").c_str());

    return sol.cost_value;
}

}  // namespace

int main()
{
    const double ref_cost = solve_discrete_reference();
    std::printf("discrete reference cost = %.10f\n", ref_cost);

    {
        ErkOcp prob(0.1);
        const double cost = solve_continuous(prob, "ERK K4");
        std::printf("ERK K4    cost = %.10f\n", cost);
        check(std::fabs(cost - ref_cost) / ref_cost < 0.05,
              "ERK K4 cost within 5% of discrete ref");
    }

    {
        IrkOcp prob(0.1);
        const double cost = solve_continuous(prob, "IRK Radau2");
        std::printf("IRK Rad2  cost = %.10f\n", cost);
        check(std::fabs(cost - ref_cost) / ref_cost < 0.05,
              "IRK Radau2 cost within 5% of discrete ref");
    }

    if (failures > 0)
    {
        std::fprintf(stderr, "sqp_continuous: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("sqp_continuous: OK\n");
    return 0;
}

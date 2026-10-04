// Phase 5a test: QP Hessian modes (GN_HESSIAN_PLAN.md sec. 3a.1).
//
// One quadratic problem is instantiated with two Dims tags:
//   ExactDims: has_dynamics_hess = true,  has_constr_hess = true
//   GnDims:    has_dynamics_hess = false, has_constr_hess = false
//
// Checks:
//   1. GN mode: first-stage hess (u;x) == H_cost exactly, terminal hess ==
//      H_cost_term exactly (no dynamics / constraint HVP terms); BA, b, DC,
//      d and the slack diagonals are unaffected by the Hessian mode.
//   2. EXACT mode: hess == H_cost - M_dyn + M_ineq (the hand-computed
//      composition also covered by assemble_2f, here for the same problem
//      so the GN/EXACT delta is exactly -M_dyn + M_ineq).
//   3. EXACT Dims with the runtime gate compute_hess = false: hess == GN.
//   4. FD cross-check of the EXACT first-stage hess: central finite
//      differences of the stage Lagrangian
//          L_cost(x,u) + lam_dyn (x1_bar - f(x,u)) + lam_ineq g(x,u)
//      w.r.t. (u, x) at the fixed iterate.  This verifies the HVP call
//      convention (multiplier w, v_x/v_u split), the minus sign of the
//      dynamics term, and the (u;x) layout end-to-end.

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/solvers/acados/sqp.hpp"

namespace
{

using namespace ocp;

int failures = 0;

void check_close(double a, double b, const char* msg, double tol = 1e-12)
{
    if (std::fabs(a - b) > tol)
    {
        std::fprintf(stderr, "FAIL: %s (%.15e vs %.15e)\n", msg, a, b);
        ++failures;
    }
}

// ---------------------------------------------------------------------
// Dims tags (identical structure, different Hessian-mode flags)
// ---------------------------------------------------------------------

namespace dims_base
{
constexpr int nx = 1;
constexpr int nu = 1;
}

struct ExactDims
{
    static constexpr int nx = dims_base::nx;
    static constexpr int nu = dims_base::nu;
    static constexpr int ng = 1;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 1;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = false;
    static constexpr bool has_dynamics_hess = true;
    static constexpr bool has_constr_hess = true;
    static constexpr std::array<int, 1> state_box_idx = {0};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 0> terminal_state_box_idx = {};
    static constexpr std::array<int, 1> ineq_soft_idx = {0};
    static constexpr std::array<int, 0> eq_soft_idx = {};
    static constexpr std::array<int, 0> lin_soft_idx = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 1> terminal_eq_soft_idx = {0};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {};
    static constexpr std::array<int, 0> state_box_soft_idx = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

struct GnDims
{
    static constexpr int nx = dims_base::nx;
    static constexpr int nu = dims_base::nu;
    static constexpr int ng = 1;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 1;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = false;
    static constexpr bool has_dynamics_hess = false;
    static constexpr bool has_constr_hess = false;
    static constexpr std::array<int, 1> state_box_idx = {0};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 0> terminal_state_box_idx = {};
    static constexpr std::array<int, 1> ineq_soft_idx = {0};
    static constexpr std::array<int, 0> eq_soft_idx = {};
    static constexpr std::array<int, 0> lin_soft_idx = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 1> terminal_eq_soft_idx = {0};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {};
    static constexpr std::array<int, 0> state_box_soft_idx = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

// ---------------------------------------------------------------------
// Problem (same math as QuadTest in assemble_2f.cpp):
//
//   p scalar state, a scalar control, Ts = 0.2, N = 1, x_0 free
//   dynamics:  p_{k+1} = p + Ts p a
//   stage cost: p^2 + 0.5 p a + 0.25 a^2
//   terminal:   4 p^2
//   ineq:       p a + p - 0.5 <= 0     (soft, penalty 7)
//   term eq:    p^2 - 1 = 0            (soft, penalty 3)
//   box:        -5 <= p <= 5, -2 <= a <= 2 (hard)
//
// H_cost (u;x) = [[0.5, 0.5], [0.5, 2.0]]
// M_dyn (lam = 1.5): [[0, 0.3], [0.3, 0]]
// M_ineq (lam = 0.4): [[0, 0.4], [0.4, 0]]
// ---------------------------------------------------------------------

template <class Dims>
class HessModeProb : public ocp::Problem<Dims>
{
public:
    using P = ocp::Problem<Dims>;
    using state_t = typename P::state_t;
    using control_t = typename P::control_t;

    static constexpr double Ts = 0.2;

    state_t dynamics_next_state(int, const state_t& x,
                                const control_t& u) const
    {
        return state_t::Constant(1, x(0) + Ts * x(0) * u(0));
    }

    void dynamics_value_jac(int, const state_t& x, const control_t& u,
        state_t& x_next,
        typename P::dyn_df_dx_t& df_dx,
        typename P::dyn_df_du_t& df_du) const
    {
        x_next(0) = x(0) + Ts * x(0) * u(0);
        df_dx(0, 0) = 1.0 + Ts * u(0);
        df_du(0, 0) = Ts * x(0);
    }

    void dynamics_value_jac_hess(int, const state_t& x, const control_t& u,
                                 const state_t& w, state_t& x_next,
                                 typename P::dyn_df_dx_t& df_dx,
                                 typename P::dyn_df_du_t& df_du,
                                 typename P::dyn_hess_t& hess) const
    {
        dynamics_value_jac(0, x, u, x_next, df_dx, df_du);
        // contracted Hessian w^T D2f, [x; u] layout:
        // D2f = Ts * [[0, 1], [1, 0]]  =>  w * Ts * [[0,1],[1,0]]
        hess.setZero();
        hess(0, 1) = w(0) * Ts;
        hess(1, 0) = w(0) * Ts;
    }

    double stage_cost_value(int, const state_t& x,
                            const control_t& u) const
    {
        return x(0) * x(0) + 0.5 * x(0) * u(0) + 0.25 * u(0) * u(0);
    }

    void stage_cost_value_grad(int, const state_t& x, const control_t& u,
                               double& value,
                               typename P::stage_grad_t& grad) const
    {
        value = x(0) * x(0) + 0.5 * x(0) * u(0) + 0.25 * u(0) * u(0);
        grad(0) = 0.5 * x(0) + 0.5 * u(0);
        grad(1) = 2.0 * x(0) + 0.5 * u(0);
    }

    void stage_cost_value_grad_hess(int, const state_t& x, const control_t& u,
                                    double& value,
                                    typename P::stage_grad_t& grad,
                                    typename P::stage_hess_t& hess) const
    {
        stage_cost_value_grad(0, x, u, value, grad);
        hess << 2.0, 0.5, 0.5, 0.5;  // [x; u] layout
    }

    double terminal_cost_value(const state_t& x) const
    {
        return 4.0 * x(0) * x(0);
    }

    void terminal_cost_value_grad(const state_t& x, double& value,
                                  typename P::term_grad_t& grad) const
    {
        value = 4.0 * x(0) * x(0);
        grad(0) = 8.0 * x(0);
    }

    void terminal_cost_value_grad_hess(const state_t& x, double& value,
                                       typename P::term_grad_t& grad,
                                       typename P::term_hess_t& hess) const
    {
        terminal_cost_value_grad(x, value, grad);
        hess(0, 0) = 8.0;
    }

    typename P::ineq_t stage_inequality_value(int, const state_t& x,
                                               const control_t& u) const
    {
        typename P::ineq_t g;
        g(0) = x(0) * u(0) + x(0) - 0.5;
        return g;
    }

    void stage_inequality_value_jac(int, const state_t& x,
                                          const control_t& u,
                                          typename P::ineq_t& g,
                                          typename P::ineq_dg_dx_t& g_dx,
                                          typename P::ineq_dg_du_t& g_du) const
    {
        g(0) = x(0) * u(0) + x(0) - 0.5;
        g_dx(0, 0) = u(0) + 1.0;
        g_du(0, 0) = x(0);
    }

    void stage_inequality_value_jac_hess(int, const state_t& x,
                                          const control_t& u,
                                          const typename P::ineq_t& lam,
                                          typename P::ineq_t& g,
                                          typename P::ineq_dg_dx_t& g_dx,
                                          typename P::ineq_dg_du_t& g_du,
                                          typename P::constr_hess_t& hess) const
    {
        stage_inequality_value_jac(0, x, u, g, g_dx, g_du);
        // contracted Hessian lam^T D2g, [x; u] layout:
        // D2g = [[0, 1], [1, 0]]
        hess.setZero();
        hess(0, 1) = lam(0);
        hess(1, 0) = lam(0);
    }

    typename P::ineq_pen_t stage_inequality_constr_soft_penalty(int) const
    {
        typename P::ineq_pen_t w;
        w(0) = 7.0;
        return w;
    }

    typename P::state_box_t stage_state_box_constr(int) const
    {
        typename P::state_box_t spec;
        spec.lo(0) = -5.0;
        spec.hi(0) = 5.0;
        spec.soft_penalty(0) = 0.0;
        return spec;
    }

    typename P::control_box_t stage_control_box_constr(int) const
    {
        typename P::control_box_t spec;
        spec.lo(0) = -2.0;
        spec.hi(0) = 2.0;
        spec.soft_penalty(0) = 0.0;
        return spec;
    }

    typename P::eq_term_t terminal_equality_value(const state_t& x) const
    {
        typename P::eq_term_t e;
        e(0) = x(0) * x(0) - 1.0;
        return e;
    }

    void terminal_equality_value_jac(const state_t& x,
                                          typename P::eq_term_t& e,
                                          typename P::eq_term_de_dx_t& e_dx)
        const
    {
        e(0) = x(0) * x(0) - 1.0;
        e_dx(0, 0) = 2.0 * x(0);
    }

    void terminal_equality_value_jac_hess(const state_t& x,
                                           const typename P::eq_term_t& lam,
                                           typename P::eq_term_t& e,
                                           typename P::eq_term_de_dx_t& e_dx,
                                           typename P::term_constr_hess_t& hess)
        const
    {
        terminal_equality_value_jac(x, e, e_dx);
        // contracted Hessian lam^T D2e, state-only: D2e = 2
        hess.setZero();
        hess(0, 0) = 2.0 * lam(0);
    }

    typename P::eq_term_pen_t terminal_equality_constr_soft_penalty() const
    {
        typename P::eq_term_pen_t w;
        w(0) = 3.0;
        return w;
    }
};

// ---------------------------------------------------------------------
// Iterate (N = 1, single stage => first)
//   x_0 = 0.3, u_0 = 0.8, x_1 = 0.5
//   lambda_dyn = 1.5, lambda_ineq = 0.4, lambda_eq_term = -2.5
// ---------------------------------------------------------------------

constexpr double x0 = 0.3, u0 = 0.8, x1b = 0.5;
constexpr double lam_dyn = 1.5, lam_ineq = 0.4, lam_eq_t = -2.5;

// FD stage-Lagrangian value for check 4.
double stage_lagrangian(const HessModeProb<ExactDims>& prob,
                        const HessModeProb<ExactDims>::state_t& x,
                        const HessModeProb<ExactDims>::control_t& u)
{
    const auto f = prob.dynamics_next_state(0, x, u);
    const auto g = prob.stage_inequality_value(0, x, u);
    return prob.stage_cost_value(0, x, u)
        + lam_dyn * (x1b - f(0)) + lam_ineq * g(0);
}

template <class P>
void check_hess_block(const Qp<P, Eigen::Dynamic>& qp, bool exact,
                      const std::string& tag)
{
    // first-stage hess (u; x; s): (u;x) block is the upper-left 2x2,
    // the slack diagonal (7) is mode-independent.
    const auto& s = qp.first;
    double huu, hux, hxx;
    if (exact)
    {
        huu = 0.5;
        hux = 0.6;
        hxx = 2.0;
    }
    else
    {
        huu = 0.5;
        hux = 0.5;
        hxx = 2.0;
    }
    const auto msg_uu = tag + " hess(u,u)";
    const auto msg_ux = tag + " hess(u,x)";
    const auto msg_xx = tag + " hess(x,x)";
    check_close(s.hess(0, 0), huu, msg_uu.c_str());
    check_close(s.hess(0, 1), hux, msg_ux.c_str());
    check_close(s.hess(1, 0), hux, msg_ux.c_str());
    check_close(s.hess(1, 1), hxx, msg_xx.c_str());
    check_close(s.hess(2, 2), 7.0, (tag + " slack diag").c_str());

    // terminal hess (x; s_lo; s_hi): exact adds the eq HVP 2 * lam_eq_t
    const double term_exp = exact ? 3.0 : 8.0;
    const auto msg_t = tag + " term hess";
    check_close(qp.term.hess(0, 0), term_exp, msg_t.c_str());
    check_close(qp.term.hess(1, 1), 3.0, (tag + " term slack lo").c_str());
    check_close(qp.term.hess(2, 2), 3.0, (tag + " term slack hi").c_str());
}

// ---------------------------------------------------------------------
// Check 1 + 2: GN and EXACT assembly of the same problem.
// ---------------------------------------------------------------------

template <class Dims>
void check_assembly(const std::string& tag)
{
    const bool exact = Dims::has_dynamics_hess;
    HessModeProb<Dims> prob;
    static constexpr int N = 1;

    Solution<HessModeProb<Dims>> sol(N);
    sol.x[0] << x0;
    sol.x[1] << x1b;
    sol.u[0] << u0;
    sol.lambda_dyn[0] << lam_dyn;
    sol.lambda_ineq_stage[0] << lam_ineq;
    sol.lambda_eq_term << lam_eq_t;

    SqpOptions opts;
    SqpSolver<HessModeProb<Dims>> solver(opts);
    solver.resize(N);

    const Status st = solver.assemble_qp(prob, sol);
    if (st != Status::kSolved)
    {
        std::fprintf(stderr, "FAIL: %s assemble status %d\n", tag.c_str(),
                     static_cast<int>(st));
        ++failures;
        return;
    }

    const auto& qp = solver.last_qp();
    check_hess_block(qp, exact, tag);

    // dynamics linearization is mode-independent
    check_close(qp.first.BA(0, 0), HessModeProb<Dims>::Ts * x0,
                (tag + " BA B").c_str());
    check_close(qp.first.BA(0, 1), 1.0 + HessModeProb<Dims>::Ts * u0,
                (tag + " BA A").c_str());
    check_close(qp.first.b(0), x0 + HessModeProb<Dims>::Ts * x0 * u0 - x1b,
                (tag + " b").c_str());
}

// ---------------------------------------------------------------------
// Check 4: FD cross-check of the EXACT first-stage hess.
// ---------------------------------------------------------------------

void check_fd_exact()
{
    HessModeProb<ExactDims> prob;
    using state_t = typename HessModeProb<ExactDims>::state_t;
    using control_t = typename HessModeProb<ExactDims>::control_t;

    const double h = 1e-5;
    state_t x;
    control_t u;
    x(0) = x0;
    u(0) = u0;

    auto fval = [&](double up, double xp)
    {
        state_t xx;
        control_t uu;
        xx(0) = xp;
        uu(0) = up;
        return stage_lagrangian(prob, xx, uu);
    };
    const double f0 = fval(u0, x0);
    const double huu = (fval(u0 + h, x0) - 2.0 * f0 + fval(u0 - h, x0))
        / (h * h);
    const double hux = (fval(u0 + h, x0 + h) - fval(u0 + h, x0 - h)
                        - fval(u0 - h, x0 + h) + fval(u0 - h, x0 - h))
        / (4.0 * h * h);
    const double hxx = (fval(u0, x0 + h) - 2.0 * f0 + fval(u0, x0 - h))
        / (h * h);

    // assembled EXACT hess (u;x) block
    Solution<HessModeProb<ExactDims>> sol(1);
    sol.x[0] << x0;
    sol.x[1] << x1b;
    sol.u[0] << u0;
    sol.lambda_dyn[0] << lam_dyn;
    sol.lambda_ineq_stage[0] << lam_ineq;
    sol.lambda_eq_term << lam_eq_t;

    SqpSolver<HessModeProb<ExactDims>> solver;
    solver.resize(1);
    if (solver.assemble_qp(prob, sol) != Status::kSolved)
    {
        std::fprintf(stderr, "FAIL: FD EXACT assemble status\n");
        ++failures;
        return;
    }
    const auto& s = solver.last_qp().first;

    check_close(s.hess(0, 0), huu, "FD exact hess(u,u)", 1e-5);
    check_close(s.hess(0, 1), hux, "FD exact hess(u,x)", 1e-5);
    check_close(s.hess(1, 1), hxx, "FD exact hess(x,x)", 1e-5);
}

}  // namespace

int run_hessmode_5a_tests()
{
    check_assembly<ExactDims>("exact");
    check_assembly<GnDims>("gn");

    // check 3: runtime gate -- EXACT Dims but compute_hess = false behaves
    // like GN (HVP terms dropped).
    {
        HessModeProb<ExactDims> prob;
        Solution<HessModeProb<ExactDims>> sol(1);
        sol.x[0] << x0;
        sol.x[1] << x1b;
        sol.u[0] << u0;
        sol.lambda_dyn[0] << lam_dyn;
        sol.lambda_ineq_stage[0] << lam_ineq;
        sol.lambda_eq_term << lam_eq_t;

        SqpOptions opts;
        opts.compute_hess = false;
        SqpSolver<HessModeProb<ExactDims>> solver(opts);
        solver.resize(1);
        if (solver.assemble_qp(prob, sol) != Status::kSolved)
        {
            std::fprintf(stderr, "FAIL: runtime-gate assemble status\n");
            ++failures;
        }
        else
        {
            const auto& s = solver.last_qp().first;
            check_close(s.hess(0, 0), 0.5, "gate-off hess(u,u)");
            check_close(s.hess(0, 1), 0.5, "gate-off hess(u,x)");
            check_close(s.hess(1, 1), 2.0, "gate-off hess(x,x)");
            check_close(solver.last_qp().term.hess(0, 0), 8.0,
                        "gate-off term hess");
        }
    }

    check_fd_exact();
    return failures;
}

// ocp/scaling.hpp
//
// Diagonal state/input scaling wrapper for the ocp++ problem interface.
//
// ScaledProblem<Base> presents the same ocp::Problem interface as the
// wrapped problem Base, but in scaled coordinates:
//
//     x_s = sx_ * x,    u_s = su_ * u
//
// All model evaluations are forwarded to the wrapped problem in the
// original (unscaled) coordinates; values, Jacobians and Hessians are
// transformed back so the scaled problem is exactly equivalent to the
// original one (same optimum, same cost value).
//
// Recovery: after a solver run, call unscale(sol) to convert the
// Solution back to the original coordinates.
//
// See docs/plans/SCALED_PROBLEM_PLAN.md for the full transform table.

#pragma once

#include "ocp/problem.hpp"

#include <utility>

#include <Eigen/Dense>

namespace ocp {

// =========================================================================
//  ScaledProblem
// =========================================================================

/// Diagonal state/input scaling wrapper.
///
/// Wraps a concrete problem `Base` (satisfying `ocp::Problem<Dims>` for
/// some Dims) and presents the identical interface in scaled coordinates:
///
///   x_s = sx_ * x,    u_s = su_ * u
///
/// `sx_` / `su_` are public, user-managed members (like problem
/// parameters); every entry must be strictly positive.  Changes take
/// effect on the next model call.  The wrapped problem is referenced,
/// not copied: keep it alive for the lifetime of the wrapper.
///
/// The wrapper is a drop-in replacement for `Base` in solvers:
///
///   DoubleIntegrator raw;
///   ScaledProblem<DoubleIntegrator> scaled(raw);
///   scaled.sx_ << 1.0, 0.1;
///   scaled.su_ << 0.5;
///
/// The solver's solution is in scaled coordinates; call `unscale()` to
/// recover the solution of the original problem.
///
/// @tparam Base  the unscaled concrete problem type
template <class Base>
class ScaledProblem
    : public Problem<typename Base::dims_t, typename Base::scalar_t>
{
    // C++ dependent-base name lookup requires explicit qualification for
    // members of the (dependent) base class.  Re-expose the type aliases
    // and static values used in this class body via local aliases.

public:
    // ---------------------------------------------------------------
    // Re-exposed type aliases (shadow the inherited names)
    // ---------------------------------------------------------------
    using scalar_t    = typename Base::scalar_t;
    using state_t     = typename Base::state_t;
    using control_t   = typename Base::control_t;
    using stage_grad_t    = typename Base::stage_grad_t;
    using stage_hess_t    = typename Base::stage_hess_t;
    using term_grad_t     = typename Base::term_grad_t;
    using term_hess_t     = typename Base::term_hess_t;
    using dyn_df_dx_t     = typename Base::dyn_df_dx_t;
    using dyn_df_du_t     = typename Base::dyn_df_du_t;
    using dyn_hess_t      = typename Base::dyn_hess_t;
    using ineq_t          = typename Base::ineq_t;
    using ineq_dg_dx_t    = typename Base::ineq_dg_dx_t;
    using ineq_dg_du_t    = typename Base::ineq_dg_du_t;
    using constr_hess_t   = typename Base::constr_hess_t;
    using eq_t            = typename Base::eq_t;
    using eq_de_dx_t      = typename Base::eq_de_dx_t;
    using eq_de_du_t      = typename Base::eq_de_du_t;
    using stage_linear_t  = typename Base::stage_linear_t;
    using state_box_t     = typename Base::state_box_t;
    using control_box_t   = typename Base::control_box_t;
    using ineq_pen_t      = typename Base::ineq_pen_t;
    using eq_pen_t        = typename Base::eq_pen_t;
    using term_state_box_t    = typename Base::term_state_box_t;
    using ineq_term_t     = typename Base::ineq_term_t;
    using ineq_term_dg_dx_t   = typename Base::ineq_term_dg_dx_t;
    using term_constr_hess_t  = typename Base::term_constr_hess_t;
    using eq_term_t       = typename Base::eq_term_t;
    using eq_term_de_dx_t = typename Base::eq_term_de_dx_t;
    using term_linear_t   = typename Base::term_linear_t;
    using ineq_term_pen_t = typename Base::ineq_term_pen_t;
    using eq_term_pen_t   = typename Base::eq_term_pen_t;
    // sens_p types
    using dyn_df_dp_t     = typename Base::dyn_df_dp_t;
    using dyn_dA_dp_t     = typename Base::dyn_dA_dp_t;
    using dyn_dB_dp_t     = typename Base::dyn_dB_dp_t;
    using param_t         = typename Base::param_t;
    using stage_grad_dp_t = typename Base::stage_grad_dp_t;
    using term_grad_dp_t  = typename Base::term_grad_dp_t;
    using ineq_dg_dp_t    = typename Base::ineq_dg_dp_t;
    using ineq_dg_dx_dp_t = typename Base::ineq_dg_dx_dp_t;
    using ineq_dg_du_dp_t = typename Base::ineq_dg_du_dp_t;
    using eq_de_dp_t      = typename Base::eq_de_dp_t;
    using eq_de_dx_dp_t   = typename Base::eq_de_dx_dp_t;
    using eq_de_du_dp_t   = typename Base::eq_de_du_dp_t;
    using ineq_term_dg_dp_t    = typename Base::ineq_term_dg_dp_t;
    using ineq_term_dg_dx_dp_t = typename Base::ineq_term_dg_dx_dp_t;
    using eq_term_de_dp_t      = typename Base::eq_term_de_dp_t;
    using eq_term_de_dx_dp_t   = typename Base::eq_term_de_dx_dp_t;
    using stage_linear_sensp_t   = typename Base::stage_linear_sensp_t;
    using term_linear_sensp_t    = typename Base::term_linear_sensp_t;
    using stage_state_box_sensp_t   = typename Base::stage_state_box_sensp_t;
    using stage_control_box_sensp_t = typename Base::stage_control_box_sensp_t;
    using term_state_box_sensp_t    = typename Base::term_state_box_sensp_t;
    using x0_dp_t           = typename Base::x0_dp_t;

    // ---------------------------------------------------------------
    // Re-exposed static dimensions and index arrays
    // ---------------------------------------------------------------
    static constexpr int nx = Base::nx;
    static constexpr int nu = Base::nu;
    static constexpr int ng = Base::ng;
    static constexpr int ne = Base::ne;
    static constexpr int nl = Base::nl;
    static constexpr int ng_t = Base::ng_t;
    static constexpr int ne_t = Base::ne_t;
    static constexpr int nl_t = Base::nl_t;
    static constexpr int np = Base::np;
    static constexpr int nbx = Base::nbx;
    static constexpr int nbu = Base::nbu;
    static constexpr int nbx_t = Base::nbx_t;
    static constexpr auto state_box_idx = Base::state_box_idx;
    static constexpr auto control_box_idx = Base::control_box_idx;
    static constexpr auto terminal_state_box_idx = Base::terminal_state_box_idx;

    // ---------------------------------------------------------------
    // Members
    // ---------------------------------------------------------------

    /// Wrapped (unscaled) problem; referenced, not owned.
    Base& base_;

    /// State scaling vector (nx); all entries must be > 0.
    state_t sx_ = state_t::Ones();

    /// Control scaling vector (nu); all entries must be > 0.
    control_t su_ = control_t::Ones();

    // ---------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------

    explicit ScaledProblem(Base& base)
        : base_(base)
    {
    }

    // ---------------------------------------------------------------
    // Solution recovery
    // ---------------------------------------------------------------

    /// Convert a solution in scaled coordinates back to the original
    /// (unscaled) coordinates of the wrapped problem.
    ///
    /// Primal: x = sx_^-1 * x_s, u = su_^-1 * u_s.
    /// Dual: dynamics and box multipliers on state/control co-vary with
    /// the scaled variables and are multiplied back by sx_ / su_ (box
    /// rows in state_box_idx / control_box_idx order, layout [lo; hi]);
    /// multipliers of (in)equality and linear constraints multiply
    /// invariant constraint values and are copied unchanged.  cost_value
    /// and status are copied as-is.
    ///
    /// @tparam NH  horizon (Eigen::Dynamic or compile-time int)
    /// @param  sol_s  solution produced by a solver on this wrapper
    /// @return Solution in the original (unscaled) coordinates
    template <int NH = Eigen::Dynamic>
    Solution<Base, NH> unscale(const Solution<ScaledProblem<Base>, NH>& sol_s) const
    {
        Solution<Base, NH> sol;
        if constexpr (NH == Eigen::Dynamic)
        {
            sol.resize(sol_s.N);
        }

        // primal: x_s -> x  (divide by sx_)
        for (int k = 0; k <= sol_s.N; ++k)
        {
            sol.x[k].array() = sol_s.x[k].array() / sx_.array();
        }
        for (int k = 0; k < sol_s.N; ++k)
        {
            sol.u[k].array() = sol_s.u[k].array() / su_.array();
        }

        // dynamics multipliers: lam_s -> lam  (multiply by sx_)
        for (int k = 0; k < sol_s.N; ++k)
        {
            sol.lambda_dyn[k].array() =
                sol_s.lambda_dyn[k].array() * sx_.array();
        }

        // box multipliers on state: row i -> multiply by sx_[state_box_idx[i]]
        for (int k = 0; k <= sol_s.N; ++k)
        {
            for (int i = 0; i < nbx; ++i)
            {
                const scalar_t s = sx_[state_box_idx[i]];
                sol.lambda_box_state[k](i) = sol_s.lambda_box_state[k](i) * s;
                sol.lambda_box_state[k](nbx + i) =
                    sol_s.lambda_box_state[k](nbx + i) * s;
            }
        }

        // box multipliers on control: row i -> multiply by su_[control_box_idx[i]]
        for (int k = 0; k < sol_s.N; ++k)
        {
            for (int i = 0; i < nbu; ++i)
            {
                const scalar_t s = su_[control_box_idx[i]];
                sol.lambda_box_control[k](i) =
                    sol_s.lambda_box_control[k](i) * s;
                sol.lambda_box_control[k](nbu + i) =
                    sol_s.lambda_box_control[k](nbu + i) * s;
            }
        }

        // (in)equality and linear multipliers: invariant
        for (int k = 0; k < sol_s.N; ++k)
        {
            sol.lambda_ineq_stage[k] = sol_s.lambda_ineq_stage[k];
            sol.lambda_eq_stage[k] = sol_s.lambda_eq_stage[k];
            sol.lambda_lin_stage[k] = sol_s.lambda_lin_stage[k];
        }
        sol.lambda_ineq_term = sol_s.lambda_ineq_term;
        sol.lambda_eq_term = sol_s.lambda_eq_term;
        sol.lambda_lin_term = sol_s.lambda_lin_term;

        sol.cost_value = sol_s.cost_value;
        sol.status = sol_s.status;
        return sol;
    }

    // ===============================================================
    //  initial state
    // ===============================================================

    state_t initial_state() const
    {
        return sx_.cwiseProduct(base_.initial_state());
    }

    // ===============================================================
    //  dynamics  (k = 0..N-1)
    // ===============================================================

    state_t dynamics_next_state(int k, const state_t& x_s,
                                const control_t& u_s) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        return sx_.cwiseProduct(base_.dynamics_next_state(k, x, u));
    }

    void dynamics_value_jac(int k, const state_t& x_s, const control_t& u_s,
                            state_t& x_next,
                            dyn_df_dx_t& df_dx,
                            dyn_df_du_t& df_du) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        state_t x_next_b;
        dyn_df_dx_t df_dx_b;
        dyn_df_du_t df_du_b;
        base_.dynamics_value_jac(k, x, u, x_next_b, df_dx_b, df_du_b);
        x_next = sx_.cwiseProduct(x_next_b);
        // df_dx_s = Sx * df_dx * Sx^-1
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                df_dx(i, j) = df_dx_b(i, j) * sx_(i) / sx_(j);
            }
        }
        // df_du_s = Sx * df_du * Su^-1
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < nu; ++j)
            {
                df_du(i, j) = df_du_b(i, j) * sx_(i) / su_(j);
            }
        }
    }

    void dynamics_value_jac_hess(int k, const state_t& x_s,
                                 const control_t& u_s,
                                 const state_t& lam_s,
                                 state_t& x_next,
                                 dyn_df_dx_t& df_dx,
                                 dyn_df_du_t& df_du,
                                 dyn_hess_t& hess) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        // the base expects the unscaled dynamics multiplier: the
        // Lagrangian term lam_s' (x_next_s - f_s) = (Sx' lam_s)' (x_next - f)
        // gives lam = Sx' lam_s = Sx * lam_s (diagonal)
        const state_t lam = lam_s.cwiseProduct(sx_);
        state_t x_next_b;
        dyn_df_dx_t df_dx_b;
        dyn_df_du_t df_du_b;
        dyn_hess_t hess_b;
        base_.dynamics_value_jac_hess(k, x, u, lam, x_next_b,
                                      df_dx_b, df_du_b, hess_b);
        x_next = sx_.cwiseProduct(x_next_b);
        // Jacobians (same as dynamics_value_jac)
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                df_dx(i, j) = df_dx_b(i, j) * sx_(i) / sx_(j);
            }
        }
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < nu; ++j)
            {
                df_du(i, j) = df_du_b(i, j) * sx_(i) / su_(j);
            }
        }
        // Hessian: H_s = D^{-T} H D^{-1}  (D = blkdiag(Sx, Su))
        for (int i = 0; i < nx + nu; ++i)
        {
            const scalar_t di = (i < nx) ? sx_(i) : su_(i - nx);
            for (int j = 0; j < nx + nu; ++j)
            {
                const scalar_t dj = (j < nx) ? sx_(j) : su_(j - nx);
                hess(i, j) = hess_b(i, j) / (di * dj);
            }
        }
    }

    // ===============================================================
    //  stage cost  (k = 0..N-1)
    // ===============================================================

    scalar_t stage_cost_value(int k, const state_t& x_s,
                              const control_t& u_s) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        return base_.stage_cost_value(k, x, u);
    }

    void stage_cost_value_grad(int k, const state_t& x_s,
                               const control_t& u_s,
                               scalar_t& value, stage_grad_t& grad) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        stage_grad_t grad_b;
        base_.stage_cost_value_grad(k, x, u, value, grad_b);
        // grad_s = D^{-T} grad  ->  element-wise divide by d = [sx; su]
        for (int i = 0; i < nx; ++i)
        {
            grad(i) = grad_b(i) / sx_(i);
        }
        for (int i = 0; i < nu; ++i)
        {
            grad(nx + i) = grad_b(nx + i) / su_(i);
        }
    }

    void stage_cost_value_grad_hess(int k, const state_t& x_s,
                                    const control_t& u_s,
                                    scalar_t& value, stage_grad_t& grad,
                                    stage_hess_t& hess) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        stage_grad_t grad_b;
        stage_hess_t hess_b;
        base_.stage_cost_value_grad_hess(k, x, u, value, grad_b, hess_b);
        // gradient
        for (int i = 0; i < nx; ++i)
        {
            grad(i) = grad_b(i) / sx_(i);
        }
        for (int i = 0; i < nu; ++i)
        {
            grad(nx + i) = grad_b(nx + i) / su_(i);
        }
        // Hessian: H_s = D^{-T} H D^{-1}
        for (int i = 0; i < nx + nu; ++i)
        {
            const scalar_t di = (i < nx) ? sx_(i) : su_(i - nx);
            for (int j = 0; j < nx + nu; ++j)
            {
                const scalar_t dj = (j < nx) ? sx_(j) : su_(j - nx);
                hess(i, j) = hess_b(i, j) / (di * dj);
            }
        }
    }

    // ===============================================================
    //  terminal cost  (stage N, state only)
    // ===============================================================

    scalar_t terminal_cost_value(const state_t& x_s) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        return base_.terminal_cost_value(x);
    }

    void terminal_cost_value_grad(const state_t& x_s, scalar_t& value,
                                  term_grad_t& grad) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        term_grad_t grad_b;
        base_.terminal_cost_value_grad(x, value, grad_b);
        for (int i = 0; i < nx; ++i)
        {
            grad(i) = grad_b(i) / sx_(i);
        }
    }

    void terminal_cost_value_grad_hess(const state_t& x_s, scalar_t& value,
                                       term_grad_t& grad,
                                       term_hess_t& hess) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        term_grad_t grad_b;
        term_hess_t hess_b;
        base_.terminal_cost_value_grad_hess(x, value, grad_b, hess_b);
        for (int i = 0; i < nx; ++i)
        {
            grad(i) = grad_b(i) / sx_(i);
        }
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                hess(i, j) = hess_b(i, j) / (sx_(i) * sx_(j));
            }
        }
    }

    // ===============================================================
    //  stage constraints  (k = 0..N-1)
    // ===============================================================

    ineq_t stage_inequality_value(int k, const state_t& x_s,
                                  const control_t& u_s) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        return base_.stage_inequality_value(k, x, u);
    }

    void stage_inequality_value_jac(int k, const state_t& x_s,
                                    const control_t& u_s,
                                    ineq_t& g,
                                    ineq_dg_dx_t& g_dx,
                                    ineq_dg_du_t& g_du) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        ineq_dg_dx_t g_dx_b;
        ineq_dg_du_t g_du_b;
        base_.stage_inequality_value_jac(k, x, u, g, g_dx_b, g_du_b);
        // g_dx_s = g_dx * Sx^-1  (column-scale by 1/sx)
        for (int i = 0; i < ng; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                g_dx(i, j) = g_dx_b(i, j) / sx_(j);
            }
        }
        // g_du_s = g_du * Su^-1
        for (int i = 0; i < ng; ++i)
        {
            for (int j = 0; j < nu; ++j)
            {
                g_du(i, j) = g_du_b(i, j) / su_(j);
            }
        }
    }

    void stage_inequality_value_jac_hess(int k, const state_t& x_s,
                                         const control_t& u_s,
                                         const ineq_t& lam,
                                         ineq_t& g,
                                         ineq_dg_dx_t& g_dx,
                                         ineq_dg_du_t& g_du,
                                         constr_hess_t& hess) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        ineq_dg_dx_t g_dx_b;
        ineq_dg_du_t g_du_b;
        constr_hess_t hess_b;
        base_.stage_inequality_value_jac_hess(k, x, u, lam, g,
                                              g_dx_b, g_du_b, hess_b);
        for (int i = 0; i < ng; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                g_dx(i, j) = g_dx_b(i, j) / sx_(j);
            }
        }
        for (int i = 0; i < ng; ++i)
        {
            for (int j = 0; j < nu; ++j)
            {
                g_du(i, j) = g_du_b(i, j) / su_(j);
            }
        }
        // Hessian: H_s = D^{-T} H D^{-1}
        for (int i = 0; i < nx + nu; ++i)
        {
            const scalar_t di = (i < nx) ? sx_(i) : su_(i - nx);
            for (int j = 0; j < nx + nu; ++j)
            {
                const scalar_t dj = (j < nx) ? sx_(j) : su_(j - nx);
                hess(i, j) = hess_b(i, j) / (di * dj);
            }
        }
    }

    eq_t stage_equality_value(int k, const state_t& x_s,
                              const control_t& u_s) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        return base_.stage_equality_value(k, x, u);
    }

    void stage_equality_value_jac(int k, const state_t& x_s,
                                  const control_t& u_s,
                                  eq_t& e,
                                  eq_de_dx_t& e_dx,
                                  eq_de_du_t& e_du) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        eq_de_dx_t e_dx_b;
        eq_de_du_t e_du_b;
        base_.stage_equality_value_jac(k, x, u, e, e_dx_b, e_du_b);
        for (int i = 0; i < ne; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                e_dx(i, j) = e_dx_b(i, j) / sx_(j);
            }
        }
        for (int i = 0; i < ne; ++i)
        {
            for (int j = 0; j < nu; ++j)
            {
                e_du(i, j) = e_du_b(i, j) / su_(j);
            }
        }
    }

    void stage_equality_value_jac_hess(int k, const state_t& x_s,
                                       const control_t& u_s,
                                       const eq_t& lam,
                                       eq_t& e,
                                       eq_de_dx_t& e_dx,
                                       eq_de_du_t& e_du,
                                       constr_hess_t& hess) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        eq_de_dx_t e_dx_b;
        eq_de_du_t e_du_b;
        constr_hess_t hess_b;
        base_.stage_equality_value_jac_hess(k, x, u, lam, e,
                                            e_dx_b, e_du_b, hess_b);
        for (int i = 0; i < ne; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                e_dx(i, j) = e_dx_b(i, j) / sx_(j);
            }
        }
        for (int i = 0; i < ne; ++i)
        {
            for (int j = 0; j < nu; ++j)
            {
                e_du(i, j) = e_du_b(i, j) / su_(j);
            }
        }
        for (int i = 0; i < nx + nu; ++i)
        {
            const scalar_t di = (i < nx) ? sx_(i) : su_(i - nx);
            for (int j = 0; j < nx + nu; ++j)
            {
                const scalar_t dj = (j < nx) ? sx_(j) : su_(j - nx);
                hess(i, j) = hess_b(i, j) / (di * dj);
            }
        }
    }

    // ---------------------------------------------------------------
    // linear stage constraint
    // ---------------------------------------------------------------

    stage_linear_t stage_linear_constr(int k) const
    {
        stage_linear_t spec = base_.stage_linear_constr(k);
        // A_s = A * Sx^-1  (column-scale by 1/sx)
        for (int i = 0; i < nl; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                spec.A(i, j) /= sx_(j);
            }
        }
        // B_s = B * Su^-1
        for (int i = 0; i < nl; ++i)
        {
            for (int j = 0; j < nu; ++j)
            {
                spec.B(i, j) /= su_(j);
            }
        }
        // bounds unchanged (the linear form value is invariant)
        return spec;
    }

    // ---------------------------------------------------------------
    // box stage constraints
    // ---------------------------------------------------------------

    state_box_t stage_state_box_constr(int k) const
    {
        state_box_t spec = base_.stage_state_box_constr(k);
        for (int i = 0; i < nbx; ++i)
        {
            const scalar_t s = sx_[state_box_idx[i]];
            spec.lo(i) *= s;
            spec.hi(i) *= s;
            spec.penalty.quad(i) /= s * s;
            spec.penalty.lin(i) /= s;
        }
        return spec;
    }

    control_box_t stage_control_box_constr(int k) const
    {
        control_box_t spec = base_.stage_control_box_constr(k);
        for (int i = 0; i < nbu; ++i)
        {
            const scalar_t s = su_[control_box_idx[i]];
            spec.lo(i) *= s;
            spec.hi(i) *= s;
            spec.penalty.quad(i) /= s * s;
            spec.penalty.lin(i) /= s;
        }
        return spec;
    }

    // ---------------------------------------------------------------
    // soft-penalty pass-throughs (nonlinear: unchanged)
    // ---------------------------------------------------------------

    ineq_pen_t stage_inequality_constr_soft_penalty(int k) const
    {
        return base_.stage_inequality_constr_soft_penalty(k);
    }

    eq_pen_t stage_equality_constr_soft_penalty(int k) const
    {
        return base_.stage_equality_constr_soft_penalty(k);
    }

    // ===============================================================
    //  terminal constraints  (stage N, state only)
    // ===============================================================

    term_state_box_t terminal_state_box_constr() const
    {
        term_state_box_t spec = base_.terminal_state_box_constr();
        for (int i = 0; i < nbx_t; ++i)
        {
            const scalar_t s = sx_[terminal_state_box_idx[i]];
            spec.lo(i) *= s;
            spec.hi(i) *= s;
            spec.penalty.quad(i) /= s * s;
            spec.penalty.lin(i) /= s;
        }
        return spec;
    }

    ineq_term_t terminal_inequality_value(const state_t& x_s) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        return base_.terminal_inequality_value(x);
    }

    void terminal_inequality_value_jac(const state_t& x_s, ineq_term_t& g,
                                       ineq_term_dg_dx_t& g_dx) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        ineq_term_dg_dx_t g_dx_b;
        base_.terminal_inequality_value_jac(x, g, g_dx_b);
        for (int i = 0; i < ng_t; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                g_dx(i, j) = g_dx_b(i, j) / sx_(j);
            }
        }
    }

    void terminal_inequality_value_jac_hess(const state_t& x_s,
                                            const ineq_term_t& lam,
                                            ineq_term_t& g,
                                            ineq_term_dg_dx_t& g_dx,
                                            term_constr_hess_t& hess) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        ineq_term_dg_dx_t g_dx_b;
        term_constr_hess_t hess_b;
        base_.terminal_inequality_value_jac_hess(x, lam, g, g_dx_b, hess_b);
        for (int i = 0; i < ng_t; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                g_dx(i, j) = g_dx_b(i, j) / sx_(j);
            }
        }
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                hess(i, j) = hess_b(i, j) / (sx_(i) * sx_(j));
            }
        }
    }

    eq_term_t terminal_equality_value(const state_t& x_s) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        return base_.terminal_equality_value(x);
    }

    void terminal_equality_value_jac(const state_t& x_s, eq_term_t& e,
                                     eq_term_de_dx_t& e_dx) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        eq_term_de_dx_t e_dx_b;
        base_.terminal_equality_value_jac(x, e, e_dx_b);
        for (int i = 0; i < ne_t; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                e_dx(i, j) = e_dx_b(i, j) / sx_(j);
            }
        }
    }

    void terminal_equality_value_jac_hess(const state_t& x_s,
                                          const eq_term_t& lam,
                                          eq_term_t& e,
                                          eq_term_de_dx_t& e_dx,
                                          term_constr_hess_t& hess) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        eq_term_de_dx_t e_dx_b;
        term_constr_hess_t hess_b;
        base_.terminal_equality_value_jac_hess(x, lam, e, e_dx_b, hess_b);
        for (int i = 0; i < ne_t; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                e_dx(i, j) = e_dx_b(i, j) / sx_(j);
            }
        }
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                hess(i, j) = hess_b(i, j) / (sx_(i) * sx_(j));
            }
        }
    }

    term_linear_t terminal_linear_constr() const
    {
        term_linear_t spec = base_.terminal_linear_constr();
        for (int i = 0; i < nl_t; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                spec.A(i, j) /= sx_(j);
            }
        }
        // bounds unchanged
        return spec;
    }

    ineq_term_pen_t terminal_inequality_constr_soft_penalty() const
    {
        return base_.terminal_inequality_constr_soft_penalty();
    }

    eq_term_pen_t terminal_equality_constr_soft_penalty() const
    {
        return base_.terminal_equality_constr_soft_penalty();
    }

    // ===============================================================
    //  parameter-vector sensitivity  (np > 0)
    // ===============================================================

    void dynamics_sens_p(int k, const state_t& x_s, const control_t& u_s,
                         dyn_df_dp_t& f_p,
                         dyn_dA_dp_t& A_p,
                         dyn_dB_dp_t& B_p) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        base_.dynamics_sens_p(k, x, u, f_p, A_p, B_p);
        // f_p_s = Sx * f_p  (row-scale by sx)
        for (int i = 0; i < nx; ++i)
        {
            for (int p = 0; p < np; ++p)
            {
                f_p(i, p) *= sx_(i);
            }
        }
        // A_p_s: element (i, j) of each column scaled by sx(i)/sx(j)
        for (int p = 0; p < np; ++p)
        {
            for (int i = 0; i < nx; ++i)
            {
                for (int j = 0; j < nx; ++j)
                {
                    A_p(i * nx + j, p) *= sx_(i) / sx_(j);
                }
            }
        }
        // B_p_s: element (i, j) scaled by sx(i)/su(j)
        for (int p = 0; p < np; ++p)
        {
            for (int i = 0; i < nx; ++i)
            {
                for (int j = 0; j < nu; ++j)
                {
                    B_p(i * nu + j, p) *= sx_(i) / su_(j);
                }
            }
        }
    }

    void stage_cost_sens_p(int k, const state_t& x_s, const control_t& u_s,
                           param_t& L_p, stage_grad_dp_t& grad_p) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        base_.stage_cost_sens_p(k, x, u, L_p, grad_p);
        // L_p unchanged (cost value is invariant)
        // grad_p: row i scaled by 1/d(i)
        for (int i = 0; i < nx; ++i)
        {
            for (int p = 0; p < np; ++p)
            {
                grad_p(i, p) /= sx_(i);
            }
        }
        for (int i = 0; i < nu; ++i)
        {
            for (int p = 0; p < np; ++p)
            {
                grad_p(nx + i, p) /= su_(i);
            }
        }
    }

    void terminal_cost_sens_p(const state_t& x_s,
                              param_t& L_p, term_grad_dp_t& grad_p) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        base_.terminal_cost_sens_p(x, L_p, grad_p);
        // L_p unchanged; grad_p row i scaled by 1/sx(i)
        for (int i = 0; i < nx; ++i)
        {
            for (int p = 0; p < np; ++p)
            {
                grad_p(i, p) /= sx_(i);
            }
        }
    }

    void stage_inequality_sens_p(int k, const state_t& x_s,
                                 const control_t& u_s,
                                 ineq_dg_dp_t& g_p,
                                 ineq_dg_dx_dp_t& gdx_p,
                                 ineq_dg_du_dp_t& gdu_p) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        base_.stage_inequality_sens_p(k, x, u, g_p, gdx_p, gdu_p);
        // g_p unchanged (value invariant)
        // gdx_p: element (r * nx + i, p) scaled by 1/sx(i)
        for (int p = 0; p < np; ++p)
        {
            for (int r = 0; r < ng; ++r)
            {
                for (int i = 0; i < nx; ++i)
                {
                    gdx_p(r * nx + i, p) /= sx_(i);
                }
            }
        }
        for (int p = 0; p < np; ++p)
        {
            for (int r = 0; r < ng; ++r)
            {
                for (int i = 0; i < nu; ++i)
                {
                    gdu_p(r * nu + i, p) /= su_(i);
                }
            }
        }
    }

    void stage_equality_sens_p(int k, const state_t& x_s,
                               const control_t& u_s,
                               eq_de_dp_t& e_p,
                               eq_de_dx_dp_t& edx_p,
                               eq_de_du_dp_t& edu_p) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        const control_t u = u_s.cwiseQuotient(su_);
        base_.stage_equality_sens_p(k, x, u, e_p, edx_p, edu_p);
        for (int p = 0; p < np; ++p)
        {
            for (int r = 0; r < ne; ++r)
            {
                for (int i = 0; i < nx; ++i)
                {
                    edx_p(r * nx + i, p) /= sx_(i);
                }
            }
        }
        for (int p = 0; p < np; ++p)
        {
            for (int r = 0; r < ne; ++r)
            {
                for (int i = 0; i < nu; ++i)
                {
                    edu_p(r * nu + i, p) /= su_(i);
                }
            }
        }
    }

    void terminal_inequality_sens_p(const state_t& x_s,
                                    ineq_term_dg_dp_t& g_p,
                                    ineq_term_dg_dx_dp_t& gdx_p) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        base_.terminal_inequality_sens_p(x, g_p, gdx_p);
        for (int p = 0; p < np; ++p)
        {
            for (int r = 0; r < ng_t; ++r)
            {
                for (int i = 0; i < nx; ++i)
                {
                    gdx_p(r * nx + i, p) /= sx_(i);
                }
            }
        }
    }

    void terminal_equality_sens_p(const state_t& x_s,
                                  eq_term_de_dp_t& e_p,
                                  eq_term_de_dx_dp_t& edx_p) const
    {
        const state_t x = x_s.cwiseQuotient(sx_);
        base_.terminal_equality_sens_p(x, e_p, edx_p);
        for (int p = 0; p < np; ++p)
        {
            for (int r = 0; r < ne_t; ++r)
            {
                for (int i = 0; i < nx; ++i)
                {
                    edx_p(r * nx + i, p) /= sx_(i);
                }
            }
        }
    }

    stage_linear_sensp_t stage_linear_sens_p(int k) const
    {
        stage_linear_sensp_t sens = base_.stage_linear_sens_p(k);
        // dA_dp: element (r * nx + i, p) scaled by 1/sx(i)
        for (int p = 0; p < np; ++p)
        {
            for (int r = 0; r < nl; ++r)
            {
                for (int i = 0; i < nx; ++i)
                {
                    sens.dA_dp(r * nx + i, p) /= sx_(i);
                }
            }
        }
        // dB_dp: element (r * nu + i, p) scaled by 1/su(i)
        for (int p = 0; p < np; ++p)
        {
            for (int r = 0; r < nl; ++r)
            {
                for (int i = 0; i < nu; ++i)
                {
                    sens.dB_dp(r * nu + i, p) /= su_(i);
                }
            }
        }
        // dlo_dp, dhi_dp unchanged (bounds are invariant)
        return sens;
    }

    term_linear_sensp_t terminal_linear_sens_p() const
    {
        term_linear_sensp_t sens = base_.terminal_linear_sens_p();
        for (int p = 0; p < np; ++p)
        {
            for (int r = 0; r < nl_t; ++r)
            {
                for (int i = 0; i < nx; ++i)
                {
                    sens.dA_dp(r * nx + i, p) /= sx_(i);
                }
            }
        }
        return sens;
    }

    stage_state_box_sensp_t stage_state_box_sens_p(int k) const
    {
        stage_state_box_sensp_t sens = base_.stage_state_box_sens_p(k);
        for (int i = 0; i < nbx; ++i)
        {
            const scalar_t s = sx_[state_box_idx[i]];
            for (int p = 0; p < np; ++p)
            {
                sens.dlo_dp(i, p) *= s;
                sens.dhi_dp(i, p) *= s;
            }
        }
        return sens;
    }

    stage_control_box_sensp_t stage_control_box_sens_p(int k) const
    {
        stage_control_box_sensp_t sens = base_.stage_control_box_sens_p(k);
        for (int i = 0; i < nbu; ++i)
        {
            const scalar_t s = su_[control_box_idx[i]];
            for (int p = 0; p < np; ++p)
            {
                sens.dlo_dp(i, p) *= s;
                sens.dhi_dp(i, p) *= s;
            }
        }
        return sens;
    }

    term_state_box_sensp_t terminal_state_box_sens_p() const
    {
        term_state_box_sensp_t sens = base_.terminal_state_box_sens_p();
        for (int i = 0; i < nbx_t; ++i)
        {
            const scalar_t s = sx_[terminal_state_box_idx[i]];
            for (int p = 0; p < np; ++p)
            {
                sens.dlo_dp(i, p) *= s;
                sens.dhi_dp(i, p) *= s;
            }
        }
        return sens;
    }

    x0_dp_t initial_state_sens_p() const
    {
        x0_dp_t x0p = base_.initial_state_sens_p();
        // x0_s = Sx * x0  ->  dx0_s/dp = Sx * dx0/dp
        for (int i = 0; i < nx; ++i)
        {
            for (int p = 0; p < np; ++p)
            {
                x0p(i, p) *= sx_(i);
            }
        }
        return x0p;
    }

    // ===============================================================
    //  shared parameters (forwarded to the wrapped problem)
    // ===============================================================

    /// Shared model parameters: the wrapper exposes exactly the
    /// parameter object of the wrapped problem, so host-side writes
    /// through the wrapper reach the same live object the wrapped
    /// model reads.
    auto params() -> decltype(base_.params())
    {
        return base_.params();
    }

    /// Const view of the wrapped problem's parameters (the trailing
    /// return type names the wrapped problem's *const* overload, since
    /// the reference member `base_` is unaffected by `this` constness).
    auto params() const
        -> decltype(std::declval<const Base&>().params())
    {
        return base_.params();
    }
};

}  // namespace ocp

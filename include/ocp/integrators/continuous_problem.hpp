// ocp/integrators/continuous_problem.hpp
//
// ContinuousProblem: adapter that turns a continuous-time ODE + a RK
// integrator into a discrete-time ocp::Problem.
//
// The user defines a concrete problem by inheriting from
// ContinuousProblem and implementing the cost / constraint / initial-state
// methods. The fused dynamics entries (dynamics_next_state,
// dynamics_value_jac, dynamics_value_jac_hess) are provided here and delegate
// to the integrator. The ODE may be time-varying: the adapter maps the stage
// index to a physical time t_k = k * h and forwards it to the integrator
// (the ODE itself is time-invariant if it simply ignores the t argument).
//
// The ODE model is stored by value; the integrator holds a const Ode&
// pointing into this object. The integrator's mutable workspace makes its
// methods const, so the dynamics_* methods are const as required by the
// Problem interface.
//
// dynamics_value_jac delegates to the integrator's fused value_jac entry
// (value + Jacobian in one pass). dynamics_value_jac_hess assembles the
// contracted (multiplier-weighted) Hessian from nIn = nx + nu unit-vector
// HVP columns: when Integ::supports_value_jac_hess_prod (AD ODE on the
// explicit integrator), each column is one nested-dual forward pass that
// carries the value and the Jacobian for free; otherwise the integrator's
// value_jac populates its forward cache and the per-column hess_prod calls
// hit that cache.
//
// has_dynamics_hess: when true (nonlinear ODE), the solver calls
// dynamics_value_jac_hess (the integrator requires the ODE to provide
// hess_prod on the non-AD path). When false (linear ODE), the solver never
// calls dynamics_value_jac_hess and the method is never instantiated — the
// ODE may omit hess_prod entirely.

#pragma once

#include "ocp/integrators/rk_explicit.hpp"
#include "ocp/problem.hpp"

namespace ocp
{

template <class Dims, class Ode, class Integ, int NH = Eigen::Dynamic>
class ContinuousProblem : public Problem<Dims>
{
public:
    using P = Problem<Dims>;
    using state_t = typename P::state_t;
    using control_t = typename P::control_t;
    using dyn_df_dx_t = typename P::dyn_df_dx_t;
    using dyn_df_du_t = typename P::dyn_df_du_t;

    /// @param ode  ODE model (stored by value).
    /// @param args forwarded to the integrator constructor (typically
    ///             the step size h and possibly scheme options).
    template <class... Args>
    explicit ContinuousProblem(const Ode& ode, double h, Args&&... args)
        : ode_(ode), h_(h), integ_(ode_, h, std::forward<Args>(args)...)
    {
    }

    // ---------------------------------------------------------------
    //  dynamics: delegate to the integrator, forwarding t_k = k * h_
    // ---------------------------------------------------------------

    state_t dynamics_next_state(int k, const state_t& x,
                                const control_t& u) const
    {
        state_t x_next;
        integ_.value(x, u, k * h_, x_next);
        return x_next;
    }

    void dynamics_value_jac(int k, const state_t& x, const control_t& u,
                            state_t& x_next, dyn_df_dx_t& df_dx,
                            dyn_df_du_t& df_du) const
    {
        integ_.value_jac(x, u, k * h_, x_next, df_dx, df_du);
    }

    void dynamics_value_jac_hess(int k, const state_t& x, const control_t& u,
                                 const state_t& w, state_t& x_next,
                                 dyn_df_dx_t& df_dx, dyn_df_du_t& df_du,
                                 typename P::dyn_hess_t& hess) const
    {
        // Contracted Hessian (w^T D2 Phi): nIn unit-vector HVP columns in
        // [x; u]-layout.
        if constexpr (Integ::supports_value_jac_hess_prod)
        {
            // AD ODE: each column is one nested-dual forward pass; the value
            // and the Jacobian come out of the same pass and are identical
            // for every column.
            for (int j = 0; j < P::nx + P::nu; ++j)
            {
                state_t v_x = state_t::Zero();
                control_t v_u = control_t::Zero();
                if (j < P::nx)
                {
                    v_x(j) = 1.0;
                }
                else
                {
                    v_u(j - P::nx) = 1.0;
                }
                state_t hv_x;
                control_t hv_u;
                integ_.value_jac_hess_prod(x, u, k * h_, x_next, df_dx,
                                           df_du, w, v_x, v_u, hv_x, hv_u);
                hess.col(j).head(P::nx) = hv_x;
                hess.col(j).tail(P::nu) = hv_u;
            }
        }
        else
        {
            // value_jac populates the integrator's (x, u, t_k)-keyed forward
            // cache; the per-column hess_prod calls below hit it.
            integ_.value_jac(x, u, k * h_, x_next, df_dx, df_du);
            for (int j = 0; j < P::nx + P::nu; ++j)
            {
                state_t v_x = state_t::Zero();
                control_t v_u = control_t::Zero();
                if (j < P::nx)
                {
                    v_x(j) = 1.0;
                }
                else
                {
                    v_u(j - P::nx) = 1.0;
                }
                state_t hv_x;
                control_t hv_u;
                integ_.hess_prod(x, u, k * h_, w, v_x, v_u, hv_x, hv_u);
                hess.col(j).head(P::nx) = hv_x;
                hess.col(j).tail(P::nu) = hv_u;
            }
        }
    }

private:
    Ode ode_;
    double h_;
    Integ integ_;
};

}  // namespace ocp

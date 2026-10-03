// ocp/integrators/continuous_problem.hpp
//
// ContinuousProblem: adapter that turns a continuous-time ODE + a RK
// integrator into a discrete-time ocp::Problem.
//
// The user defines a concrete problem by inheriting from
// ContinuousProblem and implementing the cost / constraint / initial-state
// methods. The three dynamics methods (dynamics_next_state,
// dynamics_jacobian, dynamics_hess_prod) are provided here and delegate to
// the integrator; the stage index k is ignored because the ODE is
// time-invariant this phase.
//
// The ODE model is stored by value; the integrator holds a const Ode&
// pointing into this object. The integrator's mutable workspace makes its
// methods const, so the dynamics_* methods are const as required by the
// Problem interface.
//
// has_dynamics_hess_prod: when true (nonlinear ODE), the solver calls
// dynamics_hess_prod, which delegates to the integrator's hess_prod (the
// integrator static_asserts that the ODE provides hess_prod). When false
// (linear ODE), the solver never calls dynamics_hess_prod and the method
// is never instantiated — the ODE may omit hess_prod entirely.

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
    explicit ContinuousProblem(const Ode& ode, Args&&... args)
        : ode_(ode), integ_(ode_, std::forward<Args>(args)...)
    {
    }

    // ---------------------------------------------------------------
    //  dynamics: delegate to the integrator (k ignored, time-invariant)
    // ---------------------------------------------------------------

    state_t dynamics_next_state(int /*k*/, const state_t& x,
                                const control_t& u) const
    {
        state_t x_next;
        integ_.value(x, u, x_next);
        return x_next;
    }

    void dynamics_jacobian(int /*k*/, const state_t& x, const control_t& u,
                           dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
    {
        integ_.jacobian(x, u, df_dx, df_du);
    }

    void dynamics_hess_prod(int /*k*/, const state_t& x, const control_t& u,
                            const state_t& w, const state_t& v_x,
                            const control_t& v_u, state_t& hv_x,
                            control_t& hv_u) const
    {
        integ_.hess_prod(x, u, w, v_x, v_u, hv_x, hv_u);
    }

private:
    Ode ode_;
    Integ integ_;
};

}  // namespace ocp

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
// Per-stage step size: the interval width h_k and start time t_k of each
// stage come from the CRTP hook get_step_size(k) (returns a StepSpec). The
// default (Derived = ConstantStepSize) falls back to the uniform grid
// t_k = k * h with a constant h. To use a non-uniform grid, a problem names
// its own type as the Derived template argument and defines
// `StepSpec get_step_size(int k) const`, typically reading a per-stage
// schedule from the duck-typed params() object. The hook is consumed once
// per stage (never per sub-step); the integrator's sub-step layout is
// internal. See "Per-stage step size" in the tutorial (ch. 06).
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

#include <type_traits>

#include "ocp/integrators/rk_explicit.hpp"
#include "ocp/problem.hpp"

namespace ocp
{

/// Per-stage integration step: the OCP-interval width `h` and the physical
/// start time `t` of a stage. Supplied by the problem when a non-uniform
/// time grid is used; ContinuousProblem falls back to the uniform grid
/// `t = k * h` (constant `h` from its constructor) otherwise.
struct StepSpec
{
    double h;
    double t;
};

/// Sentinel default for the `Derived` template argument of
/// ContinuousProblem: the problem does not override the step-size hook, so
/// the constant-`h` fallback grid is used.
struct ConstantStepSize
{
};

template <class Dims, class Ode, class Integ, class Derived = ConstantStepSize,
          int NH = Eigen::Dynamic>
class ContinuousProblem : public Problem<Dims>
{
public:
    using P = Problem<Dims>;
    using state_t = typename P::state_t;
    using control_t = typename P::control_t;
    using dyn_df_dx_t = typename P::dyn_df_dx_t;
    using dyn_df_du_t = typename P::dyn_df_du_t;

    /// @param ode  ODE model (stored by value).
    /// @param h    constant step size of the OCP interval (fallback grid,
    ///             used unless `Derived` overrides get_step_size).
    /// @param args forwarded to the integrator constructor (e.g. the
    ///             Newton options of the implicit integrator).
    template <class... Args>
    explicit ContinuousProblem(const Ode& ode, double h, Args&&... args)
        : ode_(ode), h_(h), integ_(ode_, std::forward<Args>(args)...)
    {
    }

    /// The ODE/DAE model stored by this adapter (by value). Exposed so
    /// user-defined problems can forward model state (e.g. a shared
    /// `params()` object) through their own interface.
    Ode& ode()
    {
        return ode_;
    }

    const Ode& ode() const
    {
        return ode_;
    }

    // ---------------------------------------------------------------
    //  step-size hook (CRTP, opt-in via the Derived template argument)
    // ---------------------------------------------------------------

    /// Per-stage step-size hook. Returns the OCP-interval width `h` and the
    /// physical start time `t` of stage `k`.
    ///
    /// Default (Derived = ConstantStepSize): the uniform grid
    /// {h, k * h} from the constructor's `h`.
    ///
    /// Opt-in: a problem names its own type as `Derived` and defines
    /// `StepSpec get_step_size(int k) const` (typically reading a per-stage
    /// schedule from its duck-typed params() object). The base then
    /// dispatches to the override; forgetting to define the hook is caught
    /// by a static_assert below (it would otherwise recurse into this
    /// method). The hook is called once per stage, never per sub-step.
    StepSpec get_step_size(int k) const
    {
        if constexpr (std::is_same_v<Derived, ConstantStepSize>)
        {
            return StepSpec{h_, (double)k * h_};
        }
        else
        {
            static_assert(
                std::is_same_v<decltype(&Derived::get_step_size),
                               StepSpec (Derived::*)(int) const>,
                "ContinuousProblem: when the problem names itself as the "
                "Derived template argument, it must define "
                "'StepSpec get_step_size(int k) const' (currently the "
                "inherited constant-step hook would be called instead).");
            return static_cast<const Derived*>(this)->get_step_size(k);
        }
    }

    // ---------------------------------------------------------------
    //  dynamics: delegate to the integrator, forwarding the per-stage
    //  (t_k, h_k) from get_step_size(k)
    // ---------------------------------------------------------------

    /// One integration step of the composed map  x_{k+1} = Phi(x, u, t_k)
    /// with  t_k = k * h.
    ///
    /// @param k stage index, 0..N-1.
    /// @param x state at stage k (nx).
    /// @param u control at stage k (nu).
    /// @return state at stage k+1 (nx).
    state_t dynamics_next_state(int k, const state_t& x,
                                const control_t& u) const
    {
        const StepSpec spec = get_step_size(k);
        state_t x_next;
        integ_.value(x, u, spec.t, spec.h, x_next);
        return x_next;
    }

    /// x_{k+1} plus the Jacobian blocks of the composed map at
    /// (x, u, t_k):  df_dx (nx x nx), df_du (nx x nu).
    ///
    /// @param k stage index, 0..N-1.
    /// @param x state at stage k (nx).
    /// @param u control at stage k (nu).
    /// @param[out] x_next state at stage k+1 (nx).
    /// @param[out] df_dx  d(Phi)/dx at (x, u, t_k) (nx x nx).
    /// @param[out] df_du  d(Phi)/du at (x, u, t_k) (nx x nu).
    void dynamics_value_jac(int k, const state_t& x, const control_t& u,
                            state_t& x_next, dyn_df_dx_t& df_dx,
                            dyn_df_du_t& df_du) const
    {
        const StepSpec spec = get_step_size(k);
        integ_.value_jac(x, u, spec.t, spec.h, x_next, df_dx, df_du);
    }

    /// x_{k+1}, the Jacobian blocks, and the contracted (multiplier-
    /// weighted) Hessian  w^T D^2 Phi  of the composed map at
    /// (x, u, t_k), assembled from nIn = nx + nu unit-vector HVP columns.
    ///
    /// @param k stage index, 0..N-1.
    /// @param x state at stage k (nx).
    /// @param u control at stage k (nu).
    /// @param w state-space multiplier weighting the Hessian (nx).
    /// @param[out] x_next state at stage k+1 (nx).
    /// @param[out] df_dx  d(Phi)/dx at (x, u, t_k) (nx x nx).
    /// @param[out] df_du  d(Phi)/du at (x, u, t_k) (nx x nu).
    /// @param[out] hess contracted Hessian (nx + nu x nx + nu), column j is
    ///                  the HVP in the j-th unit direction of [x; u].
    void dynamics_value_jac_hess(int k, const state_t& x, const control_t& u,
                                 const state_t& w, state_t& x_next,
                                 dyn_df_dx_t& df_dx, dyn_df_du_t& df_du,
                                 typename P::dyn_hess_t& hess) const
    {
        const StepSpec spec = get_step_size(k);
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
                integ_.value_jac_hess_prod(x, u, spec.t, spec.h, x_next,
                                           df_dx, df_du, w, v_x, v_u, hv_x,
                                           hv_u);
                hess.col(j).head(P::nx) = hv_x;
                hess.col(j).tail(P::nu) = hv_u;
            }
        }
        else
        {
            // value_jac populates the integrator's (x, u, t_k, h)-keyed
            // forward cache; the per-column hess_prod calls below hit it.
            integ_.value_jac(x, u, spec.t, spec.h, x_next, df_dx, df_du);
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
                integ_.hess_prod(x, u, spec.t, spec.h, w, v_x, v_u, hv_x,
                                 hv_u);
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

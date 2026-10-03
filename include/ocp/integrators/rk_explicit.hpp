// ocp/integrators/rk_explicit.hpp
//
// Allocation-free explicit Runge-Kutta integrator (phases 4b, 4g, 4h).
//
// ExplicitRkIntegrator turns a (possibly time-varying) ODE
//  xdot = f(x, u, t)  into the discrete dynamics
//  x_next = Phi(x, u, t_k)  via a fixed explicit RK scheme
// (ButcherTableau<Dims, NS, Tag>). It implements the ODE-derivative contract
// (value / jacobian / hess_prod, see ode_model.hpp) so that it can be wrapped
// by ContinuousProblem into an ocp::Problem. Nothing downstream of the
// problem interface (QP, SQP, ...) changes.
//
// Multi-step (phase 4g): the OCP interval of size h is subdivided into
// NumSteps sub-intervals of size h_ss = h/NumSteps. `value` chains the
// single-step RK sweep over the sub-steps; `jacobian` threads the first-order
// sensitivity forward across sub-steps; `hess_prod` runs one continuous
// backward adjoint sweep across all sub-steps. NumSteps = 1 (the default) is
// the plain single-step integrator and is bit-for-bit identical to 4b.
//
// Time-varying (phase 4h): each method receives `t_k` (the start of the OCP
// interval). The per-stage physical time is
//     t_s = t_k + h_ss * (ss + c[s])
// and is forwarded to every ODE evaluation (f, jacobian, hess_prod).
//
// Allocation-free: every scratch buffer is a fixed-size std::array of
// compile-time Eigen objects, sized once on NS, NumSteps and Dims::nx/nu and
// reused across calls. The methods are const; the ODE model is held by const
// reference; the per-call scratch is `mutable`.
//
// All three methods recompute the stage states from (x, u, t_k) on each call
// (no cached value is assumed), matching how the SQP invokes dynamics_* on
// its own.

#pragma once

#include <array>

#include "ocp/integrators/butcher.hpp"
#include "ocp/integrators/ode_model.hpp"

namespace ocp
{

namespace detail
{

/// Pre-allocated scratch for one explicit RK step over NumSteps sub-steps.
/// All arrays are indexed [ss][s] with ss in 0..NumSteps-1 (sub-step) and
/// s in 0..NS-1 (stage).
template <class P, int NS, int NumSteps>
struct ErkWorkspace
{
    using state_t = typename P::state_t;
    using control_t = typename P::control_t;
    using df_dx_t = typename P::dyn_df_dx_t;
    using df_du_t = typename P::dyn_df_du_t;

    using State2 = std::array<std::array<state_t, NS>, NumSteps>;
    using Dfdx2  = std::array<std::array<df_dx_t, NS>, NumSteps>;
    using Dfdu2  = std::array<std::array<df_du_t, NS>, NumSteps>;

    State2 K{};          // stage derivatives K_s (forward)
    State2 x_stage{};    // stage states x_s (forward + backward)
    Dfdx2  JX{};         // df/dx at stage s (forward + backward)
    Dfdu2  JU{};         // df/du at stage s (forward + backward)
    State2 dx{};         // first-order JVP of the stage state
    State2 dK{};         // first-order JVP of K_s (forward)

    // Backward HVP duals (explicit-dual reverse mode, phase 4g)
    State2 ddK_dual{};        // dual on the JVP node dK[ss][s]
    State2 dKval_dual{};      // dual on the value node K[ss][s]
    std::array<state_t, NumSteps + 1> dv_ss{};  // dual on v_ss (JVP boundary)
    std::array<state_t, NumSteps + 1> dx_ss{};  // dual on x_ss (value boundary)
    control_t du{};           // dual on the control u
};

}  // namespace detail

/// Explicit Runge-Kutta integrator for a time-invariant ODE.
///
/// `NS` is the number of RK stages (1..4 for the explicit tags). `Tag`
/// selects the Butcher scheme (K1Tag..K4Tag). `NumSteps` (phase 4g) is the
/// number of sub-intervals one OCP step of size `h` is divided into; the
/// default of 1 is the plain single-step scheme. The ODE model must provide
/// `f` and `jacobian`; `hess_prod` is required only for a nonlinear ODE and
/// is enforced by `ode_supports_hess_prod` at the `hess_prod` call site.
template <class Dims, class Ode, int NS, class Tag, int NumSteps = 1>
class ExplicitRkIntegrator
{
    using P = ocp::Problem<Dims>;
    using Tab = ocp::ButcherTableau<Dims, NS, Tag>;

public:
    using state_t = typename P::state_t;
    using control_t = typename P::control_t;
    using df_dx_t = typename P::dyn_df_dx_t;
    using df_du_t = typename P::dyn_df_du_t;

    /// @param ode the ODE model (held by const reference; must outlive this).
    /// @param h fixed step size for the full OCP interval (h/NumSteps per
    ///          sub-step).
    explicit ExplicitRkIntegrator(const Ode& ode, double h)
        : ode_(ode), h_(h), h_ss_(h / (double)NumSteps)
    {
    }

    /// Discrete map  x_next = Phi(x, u, t_k)  for the full OCP interval of
    /// size h, advanced in NumSteps sub-steps of size h/NumSteps.
    /// `t_k` is the physical time at the start of the OCP interval.
    void value(const state_t& x, const control_t& u, double t_k,
               state_t& x_next) const
    {
        state_t x_cur = x;
        for (int ss = 0; ss < NumSteps; ++ss)
        {
            const double t_0 = t_k + ss * h_ss_;
            ws_.K[ss][0] = ode_.f(x_cur, u, t_0 + h_ss_ * Tab::c[0]);
            for (int s = 1; s < NS; ++s)
            {
                state_t xs = x_cur;
                for (int j = 0; j < s; ++j)
                {
                    if (Tab::A[s][j] != 0.0)
                    {
                        xs += h_ss_ * Tab::A[s][j] * ws_.K[ss][j];
                    }
                }
                ws_.K[ss][s] = ode_.f(xs, u, t_0 + h_ss_ * Tab::c[s]);
            }
            for (int s = 0; s < NS; ++s)
            {
                if (Tab::b[s] != 0.0)
                {
                    x_cur += h_ss_ * Tab::b[s] * ws_.K[ss][s];
                }
            }
        }
        x_next = x_cur;
    }

    /// Jacobian  (df_dx, df_du)  of the composed map  x_next = Phi(x, u),
    /// computed by threading the first-order sensitivity forward across the
    /// sub-steps: start Sx = I, Su = 0; per sub-step form the per-sub-step
    /// Jacobian (Jx_ss, Ju_ss) and update Sx = Jx_ss*Sx, Su = Jx_ss*Su + Ju_ss.
    void jacobian(const state_t& x, const control_t& u, double t_k,
                  df_dx_t& df_dx, df_du_t& df_du) const
    {
        state_t x_cur = x;
        df_dx_t Sx = df_dx_t::Identity();
        df_du_t Su;
        Su.setZero();
        for (int ss = 0; ss < NumSteps; ++ss)
        {
            const double t_0 = t_k + ss * h_ss_;
            ws_.x_stage[ss][0] = x_cur;
            ws_.K[ss][0] = ode_.f(x_cur, u, t_0 + h_ss_ * Tab::c[0]);
            for (int s = 1; s < NS; ++s)
            {
                state_t xs = x_cur;
                for (int j = 0; j < s; ++j)
                {
                    if (Tab::A[s][j] != 0.0)
                    {
                        xs += h_ss_ * Tab::A[s][j] * ws_.K[ss][j];
                    }
                }
                ws_.x_stage[ss][s] = xs;
                ws_.K[ss][s] = ode_.f(xs, u, t_0 + h_ss_ * Tab::c[s]);
            }
            // Per-sub-step Jacobian (Jx_ss, Ju_ss): thread the stage
            // sensitivities, then accumulate the b-weighted stage Jacobians.
            df_dx_t Jx_ss = df_dx_t::Identity();
            df_du_t Ju_ss;
            Ju_ss.setZero();
            for (int s = 0; s < NS; ++s)
            {
                // d(x_s)/d(x,u):  Dxs = I + h_ss sum_{j<s} A[s][j] JX[j],
                //                Dus =     h_ss sum_{j<s} A[s][j] JU[j]
                df_dx_t Dxs = df_dx_t::Identity();
                df_du_t Dus;
                Dus.setZero();
                for (int j = 0; j < s; ++j)
                {
                    if (Tab::A[s][j] != 0.0)
                    {
                        Dxs += h_ss_ * Tab::A[s][j] * ws_.JX[ss][j];
                        Dus += h_ss_ * Tab::A[s][j] * ws_.JU[ss][j];
                    }
                }
                df_dx_t dfx_s;
                df_du_t dfdu_s;
                ode_.jacobian(ws_.x_stage[ss][s], u, t_0 + h_ss_ * Tab::c[s],
                              dfx_s, dfdu_s);
                ws_.JX[ss][s] = dfx_s * Dxs;
                ws_.JU[ss][s] = dfx_s * Dus + dfdu_s;
            }
            for (int s = 0; s < NS; ++s)
            {
                if (Tab::b[s] != 0.0)
                {
                    Jx_ss += h_ss_ * Tab::b[s] * ws_.JX[ss][s];
                    Ju_ss += h_ss_ * Tab::b[s] * ws_.JU[ss][s];
                    x_cur += h_ss_ * Tab::b[s] * ws_.K[ss][s];
                }
            }
            Sx = Jx_ss * Sx;
            Su = Jx_ss * Su + Ju_ss;
        }
        df_dx = Sx;
        df_du = Su;
    }

    /// Bilinear Hessian-vector product of the composed map, output-contracted
    /// with `w` (state; e.g. the dynamics multiplier) and input direction
    /// `v = (v_x, v_u)`; returns (hv_x, hv_u).  A single forward pass stores
    /// the per-sub-step stage values, Jacobians and first-order JVP; then an
    /// explicit-dual reverse-mode sweep propagates duals over every node of
    /// the JVP + value computation graph.  The ODE's `hess_prod` is called
    /// once per stage to inject the second-derivative terms.
    /// Only valid for a nonlinear ODE (the ODE must provide `hess_prod`).
    void hess_prod(const state_t& x, const control_t& u, double t_k,
                   const state_t& w, const state_t& v_x, const control_t& v_u,
                   state_t& hv_x, control_t& hv_u) const
    {
        static_assert(ode_supports_hess_prod<Ode, Dims>,
            "ExplicitRkIntegrator::hess_prod: the ODE must provide a "
            "'hess_prod' method (nonlinear ODE). A linear ODE has a zero "
            "Hessian and should not reach this path.");
        // Forward: per sub-step, stage states, K, Jacobians, and the
        // first-order JVP along v (dx_s, dK_s). x_cur carries the state;
        // v_ss carries the JVP of the sub-step input state.
        state_t x_cur = x;
        state_t v_ss = v_x;
        for (int ss = 0; ss < NumSteps; ++ss)
        {
            const double t_0 = t_k + ss * h_ss_;
            ws_.x_stage[ss][0] = x_cur;
            ws_.K[ss][0] = ode_.f(x_cur, u, t_0 + h_ss_ * Tab::c[0]);
            for (int s = 1; s < NS; ++s)
            {
                state_t xs = x_cur;
                for (int j = 0; j < s; ++j)
                {
                    if (Tab::A[s][j] != 0.0)
                    {
                        xs += h_ss_ * Tab::A[s][j] * ws_.K[ss][j];
                    }
                }
                ws_.x_stage[ss][s] = xs;
                ws_.K[ss][s] = ode_.f(xs, u, t_0 + h_ss_ * Tab::c[s]);
            }
            for (int s = 0; s < NS; ++s)
            {
                ode_.jacobian(ws_.x_stage[ss][s], u, t_0 + h_ss_ * Tab::c[s],
                              ws_.JX[ss][s], ws_.JU[ss][s]);
            }
            ws_.dx[ss][0] = v_ss;
            ws_.dK[ss][0] = ws_.JX[ss][0] * v_ss + ws_.JU[ss][0] * v_u;
            for (int s = 1; s < NS; ++s)
            {
                state_t dxs = v_ss;
                for (int j = 0; j < s; ++j)
                {
                    if (Tab::A[s][j] != 0.0)
                    {
                        dxs += h_ss_ * Tab::A[s][j] * ws_.dK[ss][j];
                    }
                }
                ws_.dx[ss][s] = dxs;
                ws_.dK[ss][s] = ws_.JX[ss][s] * dxs + ws_.JU[ss][s] * v_u;
            }
            for (int s = 0; s < NS; ++s)
            {
                if (Tab::b[s] != 0.0)
                {
                    x_cur += h_ss_ * Tab::b[s] * ws_.K[ss][s];
                    v_ss += h_ss_ * Tab::b[s] * ws_.dK[ss][s];
                }
            }
        }

        // Backward: explicit-dual reverse mode.  The scalar
        //   s = w^T * v_next   (v_next = DPhi * v)
        // is differentiated w.r.t. (x, u); the gradient is the HVP
        // (hv_x, hv_u).  Duals are carried on every node (value and JVP).
        //
        // Seed:  dual[v_NumSteps] = w;  all other duals = 0.
        // Answer: hv_x = dual[x_0] = dx_ss[0],  hv_u = du.
        for (int ss = 0; ss < NumSteps; ++ss)
        {
            for (int s = 0; s < NS; ++s)
            {
                ws_.ddK_dual[ss][s].setZero();
                ws_.dKval_dual[ss][s].setZero();
            }
            ws_.dv_ss[ss].setZero();
            ws_.dx_ss[ss].setZero();
        }
        ws_.dv_ss[NumSteps] = w;
        ws_.dx_ss[NumSteps].setZero();
        ws_.du.setZero();

        for (int ss = NumSteps - 1; ss >= 0; --ss)
        {
            const double t_0 = t_k + ss * h_ss_;
            const state_t& dv_next = ws_.dv_ss[ss + 1];
            const state_t& dx_next = ws_.dx_ss[ss + 1];

            // b-update: v_{ss+1} = v_ss + h_ss * sum b_s dK_s
            //           x_{ss+1} = x_ss + h_ss * sum b_s K_s
            ws_.dv_ss[ss] += dv_next;
            ws_.dx_ss[ss] += dx_next;
            for (int s = 0; s < NS; ++s)
            {
                if (Tab::b[s] != 0.0)
                {
                    ws_.ddK_dual[ss][s] += (h_ss_ * Tab::b[s]) * dv_next;
                    ws_.dKval_dual[ss][s] += (h_ss_ * Tab::b[s]) * dx_next;
                }
            }

            for (int s = NS - 1; s >= 0; --s)
            {
                // (1) K_s = f(x_s, u):  dual[K] -> dual[x_s], dual[u]
                state_t dx_stage_local =
                    ws_.JX[ss][s].transpose() * ws_.dKval_dual[ss][s];

                // (2) Hessian: JX/JU depend on (x_s, u); inject ODE HVP.
                state_t hx;
                control_t hu;
                ode_hess_prod<Ode, Dims>(ode_, ws_.x_stage[ss][s], u,
                                         t_0 + h_ss_ * Tab::c[s],
                                         ws_.ddK_dual[ss][s], ws_.dx[ss][s],
                                         v_u, hx, hu);
                dx_stage_local += hx;
                const control_t du_stage =
                    ws_.JU[ss][s].transpose() * ws_.dKval_dual[ss][s] + hu;
                ws_.du += du_stage;

                // (3) x_s = x_ss + h_ss sum_{j<s} A[s][j] K_j:
                //     dual[x_s] -> dual[x_ss], dual[K_j]
                ws_.dx_ss[ss] += dx_stage_local;
                for (int j = 0; j < s; ++j)
                {
                    if (Tab::A[s][j] != 0.0)
                    {
                        ws_.dKval_dual[ss][j] +=
                            h_ss_ * Tab::A[s][j] * dx_stage_local;
                    }
                }

                // (4) dK_s = JX_s dx_s + JU_s v_u:  dual[dK] -> dual[dx_s]
                const state_t ddx_local =
                    ws_.JX[ss][s].transpose() * ws_.ddK_dual[ss][s];

                // (5) dx_s = v_ss + h_ss sum_{j<s} A[s][j] dK_j:
                //     dual[dx_s] -> dual[v_ss], dual[dK_j]
                ws_.dv_ss[ss] += ddx_local;
                for (int j = 0; j < s; ++j)
                {
                    if (Tab::A[s][j] != 0.0)
                    {
                        ws_.ddK_dual[ss][j] +=
                            h_ss_ * Tab::A[s][j] * ddx_local;
                    }
                }
            }
        }
        hv_x = ws_.dx_ss[0];
        hv_u = ws_.du;
    }

private:
    const Ode& ode_;
    double h_;
    double h_ss_;
    mutable detail::ErkWorkspace<P, NS, NumSteps> ws_;
};

}  // namespace ocp

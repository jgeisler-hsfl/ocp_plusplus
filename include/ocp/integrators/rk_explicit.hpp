// ocp/integrators/rk_explicit.hpp
//
// Allocation-free explicit Runge-Kutta integrator (phase 4b).
//
// ExplicitRkIntegrator turns a time-invariant ODE  xdot = f(x, u)  into the
// discrete dynamics  x_next = Phi(x, u)  via a fixed explicit RK scheme
// (ButcherTableau<Dims, NS, Tag>). It implements the ODE-derivative contract
// (value / jacobian / hess_prod, see ode_model.hpp) so that it can be wrapped
// by ContinuousProblem into an ocp::Problem. Nothing downstream of the
// problem interface (QP, SQP, ...) changes.
//
// Allocation-free: every scratch buffer is a fixed-size std::array of
// compile-time Eigen objects, sized once on NS and Dims::nx/nu and reused
// across calls. The methods are const; the ODE model is held by const
// reference; the per-call scratch is `mutable`.
//
// All three methods recompute the stage states from (x, u) on each call (no
// cached value is assumed), matching how the SQP invokes dynamics_* on its
// own. The stage index is not a parameter: the ODE is time-invariant, so
// every OCP stage uses the same f (ContinuousProblem ignores its `k`).

#pragma once

#include <array>

#include "ocp/integrators/butcher.hpp"
#include "ocp/integrators/ode_model.hpp"

namespace ocp
{

namespace detail
{

/// Pre-allocated scratch for one explicit RK step (NS stages; sizes from P).
template <class P, int NS>
struct ErkWorkspace
{
    using state_t = typename P::state_t;
    using control_t = typename P::control_t;
    using df_dx_t = typename P::dyn_df_dx_t;
    using df_du_t = typename P::dyn_df_du_t;

    std::array<state_t, NS> K{};      // stage derivatives K_s
    std::array<state_t, NS> x_stage{};  // stage states x_s
    std::array<df_dx_t, NS> JX{};     // df/dx at stage s
    std::array<df_du_t, NS> JU{};     // df/du at stage s
    std::array<state_t, NS> dx{};     // first-order JVP of the stage state
    std::array<state_t, NS> dK{};     // first-order JVP of K_s
    std::array<state_t, NS> lam_dK{}; // backward adjoint on dK_s
    std::array<state_t, NS> lam_K{};  // backward adjoint on the K_s value
};

}  // namespace detail

/// Explicit Runge-Kutta integrator for a time-invariant ODE.
///
/// `NS` is the number of RK stages (1..4 for the explicit tags). `Tag`
/// selects the Butcher scheme (K1Tag..K4Tag). `h` is the fixed step size.
/// The ODE model must provide `f` and `jacobian`; `hess_prod` is required
/// only for a nonlinear ODE and is enforced by `ode_supports_hess_prod` at
/// the `hess_prod` call site.
template <class Dims, class Ode, int NS, class Tag>
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
    /// @param h fixed step size.
    explicit ExplicitRkIntegrator(const Ode& ode, double h)
        : ode_(ode), h_(h)
    {
    }

    /// Discrete map  x_next = Phi(x, u)  for a single step.
    void value(const state_t& x, const control_t& u, state_t& x_next) const
    {
        compute_stage_states(x, u);
        x_next = x;
        for (int s = 0; s < NS; ++s)
        {
            if (Tab::b[s] != 0.0)
            {
                x_next += h_ * Tab::b[s] * ws_.K[s];
            }
        }
    }

    /// Jacobian  (df_dx, df_du)  of the composed map  x_next = Phi(x, u),
    /// computed by threading first-order sensitivities forward over the stages.
    void jacobian(const state_t& x, const control_t& u, df_dx_t& df_dx,
                  df_du_t& df_du) const
    {
        compute_stage_states(x, u);
        for (int s = 0; s < NS; ++s)
        {
            // d(x_s)/d(x,u) along the direction:  Dxs = I + h sum_{j<s} A[s][j] JX_j,
            //                                      Dus =     h sum_{j<s} A[s][j] JU_j
            df_dx_t Dxs = df_dx_t::Identity();
            df_du_t Dus;
            Dus.setZero();
            for (int j = 0; j < s; ++j)
            {
                if (Tab::A[s][j] != 0.0)
                {
                    Dxs += h_ * Tab::A[s][j] * ws_.JX[j];
                    Dus += h_ * Tab::A[s][j] * ws_.JU[j];
                }
            }
            df_dx_t dfx_s;
            df_du_t dfdu_s;
            ode_.jacobian(ws_.x_stage[s], u, dfx_s, dfdu_s);
            ws_.JX[s] = dfx_s * Dxs;
            ws_.JU[s] = dfx_s * Dus + dfdu_s;
        }
        df_dx = df_dx_t::Identity();
        df_du.setZero();
        for (int s = 0; s < NS; ++s)
        {
            if (Tab::b[s] != 0.0)
            {
                df_dx += h_ * Tab::b[s] * ws_.JX[s];
                df_du += h_ * Tab::b[s] * ws_.JU[s];
            }
        }
    }

    /// Bilinear Hessian-vector product of the composed map, output-contracted
    /// with `w` (state; e.g. the dynamics multiplier) and input direction
    /// `v = (v_x, v_u)`; returns (hv_x, hv_u). Computed by a reverse-over-
    /// forward sweep: a forward pass for the first-order JVP along `v`, then a
    /// backward adjoint pass that calls the ODE's `hess_prod` once per stage.
    /// Only valid for a nonlinear ODE (the ODE must provide `hess_prod`).
    void hess_prod(const state_t& x, const control_t& u, const state_t& w,
                   const state_t& v_x, const control_t& v_u,
                   state_t& hv_x, control_t& hv_u) const
    {
        static_assert(ode_supports_hess_prod<Ode, Dims>,
            "ExplicitRkIntegrator::hess_prod: the ODE must provide a "
            "'hess_prod' method (nonlinear ODE). A linear ODE has a zero "
            "Hessian and should not reach this path.");

        compute_stage_states(x, u);
        // Forward: Jacobians at every stage + first-order JVP along v.
        for (int s = 0; s < NS; ++s)
        {
            ode_.jacobian(ws_.x_stage[s], u, ws_.JX[s], ws_.JU[s]);
        }
        for (int s = 0; s < NS; ++s)
        {
            state_t dxs = v_x;
            for (int j = 0; j < s; ++j)
            {
                if (Tab::A[s][j] != 0.0)
                {
                    dxs += h_ * Tab::A[s][j] * ws_.dK[j];
                }
            }
            ws_.dx[s] = dxs;
            ws_.dK[s] = ws_.JX[s] * dxs + ws_.JU[s] * v_u;
        }
        // Backward: seed the adjoints (dK_s multiplier from the output weights
        // h*b_s*w; the K_s-value multiplier starts at zero) and walk s downward.
        hv_x.setZero();
        hv_u.setZero();
        for (int s = 0; s < NS; ++s)
        {
            ws_.lam_dK[s] = (h_ * Tab::b[s]) * w;
            ws_.lam_K[s].setZero();
        }
        for (int s = NS - 1; s >= 0; --s)
        {
            const state_t& beta = ws_.lam_dK[s];
            state_t mu_x;
            control_t nu_u;
            ode_hess_prod<Ode, Dims>(ode_, ws_.x_stage[s], u, beta, ws_.dx[s],
                                     v_u, mu_x, nu_u);
            const state_t kappa = ws_.JX[s].transpose() * beta;      // on dx_s
            const state_t theta = ws_.JX[s].transpose() * ws_.lam_K[s];
            const control_t rho = ws_.JU[s].transpose() * ws_.lam_K[s];
            const state_t xs_adj = mu_x + theta;   // total stage-state adjoint
            hv_x += xs_adj;
            hv_u += nu_u + rho;
            for (int j = 0; j < s; ++j)
            {
                if (Tab::A[s][j] != 0.0)
                {
                    ws_.lam_K[j] += h_ * Tab::A[s][j] * xs_adj;
                    ws_.lam_dK[j] += h_ * Tab::A[s][j] * kappa;
                }
            }
        }
    }

private:
    /// Fill ws_.x_stage[s] and ws_.K[s] for s = 0..NS-1 (the ERK forward sweep).
    void compute_stage_states(const state_t& x, const control_t& u) const
    {
        ws_.x_stage[0] = x;
        ws_.K[0] = ode_.f(x, u);
        for (int s = 1; s < NS; ++s)
        {
            state_t xs = x;
            for (int j = 0; j < s; ++j)
            {
                if (Tab::A[s][j] != 0.0)
                {
                    xs += h_ * Tab::A[s][j] * ws_.K[j];
                }
            }
            ws_.x_stage[s] = xs;
            ws_.K[s] = ode_.f(xs, u);
        }
    }

    const Ode& ode_;
    double h_;
    mutable detail::ErkWorkspace<P, NS> ws_;
};

}  // namespace ocp

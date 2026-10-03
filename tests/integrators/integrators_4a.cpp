// Phase 4a: verify Butcher tableau identities and the ode_supports_hess_prod
// trait.

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/integrators/butcher.hpp"
#include "ocp/integrators/ode_model.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

namespace
{

using Dims = DoubleIntegratorDims;  // nx = 2, nu = 1; provides the type aliases.

using P = ocp::Problem<Dims>;
using state_t = P::state_t;
using control_t = P::control_t;

int failures = 0;

void check(bool cond, const char* msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

// Common identity checks: row-sum == c, sum(b) == 1.
template <int NS, class Dims, class Tag>
void check_common(const char* name)
{
    using T = ocp::ButcherTableau<Dims, NS, Tag>;

    for (int s = 0; s < NS; ++s)
    {
        double sum = 0.0;
        for (int j = 0; j < NS; ++j)
        {
            sum += T::A[s][j];
        }
        check(std::abs(sum - T::c[s]) < 1e-12,
              (std::string(name) + " row " + std::to_string(s) + " sum == c").c_str());
    }

    double bsum = 0.0;
    for (int s = 0; s < NS; ++s)
    {
        bsum += T::b[s];
    }
    check(std::abs(bsum - 1.0) < 1e-12,
          (std::string(name) + " sum(b) == 1").c_str());

    std::printf("%s (NS=%d): row-sum==c, sum(b)==1  OK\n", name, NS);
}

// Radau IIA-specific: last row == b, and collocation order conditions
// sum_j b_j c_j^k == 1/(k+1) for k = 0..2*NS-2.
template <int NS, class Dims, class Tag>
void check_irk(const char* name)
{
    using T = ocp::ButcherTableau<Dims, NS, Tag>;

    // last row of A equals b
    for (int j = 0; j < NS; ++j)
    {
        check(std::abs(T::A[NS - 1][j] - T::b[j]) < 1e-14,
              (std::string(name) + " last row == b[" + std::to_string(j) + "]").c_str());
    }

    // collocation order conditions
    for (int k = 0; k <= 2 * NS - 2; ++k)
    {
        double s = 0.0;
        for (int j = 0; j < NS; ++j)
        {
            s += T::b[j] * std::pow(T::c[j], k);
        }
        const double expected = 1.0 / (k + 1);
        check(std::abs(s - expected) < 1e-12,
              (std::string(name) + " order cond k=" + std::to_string(k)).c_str());
    }

    std::printf("%s (NS=%d): last-row==b, order conditions  OK\n", name, NS);
}

// ---- ODE trait fixtures -------------------------------------------------

/// A nonlinear ODE that has hess_prod.
struct NonlinearOde
{
    state_t f(const state_t& x, const control_t&) const
    {
        return x.array().sin().matrix();
    }
    void jacobian(const state_t& x, const control_t&,
                  P::dyn_df_dx_t& df_dx, P::dyn_df_du_t& df_du) const
    {
        df_dx = x.array().cos().matrix().asDiagonal();
        df_du.setZero();
    }
    void hess_prod(const state_t&, const control_t&, const state_t& w,
                   const state_t& v_x, const control_t&,
                   state_t& hv_x, control_t& hv_u) const
    {
        hv_x = -w.cwiseProduct(v_x);
        hv_u.setZero();
    }
};

/// A linear ODE that omits hess_prod.
struct LinearOde
{
    state_t f(const state_t& x, const control_t& u) const
    {
        state_t r;
        r << x(1), u(0);
        return r;
    }
    void jacobian(const state_t&, const control_t&,
                  P::dyn_df_dx_t& df_dx, P::dyn_df_du_t& df_du) const
    {
        df_dx.setZero();
        df_dx(0, 1) = 1.0;
        df_du(1, 0) = 1.0;
    }
};

}  // namespace

int run_integrators_4a_tests()
{
    // Explicit RK tableaus
    check_common<1, Dims, ocp::K1Tag>("K1");
    check_common<2, Dims, ocp::K2Tag>("K2");
    check_common<3, Dims, ocp::K3Tag>("K3");
    check_common<4, Dims, ocp::K4Tag>("K4");

    // Radau IIA tableaus
    check_common<2, Dims, ocp::RadauIia2Tag>("RadauIia2");
    check_irk<2, Dims, ocp::RadauIia2Tag>("RadauIia2");
    check_common<3, Dims, ocp::RadauIia3Tag>("RadauIia3");
    check_irk<3, Dims, ocp::RadauIia3Tag>("RadauIia3");
    check_common<4, Dims, ocp::RadauIia4Tag>("RadauIia4");
    check_irk<4, Dims, ocp::RadauIia4Tag>("RadauIia4");

    // Trait: positive and negative cases
    static_assert(ocp::ode_supports_hess_prod<NonlinearOde, Dims>,
                  "NonlinearOde must expose hess_prod");
    static_assert(!ocp::ode_supports_hess_prod<LinearOde, Dims>,
                  "LinearOde must not expose hess_prod");
    std::printf("ode_supports_hess_prod trait  OK\n");

    if (failures != 0)
    {
        std::fprintf(stderr, "[4a] %d check(s) failed.\n", failures);
    }
    return failures;
}

// Duck-typed params() hook unit test
// (docs/plans/finished/DUCK_TYPED_PARAMS_PLAN.md, block 5):
//   (a) read params by member and by name (scalar and matrix leaves),
//   (b) mutate a param by name through inspect::set_* and confirm a model
//       call reflects the change,
//   (c) a problem without a params() override reports zero parameter
//       names and kUnknownPath on access.

#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "ocp/problem.hpp"
#include "ocp/scaling.hpp"
#include "ocp/utils/inspect.hpp"

#include "double_integrator/double_integrator.hpp"

namespace
{

int failures = 0;

void check(bool cond, const char* msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

void check_close(const char* msg, double a, double b, double tol)
{
    if (std::abs(a - b) > tol)
    {
        std::fprintf(stderr, "FAIL: %s (%g vs %g)\n", msg, a, b);
        ++failures;
    }
}

// Minimal problem without a params() override: exercises the inherited
// EmptyParams stubs. Only initial_state() is needed; no other interface
// function is called in this test.
struct BareDims
{
    static constexpr int nx = 2, nu = 1;
    static constexpr int ng = 0, ne = 0, nl = 0;
    static constexpr int ng_t = 0, ne_t = 1, nl_t = 0;
    static constexpr bool fixed_initial_state = true;
    static constexpr bool has_dynamics_hess = false;
    static constexpr bool has_constr_hess = false;
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

struct BareProblem : ocp::Problem<BareDims>
{
    ocp::Problem<BareDims>::state_t initial_state() const
    {
        ocp::Problem<BareDims>::state_t x0;
        x0 << 1.0, 0.5;
        return x0;
    }
};

// ------------------------------------------------------------------
// (a) reading params: by member and by name
// ------------------------------------------------------------------
void test_reading()
{
    DoubleIntegrator p;

    // by member
    check_close("Ts member", p.params().Ts, 0.1, 0.0);
    check_close("w_stage member", p.params().w_stage(0), 1.0, 0.0);

    // scalar leaf by name
    double ts = 0.0;
    check(ocp::inspect::get_double(p.params(), "Ts", ts), "get Ts by name");
    check_close("Ts value", ts, 0.1, 0.0);

    // matrix leaf by name: extent + values
    int rows = 0;
    int cols = 0;
    check(ocp::inspect::matrix_extent(p.params(), "w_stage", rows, cols),
          "extent of w_stage");
    check(rows == 3 && cols == 1, "w_stage extent 3x1");

    Eigen::Matrix<double, 3, 1> w;
    check(ocp::inspect::get_matrix(p.params(), "w_stage", w),
          "get w_stage by name");
    check_close("w_stage(0)", w(0), 1.0, 0.0);
    check_close("w_stage(1)", w(1), 0.1, 0.0);
    check_close("w_stage(2)", w(2), 0.01, 0.0);

    // name discovery: all four leaves, in declaration order
    const std::vector<std::string> names = ocp::inspect::leaf_names<DiParams>();
    check(names.size() == 4, "leaf_names size 4");
    check(names[0] == "Ts" && names[1] == "v_max" && names[2] == "w_stage" &&
          names[3] == "wf",
          "leaf names in declaration order");
}

// ------------------------------------------------------------------
// (b) mutation by name reflected in model calls
// ------------------------------------------------------------------
void test_mutation()
{
    DoubleIntegrator p;

    // Ts feeds the dynamics
    check(ocp::inspect::set_double(p.params(), "Ts", 0.25).ok(),
          "set Ts by name");
    DoubleIntegrator::state_t x;
    x << 1.0, 1.0;
    DoubleIntegrator::control_t u;
    u << 1.0;
    const auto xn = p.dynamics_next_state(0, x, u);
    check_close("dynamics uses new Ts (q)", xn(0), 1.25, 0.0);
    check_close("dynamics uses new Ts (v)", xn(1), 1.25, 0.0);

    // v_max feeds the velocity-cap constraint
    check(ocp::inspect::set_double(p.params(), "v_max", 3.0).ok(),
          "set v_max by name");
    const auto g = p.stage_inequality_value(0, x, u);
    check_close("inequality uses new v_max", g(0), -2.0, 0.0);

    // w_stage feeds the stage cost
    const Eigen::Matrix<double, 3, 1> w2(2.0, 4.0, 8.0);
    check(ocp::inspect::set_matrix(p.params(), "w_stage", w2),
          "set w_stage by name");
    double L = 0.0;
    DoubleIntegrator::stage_grad_t grad;
    p.stage_cost_value_grad(0, x, u, L, grad);
    check_close("stage cost uses new weights", L, 7.0, 1e-12);
    check_close("stage grad uses new weights", grad(2), 8.0, 0.0);

    // out-of-bounds write is rejected and leaves the leaf untouched
    const auto bad = ocp::inspect::set_double(p.params(), "Ts", 0.0);
    check(bad.status == ocp::inspect::SetResult::Status::kOutOfRange,
          "Ts = 0 rejected (strict positivity bound)");
    double ts = -1.0;
    check(ocp::inspect::get_double(p.params(), "Ts", ts) && ts == 0.25,
          "Ts unchanged after rejected write");

    // wrong matrix element count is rejected, leaf untouched
    const double wrong[2] = {1.0, 1.0};
    check(!ocp::inspect::set_matrix_double(p.params(), "w_stage", wrong, 2),
          "set w_stage with wrong element count rejected");
    Eigen::Matrix<double, 3, 1> w;
    check(ocp::inspect::get_matrix(p.params(), "w_stage", w), "re-read w_stage");
    check_close("w_stage(0) after rejected write", w(0), 2.0, 0.0);

    // unknown names
    double dummy = 0.0;
    check(!ocp::inspect::get_double(p.params(), "nope", dummy),
          "unknown scalar leaf reports false");
    check(
        ocp::inspect::set_double(p.params(), "nope", 1.0).status ==
            ocp::inspect::SetResult::Status::kUnknownPath,
        "unknown scalar leaf reports kUnknownPath");
}

// ------------------------------------------------------------------
// (c) empty stub: no params() override
// ------------------------------------------------------------------
void test_empty_stub()
{
    static_assert(!ocp::detail::has_params<BareProblem>::value,
                  "base stub must read as 'no params'");
    static_assert(ocp::detail::has_params<DoubleIntegrator>::value,
                  "user override must read as 'has params'");

    BareProblem p;
    auto& params = p.params();
    static_assert(std::is_same_v<decltype(params), ocp::EmptyParams&>,
                  "stub returns EmptyParams&");

    check(ocp::inspect::leaf_names<ocp::EmptyParams>().empty(),
          "empty stub exposes zero parameter names");
    double v = 0.0;
    check(!ocp::inspect::get_double(params, "Ts", v),
          "stub read reports unknown");
    check(ocp::inspect::set_double(params, "Ts", 1.0).status ==
              ocp::inspect::SetResult::Status::kUnknownPath,
          "stub write reports kUnknownPath");

    const BareProblem cp;
    const auto& cparams = cp.params();
    static_assert(std::is_same_v<decltype(cparams), const ocp::EmptyParams&>,
                  "const stub returns const EmptyParams&");
    check(!ocp::inspect::get_double(cparams, "Ts", v),
          "const stub read reports unknown");
}

// ------------------------------------------------------------------
// ScaledProblem forwards to the wrapped problem's params object
// ------------------------------------------------------------------
void test_scaled_forwarding()
{
    DoubleIntegrator p;
    ocp::ScaledProblem<DoubleIntegrator> scaled(p);

    static_assert(
        ocp::detail::has_params<ocp::ScaledProblem<DoubleIntegrator>>::value,
        "wrapper must read as 'has params'");

    check(&scaled.params() == &p.params(),
          "wrapper exposes the wrapped problem's live params object");

    p.params().Ts = 0.5;
    double ts = 0.0;
    check(ocp::inspect::get_double(scaled.params(), "Ts", ts) && ts == 0.5,
          "read through wrapper");

    // writes through the wrapper reach the wrapped problem
    check(ocp::inspect::set_double(scaled.params(), "v_max", 5.0).ok(),
          "write through wrapper");
    check_close("wrapped object updated", p.params().v_max, 5.0, 0.0);

    const ocp::ScaledProblem<DoubleIntegrator>& cs = scaled;
    double v = 0.0;
    check(ocp::inspect::get_double(cs.params(), "v_max", v) && v == 5.0,
          "const read through wrapper");
}

}  // namespace

int main()
{
    test_reading();
    test_mutation();
    test_empty_stub();
    test_scaled_forwarding();
    if (failures == 0)
    {
        std::puts("params_unit: all checks passed");
        return 0;
    }
    std::fprintf(stderr, "params_unit: %d failure(s)\n", failures);
    return 1;
}

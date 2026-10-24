// MexSession unit test (docs/plans/MATLAB_MEX_PLAN.md, step 2): the
// host-independent core of the MATLAB MEX wrapper, exercised without
// MATLAB on the DoubleIntegrator. Checks: info extents, option discovery /
// get / set (flat and dotted, int/bool coercion, live apply_options path),
// warm-start field set/get with extent validation, a full solve, solution
// read-back and the statistics rows.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ocp/mex/mex_api.hpp"

#include "double_integrator/double_integrator.hpp"

namespace
{

using namespace ocp;
using Problem = DoubleIntegrator;
using D = QpDim<Problem>;
using Session = ocp::mex::MexSession<Problem>;

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

void check_close(double a, double b, double tol, const std::string& msg)
{
    if (std::fabs(a - b) > tol)
    {
        std::fprintf(stderr, "FAIL: %s (%.15e vs %.15e)\n", msg.c_str(), a, b);
        ++failures;
    }
}

constexpr int N = 10;

// Constant control that makes the terminal equality q_N + v_N = 1 hold
// exactly (same construction as tests/sqp_double_integrator).
double terminal_eq_feasible_control(const Problem& p)
{
    const double Ts = p.Ts_;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * N * (N - 1) / 2.0 + Ts * N;
    return (1.0 - (q0 + v0) - Ts * N * v0) / den;
}

// Forward-simulate a feasible warm start and push it through field_set.
void warm_start(Session& s)
{
    const Problem p;
    const double a = terminal_eq_feasible_control(p);

    std::vector<double> x(static_cast<std::size_t>(D::nx) * (N + 1), 0.0);
    std::vector<double> u(static_cast<std::size_t>(D::nu) * N, a);
    // column-major: column k = stage k
    for (int i = 0; i < D::nx; ++i)
    {
        x[static_cast<std::size_t>(i)] = p.initial_state()(i);
    }
    for (int k = 0; k < N; ++k)
    {
        Problem::state_t xk;
        Problem::control_t uk;
        for (int i = 0; i < D::nx; ++i)
        {
            xk(i) = x[static_cast<std::size_t>(k) * D::nx + i];
        }
        uk << a;
        const Problem::state_t xnext = p.dynamics_next_state(k, xk, uk);
        for (int i = 0; i < D::nx; ++i)
        {
            x[static_cast<std::size_t>(k + 1) * D::nx + i] = xnext(i);
        }
    }
    check(s.field_set("x", x, D::nx, N + 1), "field_set x");
    check(s.field_set("u", u, D::nu, N), "field_set u");
}

void test_info()
{
    Session s("mex_api_test");
    const auto info = s.info();
    check(info.nx == D::nx && info.nu == D::nu, "info nx/nu");
    check(info.ng == D::ng && info.ne == D::ne && info.nl == D::nl,
          "info stage constraint dims");
    check(info.ng_t == D::ng_t && info.ne_t == D::ne_t && info.nl_t == D::nl_t,
          "info terminal constraint dims");
    check(info.nbx == D::nbx && info.nbu == D::nbu && info.nbx_t == D::nbx_t,
          "info box counts");
    check(info.fixed_initial_state == Problem::fixed_initial_state,
          "info fixed_initial_state");
    check(info.N == 0, "N == 0 before init");
    check(s.name() == "mex_api_test", "configuration name");
}

void test_options()
{
    Session s("mex_api_test");

    // discovery: 33 flat fields + 5 glob.* leaves
    const std::vector<std::string> names = s.option_names();
    check(names.size() == 38, "38 discoverable option names");
    bool has_max_iter = false;
    bool has_glob_alpha_min = false;
    for (const std::string& n : names)
    {
        has_max_iter = has_max_iter || (n == "max_iter");
        has_glob_alpha_min = has_glob_alpha_min || (n == "glob.alpha_min");
    }
    check(has_max_iter && has_glob_alpha_min,
          "names include max_iter and glob.alpha_min");

    // reads (int/bool coerced to double)
    double v = 0.0;
    check(s.option_get("max_iter", v) && v == 20.0, "get max_iter");
    check(s.option_get("glob.alpha_min", v) && v == 0.05,
          "get glob.alpha_min");
    check(s.option_get("print_level", v) && v == 0.0,
          "int option read as double");
    check(s.option_get("compute_hess_dyn", v) && v == 1.0,
          "bool option read as 1.0");
    check(!s.option_get("i_max", v), "unknown option name");

    // writes: int, bool and dotted names
    check(s.option_set("max_iter", 40.0), "set max_iter");
    check(s.options().max_iter == 40, "set wrote the int member");
    check(s.option_get("max_iter", v) && v == 40.0, "get after set");
    check(s.option_set("compute_hess_dyn", 0.0), "set bool from 0.0");
    check(!s.options().compute_hess_dyn, "bool member written");
    check(s.option_set("glob.alpha_reduction", 0.9), "set dotted name");
    check_close(s.options().glob.alpha_reduction, 0.9, 1e-15,
                "dotted set wrote the glob member");
    check(!s.option_set("nope", 1.0), "set unknown name");
}

void test_live_options()
{
    Session s("mex_api_test");
    check(s.init(N), "init");
    warm_start(s);

    // max_iter = 1: the iteration cap is hit before the tolerances can be
    // met (check_termination checks the cap first), so the solve stops at
    // kMaxIterations; raising the cap then converges from the warm iterate.
    double cost = 0.0;
    check(s.option_set("max_iter", 1.0), "set max_iter = 1");
    const Status st1 = s.solve(cost);
    check(st1 == Status::kMaxIterations, "max_iter = 1 stops at the cap");
    check(s.stat_iter() >= 1, "at least one iteration recorded");

    check(s.option_set("max_iter", 20.0), "set max_iter = 20");
    const Status st2 = s.solve(cost);
    check(st2 == Status::kSolved, "max_iter = 20 converges");
}

void test_fields()
{
    Session s("mex_api_test");
    std::vector<double> x;
    int rows = 0, cols = 0;
    check(!s.field_set("x", x, D::nx, N + 1), "set x before init rejected");
    check(s.init(N), "init");

    // init sets x[0] from the fixed initial state, the rest stays zero
    check(s.field_get("x", x, rows, cols), "get x");
    check(rows == D::nx && cols == N + 1, "x extents");
    check_close(x[0], 1.0, 1e-12, "x[0] q = initial_state");
    check_close(x[static_cast<std::size_t>(1)], 0.5, 1e-12,
                "x[0] v = initial_state");
    check(s.field_get("u", x, rows, cols) && rows == D::nu && cols == N,
          "u extents");

    // wrong extents / read-only fields are rejected
    check(!s.field_set("x", std::vector<double>(2 * N, 0.0), D::nx, N),
          "x with wrong cols rejected");
    check(!s.field_set("u", std::vector<double>(2 * N, 0.0), 2, N),
          "u with wrong rows rejected");
    check(!s.field_set("lambda_dyn", std::vector<double>(2 * N, 0.0), 2, N),
          "multiplier field is not writable");

    // warm-start read-back
    warm_start(s);
    const Problem p;
    const double a = terminal_eq_feasible_control(p);
    check(s.field_get("u", x, rows, cols), "get u after warm start");
    for (int k = 0; k < N; ++k)
    {
        check_close(x[static_cast<std::size_t>(k)], a, 0.0,
                    "u read-back column " + std::to_string(k));
    }
    check(s.field_get("x", x, rows, cols), "get x after warm start");
    // column N: q_N + v_N = 1 holds exactly by construction of the control
    check_close(x[static_cast<std::size_t>(N) * D::nx]
                    + x[static_cast<std::size_t>(N) * D::nx + 1],
                1.0, 1e-10, "x read-back terminal equality");
}

void test_solve_and_readback()
{
    Session s("mex_api_test");
    double dummy = 0.0;
    check(s.solve(dummy) == Status::kAborted, "solve before init is rejected");
    check(s.init(N), "init");
    warm_start(s);

    double cost = 0.0;
    const Status st = s.solve(cost);
    check(st == Status::kSolved, "solve returns kSolved");
    check(s.solution().status == Status::kSolved, "sol.status kSolved");
    check_close(cost, s.solution().cost_value, 0.0, "cost matches solution");
    check(s.stat_status() == static_cast<int>(Status::kSolved),
          "stat status kSolved");

    // trajectory read-back matches the Solution members
    std::vector<double> x;
    int rows = 0, cols = 0;
    check(s.field_get("x", x, rows, cols), "get x after solve");
    for (int k = 0; k <= N; ++k)
    {
        for (int i = 0; i < D::nx; ++i)
        {
            check_close(x[static_cast<std::size_t>(k) * D::nx + i],
                        s.solution().x[k](i), 0.0,
                        "x read-back (" + std::to_string(k) + ", "
                        + std::to_string(i) + ")");
        }
    }

    // statistics rows: one row per recorded iteration, 10 columns
    check(s.stat_iter() >= 1, "iterations recorded");
    std::vector<double> rows_data;
    int srows = 0, scols = 0;
    check(s.stat_rows(rows_data, srows, scols), "stat_rows");
    check(srows == s.stat_iter() && scols == 10, "stat rows/cols");
}

}  // namespace

int main()
{
    test_info();
    test_options();
    test_live_options();
    test_fields();
    test_solve_and_readback();

    if (failures == 0)
    {
        std::printf("All MexSession checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}

// MexSession unit test (docs/plans/MATLAB_MEX_PLAN.md, step 2): the
// host-independent core of the MATLAB MEX wrapper, exercised without
// MATLAB on the DoubleIntegrator. Checks: info extents, option discovery /
// get / set (flat and dotted, int/bool coercion, live inspect-based path),
// warm-start field set/get with extent validation, a full solve, solution
// read-back and the statistics rows.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ocp/mex/mex_api.hpp"
#include "ocp/solvers/acados/acados.hpp"

#include "double_integrator/double_integrator.hpp"

namespace
{

using namespace ocp;
using Problem = DoubleIntegrator;
using D = QpDim<Problem>;
using Session =
    ocp::mex::MexSession<Problem, AcadosSolver<Problem, Eigen::Dynamic>>;

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
    check(s.field_set("x", x.data(), D::nx, N + 1), "field_set x");
    check(s.field_set("u", u.data(), D::nu, N), "field_set u");
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

    // discovery: 28 driver + 5 glob leaves (options.*) + 29 HPIPM
    // (qp.options.*) + 4 scaling (scaling.options.*)
    const std::vector<std::string> names = s.option_names();
    check(names.size() == 66, "66 discoverable option names");
    bool has_max_iter = false;
    bool has_glob_alpha_min = false;
    bool has_qp_warm_start = false;
    for (const std::string& n : names)
    {
        has_max_iter = has_max_iter || (n == "options.max_iter");
        has_glob_alpha_min =
            has_glob_alpha_min || (n == "options.glob.alpha_min");
        has_qp_warm_start = has_qp_warm_start || (n == "qp.options.warm_start");
    }
    check(has_max_iter && has_glob_alpha_min && has_qp_warm_start,
          "names include options.max_iter, options.glob.alpha_min and "
          "qp.options.warm_start");

    // reads (int/bool coerced to double)
    double v = 0.0;
    check(s.option_get("options.max_iter", v) && v == 20.0, "get max_iter");
    check(s.option_get("options.glob.alpha_min", v) && v == 0.05,
          "get glob.alpha_min");
    check(s.option_get("options.print_level", v) && v == 0.0,
          "int option read as double");
    check(s.option_get("options.compute_hess_dyn", v) && v == 1.0,
          "bool option read as 1.0");
    check(!s.option_get("i_max", v), "unknown option name");
    check(!s.option_get("options.max_iter.x", v), "unknown dotted tail");

    // writes: int, bool and dotted names
    check(s.option_set("options.max_iter", 40.0), "set max_iter");
    check(s.solver().options().max_iter == 40, "set wrote the int member");
    check(s.option_get("options.max_iter", v) && v == 40.0, "get after set");
    check(s.option_set("options.compute_hess_dyn", 0.0), "set bool from 0.0");
    check(!s.solver().options().compute_hess_dyn, "bool member written");
    check(s.option_set("options.glob.alpha_reduction", 0.9), "set dotted name");
    check_close(s.solver().options().glob.alpha_reduction, 0.9, 1e-15,
                "dotted set wrote the glob member");
    // live round-trips on the component option surfaces
    check(s.option_set("qp.options.warm_start", 3.0),
          "set qp.options.warm_start");
    check(s.option_get("qp.options.warm_start", v) && v == 3.0,
          "qp.options.warm_start round-trip");
    check(s.option_set("scaling.options.scale_objective", 1.0),
          "set scaling.options.scale_objective");
    check(s.option_get("scaling.options.scale_objective", v) && v == 1.0,
          "scaling.options round-trip (bool coerced)");
    check(!s.option_set("nope", 1.0), "set unknown name");

    // flat "name = value" dump (what the MEX 'opt_dump' prints)
    const std::string dump = s.option_dump_str();
    check(dump.find("options.max_iter = 40") != std::string::npos,
          "dump contains max_iter");
    check(dump.find("qp.options.warm_start = 3") != std::string::npos,
          "dump contains int qp option");
    check(dump.find("scaling.options.scale_objective = true") != std::string::npos,
          "dump contains bool scaling option");
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
    check(s.option_set("options.max_iter", 1.0), "set max_iter = 1");
    const Status st1 = s.solve(cost);
    check(st1 == Status::kMaxIterations, "max_iter = 1 stops at the cap");
    check(s.stat_iter() >= 1, "at least one iteration recorded");

    check(s.option_set("options.max_iter", 20.0), "set max_iter = 20");
    const Status st2 = s.solve(cost);
    check(st2 == Status::kSolved, "max_iter = 20 converges");
}

void test_fields()
{
    Session s("mex_api_test");
    std::vector<double> x;
    int rows = 0, cols = 0;

    // uninitialized session: stage fields unavailable, scalars readable
    check(!s.field_extent("x", rows, cols), "extent x before init rejected");
    check(!s.field_extent("u", rows, cols), "extent u before init rejected");
    check(s.field_extent("N", rows, cols) && rows == 1 && cols == 1,
          "extent N before init is 1 x 1");
    check(!s.field_extent("nope", rows, cols), "unknown field rejected");
    check(!s.field_set("x", x.data(), D::nx, N + 1),
          "set x before init rejected");

    check(s.init(N), "init");

    // init sets x[0] from the fixed initial state, the rest stays zero
    check(s.field_extent("x", rows, cols), "extent x");
    check(rows == D::nx && cols == N + 1, "x extents");
    x.assign(static_cast<std::size_t>(rows * cols), 0.0);
    check(s.field_get("x", x.data()), "get x");
    check_close(x[0], 1.0, 1e-12, "x[0] q = initial_state");
    check_close(x[static_cast<std::size_t>(1)], 0.5, 1e-12,
                "x[0] v = initial_state");
    check(s.field_extent("u", rows, cols) && rows == D::nu && cols == N,
          "u extents");
    x.assign(static_cast<std::size_t>(rows * cols), 0.0);
    check(s.field_get("u", x.data()), "get u");

    // wrong extents / read-only fields are rejected
    check(!s.field_set("x", x.data(), D::nx, N), "x with wrong cols rejected");
    check(!s.field_set("u", x.data(), 2, N), "u with wrong rows rejected");
    check(!s.field_set("lambda_dyn", x.data(), 2, N),
          "multiplier field is not writable");

    // warm-start read-back
    warm_start(s);
    const Problem p;
    const double a = terminal_eq_feasible_control(p);
    check(s.field_extent("u", rows, cols), "extent u after warm start");
    x.assign(static_cast<std::size_t>(rows * cols), 0.0);
    check(s.field_get("u", x.data()), "get u after warm start");
    for (int k = 0; k < N; ++k)
    {
        check_close(x[static_cast<std::size_t>(k)], a, 0.0,
                    "u read-back column " + std::to_string(k));
    }
    check(s.field_extent("x", rows, cols), "extent x after warm start");
    x.assign(static_cast<std::size_t>(rows * cols), 0.0);
    check(s.field_get("x", x.data()), "get x after warm start");
    // column N: q_N + v_N = 1 holds exactly by construction of the control
    check_close(x[static_cast<std::size_t>(N) * D::nx]
                    + x[static_cast<std::size_t>(N) * D::nx + 1],
                1.0, 1e-10, "x read-back terminal equality");
}

void test_solve_and_readback()
{
    Session s("mex_api_test");
    double dummy = 0.0;
    check(s.solve(dummy) == Status::kNotInitialized,
          "solve before init is rejected");
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
    check(s.field_extent("x", rows, cols), "extent x after solve");
    x.assign(static_cast<std::size_t>(rows * cols), 0.0);
    check(s.field_get("x", x.data()), "get x after solve");
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

    // statistics table: one row per recorded iteration, 10 named columns
    check(s.stat_iter() >= 1, "iterations recorded");
    const std::vector<std::string> cnames = s.stat_column_names();
    check(cnames.size() == 10, "10 stat column names");
    check(cnames.front() == "res_stat" && cnames.back() == "funnel_width",
          "stat column names first/last");
    const int srows = s.stat_iter();
    const int scols = static_cast<int>(cnames.size());
    std::vector<double> table(static_cast<std::size_t>(srows * scols), 0.0);
    s.stat_fill(table.data(), srows, scols);

    // cross-check: the named column equals the matching table column
    int c_alpha = -1;
    for (int c = 0; c < scols; ++c)
    {
        if (cnames[c] == "alpha")
        {
            c_alpha = c;
        }
    }
    check(c_alpha >= 0, "alpha column present");
    std::vector<double> alpha(static_cast<std::size_t>(srows), 0.0);
    check(s.stat_column("alpha", alpha.data()), "stat_column alpha");
    for (int r = 0; r < srows; ++r)
    {
        check_close(alpha[r],
                    table[static_cast<std::size_t>(r) * scols + c_alpha],
                    0.0, "stat_column matches table row " + std::to_string(r));
        check(alpha[r] >= 0.0 && alpha[r] <= 1.0, "alpha in [0, 1]");
    }
    check(!s.stat_column("res_statx", alpha.data()),
          "stat_column unknown name");
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

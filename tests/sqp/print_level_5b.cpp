// Phase 5 test: print_level = 2 (detailed per-iteration QP dump).
//
// Checks (SQP_HPIPM_PARITY_PLAN.md sec. 8):
//   - print_level = 2 prints, after every QP solve, the QP solver's
//     per-IPM-iteration statistics table (HpipmStatistics::print, acados
//     ocp_qp_hpipm.c:367-373) in addition to the level-1 SQP rows.
//   - print_level = 1 prints the SQP rows but not the QP stat table.
//   - print_level = 0 prints nothing.
//
// stdout is captured via dup/dup2 (cf. options_2e test_print_iteration).

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <functional>
#include <string>

#include "ocp/solvers/acados/acados.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

namespace
{

using namespace ocp;

int failures = 0;

void check(bool ok, const char* msg)
{
    if (!ok)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

// Warm start for DoubleIntegrator (same construction as driver_2g):
// x[0] = initial state, constant control chosen so the terminal equality
// q_N + v_N = 1 holds exactly, forward Euler simulation.
Solution<DoubleIntegrator>
make_warm_start(const DoubleIntegrator& p, int n_stages)
{
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double Ts = p.params().Ts;
    const double den =
        Ts * Ts * n_stages * (n_stages - 1) / 2.0 + Ts * n_stages;
    const double a = (1.0 - (q0 + v0) - Ts * n_stages * v0) / den;

    Solution<DoubleIntegrator> sol(n_stages);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < n_stages; ++k)
    {
        sol.u[k](0) = a;
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }
    return sol;
}

// Run fn with stdout redirected to a file; return the captured text.
std::string capture_stdout(const std::function<void()>& fn)
{
    const char* path = "sqp_print_5b_test.txt";
    const int saved = ::dup(1);
    if (saved < 0)
    {
        return std::string();
    }
    const int tmp = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (tmp < 0)
    {
        return std::string();
    }

    // Flush anything the C runtime buffered for stdout onto the original
    // fd first; otherwise the redirect would dump the pending output of
    // earlier tests into the capture file (block-buffered redirects).
    std::fflush(stdout);
    ::dup2(tmp, 1);
    fn();
    std::fflush(stdout);
    ::close(tmp);
    ::dup2(saved, 1);
    std::fflush(stdout);
    ::close(saved);

    std::ifstream in(path);
    std::string text, line;
    while (std::getline(in, line))
    {
        text += line + "\n";
    }
    std::remove(path);
    return text;
}

std::size_t count_occ(const std::string& text, const std::string& needle)
{
    std::size_t n = 0;
    std::size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos)
    {
        ++n;
        ++pos;
    }
    return n;
}

void run_di_solve(int print_level)
{
    DoubleIntegrator prob;
    const int N = 10;
    AcadosOptions opts;
    opts.print_level = print_level;
    AcadosSolver<DoubleIntegrator> solver(opts);
    Solution<DoubleIntegrator> sol = make_warm_start(prob, N);
    const Status st = solver.solve(prob, sol);
    check(st == Status::kSolved, "DI solve: status kSolved");
    check(solver.statistics().iter >= 1, "DI solve: at least one row");
}

}  // namespace

int run_print_level_5b_tests()
{
    failures = 0;

    // level 2: QP stat table after every QP solve + SQP rows
    {
        const std::string text = capture_stdout(
            [] { run_di_solve(2); });
        // one table per QP solve: the terminal-eq-feasible warm start
        // converges after a single QP solve (the second SQP row is a
        // termination row, no QP solve), so exactly one table
        check(count_occ(text, "alpha_prim_aff") == 1,
              "level 2: QP stat table per QP solve");
        check(count_occ(text, "singular") == 1,
              "level 2: QP stat table header (last column)");
        check(text.find("# it") != std::string::npos,
              "level 2: SQP per-iteration rows present");
    }
    // level 1: SQP rows only, no QP table
    {
        const std::string text = capture_stdout(
            [] { run_di_solve(1); });
        check(count_occ(text, "alpha_prim_aff") == 0,
              "level 1: no QP stat table");
        check(text.find("# it") != std::string::npos,
              "level 1: SQP rows present");
    }
    // level 0: silent
    {
        const std::string text = capture_stdout(
            [] { run_di_solve(0); });
        check(text.empty(), "level 0: nothing printed");
    }

    if (failures == 0)
    {
        std::printf("All print_level (5b) checks passed.\n");
    }
    return failures;
}

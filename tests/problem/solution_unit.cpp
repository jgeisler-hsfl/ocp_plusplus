// Unit tests for component naming (Problem::state_name / control_name)
// and Solution::to_csv (include/ocp/problem.hpp):
//   - default names "x_i" / "u_i" when the Dims tag declares none
//   - user-supplied names via Dims::state_names / control_names
//   - CSV layout: header from the names, one row per stage k = 0..N,
//     terminal row (k = N) with blank control fields
//   - to_csv returns false on an unopenable path

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "ocp/problem.hpp"
#include "../../examples/double_integrator/double_integrator.hpp"

namespace
{

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

// Minimal problem without name arrays: exercises the default names.
struct MinDims
{
    static constexpr int nx = 2;
    static constexpr int nu = 1;
    static constexpr int ng = 0;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 0;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = false;
    static constexpr bool has_dynamics_hess = false;
    static constexpr bool has_constr_hess = false;
    static constexpr std::array<int, 0> state_box_idx = {};
    static constexpr std::array<int, 0> control_box_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_idx = {};
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

class MinProblem : public ocp::Problem<MinDims>
{
};

std::vector<std::string> read_lines(const std::string& path)
{
    std::vector<std::string> lines;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line))
    {
        lines.push_back(line);
    }
    return lines;
}

std::vector<std::string> split(const std::string& line, char sep)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : line)
    {
        if (c == sep)
        {
            out.push_back(cur);
            cur.clear();
        }
        else
        {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

void test_default_names()
{
    check(!MinProblem::has_state_names, "MinProblem: no state names");
    check(!MinProblem::has_control_names, "MinProblem: no control names");
    check(MinProblem::state_name(0) == "x_0", "default state name x_0");
    check(MinProblem::state_name(1) == "x_1", "default state name x_1");
    check(MinProblem::control_name(0) == "u_0", "default control name u_0");
}

void test_dims_names()
{
    check(DoubleIntegrator::has_state_names, "DI: state names declared");
    check(DoubleIntegrator::has_control_names, "DI: control names declared");
    check(DoubleIntegrator::state_name(0) == "q", "state name q");
    check(DoubleIntegrator::state_name(1) == "v", "state name v");
    check(DoubleIntegrator::control_name(0) == "a", "control name a");
}

void test_to_csv_named()
{
    const int N = 3;
    ocp::Solution<DoubleIntegrator> sol(N);
    for (int k = 0; k <= N; ++k)
    {
        sol.x[k](0) = k;
        sol.x[k](1) = 2.0 * k;
    }
    for (int k = 0; k < N; ++k)
    {
        sol.u[k](0) = 0.5 * k;
    }

    const std::string path = "/tmp/opencode/ocp_solution_test.csv";
    check(sol.to_csv(path.c_str()), "to_csv returns true");

    const auto lines = read_lines(path);
    check(lines.size() == static_cast<std::size_t>(N) + 2,
          "header + N+1 data rows");
    if (lines.size() != static_cast<std::size_t>(N) + 2)
    {
        return;
    }
    check(lines[0] == "k,q,v,a", "header uses Dims names");

    // row k lives at lines[k + 1] (the header is lines[0])
    const auto r1 = split(lines[2], ',');
    check(r1.size() == 4, "row k=1 has 4 fields");
    if (r1.size() == 4)
    {
        check(r1[0] == "1", "row k=1 stage index");
        check(std::stod(r1[1]) == 1.0, "row k=1 q");
        check(std::stod(r1[2]) == 2.0, "row k=1 v");
        check(std::stod(r1[3]) == 0.5, "row k=1 a");
    }

    // terminal row: control field is blank (no u_N)
    const auto rN = split(lines[N + 1], ',');
    check(rN.size() == 4, "terminal row has 4 fields");
    if (rN.size() == 4)
    {
        check(rN[0] == "3", "terminal stage index");
        check(std::stod(rN[1]) == 3.0, "terminal q");
        check(std::stod(rN[2]) == 6.0, "terminal v");
        check(rN[3].empty(), "terminal control field blank");
    }
}

void test_to_csv_default_names()
{
    ocp::Solution<MinProblem> sol(1);
    sol.x[0](0) = 1.0;
    sol.x[0](1) = 2.0;
    sol.x[1](0) = 3.0;
    sol.x[1](1) = 4.0;
    sol.u[0](0) = 5.0;

    const std::string path = "/tmp/opencode/ocp_solution_test_min.csv";
    check(sol.to_csv(path.c_str()), "to_csv (default names) returns true");
    const auto lines = read_lines(path);
    check(lines.size() == 3, "min: header + N+1 data rows");
    if (lines.size() >= 1)
    {
        check(lines[0] == "k,x_0,x_1,u_0", "header uses default names");
    }
}

void test_to_csv_failure()
{
    ocp::Solution<MinProblem> sol(1);
    check(!sol.to_csv("/tmp/opencode/no_such_dir_ocp_unit/x.csv"),
          "to_csv fails on unopenable path");
}

}  // namespace

int main()
{
    test_default_names();
    test_dims_names();
    test_to_csv_named();
    test_to_csv_default_names();
    test_to_csv_failure();

    if (failures == 0)
    {
        std::printf("All solution names/csv checks passed.\n");
    }
    return failures;
}

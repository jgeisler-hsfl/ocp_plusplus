// Unit tests for ocp::Solution::shift (include/ocp/problem.hpp):
// the one-stage receding-horizon advance of a solution. Covers both fill
// policies (kRollover, kZero), the dynamic (NH = Eigen::Dynamic) and the
// fixed (NH >= 1) horizon extents, and verifies every shifted entry, the
// new initial state, the untouched terminal multipliers and the
// cost/status reset.

#include <cstdio>
#include <string>

#include "ocp/problem.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

namespace
{

using namespace ocp;
using DI = DoubleIntegrator;
using state_t = DI::state_t;

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

/// Exact element-wise equality (all test values are exactly
/// representable in double).
template <class Vec>
bool vec_eq(const Vec& a, const Vec& b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    return a.size() == 0 || (a - b).cwiseAbs().maxCoeff() == 0.0;
}

template <class Vec>
bool vec_zero(const Vec& v)
{
    return v.size() == 0 || v.cwiseAbs().maxCoeff() == 0.0;
}

// ---------------------------------------------------------------------
// fill a Solution with distinguishable per-stage values
// ---------------------------------------------------------------------

template <int NH>
void fill_solution(ocp::Solution<DI, NH>& sol, int N)
{
    for (int k = 0; k <= N; ++k)
    {
        sol.x[k](0) = 100.0 * (k + 1);
        sol.x[k](1) = 10.0 * (k + 1);
        sol.lambda_box_state[k](0) = 11.0 * (k + 1);
        sol.lambda_box_state[k](1) = 13.0 * (k + 1);
    }
    for (int k = 0; k < N; ++k)
    {
        sol.u[k](0) = k + 1;
        sol.lambda_dyn[k](0) = 300.0 * (k + 1);
        sol.lambda_dyn[k](1) = 400.0 * (k + 1);
        sol.lambda_ineq_stage[k](0) = 5.0 * (k + 1);
        sol.lambda_lin_stage[k](0) = 7.0 * (k + 1);
        sol.lambda_box_control[k](0) = 17.0 * (k + 1);
    }
    sol.lambda_eq_term(0) = 1.234;
    sol.cost_value = 42.0;
    sol.status = Status::kSolved;
}

// ---------------------------------------------------------------------
// the shifted-solution invariants (mapping table of Solution::shift)
// ---------------------------------------------------------------------

template <int NH>
void check_shifted(const ocp::Solution<DI, NH>& sol,
                   const ocp::Solution<DI, NH>& old,
                   const state_t& x0_new, ShiftFill fill, int N)
{
    check(sol.N == N, "N unchanged");
    check(sol.x.size() == (size_t)N + 1, "x extent unchanged");
    check(sol.u.size() == (size_t)N, "u extent unchanged");

    // states: up one index, x[0] = x0_new, x[N] per fill
    check(vec_eq(sol.x[0], x0_new), "x[0] set to x0_new");
    for (int k = 1; k < N; ++k)
    {
        check(vec_eq(sol.x[k], old.x[k + 1]),
              "x[" + std::to_string(k) + "] = old x[k+1]");
    }
    if (fill == ShiftFill::kRollover)
    {
        check(vec_eq(sol.x[N], old.x[N]), "x[N] rolled over");
    }
    else
    {
        check(vec_zero(sol.x[N]), "x[N] zeroed");
    }

    // controls: up one index, u[N-1] per fill
    for (int k = 0; k + 1 < N; ++k)
    {
        check(vec_eq(sol.u[k], old.u[k + 1]),
              "u[" + std::to_string(k) + "] = old u[k+1]");
    }
    if (fill == ShiftFill::kRollover)
    {
        check(vec_eq(sol.u[N - 1], old.u[N - 1]), "u[N-1] rolled over");
    }
    else
    {
        check(vec_zero(sol.u[N - 1]), "u[N-1] zeroed");
    }

    // stage multipliers: up one index, new last stage zeroed
    for (int k = 0; k + 1 < N; ++k)
    {
        check(vec_eq(sol.lambda_dyn[k], old.lambda_dyn[k + 1]),
              "lambda_dyn[" + std::to_string(k) + "] = old [k+1]");
        check(
            vec_eq(sol.lambda_ineq_stage[k], old.lambda_ineq_stage[k + 1]),
            "lambda_ineq_stage[" + std::to_string(k) + "] = old [k+1]");
        check(
            vec_eq(sol.lambda_eq_stage[k], old.lambda_eq_stage[k + 1]),
            "lambda_eq_stage[" + std::to_string(k) + "] = old [k+1]");
        check(
            vec_eq(sol.lambda_lin_stage[k], old.lambda_lin_stage[k + 1]),
            "lambda_lin_stage[" + std::to_string(k) + "] = old [k+1]");
        check(
            vec_eq(sol.lambda_box_control[k], old.lambda_box_control[k + 1]),
            "lambda_box_control[" + std::to_string(k) + "] = old [k+1]");
    }
    check(vec_zero(sol.lambda_dyn[N - 1]), "lambda_dyn[N-1] zeroed");
    check(vec_zero(sol.lambda_ineq_stage[N - 1]),
          "lambda_ineq_stage[N-1] zeroed");
    check(vec_zero(sol.lambda_eq_stage[N - 1]),
          "lambda_eq_stage[N-1] zeroed");
    check(vec_zero(sol.lambda_lin_stage[N - 1]),
          "lambda_lin_stage[N-1] zeroed");
    check(vec_zero(sol.lambda_box_control[N - 1]),
          "lambda_box_control[N-1] zeroed");

    // state box multipliers (N + 1 stages): up one index; the terminal
    // entry stays
    for (int k = 0; k < N; ++k)
    {
        check(
            vec_eq(sol.lambda_box_state[k], old.lambda_box_state[k + 1]),
            "lambda_box_state[" + std::to_string(k) + "] = old [k+1]");
    }
    check(vec_eq(sol.lambda_box_state[N], old.lambda_box_state[N]),
          "lambda_box_state[N] (terminal) unchanged");

    // terminal multipliers: untouched
    check(vec_eq(sol.lambda_ineq_term, old.lambda_ineq_term),
          "lambda_ineq_term unchanged");
    check(vec_eq(sol.lambda_eq_term, old.lambda_eq_term),
          "lambda_eq_term unchanged");
    check(vec_eq(sol.lambda_lin_term, old.lambda_lin_term),
          "lambda_lin_term unchanged");

    // cost / status reset
    check(sol.cost_value == 0.0, "cost_value reset");
    check(sol.status == Status::kUnset, "status reset to kUnset");
}

// ---------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------

void test_rollover_dynamic()
{
    const int N = 4;
    ocp::Solution<DI> sol(N);
    fill_solution<Eigen::Dynamic>(sol, N);
    const auto old = sol;

    state_t x0_new;
    x0_new << 3.3, -1.7;
    sol.shift(x0_new);

    check_shifted<Eigen::Dynamic>(sol, old, x0_new, ShiftFill::kRollover, N);
}

void test_zero_dynamic()
{
    const int N = 4;
    ocp::Solution<DI> sol(N);
    fill_solution<Eigen::Dynamic>(sol, N);
    const auto old = sol;

    state_t x0_new;
    x0_new << -0.5, 2.5;
    sol.shift(x0_new, ShiftFill::kZero);

    check_shifted<Eigen::Dynamic>(sol, old, x0_new, ShiftFill::kZero, N);
}

void test_rollover_dynamic_n1()
{
    // smallest horizon: the "up one index" loops are empty, only the
    // boundary entries act.
    const int N = 1;
    ocp::Solution<DI> sol(N);
    fill_solution<Eigen::Dynamic>(sol, N);
    const auto old = sol;

    state_t x0_new;
    x0_new << 0.25, -0.75;
    sol.shift(x0_new);

    check_shifted<Eigen::Dynamic>(sol, old, x0_new, ShiftFill::kRollover, N);
}

void test_rollover_fixed()
{
    // compile-time horizon: std::array-backed trajectories
    const int N = 5;
    ocp::Solution<DI, 5> sol;
    fill_solution<5>(sol, N);
    const auto old = sol;

    state_t x0_new;
    x0_new << 4.0, 0.1;
    sol.shift(x0_new);

    check_shifted<5>(sol, old, x0_new, ShiftFill::kRollover, N);
}

void test_zero_fixed()
{
    const int N = 5;
    ocp::Solution<DI, 5> sol;
    fill_solution<5>(sol, N);
    const auto old = sol;

    state_t x0_new;
    x0_new << -1.0, 1.0;
    sol.shift(x0_new, ShiftFill::kZero);

    check_shifted<5>(sol, old, x0_new, ShiftFill::kZero, N);
}

}  // namespace

int main()
{
    test_rollover_dynamic();
    test_zero_dynamic();
    test_rollover_dynamic_n1();
    test_rollover_fixed();
    test_zero_fixed();

    if (failures == 0)
    {
        std::printf("All solution_shift checks passed.\n");
    }
    return failures;
}

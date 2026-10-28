// np=2 generator fixture test (docs/plans/MULTI_PARAM_PLAN.md, block 6).
//
// The NP2 problem class is emitted by tools/acados2ocp_pp.py from
// tests/params/np2/acados_codegen/acados_ocp_nlp.json (dims.np = 2) with a
// hand-written casadi-ABI explicit ODE (np2_expl_ode_fun.c):
//   f(q, v, p) = [v; p[0]*u + p[1]*q],   p = [p0; p1]
//
// This exercises the np >= 2 generator path end to end:
//   (a) the emitted {name}Params object exposes p as a 2-element matrix
//       leaf with the JSON initial values,
//   (b) inspect round-trip on p (set/get by name; wrong element count and
//       unknown names rejected),
//   (c) the problem forwards params() to the ODE model (single shared
//       object),
//   (d) the live parameter vector actually feeds the dynamics.

#include <cmath>
#include <cstdio>

#include "ocp/scaling.hpp"
#include "ocp/utils/inspect.hpp"

#include "NP2.hpp"

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

// ------------------------------------------------------------------
// (a) p is a 2-element matrix leaf with the JSON initial values
// ------------------------------------------------------------------
void test_p_leaf()
{
    ocp::NP2 p;
    const auto& pr = p.params();

    int rows = 0;
    int cols = 0;
    check(ocp::inspect::matrix_extent(pr, "p", rows, cols), "extent of p");
    check(rows == 2 && cols == 1, "p extent 2x1");

    Eigen::Matrix<double, 2, 1> pv;
    check(ocp::inspect::get_matrix(pr, "p", pv), "get p by name");
    check_close("p(0) from JSON", pv(0), 1.0, 0.0);
    check_close("p(1) from JSON", pv(1), 0.5, 0.0);

    // leaf discovery: exactly one leaf, named "p"
    const std::vector<std::string> names =
        ocp::inspect::leaf_names<ocp::NP2Params>();
    check(names.size() == 1 && names[0] == "p", "leaf names = [p]");
}

// ------------------------------------------------------------------
// (b) inspect round-trip
// ------------------------------------------------------------------
void test_roundtrip()
{
    ocp::NP2 p;

    const double np2[2] = {2.0, 0.25};
    check(ocp::inspect::set_matrix_double(p.params(), "p", np2, 2),
          "set p by name (2 elements)");
    double buf[2] = {0.0, 0.0};
    int rows = 0;
    int cols = 0;
    check(ocp::inspect::get_matrix_double(p.params(), "p", buf, rows, cols),
          "get p by name");
    check(rows == 2 && cols == 1, "p extent after write");
    check_close("p(0) after set", buf[0], 2.0, 0.0);
    check_close("p(1) after set", buf[1], 0.25, 0.0);

    // wrong element count is rejected, leaf untouched
    const double one[1] = {7.0};
    check(!ocp::inspect::set_matrix_double(p.params(), "p", one, 1),
          "set p with 1 element rejected");
    double buf2[2] = {0.0, 0.0};
    rows = 0;
    cols = 0;
    check(ocp::inspect::get_matrix_double(p.params(), "p", buf2, rows, cols),
          "re-read p");
    check_close("p(0) after rejected write", buf2[0], 2.0, 0.0);
    check_close("p(1) after rejected write", buf2[1], 0.25, 0.0);

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
// (c) single shared object: problem forwards params() to the ODE
// ------------------------------------------------------------------
void test_shared_object()
{
    static_assert(ocp::detail::has_params<ocp::NP2>::value,
                  "generated problem must read as 'has params'");
    static_assert(ocp::detail::has_params<ocp::NP2Ode>::value,
                  "generated model must read as 'has params'");

    ocp::NP2 p;
    check(&p.params() == &p.ode().params(),
          "problem exposes the model's live params object");

    const ocp::NP2 cp;
    check(&cp.params() == &cp.ode().params(),
          "const forwarding reaches the same object");

    // the ScaledProblem wrapper forwards to the wrapped problem's params
    ocp::NP2 q;
    ocp::ScaledProblem<ocp::NP2> scaled(q);
    static_assert(
        ocp::detail::has_params<ocp::ScaledProblem<ocp::NP2>>::value,
        "wrapper must read as 'has params'");
    check(&scaled.params() == &q.params(),
          "wrapper exposes the wrapped problem's live params object");
    const double np2[2] = {3.0, 0.1};
    check(ocp::inspect::set_matrix_double(scaled.params(), "p", np2, 2),
          "write through wrapper");
    check_close("wrapped object updated", q.params().p(0), 3.0, 0.0);
}

// ------------------------------------------------------------------
// (d) the live parameter vector feeds the dynamics
// ------------------------------------------------------------------
void test_dynamics_uses_p()
{
    ocp::NP2 p;
    ocp::NP2::state_t x;
    x << 1.0, 2.0;
    ocp::NP2::control_t u;
    u << 3.0;

    // default p = [1; 0.5]: f = [v; p0*u + p1*q] = [2; 3 + 0.5]
    const auto f0 = p.ode().f(x, u, 0.0);
    check_close("f(0) = v", f0(0), 2.0, 0.0);
    check_close("f(1) at p = [1; 0.5]", f0(1), 3.5, 0.0);

    const double np2[2] = {2.0, 0.25};
    check(ocp::inspect::set_matrix_double(p.params(), "p", np2, 2),
          "set p (dynamics check)");
    const auto f1 = p.ode().f(x, u, 0.0);
    check_close("f(0) unchanged by p", f1(0), 2.0, 0.0);
    check_close("f(1) at p = [2; 0.25]", f1(1), 6.25, 0.0);

    // the FD Jacobian also reads the live p (d/du of p0*u + p1*q is p0)
    ocp::NP2::dyn_df_dx_t df_dx;
    ocp::NP2::dyn_df_du_t df_du;
    p.ode().jacobian(x, u, 0.0, df_dx, df_du);
    check_close("df/du = p0 = 2", df_du(1, 0), 2.0, 1e-6);
}

}  // namespace

int main()
{
    test_p_leaf();
    test_roundtrip();
    test_shared_object();
    test_dynamics_uses_p();
    if (failures == 0)
    {
        std::puts("params_np2: all checks passed");
        return 0;
    }
    std::fprintf(stderr, "params_np2: %d failure(s)\n", failures);
    return 1;
}

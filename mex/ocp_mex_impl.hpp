// ocp_mex_impl.hpp
//
// Thin mex.h glue around ocp::mex::MexSession (docs/plans/MATLAB_MEX_PLAN.md,
// section 2): the substance lives in the pure-C++ MexSession (unit-tested
// by tests/mex/mex_api_unit.cpp); this adapter only translates MATLAB
// arrays/strings into the session's calls and back. The configuration
// (ocp_mex::{Problem,kHorizon,Solver,kName}) comes from the config header
// selected by OCP_MEX_CONFIG (see ocp_mex_main.cpp).
//
// Command surface (v1, plan section 4):
//   info | init N | options | opt_get name | opt_set name value |
//   opt_dump | set field M | get field | solution | solve | stats
//
// Conventions: column-major stage order (MATLAB column k+1 = stage k),
// active box rows only (2*nbx, [lower; upper]), doubles throughout.

#pragma once

#include "mex.h"

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "ocp/mex/mex_api.hpp"

namespace
{

using Session =
    ocp::mex::MexSession<ocp_mex::Problem, ocp_mex::kHorizon, ocp_mex::Solver>;

/// The 14 solution field names (Solution member names, verbatim; plan 5).
constexpr const char* const kFields[] = {
    "x", "u", "lambda_dyn", "lambda_ineq_stage", "lambda_eq_stage",
    "lambda_lin_stage", "lambda_ineq_term", "lambda_eq_term",
    "lambda_lin_term", "lambda_box_state", "lambda_box_control",
    "cost_value", "status", "N"};

std::string usage()
{
    return std::string("usage: ocp_mex_") + ocp_mex::kName +
           "(cmd, ...), where cmd is one of:\n"
           "  'info'                      problem/solver extents + N\n"
           "  'init' N                    set the horizon (resizes session)\n"
           "  'options'                   all option names (cell, incl. glob.*)\n"
           "  'opt_get' name              option value (double; int/bool coerced)\n"
           "  'opt_set' name value        set an option (live: pushed to solver)\n"
           "  'opt_dump'                  print a flat option snapshot\n"
           "  'set' field M               warm start: 'x' (nx x (N+1)) or "
           "'u' (nu x N)\n"
           "  'get' field                 solution field (x, u, lambda_*, "
           "cost_value, status, N)\n"
           "  'solution'                  struct with all solution fields\n"
           "  'solve'                     [status, cost]; status = int ocp::Status\n"
           "  'stats'                     struct: iter, status, rows (iter x "
           "10)\n";
}

/// MATLAB-side error (the MEX error channel; the C++ core stays
/// exception-free). Never returns.
void die(const std::string& msg)
{
    mexErrMsgTxt(msg.c_str());
}

/// Argument i as a char string.
std::string arg_str(int nrhs, const mxArray* prhs[], int i)
{
    if (i >= nrhs || !mxIsChar(prhs[i]))
    {
        die("argument " + std::to_string(i + 1) + " must be a char string");
    }
    const std::size_t len = static_cast<std::size_t>(mxGetN(prhs[i]));
    std::string s(len, '\0');
    mxGetString(prhs[i], s.data(), static_cast<int>(len) + 1);
    return s;
}

/// Argument i as a scalar double.
double arg_scalar(int nrhs, const mxArray* prhs[], int i)
{
    if (i >= nrhs || !mxIsNumeric(prhs[i]) ||
        static_cast<std::size_t>(mxGetNumberOfElements(prhs[i])) != 1)
    {
        die("argument " + std::to_string(i + 1) + " must be a scalar");
    }
    return mxGetPr(prhs[i])[0];
}

/// Argument i as a 2-D real numeric matrix; copied column-major (MATLAB
/// and the session both use column k = stage k) into `out`.
void arg_matrix(int nrhs, const mxArray* prhs[], int i,
                std::vector<double>& out, int& rows, int& cols)
{
    if (i >= nrhs || !mxIsNumeric(prhs[i]) || mxIsComplex(prhs[i]) ||
        mxGetNumberOfDimensions(prhs[i]) != 2)
    {
        die("argument " + std::to_string(i + 1) +
            " must be a 2-D real numeric matrix");
    }
    rows = static_cast<int>(mxGetM(prhs[i]));
    cols = static_cast<int>(mxGetN(prhs[i]));
    out.assign(static_cast<std::size_t>(rows * cols), 0.0);
    if (!out.empty())
    {
        std::memcpy(out.data(), mxGetPr(prhs[i]), out.size() * sizeof(double));
    }
}

/// A flat column-major double buffer as a MATLAB matrix (zero-extent
/// fields yield 0 x n / n x 0 arrays).
mxArray* make_matrix(const std::vector<double>& data, int rows, int cols)
{
    auto* m = mxCreateDoubleMatrix(rows, cols, mxREAL);
    if (rows > 0 && cols > 0)
    {
        std::memcpy(mxGetPr(m), data.data(), data.size() * sizeof(double));
    }
    return m;
}

// mxSetField / mxSetCell take ownership of the passed value array (no deep
// copy); the caller must not destroy it afterwards.

void struct_set_num(mxArray* s, const char* name, double v)
{
    // mxAddField returns the new field number (0-based) or -1 on failure
    if (mxAddField(s, name) < 0)
    {
        die(std::string("cannot add struct field '") + name + "'");
    }
    mxSetField(s, 0, name, mxCreateDoubleScalar(v));
}

void struct_set_mat(mxArray* s, const char* name, mxArray* m)
{
    // mxAddField returns the new field number (0-based) or -1 on failure
    if (mxAddField(s, name) < 0)
    {
        die(std::string("cannot add struct field '") + name + "'");
    }
    mxSetField(s, 0, name, m);
}

void struct_set_str(mxArray* s, const char* name, const std::string& v)
{
    // mxAddField returns the new field number (0-based) or -1 on failure
    if (mxAddField(s, name) < 0)
    {
        die(std::string("cannot add struct field '") + name + "'");
    }
    mxSetField(s, 0, name, mxCreateString(v.c_str()));
}

mxArray* do_info(const Session& session)
{
    const auto i = session.info();
    auto* s = mxCreateStructMatrix(1, 1, 0, nullptr);
    struct_set_num(s, "nx", static_cast<double>(i.nx));
    struct_set_num(s, "nu", static_cast<double>(i.nu));
    struct_set_num(s, "ng", static_cast<double>(i.ng));
    struct_set_num(s, "ne", static_cast<double>(i.ne));
    struct_set_num(s, "nl", static_cast<double>(i.nl));
    struct_set_num(s, "ng_t", static_cast<double>(i.ng_t));
    struct_set_num(s, "ne_t", static_cast<double>(i.ne_t));
    struct_set_num(s, "nl_t", static_cast<double>(i.nl_t));
    struct_set_num(s, "nbx", static_cast<double>(i.nbx));
    struct_set_num(s, "nbu", static_cast<double>(i.nbu));
    struct_set_num(s, "nbx_t", static_cast<double>(i.nbx_t));
    struct_set_num(s, "fixed_initial_state",
                   i.fixed_initial_state ? 1.0 : 0.0);
    struct_set_num(s, "N", static_cast<double>(i.N));
    struct_set_str(s, "solver", ocp_mex::kName);
    return s;
}

mxArray* do_options(const Session& session)
{
    const std::vector<std::string> names = session.option_names();
    auto* c = mxCreateCellMatrix(1, static_cast<mwSize>(names.size()));
    for (std::size_t i = 0; i < names.size(); ++i)
    {
        mxSetCell(c, static_cast<mwIndex>(i),
                  mxCreateString(names[i].c_str()));
    }
    return c;
}

mxArray* do_get(const Session& session, const std::string& field)
{
    std::vector<double> data;
    int rows = 0, cols = 0;
    if (!session.field_get(field, data, rows, cols))
    {
        die("unknown solution field '" + field + "' (see 'info' and the MEX "
            "README for the field list)");
    }
    return make_matrix(data, rows, cols);
}

mxArray* do_solution(const Session& session)
{
    auto* s = mxCreateStructMatrix(1, 1, 0, nullptr);
    for (const char* f : kFields)
    {
        std::vector<double> data;
        int rows = 0, cols = 0;
        session.field_get(f, data, rows, cols);
        struct_set_mat(s, f, make_matrix(data, rows, cols));
    }
    return s;
}

mxArray* do_stats(const Session& session)
{
    std::vector<double> data;
    int rows = 0, cols = 0;
    session.stat_rows(data, rows, cols);
    auto* s = mxCreateStructMatrix(1, 1, 0, nullptr);
    struct_set_num(s, "iter", static_cast<double>(session.stat_iter()));
    struct_set_num(s, "status", static_cast<double>(session.stat_status()));
    struct_set_mat(s, "rows", make_matrix(data, rows, cols));
    return s;
}

}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // Process-persistent ("global") state of this MEX binary: the MEX is
    // loaded once into the MATLAB process and this object outlives every
    // call. Documented exception to the no-static-state rule (I/O boundary,
    // not numerical core; plan section 2).
    static Session session(ocp_mex::kName);

    if (nrhs < 1)
    {
        die(usage());
    }
    const std::string cmd = arg_str(nrhs, prhs, 0);

    if (cmd == "info")
    {
        if (nlhs < 1)
        {
            die("info: assign the output, e.g.  i = ocp_mex_" +
                std::string(ocp_mex::kName) + "('info')");
        }
        plhs[0] = do_info(session);
    }
    else if (cmd == "init")
    {
        const double n_d = arg_scalar(nrhs, prhs, 1);
        const int N = static_cast<int>(n_d);
        if (N < 1 || n_d != static_cast<double>(N))
        {
            die("init: N must be a positive integer");
        }
        if (nlhs < 1)
        {
            die("init: assign the output, e.g.  ok = ocp_mex_" +
                std::string(ocp_mex::kName) + "('init', N)");
        }
        plhs[0] = mxCreateLogicalScalar(session.init(N));
    }
    else if (cmd == "options")
    {
        if (nlhs < 1)
        {
            die("options: assign the output, e.g.  opts = ocp_mex_" +
                std::string(ocp_mex::kName) + "('options')");
        }
        plhs[0] = do_options(session);
    }
    else if (cmd == "opt_get")
    {
        const std::string name = arg_str(nrhs, prhs, 1);
        double v = 0.0;
        if (!session.option_get(name, v))
        {
            die("unknown option: '" + name + "' (see 'options')");
        }
        if (nlhs < 1)
        {
            die("opt_get: assign the output, e.g.  v = ocp_mex_" +
                std::string(ocp_mex::kName) + "('opt_get', name)");
        }
        plhs[0] = mxCreateDoubleScalar(v);
    }
    else if (cmd == "opt_set")
    {
        const std::string name = arg_str(nrhs, prhs, 1);
        const double value = arg_scalar(nrhs, prhs, 2);
        if (!session.option_set(name, value))
        {
            die("unknown option: '" + name + "' (see 'options')");
        }
        if (nlhs < 1)
        {
            die("opt_set: assign the output (logical success flag)");
        }
        plhs[0] = mxCreateLogicalScalar(1);
    }
    else if (cmd == "opt_dump")
    {
        session.option_dump(stdout);
    }
    else if (cmd == "set")
    {
        const std::string field = arg_str(nrhs, prhs, 1);
        std::vector<double> data;
        int rows = 0, cols = 0;
        arg_matrix(nrhs, prhs, 2, data, rows, cols);
        if (!session.field_set(field, data, rows, cols))
        {
            die("set '" + field + "' rejected (only 'x' and 'u' are "
                "writable; extents must match the initialized session)");
        }
        if (nlhs < 1)
        {
            die("set: assign the output (logical success flag)");
        }
        plhs[0] = mxCreateLogicalScalar(1);
    }
    else if (cmd == "get")
    {
        const std::string field = arg_str(nrhs, prhs, 1);
        if (nlhs < 1)
        {
            die("get: assign the output, e.g.  x = ocp_mex_" +
                std::string(ocp_mex::kName) + "('get', 'x')");
        }
        plhs[0] = do_get(session, field);
    }
    else if (cmd == "solution")
    {
        if (nlhs < 1)
        {
            die("solution: assign the output, e.g.  sol = ocp_mex_" +
                std::string(ocp_mex::kName) + "('solution')");
        }
        plhs[0] = do_solution(session);
    }
    else if (cmd == "solve")
    {
        if (nlhs < 2)
        {
            die("solve: use  [status, cost] = ocp_mex_" +
                std::string(ocp_mex::kName) + "('solve')");
        }
        double cost = 0.0;
        const ocp::Status st = session.solve(cost);
        plhs[0] = mxCreateDoubleScalar(static_cast<double>(st));
        plhs[1] = mxCreateDoubleScalar(cost);
    }
    else if (cmd == "stats")
    {
        if (nlhs < 1)
        {
            die("stats: assign the output, e.g.  st = ocp_mex_" +
                std::string(ocp_mex::kName) + "('stats')");
        }
        plhs[0] = do_stats(session);
    }
    else
    {
        die("unknown command: '" + cmd + "'\n\n" + usage());
    }
}

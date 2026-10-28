// ocp_mex_impl.hpp
//
// Thin mex.h glue around ocp::mex::MexSession (docs/plans/MATLAB_MEX_PLAN.md,
// section 2): the substance lives in the pure-C++ MexSession (unit-tested
// by tests/mex/mex_api_unit.cpp); this adapter only translates MATLAB
// arrays/strings into the session's calls and back. The configuration
// (ocp_mex::{Problem,kHorizon,Solver,kName}) comes from the config header
// selected by OCP_MEX_CONFIG (see ocp_mex_main.cpp).
//
// Command surface (v2, docs/plans/finished/MEX_IMPROVEMENTS_PLAN.md):
//   info | init N | options | opt_get name | opt_set name value |
//   opt_set struct | opt_dump | param_names | param_extent name |
//   param_get name | param_set name value | param_dump |
//   set field M | get field | solution |
//   solve | stats | stat_col name
//
// Conventions: column-major stage order (MATLAB column k+1 = stage k),
// active box rows only (2*nbx, [lower; upper]), doubles throughout.
// Argument types are checked strictly (real double / scalar / 2-D matrix,
// never a generic numeric); success flags are optional outputs (produced
// only when requested).

#pragma once

#include "mex.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "ocp/mex/mex_api.hpp"

namespace
{

using Session =
    ocp::mex::MexSession<ocp_mex::Problem, ocp_mex::Solver, ocp_mex::kHorizon>;

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
           "  'options'                   all option names (cell; options.*, "
           "qp.options.*, scaling.options.*)\n"
           "  'opt_get' name              option value (double; int/bool "
           "coerced)\n"
           "  'opt_set' name value        set one option (live: effective "
           "on the next solve)\n"
           "  'opt_set' struct            set options by struct (field "
           "nesting mirrors the dotted names; unknown fields warn)\n"
           "  'opt_dump'                  print a flat option snapshot\n"
            "  'param_names'               all model parameter names (cell; "
            "empty if the problem has no parameters)\n"
            "  'param_extent' name         parameter leaf extent as "
            "[rows, cols] (scalars: 1 x 1)\n"
            "  'param_get' name            parameter leaf (rows x cols; "
            "scalars 1 x 1)\n"
            "  'param_set' name M          set one parameter leaf (M must "
            "have the leaf's extent; live: effective on the next solve)\n"
            "  'param_dump'                print a flat parameter snapshot\n"
           "  'set' field M               warm start: 'x' (nx x (N+1)) or "
           "'u' (nu x N)\n"
           "  'get' field                 solution field (x, u, lambda_*, "
           "cost_value, status, N)\n"
           "  'solution'                  struct with all solution fields\n"
           "  'solve'                     [status, cost] or cost; status = "
           "int ocp::Status\n"
           "  'stats'                     [stats, status]; stats is a "
           "1 x iter struct array, one field per column name\n"
           "  'stat_col' name             one named statistics column "
           "(n x 1)\n";
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

/// Argument i as a scalar double (strict: real double class, exactly one
/// element — logicals, ints and singles are rejected).
double arg_scalar(int nrhs, const mxArray* prhs[], int i)
{
    if (i >= nrhs || !mxIsDouble(prhs[i]) || mxIsComplex(prhs[i]) ||
        mxGetNumberOfElements(prhs[i]) != 1)
    {
        die("argument " + std::to_string(i + 1) +
            " must be a scalar double");
    }
    return mxGetScalar(prhs[i]);
}

/// Argument i as a 2-D real double matrix (strict: double class, not
/// complex; its own storage is used in place, no copy).
const mxArray* arg_matrix(int nrhs, const mxArray* prhs[], int i)
{
    if (i >= nrhs || !mxIsDouble(prhs[i]) || mxIsComplex(prhs[i]) ||
        mxGetNumberOfDimensions(prhs[i]) != 2)
    {
        die("argument " + std::to_string(i + 1) +
            " must be a 2-D real double matrix");
    }
    return prhs[i];
}

// Ownership: mxSetField / mxSetCell do not deep-copy — they transfer
// ownership of the passed array to the owning struct/cell, and the
// top-level array returned from mexFunction is owned by MATLAB. Every
// call builds its own fresh arrays, so nothing crosses call boundaries;
// the only process-persistent state is the C++ Session (P9).

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

/// A solution field as a fresh mxArray filled directly by the session
/// (zero intermediate buffer): extents come from the session, values are
/// written into the mxArray's own storage.
mxArray* field_matrix(const Session& session, const char* field)
{
    int rows = 0;
    int cols = 0;
    if (!session.field_extent(field, rows, cols))
    {
        die(std::string("field '") + field +
            "' unavailable (unknown field or 'init' not called)");
    }
    auto* m = mxCreateDoubleMatrix(rows, cols, mxREAL);
    session.field_get(field, mxGetPr(m));
    return m;
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

mxArray* do_param_names(const Session& session)
{
    const std::vector<std::string> names = session.param_names();
    auto* c = mxCreateCellMatrix(1, static_cast<mwSize>(names.size()));
    for (std::size_t i = 0; i < names.size(); ++i)
    {
        mxSetCell(c, static_cast<mwIndex>(i),
                  mxCreateString(names[i].c_str()));
    }
    return c;
}

mxArray* do_solution(const Session& session)
{
    auto* s = mxCreateStructMatrix(1, 1, 0, nullptr);
    for (const char* f : kFields)
    {
        struct_set_mat(s, f, field_matrix(session, f));
    }
    return s;
}

/// The per-iteration statistics as a 1 x iter struct array: one field per
/// column name, element r = iteration r (docs/plans/
/// MEX_IMPROVEMENTS_PLAN.md P3). Filled element by element, so the layout
/// is correct by construction (the old row-major table memcpy'd into a
/// column-major mxArray was transposed).
mxArray* do_stats(const Session& session)
{
    const int n_iter = session.stat_iter();
    const std::vector<std::string> names = session.stat_column_names();
    const int n_cols = static_cast<int>(names.size());
    std::vector<const char*> cnames;
    cnames.reserve(names.size());
    for (const std::string& n : names)
    {
        cnames.push_back(n.c_str());
    }
    auto* s =
        mxCreateStructMatrix(1, static_cast<mwSize>(n_iter), n_cols,
                              cnames.data());
    if (n_iter > 0)
    {
        // row-major table: buf[r * n_cols + c] = row r, column c
        std::vector<double> buf(static_cast<std::size_t>(n_iter) * n_cols,
                                0.0);
        session.stat_fill(buf.data(), n_iter, n_cols);
        for (int r = 0; r < n_iter; ++r)
        {
            for (int c = 0; c < n_cols; ++c)
            {
                const double v =
                    buf[static_cast<std::size_t>(r) * n_cols + c];
                mxSetFieldByNumber(s, static_cast<mwIndex>(r),
                                   static_cast<int>(c),
                                   mxCreateDoubleScalar(v));
            }
        }
    }
    return s;
}

mxArray* do_stat_col(const Session& session, const std::string& name)
{
    const int n = session.stat_iter();
    auto* m = mxCreateDoubleMatrix(n, 1, mxREAL);
    if (!session.stat_column(name, mxGetPr(m)))
    {
        mxDestroyArray(m);
        die("unknown statistics column: '" + name +
            "' (see 'stats' for the column names)");
    }
    return m;
}

/// Recursively walk a MATLAB options struct: field nesting mirrors the
/// dotted option names (e.g. opt.options.max_iter,
/// opt.qp.options.warm_start). Scalar leaves are written via
/// session.option_set; unknown option names, out-of-range values and
/// non-scalar, non-struct values raise a MATLAB warning (skipped, no
/// error) — partial structs are fine
/// (docs/plans/finished/MEX_IMPROVEMENTS_PLAN.md P5).
void set_option_struct(Session& session, const mxArray* s,
                       const std::string& prefix)
{
    using Status = ocp::inspect::SetResult::Status;
    const int nfields = mxGetNumberOfFields(s);
    for (int i = 0; i < nfields; ++i)
    {
        const char* fname = mxGetFieldNameByNumber(s, i);
        const mxArray* v = mxGetFieldByNumber(s, 0, i);
        const std::string path = prefix + fname;
        if (mxIsDouble(v) && !mxIsComplex(v) &&
            mxGetNumberOfElements(v) == 1)
        {
            const auto res = session.option_set(path, mxGetScalar(v));
            if (res.status == Status::kUnknownPath)
            {
                mexWarnMsgIdAndTxt("ocp_mex:UnknownOption",
                                   "unknown option: '%s' "
                                   "(skipped; see 'options')",
                                   path.c_str());
            }
            else if (!res.ok())
            {
                mexWarnMsgIdAndTxt("ocp_mex:OptionOutOfRange",
                                   "option '%s' = %g is outside the "
                                   "allowed range [%g, %g] (skipped)",
                                   path.c_str(), mxGetScalar(v), res.lo,
                                   res.hi);
            }
        }
        else if (mxIsStruct(v))
        {
            set_option_struct(session, v, path + ".");
        }
        else
        {
            mexWarnMsgIdAndTxt("ocp_mex:BadOptionValue",
                               "option '%s' must be a scalar double or a "
                               "struct (skipped)",
                               path.c_str());
        }
    }
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
        const bool ok = session.init(N);
        if (nlhs >= 1)
        {
            plhs[0] = mxCreateLogicalScalar(ok);
        }
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
        if (nrhs < 2)
        {
            die("opt_set: usage: 'opt_set' name value or 'opt_set' struct");
        }
        if (mxIsChar(prhs[1]))
        {
            const std::string name = arg_str(nrhs, prhs, 1);
            const double value = arg_scalar(nrhs, prhs, 2);
            const auto res = session.option_set(name, value);
            if (res.status ==
                ocp::inspect::SetResult::Status::kUnknownPath)
            {
                die("unknown option: '" + name + "' (see 'options')");
            }
            else if (!res.ok())
            {
                // Distinct id + message from the unknown-name error:
                // the option exists, the value violates its bounds.
                mexErrMsgIdAndTxt("ocp_mex:OptionOutOfRange",
                                  "option '%s' = %g is outside the "
                                  "allowed range [%g, %g]",
                                  name.c_str(), value, res.lo, res.hi);
            }
        }
        else if (mxIsStruct(prhs[1]))
        {
            set_option_struct(session, prhs[1], "");
        }
        else
        {
            die("opt_set: argument 2 must be an option name (char) or an "
                "options struct");
        }
        if (nlhs >= 1)
        {
            plhs[0] = mxCreateLogicalScalar(1);
        }
    }
    else if (cmd == "opt_dump")
    {
        mexPrintf("%s", session.option_dump_str().c_str());
    }
    else if (cmd == "param_names")
    {
        if (nlhs < 1)
        {
            die("param_names: assign the output, e.g.  pn = ocp_mex_" +
                std::string(ocp_mex::kName) + "('param_names')");
        }
        plhs[0] = do_param_names(session);
    }
    else if (cmd == "param_extent")
    {
        const std::string name = arg_str(nrhs, prhs, 1);
        if (nlhs < 1)
        {
            die("param_extent: assign the output, e.g.  e = ocp_mex_" +
                std::string(ocp_mex::kName) + "('param_extent', name)");
        }
        int rows = 0;
        int cols = 0;
        if (!session.param_extent(name, rows, cols))
        {
            die("unknown parameter: '" + name + "' (see 'param_names')");
        }
        auto* m = mxCreateDoubleMatrix(1, 2, mxREAL);
        double* p = mxGetPr(m);
        p[0] = static_cast<double>(rows);
        p[1] = static_cast<double>(cols);
        plhs[0] = m;
    }
    else if (cmd == "param_get")
    {
        const std::string name = arg_str(nrhs, prhs, 1);
        if (nlhs < 1)
        {
            die("param_get: assign the output, e.g.  v = ocp_mex_" +
                std::string(ocp_mex::kName) + "('param_get', name)");
        }
        int rows = 0;
        int cols = 0;
        if (!session.param_extent(name, rows, cols))
        {
            die("unknown parameter: '" + name + "' (see 'param_names')");
        }
        auto* m = mxCreateDoubleMatrix(rows, cols, mxREAL);
        session.param_get(name, mxGetPr(m));  // extent checked; cannot fail
        plhs[0] = m;
    }
    else if (cmd == "param_set")
    {
        const std::string name = arg_str(nrhs, prhs, 1);
        const mxArray* m = arg_matrix(nrhs, prhs, 2);
        int rows = 0;
        int cols = 0;
        if (!session.param_extent(name, rows, cols))
        {
            die("unknown parameter: '" + name + "' (see 'param_names')");
        }
        const int n = static_cast<int>(mxGetNumberOfElements(m));
        if (n != rows * cols)
        {
            die("param_set: '" + name + "' has " +
                std::to_string(rows * cols) + " element(s), got " +
                std::to_string(n) + " (see 'param_extent')");
        }
        const auto res = session.param_set(name, mxGetPr(m), n);
        if (res.status == ocp::inspect::SetResult::Status::kOutOfRange)
        {
            mexErrMsgIdAndTxt("ocp_mex:ParamOutOfRange",
                              "parameter '%s' = %g is outside the allowed "
                              "range [%g, %g]",
                              name.c_str(), mxGetPr(m)[0], res.lo, res.hi);
        }
        if (nlhs >= 1)
        {
            plhs[0] = mxCreateLogicalScalar(1);
        }
    }
    else if (cmd == "param_dump")
    {
        mexPrintf("%s", session.param_dump_str().c_str());
    }
    else if (cmd == "set")
    {
        const std::string field = arg_str(nrhs, prhs, 1);
        const mxArray* m = arg_matrix(nrhs, prhs, 2);
        const int rows = static_cast<int>(mxGetM(m));
        const int cols = static_cast<int>(mxGetN(m));
        if (!session.field_set(field, mxGetPr(m), rows, cols))
        {
            die("set '" + field + "' rejected (only 'x' and 'u' are "
                "writable; extents must match the initialized session)");
        }
        if (nlhs >= 1)
        {
            plhs[0] = mxCreateLogicalScalar(1);
        }
    }
    else if (cmd == "get")
    {
        const std::string field = arg_str(nrhs, prhs, 1);
        if (nlhs < 1)
        {
            die("get: assign the output, e.g.  x = ocp_mex_" +
                std::string(ocp_mex::kName) + "('get', 'x')");
        }
        plhs[0] = field_matrix(session, field.c_str());
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
        double cost = 0.0;
        const ocp::Status st = session.solve(cost);
        if (nlhs >= 2)
        {
            plhs[0] = mxCreateDoubleScalar(static_cast<double>(st));
            plhs[1] = mxCreateDoubleScalar(cost);
        }
        else if (nlhs == 1)
        {
            plhs[0] = mxCreateDoubleScalar(cost);
        }
    }
    else if (cmd == "stats")
    {
        if (nlhs >= 1)
        {
            plhs[0] = do_stats(session);
        }
        if (nlhs >= 2)
        {
            plhs[1] = mxCreateDoubleScalar(
                static_cast<double>(session.stat_status()));
        }
    }
    else if (cmd == "stat_col")
    {
        const std::string name = arg_str(nrhs, prhs, 1);
        if (nlhs < 1)
        {
            die("stat_col: assign the output, e.g.  col = ocp_mex_" +
                std::string(ocp_mex::kName) + "('stat_col', name)");
        }
        plhs[0] = do_stat_col(session, name);
    }
    else
    {
        die("unknown command: '" + cmd + "'\n\n" + usage());
    }
}

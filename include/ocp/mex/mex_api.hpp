// ocp/mex/mex_api.hpp
//
// MexSession: the host-independent core of the MATLAB MEX wrapper
// (docs/plans/MATLAB_MEX_PLAN.md). One instance holds the main objects of a
// (problem, solver) session as persistent state and exposes the typed
// command surface used by the MEX glue (mex/ocp_mex_impl.hpp):
//
//   - option discovery / get / set by name (inspect protocol)
//   - solution field access (x, u, multipliers, cost, status, N)
//   - a full solve()
//
// Pure C++ (no mex.h): unit-testable without MATLAB. Scalars cross the host
// boundary as double (MATLAB is double-only), so option access goes through
// the inspect protocol (get_double / set_double on the live solver;
// docs/plans/finished/OPTION_INSPECT_PLAN.md).

#pragma once

#include "../problem.hpp"
#include "../utils/inspect.hpp"

#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ocp
{
namespace mex
{

/// Session state of one (problem, solver) configuration.
///
/// Owns the problem instance, the solver and the working solution. In the
/// MEX, a function-local static owns one session per loaded binary; the
/// state persists for the lifetime of the host process. Option get/set by
/// name operate on the live solver through the inspect protocol, so a write
/// takes effect on the next solve() without a separate push step.
///
/// Solver contract (docs/plans/finished/OPTION_INSPECT_PLAN.md sec. 6): `Solver`
/// must provide
///   1. `static constexpr inspect(Visitor)` exposing its option children
///      (dotted leaf paths are the option names);
///   2. `Status solve(const Problem&, Solution<Problem, NH>&)`;
///   3. `const Stats& statistics() const` where `Stats` has `int iter`,
///      `Status status`, a `Row` alias, and the accessors `int n_rows()
///      const` and `const Row& row(int) const` (fixed-capacity FIFO ring,
///      docs/plans/finished/STATS_RING_PLAN.md); `Row` has a scalar-leaf `inspect`.
/// Any NLP driver meeting this contract drops into the session unchanged.
template <class Problem, class Solver, int NH = Eigen::Dynamic>
class MexSession
{
    using P = Problem;
    using S = typename P::scalar_t;
    // The session only needs the structural problem dimensions, which the
    // Problem base exposes directly; QpDim (solver-side stage layout) is
    // not a session dependency.
    using D = P;
    using Sol = Solution<Problem, NH>;

public:
    /// Compile-time problem dimensions plus the current horizon.
    struct Info
    {
        int nx = 0, nu = 0;
        int ng = 0, ne = 0, nl = 0;
        int ng_t = 0, ne_t = 0, nl_t = 0;
        int nbx = 0, nbu = 0, nbx_t = 0;
        bool fixed_initial_state = false;
        int N = 0;
    };

    /// `name` identifies the configuration (printed in MEX error messages).
    explicit MexSession(const char* name = "") : name_(name)
    {
    }

    // ---------------------------------------------------------------
    //  Initialization
    // ---------------------------------------------------------------

    /// (Re)initialize the working solution for horizon N: resizes `sol_`
    /// (new stages are zeroed, existing stages keep their warm-start
    /// values) and sets x[0] from the problem's initial_state() when the
    /// initial state is fixed. The solver resizes its workspaces on the
    /// next solve().
    bool init(int N)
    {
        if (N < 1)
        {
            return false;
        }
        sol_.resize(N);
        if constexpr (P::fixed_initial_state)
        {
            sol_.x[0] = problem_.initial_state();
        }
        return true;
    }

    // ---------------------------------------------------------------
    //  Options (by name, inspect protocol; docs/plans/finished/OPTION_INSPECT_PLAN.md)
    // ---------------------------------------------------------------

    /// All discoverable option names (dotted paths: options.*, qp.options.*,
    /// scaling.options.*).
    std::vector<std::string> option_names() const
    {
        return inspect::leaf_names<Solver>();
    }

    /// Read an option by name (int/bool coerced to double); false if the
    /// name is unknown.
    bool option_get(std::string_view name, double& out) const
    {
        return inspect::get_double(solver_, name, out);
    }

    /// Write an option by name (dotted names allowed, value coerced to the
    /// leaf's type), enforcing the field's declared [lo, hi] bounds.
    /// Writes the live solver option directly, so it takes effect on the
    /// next solve() without a separate push step. kOk on success,
    /// kOutOfRange (bounds in lo / hi) when the value violates them,
    /// kUnknownPath when the name is unknown or ends at a child.
    inspect::SetResult option_set(std::string_view name, double value)
    {
        return inspect::set_double(solver_, name, value);
    }

    /// Print a flat "name = value" snapshot of all options.
    void option_dump(std::FILE* f = stdout) const
    {
        inspect::dump(solver_, f);
    }

    /// Flat "name = value" snapshot of all options as a single string
    /// (same format as option_dump; the MEX glue prints it via mexPrintf).
    std::string option_dump_str() const
    {
        return inspect::dump_string(solver_);
    }

    // ---------------------------------------------------------------
    //  Parameters (by name, inspect protocol over the problem's
    //  duck-typed params() object; scalar and matrix leaves)
    // ---------------------------------------------------------------

    /// All discoverable parameter names (dotted paths into the problem's
    /// parameter object). Empty when the problem does not declare
    /// parameters (the base stub returns the leaf-free EmptyParams).
    std::vector<std::string> param_names() const
    {
        using ParamType = std::decay_t<decltype(problem_.params())>;
        return inspect::leaf_names<ParamType>();
    }

    /// Extent of the parameter leaf at dotted `name`: a matrix leaf
    /// reports its rows/cols, a scalar leaf reports 1 x 1. False if the
    /// name is unknown or the problem has no parameters.
    bool param_extent(std::string_view name, int& rows, int& cols) const
    {
        if (inspect::matrix_extent(problem_.params(), name, rows, cols))
        {
            return true;
        }
        double v = 0.0;
        if (inspect::get_double(problem_.params(), name, v))
        {
            rows = 1;
            cols = 1;
            return true;
        }
        rows = 0;
        cols = 0;
        return false;
    }

    /// Read the parameter leaf at dotted `name` double-flattened,
    /// column-major: `out` must hold rows*cols doubles (rows/cols from
    /// param_extent). False if the name is unknown.
    bool param_get(std::string_view name, double* out) const
    {
        int rows = 0;
        int cols = 0;
        if (inspect::matrix_extent(problem_.params(), name, rows, cols))
        {
            return inspect::get_matrix_double(
                problem_.params(), name, out, rows, cols);
        }
        double v = 0.0;
        if (inspect::get_double(problem_.params(), name, v))
        {
            out[0] = v;
            return true;
        }
        return false;
    }

    /// Write the double-flattened, column-major `in` (n elements) into
    /// the parameter leaf at dotted `name`: n must equal rows*cols for a
    /// matrix leaf and 1 for a scalar leaf, where the leaf's declared
    /// [lo, hi] bounds are enforced. kOk on success; kOutOfRange (bounds
    /// in lo / hi) for an out-of-range scalar; kUnknownPath for an
    /// unknown name or a wrong element count (the leaf stays untouched).
    inspect::SetResult param_set(std::string_view name, const double* in,
                                 int n)
    {
        inspect::SetResult r;
        int rows = 0;
        int cols = 0;
        if (inspect::matrix_extent(problem_.params(), name, rows, cols))
        {
            if (n == rows * cols &&
                inspect::set_matrix_double(problem_.params(), name, in, n))
            {
                r.status = inspect::SetResult::Status::kOk;
            }
            return r;
        }
        if (n != 1)
        {
            return r;
        }
        return inspect::set_double(problem_.params(), name, in[0]);
    }

    /// Print a flat "name = value" snapshot of all parameters (matrix
    /// leaves flattened column-major onto one line).
    void param_dump(std::FILE* f = stdout) const
    {
        inspect::dump(problem_.params(), f);
    }

    /// Flat "name = value" snapshot of all parameters as a single string
    /// (same format as param_dump; the MEX glue prints it via mexPrintf).
    /// Empty when the problem has no parameters.
    std::string param_dump_str() const
    {
        return inspect::dump_string(problem_.params());
    }

    // ---------------------------------------------------------------
    //  Solution fields (names = Solution member names, verbatim)
    // ---------------------------------------------------------------

    /// Extent of a solution field in the MATLAB (column-major) layout:
    /// x / lambda_box_state: dim x (N+1); u and all other stage fields:
    /// dim x N; terminal lambdas: dim x 1; cost_value / status / N: 1 x 1.
    /// Returns false for unknown field names and, for all stage/terminal
    /// fields, before init() (N < 1); the three scalars are always 1 x 1.
    bool field_extent(std::string_view field, int& rows, int& cols) const
    {
        const FieldId f = parse_field(field);
        const int N = sol_.N;
        switch (f)
        {
        case FieldId::kCost:
        case FieldId::kStatus:
        case FieldId::kN:
            rows = 1;
            cols = 1;
            return true;
        case FieldId::kX:
            rows = D::nx;
            cols = N + 1;
            break;
        case FieldId::kU:
            rows = D::nu;
            cols = N;
            break;
        case FieldId::kLambdaDyn:
            rows = D::nx;
            cols = N;
            break;
        case FieldId::kLambdaIneqStage:
            rows = D::ng;
            cols = N;
            break;
        case FieldId::kLambdaEqStage:
            rows = D::ne;
            cols = N;
            break;
        case FieldId::kLambdaLinStage:
            rows = D::nl;
            cols = N;
            break;
        case FieldId::kLambdaIneqTerm:
            rows = D::ng_t;
            cols = 1;
            break;
        case FieldId::kLambdaEqTerm:
            rows = D::ne_t;
            cols = 1;
            break;
        case FieldId::kLambdaLinTerm:
            rows = D::nl_t;
            cols = 1;
            break;
        case FieldId::kLambdaBoxState:
            rows = 2 * D::nbx;
            cols = N + 1;
            break;
        case FieldId::kLambdaBoxControl:
            rows = 2 * D::nbu;
            cols = N;
            break;
        case FieldId::kUnknown:
            return false;
        }
        return N >= 1;
    }

    /// Read a solution field into `out` (column-major, column k = stage k;
    /// extents as in field_extent). `out` must hold rows*cols doubles;
    /// zero-extent fields write nothing. False if the field name is unknown
    /// or the session is not initialized (stage/terminal fields only).
    bool field_get(std::string_view field, double* out) const
    {
        int rows = 0;
        int cols = 0;
        if (!field_extent(field, rows, cols))
        {
            return false;
        }
        switch (parse_field(field))
        {
        case FieldId::kX:
            traj_fill(sol_.x, rows, cols, out);
            return true;
        case FieldId::kU:
            traj_fill(sol_.u, rows, cols, out);
            return true;
        case FieldId::kLambdaDyn:
            traj_fill(sol_.lambda_dyn, rows, cols, out);
            return true;
        case FieldId::kLambdaIneqStage:
            traj_fill(sol_.lambda_ineq_stage, rows, cols, out);
            return true;
        case FieldId::kLambdaEqStage:
            traj_fill(sol_.lambda_eq_stage, rows, cols, out);
            return true;
        case FieldId::kLambdaLinStage:
            traj_fill(sol_.lambda_lin_stage, rows, cols, out);
            return true;
        case FieldId::kLambdaIneqTerm:
            term_fill(sol_.lambda_ineq_term, rows, out);
            return true;
        case FieldId::kLambdaEqTerm:
            term_fill(sol_.lambda_eq_term, rows, out);
            return true;
        case FieldId::kLambdaLinTerm:
            term_fill(sol_.lambda_lin_term, rows, out);
            return true;
        case FieldId::kLambdaBoxState:
            traj_fill(sol_.lambda_box_state, rows, cols, out);
            return true;
        case FieldId::kLambdaBoxControl:
            traj_fill(sol_.lambda_box_control, rows, cols, out);
            return true;
        case FieldId::kCost:
            out[0] = static_cast<double>(sol_.cost_value);
            return true;
        case FieldId::kStatus:
            out[0] = static_cast<double>(sol_.status);
            return true;
        case FieldId::kN:
            out[0] = static_cast<double>(sol_.N);
            return true;
        case FieldId::kUnknown:
            return false;
        default:  // keeps GCC's flow analysis happy on scoped enums
            return false;
        }
    }

    /// Write a solution field (warm start). Only `x` (nx x (N+1)) and
    /// `u` (nu x N) are writable; `in` must be column-major (column k =
    /// stage k) with exactly rows*cols entries. False on unknown field,
    /// wrong extents, or a session that was not init()'d.
    bool field_set(std::string_view field, const double* in, int rows,
                   int cols)
    {
        const int N = sol_.N;
        if (N < 1)
        {
            return false;
        }
        if (field == "x" && rows == D::nx && cols == N + 1)
        {
            // column-major: column k = stage k
            for (int k = 0; k <= N; ++k)
            {
                for (int i = 0; i < D::nx; ++i)
                {
                    sol_.x[k](i) = static_cast<S>(
                        in[static_cast<std::size_t>(k) * rows + i]);
                }
            }
            return true;
        }
        if (field == "u" && rows == D::nu && cols == N)
        {
            // column-major: column k = stage k
            for (int k = 0; k < N; ++k)
            {
                for (int i = 0; i < D::nu; ++i)
                {
                    sol_.u[k](i) = static_cast<S>(
                        in[static_cast<std::size_t>(k) * rows + i]);
                }
            }
            return true;
        }
        return false;
    }

    // ---------------------------------------------------------------
    //  Solve
    // ---------------------------------------------------------------

    /// Run a full solve on the current working solution. `cost` receives
    /// sol_.cost_value (includes the soft-slack penalty). Returns
    /// kNotInitialized if init() was not called yet.
    Status solve(double& cost)
    {
        if (sol_.N < 1)
        {
            return Status::kNotInitialized;
        }
        const Status st = solver_.solve(problem_, sol_);
        cost = sol_.cost_value;
        return st;
    }

    // ---------------------------------------------------------------
    //  Statistics (named columns via the inspect protocol;
    //  docs/plans/finished/OPTION_INSPECT_PLAN.md sec. 5)
    // ---------------------------------------------------------------

    int stat_iter() const { return solver_.statistics().iter; }

    /// Number of rows currently stored in the statistics ring (the table
    /// row count; equals stat_iter() unless the ring wrapped).
    int stat_nrows() const { return solver_.statistics().n_rows(); }

    int stat_status() const
    {
        return static_cast<int>(solver_.statistics().status);
    }

    /// Column names of the per-iteration statistics table (the row struct's
    /// leaf field names, in declaration order).
    std::vector<std::string> stat_column_names() const
    {
        return inspect::stat_column_names(solver_.statistics());
    }

    /// Fill the per-iteration statistics table into `out`, row-major:
    /// out[r * cols + c] is row r, column c (column order =
    /// stat_column_names()). One row per stored statistics row (oldest
    /// first; rows carry their own iteration number in the `iter` column).
    /// `rows` must be stat_nrows() and `cols` stat_column_names().size();
    /// `out` must hold rows*cols doubles.
    void stat_fill(double* out, int rows, int cols) const
    {
        const auto& st = solver_.statistics();
        for (int r = 0; r < rows; ++r)
        {
            inspect::fill_row(st.row(r),
                              out + static_cast<std::size_t>(r) * cols);
        }
    }

    /// One named statistics column: `out[i]` is stored row i's value
    /// (int/bool coerced to double). `out` must hold stat_nrows() doubles.
    /// False if the column name is unknown.
    bool stat_column(std::string_view name, double* out) const
    {
        return inspect::stat_column(solver_.statistics(), name, out);
    }

    // ---------------------------------------------------------------
    //  Introspection
    // ---------------------------------------------------------------

    Info info() const
    {
        Info i;
        i.nx = D::nx;
        i.nu = D::nu;
        i.ng = D::ng;
        i.ne = D::ne;
        i.nl = D::nl;
        i.ng_t = D::ng_t;
        i.ne_t = D::ne_t;
        i.nl_t = D::nl_t;
        i.nbx = D::nbx;
        i.nbu = D::nbu;
        i.nbx_t = D::nbx_t;
        i.fixed_initial_state = P::fixed_initial_state;
        i.N = sol_.N;
        return i;
    }

    /// The configuration name given at construction.
    std::string name() const { return name_; }

    const Sol& solution() const { return sol_; }
    Sol& solution() { return sol_; }
    const Solver& solver() const { return solver_; }

private:
    /// The solution fields by name (Solution member names, verbatim).
    enum class FieldId
    {
        kUnknown, kX, kU, kLambdaDyn, kLambdaIneqStage, kLambdaEqStage,
        kLambdaLinStage, kLambdaIneqTerm, kLambdaEqTerm, kLambdaLinTerm,
        kLambdaBoxState, kLambdaBoxControl, kCost, kStatus, kN
    };

    static FieldId parse_field(std::string_view field)
    {
        if (field == "x")
        {
            return FieldId::kX;
        }
        if (field == "u")
        {
            return FieldId::kU;
        }
        if (field == "lambda_dyn")
        {
            return FieldId::kLambdaDyn;
        }
        if (field == "lambda_ineq_stage")
        {
            return FieldId::kLambdaIneqStage;
        }
        if (field == "lambda_eq_stage")
        {
            return FieldId::kLambdaEqStage;
        }
        if (field == "lambda_lin_stage")
        {
            return FieldId::kLambdaLinStage;
        }
        if (field == "lambda_ineq_term")
        {
            return FieldId::kLambdaIneqTerm;
        }
        if (field == "lambda_eq_term")
        {
            return FieldId::kLambdaEqTerm;
        }
        if (field == "lambda_lin_term")
        {
            return FieldId::kLambdaLinTerm;
        }
        if (field == "lambda_box_state")
        {
            return FieldId::kLambdaBoxState;
        }
        if (field == "lambda_box_control")
        {
            return FieldId::kLambdaBoxControl;
        }
        if (field == "cost_value")
        {
            return FieldId::kCost;
        }
        if (field == "status")
        {
            return FieldId::kStatus;
        }
        if (field == "N")
        {
            return FieldId::kN;
        }
        return FieldId::kUnknown;
    }

    /// Write a stage-indexed sequence of fixed-size vectors into `out`,
    /// column-major with column k = stage k (MATLAB layout); zero-extent
    /// fields write nothing.
    template <class Seq>
    static void traj_fill(const Seq& traj, int rows, int cols, double* out)
    {
        for (int k = 0; k < cols; ++k)
        {
            for (int i = 0; i < rows; ++i)
            {
                out[static_cast<std::size_t>(k) * rows + i] =
                    static_cast<double>(traj[k](i));
            }
        }
    }

    /// Write a terminal (single-stage) vector into `out`.
    template <class Vec>
    static void term_fill(const Vec& v, int rows, double* out)
    {
        for (int i = 0; i < rows; ++i)
        {
            out[i] = static_cast<double>(v(i));
        }
    }


    Problem problem_;
    Solver solver_;
    Sol sol_;
    const char* name_ = "";
};

}  // namespace mex
}  // namespace ocp

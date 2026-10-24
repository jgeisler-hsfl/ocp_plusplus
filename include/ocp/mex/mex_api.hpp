// ocp/mex/mex_api.hpp
//
// MexSession: the host-independent core of the MATLAB MEX wrapper
// (docs/plans/MATLAB_MEX_PLAN.md). One instance holds the main objects of a
// (problem, solver) session as persistent state and exposes the typed
// command surface used by the MEX glue (mex/ocp_mex_impl.hpp):
//
//   - option discovery / get / set by name (ValueRegistry, incl. glob.*)
//   - solution field access (x, u, multipliers, cost, status, N)
//   - a full solve()
//
// Pure C++ (no mex.h): unit-testable without MATLAB. Scalars cross the host
// boundary as double (MATLAB is double-only), so option access goes through
// ValueRegistry::get_as_double / set_from_double.

#pragma once

#include "../problem.hpp"
#include "../solvers/acados/acados.hpp"

#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace ocp
{
namespace mex
{

/// Session state of one (problem, solver) configuration.
///
/// Owns the problem instance, the driver options (source of truth), the
/// solver and the working solution. In the MEX, a function-local static
/// owns one session per loaded binary; the state persists for the lifetime
/// of the host process. `Solver` is expected to be an NLP driver of type
/// `AcadosSolver<Problem, NH>` (v1); option updates are pushed into it via
/// `apply_options`.
template <class Problem, int NH = Eigen::Dynamic,
          class Solver = AcadosSolver<Problem, NH>>
class MexSession
{
    using P = Problem;
    using S = typename P::scalar_t;
    using D = QpDim<P>;
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
    //  Options (by name, docs/plans/DUAL_ACCESS_PLAN.md)
    // ---------------------------------------------------------------

    /// All discoverable option names (flat + dotted `glob.*` leaves).
    std::vector<std::string> option_names() const
    {
        return opts_.reg.names();
    }

    /// Read an option by name (int/bool coerced to double); false if the
    /// name is unknown.
    bool option_get(std::string_view name, double& out) const
    {
        return opts_.reg.get_as_double(name, out);
    }

    /// Write an option by name (dotted names allowed, value coerced to the
    /// entry's type), then push the updated options into the solver (the
    /// solver holds its own copy; glob.* is re-synced at the next solve()).
    bool option_set(std::string_view name, double value)
    {
        if (!opts_.reg.set_from_double(name, value))
        {
            return false;
        }
        solver_.apply_options(opts_);
        return true;
    }

    /// Print a flat "name = value" snapshot of all options.
    void option_dump(std::FILE* f = stdout) const
    {
        opts_.reg.dump(f);
    }

    // ---------------------------------------------------------------
    //  Solution fields (names = Solution member names, verbatim)
    // ---------------------------------------------------------------

    /// Read a solution field into `out`, column-major with one column per
    /// stage (column k = stage k). Extents on success:
    /// x / lambda_box_state: dim x (N+1); u and all other stage fields:
    /// dim x N; terminal lambdas: dim x 1; cost_value / status / N: 1 x 1.
    /// `out` is empty with rows = 0 for zero-extent fields. False if the
    /// field name is unknown.
    bool field_get(std::string_view field, std::vector<double>& out,
                   int& rows, int& cols) const
    {
        const int N = sol_.N;
        if (field == "x")
        {
            return traj_get(sol_.x, D::nx, N + 1, out, rows, cols);
        }
        if (field == "u")
        {
            return traj_get(sol_.u, D::nu, N, out, rows, cols);
        }
        if (field == "lambda_dyn")
        {
            return traj_get(sol_.lambda_dyn, D::nx, N, out, rows, cols);
        }
        if (field == "lambda_ineq_stage")
        {
            return traj_get(sol_.lambda_ineq_stage, D::ng, N, out, rows, cols);
        }
        if (field == "lambda_eq_stage")
        {
            return traj_get(sol_.lambda_eq_stage, D::ne, N, out, rows, cols);
        }
        if (field == "lambda_lin_stage")
        {
            return traj_get(sol_.lambda_lin_stage, D::nl, N, out, rows, cols);
        }
        if (field == "lambda_ineq_term")
        {
            return term_get(sol_.lambda_ineq_term, D::ng_t, out, rows, cols);
        }
        if (field == "lambda_eq_term")
        {
            return term_get(sol_.lambda_eq_term, D::ne_t, out, rows, cols);
        }
        if (field == "lambda_lin_term")
        {
            return term_get(sol_.lambda_lin_term, D::nl_t, out, rows, cols);
        }
        if (field == "lambda_box_state")
        {
            return traj_get(sol_.lambda_box_state, 2 * D::nbx, N + 1,
                            out, rows, cols);
        }
        if (field == "lambda_box_control")
        {
            return traj_get(sol_.lambda_box_control, 2 * D::nbu, N,
                            out, rows, cols);
        }
        if (field == "cost_value")
        {
            return scalar_get(static_cast<double>(sol_.cost_value),
                              out, rows, cols);
        }
        if (field == "status")
        {
            return scalar_get(static_cast<double>(sol_.status),
                              out, rows, cols);
        }
        if (field == "N")
        {
            return scalar_get(static_cast<double>(N), out, rows, cols);
        }
        return false;
    }

    /// Write a solution field (warm start). Only `x` (dim x (N+1)) and
    /// `u` (dim x N) are writable; `in` must be column-major (column k =
    /// stage k) with exactly rows*cols entries. False on unknown field,
    /// wrong extents, or a session that was not init()'d.
    bool field_set(std::string_view field, const std::vector<double>& in,
                   int rows, int cols)
    {
        const int N = sol_.N;
        if (N < 1)
        {
            return false;
        }
        if (field == "x" && rows == D::nx && cols == N + 1 &&
            in.size() == static_cast<std::size_t>(rows * cols))
        {
            // column-major: column k = stage k
            for (int k = 0; k <= N; ++k)
            {
                for (int i = 0; i < D::nx; ++i)
                {
                    sol_.x[k](i) =
                        static_cast<S>(in[static_cast<std::size_t>(k) * rows
                                           + i]);
                }
            }
            return true;
        }
        if (field == "u" && rows == D::nu && cols == N &&
            in.size() == static_cast<std::size_t>(rows * cols))
        {
            // column-major: column k = stage k
            for (int k = 0; k < N; ++k)
            {
                for (int i = 0; i < D::nu; ++i)
                {
                    sol_.u[k](i) =
                        static_cast<S>(in[static_cast<std::size_t>(k) * rows
                                           + i]);
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
    /// sol_.cost_value (includes the soft-slack penalty). Returns kAborted
    /// if init() was not called yet.
    Status solve(double& cost)
    {
        if (sol_.N < 1)
        {
            return Status::kAborted;
        }
        const Status st = solver_.solve(problem_, sol_);
        cost = sol_.cost_value;
        return st;
    }

    // ---------------------------------------------------------------
    //  Statistics (raw fields; statistics dual access is a follow-up)
    // ---------------------------------------------------------------

    int stat_iter() const { return solver_.statistics().iter; }

    int stat_status() const
    {
        return static_cast<int>(solver_.statistics().status);
    }

    /// The per-iteration rows (10 columns: res_stat, res_eq, res_ineq,
    /// res_comp, qp_status, qp_iter, step_norm, alpha, funnel_iter_type,
    /// funnel_width), one row per recorded iteration.
    bool stat_rows(std::vector<double>& out, int& rows, int& cols) const
    {
        const AcadosStatistics& stat = solver_.statistics();
        rows = stat.iter;
        cols = 10;
        out.assign(static_cast<std::size_t>(rows * cols), 0.0);
        for (int i = 0; i < stat.iter; ++i)
        {
            const SqpIteration& r = stat.rows[i];
            const std::size_t o = static_cast<std::size_t>(i) * cols;
            out[o + 0] = r.res_stat;
            out[o + 1] = r.res_eq;
            out[o + 2] = r.res_ineq;
            out[o + 3] = r.res_comp;
            out[o + 4] = static_cast<double>(r.qp_status);
            out[o + 5] = static_cast<double>(r.qp_iter);
            out[o + 6] = r.step_norm;
            out[o + 7] = r.alpha;
            out[o + 8] = static_cast<double>(r.funnel_iter_type);
            out[o + 9] = r.funnel_width;
        }
        return true;
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
    const AcadosOptions& options() const { return opts_; }
    const Solver& solver() const { return solver_; }

private:
    /// Flatten a stage-indexed sequence of fixed-size vectors into `out`,
    /// column-major with column k = stage k (MATLAB layout); zero-extent
    /// fields yield an empty vector.
    template <class Seq>
    static bool traj_get(const Seq& traj, int dim, int n_stages,
                          std::vector<double>& out, int& rows, int& cols)
    {
        rows = dim;
        cols = n_stages;
        out.assign(static_cast<std::size_t>(rows * cols), 0.0);
        for (int k = 0; k < n_stages; ++k)
        {
            for (int i = 0; i < dim; ++i)
            {
                out[static_cast<std::size_t>(k) * dim + i] =
                    static_cast<double>(traj[k](i));
            }
        }
        return true;
    }

    /// Flatten a terminal (single-stage) vector.
    template <class T>
    static bool term_get(const T& v, int dim,
                         std::vector<double>& out, int& rows, int& cols)
    {
        rows = dim;
        cols = 1;
        out.assign(static_cast<std::size_t>(dim), 0.0);
        for (int i = 0; i < dim; ++i)
        {
            out[i] = static_cast<double>(v(i));
        }
        return true;
    }

    static bool scalar_get(double value, std::vector<double>& out,
                           int& rows, int& cols)
    {
        rows = 1;
        cols = 1;
        out.assign(1, value);
        return true;
    }

    Problem problem_;
    AcadosOptions opts_;
    Solver solver_;
    Sol sol_;
    const char* name_ = "";
};

}  // namespace mex
}  // namespace ocp

// Phase 2 sub-step 2e test: AcadosOptions, SqpIteration, AcadosStatistics,
// print_sqp_iteration from include/ocp/solvers/acados/acados.hpp.
//
// Checks (SQP_PHASE2_PLAN.md sec. 2e):
//   - AcadosOptions defaults match the acados defaults verified at 2e
//     (ocp_nlp_common.c:1215,1276,1282-1287, ocp_nlp_sqp.c:114-117,
//     GlobOptions per ocp_nlp_globalization_common.h:82-90)
//   - AcadosStatistics::record appends rows and increments the counter
//   - AcadosStatistics::write_csv writes the planned header and one row per
//     iteration with all nine values round-tripping through the file
//   - print_sqp_iteration prints the header at iterations 0 and 10 but not
//     in between (acados print_iteration, ocp_nlp_sqp.c:449-466)

#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "ocp/utils/inspect.hpp"
#include "ocp/solvers/acados/acados.hpp"

#include "double_integrator/double_integrator.hpp"

namespace
{

using namespace ocp;

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

void check_close(double a, double b, const std::string& msg)
{
    if (std::fabs(a - b) > 1e-12)
    {
        std::fprintf(stderr, "FAIL: %s (%.12e vs %.12e)\n", msg.c_str(), a, b);
        ++failures;
    }
}

// ---------------------------------------------------------------------
// AcadosOptions defaults
// ---------------------------------------------------------------------

void test_options_defaults()
{
    AcadosOptions o;

    check(o.max_iter == 20, "max_iter default 20");
    check_close(o.tol_stat, 1e-8, "tol_stat default 1e-8");
    check_close(o.tol_eq, 1e-8, "tol_eq default 1e-8");
    check_close(o.tol_ineq, 1e-8, "tol_ineq default 1e-8");
    check_close(o.tol_comp, 1e-8, "tol_comp default 1e-8");
    check_close(o.tol_min_step_norm, 1e-12, "tol_min_step_norm default 1e-12");
    check_close(o.tol_unbounded, -1e10, "tol_unbounded default -1e10");
    check(o.compute_hess_dyn, "compute_hess_dyn default true");
    check(o.compute_hess_constr, "compute_hess_constr default true");
    check_close(o.levenberg_marquardt, 0.0, "levenberg_marquardt default 0");
    check(!o.with_adaptive_lm, "with_adaptive_lm default false");
    check(o.print_level == 0, "print_level default 0");
    check(!o.warm_start_first_qp, "warm_start_first_qp default false");
    check(o.qp_mode == QpMode::kStaged, "qp_mode default kStaged");
    check(o.qp_cond_N == 0, "qp_cond_N default 0 (identity fast path)");
    check(!o.eval_residual_at_max_iter,
          "eval_residual_at_max_iter default false");
    check_close(o.timeout_max_time, 0.0, "timeout_max_time default 0");
    check_close(o.tau_min, 1e-16, "tau_min default 1e-16");

    // embedded GlobOptions defaults (ocp_nlp_globalization_common.h)
    check_close(o.glob.alpha_min, 0.05, "glob.alpha_min default 0.05");
    check_close(o.glob.alpha_reduction, 0.7, "glob.alpha_reduction default 0.7");
    check_close(o.glob.eps_sufficient_descent, 1e-4,
                "glob.eps_sufficient_descent default 1e-4");
    check(!o.glob.full_step_dual, "glob.full_step_dual default false");

    // the 5 options removed from AcadosOptions now live on the owning
    // components (docs/plans/OPTION_INSPECT_PLAN.md sec. 2.1)
    HpipmOptions h;
    check(h.warm_start == 0, "HpipmOptions warm_start default 0");
    QpScaler<DoubleIntegrator>::Opts so;
    check(!so.scale_objective, "QpScaler::Opts scale_objective default false");
    check(!so.scale_constraints,
          "QpScaler::Opts scale_constraints default false");
    check_close(so.ub_max_abs_eig, 1e5,
                "QpScaler::Opts ub_max_abs_eig default 1e5");
    check_close(so.lb_norm_inf_grad_obj, 1e-4,
                "QpScaler::Opts lb_norm_inf_grad_obj default 1e-4");
}

// ---------------------------------------------------------------------
// AcadosOptions by-name access (inspect protocol;
// docs/plans/OPTION_INSPECT_PLAN.md)
// ---------------------------------------------------------------------

void test_options_inspect()
{
    AcadosOptions o;

    // set via name (double coercion, host-style path) -> read the member
    check(inspect::set_double(o, "max_iter", 40.0).ok(), "set max_iter");
    check(o.max_iter == 40, "set wrote max_iter");
    check(inspect::set_double(o, "tau_min", 2e-16).ok(), "set tau_min");
    check_close(o.tau_min, 2e-16, "set wrote tau_min");
    check(inspect::set_double(o, "print_level", 1.0).ok(), "set int from double");
    check(o.print_level == 1, "set wrote the int member");

    // set the member -> read via name
    o.compute_hess_dyn = false;
    double d = 0.0;
    check(inspect::get_double(o, "print_level", d) && d == 1.0,
          "get print_level (int coerced to double)");
    check(inspect::get_double(o, "compute_hess_dyn", d) && d == 0.0,
          "get bool as 0.0");
    check(inspect::get_double(o, "glob.alpha_min", d) && d == 0.05,
          "get glob.alpha_min (dotted)");

    // strict typed access
    int i = 0;
    check(inspect::get_field(o, "max_iter", i) && i == 40, "typed get int");
    check(inspect::set_field(o, "tau_min", 3e-16).ok(), "typed set double");
    check_close(o.tau_min, 3e-16, "typed set wrote tau_min");

    // unknown names
    check(!inspect::get_double(o, "i_max", d), "get unknown name");
    check(!inspect::set_double(o, "nope", 1.0), "set unknown name");
    check(!inspect::get_double(o, "glob.alpha_min.x", d),
          "get unknown dotted tail");

    // names(): 28 flat leaves + 5 glob leaves
    const std::vector<std::string> names =
        inspect::leaf_names<AcadosOptions>();
    check(names.size() == 33, "leaf_names: 28 flat + 5 glob leaves");
    bool has_glob_alpha_min = false;
    bool has_glob_use_soc = false;
    for (const std::string& n : names)
    {
        has_glob_alpha_min = has_glob_alpha_min || (n == "glob.alpha_min");
        has_glob_use_soc = has_glob_use_soc || (n == "glob.use_soc");
    }
    check(has_glob_alpha_min && has_glob_use_soc,
          "leaf_names: dotted glob leaves present");

    // copy safety: sets through the copy (flat *and* dotted) must not
    // touch the original
    AcadosOptions o2 = o;
    check(inspect::set_double(o2, "max_iter", 99.0).ok(), "copy: set flat name");
    check(o2.max_iter == 99 && o.max_iter == 40,
          "copy: flat set left the original untouched");
    check(inspect::set_double(o2, "glob.alpha_min", 0.11).ok(),
          "copy: set dotted name");
    check_close(o2.glob.alpha_min, 0.11, "copy: dotted set wrote the copy");
    check_close(o.glob.alpha_min, 0.05, "copy: dotted set left the original");
    double ar = 0.0;
    check(inspect::get_double(o2, "glob.alpha_reduction", ar) && ar == 0.7,
          "copy: get dotted name");
}

// ---------------------------------------------------------------------
// Statistics table access (inspect protocol;
// docs/plans/OPTION_INSPECT_PLAN.md sec. 5)
// ---------------------------------------------------------------------

void test_stat_table()
{
    AcadosStatistics stat;
    SqpIteration r0;
    r0.res_stat = 1.0;
    r0.qp_status = 1;
    r0.qp_iter = 4;
    r0.step_norm = 0.5;
    r0.alpha = 0.7;
    stat.record(r0);
    SqpIteration r1;
    r1.res_stat = 1e-3;
    r1.funnel_iter_type = 2;
    r1.funnel_width = 0.25;
    stat.record(r1);

    // column names: the SqpIteration leaves in declaration order
    const std::vector<std::string> cols = inspect::stat_column_names(stat);
    check(cols.size() == 11, "stat_column_names: 11 columns");
    check(cols[0] == "res_stat" && cols[4] == "qp_status" &&
          cols[9] == "funnel_width" && cols[10] == "cond_time",
          "stat_column_names order");

    // one named column, int coerced to double
    std::vector<double> qp_iter(2, 0.0);
    check(inspect::stat_column(stat, "qp_iter", qp_iter.data()),
          "stat_column qp_iter");
    check(qp_iter[0] == 4.0 && qp_iter[1] == 0.0,
          "stat_column qp_iter values");
    check(!inspect::stat_column(stat, "res_statx", qp_iter.data()),
          "stat_column unknown name");
    AcadosStatistics empty;
    check(!inspect::stat_column(empty, "res_statx", qp_iter.data()),
          "stat_column unknown name on zero-row table");

    // full table: row-major rows x cols
    std::vector<double> tab(2 * 11, 0.0);
    int n_rows = 0, n_cols = 0;
    check(inspect::stat_table(stat, tab.data(), n_rows, n_cols) &&
          n_rows == 2 && n_cols == 11,
          "stat_table extents");
    check(tab[0] == 1.0 && tab[4] == 1.0,
          "stat_table row 0 (res_stat, qp_status)");
    check(tab[11 + 0] == 1e-3 && tab[11 + 8] == 2.0,
          "stat_table row 1 (res_stat, funnel_iter_type)");
}

// ---------------------------------------------------------------------
// AcadosStatistics::record
// ---------------------------------------------------------------------

void test_statistics_record()
{
    AcadosStatistics stat;
    check(stat.iter == 0, "initial iter is 0");
    check(stat.rows.empty(), "initially no rows");
    check(stat.status == Status::kUnset, "initial status kUnset");

    SqpIteration r0;
    r0.res_stat = 1.0;
    r0.res_eq = 2.0;
    r0.res_ineq = 3.0;
    r0.res_comp = 4.0;
    r0.qp_status = 1;
    r0.qp_iter = 2;
    r0.step_norm = 0.5;
    r0.alpha = 0.7;
    stat.record(r0);

    SqpIteration r1;
    r1.res_stat = 1e-3;
    r1.alpha = 1.0;
    stat.record(r1);

    check(stat.iter == 2, "iter counts recorded rows");
    check(stat.rows.size() == 2, "rows size");
    check_close(stat.rows[0].res_stat, 1.0, "row 0 res_stat");
    check(stat.rows[0].qp_iter == 2, "row 0 qp_iter");
    check_close(stat.rows[1].res_stat, 1e-3, "row 1 res_stat");
    check(stat.rows[1].qp_status == 0, "row 1 default qp_status");
    check(stat.rows[0].cond_time == 0.0, "row 0 default cond_time 0");
}

// ---------------------------------------------------------------------
// AcadosStatistics::write_csv
// ---------------------------------------------------------------------

void test_statistics_csv()
{
    const char* path = "sqp_stats_2e_test.csv";

    AcadosStatistics stat;
    SqpIteration r;
    r.res_stat = 1.5e-3;
    r.res_eq = 2.5e-4;
    r.res_ineq = 3.5e-5;
    r.res_comp = 4.5e-6;
    r.qp_status = 1;
    r.qp_iter = 4;
    r.step_norm = 1e-2;
    r.alpha = 0.7;
    stat.record(r);
    stat.record(r);
    stat.status = Status::kSolved;
    stat.write_csv(path);

    std::ifstream in(path);
    check(in.good(), "csv file opened");
    if (in.good())
    {
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(in, line))
        {
            lines.push_back(line);
        }
        check(lines.size() == 3, "csv has header + 2 rows");
        check(lines[0] == "iter,res_stat,res_eq,res_ineq,res_comp,"
                          "qp_status,qp_iter,step_norm,alpha,"
                          "funnel_iter_type,funnel_width,cond_time",
              "csv header");
        for (int i = 0; i < 2; ++i)
        {
            int iter_i = -1, qp_status = -1, qp_iter = -1, ftype = 0;
            double rs = 0.0, re = 0.0, ri = 0.0, rc = 0.0, sn = 0.0,
                   al = 0.0, fw = 0.0, ct = 0.0;
            const int n = std::sscanf(lines[i + 1].c_str(),
                                      "%d,%lf,%lf,%lf,%lf,%d,%d,%lf,%lf,"
                                      "%d,%lf,%lf",
                                      &iter_i, &rs, &re, &ri, &rc,
                                      &qp_status, &qp_iter, &sn, &al,
                                      &ftype, &fw, &ct);
            check(n == 12, "csv row parsed");
            check(iter_i == i, "csv iter column");
            check_close(rs, 1.5e-3, "csv res_stat");
            check_close(re, 2.5e-4, "csv res_eq");
            check_close(ri, 3.5e-5, "csv res_ineq");
            check_close(rc, 4.5e-6, "csv res_comp");
            check(qp_status == 1, "csv qp_status");
            check(qp_iter == 4, "csv qp_iter");
            check_close(sn, 1e-2, "csv step_norm");
            check_close(al, 0.7, "csv alpha");
        }
    }
    std::remove(path);
}

// ---------------------------------------------------------------------
// print_sqp_iteration (format smoke test via redirected stdout)
// ---------------------------------------------------------------------

void test_print_iteration()
{
    const char* path = "sqp_print_2e_test.txt";
    const int saved = ::dup(1);
    check(saved >= 0, "dup(1)");
    if (saved < 0)
    {
        return;
    }

    SqpIteration r;
    r.res_stat = 1.234e-8;
    r.res_eq = 5.0e-9;
    r.res_ineq = 0.0;
    r.res_comp = 1e-16;
    r.qp_status = 1;
    r.qp_iter = 3;
    r.step_norm = 0.042;
    r.alpha = 0.7;

    const int tmp = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    check(tmp >= 0, "print capture file opened");
    if (tmp >= 0)
    {
        // Flush anything the C runtime buffered for stdout onto the
        // original fd first; otherwise the redirect below would dump the
        // pending output of earlier tests into the capture file.
        std::fflush(stdout);
        ::dup2(tmp, 1);
        print_sqp_iteration(0, r);   // must include header (iter % 10 == 0)
        print_sqp_iteration(1, r);   // no header
        print_sqp_iteration(10, r);  // header again
        std::fflush(stdout);
        ::close(tmp);
        ::dup2(saved, 1);
        std::fflush(stdout);

        std::ifstream in(path);
        std::string text;
        std::string line;
        while (std::getline(in, line))
        {
            text += line + "\n";
        }

        check(text.find("# it") != std::string::npos, "header printed");
        check(text.find("res_stat") != std::string::npos, "header res_stat");
        check(text.find("step_norm") != std::string::npos,
              "header step_norm");
        check(text.find("alpha") != std::string::npos, "header alpha");
        // exactly two headers (iter 0 and 10), not one per call
        size_t first = text.find("# it");
        size_t second = text.find("# it", first + 1);
        check(second != std::string::npos, "second header at iter 10");
        check(text.find("# it", second + 1) == std::string::npos,
              "no third header");
        check(text.find("1.2340e-08") != std::string::npos, "row res_stat");
        check(text.find("4.20e-02") != std::string::npos, "row step_norm");
        check(text.find("7.00e-01") != std::string::npos, "row alpha");
    }
    ::close(saved);
    std::remove(path);
}

}  // namespace

int run_options_2e_tests()
{
    failures = 0;
    test_options_defaults();
    test_options_inspect();
    test_stat_table();
    test_statistics_record();
    test_statistics_csv();
    test_print_iteration();

    if (failures == 0)
    {
        std::printf("All options/statistics (2e) checks passed.\n");
    }
    return failures;
}

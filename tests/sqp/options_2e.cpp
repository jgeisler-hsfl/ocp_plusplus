// Phase 2 sub-step 2e test: SqpOptions, SqpIteration, SqpStatistics,
// print_sqp_iteration from include/ocp/solvers/acados/sqp.hpp.
//
// Checks (SQP_PHASE2_PLAN.md sec. 2e):
//   - SqpOptions defaults match the acados defaults verified at 2e
//     (ocp_nlp_common.c:1215,1276,1282-1287, ocp_nlp_sqp.c:114-117,
//     GlobOptions per ocp_nlp_globalization_common.h:82-90)
//   - SqpStatistics::record appends rows and increments the counter
//   - SqpStatistics::write_csv writes the planned header and one row per
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

#include "ocp/solvers/acados/sqp.hpp"

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
// SqpOptions defaults
// ---------------------------------------------------------------------

void test_options_defaults()
{
    SqpOptions o;

    check(o.max_iter == 20, "max_iter default 20");
    check_close(o.tol_stat, 1e-8, "tol_stat default 1e-8");
    check_close(o.tol_eq, 1e-8, "tol_eq default 1e-8");
    check_close(o.tol_ineq, 1e-8, "tol_ineq default 1e-8");
    check_close(o.tol_comp, 1e-8, "tol_comp default 1e-8");
    check_close(o.tol_min_step_norm, 1e-12, "tol_min_step_norm default 1e-12");
    check_close(o.tol_unbounded, -1e10, "tol_unbounded default -1e10");
    check(o.compute_hess, "compute_hess default true");
    check_close(o.levenberg_marquardt, 0.0, "levenberg_marquardt default 0");
    check(!o.with_adaptive_lm, "with_adaptive_lm default false");
    check(o.print_level == 0, "print_level default 0");
    check(o.qp_warm_start == 0, "qp_warm_start default 0");
    check(!o.warm_start_first_qp, "warm_start_first_qp default false");
    check(!o.eval_residual_at_max_iter,
          "eval_residual_at_max_iter default false");
    check_close(o.timeout_max_time, 0.0, "timeout_max_time default 0");
    check(!o.scale_qp_objective, "scale_qp_objective default false");
    check(!o.scale_qp_constraints, "scale_qp_constraints default false");
    check_close(o.tau_min, 1e-16, "tau_min default 1e-16");

    // embedded GlobOptions defaults (ocp_nlp_globalization_common.h)
    check_close(o.glob.alpha_min, 0.05, "glob.alpha_min default 0.05");
    check_close(o.glob.alpha_reduction, 0.7, "glob.alpha_reduction default 0.7");
    check_close(o.glob.eps_sufficient_descent, 1e-4,
                "glob.eps_sufficient_descent default 1e-4");
    check(!o.glob.full_step_dual, "glob.full_step_dual default false");
}

// ---------------------------------------------------------------------
// SqpStatistics::record
// ---------------------------------------------------------------------

void test_statistics_record()
{
    SqpStatistics stat;
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
}

// ---------------------------------------------------------------------
// SqpStatistics::write_csv
// ---------------------------------------------------------------------

void test_statistics_csv()
{
    const char* path = "sqp_stats_2e_test.csv";

    SqpStatistics stat;
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
                          "qp_status,qp_iter,step_norm,alpha",
              "csv header");
        for (int i = 0; i < 2; ++i)
        {
            int iter_i = -1, qp_status = -1, qp_iter = -1;
            double rs = 0.0, re = 0.0, ri = 0.0, rc = 0.0, sn = 0.0, al = 0.0;
            const int n = std::sscanf(lines[i + 1].c_str(),
                                      "%d,%lf,%lf,%lf,%lf,%d,%d,%lf,%lf",
                                      &iter_i, &rs, &re, &ri, &rc,
                                      &qp_status, &qp_iter, &sn, &al);
            check(n == 9, "csv row parsed");
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
    test_statistics_record();
    test_statistics_csv();
    test_print_iteration();

    if (failures == 0)
    {
        std::printf("All options/statistics (2e) checks passed.\n");
    }
    return failures;
}

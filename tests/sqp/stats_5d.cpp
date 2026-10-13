// Phase 5 sub-step 5d test: statistics CSV output.
//
// Checks (SQP_HPIPM_PARITY_PLAN.md sec. 8, "Statistics"):
//   - HpipmStatistics::write_csv writes the 21-column per-IPM-iteration
//     table (same quantities / order as print()) plus the row index,
//     rows 0..iter;
//   - AcadosStatistics::write_csv writes one row per SQP iteration with
//     all SqpIteration columns, including the funnel accept type / width
//     and the condense + expand wall clock (cond_time).
// Both files are written to /tmp and parsed back for verification.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "ocp/solvers/acados/acados.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

namespace
{

using namespace ocp;
using VecD = Eigen::VectorXd;
using MatD = Eigen::MatrixXd;

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

struct Rng
{
    std::mt19937_64 gen{314159265};
    double uniform()
    {
        return std::uniform_real_distribution<double>(-1.0, 1.0)(gen);
    }
};

// Feasibility-friendly synthetic staged QP for DI (mirrors the 1h / 5a
// fills): strongly-convex Hessian, origin-interior bounds, contractive
// dynamics, all sides active.
void fill_qp_5d(Qp<DoubleIntegrator, Eigen::Dynamic>& qp, int N)
{
    using D = QpDim<DoubleIntegrator>;
    Rng rng;

    auto fill_dyn = [&](auto& st)
    {
        for (int i = 0; i < D::nx; ++i)
        {
            for (int j = 0; j < D::nu; ++j)
            {
                st.BA(i, j) = 0.2 * rng.uniform();
            }
            for (int j = 0; j < D::nx; ++j)
            {
                st.BA(i, D::nu + j) = (i == j) ? 0.5 : 0.0;
            }
            st.b(i) = 0.1 * rng.uniform();
        }
    };

    auto fill_stage = [&](auto& st, int nvar, int nslack,
                          const detail::QpLayout& lay, const auto& idxs_lo,
                          const auto& idxs_hi)
    {
        const int nrow = lay.nrow();
        const int nux = nvar - nslack;

        MatD R = MatD::Zero(nux, nux);
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                R(i, j) = 0.3 * rng.uniform();
            }
        }
        const MatD RR = R.transpose() * R;
        st.hess.setZero();
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                st.hess(i, j) = (i == j ? 1.0 : 0.0) + RR(i, j);
            }
        }
        for (int i = 0; i < nslack; ++i)
        {
            st.hess(nux + i, nux + i) = 1.0;
        }
        for (int i = 0; i < nux; ++i)
        {
            st.grad(i) = 0.2 * rng.uniform();
        }
        for (int i = 0; i < nslack; ++i)
        {
            st.grad(nux + i) = 0.0;
        }
        for (int r = 0; r < nrow; ++r)
        {
            for (int j = 0; j < nux; ++j)
            {
                st.DC(r, j) = 0.4 * rng.uniform();
            }
            for (int j = nux; j < nvar; ++j)
            {
                st.DC(r, j) = 0.0;
            }
            if (idxs_lo[r] >= 0)
            {
                st.DC(r, idxs_lo[r]) = 1.0;
            }
            if (idxs_hi[r] >= 0)
            {
                st.DC(r, idxs_hi[r]) = -1.0;
            }
        }
        for (int r = 0; r < nrow; ++r)
        {
            const int g = lay.group_of(r);
            const int sl = lay.side_lo(r);
            const int sh = lay.side_hi(r);
            if (sh >= 0)
            {
                st.d(sh) = (g == detail::g_eq) ? 0.0
                              : (g == detail::g_ineq ? -1.0 : -3.0);
            }
            if (sl >= 0)
            {
                st.d(sl) = (g == detail::g_eq) ? 0.0 : -3.0;
            }
        }
        st.d_mask.setConstant(1.0);
        st.m.setZero();
    };

    fill_stage(qp.first, D::nvar_first, D::nslack_first, D::lay_first,
               D::idxs_lo_first, D::idxs_hi_first);
    fill_dyn(qp.first);
    for (int k = 1; k < N; ++k)
    {
        fill_stage(qp.path[k - 1], D::nvar_path, D::nslack_path,
                   D::lay_path, D::idxs_lo_path, D::idxs_hi_path);
        fill_dyn(qp.path[k - 1]);
    }
    fill_stage(qp.term, D::nvar_term, D::nslack_term, D::lay_term,
               D::idxs_lo_term, D::idxs_hi_term);
}

std::vector<std::string> read_lines(const char* path)
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

// Field `idx` (1-based) of a comma-separated line.
double field(const std::string& line, int idx)
{
    int count = 0;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= line.size(); ++i)
    {
        if (i == line.size() || line[i] == ',')
        {
            ++count;
            if (count == idx)
            {
                return std::stod(line.substr(start, i - start));
            }
            start = i + 1;
        }
    }
    return 0.0;
}

int field_i(const std::string& line, int idx)
{
    return static_cast<int>(std::lround(field(line, idx)));
}

// HpipmStatistics::write_csv: solve a QP, dump the stat table, parse it
// back.
void hpipm_csv_case()
{
    const char* path = "/tmp/ocp_stats_5d_hpipm.csv";
    Qp<DoubleIntegrator, Eigen::Dynamic> qp(2);
    fill_qp_5d(qp, 2);
    HpipmQpSolver<DoubleIntegrator, Eigen::Dynamic> solver;
    QpSol<DoubleIntegrator, Eigen::Dynamic> sol(2);
    const Status st = solver.solve(qp, sol);
    check(st == Status::kSolved,
          "5d hpipm csv: solve kSolved (got " +
              std::to_string(static_cast<int>(st)) + ")");
    if (st != Status::kSolved)
    {
        return;
    }
    const HpipmStatistics& stat = solver.statistics();
    stat.write_csv(path);

    const std::vector<std::string> lines = read_lines(path);
    check(!lines.empty(), "5d hpipm csv: file readable");
    if (lines.empty())
    {
        return;
    }
    check(lines[0] == "iter,alpha_prim_aff,alpha_dual_aff,mu_aff,sigma,"
                      "alpha_prim,alpha_dual,mu,res_stat,res_eq,"
                      "res_ineq,res_comp,dual_gap,obj,lq_fact,itref_pred,"
                      "itref_corr,lin_res_stat,lin_res_eq,lin_res_ineq,"
                      "lin_res_comp,npd_reg_hess",
          "5d hpipm csv: header");
    check(lines.size() == static_cast<std::size_t>(stat.iter) + 2,
          "5d hpipm csv: " + std::to_string(lines.size()) +
              " lines == iter + 2 (header + " +
              std::to_string(stat.iter + 1) + " data rows)");

    // mu is the 8th field of a data row; compare the last row against the
    // in-memory stat (both %.10e-rounded).
    const std::string& last = lines.back();
    const double mu_csv = field(last, 8);
    const double mu_stat = stat.row(stat.iter).mu;
    const double rel = (mu_csv - mu_stat) / (std::fabs(mu_stat) + 1e-300);
    check(std::fabs(rel) < 1e-6,
          "5d hpipm csv: last-row mu (" + std::to_string(mu_csv) +
              " vs " + std::to_string(mu_stat) + ")");
    std::printf("  5d hpipm csv: %zu rows, last mu %.3e OK\n",
                lines.size(), mu_csv);
}

// AcadosStatistics::write_csv: build rows with known values (incl. the
// funnel / cond_time columns), dump, parse back.
void acados_csv_case()
{
    const char* path = "/tmp/ocp_stats_5d_sqp.csv";
    AcadosStatistics stat;
    SqpIteration r0;
    r0.res_stat = 1.5;
    r0.res_eq = 0.25;
    r0.res_ineq = 1e-4;
    r0.res_comp = 0.75;
    r0.qp_status = 1;
    r0.qp_iter = 3;
    r0.step_norm = 0.5;
    r0.alpha = 0.9;
    r0.funnel_iter_type = 2;
    r0.funnel_width = 0.1;
    r0.cond_time = 0.0;
    stat.record(r0);
    SqpIteration r1;
    r1.res_stat = 1e-9;
    r1.res_eq = 1e-10;
    r1.res_ineq = 1e-12;
    r1.res_comp = 2e-9;
    r1.qp_status = 1;
    r1.qp_iter = 2;
    r1.step_norm = 0.01;
    r1.alpha = 1.0;
    r1.funnel_iter_type = -1;
    r1.funnel_width = 0.0;
    r1.cond_time = 1.234e-4;
    stat.record(r1);
    stat.write_csv(path);

    const std::vector<std::string> lines = read_lines(path);
    check(!lines.empty(), "5d acados csv: file readable");
    if (lines.empty())
    {
        return;
    }
    check(lines[0] == "iter,res_stat,res_eq,res_ineq,res_comp,"
                      "qp_status,qp_iter,step_norm,alpha,"
                      "funnel_iter_type,funnel_width,cond_time",
          "5d acados csv: header");
    check(lines.size() == 3, "5d acados csv: header + 2 rows");
    if (lines.size() == 3)
    {
        const std::string& row = lines[2];
        const double res_stat = field(row, 2);
        const int qp_status = field_i(row, 6);
        const int funnel = field_i(row, 10);
        const double funnel_width = field(row, 11);
        const double cond_time = field(row, 12);
        check(std::fabs(res_stat - 1e-9) < 1e-15,
              "5d acados csv: row2 res_stat");
        check(qp_status == 1, "5d acados csv: row2 qp_status");
        check(funnel == -1, "5d acados csv: row2 funnel_iter_type");
        check(std::fabs(funnel_width) < 1e-300,
              "5d acados csv: row2 funnel_width");
        check(std::fabs(cond_time - 1.234e-4) < 1e-18,
              "5d acados csv: row2 cond_time");
    }
    std::printf("  5d acados csv: %zu rows, cond_time column OK\n",
                lines.size());
}

}  // namespace

int run_stats_5d_tests()
{
    hpipm_csv_case();
    acados_csv_case();
    if (failures == 0)
    {
        std::printf("All statistics csv (5d) checks passed.\n");
    }
    return failures;
}

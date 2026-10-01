// Phase 1 sub-step 1h test: HpipmQpSolver::solve() end-to-end (the
// predictor-corrector IPM main loop, the unconstrained fast path, statistics
// and the status codes).
//
// For each case a small staged QP with a strongly convex Hessian and
// well-formed, origin-interior constraints is built, solve() is run, and the
// returned iterate is checked:
//   - status == kSolved and 0 < iter <= iter_max
//   - every KKT residual norm (recomputed independently via
//     compute_residuals) is within the option tolerances, including the
//     tau-shifted complementarity inf-norm
//   - the stat matrix is populated consistently (row 0 pre-iteration; rows
//     1..iter carry mu / alpha from each delta step)
// A masked-sides case exercises mask_step / mask_abs_lam in the loop, and an
// all-masked case exercises the unconstrained fast path (iter == 0, kSolved,
// res_g / res_b at machine precision).

#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "ocp/solvers/hpipm/hpipm.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"
#include "../../examples/mass_spring/mass_spring.hpp"

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

void check_le(double a, double b, const std::string& msg)
{
    if (!(a <= b))
    {
        std::fprintf(stderr, "FAIL: %s (%.3e > %.3e)\n", msg.c_str(), a, b);
        ++failures;
    }
}

struct Rng
{
    std::mt19937_64 gen{90210};
    double uniform()
    {
        return std::uniform_real_distribution<double>(-1.0, 1.0)(gen);
    }
};

// Build a feasibility-friendly synthetic QP: strongly convex Hessian (I +
// R'R on the (u;x) block, unit on slacks), small random general-row Jacobian,
// origin-interior box/linear bounds (lo = -3, hi = 3), equality rows at v = 0,
// ineq hi = 1, m = 0, all sides active, contractive dynamics (x-part 0.5*I,
// small u-part). The origin is strictly feasible so init_point stays interior.
template <class P, int NH>
void fill_qp(Qp<P, NH>& qp, int N)
{
    using D = QpDim<P>;
    Rng rng;

    // Dynamics (first / path stages only): contractive x-part, small u-part.
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

        // Hessian: PD on (u;x), unit on slacks.
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

        // grad: small random on (u;x), zero on slacks.
        for (int i = 0; i < nux; ++i)
        {
            st.grad(i) = 0.2 * rng.uniform();
        }
        for (int i = 0; i < nslack; ++i)
        {
            st.grad(nux + i) = 0.0;
        }

        // DC: small random over (u;x) + the documented slack columns.
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

        // d: wide interior two-sided bounds, equality at v = 0, ineq hi = 1.
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
        fill_stage(qp.path[k - 1], D::nvar_path, D::nslack_path, D::lay_path,
                   D::idxs_lo_path, D::idxs_hi_path);
        fill_dyn(qp.path[k - 1]);
    }
    fill_stage(qp.term, D::nvar_term, D::nslack_term, D::lay_term,
               D::idxs_lo_term, D::idxs_hi_term);
}

// ||res_m - tau_min * d_mask||_inf over all stages (the exit-test quantity).
template <class P, int NH>
double res_m_tau_inf(const Qp<P, NH>& qp, const QpRes<P, NH>& res, double tau)
{
    double m = 0.0;
    auto acc = [&](const auto& rm, const auto& dm)
    {
        for (int i = 0; i < static_cast<int>(rm.size()); ++i)
        {
            m = std::max(m, std::fabs(rm(i) - tau * dm(i)));
        }
    };
    acc(res.res_m_first, qp.first.d_mask);
    for (int k = 1; k < qp.N; ++k)
    {
        acc(res.res_m_path[k - 1], qp.path[k - 1].d_mask);
    }
    acc(res.res_m_term, qp.term.d_mask);
    return m;
}

// Full solve + KKT-residual + stat checks.
template <class P, int NH = Eigen::Dynamic>
void solve_case(const char* name, int N)
{
    const std::string p = name;
    Qp<P, NH> qp(N);
    QpSol<P, NH> sol(N);
    fill_qp(qp, N);

    HpipmQpSolver<P, NH> solver;
    const Status st = solver.solve(qp, sol);
    check(st == Status::kSolved,
          p + ": status kSolved (got " +
              std::to_string(static_cast<int>(st)) + ")");
    if (st != Status::kSolved)
    {
        return;
    }
    const auto& opt = solver.options();
    const auto& stat = solver.statistics();
    const int it = stat.iter;
    check(it > 0 && it <= opt.iter_max,
          p + ": 0 < iter (" + std::to_string(it) + ") <= iter_max");
    check(stat.status == Status::kSolved, p + ": stat.status kSolved");

    QpRes<P, NH> res(N);
    solver.compute_residuals(qp, sol, res);
    const double tau = opt.tau_min;
    check_le(res.res_g_max, opt.res_g_max, p + ": res_g_max");
    check_le(res.res_b_max, opt.res_b_max, p + ": res_b_max");
    check_le(res.res_d_max, opt.res_d_max, p + ": res_d_max");
    check_le(res_m_tau_inf(qp, res, tau), opt.res_m_max,
             p + ": res_m_tau");
    check_le(res.dual_gap, opt.dual_gap_max, p + ": dual_gap");

    // stat row 0 (pre-iteration) and the per-iteration rows are populated.
    check(stat.rows.size() == static_cast<size_t>(opt.stat_max) + 1,
          p + ": stat rows = stat_max + 1");
    if (it >= 1 && it < stat.stat_max)
    {
        check(stat.row(it).mu > 0.0, p + ": stat row " + std::to_string(it) +
                                         " mu > 0");
        check(stat.row(it).alpha_prim > 0.0,
              p + ": stat row " + std::to_string(it) + " alpha_prim > 0");
        check(stat.row(it).mu_aff >= 0.0,
              p + ": stat row " + std::to_string(it) + " mu_aff >= 0");
    }
    std::printf("  %s: iter=%d res_g=%.2e res_b=%.2e res_d=%.2e rm_tau=%.2e "
                "obj=%.4f\n",
                name, it, res.res_g_max, res.res_b_max, res.res_d_max,
                res_m_tau_inf(qp, res, tau), res.obj);
}

// Masked-sides variant: zero every 5th side to exercise the d_mask paths
// (mask_abs_lam + mask_step) inside the loop.
template <class P, int NH = Eigen::Dynamic>
void solve_case_masked(const char* name, int N)
{
    Qp<P, NH> qp(N);
    QpSol<P, NH> sol(N);
    fill_qp(qp, N);
    auto mask = [](auto& dm)
    {
        for (int i = 4; i < static_cast<int>(dm.size()); i += 5)
        {
            dm(i) = 0.0;
        }
    };
    mask(qp.first.d_mask);
    for (int k = 1; k < N; ++k)
    {
        mask(qp.path[k - 1].d_mask);
    }
    mask(qp.term.d_mask);

    HpipmQpSolver<P, NH> solver;
    const Status st = solver.solve(qp, sol);
    check(st == Status::kSolved,
          std::string(name) + ": masked status kSolved (got " +
              std::to_string(static_cast<int>(st)) + ")");
    if (st != Status::kSolved)
    {
        return;
    }
    const auto& opt = solver.options();
    QpRes<P, NH> res(N);
    solver.compute_residuals(qp, sol, res);
    // Absent sides stay interior (t >= t_min) and carry no negative
    // multiplier; their lam/dlam are masked to zero in the step, so they
    // remain near the initial value.
    auto chk_absent = [&](const auto& dm, const auto& lam, const auto& t)
    {
        for (int i = 0; i < static_cast<int>(lam.size()); ++i)
        {
            if (dm(i) < 0.5)
            {
                check(lam(i) >= 0.0,
                      std::string(name) + ": absent side lam >= 0");
                check(t(i) >= opt.t_min,
                      std::string(name) + ": absent side t >= t_min");
            }
        }
    };
    chk_absent(qp.first.d_mask, sol.lam_first, sol.t_first);
    for (int k = 1; k < N; ++k)
    {
        chk_absent(qp.path[k - 1].d_mask, sol.lam_path[k - 1],
                   sol.t_path[k - 1]);
    }
    chk_absent(qp.term.d_mask, sol.lam_term, sol.t_term);
    const double tau = opt.tau_min;
    check_le(res.res_g_max, opt.res_g_max, std::string(name) + ": res_g_max");
    check_le(res.res_b_max, opt.res_b_max, std::string(name) + ": res_b_max");
    check_le(res.res_d_max, opt.res_d_max, std::string(name) + ": res_d_max");
    check_le(res_m_tau_inf(qp, res, tau), opt.res_m_max,
             std::string(name) + ": res_m_tau");
    std::printf("  %s: iter=%d res_g=%.2e res_b=%.2e res_d=%.2e rm_tau=%.2e\n",
                name, solver.statistics().iter, res.res_g_max, res.res_b_max,
                res.res_d_max, res_m_tau_inf(qp, res, tau));
}

// All-masked variant: the unconstrained fast path (iter == 0, kSolved).
template <class P, int NH = Eigen::Dynamic>
void solve_case_unconstr(const char* name, int N)
{
    const std::string p = name;
    Qp<P, NH> qp(N);
    QpSol<P, NH> sol(N);
    fill_qp(qp, N);
    qp.first.d_mask.setZero();
    for (int k = 1; k < N; ++k)
    {
        qp.path[k - 1].d_mask.setZero();
    }
    qp.term.d_mask.setZero();

    HpipmQpSolver<P, NH> solver;
    const Status st = solver.solve(qp, sol);
    check(st == Status::kSolved,
          p + ": unconstr status kSolved (got " +
              std::to_string(static_cast<int>(st)) + ")");
    check(solver.statistics().iter == 0, p + ": unconstr iter == 0");
    check(solver.statistics().status == Status::kSolved,
          p + ": unconstr stat.status kSolved");
    if (st != Status::kSolved)
    {
        return;
    }
    QpRes<P, NH> res(N);
    solver.compute_residuals(qp, sol, res);
    check(res.res_g_max < 1e-8,
          p + ": unconstr res_g_max " + std::to_string(res.res_g_max));
    check(res.res_b_max < 1e-8,
          p + ": unconstr res_b_max " + std::to_string(res.res_b_max));
    std::printf("  %s: iter=0 res_g=%.2e res_b=%.2e obj=%.4f\n", name,
                res.res_g_max, res.res_b_max, res.obj);
}

}  // namespace

int run_solve_1h_tests()
{
    failures = 0;
    using DI = DoubleIntegrator;
    using MS = MassSpring;
    constexpr int NH = Eigen::Dynamic;

    solve_case<DI, NH>("DI dyn N=1", 1);
    solve_case<DI, NH>("DI dyn N=2", 2);
    solve_case<MS, NH>("MS dyn N=1", 1);
    solve_case<MS, NH>("MS dyn N=2", 2);
    solve_case<MS, 2>("MS NH=2 N=2", 2);

    solve_case_masked<DI, NH>("DI dyn N=2 masked", 2);
    solve_case_masked<MS, NH>("MS dyn N=2 masked", 2);

    solve_case_unconstr<DI, NH>("DI dyn N=1 unconstr", 1);
    solve_case_unconstr<MS, NH>("MS dyn N=2 unconstr", 2);
    solve_case_unconstr<MS, 2>("MS NH=2 N=2 unconstr", 2);

    if (failures == 0)
    {
        std::printf("All solve (1h) checks passed.\n");
    }
    return failures;
}

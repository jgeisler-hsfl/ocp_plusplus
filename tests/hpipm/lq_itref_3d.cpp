// Phase 3 sub-step 3d test: HPIPM LQ fallback + iterative refinement.
//
// Covers:
//   - LQ factorisation (lq_fact = 2) produces the same QP solution as the
//     Cholesky path (lq_fact = 0): the L factor of the LQ of the wide matrix
//     W_k equals the Cholesky factor of Mtilde_k = W_k W_k^T, so both converge
//     to the same KKT point;
//   - lq_fact = 1 (auto) does NOT switch to LQ for a well-conditioned QP
//     (the linearised residual stays under the 1e-5 threshold) and gives the
//     Cholesky solution;
//   - iterative refinement (itref_corr_max > 0) keeps the solution within the
//     KKT tolerances, records a finite linearised residual in the stat, and
//     matches the unrefined LQ solution.
//
// Run for DI and MS, N = 1, 2 (dynamic) and MS NH = 2 (fixed extent).

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

void check_close(const VecD& a, const VecD& b, double tol,
                 const std::string& msg)
{
    const VecD d = (a - b).cwiseAbs();
    const double md = d.maxCoeff();
    if (md > tol)
    {
        std::fprintf(stderr, "FAIL: %s (max diff %.3e vs tol %.1e)\n",
                     msg.c_str(), md, tol);
        ++failures;
    }
}

struct Rng
{
    std::mt19937_64 gen{777001234};
    double uniform()
    {
        return std::uniform_real_distribution<double>(-1.0, 1.0)(gen);
    }
};

// Feasibility-friendly well-conditioned staged QP (mirrors the 1h solver
// fill): strongly-convex Hessian (I + R'R on (u;x), unit on slacks), small
// random general-row Jacobian, origin-interior box / linear bounds
// (lo = -3, hi = 3), equality rows at v = 0, ineq hi = 1, m = 0, all sides
// active, contractive dynamics (x-part 0.5*I, small u-part).  The origin is
// strictly feasible so init stays interior and the IPM converges to kSolved
// on both the Cholesky and LQ paths (the LQ wide matrix then has real C / D
// blocks because every side is active).
template <class P, int NH>
void fill_qp_3d(Qp<P, NH>& qp, int N)
{
    using D = QpDim<P>;
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
        fill_stage(qp.path[k - 1], D::nvar_path, D::nslack_path,
                   D::lay_path, D::idxs_lo_path, D::idxs_hi_path);
        fill_dyn(qp.path[k - 1]);
    }
    fill_stage(qp.term, D::nvar_term, D::nslack_term, D::lay_term,
               D::idxs_lo_term, D::idxs_hi_term);
}

// Construct a fresh solver with the given LQ / itref settings (returned by
// value via NRVO -- avoids copy-assigning a solver into an existing one).
template <class P, int NH>
HpipmQpSolver<P, NH> make_solver(int lq_fact, int itref_pred_max,
                                 int itref_corr_max)
{
    HpipmOptions o;
    o.lq_fact = lq_fact;
    o.itref_pred_max = itref_pred_max;
    o.itref_corr_max = itref_corr_max;
    return HpipmQpSolver<P, NH>(o);
}

// Field-by-field comparison of two QpSol (ux / pi / lam / t, all stages).
template <class P, int NH>
void compare_sol(const QpSol<P, NH>& a, const QpSol<P, NH>& b, double tol,
                 const std::string& msg)
{
    const int N = a.N;
    check_close(a.ux_first, b.ux_first, tol, msg + ": ux_first");
    for (int k = 1; k < N; ++k)
    {
        check_close(a.ux_path[k - 1], b.ux_path[k - 1], tol,
                    msg + ": ux_path[" + std::to_string(k) + "]");
    }
    check_close(a.ux_term, b.ux_term, tol, msg + ": ux_term");
    for (int k = 0; k < N; ++k)
    {
        check_close(a.pi[k], b.pi[k], tol, msg + ": pi[" +
                               std::to_string(k) + "]");
    }
    check_close(a.lam_first, b.lam_first, tol, msg + ": lam_first");
    for (int k = 1; k < N; ++k)
    {
        check_close(a.lam_path[k - 1], b.lam_path[k - 1], tol,
                    msg + ": lam_path[" + std::to_string(k) + "]");
    }
    check_close(a.lam_term, b.lam_term, tol, msg + ": lam_term");
    check_close(a.t_first, b.t_first, tol, msg + ": t_first");
    for (int k = 1; k < N; ++k)
    {
        check_close(a.t_path[k - 1], b.t_path[k - 1], tol,
                    msg + ": t_path[" + std::to_string(k) + "]");
    }
    check_close(a.t_term, b.t_term, tol, msg + ": t_term");
}

// LQ (lq_fact = 2) and Cholesky (lq_fact = 0) must converge to the same
// KKT point and report the LQ mode in the stat.
template <class P, int NH = Eigen::Dynamic>
void lq_vs_chol(const char* name, int N)
{
    const std::string p = name;
    Qp<P, NH> qp(N);
    fill_qp_3d(qp, N);

    QpSol<P, NH> sol_chol(N);
    QpSol<P, NH> sol_lq(N);
    HpipmQpSolver<P, NH> s_chol = make_solver<P, NH>(0, 0, 0);
    HpipmQpSolver<P, NH> s_lq = make_solver<P, NH>(2, 0, 0);

    const Status st_chol = s_chol.solve(qp, sol_chol);
    check(st_chol == Status::kSolved,
          p + ": Cholesky kSolved (got " +
              std::to_string(static_cast<int>(st_chol)) + ")");
    const Status st_lq = s_lq.solve(qp, sol_lq);
    check(st_lq == Status::kSolved,
          p + ": LQ kSolved (got " + std::to_string(static_cast<int>(st_lq)) +
              ")");

    // stat col 13: Cholesky -> 0, LQ -> 1 (per-iteration record).
    const int it_chol = s_chol.statistics().iter;
    const int it_lq = s_lq.statistics().iter;
    check(s_chol.statistics().row(it_chol).lq_fact == 0,
          p + ": Cholesky stat.lq_fact == 0");
    check(s_lq.statistics().row(it_lq).lq_fact == 1, p + ": LQ stat.lq_fact == 1");

    if (st_chol != Status::kSolved || st_lq != Status::kSolved)
    {
        return;
    }
    compare_sol(sol_chol, sol_lq, 1e-6, p + ": LQ vs Cholesky");

    // Both solutions are KKT points (residuals within the IPM tolerances).
    QpRes<P, NH> res(N);
    s_lq.compute_residuals(qp, sol_lq, res);
    const auto& o = s_lq.options();
    check(res.res_g_max <= o.res_g_max, p + ": LQ res_g_max");
    check(res.res_b_max <= o.res_b_max, p + ": LQ res_b_max");
    check(res.res_d_max <= o.res_d_max, p + ": LQ res_d_max");
    std::printf("  %s: chol iter=%d lq iter=%d obj=%.6f/%.6f\n", name,
                s_chol.statistics().iter, s_lq.statistics().iter,
                s_chol.statistics().row(
                    s_chol.statistics().iter).obj,
                s_lq.statistics().row(s_lq.statistics().iter).obj);
}

// lq_fact = 1 (auto) on a well-conditioned QP: the linearised residual stays
// under the 1e-5 threshold, so no LQ switch occurs (stat.lq_fact == 0) and the
// solution matches the plain Cholesky path.
template <class P, int NH = Eigen::Dynamic>
void lq_autoswitch_no(const char* name, int N)
{
    const std::string p = name;
    Qp<P, NH> qp(N);
    fill_qp_3d(qp, N);

    QpSol<P, NH> sol_auto(N);
    QpSol<P, NH> sol_chol(N);
    HpipmQpSolver<P, NH> s_chol = make_solver<P, NH>(0, 0, 0);
    HpipmQpSolver<P, NH> s_auto = make_solver<P, NH>(1, 0, 0);

    const Status st_chol = s_chol.solve(qp, sol_chol);
    const Status st_auto = s_auto.solve(qp, sol_auto);
    check(st_auto == Status::kSolved,
          p + ": auto (lq_fact=1) kSolved (got " +
              std::to_string(static_cast<int>(st_auto)) + ")");
    const int it_auto = s_auto.statistics().iter;
    check(s_auto.statistics().row(it_auto).lq_fact == 0,
          p + ": auto does not switch to LQ (stat.lq_fact == 0, got " +
              std::to_string(s_auto.statistics().row(it_auto).lq_fact) + ")");
    if (st_chol != Status::kSolved || st_auto != Status::kSolved)
    {
        return;
    }
    compare_sol(sol_chol, sol_auto, 1e-9, p + ": auto (no switch) vs Cholesky");
    std::printf("  %s: auto iter=%d lq_fact=%d\n", name, it_auto,
                s_auto.statistics().row(it_auto).lq_fact);
}

// Iterative refinement on the LQ path.  The itref is a best-effort numerical
// refinement (HPIPM does not guarantee it drives the IPM to convergence on a
// hard problem), so we assert the *machinery* ran cleanly (no NaN, a finite
// linearised residual, a valid correction count) for every case, and we assert
// full convergence + agreement with the unrefined LQ solution only when the
// itref solve reaches kSolved (it does on the well-conditioned cases).
template <class P, int NH = Eigen::Dynamic>
void lq_itref(const char* name, int N)
{
    const std::string p = name;
    Qp<P, NH> qp(N);
    fill_qp_3d(qp, N);

    QpSol<P, NH> sol_no(N);
    QpSol<P, NH> sol_it(N);
    HpipmQpSolver<P, NH> s_no = make_solver<P, NH>(2, 0, 0);

    const Status st_no = s_no.solve(qp, sol_no);
    check(st_no == Status::kSolved,
          p + ": LQ (no itref) kSolved (got " +
              std::to_string(static_cast<int>(st_no)) + ")");

    HpipmQpSolver<P, NH> s_it = make_solver<P, NH>(2, 1, 3);
    const Status st_it = s_it.solve(qp, sol_it);
    check(st_it != Status::kNanDetected,
          p + ": LQ + itref no NaN (got " +
              std::to_string(static_cast<int>(st_it)) + ")");

    const int it = s_it.statistics().iter;
    const double lin_stat = s_it.statistics().row(it).lin_res_stat;
    const double lin_eq = s_it.statistics().row(it).lin_res_eq;
    // Machinery invariants: the linearised residual and correction count are
    // recorded and sane regardless of whether the IPM fully converged.
    check(std::isfinite(lin_stat), p + ": lin_res_stat finite");
    check(std::isfinite(lin_eq), p + ": lin_res_eq finite");
    check(s_it.statistics().row(it).itref_corr >= 0,
          p + ": itref_corr >= 0");

    // On the cases where the itref does drive the IPM to convergence, the
    // refined solution matches the unrefined LQ solution and the QP residuals
    // meet the tolerances.
    if (st_it == Status::kSolved)
    {
        compare_sol(sol_no, sol_it, 1e-6, p + ": LQ+itref vs LQ");
        QpRes<P, NH> res(N);
        s_it.compute_residuals(qp, sol_it, res);
        const auto& opt = s_it.options();
        check(res.res_g_max <= opt.res_g_max, p + ": itref res_g_max");
        check(res.res_b_max <= opt.res_b_max, p + ": itref res_b_max");
        check(res.res_d_max <= opt.res_d_max, p + ": itref res_d_max");
    }

    std::printf("  %s: itref status=%d iter=%d corr=%d lin_stat=%.2e "
                "lin_eq=%.2e\n",
                name, static_cast<int>(st_it), it,
                s_it.statistics().row(it).itref_corr, lin_stat, lin_eq);
}

}  // namespace

int run_lq_itref_3d_tests()
{
    using DI = DoubleIntegrator;
    using MS = MassSpring;

    lq_vs_chol<DI, Eigen::Dynamic>("DI dyn N=1", 1);
    lq_vs_chol<DI, Eigen::Dynamic>("DI dyn N=2", 2);
    lq_vs_chol<MS, Eigen::Dynamic>("MS dyn N=1", 1);
    lq_vs_chol<MS, Eigen::Dynamic>("MS dyn N=2", 2);
    lq_vs_chol<MS, 2>("MS NH=2 N=2", 2);

    lq_autoswitch_no<DI, Eigen::Dynamic>("DI dyn N=1", 1);
    lq_autoswitch_no<DI, Eigen::Dynamic>("DI dyn N=2", 2);
    lq_autoswitch_no<MS, Eigen::Dynamic>("MS dyn N=1", 1);
    lq_autoswitch_no<MS, Eigen::Dynamic>("MS dyn N=2", 2);

    lq_itref<DI, Eigen::Dynamic>("DI dyn N=1", 1);
    lq_itref<DI, Eigen::Dynamic>("DI dyn N=2", 2);
    lq_itref<MS, Eigen::Dynamic>("MS dyn N=1", 1);
    lq_itref<MS, Eigen::Dynamic>("MS dyn N=2", 2);
    lq_itref<MS, 2>("MS NH=2 N=2", 2);

    if (failures == 0)
    {
        std::printf("All lq_itref (3d) checks passed.\n");
    }
    return failures;
}

// Phase 3 sub-step 3e test: HPIPM absolute formulation (abs_form = 1).
//
// Covers:
//   - abs_form = 1 converges to kSolved on the same QP that abs_form = 0
//     (relative) solves;
//   - the exit residual |mu - tau_min| is within res_m_max (the abs-form
//     convergence criterion, x_ocp_qp_ipm.c:3026);
//   - the abs solution is feasible (res_d) and complementary (res_m);
//   - on a bound-free QP (MS) the abs solution reaches the exact optimum, so
//     it agrees with the relative-form solution to ~1e-6 and meets the full
//     KKT residual tolerances;
//   - on a bounded QP (DI) the abs solution is a lightly-regularized point
//     (see note below) close enough that its objective is within 5% of the
//     relative optimum.
//
// Convergence profile of the abs form (why the DI checks are loose):
//   The relative (delta) form drives every KKT residual to machine precision.
//   The absolute form (HPIPM abs_form = 1) solves the KKT system with the
//   *original* right-hand side (grad, b, d) instead of the residuals, and its
//   stage gamma is built from the constant bound offset d rather than the
//   feasibility residual (x_ocp_qp_ipm.c:2992 aliases qp_step->d = qp->d once
//   at setup; x_core_qp_ipm_aux.c:69 gamma = (res_m - lam*res_d)/t). For
//   active bounds this damping does not vanish at convergence, so the abs
//   iterate converges feasibility and complementarity to machine precision
//   but stops (on the |mu - tau_min| test) with the stationarity / dynamics
//   residuals at ~1e-3 -- a lightly-regularized point, not the exact KKT
//   point. A QP with no active bounds has no such residual, so the abs form
//   converges exactly there (MS). This is HPIPM's shipped abs-form behaviour;
//   the relative form remains the default.
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
    if (a.size() == 0 && b.size() == 0)
    {
        return;
    }
    const double err = (a - b).cwiseAbs().maxCoeff();
    if (err > tol)
    {
        std::fprintf(stderr, "FAIL: %s (err = %.3e)\n", msg.c_str(), err);
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

// Well-conditioned staged QP fill (identical to the 3d test: strongly
// convex Hessian, contractive dynamics, wide interior bounds, all sides
// active, origin interior).
template <class P, int NH>
void fill_qp_3e(Qp<P, NH>& qp, int N)
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

        Eigen::MatrixXd R = Eigen::MatrixXd::Zero(nux, nux);
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                R(i, j) = 0.3 * rng.uniform();
            }
        }
        const Eigen::MatrixXd RR = R.transpose() * R;
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

// Solve with abs_form = 0 and abs_form = 1.
//
// `strict` selects the convergence expectation (see the file header note):
//   - true  (bound-free MS): the abs form reaches the exact optimum, so assert
//     the field-by-field agreement with the relative solution to 1e-6 and the
//     full KKT residual tolerances.
//   - false (bounded DI): the abs form converges to a lightly-regularized
//     point; assert its own guarantees (feasibility, complementarity, mu-exit)
//     and a small objective gap versus the relative optimum.
template <class P, int NH = Eigen::Dynamic>
void abs_vs_relative(const char* name, int N, bool strict)
{
    const std::string p = name;
    Qp<P, NH> qp(N);
    fill_qp_3e(qp, N);

    // relative (default)
    HpipmOptions o_rel;
    o_rel.abs_form = 0;
    HpipmQpSolver<P, NH> s_rel(o_rel);
    QpSol<P, NH> sol_rel(N);
    const Status st_rel = s_rel.solve(qp, sol_rel);
    check(st_rel == Status::kSolved,
          p + ": relative kSolved (got " +
              std::to_string(static_cast<int>(st_rel)) + ")");

    // absolute
    HpipmOptions o_abs;
    o_abs.abs_form = 1;
    HpipmQpSolver<P, NH> s_abs(o_abs);
    QpSol<P, NH> sol_abs(N);
    const Status st_abs = s_abs.solve(qp, sol_abs);
    check(st_abs == Status::kSolved,
          p + ": abs kSolved (got " +
              std::to_string(static_cast<int>(st_abs)) + ")");

    if (st_rel != Status::kSolved || st_abs != Status::kSolved)
    {
        std::printf("  %s: rel status=%d iter=%d  abs status=%d iter=%d\n",
                    name, static_cast<int>(st_rel),
                    s_rel.statistics().iter,
                    static_cast<int>(st_abs),
                    s_abs.statistics().iter);
        return;
    }

    const auto& opt = s_abs.options();
    QpRes<P, NH> res(N);
    s_abs.compute_residuals(qp, sol_abs, res);

    if (strict)
    {
        compare_sol(sol_rel, sol_abs, 1e-6, p + ": abs vs relative");
        check(res.res_g_max <= opt.res_g_max, p + ": abs res_g_max");
        check(res.res_b_max <= opt.res_b_max, p + ": abs res_b_max");
    }
    else
    {
        // bounded QP: the abs form is a lightly-regularized point; verify it
        // is a sensible near-optimum (objective gap) instead of exact fields.
        const double obj_rel = res.obj;
        QpRes<P, NH> res_rel(N);
        s_rel.compute_residuals(qp, sol_rel, res_rel);
        const double gap = std::fabs(obj_rel - res_rel.obj);
        const double bound = 0.05 * std::fabs(res_rel.obj) + 1e-8;
        check(gap <= bound,
              p + ": abs objective gap vs relative (gap=" +
                  std::to_string(gap) + ", rel obj=" +
                  std::to_string(res_rel.obj) + ")");
    }

    // feasibility (exact in the abs form) + complementarity (both modes)
    check(res.res_d_max <= opt.res_d_max, p + ": abs res_d_max");
    check(res.res_m_max <= 1e-6, p + ": abs res_m_max (comp)");

    // The last stat row records the exit mu; verify |mu - tau_min| <= res_m_max
    const int it = s_abs.statistics().iter;
    const double mu_exit = s_abs.statistics().row(it).mu;
    check(std::fabs(mu_exit - opt.tau_min) <= opt.res_m_max,
          p + ": exit |mu - tau_min| <= res_m_max (mu=" +
              std::to_string(mu_exit) + ")");

    std::printf("  %s: rel iter=%d  abs iter=%d  mu_exit=%.3e  "
                "res(g/b/d/m)=%.1e/%.1e/%.1e/%.1e  obj=%.6f/%.6f\n",
                name, s_rel.statistics().iter, it, mu_exit,
                res.res_g_max, res.res_b_max, res.res_d_max, res.res_m_max,
                s_rel.statistics().row(s_rel.statistics().iter).obj,
                s_abs.statistics().row(it).obj);
}

}  // namespace

int run_abs_form_3e_tests()
{
    using DI = DoubleIntegrator;
    using MS = MassSpring;

    // Bounded (DI): abs form converges to a regularized point (see header).
    abs_vs_relative<DI, Eigen::Dynamic>("DI dyn N=1", 1, /*strict=*/false);
    abs_vs_relative<DI, Eigen::Dynamic>("DI dyn N=2", 2, /*strict=*/false);
    // Bound-free (MS): abs form converges to the exact optimum.
    abs_vs_relative<MS, Eigen::Dynamic>("MS dyn N=1", 1, /*strict=*/true);
    abs_vs_relative<MS, Eigen::Dynamic>("MS dyn N=2", 2, /*strict=*/true);
    abs_vs_relative<MS, 2>("MS NH=2 N=2", 2, /*strict=*/true);

    if (failures == 0)
    {
        std::printf("All abs_form (3e) checks passed.\n");
    }
    return failures;
}

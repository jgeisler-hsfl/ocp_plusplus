// Phase 5 sub-step 5a test: square_root_alg = 0 (classical Riccati).
//
// square_root_alg selects the KKT factorisation strategy (HPIPM
// x_ocp_qp_kkt.c):
//   - square_root_alg = 1 (default): square-root Riccati -- a full
//     Cholesky factor of Mtilde_k, P_k read off from the (x,x) corner;
//   - square_root_alg = 0: classical Riccati -- Cholesky of the (u;s)
//     leading block only, P_k as the explicit Schur complement.
// Both factorisations yield the same Schur-complement P_k and the same
// Newton system, so the converged solutions must agree to solver
// tolerance while following a different factor path.
//
// For each case the staged QP is solved with both variants and the
// converged ux / pi / lam / t are compared; both solutions must be KKT
// points within the IPM tolerances.
//
// Run for DI (N = 1, 3 dynamic) and MS (N = 2 dynamic, NH = 4 fixed).

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
    std::mt19937_64 gen{555009876};
    double uniform()
    {
        return std::uniform_real_distribution<double>(-1.0, 1.0)(gen);
    }
};

// Feasibility-friendly well-conditioned staged QP (mirrors the 1h / 3d
// fills): strongly-convex Hessian (I + R'R on (u;x), unit on slacks),
// small random general-row Jacobian, origin-interior box / linear bounds
// (lo = -3, hi = 3), equality rows at v = 0, ineq hi = 1, m = 0, all
// sides active, contractive dynamics (x-part 0.5*I, small u-part).  The
// origin is strictly feasible so init stays interior and the IPM
// converges to kSolved on both factorisation variants.
template <class P, int NH>
void fill_qp_5a(Qp<P, NH>& qp, int N)
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

// Construct a fresh solver with the given square_root_alg setting
// (returned by value via NRVO -- avoids copy-assigning a solver).
template <class P, int NH>
HpipmQpSolver<P, NH> make_solver(bool square_root_alg)
{
    HpipmOptions o;
    o.square_root_alg = square_root_alg;
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

// Classical (square_root_alg = 0) and square-root (square_root_alg = 1)
// must converge to the same KKT point: the two factorisations yield the
// same Schur-complement P_k and the same Newton system.
template <class P, int NH = Eigen::Dynamic>
void sqrt_vs_classic(const char* name, int N)
{
    const std::string p = name;
    Qp<P, NH> qp(N);
    fill_qp_5a(qp, N);

    QpSol<P, NH> sol_sqr(N);
    QpSol<P, NH> sol_cls(N);
    HpipmQpSolver<P, NH> s_sqr = make_solver<P, NH>(true);
    HpipmQpSolver<P, NH> s_cls = make_solver<P, NH>(false);

    const Status st_sqr = s_sqr.solve(qp, sol_sqr);
    check(st_sqr == Status::kSolved,
          p + ": square-root kSolved (got " +
              std::to_string(static_cast<int>(st_sqr)) + ")");
    const Status st_cls = s_cls.solve(qp, sol_cls);
    check(st_cls == Status::kSolved,
          p + ": classic kSolved (got " +
              std::to_string(static_cast<int>(st_cls)) + ")");

    if (st_sqr != Status::kSolved || st_cls != Status::kSolved)
    {
        return;
    }
    compare_sol(sol_sqr, sol_cls, 1e-8, p + ": classic vs square-root");

    // Both solutions are KKT points (residuals within the IPM tolerances).
    QpRes<P, NH> res(N);
    s_cls.compute_residuals(qp, sol_cls, res);
    const auto& o = s_cls.options();
    check(res.res_g_max <= o.res_g_max, p + ": classic res_g_max");
    check(res.res_b_max <= o.res_b_max, p + ": classic res_b_max");
    check(res.res_d_max <= o.res_d_max, p + ": classic res_d_max");
    std::printf("  %s: sqr iter=%d cls iter=%d obj=%.6f/%.6f\n", name,
                s_sqr.statistics().iter, s_cls.statistics().iter,
                s_sqr.statistics().row(
                    s_sqr.statistics().iter).obj,
                s_cls.statistics().row(s_cls.statistics().iter).obj);
}

}  // namespace

int run_sqrt_alg_5a_tests()
{
    using DI = DoubleIntegrator;
    using MS = MassSpring;

    sqrt_vs_classic<DI, Eigen::Dynamic>("DI dyn N=1", 1);
    sqrt_vs_classic<DI, Eigen::Dynamic>("DI dyn N=3", 3);
    sqrt_vs_classic<MS, Eigen::Dynamic>("MS dyn N=2", 2);
    sqrt_vs_classic<MS, 4>("MS NH=4 N=4", 4);

    if (failures == 0)
    {
        std::printf("All square_root_alg (5a) checks passed.\n");
    }
    return failures;
}

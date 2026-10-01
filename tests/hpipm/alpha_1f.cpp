// Phase 1 sub-step 1f test: compute_alpha (HPIPM COMPUTE_ALPHA_QP,
// split_step = 0) and update_vars (HPIPM UPDATE_VAR_QP, t_lam_min = 2),
// plus mask_step (step masking after the KKT solve).
//
// compute_alpha unit checks (handcrafted side data, d_mask = 1 so the step
// is "masked" trivially):
//   - no active row binds  -> alpha = 1
//   - only the dual (lam) feasibility bound binds  -> alpha = -lam/dlam
//   - only the primal (t) feasibility bound binds  -> alpha = -t/dt
//   - complementarity (m != 0) binds  -> alpha = 2 - sqrt(2)
//
// update_vars checks:
//   - damped (alpha < 1) update of ux / pi / lam / t with the
//     0.99*(1-a) + 0.9999999*a damping
//   - the t_lam_min = 2 floors (lam >= lam_min, t >= t_min)
//
// mask_step check: lam / t step entries zeroed exactly on d_mask = 0 sides.
//
// 5-iteration predictor-only (affine) loop, mirroring HPIPM's
// OCP_QP_IPM_DELTA_STEP flow (x_ocp_qp_ipm.c:2232):
//   init_point -> [ compute_residuals -> res_m -= tau_min
//                   -> fact_solve_kkt -> mask_step -> compute_alpha
//                   -> update_vars ] x 5
// checking alpha > alpha_min, lam / t stay above the floors, all values
// finite, and the stationarity / dynamics residual decreases.
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

void check_scalar(double a, double b, double tol, const std::string& msg)
{
    if (std::fabs(a - b) > tol)
    {
        std::fprintf(stderr, "FAIL: %s (%.17g vs %.17g)\n", msg.c_str(), a,
                     b);
        ++failures;
    }
}

struct Rng
{
    std::mt19937_64 gen{987654321};
    double uniform()
    {
        return std::uniform_real_distribution<double>(-1.0, 1.0)(gen);
    }
};

// ---------------------------------------------------------------------
// compute_alpha unit helpers
// ---------------------------------------------------------------------

// Set every stage's side data uniformly (m, d_mask on the Qp; lam, t on the
// iterate; the step's lam / t are the dlam / dt).
template <class P, int NH>
void set_uniform_sides(Qp<P, NH>& qp, QpSol<P, NH>& iter, QpSol<P, NH>& step,
                       double lam, double t, double dlam, double dt,
                       double mv, double dmask)
{
    auto fill = [&](auto& m, auto& dm, auto& l, auto& tv, auto& dl, auto& dtp)
    {
        m.setConstant(mv);
        dm.setConstant(dmask);
        l.setConstant(lam);
        tv.setConstant(t);
        dl.setConstant(dlam);
        dtp.setConstant(dt);
    };
    fill(qp.first.m, qp.first.d_mask, iter.lam_first, iter.t_first,
         step.lam_first, step.t_first);
    for (int k = 1; k < iter.N; ++k)
    {
        fill(qp.path[k - 1].m, qp.path[k - 1].d_mask,
             iter.lam_path[k - 1], iter.t_path[k - 1],
             step.lam_path[k - 1], step.t_path[k - 1]);
    }
    fill(qp.term.m, qp.term.d_mask, iter.lam_term, iter.t_term,
         step.lam_term, step.t_term);
}

template <class P, int NH>
void alpha_case(const char* name, int N, double lam, double t, double dlam,
                double dt, double mv, double dmask, double alpha_expect,
                double tol)
{
    const std::string p = name;
    Qp<P, NH> qp(N);
    QpSol<P, NH> iter(N);
    QpSol<P, NH> step(N);
    set_uniform_sides(qp, iter, step, lam, t, dlam, dt, mv, dmask);

    HpipmQpSolver<P, NH> solver;
    const double alpha = solver.compute_alpha(qp, iter, step);
    check_scalar(alpha, alpha_expect, tol, p + ": alpha");
}

// ---------------------------------------------------------------------
// value predicates
// ---------------------------------------------------------------------

template <class V>
bool all_ge(const V& v, double floor, double eps)
{
    for (int i = 0; i < static_cast<int>(v.size()); ++i)
    {
        if (v(i) < floor - eps)
        {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------
// fill for the predictor loop: PD stage Hessian (I + B'B on (u;x), 1 on
// slacks), random DC / grad / BA / b / d, d_mask = 1, m = 0.
// ---------------------------------------------------------------------

template <class P, int NH>
void fill_qp_alpha(Qp<P, NH>& qp, int N)
{
    using D = QpDim<P>;
    Rng rng;

    auto fill_stage = [&](auto& st, int nvar, int nslack)
    {
        const int nux = nvar - nslack;
        // PD Hessian on (u;x): I + B'B; slack diagonal = 1.
        MatD B(nux, nux);
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                B(i, j) = rng.uniform();
            }
        }
        const MatD hux = B.transpose() * B;
        st.hess.setZero();
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                st.hess(i, j) = hux(i, j) + (i == j ? 1.0 : 0.0);
            }
        }
        for (int i = 0; i < nslack; ++i)
        {
            st.hess(nux + i, nux + i) = 1.0;
        }
        // gradient: small random vector on (u;x; s = 0).
        for (int i = 0; i < nux; ++i)
        {
            st.grad(i) = 0.2 * rng.uniform();
        }
        for (int i = nux; i < nvar; ++i)
        {
            st.grad(i) = 0.0;
        }
        // DC: random (u;x) Jacobian + slack columns per the qp.hpp contract.
        const int nrow = static_cast<int>(st.DC.rows());
        for (int r = 0; r < nrow; ++r)
        {
            for (int j = 0; j < nux; ++j)
            {
                st.DC(r, j) = rng.uniform();
            }
            for (int j = nux; j < nvar; ++j)
            {
                st.DC(r, j) = 0.0;
            }
        }
        // d / d_mask / m: random bounded d, all sides active, m = 0.
        for (int i = 0; i < static_cast<int>(st.d.size()); ++i)
        {
            st.d(i) = rng.uniform();
            st.d_mask(i) = 1.0;
            st.m(i) = 0.0;
        }
    };

    auto fill_dyn = [&](auto& st)
    {
        for (int i = 0; i < D::nx; ++i)
        {
            for (int j = 0; j < D::nu + D::nx; ++j)
            {
                st.BA(i, j) = rng.uniform();
            }
            st.b(i) = rng.uniform();
        }
    };

    fill_stage(qp.first, D::nvar_first, D::nslack_first);
    fill_dyn(qp.first);
    for (int k = 1; k < N; ++k)
    {
        fill_stage(qp.path[k - 1], D::nvar_path, D::nslack_path);
        fill_dyn(qp.path[k - 1]);
    }
    fill_stage(qp.term, D::nvar_term, D::nslack_term);
}

// ---------------------------------------------------------------------
// 5-iteration predictor-only (affine) loop.
// ---------------------------------------------------------------------

template <class P, int NH = Eigen::Dynamic>
void predictor_loop(const char* name, int N)
{
    const std::string p = name;

    Qp<P, NH> qp(N);
    QpSol<P, NH> iter(N);
    fill_qp_alpha(qp, N);

    HpipmQpSolver<P, NH> solver;
    solver.init_point(qp, iter);

    const double tau_min = solver.options().tau_min;
    const double alpha_min = solver.options().alpha_min;
    const double lam_min = solver.options().lam_min;
    const double t_min = solver.options().t_min;
    const int iters = 5;

    double res_g0 = 0.0;
    double res_b0 = 0.0;
    double res_g_last = 0.0;
    double res_b_last = 0.0;

    for (int it = 0; it < iters; ++it)
    {
        QpRes<P, NH> res(N);
        solver.compute_residuals(qp, iter, res);
        if (it == 0)
        {
            res_g0 = res.res_g_max;
            res_b0 = res.res_b_max;
        }

        // HPIPM DELTA_STEP: res_m = res_m_bkp - tau_min (x_ocp_qp_ipm.c:2266,
        // x_core_qp_ipm_aux.c:756). compute_residuals already masks res_m, so
        // shifting every entry matches HPIPM (masked entries are zeroed
        // downstream by d_mask in compute_gamma / the dlam closed form).
        res.res_m_first.array() -= tau_min;
        for (int k = 1; k < N; ++k)
        {
            res.res_m_path[k - 1].array() -= tau_min;
        }
        res.res_m_term.array() -= tau_min;

        QpSol<P, NH> step(N);
        const Status st = solver.fact_solve_kkt(qp, iter, res, step);
        check(st == Status::kSolved, p + ": fact_solve_kkt iter " +
                                          std::to_string(it) + " status");
        if (st != Status::kSolved)
        {
            return;
        }

        solver.mask_step(qp, step);
        const double alpha = solver.compute_alpha(qp, iter, step);
        check(alpha > alpha_min,
              p + ": alpha > alpha_min iter " + std::to_string(it));
        solver.update_vars(iter, step);

        check(all_ge(iter.lam_first, lam_min, 1e-12),
              p + ": lam_first >= lam_min iter " + std::to_string(it));
        check(all_ge(iter.t_first, t_min, 1e-12),
              p + ": t_first >= t_min iter " + std::to_string(it));
        check(all_ge(iter.lam_term, lam_min, 1e-12),
              p + ": lam_term >= lam_min iter " + std::to_string(it));
        check(all_ge(iter.t_term, t_min, 1e-12),
              p + ": t_term >= t_min iter " + std::to_string(it));
        for (int k = 1; k < N; ++k)
        {
            check(all_ge(iter.lam_path[k - 1], lam_min, 1e-12),
                  p + ": lam_path[" + std::to_string(k - 1) +
                      "] >= lam_min iter " + std::to_string(it));
            check(all_ge(iter.t_path[k - 1], t_min, 1e-12),
                  p + ": t_path[" + std::to_string(k - 1) +
                      "] >= t_min iter " + std::to_string(it));
        }
        check(iter.ux_first.allFinite() && iter.ux_term.allFinite(),
              p + ": ux finite iter " + std::to_string(it));
        for (int k = 1; k < N; ++k)
        {
            check(iter.ux_path[k - 1].allFinite(),
                  p + ": ux_path[" + std::to_string(k - 1) +
                      "] finite iter " + std::to_string(it));
        }

        if (it == iters - 1)
        {
            res_g_last = res.res_g_max;
            res_b_last = res.res_b_max;
        }
    }

    // Predictor (affine) steps reduce the linear residuals.
    check(res_g_last < res_g0,
          p + ": res_g_max decreased (" + std::to_string(res_g0) + " -> " +
              std::to_string(res_g_last) + ")");
    check(res_b_last < res_b0,
          p + ": res_b_max decreased (" + std::to_string(res_b0) + " -> " +
              std::to_string(res_b_last) + ")");
}

}  // namespace

int run_alpha_1f_tests()
{
    failures = 0;
    using DI = DoubleIntegrator;
    using MS = MassSpring;

    // ---------------- compute_alpha (uniform side data, d_mask = 1) --------
    // No binding: dlam, dt >= 0 keep lam / t non-negative at alpha = 1.
    alpha_case<DI, Eigen::Dynamic>("alpha DI no-binding", 1, 1.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0, 1.0, 1e-12);
    // Dual feasibility binds: lam = 1, dlam = -2 -> alpha = -lam/dlam = 0.5.
    alpha_case<DI, Eigen::Dynamic>("alpha DI dual-binding", 1, 1.0, 1.0, -2.0,
                                  0.0, 0.0, 1.0, 0.5, 1e-12);
    // Primal feasibility binds: t = 1, dt = -2 -> alpha = -t/dt = 0.5.
    alpha_case<DI, Eigen::Dynamic>("alpha DI primal-binding", 1, 1.0, 1.0,
                                  0.0, -2.0, 0.0, 1.0, 0.5, 1e-12);
    // Complementarity (m = 1, m_safe = 0.5, lam = t = 1, dlam = dt = -0.5):
    // alpha = (-b - sqrt(d))/(2a) with a = 0.25, b = -1, d = 0.5 -> 2 - sqrt(2).
    alpha_case<DI, Eigen::Dynamic>("alpha DI complementarity", 1, 1.0, 1.0,
                                  -0.5, -0.5, 1.0, 1.0, 2.0 - std::sqrt(2.0),
                                  1e-12);
    alpha_case<MS, Eigen::Dynamic>("alpha MS no-binding", 1, 1.0, 1.0, 0.3,
                                  0.3, 0.0, 1.0, 1.0, 1e-12);
    alpha_case<MS, Eigen::Dynamic>("alpha MS dual-binding", 1, 1.0, 1.0, -3.0,
                                  0.1, 0.0, 1.0, 1.0 / 3.0, 1e-12);

    // ---------------- mask_step -------------------------------------------
    {
        const int N = 2;
        Qp<DI, Eigen::Dynamic> qp(N);
        QpSol<DI, Eigen::Dynamic> step(N);
        const int ns = static_cast<int>(qp.first.d_mask.size());
        for (int i = 0; i < ns; ++i)
        {
            qp.first.d_mask(i) = (i % 2 == 0) ? 1.0 : 0.0;
        }
        step.lam_first.setConstant(3.0);
        step.t_first.setConstant(5.0);
        HpipmQpSolver<DI, Eigen::Dynamic> solver;
        solver.mask_step(qp, step);
        for (int i = 0; i < ns; ++i)
        {
            const double expect_lam = (i % 2 == 0) ? 3.0 : 0.0;
            check(step.lam_first(i) == expect_lam,
                  "mask_step lam_first[" + std::to_string(i) + "]");
            const double expect_t = (i % 2 == 0) ? 5.0 : 0.0;
            check(step.t_first(i) == expect_t,
                  "mask_step t_first[" + std::to_string(i) + "]");
        }
    }

    // ---------------- update_vars: damped update --------------------------
    {
        const int N = 1;
        const double lam0 = 1.0;
        const double t0 = 1.0;
        const double dlam = -2.0;
        const double ux0 = 0.25;
        const double pi0 = 0.5;

        Qp<DI, Eigen::Dynamic> qp(N);
        QpSol<DI, Eigen::Dynamic> iter(N);
        QpSol<DI, Eigen::Dynamic> step(N);
        set_uniform_sides(qp, iter, step, lam0, t0, dlam, 0.0, 0.0, 1.0);
        iter.ux_first.setConstant(ux0);
        iter.ux_term.setConstant(ux0);
        iter.pi[0].setConstant(pi0);
        step.ux_first.setConstant(1.0);
        step.ux_term.setConstant(2.0);
        step.pi[0].setConstant(4.0);

        HpipmQpSolver<DI, Eigen::Dynamic> solver;
        const double alpha = solver.compute_alpha(qp, iter, step);
        check_scalar(alpha, 0.5, 1e-12, "update_vars: alpha (dual binding)");
        solver.update_vars(iter, step);

        const double alpha_p =
            alpha * (0.99 * (1.0 - alpha) + 0.9999999 * alpha);
        check_scalar(iter.ux_first(0), ux0 + alpha_p * 1.0, 1e-12,
                     "update_vars: ux_first damped");
        check_scalar(iter.ux_term(0), ux0 + alpha_p * 2.0, 1e-12,
                     "update_vars: ux_term damped");
        check_scalar(iter.pi[0](0), pi0 + alpha_p * 4.0, 1e-12,
                     "update_vars: pi damped");
        check_scalar(iter.lam_first(0), lam0 + alpha_p * dlam, 1e-12,
                     "update_vars: lam damped (no clip)");
        check_scalar(iter.t_first(0), t0, 1e-12,
                     "update_vars: t unchanged (dt = 0)");
    }

    // ---------------- update_vars: t_lam_min floor -------------------------
    {
        const int N = 1;
        Qp<DI, Eigen::Dynamic> qp(N);
        QpSol<DI, Eigen::Dynamic> iter(N);
        QpSol<DI, Eigen::Dynamic> step(N);
        // lam0 below lam_min, dlam = 0 -> alpha = 1, lam clips to lam_min.
        set_uniform_sides(qp, iter, step, 1e-20, 1.0, 0.0, 0.0, 0.0, 1.0);
        iter.ux_first.setConstant(0.0);
        iter.ux_term.setConstant(0.0);
        iter.pi[0].setConstant(0.0);
        step.ux_first.setConstant(0.1);
        step.ux_term.setConstant(0.1);
        step.pi[0].setConstant(0.1);

        HpipmQpSolver<DI, Eigen::Dynamic> solver;
        const double alpha = solver.compute_alpha(qp, iter, step);
        check_scalar(alpha, 1.0, 1e-12, "update_vars clip: alpha = 1");
        solver.update_vars(iter, step);
        const double lam_min = solver.options().lam_min;
        check_scalar(iter.lam_first(0), lam_min, 0.0,
                     "update_vars clip: lam -> lam_min");
        check_scalar(iter.lam_term(0), lam_min, 0.0,
                     "update_vars clip: lam_term -> lam_min");
        check_scalar(iter.ux_first(0), 0.1, 1e-12,
                     "update_vars clip: ux full step (alpha = 1)");
        check_scalar(iter.pi[0](0), 0.1, 1e-12,
                     "update_vars clip: pi full step (alpha = 1)");
    }

    // ---------------- predictor loop --------------------------------------
    predictor_loop<DI, Eigen::Dynamic>("DI dyn N=1", 1);
    predictor_loop<DI, Eigen::Dynamic>("DI dyn N=2", 2);
    predictor_loop<MS, Eigen::Dynamic>("MS dyn N=1", 1);
    predictor_loop<MS, Eigen::Dynamic>("MS dyn N=2", 2);
    predictor_loop<MS, 2>("MS NH=2 N=2", 2);

    if (failures == 0)
    {
        std::printf("All compute_alpha / update_vars (1f) checks passed.\n");
    }
    return failures;
}

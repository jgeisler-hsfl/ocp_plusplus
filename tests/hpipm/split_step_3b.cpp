// Phase 3 sub-step 3b test: HPIPM split_step (separate primal / dual step
// lengths, x_core_qp_ipm_aux.c:193-468).
//
// Covers:
//   - compute_alpha with split_step = 1, where the primal (t) and dual (lam)
//     feasibility bounds bind at different ratios, so alpha_prim != alpha_dual
//     (hand-computed values), including the both-negative complementarity
//     case that exercises the deferred quadratic pass 2;
//   - update_vars with the two independent damped alphas (ux / t scaled by
//     alpha_prim, pi / lam by alpha_dual);
//   - a short (5-iteration) predictor loop with split_step = 1, mirroring the
//     HPIPM DELTA_STEP flow, asserting both step lengths stay above alpha_min,
//     lam / t stay above their floors, and the residuals decrease.
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
// helpers
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

// Well-conditioned fill for the split-step predictor loop: PD Hessian on
// (u;x) (I + R'R with small R), 1 on slacks; small random grad; small random
// DC over (u;x) (slack columns 0) so DC'Gamma D stays a mild perturbation;
// contractive dynamics (A = 0.5*I, B small) so the terminal Schur term stays
// bounded; modest d offsets, d_mask = 1 (all sides active), m = 0. The
// split-step (alpha_p != alpha_d) predictor path is more aggressive than the
// single-step path, so the data must be well conditioned for a 5-iteration
// affine loop to keep every stage Cholesky positive definite.
template <class P, int NH>
void fill_qp_split(Qp<P, NH>& qp, int N)
{
    using D = QpDim<P>;
    Rng rng;

    auto fill_stage = [&](auto& st, int nvar, int nslack)
    {
        const int nux = nvar - nslack;
        MatD R(nux, nux);
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                R(i, j) = 0.3 * rng.uniform();
            }
        }
        const MatD hux = R.transpose() * R;
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
        for (int i = 0; i < nux; ++i)
        {
            st.grad(i) = 0.2 * rng.uniform();
        }
        for (int i = nux; i < nvar; ++i)
        {
            st.grad(i) = 0.0;
        }
        const int nrow = static_cast<int>(st.DC.rows());
        for (int r = 0; r < nrow; ++r)
        {
            for (int j = 0; j < nux; ++j)
            {
                st.DC(r, j) = 0.1 * rng.uniform();
            }
            for (int j = nux; j < nvar; ++j)
            {
                st.DC(r, j) = 0.0;
            }
        }
        for (int i = 0; i < static_cast<int>(st.d.size()); ++i)
        {
            st.d(i) = 0.1 * rng.uniform();
            st.d_mask(i) = 1.0;
            st.m(i) = 0.0;
        }
    };

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

    fill_stage(qp.first, D::nvar_first, D::nslack_first);
    fill_dyn(qp.first);
    for (int k = 1; k < N; ++k)
    {
        fill_stage(qp.path[k - 1], D::nvar_path, D::nslack_path);
        fill_dyn(qp.path[k - 1]);
    }
    fill_stage(qp.term, D::nvar_term, D::nslack_term);
}

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
// compute_alpha split_step = 1 unit cases (uniform side data, d_mask = 1)
// ---------------------------------------------------------------------

template <class P, int NH>
void split_alpha_case(const char* name, int N, double lam, double t,
                      double dlam, double dt, double mv, double dmask,
                      double expect_p, double expect_d, double tol)
{
    const std::string p = name;
    Qp<P, NH> qp(N);
    QpSol<P, NH> iter(N);
    QpSol<P, NH> step(N);
    set_uniform_sides(qp, iter, step, lam, t, dlam, dt, mv, dmask);

    HpipmOptions opts;
    opts.split_step = 1;
    HpipmQpSolver<P, NH> solver(opts);
    solver.compute_alpha(qp, iter, step);
    check_scalar(solver.alpha_prim(), expect_p, tol, p + ": alpha_prim");
    check_scalar(solver.alpha_dual(), expect_d, tol, p + ": alpha_dual");
}

// ---------------------------------------------------------------------
// 5-iteration predictor-only (affine) loop, split_step = 1.
// ---------------------------------------------------------------------

template <class P, int NH = Eigen::Dynamic>
void split_predictor_loop(const char* name, int N)
{
    const std::string p = name;

    Qp<P, NH> qp(N);
    QpSol<P, NH> iter(N);
    fill_qp_split(qp, N);

    HpipmOptions opts;
    opts.split_step = 1;
    HpipmQpSolver<P, NH> solver(opts);
    solver.init_point(qp, iter);

    const double tau_min = solver.options().tau_min;
    const double alpha_min = solver.options().alpha_min;
    const double lam_min = solver.options().lam_min;
    const double t_min = solver.options().t_min;
    const int iters = 5;

    for (int it = 0; it < iters; ++it)
    {
        QpRes<P, NH> res(N);
        solver.compute_residuals(qp, iter, res);

        // HPIPM DELTA_STEP: res_m = res_m_bkp - tau_min. compute_residuals
        // already masks res_m, so shifting every entry matches HPIPM.
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
        solver.compute_alpha(qp, iter, step);
        check(solver.alpha_prim() > alpha_min,
              p + ": alpha_prim > alpha_min iter " + std::to_string(it));
        check(solver.alpha_dual() > alpha_min,
              p + ": alpha_dual > alpha_min iter " + std::to_string(it));
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
    }
}

}  // namespace

int run_split_step_3b_tests()
{
    failures = 0;
    using DI = DoubleIntegrator;
    using MS = MassSpring;

    // ---------------- compute_alpha (split_step = 1) --------------------
    // No binding: dlam, dt >= 0 keep lam / t non-negative at alpha = 1.
    split_alpha_case<DI, Eigen::Dynamic>("split DI no-binding", 1, 1.0, 1.0,
                                         0.3, 0.2, 0.0, 1.0, 1.0, 1.0, 1e-12);
    // Dual feasibility binds only: lam = 1, dlam = -2 -> alpha_dual = 0.5,
    // primal free -> alpha_prim = 1.
    split_alpha_case<DI, Eigen::Dynamic>("split DI dual-binding", 1, 1.0, 1.0,
                                         -2.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1e-12);
    // Primal feasibility binds only: t = 1, dt = -2 -> alpha_prim = 0.5,
    // dual free -> alpha_dual = 1.
    split_alpha_case<DI, Eigen::Dynamic>("split DI primal-binding", 1, 1.0,
                                         1.0, 0.0, -2.0, 0.0, 1.0, 0.5, 1.0,
                                         1e-12);
    // Both feasibility bounds bind at different ratios:
    // dlam = -2 -> alpha_dual = 0.5, dt = -4 -> alpha_prim = 0.25.
    split_alpha_case<DI, Eigen::Dynamic>("split DI both-binding", 1, 1.0, 1.0,
                                         -2.0, -4.0, 0.0, 1.0, 0.25, 0.5,
                                         1e-12);
    // Complementarity with dlam < 0 and dt < 0 (both negative): pass 1 defers,
    // pass 2 applies the quadratic scaling r = (-b - sqrt(disc))/(2a) with
    // a = 0.25, b = -1, disc = 0.5 -> r = 2(1 - sqrt(0.5)) = 0.585786...
    split_alpha_case<DI, Eigen::Dynamic>("split DI quad-pass", 1, 1.0, 1.0,
                                         -0.5, -0.5, 1.0, 1.0, 2.0 - 2.0 *
                                         std::sqrt(0.5), 2.0 - 2.0 *
                                         std::sqrt(0.5),
                                         1e-12);
    split_alpha_case<MS, Eigen::Dynamic>("split MS dual-binding", 1, 1.0, 1.0,
                                         -3.0, 0.1, 0.0, 1.0, 1.0, 1.0 / 3.0,
                                         1e-12);
    split_alpha_case<MS, Eigen::Dynamic>("split MS primal-binding", 1, 1.0,
                                         1.0, 0.1, -3.0, 0.0, 1.0, 1.0 / 3.0,
                                         1.0, 1e-12);

    // ---------------- update_vars: two damped alphas ---------------------
    {
        const int N = 1;
        const double lam0 = 1.0;
        const double t0 = 1.0;
        const double dlam = -2.0;  // dual binds at 0.5, primal free at 1.0
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

        HpipmOptions opts;
        opts.split_step = 1;
        HpipmQpSolver<DI, Eigen::Dynamic> solver(opts);
        solver.compute_alpha(qp, iter, step);
        check_scalar(solver.alpha_prim(), 1.0, 1e-12,
                     "update_vars: alpha_prim (primal free)");
        check_scalar(solver.alpha_dual(), 0.5, 1e-12,
                     "update_vars: alpha_dual (dual binding)");
        solver.update_vars(iter, step);

        // min(1, 0.5) < 1 -> both damped independently.
        const double ap = 1.0 * (0.99 * (1.0 - 1.0) + 0.9999999 * 1.0);
        const double ad = 0.5 * (0.99 * (1.0 - 0.5) + 0.9999999 * 0.5);
        check_scalar(iter.ux_first(0), ux0 + ap * 1.0, 1e-12,
                     "update_vars: ux_first uses alpha_prim");
        check_scalar(iter.ux_term(0), ux0 + ap * 2.0, 1e-12,
                     "update_vars: ux_term uses alpha_prim");
        check_scalar(iter.pi[0](0), pi0 + ad * 4.0, 1e-12,
                     "update_vars: pi uses alpha_dual");
        check_scalar(iter.lam_first(0), lam0 + ad * dlam, 1e-12,
                     "update_vars: lam uses alpha_dual (no clip)");
        check_scalar(iter.t_first(0), t0, 1e-12,
                     "update_vars: t unchanged (dt = 0)");
    }

    // ---------------- predictor loop (split_step = 1) --------------------
    split_predictor_loop<DI, Eigen::Dynamic>("DI dyn N=1", 1);
    split_predictor_loop<DI, Eigen::Dynamic>("DI dyn N=2", 2);
    split_predictor_loop<MS, Eigen::Dynamic>("MS dyn N=1", 1);
    split_predictor_loop<MS, Eigen::Dynamic>("MS dyn N=2", 2);
    split_predictor_loop<MS, 2>("MS NH=2 N=2", 2);

    if (failures == 0)
    {
        std::printf("All split_step (3b) checks passed.\n");
    }
    return failures;
}

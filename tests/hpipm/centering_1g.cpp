// Phase 1 sub-step 1g test: compute_mu_aff (HPIPM COMPUTE_MU_AFF_QP,
// x_core_qp_ipm_aux.c:636) and apply_centering (HPIPM
// COMPUTE_CENTERING_CORRECTION_QP / COMPUTE_CENTERING_QP,
// x_core_qp_ipm_aux.c:695/729).
//
// Hand-formula checks (uniform side data, DI N=1):
//   - mu_aff = (1/nc) * sum | -m + (lam + a*dlam)(t + a*dt) |
//   - corrector res_m = d_mask * (res_m_bkp + dt*dlam - sigma_mu)
//   - pure centering res_m = d_mask * (res_m_bkp - sigma_mu)
//   where sigma_mu = max((mu_aff/mu)^3 * mu, tau_min)
//   - sigma_mu floor: when sigma*mu < tau_min, sigma_mu = tau_min
//   - masked sides: res_m = 0 on d_mask = 0
//
// Pipeline check (mu_aff < mu after one affine step):
//   init_point -> compute_residuals -> res_m -= tau_min -> fact_solve_kkt
//   -> mask_step -> compute_alpha -> compute_mu_aff; on the toy QP the
//   affine step must reduce the barrier parameter.
//
// Run for DI (slacks) and MS (no slacks), N = 1, 2 (dynamic) and MS NH = 2.

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

template <class A, class B>
void check_close(const A& a, const B& b, double tol, const std::string& msg)
{
    const double err = (a - b).cwiseAbs().maxCoeff();
    if (err > tol)
    {
        std::fprintf(stderr, "FAIL: %s (err = %.3e)\n", msg.c_str(), err);
        ++failures;
    }
}

struct Rng
{
    std::mt19937_64 gen{123456789};
    double uniform()
    {
        return std::uniform_real_distribution<double>(-1.0, 1.0)(gen);
    }
    double pos()
    {
        return 0.1 + std::uniform_real_distribution<double>(0.0, 1.0)(gen);
    }
};

// ---------------------------------------------------------------------
// Set uniform side data across all stages.
// ---------------------------------------------------------------------

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

// ---------------------------------------------------------------------
// Hand-computed mu_aff: (1/nc_active) * sum_active | -m + (lam+a*dlam)
// * (t+a*dt) |
// ---------------------------------------------------------------------

template <class P, int NH>
double hand_mu_aff(const Qp<P, NH>& qp, const QpSol<P, NH>& iter,
                   const QpSol<P, NH>& step, double alpha)
{
    double sum = 0.0;
    int nc = 0;
    auto acc = [&](const auto& m, const auto& lam, const auto& t,
                   const auto& dlam, const auto& dt, const auto& dmask)
    {
        for (int i = 0; i < static_cast<int>(lam.size()); ++i)
        {
            if (dmask(i) > 0.5)
            {
                const double lam1 = lam(i) + alpha * dlam(i);
                const double t1 = t(i) + alpha * dt(i);
                sum += std::fabs(-m(i) + lam1 * t1);
                ++nc;
            }
        }
    };
    acc(qp.first.m, iter.lam_first, iter.t_first, step.lam_first,
        step.t_first, qp.first.d_mask);
    for (int k = 1; k < qp.N; ++k)
    {
        acc(qp.path[k - 1].m, iter.lam_path[k - 1], iter.t_path[k - 1],
            step.lam_path[k - 1], step.t_path[k - 1],
            qp.path[k - 1].d_mask);
    }
    acc(qp.term.m, iter.lam_term, iter.t_term, step.lam_term,
        step.t_term, qp.term.d_mask);
    return (nc > 0) ? sum / nc : 0.0;
}

// ---------------------------------------------------------------------
// Pipeline: fill data, init, one affine step, check mu_aff < mu.
// ---------------------------------------------------------------------

template <class P, int NH = Eigen::Dynamic>
void pipeline_case(const char* name, int N)
{
    using D = QpDim<P>;
    const std::string p = name;
    Rng rng;

    auto fill_stage = [&](auto& st, int nvar, int nslack,
                          const detail::QpLayout& lay)
    {
        const int nux = nvar - nslack;
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
        const int nside = lay.nside();
        for (int i = 0; i < nside; ++i)
        {
            st.d(i) = rng.uniform();
            st.d_mask(i) = (rng.pos() < 0.85) ? 1.0 : 0.0;
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

    Qp<P, NH> qp(N);
    QpSol<P, NH> sol(N);
    QpRes<P, NH> res(N);

    fill_stage(qp.first, D::nvar_first, D::nslack_first, D::lay_first);
    fill_dyn(qp.first);
    for (int k = 1; k < N; ++k)
    {
        fill_stage(qp.path[k - 1], D::nvar_path, D::nslack_path,
                   D::lay_path);
        fill_dyn(qp.path[k - 1]);
    }
    fill_stage(qp.term, D::nvar_term, D::nslack_term, D::lay_term);

    HpipmQpSolver<P, NH> solver;
    solver.init_point(qp, sol);
    solver.compute_residuals(qp, sol, res);
    const double mu0 = res.res_mu;

    const double tau_min = solver.options().tau_min;
    res.res_m_first.array() -= tau_min;
    for (int k = 1; k < N; ++k)
    {
        res.res_m_path[k - 1].array() -= tau_min;
    }
    res.res_m_term.array() -= tau_min;

    QpSol<P, NH> step(N);
    const Status st = solver.fact_solve_kkt(qp, sol, res, step);
    check(st == Status::kSolved, p + ": fact_solve_kkt");
    if (st != Status::kSolved)
    {
        return;
    }
    solver.mask_step(qp, step);
    const double alpha = solver.compute_alpha(qp, sol, step);
    check(alpha > solver.options().alpha_min, p + ": alpha > alpha_min");
    const double mu_aff = solver.compute_mu_aff(qp, sol, step);
    check(mu_aff < mu0,
          p + ": mu_aff (" + std::to_string(mu_aff) + ") < mu (" +
              std::to_string(mu0) + ")");
    const double mu_aff_ref = hand_mu_aff(qp, sol, step, alpha);
    check_scalar(mu_aff, mu_aff_ref, 1e-12, p + ": mu_aff hand formula");
}

}  // namespace

int run_centering_1g_tests()
{
    failures = 0;
    using DI = DoubleIntegrator;
    using MS = MassSpring;
    constexpr int NH = Eigen::Dynamic;
    const double tau_min = HpipmOptions{}.tau_min;

    // ---------- compute_mu_aff hand formula (m = 0) ----------------------
    {
        const int N = 1;
        Qp<DI, NH> qp(N);
        QpSol<DI, NH> iter(N);
        QpSol<DI, NH> step(N);
        set_uniform_sides(qp, iter, step, 1.0, 2.0, -0.5, -0.3, 0.0, 1.0);

        HpipmQpSolver<DI, NH> solver;
        const double alpha = solver.compute_alpha(qp, iter, step);
        check_scalar(alpha, 1.0, 1e-12, "mu_aff m=0: alpha = 1 (no binding)");
        const double mu_aff = solver.compute_mu_aff(qp, iter, step);
        // |0 + (1 + 1*(-0.5)) * (2 + 1*(-0.3))| = |0.5 * 1.7| = 0.85
        check_scalar(mu_aff, 0.85, 1e-12, "mu_aff m=0: hand value");
    }

    // ---------- compute_mu_aff hand formula (m != 0) --------------------
    {
        const int N = 1;
        Qp<DI, NH> qp(N);
        QpSol<DI, NH> iter(N);
        QpSol<DI, NH> step(N);
        set_uniform_sides(qp, iter, step, 1.0, 2.0, -0.5, -0.3, 0.3, 1.0);

        HpipmQpSolver<DI, NH> solver;
        const double alpha = solver.compute_alpha(qp, iter, step);
        check_scalar(alpha, 1.0, 1e-12, "mu_aff m=0.3: alpha = 1");
        const double mu_aff = solver.compute_mu_aff(qp, iter, step);
        // |-0.3 + (1 - 0.5) * (2 - 0.3)| = |-0.3 + 0.85| = 0.55
        check_scalar(mu_aff, 0.55, 1e-12, "mu_aff m=0.3: hand value");
    }

    // ---------- compute_mu_aff: masked sides contribute 0 ----------------
    {
        const int N = 1;
        Qp<DI, NH> qp(N);
        QpSol<DI, NH> iter(N);
        QpSol<DI, NH> step(N);
        set_uniform_sides(qp, iter, step, 1.0, 2.0, -0.5, -0.3, 0.0, 1.0);
        for (int i = 1;
             i < static_cast<int>(qp.first.d_mask.size()); i += 2)
        {
            qp.first.d_mask(i) = 0.0;
        }
        for (int i = 1; i < static_cast<int>(qp.term.d_mask.size()); i += 2)
        {
            qp.term.d_mask(i) = 0.0;
        }

        HpipmQpSolver<DI, NH> solver;
        const double alpha = solver.compute_alpha(qp, iter, step);
        const double mu_aff = solver.compute_mu_aff(qp, iter, step);
        // Active sides all have the same value: 0.5 * 1.7 = 0.85
        check_scalar(mu_aff, 0.85, 1e-12, "mu_aff masked: active-only");
        const double ref = hand_mu_aff(qp, iter, step, alpha);
        check_scalar(mu_aff, ref, 1e-12, "mu_aff masked: hand formula");
    }

    // ---------- apply_centering: corrector hand check --------------------
    {
        constexpr int N = 1;
        const double lam = 1.0, t = 2.0, dlam = -0.5, dt = -0.3;
        const double mu_val = 1.0;
        const double rm_bkp = 0.1;

        Qp<DI, NH> qp(N);
        QpSol<DI, NH> iter(N);
        QpSol<DI, NH> step(N);
        set_uniform_sides(qp, iter, step, lam, t, dlam, dt, 0.0, 1.0);

        HpipmQpSolver<DI, NH> solver;
        solver.compute_alpha(qp, iter, step);
        solver.compute_mu_aff(qp, iter, step);  // mu_aff = 0.85

        QpRes<DI, NH> res_bkp(N);
        res_bkp.res_m_first.setConstant(rm_bkp);
        res_bkp.res_m_term.setConstant(rm_bkp);
        res_bkp.res_mu = mu_val;
        QpRes<DI, NH> res_out(N);

        solver.apply_centering(qp, res_bkp, step, /*correction=*/true,
                               res_out);
        const double sigma = 0.85 * 0.85 * 0.85;
        const double sigma_mu = std::max(sigma * mu_val, tau_min);
        const double expect = rm_bkp + dt * dlam - sigma_mu;
        constexpr int nside_f = QpDim<DI>::nside_first;
        constexpr int nside_t = QpDim<DI>::nside_term;
        check_close(res_out.res_m_first,
                    Eigen::Matrix<double, nside_f, 1>::Constant(expect),
                    1e-12, "centering corr: res_m_first");
        check_close(res_out.res_m_term,
                    Eigen::Matrix<double, nside_t, 1>::Constant(expect),
                    1e-12, "centering corr: res_m_term");
    }

    // ---------- apply_centering: pure centering hand check ---------------
    {
        constexpr int N = 1;
        const double lam = 1.0, t = 2.0, dlam = -0.5, dt = -0.3;
        const double mu_val = 1.0;
        const double rm_bkp = 0.1;

        Qp<DI, NH> qp(N);
        QpSol<DI, NH> iter(N);
        QpSol<DI, NH> step(N);
        set_uniform_sides(qp, iter, step, lam, t, dlam, dt, 0.0, 1.0);

        HpipmQpSolver<DI, NH> solver;
        solver.compute_alpha(qp, iter, step);
        solver.compute_mu_aff(qp, iter, step);

        QpRes<DI, NH> res_bkp(N);
        res_bkp.res_m_first.setConstant(rm_bkp);
        res_bkp.res_m_term.setConstant(rm_bkp);
        res_bkp.res_mu = mu_val;
        QpRes<DI, NH> res_out(N);

        solver.apply_centering(qp, res_bkp, step, /*correction=*/false,
                               res_out);
        const double sigma = 0.85 * 0.85 * 0.85;
        const double sigma_mu = std::max(sigma * mu_val, tau_min);
        const double expect = rm_bkp - sigma_mu;
        constexpr int nside_f = QpDim<DI>::nside_first;
        check_close(res_out.res_m_first,
                    Eigen::Matrix<double, nside_f, 1>::Constant(expect),
                    1e-12, "centering pure: res_m_first");
    }

    // ---------- apply_centering: sigma_mu floor ---------------------------
    {
        const int N = 1;
        // dlam = -lam/alpha => lam1 = 0 => mu_aff = 0 => sigma_mu = tau_min
        const double lam = 1.0, t = 2.0, dlam = -1.0, dt = 0.0;
        const double mu_val = 1.0;
        const double rm_bkp = 0.5;

        Qp<DI, NH> qp(N);
        QpSol<DI, NH> iter(N);
        QpSol<DI, NH> step(N);
        set_uniform_sides(qp, iter, step, lam, t, dlam, dt, 0.0, 1.0);

        HpipmQpSolver<DI, NH> solver;
        const double alpha = solver.compute_alpha(qp, iter, step);
        check_scalar(alpha, 1.0, 1e-12, "sigma floor: alpha");
        const double mu_aff = solver.compute_mu_aff(qp, iter, step);
        check_scalar(mu_aff, 0.0, 1e-15, "sigma floor: mu_aff = 0");

        QpRes<DI, NH> res_bkp(N);
        res_bkp.res_m_first.setConstant(rm_bkp);
        res_bkp.res_m_term.setConstant(rm_bkp);
        res_bkp.res_mu = mu_val;
        QpRes<DI, NH> res_out(N);

        solver.apply_centering(qp, res_bkp, step, /*correction=*/true,
                               res_out);
        const double expect = rm_bkp + dt * dlam - tau_min;
        constexpr int nside_f = QpDim<DI>::nside_first;
        check_close(res_out.res_m_first,
                    Eigen::Matrix<double, nside_f, 1>::Constant(expect),
                    1e-14, "sigma floor: res_m_first");
    }

    // ---------- apply_centering: masked sides are zero -------------------
    {
        const int N = 1;
        Qp<DI, NH> qp(N);
        QpSol<DI, NH> iter(N);
        QpSol<DI, NH> step(N);
        set_uniform_sides(qp, iter, step, 1.0, 1.0, -0.2, -0.2, 0.0, 1.0);
        for (int i = 1; i < static_cast<int>(qp.first.d_mask.size()); i += 2)
        {
            qp.first.d_mask(i) = 0.0;
        }

        HpipmQpSolver<DI, NH> solver;
        solver.compute_alpha(qp, iter, step);
        solver.compute_mu_aff(qp, iter, step);

        QpRes<DI, NH> res_bkp(N);
        res_bkp.res_m_first.setConstant(0.5);
        res_bkp.res_m_term.setConstant(0.5);
        res_bkp.res_mu = 1.0;
        QpRes<DI, NH> res_out(N);

        solver.apply_centering(qp, res_bkp, step, /*correction=*/true,
                               res_out);
        for (int i = 0; i < static_cast<int>(res_out.res_m_first.size());
             ++i)
        {
            if (qp.first.d_mask(i) < 0.5)
            {
                check(res_out.res_m_first(i) == 0.0,
                      "masking: res_m_first[" + std::to_string(i) +
                          "] = 0 on masked side");
            }
        }
    }

    // ---------- pipeline: mu_aff < mu after one affine step --------------
    pipeline_case<DI>("DI dyn N=1", 1);
    pipeline_case<DI>("DI dyn N=2", 2);
    pipeline_case<MS>("MS dyn N=1", 1);
    pipeline_case<MS>("MS dyn N=2", 2);
    pipeline_case<MS, 2>("MS NH=2 N=2", 2);

    if (failures == 0)
    {
        std::printf("All centering (1g) checks passed.\n");
    }
    return failures;
}

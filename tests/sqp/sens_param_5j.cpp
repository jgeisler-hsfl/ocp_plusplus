// Phase 5 test 5j: NLP-level parameter sensitivity in the condensing QP
// modes (kFullCondensed / kPartialCondensed).
//
// Verifies AcadosSolver::sens_param (staged 5i, extended to the condensed
// QP forms by 5j, plan
// SQP_HPIPM_PARITY_PHASE5_5J_SENS_PARAM_CONDENSING_PLAN.md): the staged
// sensitivity QP (converged LHS + staged seed as RHS) is condensed with
// the same flow-map machinery as the QP data, and the condensed QP is
// seed-solved.
//
// For each parameter p_j with dp = e_j:
//   K_j = d(x*, u*)/dp_j  from one sens_param call,
//   FD  = (sol(p + h e_j) - sol(p - h e_j)) / (2 h)   (full SQP re-solve
//         in the SAME qp_mode).
// Plus a cross-check of the condensed sens_param against the staged
// sens_param on the respective converged base solutions.  The
// DoubleIntegrator's x_0 is p-independent and pinned, so its response
// component must stay exactly zero.
//
// FD step / tolerance per parameter: same calibration as 5i (Ts is
// quadratic in the dynamics -> h = 1e-5, tol 1e-5; the other parameters
// enter linearly -> h = 1e-4 * max(1, |p|), tol 1e-6).

#include <cmath>
#include <cstdio>
#include <string>

#include "ocp/solvers/acados/acados.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"

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

// p = (Ts, v_max, wq, wv, wa, wf)
double get_p(const DoubleIntegrator& p, int j)
{
    switch (j)
    {
    case 0:
        return p.Ts_;
    case 1:
        return p.v_max_;
    case 2:
        return p.wq_;
    case 3:
        return p.wv_;
    case 4:
        return p.wa_;
    case 5:
        return p.wf_;
    }
    return 0.0;
}

void set_p(DoubleIntegrator& p, int j, double v)
{
    switch (j)
    {
    case 0:
        p.Ts_ = v;
        break;
    case 1:
        p.v_max_ = v;
        break;
    case 2:
        p.wq_ = v;
        break;
    case 3:
        p.wv_ = v;
        break;
    case 4:
        p.wa_ = v;
        break;
    case 5:
        p.wf_ = v;
        break;
    default:
        break;
    }
}

Solution<DoubleIntegrator> make_start(const DoubleIntegrator& p, int N)
{
    // Constant control that lands exactly on q_N + v_N = 1 (feasible for
    // |u| <= 1 at N >= 10); same construction as sens_param_5i.
    const double Ts = p.Ts_;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * N * (N - 1) / 2.0 + Ts * N;
    const double a = (1.0 - (q0 + v0) - Ts * N * v0) / den;
    Solution<DoubleIntegrator> sol(N);
    sol.x[0] = p.initial_state();
    for (int k = 0; k < N; ++k)
    {
        sol.u[k](0) = a;
        sol.x[k + 1] = p.dynamics_next_state(k, sol.x[k], sol.u[k]);
    }
    return sol;
}

// k-th (du, dx) block of a QpSol response; ux layout is (u; x; s) per
// stage (ux_term is (x; s)).
void ux_block(const QpSol<DoubleIntegrator, Eigen::Dynamic>& s, int N, int k,
              Eigen::VectorXd& du, Eigen::VectorXd& dx)
{
    using D = QpDim<DoubleIntegrator>;
    if (k == 0)
    {
        du = s.ux_first.head(D::nu);
        dx = s.ux_first.segment(D::nu, D::nx);
    }
    else if (k < N)
    {
        du = s.ux_path[k - 1].head(D::nu);
        dx = s.ux_path[k - 1].segment(D::nu, D::nx);
    }
    else
    {
        du = Eigen::VectorXd::Zero(D::nu);
        dx = s.ux_term.head(D::nx);
    }
}

// Full per-parameter check in one qp_mode: analytic sens_param, dx_0 == 0,
// cross-check vs the staged sens_param, and central FD of the full SQP
// re-solve in the same mode.
template <class SolvT>
void run_mode(const std::string& name, const AcadosOptions& opts, int N)
{
    const int np = DoubleIntegratorDims::np;

    DoubleIntegrator p;
    Solution<DoubleIntegrator> sol = make_start(p, N);
    SolvT solver(opts);
    const Status st = solver.solve(p, sol);
    check(st == Status::kSolved, name + "base solve kSolved (got " +
                                     std::to_string(static_cast<int>(st)) +
                                     ")");
    if (st != Status::kSolved)
    {
        return;
    }

    // staged reference on its own converged solution
    AcadosOptions opts_st;
    AcadosSolver<DoubleIntegrator, Eigen::Dynamic> solver_st(opts_st);
    Solution<DoubleIntegrator> sol_st = make_start(p, N);
    check(solver_st.solve(p, sol_st) == Status::kSolved,
          name + "staged base solve kSolved");
    if (failures > 0)
    {
        return;
    }

    for (int j = 0; j < np; ++j)
    {
        const std::string pfx = name + "p_" + std::to_string(j);
        const double pv = get_p(p, j);
        // Ts (j == 0) enters the dynamics twice (x_{k+1} = x_k + Ts*v_k +
        // 0.5*Ts^2*u_k), so the central-FD truncation error at h = 1e-4 is
        // ~5e-5; use h = 1e-5 there with a matching relaxed tolerance.
        // Same calibration as 5i.
        const double h = (j == 0) ? 1e-5 : 1e-4 * std::max(1.0, std::fabs(pv));
        const double tol_j = (j == 0) ? 1e-5 : 1e-6;

        // analytic: dp = e_j
        DoubleIntegrator::param_t dp;
        dp.setZero();
        dp(j) = 1.0;
        QpSol<DoubleIntegrator, Eigen::Dynamic> dsol(N);
        const Status ss = solver.sens_param(p, sol, dp, dsol);
        check(ss == Status::kSolved,
              pfx + ": sens_param kSolved (got " +
                  std::to_string(static_cast<int>(ss)) + ")");
        if (ss != Status::kSolved)
        {
            continue;
        }

        // x_0 is p-independent and pinned: it must not move.
        const Eigen::VectorXd dx0_an =
            dsol.ux_first.segment(QpDim<DoubleIntegrator>::nu,
                                  QpDim<DoubleIntegrator>::nx);
        check(dx0_an.cwiseAbs().maxCoeff() < 1e-12,
              pfx + ": dx_0 == 0 (p-independent pinned initial state, got " +
                  std::to_string(dx0_an.cwiseAbs().maxCoeff()) + ")");

        // cross-check: condensed sens vs staged sens on the (mode-
        // converged) base solutions
        QpSol<DoubleIntegrator, Eigen::Dynamic> dsol_st(N);
        check(solver_st.sens_param(p, sol_st, dp, dsol_st) == Status::kSolved,
              pfx + ": staged sens_param kSolved");
        if (failures > 0)
        {
            return;
        }
        double xdiff = 0.0;
        for (int k = 0; k <= N; ++k)
        {
            Eigen::VectorXd du_m, dx_m, du_s, dx_s;
            ux_block(dsol, N, k, du_m, dx_m);
            ux_block(dsol_st, N, k, du_s, dx_s);
            xdiff = std::max(xdiff, (du_m - du_s).cwiseAbs().maxCoeff());
            xdiff = std::max(xdiff, (dx_m - dx_s).cwiseAbs().maxCoeff());
        }
        check(xdiff < 1e-6, pfx + ": vs staged sens agree < 1e-6 (got " +
                                std::to_string(xdiff) + ")");

        // finite differences: p_j +/- h, full re-solve in the same mode
        DoubleIntegrator pp = p, pm = p;
        set_p(pp, j, pv + h);
        set_p(pm, j, pv - h);
        Solution<DoubleIntegrator> sol_p = make_start(pp, N);
        Solution<DoubleIntegrator> sol_m = make_start(pm, N);
        SolvT sp(opts);
        SolvT sm(opts);
        check(sp.solve(pp, sol_p) == Status::kSolved, pfx + ": FD+ kSolved");
        check(sm.solve(pm, sol_m) == Status::kSolved, pfx + ": FD- kSolved");
        if (failures > 0)
        {
            return;
        }

        double max_du = 0.0;
        double max_dx = 0.0;
        for (int k = 0; k <= N; ++k)
        {
            Eigen::VectorXd du_an, dx_an;
            ux_block(dsol, N, k, du_an, dx_an);
            const Eigen::VectorXd dx_fd =
                (sol_p.x[k].cast<double>() - sol_m.x[k].cast<double>()) /
                (2.0 * h);
            if (k < N)
            {
                const Eigen::VectorXd du_fd =
                    (sol_p.u[k].cast<double>() -
                     sol_m.u[k].cast<double>()) /
                    (2.0 * h);
                max_du = std::max(max_du,
                                  (du_an - du_fd).cwiseAbs().maxCoeff());
            }
            max_dx = std::max(max_dx, (dx_an - dx_fd).cwiseAbs().maxCoeff());
        }
        check(max_du < tol_j, pfx + ": du/dp matches FD < " +
                                  std::to_string(tol_j) + " (got " +
                                  std::to_string(max_du) + ")");
        check(max_dx < tol_j, pfx + ": dx/dp matches FD < " +
                                  std::to_string(tol_j) + " (got " +
                                  std::to_string(max_dx) + ")");
        std::printf("  %s: max |du diff| = %.3e, max |dx diff| = %.3e\n",
                    pfx.c_str(), max_du, max_dx);
    }
}

void test_sens_param_dense()
{
    AcadosOptions opts;
    opts.qp_mode = QpMode::kFullCondensed;
    run_mode<AcadosSolver<DoubleIntegrator, Eigen::Dynamic>>(
        "sens_param dense DI N=10 ", opts, 10);
}

void test_sens_param_block()
{
    using BlockSolv = AcadosSolver<DoubleIntegrator, Eigen::Dynamic,
                                   HpipmQpSolver<DoubleIntegrator,
                                                 Eigen::Dynamic>,
                                   GlmRegularizer<DoubleIntegrator,
                                                  Eigen::Dynamic>,
                                   MeritBacktracking<DoubleIntegrator,
                                                     Eigen::Dynamic>,
                                   2>;
    AcadosOptions opts;
    opts.qp_mode = QpMode::kPartialCondensed;
    opts.qp_cond_N = 5;  // 5 blocks of NB = 2
    run_mode<BlockSolv>("sens_param block DI N=10 ", opts, 10);
}

}  // namespace

int run_sens_param_5j_tests()
{
    test_sens_param_dense();
    test_sens_param_block();
    if (failures == 0)
    {
        std::printf("All sens_param (5j) checks passed.\n");
    }
    return failures;
}

// Phase 5 test 5i: NLP-level parameter sensitivity (staged mode).
//
// Verifies AcadosSolver::sens_param (SQP_HPIPM_PARITY_PHASE5_5I plan)
// against central finite differences of the full SQP re-solve:
//   K_j = d(x*, u*)/dp_j  from one sens_param call (dp = e_j),
//   FD  = (sol(p + h e_j) - sol(p - h e_j)) / (2 h).
// The DoubleIntegrator's x_0 is p-independent and pinned, so its
// response component must stay exactly zero.  5j made the condensing
// modes work (they must return kSolved; the dense/block FD checks live
// in sens_param_5j.cpp).  A zero direction must give a zero response
// (zero seed -> zero step).
//
// FD step / tolerance per parameter:
//   p_0 (Ts): enters the dynamics quadratically, so central-FD truncation
//     error at h = 1e-4 is ~5e-5; use h = 1e-5 (observed diff ~2e-6) with
//     tol 1e-5.  All other parameters enter linearly; h = 1e-4 *
//     max(1,|p|) and tol 1e-6 suffice.
//   p_1 (v_max): the stage inequality g = v - v_max is inactive at the
//     converged solution (margin ~1.5), so the exact first-order response
//     is zero (active-set gating in fill_param_seed drops the d-seed on
//     inactive sides); FD confirms ~0.

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
        return p.params().Ts;
    case 1:
        return p.params().v_max;
    case 2:
        return p.params().w_stage(0);
    case 3:
        return p.params().w_stage(1);
    case 4:
        return p.params().w_stage(2);
    case 5:
        return p.params().wf;
    }
    return 0.0;
}

void set_p(DoubleIntegrator& p, int j, double v)
{
    switch (j)
    {
    case 0:
        p.params().Ts = v;
        break;
    case 1:
        p.params().v_max = v;
        break;
    case 2:
        p.params().w_stage(0) = v;
        break;
    case 3:
        p.params().w_stage(1) = v;
        break;
    case 4:
        p.params().w_stage(2) = v;
        break;
    case 5:
        p.params().wf = v;
        break;
    default:
        break;
    }
}

Solution<DoubleIntegrator> make_start(const DoubleIntegrator& p, int N)
{
    // Constant control that lands exactly on q_N + v_N = 1 (feasible for
    // |u| <= 1 at N >= 10); same construction as sens_x0_5f.
    const double Ts = p.params().Ts;
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

// dp = 0 gives a zero seed, hence a zero KKT response.
void test_sens_param_zero_dp()
{
    const std::string name = "sens_param DI N=10 dp=0 ";
    const int N = 10;
    DoubleIntegrator p;
    Solution<DoubleIntegrator> sol = make_start(p, N);
    AcadosSolver<DoubleIntegrator, Eigen::Dynamic> solver;
    if (solver.solve(p, sol) != Status::kSolved)
    {
        check(false, name + ": base solve kSolved");
        return;
    }
    DoubleIntegrator::param_t dp_zero;
    dp_zero.setZero();
    QpSol<DoubleIntegrator, Eigen::Dynamic> dsol(N);
    const Status ss = solver.sens_param(p, sol, dp_zero, dsol);
    check(ss == Status::kSolved,
          name + ": sens_param kSolved (got " +
              std::to_string(static_cast<int>(ss)) + ")");
    if (ss != Status::kSolved)
    {
        return;
    }
    double mx = 0.0;
    for (int k = 0; k <= N; ++k)
    {
        Eigen::VectorXd du, dx;
        ux_block(dsol, N, k, du, dx);
        mx = std::max(mx, du.cwiseAbs().maxCoeff());
        mx = std::max(mx, dx.cwiseAbs().maxCoeff());
    }
    check(mx < 1e-12,
          name + ": zero response for dp = 0 (got " + std::to_string(mx) +
              ")");
}

// 5j: the condensing modes must now solve (the full FD checks for them
// live in sens_param_5j.cpp); here we only check status + a sane,
// non-trivial response for Ts (p_0).
void test_sens_param_condensed_modes_solved()
{
    const std::string name = "sens_param DI N=10 non-staged ";
    const int N = 10;
    DoubleIntegrator p;
    Solution<DoubleIntegrator> sol = make_start(p, N);

    for (int dense = 0; dense < 2; ++dense)
    {
        AcadosOptions opts;
        if (dense)
        {
            opts.qp_mode = QpMode::kFullCondensed;
        }
        else
        {
            opts.qp_mode = QpMode::kPartialCondensed;
            // NB == 0 default solver: 5j falls back to the staged path.
        }
        AcadosSolver<DoubleIntegrator, Eigen::Dynamic> solver(opts);
        if (solver.solve(p, sol) != Status::kSolved)
        {
            check(false, name + (dense ? "dense" : "block-fallback") +
                          ": base solve kSolved");
            return;
        }
        DoubleIntegrator::param_t dp;
        dp.setZero();
        dp(0) = 1.0;
        QpSol<DoubleIntegrator, Eigen::Dynamic> dsol(N);
        const Status ss = solver.sens_param(p, sol, dp, dsol);
        const std::string tag =
            name + (dense ? "kFullCondensed" : "kPartialCondensed (NB=0)");
        check(ss == Status::kSolved,
              tag + " -> kSolved (got " + std::to_string(static_cast<int>(ss)) +
                  ")");
        if (ss != Status::kSolved)
        {
            return;
        }
        // dx_0 must stay exactly zero (p-independent pinned state).
        const Eigen::VectorXd dx0 =
            dsol.ux_first.segment(QpDim<DoubleIntegrator>::nu,
                                  QpDim<DoubleIntegrator>::nx);
        check(dx0.cwiseAbs().maxCoeff() < 1e-12,
              tag + ": dx_0 == 0 (got " +
                  std::to_string(dx0.cwiseAbs().maxCoeff()) + ")");
        // Ts moves the trajectory: at least one du/dTs entry must be
        // O(1), not numerically zero.
        double mx = 0.0;
        for (int k = 0; k < N; ++k)
        {
            Eigen::VectorXd du, dx;
            ux_block(dsol, N, k, du, dx);
            mx = std::max(mx, du.cwiseAbs().maxCoeff());
        }
        check(mx > 1e-3,
              tag + ": d(u)/dTs is non-trivial (got max |du| = " +
                  std::to_string(mx) + ")");
    }
}

void test_sens_param_fd()
{
    const std::string name = "sens_param DI N=10 ";
    const int N = 10;
    const int np = DoubleIntegratorDims::np;
    const double tol = 1e-6;

    DoubleIntegrator p;
    Solution<DoubleIntegrator> sol = make_start(p, N);
    AcadosOptions opts;  // kStaged (default)
    AcadosSolver<DoubleIntegrator, Eigen::Dynamic> solver(opts);
    const Status st = solver.solve(p, sol);
    check(st == Status::kSolved,
          name + "base solve kSolved (got " +
              std::to_string(static_cast<int>(st)) + ")");
    if (st != Status::kSolved)
    {
        return;
    }

    for (int j = 0; j < np; ++j)
    {
        const std::string pfx = name + "p_" + std::to_string(j);
        const double pv = get_p(p, j);
        // Ts (j == 0) enters the dynamics twice (x_{k+1} = x_k + Ts*v_k +
        // 0.5*Ts^2*u_k), so the central-FD truncation error at h = 1e-4 is
        // ~5e-5; use h = 1e-5 there (observed diff ~2e-6) with a matching
        // relaxed tolerance. The weights enter the cost linearly, so the
        // default step is sufficient.
        const double h = (j == 0) ? 1e-5 : 1e-4 * std::max(1.0, std::fabs(pv));
        const double tol_j = (j == 0) ? 1e-5 : tol;

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

        // finite differences: p_j +/- h, full re-solve
        DoubleIntegrator pp = p, pm = p;
        set_p(pp, j, pv + h);
        set_p(pm, j, pv - h);
        Solution<DoubleIntegrator> sol_p = make_start(pp, N);
        Solution<DoubleIntegrator> sol_m = make_start(pm, N);
        AcadosSolver<DoubleIntegrator, Eigen::Dynamic> sp(opts);
        AcadosSolver<DoubleIntegrator, Eigen::Dynamic> sm(opts);
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

}  // namespace

int run_sens_param_5i_tests()
{
    test_sens_param_zero_dp();
    test_sens_param_condensed_modes_solved();
    test_sens_param_fd();
    if (failures == 0)
    {
        std::printf("All sens_param (5i) checks passed.\n");
    }
    return failures;
}

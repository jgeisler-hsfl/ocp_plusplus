// Phase 5 test 5f: OCP-level sensitivity w.r.t. the fixed initial state.
//
// Verifies AcadosSolver::sens_x0 (ARCHITECTURE.md sec. 8) against central
// finite differences of the full SQP solve:
//   K_i = d(x*, u*)/dx0 * e_i  from one sens_x0 call,
//   FD  = (sol(x0 + h e_i) - sol(x0 - h e_i)) / (2 h).
// The double integrator has linear dynamics and a quadratic cost, so the
// optimal policy is affine in x0 and both routes must agree closely
// (limited only by the SQP/QP tolerances and the finite-difference step).
// The pinned initial state must also be returned exactly: dx_0 = e_i.

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

// DoubleIntegrator with a settable initial state (the example's x0 is
// hardcoded to [1; 0.5]).
struct DiSens : DoubleIntegrator
{
    state_t x0_;

    DiSens()
    {
        x0_ << 1.0, 0.5;
    }

    state_t initial_state() const { return x0_; }
};

double di_feasible_control(const DiSens& p, int n_stages)
{
    const double Ts = p.Ts_;
    const double q0 = p.initial_state()(0);
    const double v0 = p.initial_state()(1);
    const double den = Ts * Ts * n_stages * (n_stages - 1) / 2.0 + Ts * n_stages;
    return (1.0 - (q0 + v0) - Ts * n_stages * v0) / den;
}

Solution<DiSens> make_start(const DiSens& p, int N)
{
    const double a = di_feasible_control(p, N);
    Solution<DiSens> sol(N);
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
void ux_block(const QpSol<DiSens, Eigen::Dynamic>& s, int N, int k,
              Eigen::VectorXd& du, Eigen::VectorXd& dx)
{
    using D = QpDim<DiSens>;
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

void test_sens_x0_fd()
{
    const std::string name = "sens_x0 DI N=10 ";
    const int N = 10;  // N < 7 is infeasible (terminal eq unreachable
                       // with |u| <= 1 from x0 = [1; 0.5])
    const double tol = 1e-6;
    const double h = 1e-6;

    DiSens p;
    Solution<DiSens> sol = make_start(p, N);
    AcadosOptions opts;
    AcadosSolver<DiSens, Eigen::Dynamic> solver(opts);
    const Status st = solver.solve(p, sol);
    check(st == Status::kSolved, name + ": base solve kSolved (got " +
                                      std::to_string(static_cast<int>(st)) +
                                      ")");
    if (st != Status::kSolved)
    {
        return;
    }

    for (int i = 0; i < QpDim<DiSens>::nx; ++i)
    {
        const std::string pfx = name + " e_" + std::to_string(i);

        // analytic: dx0 = e_i
        DiSens::state_t e_i;
        e_i.setZero();
        e_i(i) = 1.0;
        QpSol<DiSens, Eigen::Dynamic> dsol(N);
        const Status ss = solver.sens_x0(p, sol, e_i, dsol);
        check(ss == Status::kSolved, pfx + ": sens_x0 kSolved (got " +
                                         std::to_string(static_cast<int>(ss)) +
                                         ")");
        if (ss != Status::kSolved)
        {
            continue;
        }

        // the pinned initial state must move exactly by dx0
        Eigen::VectorXd dx0_an = dsol.ux_first.segment(QpDim<DiSens>::nu,
                                                       QpDim<DiSens>::nx);
        check((dx0_an - e_i).cwiseAbs().maxCoeff() < 1e-12,
              pfx + ": dx_0 == e_i (pinned initial state)");

        // finite differences: x0 +/- h*e_i, re-solved
        DiSens pp = p;
        pp.x0_(i) += h;
        DiSens pm = p;
        pm.x0_(i) -= h;
        Solution<DiSens> sol_p = make_start(pp, N);
        Solution<DiSens> sol_m = make_start(pm, N);
        AcadosSolver<DiSens, Eigen::Dynamic> sp(opts);
        AcadosSolver<DiSens, Eigen::Dynamic> sm(opts);
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
            Eigen::VectorXd dx_fd =
                (sol_p.x[k].cast<double>() - sol_m.x[k].cast<double>()) /
                (2.0 * h);
            if (k < N)
            {
                const Eigen::VectorXd du_fd =
                    (sol_p.u[k].cast<double>() -
                     sol_m.u[k].cast<double>()) / (2.0 * h);
                max_du = std::max(max_du,
                                  (du_an - du_fd).cwiseAbs().maxCoeff());
            }
            max_dx = std::max(max_dx, (dx_an - dx_fd).cwiseAbs().maxCoeff());
        }
        check(max_du < tol, pfx + ": du/dx0 matches FD < " +
                                std::to_string(tol) + " (got " +
                                std::to_string(max_du) + ")");
        check(max_dx < tol, pfx + ": dx/dx0 matches FD < " +
                                std::to_string(tol) + " (got " +
                                std::to_string(max_dx) + ")");
        std::printf("  %s: max |du diff| = %.3e, max |dx diff| = %.3e\n",
                    pfx.c_str(), max_du, max_dx);
    }
}

// 5g: the same FD check with qp_mode = kFullCondensed (dense QP + dense
// sens path, hard-equality block KKT), plus a cross-check of the dense
// sens_x0 against the staged sens_x0 on the same base solution.
void test_sens_x0_fd_dense()
{
    const std::string name = "sens_x0 dense DI N=10 ";
    const int N = 10;
    const double tol = 1e-6;
    const double h = 1e-6;

    DiSens p;
    Solution<DiSens> sol = make_start(p, N);
    AcadosOptions opts;
    opts.qp_mode = QpMode::kFullCondensed;
    AcadosSolver<DiSens, Eigen::Dynamic> solver(opts);
    const Status st = solver.solve(p, sol);
    check(st == Status::kSolved, name + " base solve kSolved (got " +
                                      std::to_string(static_cast<int>(st)) +
                                      ")");
    if (st != Status::kSolved)
    {
        return;
    }

    // staged reference (same base iterate)
    AcadosOptions opts_st;
    AcadosSolver<DiSens, Eigen::Dynamic> solver_st(opts_st);
    Solution<DiSens> sol_st = make_start(p, N);
    check(solver_st.solve(p, sol_st) == Status::kSolved,
          name + " staged base solve kSolved");

    for (int i = 0; i < QpDim<DiSens>::nx; ++i)
    {
        const std::string pfx = name + " e_" + std::to_string(i);

        DiSens::state_t e_i;
        e_i.setZero();
        e_i(i) = 1.0;
        QpSol<DiSens, Eigen::Dynamic> dsol(N);
        const Status ss = solver.sens_x0(p, sol, e_i, dsol);
        check(ss == Status::kSolved, pfx + ": sens_x0 kSolved (got " +
                                         std::to_string(static_cast<int>(ss)) +
                                         ")");
        if (ss != Status::kSolved)
        {
            continue;
        }

        // pinned initial state moves exactly by dx0
        Eigen::VectorXd dx0_an = dsol.ux_first.segment(QpDim<DiSens>::nu,
                                                       QpDim<DiSens>::nx);
        check((dx0_an - e_i).cwiseAbs().maxCoeff() < 1e-12,
              pfx + ": dx_0 == e_i (pinned initial state)");

        // cross-check: dense sens vs staged sens on the same base solution
        QpSol<DiSens, Eigen::Dynamic> dsol_st(N);
        check(solver_st.sens_x0(p, sol_st, e_i, dsol_st) == Status::kSolved,
              pfx + ": staged sens_x0 kSolved");
        double xdiff = 0.0;
        for (int k = 0; k <= N; ++k)
        {
            Eigen::VectorXd du_d, dx_d, du_s, dx_s;
            ux_block(dsol, N, k, du_d, dx_d);
            ux_block(dsol_st, N, k, du_s, dx_s);
            xdiff = std::max(xdiff, (du_d - du_s).cwiseAbs().maxCoeff());
            xdiff = std::max(xdiff, (dx_d - dx_s).cwiseAbs().maxCoeff());
        }
        check(xdiff < 1e-6, pfx + ": dense vs staged sens agree < 1e-6 (got "
                                  + std::to_string(xdiff) + ")");

        // finite differences: x0 +/- h*e_i, re-solved (dense mode)
        DiSens pp = p;
        pp.x0_(i) += h;
        DiSens pm = p;
        pm.x0_(i) -= h;
        Solution<DiSens> sol_p = make_start(pp, N);
        Solution<DiSens> sol_m = make_start(pm, N);
        AcadosSolver<DiSens, Eigen::Dynamic> sp(opts);
        AcadosSolver<DiSens, Eigen::Dynamic> sm(opts);
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
            Eigen::VectorXd dx_fd =
                (sol_p.x[k].cast<double>() - sol_m.x[k].cast<double>()) /
                (2.0 * h);
            if (k < N)
            {
                const Eigen::VectorXd du_fd =
                    (sol_p.u[k].cast<double>() -
                     sol_m.u[k].cast<double>()) / (2.0 * h);
                max_du = std::max(max_du,
                                  (du_an - du_fd).cwiseAbs().maxCoeff());
            }
            max_dx = std::max(max_dx,
                              (dx_an - dx_fd).cwiseAbs().maxCoeff());
        }
        check(max_du < tol, pfx + ": du/dx0 matches FD < " +
                                std::to_string(tol) + " (got " +
                                std::to_string(max_du) + ")");
        check(max_dx < tol, pfx + ": dx/dx0 matches FD < " +
                                std::to_string(tol) + " (got " +
                                std::to_string(max_dx) + ")");
        std::printf("  %s: max |du diff| = %.3e, max |dx diff| = %.3e\n",
                    pfx.c_str(), max_du, max_dx);
    }
}

// 5h: the same FD check with qp_mode = kPartialCondensed (block QP,
// NB = 2), plus a cross-check of the block sens_x0 against the staged
// sens_x0 on the same base solution.
void test_sens_x0_fd_block()
{
    using BlockSolv = AcadosSolver<DiSens, Eigen::Dynamic,
                                   HpipmQpSolver<DiSens, Eigen::Dynamic>,
                                   GlmRegularizer<DiSens, Eigen::Dynamic>,
                                   MeritBacktracking<DiSens, Eigen::Dynamic>, 2>;
    const std::string name = "sens_x0 block DI N=10 ";
    const int N = 10;
    const double tol = 1e-6;
    const double h = 1e-6;

    DiSens p;
    Solution<DiSens> sol = make_start(p, N);
    AcadosOptions opts;
    opts.qp_mode = QpMode::kPartialCondensed;
    opts.qp_cond_N = N / 2;  // 5 blocks of NB = 2
    BlockSolv solver(opts);
    const Status st = solver.solve(p, sol);
    check(st == Status::kSolved, name + " base solve kSolved (got " +
                                     std::to_string(static_cast<int>(st)) +
                                     ")");
    if (st != Status::kSolved)
    {
        return;
    }

    // staged reference (same base iterate)
    AcadosOptions opts_st;
    AcadosSolver<DiSens, Eigen::Dynamic> solver_st(opts_st);
    Solution<DiSens> sol_st = make_start(p, N);
    check(solver_st.solve(p, sol_st) == Status::kSolved,
          name + " staged base solve kSolved");

    for (int i = 0; i < QpDim<DiSens>::nx; ++i)
    {
        const std::string pfx = name + " e_" + std::to_string(i);

        DiSens::state_t e_i;
        e_i.setZero();
        e_i(i) = 1.0;
        QpSol<DiSens, Eigen::Dynamic> dsol(N);
        const Status ss = solver.sens_x0(p, sol, e_i, dsol);
        check(ss == Status::kSolved, pfx + ": sens_x0 kSolved (got " +
                                         std::to_string(static_cast<int>(ss)) +
                                         ")");
        if (ss != Status::kSolved)
        {
            continue;
        }

        // pinned initial state moves exactly by dx0
        Eigen::VectorXd dx0_an = dsol.ux_first.segment(QpDim<DiSens>::nu,
                                                       QpDim<DiSens>::nx);
        check((dx0_an - e_i).cwiseAbs().maxCoeff() < 1e-12,
              pfx + ": dx_0 == e_i (pinned initial state)");

        // cross-check: block sens vs staged sens on the same base solution
        QpSol<DiSens, Eigen::Dynamic> dsol_st(N);
        check(solver_st.sens_x0(p, sol_st, e_i, dsol_st) == Status::kSolved,
              pfx + ": staged sens_x0 kSolved");
        double xdiff = 0.0;
        for (int k = 0; k <= N; ++k)
        {
            Eigen::VectorXd du_b, dx_b, du_s, dx_s;
            ux_block(dsol, N, k, du_b, dx_b);
            ux_block(dsol_st, N, k, du_s, dx_s);
            xdiff = std::max(xdiff, (du_b - du_s).cwiseAbs().maxCoeff());
            xdiff = std::max(xdiff, (dx_b - dx_s).cwiseAbs().maxCoeff());
        }
        check(xdiff < 1e-6, pfx + ": block vs staged sens agree < 1e-6 (got "
                                  + std::to_string(xdiff) + ")");

        // finite differences: x0 +/- h*e_i, re-solved (block mode)
        DiSens pp = p;
        pp.x0_(i) += h;
        DiSens pm = p;
        pm.x0_(i) -= h;
        Solution<DiSens> sol_p = make_start(pp, N);
        Solution<DiSens> sol_m = make_start(pm, N);
        BlockSolv sp(opts);
        BlockSolv sm(opts);
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
            Eigen::VectorXd dx_fd =
                (sol_p.x[k].cast<double>() - sol_m.x[k].cast<double>()) /
                (2.0 * h);
            if (k < N)
            {
                const Eigen::VectorXd du_fd =
                    (sol_p.u[k].cast<double>() -
                     sol_m.u[k].cast<double>()) / (2.0 * h);
                max_du = std::max(max_du,
                                  (du_an - du_fd).cwiseAbs().maxCoeff());
            }
            max_dx = std::max(max_dx, (dx_an - dx_fd).cwiseAbs().maxCoeff());
        }
        check(max_du < tol, pfx + ": du/dx0 matches FD < " +
                                std::to_string(tol) + " (got " +
                                std::to_string(max_du) + ")");
        check(max_dx < tol, pfx + ": dx/dx0 matches FD < " +
                                std::to_string(tol) + " (got " +
                                std::to_string(max_dx) + ")");
        std::printf("  %s: max |du diff| = %.3e, max |dx diff| = %.3e\n",
                    pfx.c_str(), max_du, max_dx);
    }
}

}  // namespace

int run_sens_x0_5f_tests()
{
    test_sens_x0_fd();
    test_sens_x0_fd_dense();
    test_sens_x0_fd_block();
    if (failures == 0)
    {
        std::printf("All sens_x0 (5f) checks passed.\n");
    }
    return failures;
}

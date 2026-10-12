// Phase 4 block 4d test: QP sensitivities (HPIPM seed module).
//
// HpipmQpSolver::sens_forward / sens_adj are cross-checked against
// central finite differences of perturbed re-solves (tightened solver
// tolerances, small h), plus:
//   - hessian "seed" via the equivalent grad seed -E z* (open item 4)
//   - adjoint == forward on the ux part (m = 0 seeds)
//   - the K^{-1} symmetry duality identity on two seeds
//   - state hygiene (solve -> sens -> solve on one instance)
//   - the no-prior-solve guard (kQpFailure)
//   - QpSeed sizing / setZero smoke

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>

#include "ocp/solvers/hpipm/hpipm.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"
#include "../../examples/mass_spring/mass_spring.hpp"

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

struct Rng
{
    std::mt19937_64 gen{90210};
    double uniform()
    {
        return std::uniform_real_distribution<double>(-1.0, 1.0)(gen);
    }
};

// Strongly-convex feasibility-friendly synthetic QP (same construction as
// solve_1h.cpp): I + R'R on (u;x), unit on slacks; wide interior two-sided
// bounds (lo = -3, hi = 3, i.e. d = -3 on both sides at v = 0); equality
// rows at v = 0; ineq hi = 1; m = 0; all sides active; contractive
// dynamics. The origin is strictly feasible so init_point stays interior.
template <class P, int NH>
void fill_qp(Qp<P, NH>& qp, int N)
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
            // g_eq rows are filled as inactive two-sided boxes (+-3, d = -3
            // on both sides gives d_lo <= v <= -d_hi) rather than exact
            // equalities (d = 0 both sides): an exact equality pins the
            // solution onto the boundary (t -> 0, Gamma = lambda/t ~ 1e20),
            // which puts the barrier KKT at the numerical floor and makes
            // the sensitivity ill-conditioned.  The sensitivity machinery
            // is what 4d validates, not the equality regime.
            if (sh >= 0)
            {
                st.d(sh) = (g == detail::g_ineq) ? -1.0 : -3.0;
            }
            if (sl >= 0)
            {
                st.d(sl) = -3.0;
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

// Tight-tolerance solver for the FD re-solves (a small h amplifies the
// absolute solver error by 1/h).
template <class P, int NH>
HpipmQpSolver<P, NH> fd_solver()
{
    HpipmOptions o;
    o.res_g_max = 3e-7;
    o.res_b_max = 1e-11;
    o.res_d_max = 1e-10;
    o.res_m_max = 1e-10;
    o.alpha_min = 1e-13;
    o.iter_max = 200;
    return HpipmQpSolver<P, NH>(o);
}

// Relative error of (a - b) scaled by max(1, ||b||_inf).
template <class A, class B>
bool rel_close(const A& a, const B& b, double tol, const std::string& msg)
{
    const double num = (a - b).cwiseAbs().maxCoeff();
    const double den = std::max(1.0, (double)b.cwiseAbs().maxCoeff());
    if (num / den > tol)
    {
        std::fprintf(stderr, "FAIL: %s (err = %.3e)\n", msg.c_str(), num / den);
        return false;
    }
    return true;
}

// Compare every part of a QpSol pair with a relative tolerance.
template <class P, int NH>
void check_sol_close(const QpSol<P, NH>& fd, const QpSol<P, NH>& sens,
                     double tol, const char* name)
{
    const std::string p = name;
    check(rel_close(fd.ux_first, sens.ux_first, tol, p + " ux_first"), "");
    for (int k = 1; k < fd.N; ++k)
    {
        check(rel_close(fd.ux_path[k - 1], sens.ux_path[k - 1], tol,
                        p + " ux_path"), "");
    }
    check(rel_close(fd.ux_term, sens.ux_term, tol, p + " ux_term"), "");
    for (int k = 0; k < fd.N; ++k)
    {
        check(rel_close(fd.pi[k], sens.pi[k], tol, p + " pi"), "");
    }
    check(rel_close(fd.lam_first, sens.lam_first, tol, p + " lam_first"), "");
    for (int k = 1; k < fd.N; ++k)
    {
        check(rel_close(fd.lam_path[k - 1], sens.lam_path[k - 1], tol,
                        p + " lam_path"), "");
    }
    check(rel_close(fd.lam_term, sens.lam_term, tol, p + " lam_term"), "");
    check(rel_close(fd.t_first, sens.t_first, tol, p + " t_first"), "");
    for (int k = 1; k < fd.N; ++k)
    {
        check(rel_close(fd.t_path[k - 1], sens.t_path[k - 1], tol,
                        p + " t_path"), "");
    }
    check(rel_close(fd.t_term, sens.t_term, tol, p + " t_term"), "");
}

// Central difference of a re-solved QP solution: perturb `qpp` / `qpm`,
// solve both with tight tolerances, return (sp - sm) / 2h.
template <class P, int NH, class F1, class F2>
QpSol<P, NH> fd_step(const Qp<P, NH>& qp, double h, const F1& plus,
                     const F2& minus)
{
    Qp<P, NH> qpp(qp.N), qpm(qp.N);
    qpp = qp;
    qpm = qp;
    plus(qpp);
    minus(qpm);
    QpSol<P, NH> sp(qp.N), sm(qp.N);
    HpipmQpSolver<P, NH> svp = fd_solver<P, NH>();
    HpipmQpSolver<P, NH> svm = fd_solver<P, NH>();
    check(svp.solve(qpp, sp) == Status::kSolved, "FD plus solve");
    check(svm.solve(qpm, sm) == Status::kSolved, "FD minus solve");
    QpSol<P, NH> fd(qp.N);
    const double inv = 1.0 / (2.0 * h);
    fd.ux_first = (sp.ux_first - sm.ux_first) * inv;
    for (int k = 1; k < qp.N; ++k)
    {
        fd.ux_path[k - 1] = (sp.ux_path[k - 1] - sm.ux_path[k - 1]) * inv;
    }
    fd.ux_term = (sp.ux_term - sm.ux_term) * inv;
    for (int k = 0; k < qp.N; ++k)
    {
        fd.pi[k] = (sp.pi[k] - sm.pi[k]) * inv;
    }
    fd.lam_first = (sp.lam_first - sm.lam_first) * inv;
    for (int k = 1; k < qp.N; ++k)
    {
        fd.lam_path[k - 1] = (sp.lam_path[k - 1] - sm.lam_path[k - 1]) * inv;
    }
    fd.lam_term = (sp.lam_term - sm.lam_term) * inv;
    fd.t_first = (sp.t_first - sm.t_first) * inv;
    for (int k = 1; k < qp.N; ++k)
    {
        fd.t_path[k - 1] = (sp.t_path[k - 1] - sm.t_path[k - 1]) * inv;
    }
    fd.t_term = (sp.t_term - sm.t_term) * inv;
    return fd;
}

template <class P, int NH>
void check_seed_smoke(const char* name, int N)
{
    const std::string p = name;

    // sizing / setZero
    QpSeed<P, NH> seed(N);
    check(seed.N == N, p + " QpSeed::N");
    seed.grad_first(0) = 1.0;
    seed.b[1](0) = 1.0;
    seed.d_first(0) = 1.0;
    seed.m_path[N - 2](0) = 1.0;
    seed.setZero();
    check(seed.grad_first.isZero() && seed.b[1].isZero(),
          p + " QpSeed::setZero (first)");
    check(seed.d_first.isZero() && seed.m_path[N - 2].isZero(),
          p + " QpSeed::setZero (d/m)");

    // guard: no prior solve -> kQpFailure
    Qp<P, NH> qp(N);
    fill_qp(qp, N);
    HpipmQpSolver<P, NH> fresh;
    QpSol<P, NH> sol(N);
    QpSol<P, NH> sens(N);
    check(fresh.sens_forward(qp, sol, seed, sens) == Status::kQpFailure,
          p + " sens_forward guard (fresh solver)");
    check(fresh.sens_adj(qp, sol, seed, sens) == Status::kQpFailure,
          p + " sens_adj guard (fresh solver)");
}

template <class P, int NH>
void check_seed_fd(const char* name, int N)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    const std::string p = name;
    const double h = 1e-4;
    const double tol = 1e-5;

    Qp<P, NH> qp(N);
    fill_qp(qp, N);
    HpipmQpSolver<P, NH> solver = fd_solver<P, NH>();
    QpSol<P, NH> sol(N);
    check(solver.solve(qp, sol) == Status::kSolved, p + " base solve");

    // --- grad seed (first stage, variable 0) -----------------------
    {
        QpSeed<P, NH> seed(N);
        seed.grad_first(0) = 1.0;
        QpSol<P, NH> sens(N);
        check(solver.sens_forward(qp, sol, seed, sens) == Status::kSolved,
              p + " sens_forward (grad)");
        QpSol<P, NH> fd = fd_step(
            qp, h,
            [&](Qp<P, NH>& q) { q.first.grad(0) += h; },
            [&](Qp<P, NH>& q) { q.first.grad(0) -= h; });
        check_sol_close(fd, sens, tol, (p + " grad seed").c_str());
    }

    // --- b seed (dynamics step k = 2, state component 0) ------------
    {
        QpSeed<P, NH> seed(N);
        seed.b[2](0) = 1.0;
        QpSol<P, NH> sens(N);
        check(solver.sens_forward(qp, sol, seed, sens) == Status::kSolved,
              p + " sens_forward (b)");
        QpSol<P, NH> fd = fd_step(
            qp, h,
            [&](Qp<P, NH>& q) { q.path[1].b(0) += h; },
            [&](Qp<P, NH>& q) { q.path[1].b(0) -= h; });
        check_sol_close(fd, sens, tol, (p + " b seed").c_str());
    }

    // --- hessian "seed": FD of H +- h E vs -sens_forward(E z*) ------
    {
        const int nux = D::nvar_path - D::nslack_path;
        (void)nux;
        // explicit type: this Eigen's Zero() returns a const expression,
        // so `auto` would deduce a non-assignable constant object
        Eigen::Matrix<S, D::nvar_path, D::nvar_path> E =
            Eigen::Matrix<S, D::nvar_path, D::nvar_path>::Zero();
        E(0, 0) = 1.0;
        E(1, 1) = 1.0;
        E(0, 1) = 0.5;
        E(1, 0) = 0.5;

        QpSol<P, NH> fd = fd_step(
            qp, h,
            [&](Qp<P, NH>& q) { q.path[0].hess += h * E; },
            [&](Qp<P, NH>& q) { q.path[0].hess -= h * E; });

        QpSeed<P, NH> seed(N);
        seed.grad_path[0] = E * sol.ux_path[0];
        QpSol<P, NH> sens(N);
        check(solver.sens_forward(qp, sol, seed, sens) == Status::kSolved,
              p + " sens_forward (hessian-via-grad)");

        // FD of H +- h E: the KKT RHS shifts by +h E z* (stationarity) and
        // the KKT Hessian by h E; to first order in h both give
        // dz/dh = -K^-1 (E z*).  The seed machinery applies the same
        // negative sign to the grad seed as for an explicit grad
        // perturbation, so fd and sens have the same sign (no flip).
        check_sol_close(fd, sens, tol, (p + " hessian seed").c_str());
    }

    // --- adjoint == forward on the ux part (m = 0 seeds) ------------
    {
        QpSeed<P, NH> seed(N);
        seed.grad_first(0) = 1.0;
        seed.b[0](0) = 1.0;
        QpSol<P, NH> sfwd(N), sadj(N);
        check(solver.sens_forward(qp, sol, seed, sfwd) == Status::kSolved,
              p + " sens_forward (adjoint pair)");
        check(solver.sens_adj(qp, sol, seed, sadj) == Status::kSolved,
              p + " sens_adj (adjoint pair)");
        check(rel_close(sadj.ux_first, sfwd.ux_first, 1e-12,
                        p + " adjoint ux_first"), "");
        for (int k = 1; k < N; ++k)
        {
            check(rel_close(sadj.ux_path[k - 1], sfwd.ux_path[k - 1], 1e-12,
                            p + " adjoint ux_path"), "");
        }
        check(rel_close(sadj.ux_term, sfwd.ux_term, 1e-12,
                        p + " adjoint ux_term"), "");
    }

    // --- duality: s1' K^{-1} s2 == s2' K^{-1} s1 --------------------
    {
        QpSeed<P, NH> s1(N), s2(N);
        s1.grad_first(1) = 1.0;
        s1.b[0](0) = 1.0;
        s2.grad_path[2](0) = 1.0;
        s2.b[3](1) = 1.0;
        QpSol<P, NH> v1(N), v2(N);
        check(solver.sens_forward(qp, sol, s1, v1) == Status::kSolved,
              p + " sens_forward (duality s1)");
        check(solver.sens_forward(qp, sol, s2, v2) == Status::kSolved,
              p + " sens_forward (duality s2)");
        auto ip = [&](const QpSeed<P, NH>& s, const QpSol<P, NH>& v)
        {
            double acc = s.grad_first.dot(v.ux_first);
            for (int k = 1; k < N; ++k)
            {
                acc += s.grad_path[k - 1].dot(v.ux_path[k - 1]);
            }
            acc += s.grad_term.dot(v.ux_term);
            for (int k = 0; k < N; ++k)
            {
                acc += s.b[k].dot(v.pi[k]);
            }
            return acc;
        };
        const double d = std::fabs(ip(s1, v2) - ip(s2, v1));
        check(d < 1e-6, p + " duality identity (diff = " +
                            std::to_string(d) + ")");
    }

    // --- state hygiene: solve -> sens -> solve on the same instance -
    {
        QpSol<P, NH> sol2(N);
        check(solver.solve(qp, sol2) == Status::kSolved,
              p + " re-solve after sens");
        check_sol_close(sol2, sol, 1e-9, (p + " hygiene").c_str());
    }
}

// d seed on an ACTIVE bound: push u_0 onto its upper bound (hi = 3 in
// fill_qp) with a strong negative gradient, then check the sensitivity of
// the hi-side offset (seed = -e_j on the hi side, d_hi = v_cur - hi).
template <class P, int NH>
void check_seed_fd_d(const char* name, int N)
{
    using D = QpDim<P>;
    static_assert(D::nbu > 0, "check_seed_fd_d needs control box rows");
    const std::string p = name;
    const double h = 1e-4;
    const double tol = 1e-5;

    Qp<P, NH> qp(N);
    fill_qp(qp, N);
    qp.first.grad(0) = -50.0;  // u_0 -> hi bound
    HpipmQpSolver<P, NH> solver = fd_solver<P, NH>();
    QpSol<P, NH> sol(N);
    check(solver.solve(qp, sol) == Status::kSolved, p + " base solve (active)");
    check(sol.ux_first(0) > 2.9, p + " u_0 at upper bound (active)");
    if (sol.ux_first(0) <= 2.9)
    {
        return;
    }

    const int r = D::lay_first.row_off(detail::g_bu);  // first bu row = u_0
    const int side = D::lay_first.side_hi(r);
    QpSeed<P, NH> seed(N);
    seed.d_first(side) = -1.0;  // hi += h  <=>  d_hi -= h
    QpSol<P, NH> sens(N);
    check(solver.sens_forward(qp, sol, seed, sens) == Status::kSolved,
          p + " sens_forward (d, active hi)");
    QpSol<P, NH> fd = fd_step(
        qp, h,
        [&](Qp<P, NH>& q) { q.first.d(side) -= h; },
        [&](Qp<P, NH>& q) { q.first.d(side) += h; });
    check_sol_close(fd, sens, tol, (p + " d seed (active hi)").c_str());
}

}  // namespace

int run_seed_4d_tests()
{
    failures = 0;
    const int N = 5;
    check_seed_smoke<DoubleIntegrator, Eigen::Dynamic>("DI dyn", N);
    check_seed_smoke<MassSpring, 5>("MS NH=5", N);
    check_seed_fd<DoubleIntegrator, Eigen::Dynamic>("DI dyn N=5", N);
    check_seed_fd_d<DoubleIntegrator, Eigen::Dynamic>("DI dyn N=5", N);
    check_seed_fd<MassSpring, 5>("MS NH=5 N=5", N);
    check_seed_fd_d<MassSpring, 5>("MS NH=5 N=5", N);
    if (failures == 0)
    {
        std::printf("All seed (4d) checks passed.\n");
    }
    return failures;
}

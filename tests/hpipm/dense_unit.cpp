// Phase 1 (parity) block 1c test: HpipmDenseQpSolver.
//
//  (1) compute_residuals vs an independent Eigen evaluation
//  (2) kkt_step Newton step vs the full block KKT system (Eigen LU)
//  (3) solve() on a random dense QP (m = 0, as in production use):
//      status kSolved, KKT residuals within the option tolerances;
//      hot warm-start converges immediately
//  (4) unconstrained fast path: v = (H + reg I_(x0;u))^{-1} (-g), iter = 0
//  (5) fixed-horizon (NH) compile-time path

#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "ocp/solvers/hpipm/dense_qp.hpp"
#include "ocp/solvers/hpipm/hpipm.hpp"

#include <Eigen/Dense>

using namespace ocp;
using S = double;

namespace
{

int failures = 0;

void check(bool cond, const char* msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

template <class A, class B>
void check_close(const A& a, const B& b, double tol, const char* msg)
{
    const double err = (a - b).cwiseAbs().maxCoeff();
    if (err > tol)
    {
        std::fprintf(stderr, "FAIL: %s (err = %.3e)\n", msg, err);
        ++failures;
    }
}

void check_close_scalar(double a, double b, double tol, const char* msg)
{
    if (std::fabs(a - b) > tol)
    {
        std::fprintf(stderr, "FAIL: %s (%.3e vs %.3e)\n", msg, a, b);
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
    double pos()
    {
        return std::uniform_real_distribution<double>(0.0, 1.0)(gen);
    }
};

// Synthetic problem (copied from condense_full.cpp / dense_qp_dim.cpp)
struct TinyFree
{
    using scalar_t = double;
    static constexpr int nx = 2, nu = 2;
    static constexpr bool fixed_initial_state = false;
    static constexpr int nbx = 2, nbu = 1, ng = 1, ne = 0, nl = 1;
    static constexpr int nbx_t = 1, ng_t = 1, ne_t = 1, nl_t = 0;
    static constexpr int nbx_soft = 1, nbu_soft = 0, ng_soft = 1, ne_soft = 0,
        nl_soft = 0;
    static constexpr int nbx_t_soft = 1, ng_t_soft = 0, ne_t_soft = 0,
        nl_t_soft = 0;
    static constexpr std::array<int, 2> state_box_idx = {0, 1};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 1> terminal_state_box_idx = {0};
    static constexpr std::array<int, 1> state_box_soft_idx = {0};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 1> ineq_soft_idx = {0};
    static constexpr std::array<int, 0> eq_soft_idx = {};
    static constexpr std::array<int, 0> lin_soft_idx = {};
    static constexpr std::array<int, 1> terminal_state_box_soft_idx = {0};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {};
};

// ---------------------------------------------------------------------
// Random dense QP with a strictly interior v = 0 and distinct box vars
// ---------------------------------------------------------------------

template <class P, int NH>
DenseQp<P, NH> make_qp(int N, Rng& rng)
{
    using D = DenseQpDim<P>;
    DenseQp<P, NH> qp;
    qp.resize(N);

    const int nv = D::nv_total(N);
    Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic> A =
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>::Random(nv, nv);
    qp.H = A.transpose() * A
           + S(0.1) * Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>::Identity
                 (nv, nv);
    for (int j = 0; j < nv; ++j)
    {
        qp.g(j) = rng.uniform();
    }
    for (int r = 0; r < qp.ng; ++r)
    {
        for (int j = 0; j < nv; ++j)
        {
            qp.C_lo(r, j) = rng.uniform();
            qp.C_hi(r, j) = rng.uniform();
        }
    }
    for (int r = 0; r < qp.nb; ++r)
    {
        check(r < nv, "box row index in range");
        qp.idxb[r] = r;
    }
    // lo bounds negative, hi bounds stored as -hi (also negative) so that
    // v = 0 is strictly interior on every side
    for (int i = 0; i < qp.nside; ++i)
    {
        qp.d(i) = -(S(1.0) + rng.pos());
        qp.d_mask(i) = 1.0;
        qp.m(i) = rng.uniform();
    }
    return qp;
}

// Independent copy of the signed side matrix (test-side oracle)
template <class P, int NH>
Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>
side_matrix(const DenseQp<P, NH>& qp)
{
    const int nv = qp.nv, nb = qp.nb, ng = qp.ng;
    Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic> A(qp.nside, nv);
    A.setZero();
    for (int r = 0; r < nb; ++r)
    {
        A(r, qp.idxb[r]) = 1.0;
        A(nb + ng + r, qp.idxb[r]) = -1.0;
    }
    for (int r = 0; r < ng; ++r)
    {
        A.row(nb + r) = qp.C_lo.row(r);
        A.row(nb + ng + nb + r) = -qp.C_hi.row(r);
    }
    return A;
}

// ---------------------------------------------------------------------
// (1) residuals
// ---------------------------------------------------------------------

template <class P, int NH>
void test_residuals(Rng& rng)
{
    const std::string tag = "residuals ";
    DenseQp<P, NH> qp = make_qp<P, NH>(3, rng);
    DenseQpSol<P, NH> sol;
    sol.resize(3);
    for (int j = 0; j < qp.nv; ++j)
    {
        sol.v(j) = rng.uniform();
    }
    for (int i = 0; i < qp.nside; ++i)
    {
        sol.lam(i) = S(0.05) + rng.pos();
        sol.t(i) = S(0.05) + rng.pos();
    }

    HpipmDenseQpSolver<P, NH> solver;
    DenseQpRes<P, NH> res;
    res.resize(3);
    solver.compute_residuals(qp, sol, res);

    const auto A = side_matrix(qp);
    const auto lam_m = qp.d_mask.cwiseProduct(sol.lam);
    const auto Hz = qp.H * sol.v;
    const auto res_g_ref = Hz + qp.g - A.transpose() * lam_m;
    const auto res_d_ref = (qp.d + sol.t - A * sol.v)
                           .cwiseProduct(qp.d_mask);
    const auto res_m_ref = qp.d_mask.cwiseProduct(
        sol.lam.cwiseProduct(sol.t) - qp.m);
    const double obj_ref = S(0.5) * sol.v.dot(Hz) + sol.v.dot(qp.g);
    const double gap_ref = sol.v.dot(Hz) + sol.v.dot(qp.g)
                           - qp.d.dot(lam_m);
    int nc = 0;
    for (int i = 0; i < qp.nside; ++i)
    {
        nc += (qp.d_mask(i) > 0.5) ? 1 : 0;
    }
    const double mu_ref = res_m_ref.cwiseAbs().sum() / nc;

    check_close(res.res_g, res_g_ref, 1e-11, (tag + " res_g").c_str());
    check_close(res.res_d, res_d_ref, 1e-11, (tag + " res_d").c_str());
    check_close(res.res_m, res_m_ref, 1e-11, (tag + " res_m").c_str());
    check_close_scalar(res.obj, obj_ref, 1e-10, (tag + " obj").c_str());
    check_close_scalar(res.dual_gap, gap_ref, 1e-10,
                       (tag + " dual_gap").c_str());
    check_close_scalar(res.res_mu, mu_ref, 1e-12, (tag + " res_mu").c_str());
    check_close_scalar(res.res_g_max, res_g_ref.cwiseAbs().maxCoeff(), 1e-12,
                       (tag + " res_g_max").c_str());
}

// ---------------------------------------------------------------------
// (2) kkt_step vs full block KKT solve
// ---------------------------------------------------------------------

template <class P, int NH>
void test_kkt_step(Rng& rng)
{
    const std::string tag = "kkt_step  ";
    DenseQp<P, NH> qp = make_qp<P, NH>(3, rng);
    DenseQpSol<P, NH> sol;
    sol.resize(3);
    for (int j = 0; j < qp.nv; ++j)
    {
        sol.v(j) = rng.uniform();
    }
    for (int i = 0; i < qp.nside; ++i)
    {
        sol.lam(i) = S(0.05) + rng.pos();
        sol.t(i) = S(0.05) + rng.pos();
    }

    HpipmDenseQpSolver<P, NH> solver;
    DenseQpRes<P, NH> res;
    res.resize(3);
    solver.compute_residuals(qp, sol, res);

    const S reg = S(1e-6);
    DenseQpSol<P, NH> step;
    step.resize(3);
    const Status st = solver.kkt_step(qp, sol, res, step, reg);
    check(st == Status::kSolved, (tag + " status").c_str());

    // reference: full KKT block [ W -A'; A 0 ] [dv; dt] = [-res_g - A'g; -res_d]
    const auto A = side_matrix(qp);
    Eigen::VectorXd gamma(qp.nside), Gamma(qp.nside);
    gamma.setZero();
    Gamma.setZero();
    for (int i = 0; i < qp.nside; ++i)
    {
        if (qp.d_mask(i) > 0.5)
        {
            Gamma(i) = sol.lam(i) / sol.t(i);
            gamma(i) = (res.res_m(i) - sol.lam(i) * res.res_d(i)) / sol.t(i);
        }
    }
    const int nvh = DenseQpDim<P>::nx + 3 * DenseQpDim<P>::nu;
    Eigen::MatrixXd W = qp.H;
    for (int j = 0; j < nvh; ++j)
    {
        W(j, j) += reg;
    }
    W += A.transpose() * (Gamma.asDiagonal() * A);

    // The full KKT block [ W -A'; A 0 ] is singular whenever nside > nv
    // (A is rank-deficient), so it cannot be used as a reference. Instead
    // verify the closed-form Newton relations that kkt_step implements:
    //   (1) W * dv  = -res_g - A' gamma
    //   (2) dt      = A dv - res_d
    //   (3) dlam    = d_mask . ( -res_m - (d_mask . lam) . dt ) ./ t
    const Eigen::VectorXd rhs = -res.res_g - A.transpose() * gamma;
    const auto dv_res = W * step.v - rhs;
    check(dv_res.cwiseAbs().maxCoeff() < S(1e-7), (tag + " W dv = rhs").c_str());

    const auto dt_ref = A * step.v - res.res_d;
    check_close(step.t, dt_ref, 1e-7, (tag + " dt").c_str());

    const auto dlam_ref = qp.d_mask.cwiseProduct(
        (-res.res_m - qp.d_mask.cwiseProduct(sol.lam).cwiseProduct(step.t))
        .cwiseQuotient(sol.t));
    check_close(step.lam, dlam_ref, 1e-7, (tag + " dlam").c_str());
}

// ---------------------------------------------------------------------
// (3) full IPM solve
// ---------------------------------------------------------------------

template <class P, int NH>
void test_solve(Rng& rng)
{
    const std::string tag = "solve     ";
    DenseQp<P, NH> qp = make_qp<P, NH>(3, rng);
    qp.m.setZero();  // production QPs carry m = 0

    HpipmDenseQpSolver<P, NH> solver;
    DenseQpSol<P, NH> sol;
    sol.resize(3);
    const Status st = solver.solve(qp, sol);
    check(st == Status::kSolved, (tag + " status").c_str());
    check(solver.statistics().iter > 0, (tag + " did iterate").c_str());
    check(solver.statistics().iter < solver.options().iter_max,
          (tag + " within iter_max").c_str());

    // interior solution
    check(sol.lam.minCoeff() >= 0.0, (tag + " lam >= 0").c_str());
    check(sol.t.minCoeff() > 0.0, (tag + " t > 0").c_str());

    // KKT residuals at the returned iterate
    DenseQpRes<P, NH> res;
    res.resize(3);
    solver.compute_residuals(qp, sol, res);
    check(res.res_g_max <= S(1e-6), (tag + " res_g").c_str());
    check(res.res_d_max <= S(1e-7), (tag + " res_d").c_str());
    check(res.res_m.cwiseAbs().maxCoeff() <= S(1e-6),
          (tag + " complementarity").c_str());

    // hot warm-start: already solved -> converges without iterating
    solver.set_warm_start(3);
    const Eigen::VectorXd v_bkp = sol.v;
    const Status st2 = solver.solve(qp, sol);
    check(st2 == Status::kSolved, (tag + " hot status").c_str());
    check(solver.statistics().iter <= 1, (tag + " hot iterations").c_str());
    check_close(sol.v, v_bkp, 1e-9, (tag + " hot keeps v").c_str());
    solver.set_warm_start(0);
}

// ---------------------------------------------------------------------
// (4) unconstrained fast path
// ---------------------------------------------------------------------

template <class P, int NH>
void test_unconstrained(Rng& rng)
{
    const std::string tag = "unconstr  ";
    DenseQp<P, NH> qp = make_qp<P, NH>(3, rng);
    qp.m.setZero();
    qp.d_mask.setZero();

    HpipmDenseQpSolver<P, NH> solver;
    DenseQpSol<P, NH> sol;
    sol.resize(3);
    const Status st = solver.solve(qp, sol);
    check(st == Status::kSolved, (tag + " status").c_str());
    check(solver.statistics().iter == 0, (tag + " fast path iter = 0").c_str());

    const S reg = S(solver.options().reg_prim);
    const int nvh = DenseQpDim<P>::nx + 3 * DenseQpDim<P>::nu;
    Eigen::MatrixXd W = qp.H;
    for (int j = 0; j < nvh; ++j)
    {
        W(j, j) += reg;
    }
    const Eigen::LLT<Eigen::MatrixXd> llt(W);
    const Eigen::VectorXd v_ref = llt.solve(-qp.g);
    check_close(sol.v, v_ref, 1e-9, (tag + " v").c_str());
    check_close_scalar(sol.lam.maxCoeff(), 0.0, 0.0, (tag + " lam = 0").c_str());
    check_close_scalar(sol.t.maxCoeff(), 0.0, 0.0, (tag + " t = 0").c_str());
}

}  // namespace

int main()
{
    Rng rng;

    test_residuals<TinyFree, Eigen::Dynamic>(rng);
    test_kkt_step<TinyFree, Eigen::Dynamic>(rng);
    test_solve<TinyFree, Eigen::Dynamic>(rng);
    test_unconstrained<TinyFree, Eigen::Dynamic>(rng);

    // (5) fixed-horizon path (compile-time extents)
    test_solve<TinyFree, 3>(rng);
    test_kkt_step<TinyFree, 3>(rng);

    if (failures == 0)
    {
        std::printf("All dense_unit checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}

// Phase 5 test 5g: HpipmDenseQpSolver QP sensitivities (forward / adjoint).
//
// Verifies HpipmDenseQpSolver::sens_forward / sens_adj (HPIPM
// DENSE_QP_IPM_SENS_FRW / _ADJ) against central finite differences of the
// dense IPM re-solve:
//   K = d(v*, lam*, t*)/d(rhs)  from one sens call,
//   FD  = (sol(rhs + h s) - sol(rhs - h s)) / (2 h).
// Two regimes are exercised:
//   (a) a dense QP with a hard-equality row (lo == hi) -> the sens path takes
//       the hard-equality block KKT (Eigen LDLT), since the barrier KKT is
//       singular at the converged iterate on that row;
//   (b) a dense QP with only two-sided inequality rows -> the sens path reuses
//       the barrier factor (W = H + reg I + A' diag(Gamma) A).
// The adjoint is checked on its defining property for a pure cost seed
// (seed_d = seed_m = 0): the primal step must equal the forward primal step.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "ocp/solvers/hpipm/dense_qp.hpp"
#include "ocp/solvers/hpipm/seed.hpp"
#include "ocp/solvers/hpipm/hpipm.hpp"

#include <Eigen/Dense>

using namespace ocp;
using S = double;

namespace
{

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
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
        return std::uniform_real_distribution<double>(0.0, 1.0)(gen);
    }
};

// Synthetic problem (copied from dense_unit.cpp / condense_full.cpp).
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

// Random dense QP with a strictly interior v = 0 (all sides two-sided).
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
    for (int i = 0; i < qp.nside; ++i)
    {
        qp.d(i) = -(S(1.0) + rng.pos());
        qp.d_mask(i) = 1.0;
        qp.m(i) = 0.0;
    }
    return qp;
}

// Turn general row `r` into a hard equality: both sides active, d(lo) = c,
// d(hi) = -c, so d(lo) + d(hi) == 0 (lo == hi).
template <class P, int NH>
void make_eq_row(DenseQp<P, NH>& qp, int r, S c)
{
    const int nb = qp.nb, ng = qp.ng;
    const int lo = nb + r;
    const int hi = nb + ng + nb + r;
    qp.d(lo) = c;
    qp.d(hi) = -c;
    qp.d_mask(lo) = 1.0;
    qp.d_mask(hi) = 1.0;
}

template <class P, int NH>
void run_dense_sens_case(bool with_eq, int N)
{
    const std::string tag =
        (with_eq ? "dense sens (eq row) N=" : "dense sens (ineq) N=")
        + std::to_string(N) + " ";
    const double tol = 1e-5;
    const double h = 1e-6;
    Rng rng;

    DenseQp<P, NH> qp = make_qp<P, NH>(N, rng);
    if (with_eq)
    {
        make_eq_row(qp, 0, S(0.5));
    }

    HpipmDenseQpSolver<P, NH> solv;
    DenseQpSol<P, NH> sol;
    sol.resize(N);
    const Status st = solv.solve(qp, sol);
    check(st == Status::kSolved,
          tag + "base solve kSolved (got "
              + std::to_string(static_cast<int>(st)) + ")");
    if (st != Status::kSolved)
    {
        return;
    }

    // ---- (i) cost seed: seed_g unit on variable 0 ----
    {
        const std::string pfx = tag + "seed_g ";
        DenseQpSeed<P, NH> seed(N);
        seed.seed_g(0) = S(1.0);

        DenseQpSol<P, NH> sens;
        sens.resize(N);
        const Status ss = solv.sens_forward(qp, sol, seed, sens);
        check(ss == Status::kSolved,
              pfx + "sens_forward kSolved (got "
                  + std::to_string(static_cast<int>(ss)) + ")");
        if (ss != Status::kSolved)
        {
            return;
        }

        DenseQp<P, NH> qp_p = qp, qp_m = qp;
        qp_p.g(0) += h;
        qp_m.g(0) -= h;
        HpipmDenseQpSolver<P, NH> sp, sm;
        DenseQpSol<P, NH> sol_p, sol_m;
        sol_p.resize(N);
        sol_m.resize(N);
        check(sp.solve(qp_p, sol_p) == Status::kSolved, pfx + "FD+ kSolved");
        check(sm.solve(qp_m, sol_m) == Status::kSolved, pfx + "FD- kSolved");
        const auto fd = (sol_p.v - sol_m.v) / S(2.0 * h);
        check_close(sens.v, fd, tol, pfx + "dv matches FD");
        std::printf("  %s: max |dv - FD| = %.3e\n", pfx.c_str(),
                    (sens.v - fd).cwiseAbs().maxCoeff());
    }

    // ---- (ii) bound seed on an ACTIVE, non-pinned side (exercises the
    //          active-set row: A_act dv = seed_d).  A pinned side (both
    //          sides of a lo == hi row active) is skipped: a one-sided seed
    //          on it is an inconsistent perturbation. ----
    {
        const std::string pfx = tag + "seed_d ";
        const int nlo = qp.nb + qp.ng;  // first hi-side index
        int lo = -1;
        for (int i = 0; i < qp.nside && lo < 0; ++i)
        {
            if (qp.d_mask(i) <= 0.5 || sol.t(i) >= 1e-10)
            {
                continue;
            }
            const int p = (i < nlo) ? i + nlo : i - nlo;  // partner side
            if (qp.d_mask(p) > 0.5 && sol.t(p) < 1e-10)
            {
                continue;  // pin / hard-equality pair
            }
            lo = i;
        }
        check(lo >= 0, pfx + "found an active non-pinned side");
        if (lo < 0)
        {
            return;
        }
        DenseQpSeed<P, NH> seed(N);
        seed.seed_d(lo) = S(1.0);

        DenseQpSol<P, NH> sens;
        sens.resize(N);
        const Status ss = solv.sens_forward(qp, sol, seed, sens);
        check(ss == Status::kSolved,
              pfx + "sens_forward kSolved (got "
                  + std::to_string(static_cast<int>(ss)) + ")");
        if (ss != Status::kSolved)
        {
            return;
        }

        DenseQp<P, NH> qp_p = qp, qp_m = qp;
        qp_p.d(lo) += h;
        qp_m.d(lo) -= h;
        HpipmDenseQpSolver<P, NH> sp, sm;
        DenseQpSol<P, NH> sol_p, sol_m;
        sol_p.resize(N);
        sol_m.resize(N);
        check(sp.solve(qp_p, sol_p) == Status::kSolved, pfx + "FD+ kSolved");
        check(sm.solve(qp_m, sol_m) == Status::kSolved, pfx + "FD- kSolved");
        const auto fd = (sol_p.v - sol_m.v) / S(2.0 * h);
        check_close(sens.v, fd, tol, pfx + "dv matches FD");
        std::printf("  %s: max |dv - FD| = %.3e\n", pfx.c_str(),
                    (sens.v - fd).cwiseAbs().maxCoeff());
    }

    // ---- (iii) adjoint: pure cost seed -> primal step equals forward ----
    {
        const std::string pfx = tag + "adj ";
        DenseQpSeed<P, NH> seed(N);
        seed.seed_g(0) = S(1.0);

        DenseQpSol<P, NH> fwd, adj;
        fwd.resize(N);
        adj.resize(N);
        const Status sf = solv.sens_forward(qp, sol, seed, fwd);
        const Status sa = solv.sens_adj(qp, sol, seed, adj);
        check(sf == Status::kSolved && sa == Status::kSolved,
              pfx + "forward & adjoint kSolved");
        if (sf == Status::kSolved && sa == Status::kSolved)
        {
            check_close(adj.v, fwd.v, 1e-10, pfx + "adjoint dv == forward dv");
        }
    }
}

}  // namespace

int run_dense_seed_5g_tests()
{
    run_dense_sens_case<TinyFree, Eigen::Dynamic>(true, 3);
    run_dense_sens_case<TinyFree, Eigen::Dynamic>(false, 3);
    if (failures == 0)
    {
        std::printf("All dense seed (5g) checks passed.\n");
    }
    return failures;
}

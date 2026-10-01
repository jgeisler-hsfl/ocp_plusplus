// Phase 1 sub-step 1e test: KKT solve (two-pass Riccati Cholesky) for
// fact_solve_kkt / solve_kkt. The solver's step (delta_z, delta_pi, delta_t,
// delta_lam) is checked against an independent dense solve of the coupled
// Newton system
//
//   M_k dz_k + D_k' dpi_k - J_k dpi_{k-1} = s_k      (k = 0..N)
//   D_k dz_k - J_{k+1} dz_{k+1} = -res_b_k           (k = 0..N-1)
//
// with M_k = H_k + C_k' diag(Gamma_k) C_k + reg*I_(u;x),
//      s_k = -res_g_k - C_k' gamma_k,
// and the closed forms dt = C dz - res_d, dlam = mask(-(res_m + lam dt)/t).
// C is the effective side-DC (same construction as the solver's
// usx_build_side_dc). Data is synthetic: a strongly-PD stage Hessian, random
// DC / BA / residuals, some absent sides (d_mask = 0), and a couple of
// one-sided (ineq) rows so the lo-side masking is exercised.

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
    std::mt19937_64 gen{7654321};
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
// Per-stage accessors (first / path / term) into Dynamic matrices.
// ---------------------------------------------------------------------

template <class IdxLo, class IdxHi>
MatD build_side_dc(const MatD& dc, const detail::QpLayout& lay,
                   const IdxLo& idxs_lo, const IdxHi& idxs_hi, int nvar,
                   int nslack, int nside);

template <class P, int NH>
struct Stg
{
    using D = QpDim<P>;
    const Qp<P, NH>& qp;
    const QpSol<P, NH>& sol;
    const QpRes<P, NH>& res;
    int N;

    int nvar(int k) const
    {
        return k == 0 ? D::nvar_first : (k == N ? D::nvar_term : D::nvar_path);
    }
    int zoff(int k) const
    {
        int s = 0;
        for (int j = 0; j < k; ++j)
        {
            s += nvar(j);
        }
        return s;
    }
    int nslack(int k) const
    {
        return k == 0 ? D::nslack_first
                      : (k == N ? D::nslack_term : D::nslack_path);
    }
    int nu(int k) const
    {
        return (k == N) ? 0 : D::nu;
    }
    const detail::QpLayout& lay(int k) const
    {
        return k == 0 ? D::lay_first
                      : (k == N ? D::lay_term : D::lay_path);
    }
    MatD hess(int k) const
    {
        if (k == 0)
        {
            return qp.first.hess;
        }
        if (k == N)
        {
            return qp.term.hess;
        }
        return qp.path[k - 1].hess;
    }
    MatD dc(int k) const
    {
        if (k == 0)
        {
            return qp.first.DC;
        }
        if (k == N)
        {
            return qp.term.DC;
        }
        return qp.path[k - 1].DC;
    }
    MatD ba(int k) const
    {
        if (k == 0)
        {
            return qp.first.BA;
        }
        if (k == N)
        {
            return MatD::Zero(D::nx, D::nu + D::nx);
        }
        return qp.path[k - 1].BA;
    }
    VecD res_g(int k) const
    {
        if (k == 0)
        {
            return res.res_g_first;
        }
        if (k == N)
        {
            return res.res_g_term;
        }
        return res.res_g_path[k - 1];
    }
    VecD res_b(int k) const
    {
        return res.res_b[k];
    }
    VecD res_d(int k) const
    {
        if (k == 0)
        {
            return res.res_d_first;
        }
        if (k == N)
        {
            return res.res_d_term;
        }
        return res.res_d_path[k - 1];
    }
    VecD res_m(int k) const
    {
        if (k == 0)
        {
            return res.res_m_first;
        }
        if (k == N)
        {
            return res.res_m_term;
        }
        return res.res_m_path[k - 1];
    }
    VecD lam(int k) const
    {
        if (k == 0)
        {
            return sol.lam_first;
        }
        if (k == N)
        {
            return sol.lam_term;
        }
        return sol.lam_path[k - 1];
    }
    VecD t(int k) const
    {
        if (k == 0)
        {
            return sol.t_first;
        }
        if (k == N)
        {
            return sol.t_term;
        }
        return sol.t_path[k - 1];
    }
    VecD dmask(int k) const
    {
        if (k == 0)
        {
            return qp.first.d_mask;
        }
        if (k == N)
        {
            return qp.term.d_mask;
        }
        return qp.path[k - 1].d_mask;
    }
    VecD dz(int k, const QpSol<P, NH>& step) const
    {
        if (k == 0)
        {
            return step.ux_first;
        }
        if (k == N)
        {
            return step.ux_term;
        }
        return step.ux_path[k - 1];
    }
    MatD side_dc(int k) const
    {
        const int nv = nvar(k);
        const int ns = nslack(k);
        const int nside = static_cast<int>(dmask(k).size());
        const MatD dc = this->dc(k);
        if (k == 0)
        {
            return build_side_dc(dc, D::lay_first, D::idxs_lo_first,
                                 D::idxs_hi_first, nv, ns, nside);
        }
        if (k == N)
        {
            return build_side_dc(dc, D::lay_term, D::idxs_lo_term,
                                 D::idxs_hi_term, nv, ns, nside);
        }
        return build_side_dc(dc, D::lay_path, D::idxs_lo_path,
                             D::idxs_hi_path, nv, ns, nside);
    }
};

// Effective side-DC C (nside x nvar, natural (u;x;s) order): mirrors the
// solver's usx_build_side_dc. lo side of row r = +DC row (hi-slack col
// zeroed); hi side = -DC row (lo-slack col zeroed); slack side j = unit at
// column (nvar - nslack + j).
template <class IdxLo, class IdxHi>
MatD build_side_dc(const MatD& dc, const detail::QpLayout& lay,
                   const IdxLo& idxs_lo, const IdxHi& idxs_hi, int nvar,
                   int nslack, int nside)
{
    MatD c = MatD::Zero(nside, nvar);
    const int nrow = static_cast<int>(dc.rows());
    const int lo = lay.lo_size();
    for (int r = 0; r < nrow; ++r)
    {
        const int sl = lay.side_lo(r);
        const int sh = lay.side_hi(r);
        if (sl >= 0)
        {
            c.row(sl) = dc.row(r);
            const int ch = idxs_hi[r];
            if (ch >= 0)
            {
                c(sl, ch) = 0.0;
            }
        }
        c.row(sh) = -dc.row(r);
        const int cl = idxs_lo[r];
        if (cl >= 0)
        {
            c(sh, cl) = 0.0;
        }
    }
    const int s0 = nvar - nslack;
    for (int j = 0; j < nslack; ++j)
    {
        c(lo + nrow + j, s0 + j) = 1.0;
    }
    return c;
}

// ---------------------------------------------------------------------
// Data fill: strongly-PD stage Hessian, random DC / BA / residuals, a few
// absent sides.
// ---------------------------------------------------------------------

template <class P, int NH>
void fill_kkt_data(Qp<P, NH>& qp, QpSol<P, NH>& sol, QpRes<P, NH>& res, int N)
{
    using D = QpDim<P>;
    Rng rng;

    auto fill_stage = [&](auto& st, auto& lam, auto& t, auto& res_g,
                          auto& res_d, auto& res_m, const auto& idxs_lo,
                          const auto& idxs_hi, int nvar, int nslack,
                          const detail::QpLayout& lay)
    {
        const int nside = lay.nside();
        const int nux = nvar - nslack;
        // PD Hessian on (u;x): I + B' B; slack diagonal = 1.0.
        MatD B(nux, nux);
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                B(i, j) = rng.uniform();
            }
        }
        const MatD hux = B.transpose() * B;
        for (int i = 0; i < nvar; ++i)
        {
            for (int j = 0; j < nvar; ++j)
            {
                st.hess(i, j) = 0.0;
            }
        }
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
        // DC: random (u;x) Jacobian + slack columns per the qp.hpp contract.
        for (int r = 0; r < static_cast<int>(st.DC.rows()); ++r)
        {
            for (int j = 0; j < nux; ++j)
            {
                st.DC(r, j) = rng.uniform();
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
        for (int i = 0; i < nside; ++i)
        {
            st.d(i) = rng.uniform();
            st.d_mask(i) = (rng.pos() < 0.85) ? 1.0 : 0.0;
            st.m(i) = rng.uniform();
        }
        for (int i = 0; i < nvar; ++i)
        {
            res_g(i) = rng.uniform();
        }
        for (int i = 0; i < nside; ++i)
        {
            res_d(i) = rng.uniform();
            res_m(i) = rng.uniform();
            lam(i) = rng.pos();
            t(i) = rng.pos();
        }
    };

    fill_stage(qp.first, sol.lam_first, sol.t_first, res.res_g_first,
               res.res_d_first, res.res_m_first, D::idxs_lo_first,
               D::idxs_hi_first, D::nvar_first, D::nslack_first,
               D::lay_first);
    for (int i = 0; i < D::nx; ++i)
    {
        for (int j = 0; j < D::nu + D::nx; ++j)
        {
            qp.first.BA(i, j) = rng.uniform();
        }
        qp.first.b(i) = rng.uniform();
        res.res_b[0](i) = rng.uniform();
    }
    for (int k = 1; k < N; ++k)
    {
        fill_stage(qp.path[k - 1], sol.lam_path[k - 1],
                   sol.t_path[k - 1], res.res_g_path[k - 1],
                   res.res_d_path[k - 1], res.res_m_path[k - 1],
                   D::idxs_lo_path, D::idxs_hi_path, D::nvar_path,
                   D::nslack_path, D::lay_path);
        for (int i = 0; i < D::nx; ++i)
        {
            for (int j = 0; j < D::nu + D::nx; ++j)
            {
                qp.path[k - 1].BA(i, j) = rng.uniform();
            }
            qp.path[k - 1].b(i) = rng.uniform();
            res.res_b[k](i) = rng.uniform();
        }
    }
    fill_stage(qp.term, sol.lam_term, sol.t_term, res.res_g_term,
               res.res_d_term, res.res_m_term, D::idxs_lo_term,
               D::idxs_hi_term, D::nvar_term, D::nslack_term, D::lay_term);
}

// ---------------------------------------------------------------------
// Dense oracle: assemble the coupled (delta_z, delta_pi) Newton system and
// solve it directly.
// ---------------------------------------------------------------------

template <class P, int NH>
void dense_kkt(const Stg<P, NH>& sg, double reg, VecD& dz_out,
               VecD& dpi_out)
{
    using D = QpDim<P>;
    const int N = sg.N;
    const int nx = D::nx;

    // unknown layout: [dz_0..dz_N (zdim)] [dpi_0..dpi_{N-1} (N*nx)]
    int zdim = 0;
    for (int k = 0; k <= N; ++k)
    {
        zdim += sg.nvar(k);
    }
    const int pdim = N * nx;
    const int dim = zdim + pdim;

    auto zoff = [&](int k)
    {
        int s = 0;
        for (int j = 0; j < k; ++j)
        {
            s += sg.nvar(j);
        }
        return s;
    };
    auto poff = [&](int k)
    {
        return zdim + k * nx;
    };

    MatD K = MatD::Zero(dim, dim);
    VecD r = VecD::Zero(dim);

    for (int k = 0; k <= N; ++k)
    {
        const int nv = sg.nvar(k);
        const int ns = sg.nslack(k);
        const int nside = static_cast<int>(sg.dmask(k).size());
        const int nux = nv - ns;
        const int xoff = nux - nx;  // x-part offset inside z_k

        // C_k, gamma_k, Gamma_k (masked), M_k, s_k
        const MatD c = sg.side_dc(k);
        const VecD dmask = sg.dmask(k);
        const VecD lam = sg.lam(k);
        const VecD tv = sg.t(k);
        const VecD res_d = sg.res_d(k);
        const VecD res_m = sg.res_m(k);
        VecD gamma = VecD::Zero(nside);
        VecD Gamma = VecD::Zero(nside);
        for (int i = 0; i < nside; ++i)
        {
            if (dmask(i) > 0.5)
            {
                Gamma(i) = lam(i) / tv(i);
                gamma(i) = (res_m(i) - lam(i) * res_d(i)) / tv(i);
            }
        }
        MatD M = sg.hess(k);
        M += c.transpose() * (Gamma.asDiagonal() * c);
        for (int i = 0; i < nux; ++i)
        {
            M(i, i) += reg;
        }
        const VecD s = -sg.res_g(k) - c.transpose() * gamma;

        // stationarity row block (nv rows)
        for (int a = 0; a < nv; ++a)
        {
            const int row = zoff(k) + a;
            for (int b = 0; b < nv; ++b)
            {
                K(row, zoff(k) + b) += M(a, b);
            }
            if (k <= N - 1)
            {
                // D_k' dpi_k, a in the (u;x) part only
                if (a < nux)
                {
                    const MatD dk = sg.ba(k);
                    for (int j = 0; j < nx; ++j)
                    {
                        K(row, poff(k) + j) += dk(j, a);
                    }
                }
            }
            if (k >= 1 && a >= xoff && a < xoff + nx)
            {
                // -J_k dpi_{k-1} (x-part)
                K(row, poff(k - 1) + (a - xoff)) -= 1.0;
            }
            r(row) = s(a);
        }
    }
    // dynamics rows (N * nx)
    for (int k = 0; k < N; ++k)
    {
        const MatD dk = sg.ba(k);
        const int nv_next = sg.nvar(k + 1);
        const int ns_next = sg.nslack(k + 1);
        const int nux_next = nv_next - ns_next;
        const int xoff_next = nux_next - nx;
        for (int i = 0; i < nx; ++i)
        {
            const int row = zdim + k * nx + i;
            for (int a = 0; a < sg.nvar(k) - sg.nslack(k); ++a)
            {
                K(row, zoff(k) + a) += dk(i, a);
            }
            K(row, zoff(k + 1) + xoff_next + i) -= 1.0;
            r(row) = -sg.res_b(k)(i);
        }
    }

    Eigen::FullPivLU<MatD> lu(K);
    const VecD x = lu.solve(r);
    // sanity: dense solve residual (also flags a singular KKT matrix)
    const VecD resid = K * x - r;
    const double kkt_res = resid.cwiseAbs().maxCoeff();
    check(kkt_res < 1e-7, "dense KKT solve residual too large: " + std::to_string(kkt_res));
    dz_out = x.head(zdim);
    dpi_out = x.tail(pdim);
}

// ---------------------------------------------------------------------
// One case: fill, solve (fact_solve_kkt), compare against the dense oracle.
// ---------------------------------------------------------------------

template <class P, int NH = Eigen::Dynamic>
void kkt_case(const char* name, int N)
{
    using D = QpDim<P>;
    const double reg = 1e-15;  // HpipmOptions::reg_prim
    const double tol = 1e-9;

    Qp<P, NH> qp(N);
    QpSol<P, NH> sol(N);
    QpRes<P, NH> res(N);
    fill_kkt_data(qp, sol, res, N);

    HpipmQpSolver<P, NH> solver;
    QpSol<P, NH> step(N);
    const Status st = solver.fact_solve_kkt(qp, sol, res, step);
    check(st == Status::kSolved,
          std::string(name) + ": fact_solve_kkt status");
    if (st != Status::kSolved)
    {
        return;
    }

    Stg<P, NH> sg{qp, sol, res, N};
    VecD dz_ref, dpi_ref;
    dense_kkt(sg, reg, dz_ref, dpi_ref);

    const std::string p = name;
    // compare delta_z (natural order) per stage
    auto cmp_dz = [&](int k)
    {
        const VecD dz = sg.dz(k, step);
        const int nv = sg.nvar(k);
        const VecD ref = dz_ref.segment(sg.zoff(k), nv);
        check_close(dz, ref, tol, p + ": dz stage " + std::to_string(k));
    };
    for (int k = 0; k <= N; ++k)
    {
        cmp_dz(k);
    }
    // compare delta_pi (k = 0..N-1)
    for (int k = 0; k < N; ++k)
    {
        check_close(step.pi[k], dpi_ref.segment(k * D::nx, D::nx), tol,
                    p + ": dpi[" + std::to_string(k) + "]");
    }
    // closed forms: dt = C dz - res_d; dlam = mask(-(res_m + lam dt)/t)
    for (int k = 0; k <= N; ++k)
    {
        const int nside = static_cast<int>(sg.dmask(k).size());
        const MatD c = sg.side_dc(k);
        const VecD dz = sg.dz(k, step);
        const VecD dt_ref = c * dz - sg.res_d(k);
        const VecD dmask = sg.dmask(k);
        const VecD lam = sg.lam(k);
        const VecD tv = sg.t(k);
        VecD dlam_ref = VecD::Zero(nside);
        for (int i = 0; i < nside; ++i)
        {
            if (dmask(i) > 0.5)
            {
                dlam_ref(i) = -(sg.res_m(k)(i) + lam(i) * dt_ref(i)) / tv(i);
            }
        }
        if (k == 0)
        {
            check_close(step.t_first, dt_ref, tol,
                        p + ": dt_first");
            check_close(step.lam_first, dlam_ref, tol, p + ": dlam_first");
        }
        else if (k == N)
        {
            check_close(step.t_term, dt_ref, tol, p + ": dt_term");
            check_close(step.lam_term, dlam_ref, tol, p + ": dlam_term");
        }
        else
        {
            check_close(step.t_path[k - 1], dt_ref, tol,
                        p + ": dt_path[" + std::to_string(k - 1) + "]");
            check_close(step.lam_path[k - 1], dlam_ref, tol,
                        p + ": dlam_path[" + std::to_string(k - 1) + "]");
        }
    }

    // solve_kkt (factor reuse) must agree with fact_solve_kkt here since the
    // gamma is unchanged: re-run the reduced/forward passes and compare.
    QpSol<P, NH> step2(N);
    solver.solve_kkt(qp, sol, res, step2);
    for (int k = 0; k < N; ++k)
    {
        check_close(step2.pi[k], step.pi[k], tol,
                    p + ": solve_kkt dpi[" + std::to_string(k) + "]");
    }
    check_close(step2.ux_first, step.ux_first, tol, p + ": solve_kkt dz0");
}

}  // namespace

int run_kkt_1e_tests()
{
    failures = 0;
    kkt_case<DoubleIntegrator, Eigen::Dynamic>("DI dyn N=1", 1);
    kkt_case<DoubleIntegrator, Eigen::Dynamic>("DI dyn N=2", 2);
    kkt_case<MassSpring, Eigen::Dynamic>("MS dyn N=1", 1);
    kkt_case<MassSpring, Eigen::Dynamic>("MS dyn N=2", 2);
    kkt_case<MassSpring, 2>("MS NH=2 N=2", 2);

    if (failures == 0)
    {
        std::printf("All KKT solve (1e) checks passed.\n");
    }
    return failures;
}

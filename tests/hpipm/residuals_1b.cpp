// Phase 1 sub-step 1b test: compute_residuals (port of HPIPM
// OCP_QP_RES_COMPUTE). Checks:
//   - zero-solution case with hand-computed expected values (N = 1)
//   - random data + random iterate against an independent, explicitly
//     looped reference implementation (N = 1 and 2, with/without slacks,
//     dynamic and fixed horizon)

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

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

void check_scalar(double a, double b, double tol, const std::string& msg)
{
    if (std::fabs(a - b) > tol)
    {
        std::fprintf(stderr, "FAIL: %s (%.17g vs %.17g)\n", msg.c_str(), a, b);
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

// ---------------------------------------------------------------------
// Independent reference: explicit per-side / per-element loops, no
// matrix products, mirroring the worklog sec. 1 definitions directly.
// ---------------------------------------------------------------------

template <class P, int NH>
void ref_residuals(const Qp<P, NH>& in, const QpSol<P, NH>& sol,
                   QpRes<P, NH>& out)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    using Pi = Eigen::Matrix<S, D::nx, 1>;
    using Ba = Eigen::Matrix<S, D::nx, D::nu + D::nx>;
    const int N = in.N;

    S obj = 0.0;
    S dual_gap = 0.0;
    S res_mu_sum = 0.0;
    int nc_mask = 0;
    S gmax = 0.0, bmax = 0.0, dmax = 0.0, mmax = 0.0;

    auto process = [&](auto& hess, auto& grad, auto& dc, auto& d,
                       auto& dmask, auto& m, auto& z, auto& lam, auto& t,
                       const Pi* pi_prev, const Pi* pi_cur, const Pi* x_next,
                       const Ba* ba, const Pi* b, detail::QpLayout lay,
                       const auto& idxs_lo, const auto& idxs_hi,
                       auto& res_g, auto& res_d, auto& res_m, Pi* res_b)
    {
        const int nvar = static_cast<int>(z.size());
        const int nrow = static_cast<int>(dc.rows());
        const int lo = lay.lo_size();
        const int nside = static_cast<int>(d.size());
        const int nslack = nside - lo - nrow;
        const int s0 = nvar - nslack;
        const int xoff = nvar - nslack - D::nx;
        const int nux = nvar - nslack;  // (u;x)-part width

        // stationarity base: H z + g, element-wise
        std::vector<S> hz(nvar);
        for (int j = 0; j < nvar; ++j)
        {
            S acc = 0.0;
            for (int i = 0; i < nvar; ++i)
            {
                acc += hess(j, i) * z(i);
            }
            hz[j] = acc;
            res_g(j) = acc + grad(j);
        }
        S hz_z = 0.0;
        S g_z = 0.0;
        for (int j = 0; j < nvar; ++j)
        {
            hz_z += hz[j] * z(j);
            g_z += grad(j) * z(j);
        }
        obj += 0.5 * hz_z + g_z;
        dual_gap += hz_z + g_z;

        // row values on the (u;x) part only (HPIPM: DCt covers (u;x))
        std::vector<S> v(nrow, 0.0);
        for (int r = 0; r < nrow; ++r)
        {
            for (int j = 0; j < nux; ++j)
            {
                v[r] += dc(r, j) * z(j);
            }
        }

        // per-side feasibility / complementarity / multiplier coupling
        for (int i = 0; i < nside; ++i)
        {
            const S mask = (dmask(i) > 0.5) ? 1.0 : 0.0;
            nc_mask += (mask > 0.5) ? 1 : 0;
            const S lami = mask * lam(i);
            dual_gap -= d(i) * lami;  // x_ocp_qp_res.c:497
            if (i < lo)
            {
                // lo side of the row whose side_lo index is i
                int r = -1;
                for (int rr = 0; rr < nrow; ++rr)
                {
                    if (lay.side_lo(rr) == i)
                    {
                        r = rr;
                        break;
                    }
                }
                S vlo = v[r];
                const int cl = idxs_lo[r];
                if (cl >= 0)
                {
                    vlo += z(cl);
                }
                res_d(i) = mask * (d(i) + t(i) - vlo);
                for (int j = 0; j < nux; ++j)
                {
                    res_g(j) -= lami * dc(r, j);
                }
                if (cl >= 0)
                {
                    res_g(cl) -= lami;
                }
            }
            else if (i < lo + nrow)
            {
                // hi side of row i - lo
                const int r = i - lo;
                S vhi = v[r];
                const int ch = idxs_hi[r];
                if (ch >= 0)
                {
                    vhi -= z(ch);
                }
                res_d(i) = mask * (d(i) + t(i) + vhi);
                for (int j = 0; j < nux; ++j)
                {
                    res_g(j) += lami * dc(r, j);
                }
                if (ch >= 0)
                {
                    res_g(ch) -= lami;
                }
            }
            else
            {
                // slack side j: value is the slack variable itself
                const int j = i - lo - nrow;
                res_d(i) = mask * (d(i) + t(i) - z(s0 + j));
                res_g(s0 + j) -= lami;
            }
            res_m(i) = mask * (lam(i) * t(i) - m(i));
            dmax = std::max(dmax, std::fabs(res_d(i)));
            mmax = std::max(mmax, std::fabs(res_m(i)));
            res_mu_sum += std::fabs(res_m(i));
        }
        // dynamics coupling
        if (ba != nullptr)
        {
            for (int i = 0; i < D::nx; ++i)
            {
                S acc = (*b)(i) - (*x_next)(i);
                for (int j = 0; j < D::nu + D::nx; ++j)
                {
                    acc += (*ba)(i, j) * z(j);
                }
                (*res_b)(i) = acc;
                bmax = std::max(bmax, std::fabs(acc));
            }
            for (int i = 0; i < D::nx; ++i)
            {
                dual_gap -= (*b)(i) * (*pi_cur)(i);
                for (int j = 0; j < D::nu + D::nx; ++j)
                {
                    res_g(j) += (*ba)(i, j) * (*pi_cur)(i);
                }
            }
        }
        if (pi_prev != nullptr)
        {
            for (int i = 0; i < D::nx; ++i)
            {
                res_g(xoff + i) -= (*pi_prev)(i);
            }
        }
        // res_g_max over the fully coupled res_g (HPIPM inf-norm is a
        // separate pass after the stage loop, x_ocp_qp_res.c:689)
        for (int j = 0; j < nvar; ++j)
        {
            gmax = std::max(gmax, std::fabs(res_g(j)));
        }
    };

    // first stage (k = 0)
    {
        Pi x1;
        if (N > 1)
        {
            x1 = sol.ux_path[0].segment(D::nu, D::nx);
        }
        else
        {
            x1 = sol.ux_term.head(D::nx);
        }
        process(in.first.hess, in.first.grad, in.first.DC, in.first.d,
                in.first.d_mask, in.first.m, sol.ux_first, sol.lam_first,
                sol.t_first, nullptr, &sol.pi[0], &x1, &in.first.BA,
                &in.first.b, D::lay_first, D::idxs_lo_first,
                D::idxs_hi_first, out.res_g_first, out.res_d_first,
                out.res_m_first, &out.res_b[0]);
    }
    // path stages (k = 1..N-1)
    for (int k = 1; k < N; ++k)
    {
        const int i = k - 1;
        Pi x_next;
        if (k + 1 < N)
        {
            x_next = sol.ux_path[k].segment(D::nu, D::nx);
        }
        else
        {
            x_next = sol.ux_term.head(D::nx);
        }
        process(in.path[i].hess, in.path[i].grad, in.path[i].DC,
                in.path[i].d, in.path[i].d_mask, in.path[i].m,
                sol.ux_path[i], sol.lam_path[i], sol.t_path[i],
                &sol.pi[k - 1], &sol.pi[k], &x_next, &in.path[i].BA,
                &in.path[i].b, D::lay_path, D::idxs_lo_path,
                D::idxs_hi_path, out.res_g_path[i], out.res_d_path[i],
                out.res_m_path[i], &out.res_b[k]);
    }
    // terminal stage (k = N)
    process(in.term.hess, in.term.grad, in.term.DC, in.term.d,
            in.term.d_mask, in.term.m, sol.ux_term, sol.lam_term,
            sol.t_term, &sol.pi[N - 1], nullptr, nullptr, nullptr,
            nullptr, D::lay_term, D::idxs_lo_term, D::idxs_hi_term,
            out.res_g_term, out.res_d_term, out.res_m_term, nullptr);

    out.res_g_max = gmax;
    out.res_b_max = bmax;
    out.res_d_max = dmax;
    out.res_m_max = mmax;
    out.res_mu_sum = res_mu_sum;
    out.res_mu = (nc_mask > 0) ? res_mu_sum / nc_mask : 0.0;
    out.obj = obj;
    out.dual_gap = dual_gap;
}

// ---------------------------------------------------------------------
// Data fillers
// ---------------------------------------------------------------------

template <class P, int NH>
void zero_all(Qp<P, NH>& qp, QpSol<P, NH>& sol, int N)
{
    auto zero_base = [&](auto& st)
    {
        st.hess.setZero();
        st.grad.setZero();
        st.DC.setZero();
        st.d.setZero();
        st.d_mask.setZero();
        st.m.setZero();
    };
    auto zero_dyn = [&](auto& st)
    {
        st.BA.setZero();
        st.b.setZero();
    };
    zero_base(qp.first);
    zero_dyn(qp.first);
    for (int k = 1; k < N; ++k)
    {
        zero_base(qp.path[k - 1]);
        zero_dyn(qp.path[k - 1]);
    }
    zero_base(qp.term);

    sol.ux_first.setZero();
    for (int k = 1; k < N; ++k)
    {
        sol.ux_path[k - 1].setZero();
    }
    sol.ux_term.setZero();
    for (int k = 0; k < N; ++k)
    {
        sol.pi[k].setZero();
    }
    sol.lam_first.setZero();
    for (int k = 1; k < N; ++k)
    {
        sol.lam_path[k - 1].setZero();
    }
    sol.lam_term.setZero();
    sol.t_first.setZero();
    for (int k = 1; k < N; ++k)
    {
        sol.t_path[k - 1].setZero();
    }
    sol.t_term.setZero();
}

template <class P, int NH>
void fill_random(Qp<P, NH>& qp, QpSol<P, NH>& sol, int N)
{
    using D = QpDim<P>;
    Rng rng;

    // Fill a stage's DC: random (u;x)-jacobian part, then the slack columns
    // per the qp.hpp contract (0 for non-soft rows, +1 at idxs_lo,
    // -1 at idxs_hi for soft sides).
    auto fill_dc = [&](auto& dc, const auto& idxs_lo, const auto& idxs_hi,
                       int nux)
    {
        const int nrow = static_cast<int>(dc.rows());
        const int nv = static_cast<int>(dc.cols());
        for (int r = 0; r < nrow; ++r)
        {
            for (int j = 0; j < nux; ++j)
            {
                dc(r, j) = rng.uniform();
            }
            for (int j = nux; j < nv; ++j)
            {
                dc(r, j) = 0.0;
            }
            if (idxs_lo[r] >= 0)
            {
                dc(r, idxs_lo[r]) = 1.0;
            }
            if (idxs_hi[r] >= 0)
            {
                dc(r, idxs_hi[r]) = -1.0;
            }
        }
    };
    auto fill_base = [&](auto& st)
    {
        const int nv = static_cast<int>(st.hess.rows());
        for (int i = 0; i < nv; ++i)
        {
            for (int j = 0; j < nv; ++j)
            {
                st.hess(i, j) = rng.uniform();
            }
        }
        st.hess = 0.5 * (st.hess + st.hess.transpose());
        for (int i = 0; i < nv; ++i)
        {
            st.grad(i) = rng.uniform();
        }
        for (int i = 0; i < static_cast<int>(st.d.size()); ++i)
        {
            st.d(i) = rng.uniform();
            st.d_mask(i) = (rng.pos() < 0.8) ? 1.0 : 0.0;
        }
        for (int i = 0; i < static_cast<int>(st.m.size()); ++i)
        {
            st.m(i) = rng.uniform();
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
        }
        for (int i = 0; i < D::nx; ++i)
        {
            st.b(i) = rng.uniform();
        }
    };
    auto fill_sol = [&](auto& z, auto& lam, auto& t)
    {
        for (int i = 0; i < static_cast<int>(z.size()); ++i)
        {
            z(i) = rng.uniform();
        }
        for (int i = 0; i < static_cast<int>(lam.size()); ++i)
        {
            lam(i) = rng.pos();
        }
        for (int i = 0; i < static_cast<int>(t.size()); ++i)
        {
            t(i) = rng.pos() + 0.01;
        }
    };

    fill_base(qp.first);
    fill_dyn(qp.first);
    fill_dc(qp.first.DC, D::idxs_lo_first, D::idxs_hi_first, D::nu + D::nx);
    fill_sol(sol.ux_first, sol.lam_first, sol.t_first);
    for (int k = 1; k < N; ++k)
    {
        fill_base(qp.path[k - 1]);
        fill_dyn(qp.path[k - 1]);
        fill_dc(qp.path[k - 1].DC, D::idxs_lo_path, D::idxs_hi_path,
                D::nu + D::nx);
        fill_sol(sol.ux_path[k - 1], sol.lam_path[k - 1],
                 sol.t_path[k - 1]);
    }
    fill_base(qp.term);
    fill_dc(qp.term.DC, D::idxs_lo_term, D::idxs_hi_term, D::nx);
    fill_sol(sol.ux_term, sol.lam_term, sol.t_term);
    for (int k = 0; k < N; ++k)
    {
        for (int i = 0; i < D::nx; ++i)
        {
            sol.pi[k](i) = rng.uniform();
        }
    }
}

// ---------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------

// Zero iterate, known data: every residual has a hand-computed value.
template <class P, int NH = Eigen::Dynamic>
void check_residuals_zero(const char* name)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    Qp<P, NH> qp(1);
    QpSol<P, NH> sol(1);
    zero_all(qp, sol, 1);

    // known non-zero data (pattern-filled, so it works for any problem dims)
    for (int i = 0; i < D::nvar_first; ++i)
    {
        qp.first.grad(i) = double(i) + 1.0;
    }
    for (int i = 0; i < D::nx; ++i)
    {
        qp.first.b(i) = 10.0 + double(i);
    }
    for (int i = 0; i < D::nvar_term; ++i)
    {
        qp.term.grad(i) = -double(i) - 1.0;
    }
    for (int i = 0; i < D::nside_first; ++i)
    {
        qp.first.d(i) = double(i) + 1.0;
        qp.first.d_mask(i) = (i % 2 == 0) ? 1.0 : 0.0;
    }
    for (int i = 0; i < D::nside_term; ++i)
    {
        qp.term.d(i) = 2.0 * (i + 1);
        qp.term.d_mask(i) = 1.0;
    }

    HpipmQpSolver<P, NH> solver;
    QpRes<P, NH> res(1);
    solver.compute_residuals(qp, sol, res);

    // lam = t = pi = z = 0  =>  res_g = grad, res_b = b, res_d = d*mask,
    // res_m = 0, obj = 0, dual_gap = 0.
    check_close(res.res_g_first, qp.first.grad, 1e-14, "zero: res_g_first");
    check_close(res.res_g_term, qp.term.grad, 1e-14, "zero: res_g_term");
    check_close(res.res_b[0], qp.first.b, 1e-14, "zero: res_b");
    for (int i = 0; i < D::nside_first; ++i)
    {
        const S exp = qp.first.d_mask(i) * qp.first.d(i);
        check(std::fabs(res.res_d_first(i) - exp) < 1e-14,
              "zero: res_d_first(" + std::to_string(i) + ")");
    }
    for (int i = 0; i < D::nside_term; ++i)
    {
        check(std::fabs(res.res_d_term(i) - qp.term.d(i)) < 1e-14,
              "zero: res_d_term(" + std::to_string(i) + ")");
    }
    check_close(res.res_m_first, Eigen::Matrix<S, D::nside_first, 1>::Zero(),
                1e-14, "zero: res_m_first");
    check_close(res.res_m_term, Eigen::Matrix<S, D::nside_term, 1>::Zero(),
                1e-14, "zero: res_m_term");
    check_scalar(res.obj, 0.0, 1e-14, "zero: obj");
    check_scalar(res.dual_gap, 0.0, 1e-14, "zero: dual_gap");
    check_scalar(res.res_mu, 0.0, 1e-14, "zero: res_mu");

    // inf-norm maxima, hand-computed from the data set above
    double exp_gmax = 0.0, exp_bmax = 0.0, exp_dmax = 0.0;
    for (int i = 0; i < D::nvar_first; ++i)
    {
        exp_gmax = std::max(exp_gmax, std::fabs(qp.first.grad(i)));
    }
    for (int i = 0; i < D::nvar_term; ++i)
    {
        exp_gmax = std::max(exp_gmax, std::fabs(qp.term.grad(i)));
    }
    for (int i = 0; i < D::nx; ++i)
    {
        exp_bmax = std::max(exp_bmax, std::fabs(qp.first.b(i)));
    }
    for (int i = 0; i < D::nside_first; ++i)
    {
        exp_dmax = std::max(exp_dmax,
                            std::fabs(qp.first.d_mask(i) * qp.first.d(i)));
    }
    for (int i = 0; i < D::nside_term; ++i)
    {
        exp_dmax = std::max(exp_dmax, std::fabs(qp.term.d(i)));
    }
    check_scalar(res.res_g_max, exp_gmax, 1e-14, std::string(name) + ": res_g_max");
    check_scalar(res.res_b_max, exp_bmax, 1e-14, std::string(name) + ": res_b_max");
    check_scalar(res.res_d_max, exp_dmax, 1e-14, std::string(name) + ": res_d_max");
    check(res.res_m_max == 0.0, std::string(name) + ": res_m_max");
}

template <class P, int NH>
void check_residuals_random(const char* name, int N)
{
    Qp<P, NH> qp(N);
    QpSol<P, NH> sol(N);
    fill_random(qp, sol, N);

    HpipmQpSolver<P, NH> solver;
    QpRes<P, NH> res(N);
    solver.compute_residuals(qp, sol, res);

    QpRes<P, NH> ref(N);
    ref_residuals(qp, sol, ref);

    const std::string p = name;
    const double tol = 1e-10;
    check_close(res.res_g_first, ref.res_g_first, tol, p + ": res_g_first");
    for (int k = 1; k < N; ++k)
    {
        check_close(res.res_g_path[k - 1], ref.res_g_path[k - 1], tol,
                    p + ": res_g_path[" + std::to_string(k - 1) + "]");
        check_close(res.res_b[k], ref.res_b[k], tol,
                    p + ": res_b[" + std::to_string(k) + "]");
    }
    check_close(res.res_g_term, ref.res_g_term, tol, p + ": res_g_term");
    check_close(res.res_b[0], ref.res_b[0], tol, p + ": res_b[0]");
    check_close(res.res_d_first, ref.res_d_first, tol, p + ": res_d_first");
    check_close(res.res_d_term, ref.res_d_term, tol, p + ": res_d_term");
    check_close(res.res_m_first, ref.res_m_first, tol, p + ": res_m_first");
    check_close(res.res_m_term, ref.res_m_term, tol, p + ": res_m_term");
    for (int k = 1; k < N; ++k)
    {
        check_close(res.res_d_path[k - 1], ref.res_d_path[k - 1], tol,
                    p + ": res_d_path[" + std::to_string(k - 1) + "]");
        check_close(res.res_m_path[k - 1], ref.res_m_path[k - 1], tol,
                    p + ": res_m_path[" + std::to_string(k - 1) + "]");
    }
    check_scalar(res.res_g_max, ref.res_g_max, tol, p + ": res_g_max");
    check_scalar(res.res_b_max, ref.res_b_max, tol, p + ": res_b_max");
    check_scalar(res.res_d_max, ref.res_d_max, tol, p + ": res_d_max");
    check_scalar(res.res_m_max, ref.res_m_max, tol, p + ": res_m_max");
    check_scalar(res.res_mu_sum, ref.res_mu_sum, tol, p + ": res_mu_sum");
    check_scalar(res.res_mu, ref.res_mu, tol, p + ": res_mu");
    check_scalar(res.obj, ref.obj, tol, p + ": obj");
    check_scalar(res.dual_gap, ref.dual_gap, tol, p + ": dual_gap");
}

}  // namespace

int run_residuals_1b_tests()
{
    failures = 0;
    check_residuals_zero<DoubleIntegrator, Eigen::Dynamic>("DI dyn N=1 zero");
    check_residuals_zero<MassSpring, Eigen::Dynamic>("MS dyn N=1 zero");
    check_residuals_random<DoubleIntegrator, Eigen::Dynamic>("DI dyn N=1", 1);
    check_residuals_random<DoubleIntegrator, Eigen::Dynamic>("DI dyn N=2", 2);
    check_residuals_random<MassSpring, Eigen::Dynamic>("MS dyn N=1", 1);
    check_residuals_random<MassSpring, Eigen::Dynamic>("MS dyn N=2", 2);
    check_residuals_random<MassSpring, 2>("MS NH=2 N=2", 2);

    if (failures == 0)
    {
        std::printf("All compute_residuals (1b) checks passed.\n");
    }
    return failures;
}

// Phase 1 sub-step 1c test: init_point (port of HPIPM OCP_QP_INIT_VAR,
// var_init_scheme = 1, t0_init = 2, warm_start = 0). Checks:
//   - interior: t > 0 and lam > 0 (finite) on every side of every stage
//   - decision variables repaired into the interior (|z| finite)
//   - res_mu == mu0 after init (lam_i = mu0/t_i => lam_i*t_i = mu0, m = 0),
//     including with some sides masked (d_mask = 0)
// for DI and MS, N = 1 and 2, dynamic and fixed horizon.

#include <cmath>
#include <cstdio>
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

void check_scalar(double a, double b, double tol, const std::string& msg)
{
    if (std::fabs(a - b) > tol)
    {
        std::fprintf(stderr, "FAIL: %s (%.17g vs %.17g)\n", msg.c_str(), a, b);
        ++failures;
    }
}

// Fill a stage's side data with a feasible [-1, 1] bound pattern:
//   lo sides   d = -1   (lo  = -1)
//   hi sides   d = -1   (hi  = +1), except ineq rows (hi = 0 => d = 0)
//   slack sides d = 0
// d_mask is 1 everywhere except every 5th side (= 0) to exercise masking.
// hess = I, grad = 0, DC = 0 (general-row values v = 0 at the origin), m = 0.
template <class St>
void fill_base(St& st, detail::QpLayout lay)
{
    st.hess.setIdentity();
    st.grad.setZero();
    st.DC.setZero();
    st.m.setZero();

    const int nrow = static_cast<int>(st.DC.rows());
    const int lo = lay.lo_size();
    for (int i = 0; i < lo; ++i)
    {
        st.d(i) = -1.0;
    }
    for (int r = 0; r < nrow; ++r)
    {
        st.d(lay.side_hi(r)) =
            (lay.group_of(r) == detail::g_ineq) ? 0.0 : -1.0;
    }
    for (int j = lo + nrow; j < static_cast<int>(st.d.size()); ++j)
    {
        st.d(j) = 0.0;
    }
    for (int i = 0; i < static_cast<int>(st.d.size()); ++i)
    {
        st.d_mask(i) = (i % 5 == 4) ? 0.0 : 1.0;
    }
}

template <class St>
void fill_dyn(St& st, int nx)
{
    st.BA.setZero();
    for (int i = 0; i < nx; ++i)
    {
        st.BA(i, i) = 1.0;
    }
    st.b.setZero();
}

template <class P, int NH>
void fill_qp_init(Qp<P, NH>& qp, int N)
{
    using D = QpDim<P>;
    fill_base(qp.first, D::lay_first);
    fill_dyn(qp.first, D::nx);
    for (int k = 1; k < N; ++k)
    {
        fill_base(qp.path[k - 1], D::lay_path);
        fill_dyn(qp.path[k - 1], D::nx);
    }
    fill_base(qp.term, D::lay_term);
}

template <class V>
bool all_interior(const V& v)
{
    for (int i = 0; i < static_cast<int>(v.size()); ++i)
    {
        if (!(v(i) > 0.0) || !std::isfinite(v(i)))
        {
            return false;
        }
    }
    return true;
}

template <class P, int NH>
void check_init(const char* name, int N)
{
    const std::string p = name;

    Qp<P, NH> qp(N);
    QpSol<P, NH> sol(N);
    fill_qp_init(qp, N);

    HpipmQpSolver<P, NH> solver;
    solver.init_point(qp, sol);

    // interior: t > 0 and lam > 0 (finite) on every side
    check(all_interior(sol.t_first), p + ": t_first interior");
    check(all_interior(sol.lam_first), p + ": lam_first interior");
    check(all_interior(sol.t_term), p + ": t_term interior");
    check(all_interior(sol.lam_term), p + ": lam_term interior");
    for (int k = 1; k < N; ++k)
    {
        check(all_interior(sol.t_path[k - 1]),
              p + ": t_path[" + std::to_string(k - 1) + "] interior");
        check(all_interior(sol.lam_path[k - 1]),
              p + ": lam_path[" + std::to_string(k - 1) + "] interior");
    }

    // decision variables finite (bound repair must not produce NaN/Inf)
    check(sol.ux_first.allFinite(), p + ": ux_first finite");
    check(sol.ux_term.allFinite(), p + ": ux_term finite");
    for (int k = 1; k < N; ++k)
    {
        check(sol.ux_path[k - 1].allFinite(),
              p + ": ux_path[" + std::to_string(k - 1) + "] finite");
    }

    // lam_i = mu0 / t_i  =>  res_m_i = lam_i*t_i - m_i = mu0 on active sides
    //  =>  res_mu == mu0 (masked sides excluded via d_mask)
    QpRes<P, NH> res(N);
    solver.compute_residuals(qp, sol, res);
    check_scalar(res.res_mu, solver.options().mu0, 1e-12, p + ": res_mu == mu0");

    // sanity: the init must be strictly interior in t (t >= thr0 = 0.1)
    auto t_min = [](const auto& t)
    {
        double m = t(0);
        for (int i = 1; i < static_cast<int>(t.size()); ++i)
        {
            m = std::min(m, t(i));
        }
        return m;
    };
    check(t_min(sol.t_first) >= 1e-1 - 1e-12, p + ": t_first >= thr0");
    check(t_min(sol.t_term) >= 1e-1 - 1e-12, p + ": t_term >= thr0");
    for (int k = 1; k < N; ++k)
    {
        check(t_min(sol.t_path[k - 1]) >= 1e-1 - 1e-12,
              p + ": t_path[" + std::to_string(k - 1) + "] >= thr0");
    }
}

}  // namespace

int run_init_1c_tests()
{
    failures = 0;
    check_init<DoubleIntegrator, Eigen::Dynamic>("DI dyn N=1", 1);
    check_init<DoubleIntegrator, Eigen::Dynamic>("DI dyn N=2", 2);
    check_init<MassSpring, Eigen::Dynamic>("MS dyn N=1", 1);
    check_init<MassSpring, Eigen::Dynamic>("MS dyn N=2", 2);
    check_init<MassSpring, 2>("MS NH=2 N=2", 2);

    if (failures == 0)
    {
        std::printf("All init_point (1c) checks passed.\n");
    }
    return failures;
}

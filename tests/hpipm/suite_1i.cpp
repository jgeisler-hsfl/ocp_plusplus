// Phase 1 sub-step 1i: the HPIPM QP solver test suite.
//
// Covers the five cases from the worklog:
//   (1) staged LQR-like QP, hard constraints only, double-integrator-like
//       data  -> the primal solution / dynamics multipliers / objective are
//       cross-checked against an independent dense full-KKT solve;
//   (2) soft rows + slacks -> the optimum satisfies the weighted-slack KKT
//       (w s = lambda on the relaxed side) and matches the dense oracle;
//   (3) fixed x_0 pin rows -> x_0 is exactly the pinned value;
//   (4) terminal ineq / eq / lin + terminal box (custom problem type);
//   (5) edge cases -> the correct Status codes (kMinStep, kQpFailure,
//       kMaxIterations).
//
// The dense oracle solves the equality-constrained KKT of the primal QP
// (dynamics + pin rows + any explicitly listed active sides) with a
// FullPivLU factorisation, independently of the two-pass Riccati solve the
// solver uses. A case is valid only when every side not listed as active is
// strictly interior at the solution (verified explicitly).

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

void check_close(const VecD& a, const VecD& b, double tol,
                 const std::string& msg)
{
    if (a.size() == 0 && b.size() == 0)
    {
        return;
    }
    const double err = (a - b).cwiseAbs().maxCoeff();
    if (err > tol)
    {
        std::fprintf(stderr, "FAIL: %s (err = %.3e)\n", msg.c_str(), err);
        ++failures;
    }
}

void check_close_scalar(double a, double b, double tol, const std::string& msg)
{
    if (std::fabs(a - b) > tol)
    {
        std::fprintf(stderr, "FAIL: %s (%.6e vs %.6e)\n", msg.c_str(), a, b);
        ++failures;
    }
}

void check_le(double a, double b, const std::string& msg)
{
    if (!(a <= b))
    {
        std::fprintf(stderr, "FAIL: %s (%.3e > %.3e)\n", msg.c_str(), a, b);
        ++failures;
    }
}

struct Rng
{
    std::mt19937_64 gen{424242};
    double uniform()
    {
        return std::uniform_real_distribution<double>(-1.0, 1.0)(gen);
    }
};

// ---------------------------------------------------------------------
// Custom problem type for the terminal-constraint case: a 2-state /
// 1-control system with terminal ineq / eq / lin rows and a terminal box.
// No soft rows. The QP data is filled by hand in the test; only the
// compile-time dimensions / index sets are used.
// ---------------------------------------------------------------------

struct TermConstrDims
{
    static constexpr int nx = 2;
    static constexpr int nu = 1;
    static constexpr int ng = 0;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 1;
    static constexpr int ne_t = 1;
    static constexpr int nl_t = 1;
    static constexpr bool fixed_initial_state = true;
    static constexpr bool has_dynamics_hess_prod = false;
    static constexpr bool has_constr_hess_prod = true;
    static constexpr std::array<int, 2> state_box_idx = {0, 1};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 2> terminal_state_box_idx = {0, 1};
    static constexpr std::array<int, 0> ineq_soft_idx = {};
    static constexpr std::array<int, 0> eq_soft_idx = {};
    static constexpr std::array<int, 0> lin_soft_idx = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {};
    static constexpr std::array<int, 0> state_box_soft_idx = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

struct TermConstr : ocp::Problem<TermConstrDims>
{
};

// ---------------------------------------------------------------------
// SoftBox (phase 3a): 1 state / 1 control, pinned x_0, one soft control
// box (first stage) and one soft terminal state box. The inactive side of
// each box is masked (d_mask = 0, mirroring an -inf bound). No ineq / eq /
// lin rows, so the only slacks are the four box slacks. The QP data is
// filled by hand in case3_softbox; only the compile-time dimensions /
// index sets are used.
// ---------------------------------------------------------------------

struct SoftBoxDims
{
    static constexpr int nx = 1;
    static constexpr int nu = 1;
    static constexpr int ng = 0;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 0;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = true;
    static constexpr bool has_dynamics_hess_prod = false;
    static constexpr bool has_constr_hess_prod = false;
    static constexpr std::array<int, 1> state_box_idx = {0};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 1> terminal_state_box_idx = {0};
    static constexpr std::array<int, 0> ineq_soft_idx = {};
    static constexpr std::array<int, 0> eq_soft_idx = {};
    static constexpr std::array<int, 0> lin_soft_idx = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {};
    static constexpr std::array<int, 1> state_box_soft_idx = {0};
    static constexpr std::array<int, 1> control_box_soft_idx = {0};
    static constexpr std::array<int, 1> terminal_state_box_soft_idx = {0};
};

struct SoftBox : ocp::Problem<SoftBoxDims>
{
};

// ---------------------------------------------------------------------
// Per-stage accessors into a Qp (Dynamic-extent mirrors for the oracle).
// ---------------------------------------------------------------------

template <class P, int NH>
struct Stg
{
    using D = QpDim<P>;
    const Qp<P, NH>& qp;
    int N;

    int nx() const { return D::nx; }
    int nu() const { return D::nu; }
    bool fixed() const { return P::fixed_initial_state; }

    int nvar(int k) const
    {
        return k == 0 ? D::nvar_first
                      : (k == N ? D::nvar_term : D::nvar_path);
    }
    int nslack(int k) const
    {
        return k == 0 ? D::nslack_first
                      : (k == N ? D::nslack_term : D::nslack_path);
    }
    int nux(int k) const { return nvar(k) - nslack(k); }
    int xoff(int k) const { return (k == N) ? 0 : nu(); }
    int zoff(int k) const
    {
        int s = 0;
        for (int j = 0; j < k; ++j)
        {
            s += nvar(j);
        }
        return s;
    }
    const detail::QpLayout& lay(int k) const
    {
        return k == 0 ? D::lay_first
                      : (k == N ? D::lay_term : D::lay_path);
    }
    MatD H(int k) const
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
    VecD g(int k) const
    {
        if (k == 0)
        {
            return qp.first.grad;
        }
        if (k == N)
        {
            return qp.term.grad;
        }
        return qp.path[k - 1].grad;
    }
    MatD BA(int k) const
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
    VecD b(int k) const
    {
        if (k == 0)
        {
            return qp.first.b;
        }
        return qp.path[k - 1].b;
    }
    VecD d(int k) const
    {
        if (k == 0)
        {
            return qp.first.d;
        }
        if (k == N)
        {
            return qp.term.d;
        }
        return qp.path[k - 1].d;
    }
    MatD DC(int k) const
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
};

// ---------------------------------------------------------------------
// Dense equality-KKT oracle.
//
// Solves the primal QP's KKT system in (z_0..z_N, pi_0..pi_{N-1}) plus a
// free multiplier per pin row and per explicitly-listed active side, using
// FullPivLU. Valid when every side not listed as active is interior at the
// solution.
// ---------------------------------------------------------------------

struct ActiveRow
{
    int k;    // stage
    int r;    // row index
    int which;  // 0 = lo side, 1 = hi side
};

struct ActiveSlack
{
    int k;  // stage
    int j;  // slack index within the stage
};

template <class P, int NH>
struct OracleOut
{
    std::vector<VecD> z;  // N + 1, per stage (natural (u;x;s) order)
    std::vector<VecD> pi;  // N
    VecD act_row_mult;  // one per ActiveRow (in input order)
    VecD act_slack_mult;  // one per ActiveSlack
    double obj = 0.0;
    bool ok = false;
};

template <class P, int NH>
OracleOut<P, NH> dense_oracle(const Stg<P, NH>& sg, const VecD& x0,
                              const std::vector<ActiveRow>& act_rows,
                               const std::vector<ActiveSlack>& act_slacks)
{
    const int N = sg.N;
    const int nx = sg.nx();
    const int nu = sg.nu();
    const bool fixed = sg.fixed();

    int zdim = 0;
    for (int k = 0; k <= N; ++k)
    {
        zdim += sg.nvar(k);
    }
    const int npi = N * nx;
    const int npin = fixed ? nx : 0;
    const int nact = static_cast<int>(act_rows.size() + act_slacks.size());
    const int dim = zdim + npi + npin + nact;

    MatD K = MatD::Zero(dim, dim);
    VecD r = VecD::Zero(dim);

    // stationarity rows (one per variable of each stage)
    for (int k = 0; k <= N; ++k)
    {
        const int nv = sg.nvar(k);
        const int nux = sg.nux(k);
        const int xo = sg.xoff(k);
        const MatD Hk = sg.H(k);
        const VecD gk = sg.g(k);
        const MatD BAk = sg.BA(k);
        for (int a = 0; a < nv; ++a)
        {
            const int row = sg.zoff(k) + a;
            for (int b = 0; b < nv; ++b)
            {
                K(row, sg.zoff(k) + b) += Hk(a, b);
            }
            r(row) = -gk(a);
            if (k < N && a < nux)
            {
                for (int j = 0; j < nx; ++j)
                {
                    K(row, zdim + k * nx + j) += BAk(j, a);
                }
            }
            if (k >= 1 && a >= xo && a < xo + nx)
            {
                const int i = a - xo;
                K(row, zdim + (k - 1) * nx + i) -= 1.0;
            }
            if (k == 0 && fixed && a >= xo && a < xo + nx)
            {
                const int i = a - xo;
                K(row, zdim + npi + i) += 1.0;
            }
        }
    }

    // dynamics rows
    for (int k = 0; k < N; ++k)
    {
        const MatD BAk = sg.BA(k);
        const int nux = sg.nux(k);
        const int xoff_next = (k + 1 == N) ? 0 : nu;
        for (int i = 0; i < nx; ++i)
        {
            const int row = zdim + k * nx + i;
            for (int a = 0; a < nux; ++a)
            {
                K(row, sg.zoff(k) + a) += BAk(i, a);
            }
            K(row, sg.zoff(k + 1) + xoff_next + i) -= 1.0;
            r(row) = -sg.b(k)(i);
        }
    }

    // pin rows
    if (fixed)
    {
        const int xoff0 = (N == 0) ? 0 : nu;
        for (int i = 0; i < nx; ++i)
        {
            const int row = zdim + npi + i;
            K(row, sg.zoff(0) + xoff0 + i) = 1.0;
            r(row) = x0(i);
        }
    }

    // active-side rows / multipliers
    const int mbase = zdim + npi + npin;
    for (int a = 0; a < static_cast<int>(act_rows.size()); ++a)
    {
        const auto& A = act_rows[a];
        const int k = A.k;
        const int rr = A.r;
        const int nv = sg.nvar(k);
        const int nux = sg.nux(k);
        const detail::QpLayout& lay = sg.lay(k);
        const MatD DCk = sg.DC(k);
        const int side = (A.which == 0) ? lay.side_lo(rr) : lay.side_hi(rr);
        const double dval = sg.d(k)(side);
        const double target = (A.which == 0) ? dval : -dval;
        const double sign = (A.which == 0) ? -1.0 : 1.0;
        const int mcol = mbase + a;
        // equality row: DC row * z = target
        for (int c = 0; c < nv; ++c)
        {
            K(mcol, sg.zoff(k) + c) += DCk(rr, c);
        }
        r(mcol) = target;
        // stationarity coupling on the (u;x) part
        for (int c = 0; c < nux; ++c)
        {
            K(sg.zoff(k) + c, mcol) += sign * DCk(rr, c);
        }
        // stationarity coupling on the relaxed slack column (if soft)
        int slack_col = -1;
        for (int c = nux; c < nv; ++c)
        {
            if ((A.which == 0) && std::fabs(DCk(rr, c) - 1.0) < 1e-12)
            {
                slack_col = c;
            }
            if ((A.which == 1) && std::fabs(DCk(rr, c) + 1.0) < 1e-12)
            {
                slack_col = c;
            }
        }
        if (slack_col >= 0)
        {
            K(sg.zoff(k) + slack_col, mcol) -= 1.0;
        }
    }
    for (int a = 0; a < static_cast<int>(act_slacks.size()); ++a)
    {
        const auto& A = act_slacks[a];
        const int k = A.k;
        const int nux = sg.nux(k);
        const int col = nux + A.j;
        const int mcol = mbase + static_cast<int>(act_rows.size()) + a;
        K(mcol, sg.zoff(k) + col) = 1.0;  // equality: s_j = d_slack = 0
        r(mcol) = sg.d(k)(sg.lay(k).lo_size() + sg.lay(k).nrow() + A.j);
        K(sg.zoff(k) + col, mcol) -= 1.0;  // stationarity: -mu on s_j
    }

    Eigen::FullPivLU<MatD> lu(K);
    if (!lu.isInvertible())
    {
        return {};
    }
    const VecD x = lu.solve(r);
    const double resid = (K * x - r).cwiseAbs().maxCoeff();
    if (resid > 1e-7)
    {
        std::fprintf(stderr,
                     "oracle: dense KKT residual too large %.3e\n", resid);
        return {};
    }

    OracleOut<P, NH> out;
    out.z.resize(N + 1);
    for (int k = 0; k <= N; ++k)
    {
        out.z[k] = x.segment(sg.zoff(k), sg.nvar(k));
    }
    out.pi.resize(N);
    for (int k = 0; k < N; ++k)
    {
        out.pi[k] = x.segment(zdim + k * nx, nx);
    }
    out.act_row_mult = x.segment(mbase, static_cast<int>(act_rows.size()));
    out.act_slack_mult =
        x.segment(mbase + static_cast<int>(act_rows.size()),
                  static_cast<int>(act_slacks.size()));
    double obj = 0.0;
    for (int k = 0; k <= N; ++k)
    {
        obj += 0.5 * out.z[k].dot(sg.H(k) * out.z[k])
               + out.z[k].dot(sg.g(k));
    }
    out.obj = obj;
    out.ok = true;
    return out;
}

// ||res_m - tau_min * d_mask||_inf over all stages (the exit-test quantity).
template <class P, int NH>
double res_m_tau_inf(const Qp<P, NH>& qp, const QpRes<P, NH>& res, double tau)
{
    double m = 0.0;
    auto acc = [&](const auto& rm, const auto& dm)
    {
        for (int i = 0; i < static_cast<int>(rm.size()); ++i)
        {
            m = std::max(m, std::fabs(rm(i) - tau * dm(i)));
        }
    };
    acc(res.res_m_first, qp.first.d_mask);
    for (int k = 1; k < qp.N; ++k)
    {
        acc(res.res_m_path[k - 1], qp.path[k - 1].d_mask);
    }
    acc(res.res_m_term, qp.term.d_mask);
    return m;
}

// Recompute the KKT residuals and assert every norm is within the option
// tolerances (including the tau-shifted complementarity).
template <class P, int NH>
void check_kkt_residuals(HpipmQpSolver<P, NH>& solver, const Qp<P, NH>& qp,
                         QpSol<P, NH>& sol, const std::string& name)
{
    const auto& opt = solver.options();
    QpRes<P, NH> res(qp.N);
    solver.compute_residuals(qp, sol, res);
    check_le(res.res_g_max, opt.res_g_max, name + ": res_g_max");
    check_le(res.res_b_max, opt.res_b_max, name + ": res_b_max");
    check_le(res.res_d_max, opt.res_d_max, name + ": res_d_max");
    check_le(res_m_tau_inf(qp, res, opt.tau_min), opt.res_m_max,
             name + ": res_m_tau");
    check_le(res.dual_gap, opt.dual_gap_max, name + ": dual_gap");
}

// ---------------------------------------------------------------------
// Case 1 fill: LQR-like data, strongly-PD (u;x) Hessian, contractive
// dynamics, wide (non-binding) box / ineq / lin bounds, m = 0, all sides
// active. The pin rows pin x_0 = x0.
// ---------------------------------------------------------------------

template <class P, int NH>
void fill_lqr(Qp<P, NH>& qp, int N, const VecD& x0)
{
    using D = QpDim<P>;
    Rng rng;
    const double BIG = 1e3;

    auto fill_ux_stage = [&](auto& st, int nslack)
    {
        const int nux = D::nu + D::nx;
        MatD R(nux, nux);
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                R(i, j) = 0.3 * rng.uniform();
            }
        }
        const MatD Hux = R.transpose() * R;
        st.hess.setZero();
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                st.hess(i, j) = Hux(i, j) + (i == j ? 1.0 : 0.0);
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

    // first stage
    fill_ux_stage(qp.first, D::nslack_first);
    {
        const auto& lay = D::lay_first;
        for (int r = 0; r < D::nrow_first; ++r)
        {
            const int g = lay.group_of(r);
            const int sl = lay.side_lo(r);
            const int sh = lay.side_hi(r);
            if (g == detail::g_pin)
            {
                const int var = D::idx_x0[r];
                qp.first.DC(r, var) = 1.0;
                if (sl >= 0)
                {
                    qp.first.d(sl) = x0(r);
                }
                if (sh >= 0)
                {
                    qp.first.d(sh) = -x0(r);
                }
            }
            else if (g == detail::g_bx || g == detail::g_bu)
            {
                const int nbx = D::nbx_first;
                const int var = (g == detail::g_bx)
                                    ? D::idxb_first[r]
                                    : D::idxb_first[nbx + (r - lay.row_off(detail::g_bu))];
                qp.first.DC(r, var) = 1.0;
                if (sl >= 0)
                {
                    qp.first.d(sl) = -BIG;
                }
                if (sh >= 0)
                {
                    qp.first.d(sh) = -BIG;
                }
            }
            else
            {
                // ineq / eq / lin: fixed Jacobian, wide (non-binding) bounds
                for (int j = 0; j < D::nu + D::nx; ++j)
                {
                    qp.first.DC(r, j) = 0.1 * rng.uniform();
                }
                if (sl >= 0)
                {
                    qp.first.d(sl) = -BIG;
                }
                if (sh >= 0)
                {
                    qp.first.d(sh) = -BIG;
                }
            }
            if (D::idxs_lo_first[r] >= 0)
            {
                qp.first.DC(r, D::idxs_lo_first[r]) = 1.0;
            }
            if (D::idxs_hi_first[r] >= 0)
            {
                qp.first.DC(r, D::idxs_hi_first[r]) = -1.0;
            }
        }
        qp.first.d_mask.setConstant(1.0);
        qp.first.m.setZero();
    }

    // path stages
    for (int k = 1; k < N; ++k)
    {
        fill_ux_stage(qp.path[k - 1], D::nslack_path);
        const auto& lay = D::lay_path;
        for (int r = 0; r < D::nrow_path; ++r)
        {
            const int g = lay.group_of(r);
            const int sl = lay.side_lo(r);
            const int sh = lay.side_hi(r);
            if (g == detail::g_bx || g == detail::g_bu)
            {
                const int var = (g == detail::g_bx)
                                    ? D::idxb_path[r]
                                    : D::idxb_path[D::nbx + (r - lay.row_off(detail::g_bu))];
                qp.path[k - 1].DC(r, var) = 1.0;
                if (sl >= 0)
                {
                    qp.path[k - 1].d(sl) = -BIG;
                }
                if (sh >= 0)
                {
                    qp.path[k - 1].d(sh) = -BIG;
                }
            }
            else
            {
                for (int j = 0; j < D::nu + D::nx; ++j)
                {
                    qp.path[k - 1].DC(r, j) = 0.1 * rng.uniform();
                }
                if (sl >= 0)
                {
                    qp.path[k - 1].d(sl) = -BIG;
                }
                if (sh >= 0)
                {
                    qp.path[k - 1].d(sh) = -BIG;
                }
            }
            if (D::idxs_lo_path[r] >= 0)
            {
                qp.path[k - 1].DC(r, D::idxs_lo_path[r]) = 1.0;
            }
            if (D::idxs_hi_path[r] >= 0)
            {
                qp.path[k - 1].DC(r, D::idxs_hi_path[r]) = -1.0;
            }
        }
        qp.path[k - 1].d_mask.setConstant(1.0);
        qp.path[k - 1].m.setZero();
    }

    // terminal stage (x;s)
    {
        const int nux = D::nx;
        const int nslack = D::nslack_term;
        MatD R(nux, nux);
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                R(i, j) = 0.3 * rng.uniform();
            }
        }
        const MatD Hux = R.transpose() * R;
        qp.term.hess.setZero();
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                qp.term.hess(i, j) = Hux(i, j) + (i == j ? 1.0 : 0.0);
            }
        }
        for (int i = 0; i < nslack; ++i)
        {
            qp.term.hess(nux + i, nux + i) = 1.0;
        }
        for (int i = 0; i < nux; ++i)
        {
            qp.term.grad(i) = 0.2 * rng.uniform();
        }
        const auto& lay = D::lay_term;
        for (int r = 0; r < D::nrow_term; ++r)
        {
            const int g = lay.group_of(r);
            const int sl = lay.side_lo(r);
            const int sh = lay.side_hi(r);
            if (g == detail::g_bx)
            {
                qp.term.DC(r, D::idxb_term[r]) = 1.0;
                if (sl >= 0)
                {
                    qp.term.d(sl) = -BIG;
                }
                if (sh >= 0)
                {
                    qp.term.d(sh) = -BIG;
                }
            }
            else
            {
                for (int j = 0; j < D::nx; ++j)
                {
                    qp.term.DC(r, j) = 0.1 * rng.uniform();
                }
                if (sl >= 0)
                {
                    qp.term.d(sl) = -BIG;
                }
                if (sh >= 0)
                {
                    qp.term.d(sh) = -BIG;
                }
            }
            if (D::idxs_lo_term[r] >= 0)
            {
                qp.term.DC(r, D::idxs_lo_term[r]) = 1.0;
            }
            if (D::idxs_hi_term[r] >= 0)
            {
                qp.term.DC(r, D::idxs_hi_term[r]) = -1.0;
            }
        }
        qp.term.d_mask.setConstant(1.0);
        qp.term.m.setZero();
    }
}

// ---------------------------------------------------------------------
// Case 1: LQR-like QP cross-checked against the dense oracle.
// ---------------------------------------------------------------------

template <class P, int NH = Eigen::Dynamic>
void case1_lqr(const char* name, int N, const VecD& x0)
{
    using D = QpDim<P>;
    const std::string p = name;
    Qp<P, NH> qp(N);
    QpSol<P, NH> sol(N);
    fill_lqr(qp, N, x0);

    HpipmQpSolver<P, NH> solver;
    const Status st = solver.solve(qp, sol);
    check(st == Status::kSolved,
          p + ": status kSolved (got " +
              std::to_string(static_cast<int>(st)) + ")");
    if (st != Status::kSolved)
    {
        return;
    }

    // independent dense oracle (no active side: box / ineq / lin are wide)
    Stg<P, NH> sg{qp, N};
    OracleOut<P, NH> ref = dense_oracle(sg, x0, {}, {});
    check(ref.ok, p + ": dense oracle factorised");
    if (!ref.ok)
    {
        return;
    }
    const double tol_z = 1e-3;
    for (int k = 0; k <= N; ++k)
    {
        VecD got;
        if (k == 0)
        {
            got = sol.ux_first;
        }
        else if (k == N)
        {
            got = sol.ux_term;
        }
        else
        {
            got = sol.ux_path[k - 1];
        }
        check_close(got, ref.z[k], tol_z, p + ": z_k " + std::to_string(k));
    }
    for (int k = 0; k < N; ++k)
    {
        check_close(sol.pi[k], ref.pi[k], tol_z,
                    p + ": pi_k " + std::to_string(k));
    }
    QpRes<P, NH> res(N);
    solver.compute_residuals(qp, sol, res);
    check_close_scalar(res.obj, ref.obj, 1e-6, p + ": objective");

    // pin: x_0 exactly equals the pinned value
    {
        const int xoff = QpDim<P>::nu;
        for (int i = 0; i < QpDim<P>::nx; ++i)
        {
            check_close_scalar(sol.ux_first(xoff + i), x0(i), 1e-8,
                               p + ": x_0 pin comp " + std::to_string(i));
        }
    }

    // every non-pin side is interior (multiplier ~ 0) -> empty active set.
    // The hi-side block is 1:1 with rows in group order; skip the pin group
    // (its hi sides carry the pinned-state costates, legitimately nonzero).
    auto nonbinding = [&](const detail::QpLayout& lay, const auto& dm,
                          const auto& lam)
    {
        const int nrow = lay.nrow();
        const int lo = lay.lo_size();
        for (int i = lo; i < lo + nrow; ++i)
        {
            if (dm(i) <= 0.5)
            {
                continue;
            }
            const int r = i - lo;
            if (lay.group_of(r) == detail::g_pin)
            {
                continue;
            }
            check(std::fabs(lam(i)) < 1e-3,
                  p + ": side " + std::to_string(i) + " non-binding");
        }
    };
    nonbinding(D::lay_first, qp.first.d_mask, sol.lam_first);

    check_kkt_residuals(solver, qp, sol, p);
    std::printf("  %s: N=%d obj=%.6f res_g=%.1e res_b=%.1e\n", name, N,
                res.obj, res.res_g_max, res.res_b_max);
}

// ---------------------------------------------------------------------
// Case 2 fill: a soft ineq is violated at the optimum so its slack is
// positive. The ineq row (g <= 0) has Jacobian jg over (u;x) and a
// non-negative weight w on its slack.
// ---------------------------------------------------------------------

template <class P, int NH>
void fill_soft(Qp<P, NH>& qp, int N, const VecD& x0, const VecD& jg, double w)
{
    using D = QpDim<P>;
    Rng rng;
    const double BIG = 1e3;

    // Make the ineq clearly violated: the row value at the "natural"
    // (zero-cost-gradient) point is positive. Set the ineq offset so that
    // g(x*,u*) > 0 at the unrelaxed optimum.
    auto fill_ux_stage = [&](auto& st, int nslack, bool is_first)
    {
        const int nux = D::nu + D::nx;
        MatD R(nux, nux);
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                R(i, j) = 0.3 * rng.uniform();
            }
        }
        const MatD Hux = R.transpose() * R;
        st.hess.setZero();
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                st.hess(i, j) = Hux(i, j) + (i == j ? 1.0 : 0.0);
            }
        }
        for (int i = 0; i < nslack; ++i)
        {
            st.hess(nux + i, nux + i) = 1.0;
        }
        for (int i = 0; i < nslack; ++i)
        {
            st.grad(nux + i) = 0.0;
        }
        for (int i = 0; i < nux; ++i)
        {
            st.grad(i) = 0.2 * rng.uniform();
        }
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
        (void)is_first;
    };

    auto fill_dc_ineq = [&](auto& st, bool is_first)
    {
        const auto& lay = is_first ? D::lay_first : D::lay_path;
        const int nrow = is_first ? D::nrow_first : D::nrow_path;
        const int nvar = is_first ? D::nvar_first : D::nvar_path;
        for (int r = 0; r < nrow; ++r)
        {
            const int g = lay.group_of(r);
            const int sl = lay.side_lo(r);
            const int sh = lay.side_hi(r);
            if (g == detail::g_pin)
            {
                const int var = D::idx_x0[r];
                st.DC(r, var) = 1.0;
                if (sl >= 0)
                {
                    st.d(sl) = x0(r);
                }
                if (sh >= 0)
                {
                    st.d(sh) = -x0(r);
                }
            }
            else if (g == detail::g_bx || g == detail::g_bu)
            {
                const int var = (is_first)
                                    ? (g == detail::g_bx
                                           ? D::idxb_first[r]
                                           : D::idxb_first[D::nbx_first + (r - lay.row_off(detail::g_bu))])
                                    : (g == detail::g_bx
                                           ? D::idxb_path[r]
                                           : D::idxb_path[D::nbx + (r - lay.row_off(detail::g_bu))]);
                st.DC(r, var) = 1.0;
                if (sl >= 0)
                {
                    st.d(sl) = -BIG;
                }
                if (sh >= 0)
                {
                    st.d(sh) = -BIG;
                }
            }
            else if (g == detail::g_ineq)
            {
                for (int j = 0; j < D::nu + D::nx; ++j)
                {
                    st.DC(r, j) = jg(j);
                }
                // ineq is one-sided (hi only), soft -> slack column
                const int slo = (is_first) ? D::idxs_hi_first[r] : D::idxs_hi_path[r];
                if (slo >= 0)
                {
                    st.DC(r, slo) = -1.0;
                    st.hess(slo, slo) = w;
                }
                if (sh >= 0)
                {
                    st.d(sh) = 0.0;  // g <= 0
                }
            }
            else
            {
                for (int j = 0; j < D::nu + D::nx; ++j)
                {
                    st.DC(r, j) = 0.1 * rng.uniform();
                }
                if (sl >= 0)
                {
                    st.d(sl) = -BIG;
                }
                if (sh >= 0)
                {
                    st.d(sh) = -BIG;
                }
            }
        }
        st.d_mask.setConstant(1.0);
        st.m.setZero();
        (void)nvar;
    };

    // first
    fill_ux_stage(qp.first, D::nslack_first, true);
    fill_dc_ineq(qp.first, true);
    // path
    for (int k = 1; k < N; ++k)
    {
        fill_ux_stage(qp.path[k - 1], D::nslack_path, false);
        fill_dc_ineq(qp.path[k - 1], false);
    }
    // terminal (x;s)
    {
        const int nux = D::nx;
        const int nslack = D::nslack_term;
        MatD R(nux, nux);
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                R(i, j) = 0.3 * rng.uniform();
            }
        }
        const MatD Hux = R.transpose() * R;
        qp.term.hess.setZero();
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                qp.term.hess(i, j) = Hux(i, j) + (i == j ? 1.0 : 0.0);
            }
        }
        for (int i = 0; i < nslack; ++i)
        {
            qp.term.hess(nux + i, nux + i) = 1.0;
        }
        for (int i = 0; i < nux; ++i)
        {
            qp.term.grad(i) = 0.2 * rng.uniform();
        }
        const auto& lay = D::lay_term;
        for (int r = 0; r < D::nrow_term; ++r)
        {
            const int g = lay.group_of(r);
            const int sl = lay.side_lo(r);
            const int sh = lay.side_hi(r);
            if (g == detail::g_bx)
            {
                qp.term.DC(r, D::idxb_term[r]) = 1.0;
                if (sl >= 0)
                {
                    qp.term.d(sl) = -BIG;
                }
                if (sh >= 0)
                {
                    qp.term.d(sh) = -BIG;
                }
            }
            else
            {
                for (int j = 0; j < D::nx; ++j)
                {
                    qp.term.DC(r, j) = 0.1 * rng.uniform();
                }
                if (sl >= 0)
                {
                    qp.term.d(sl) = -BIG;
                }
                if (sh >= 0)
                {
                    qp.term.d(sh) = -BIG;
                }
            }
            if (D::idxs_lo_term[r] >= 0)
            {
                qp.term.DC(r, D::idxs_lo_term[r]) = 1.0;
            }
            if (D::idxs_hi_term[r] >= 0)
            {
                qp.term.DC(r, D::idxs_hi_term[r]) = -1.0;
            }
        }
        qp.term.d_mask.setConstant(1.0);
        qp.term.m.setZero();
    }
}

// ---------------------------------------------------------------------
// Case 2: soft ineq with a positive slack. Verify the weighted-slack KKT
// (w * s == lambda on the relaxed side) and, for N = 1, cross-check the
// full solution against the dense oracle with the ineq listed active.
// ---------------------------------------------------------------------

template <class P, int NH = Eigen::Dynamic>
void case2_soft(const char* name, int N, const VecD& x0, const VecD& jg,
                double w)
{
    const std::string p = name;
    Qp<P, NH> qp(N);
    QpSol<P, NH> sol(N);
    fill_soft(qp, N, x0, jg, w);

    HpipmQpSolver<P, NH> solver;
    const Status st = solver.solve(qp, sol);
    check(st == Status::kSolved,
          p + ": status kSolved (got " +
              std::to_string(static_cast<int>(st)) + ")");
    if (st != Status::kSolved)
    {
        return;
    }

    using D = QpDim<P>;
    // locate the ineq row (group g_ineq, first stage) and its hi side / slack
    const auto& lay = D::lay_first;
    const int ineq_r = lay.row_off(detail::g_ineq);
    const int shi = lay.side_hi(ineq_r);
    const int slack_col = D::idxs_hi_first[ineq_r];
    check(slack_col >= 0, p + ": ineq is soft (has a slack column)");
    const int slack_side = lay.lo_size() + lay.nrow() + 0;  // ineq slack

    // the ineq is violated at the optimum -> slack positive
    const double s = sol.ux_first(slack_col);
    check(s > 1e-3, p + ": ineq slack positive (s=" + std::to_string(s) + ")");

    // weighted-slack KKT: lambda_hi == w * s
    const double lam_hi = sol.lam_first(shi);
    const double lam_slack = sol.lam_first(slack_side);
    check_close_scalar(lam_hi, w * s, 1e-6, p + ": weighted-slack w*s == lam_hi");
    check(std::fabs(lam_slack) < 1e-4,
          p + ": slack-side multiplier ~ 0 (s > 0)");

    check_kkt_residuals(solver, qp, sol, p);

    if (N == 1)
    {
        // dense oracle with the ineq hi side active
        Stg<P, NH> sg{qp, N};
        const std::vector<ActiveRow> act = {{0, ineq_r, 1}};
        OracleOut<P, NH> ref = dense_oracle(sg, x0, act, {});
        check(ref.ok, p + ": dense oracle (active ineq) factorised");
        if (ref.ok)
        {
            check_close(sol.ux_first, ref.z[0], 1e-5, p + ": z_0 (soft)");
            check_close(sol.ux_term, ref.z[1], 1e-5, p + ": z_1 (soft)");
            check_close(sol.pi[0], ref.pi[0], 1e-5, p + ": pi_0 (soft)");
            check_close_scalar(ref.act_row_mult(0), lam_hi, 1e-6,
                               p + ": oracle lam_hi matches solver");
        }
    }

    QpRes<P, NH> res(N);
    solver.compute_residuals(qp, sol, res);
    std::printf("  %s: N=%d s=%.4f w*s=%.4f lam_hi=%.4f res_g=%.1e res_b=%.1e\n",
                name, N, s, w * s, lam_hi, res.res_g_max, res.res_b_max);
}

// ---------------------------------------------------------------------
// Case 3 fill (phase 3a): soft box rows carry the idxs_rev slack columns.
//
//   N = 1, x_0 = 0 pinned.  dynamics: x_1 = u_0 + x_0   (BA = [1 1], b = 0).
//   stage cost:  0.5 (u^2 + x_0^2) + g*u      (g = -12 pulls u up, or +12
//                                              pulls u down)
//   terminal:    0.5 x_1^2
//   box:         -1 <= u <= 1   (soft, w = 4)   [first stage]
//                -1 <= x_1 <= 1  (soft, w = 25)  [terminal]
//   inactive box sides are masked (d_mask = 0, d = -1e3), mirroring -inf.
//
// Because x_1 = u, both boxes see the same value, so the violated side is
// whichever the gradient pushes into. The weighted-slack KKT on the active
// side is lambda = w * s, and the dense oracle (active side + zeroed
// opposite slack) reproduces z / pi / obj.
// ---------------------------------------------------------------------

template <class P, int NH>
void fill_softbox(Qp<P, NH>& qp, bool lo_violated)
{
    using D = QpDim<P>;
    const double BIG = 1e3;
    const double g = lo_violated ? 12.0 : -12.0;

    // first stage: (u; x; s_bu_lo; s_bu_hi)
    {
        auto& st = qp.first;
        st.hess.setZero();
        st.hess(0, 0) = 1.0;
        st.hess(1, 1) = 1.0;
        st.hess(2, 2) = 4.0;  // s_bu_lo
        st.hess(3, 3) = 4.0;  // s_bu_hi
        st.grad(0) = g;
        st.grad(1) = 0.0;
        st.BA(0, 0) = 1.0;
        st.BA(0, 1) = 1.0;
        // rows: 0 = pin, 1 = bu
        st.DC(0, D::idx_x0[0]) = 1.0;
        st.DC(1, D::idxb_first[0]) = 1.0;
        st.DC(1, D::idxs_lo_first[1]) = 1.0;
        st.DC(1, D::idxs_hi_first[1]) = -1.0;
        // sides: 0=pin lo, 1=bu lo, 2=pin hi, 3=bu hi, 4=s_bu_lo, 5=s_bu_hi
        st.d(0) = 0.0;
        st.d_mask(0) = 1.0;
        st.d(2) = 0.0;
        st.d_mask(2) = 1.0;
        st.d(3) = -1.0;  // bu hi: -hi
        st.d_mask(3) = 1.0;
        st.d(4) = 0.0;
        st.d_mask(4) = 1.0;
        st.d(5) = 0.0;
        st.d_mask(5) = 1.0;
        if (lo_violated)
        {
            st.d(1) = -1.0;   // bu lo: lo - w_cur = -1
            st.d_mask(1) = 1.0;
        }
        else
        {
            st.d(1) = -BIG;   // bu lo masked (-inf analogue)
            st.d_mask(1) = 0.0;
        }
    }

    // terminal stage: (x; s_bx_lo; s_bx_hi)
    {
        auto& st = qp.term;
        st.hess.setZero();
        st.hess(0, 0) = 1.0;
        st.hess(1, 1) = 25.0;  // s_bx_lo
        st.hess(2, 2) = 25.0;  // s_bx_hi
        // grad zero
        st.DC(0, D::idxb_term[0]) = 1.0;
        st.DC(0, D::idxs_lo_term[0]) = 1.0;
        st.DC(0, D::idxs_hi_term[0]) = -1.0;
        // sides: 0=bx lo, 1=bx hi, 2=s_lo, 3=s_hi
        st.d(1) = -1.0;   // bx hi: -hi
        st.d_mask(1) = 1.0;
        st.d(2) = 0.0;
        st.d_mask(2) = 1.0;
        st.d(3) = 0.0;
        st.d_mask(3) = 1.0;
        if (lo_violated)
        {
            st.d(0) = -1.0;   // bx lo: lo - w_cur = -1
            st.d_mask(0) = 1.0;
        }
        else
        {
            st.d(0) = -BIG;   // bx lo masked (-inf analogue)
            st.d_mask(0) = 0.0;
        }
    }
}

// ---------------------------------------------------------------------
// Case 3: soft box rows (phase 3a). Verifies the weighted-slack KKT on the
// violated side (lambda = w * s, s >= 0) and cross-checks z / pi / obj /
// active multipliers against the dense full-KKT oracle.
// ---------------------------------------------------------------------

template <class P, int NH = Eigen::Dynamic>
void case3_softbox(const char* name, bool lo_violated)
{
    const std::string p = name;
    Qp<P, NH> qp(1);
    QpSol<P, NH> sol(1);
    fill_softbox(qp, lo_violated);

    HpipmQpSolver<P, NH> solver;
    const Status st = solver.solve(qp, sol);
    check(st == Status::kSolved,
          p + ": status kSolved (got " +
              std::to_string(static_cast<int>(st)) + ")");
    if (st != Status::kSolved)
    {
        return;
    }

    using D = QpDim<P>;
    const auto& lay_f = D::lay_first;
    const auto& lay_t = D::lay_term;
    const int r_bu = lay_f.row_off(detail::g_bu);
    const int r_bx = lay_t.row_off(detail::g_bx);
    const int which = lo_violated ? 0 : 1;  // 0 = lo, 1 = hi

    const int s_lo_f = D::idxs_lo_first[r_bu];
    const int s_hi_f = D::idxs_hi_first[r_bu];
    const int s_lo_t = D::idxs_lo_term[r_bx];
    const int s_hi_t = D::idxs_hi_term[r_bx];
    const double s_f = (which == 0) ? sol.ux_first(s_lo_f)
                                    : sol.ux_first(s_hi_f);
    const double s_t = (which == 0) ? sol.ux_term(s_lo_t)
                                    : sol.ux_term(s_hi_t);
    check(s_f >= 0.0, p + ": box slack (bu) >= 0");
    check(s_t >= 0.0, p + ": box slack (term bx) >= 0");
    check(s_f > 1e-3, p + ": box slack (bu) positive (s=" +
                          std::to_string(s_f) + ")");
    check(s_t > 1e-3, p + ": box slack (term bx) positive (s=" +
                          std::to_string(s_t) + ")");

    // weighted-slack KKT: lambda_side == w * s on the violated side
    const int side_f = (which == 0) ? lay_f.side_lo(r_bu)
                                    : lay_f.side_hi(r_bu);
    const int side_t = (which == 0) ? lay_t.side_lo(r_bx)
                                    : lay_t.side_hi(r_bx);
    check_close_scalar(sol.lam_first(side_f), 4.0 * s_f, 1e-6,
                       p + ": weighted-slack bu (lam == 4*s)");
    check_close_scalar(sol.lam_term(side_t), 25.0 * s_t, 1e-6,
                       p + ": weighted-slack term bx (lam == 25*s)");

    check_kkt_residuals(solver, qp, sol, p);

    // dense oracle: the violated side active + the opposite (zero) box
    // slacks pinned (their slack sides are active at s = 0)
    Stg<P, NH> sg{qp, 1};
    const std::vector<ActiveRow> act = {{0, r_bu, which}, {1, r_bx, which}};
    const int other = (which == 0) ? 1 : 0;
    const std::vector<ActiveSlack> act_s = {{0, other}, {1, other}};
    VecD x0 = VecD::Zero(D::nx);
    OracleOut<P, NH> ref = dense_oracle(sg, x0, act, act_s);
    check(ref.ok, p + ": dense oracle (soft box) factorised");
    if (ref.ok)
    {
        // The masked (-inf) box side keeps a tiny positive multiplier in the
        // solver, so its slack is s = lam/w ~ 1e-5 instead of exactly 0; zero
        // those entries out before the tight comparison.
        const int mask_f = (which == 0) ? s_hi_f : s_lo_f;
        const int mask_t = (which == 0) ? s_hi_t : s_lo_t;
        check(std::fabs(sol.ux_first(mask_f)) < 1e-3,
              p + ": masked bu slack small");
        check(std::fabs(sol.ux_term(mask_t)) < 1e-3,
              p + ": masked term bx slack small");
        VecD z0 = sol.ux_first;
        z0(mask_f) = 0.0;
        VecD zn = sol.ux_term;
        zn(mask_t) = 0.0;
        check_close(z0, ref.z[0], 1e-6, p + ": z_0 (soft box)");
        check_close(zn, ref.z[1], 1e-6, p + ": z_N (soft box)");
        check_close(sol.pi[0], ref.pi[0], 1e-5, p + ": pi_0 (soft box)");
        check_close_scalar(ref.act_row_mult(0), sol.lam_first(side_f), 1e-6,
                           p + ": oracle lam bu matches solver");
        check_close_scalar(ref.act_row_mult(1), sol.lam_term(side_t), 1e-6,
                           p + ": oracle lam term matches solver");
    }

    QpRes<P, NH> res(1);
    solver.compute_residuals(qp, sol, res);
    check_close_scalar(res.obj, ref.obj, 1e-6, p + ": objective (soft box)");
    std::printf("  %s: N=1 s_bu=%.5f s_term=%.5f lam_bu=%.5f lam_term=%.5f "
                "obj=%.6f res_g=%.1e\n",
                name, s_f, s_t, sol.lam_first(side_f), sol.lam_term(side_t),
                res.obj, res.res_g_max);
}

// ---------------------------------------------------------------------
// Case 4 fill: terminal ineq / eq / lin + terminal box for TermConstr.
// The terminal equality is active; ineq / lin / box are wide (non-binding).
// ---------------------------------------------------------------------

template <class P, int NH>
void fill_terminal(Qp<P, NH>& qp, int N, const VecD& x0)
{
    using D = QpDim<P>;
    Rng rng;
    const double BIG = 1e3;

    auto fill_ux_stage = [&](auto& st, int nslack)
    {
        const int nux = D::nu + D::nx;
        MatD R(nux, nux);
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                R(i, j) = 0.3 * rng.uniform();
            }
        }
        const MatD Hux = R.transpose() * R;
        st.hess.setZero();
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                st.hess(i, j) = Hux(i, j) + (i == j ? 1.0 : 0.0);
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

    // first + path (state / control box only)
    auto fill_box_stage = [&](auto& st, bool is_first)
    {
        const auto& lay = is_first ? D::lay_first : D::lay_path;
        for (int r = 0; r < (is_first ? D::nrow_first : D::nrow_path); ++r)
        {
            const int g = lay.group_of(r);
            const int sl = lay.side_lo(r);
            const int sh = lay.side_hi(r);
            if (g == detail::g_pin)
            {
                st.DC(r, D::idx_x0[r]) = 1.0;
                if (sl >= 0)
                {
                    st.d(sl) = x0(r);
                }
                if (sh >= 0)
                {
                    st.d(sh) = -x0(r);
                }
            }
            else if (g == detail::g_bx || g == detail::g_bu)
            {
                const int var = (is_first)
                                    ? (g == detail::g_bx
                                           ? D::idxb_first[r]
                                           : D::idxb_first[D::nbx_first + (r - lay.row_off(detail::g_bu))])
                                    : (g == detail::g_bx
                                           ? D::idxb_path[r]
                                           : D::idxb_path[D::nbx + (r - lay.row_off(detail::g_bu))]);
                st.DC(r, var) = 1.0;
                if (sl >= 0)
                {
                    st.d(sl) = -BIG;
                }
                if (sh >= 0)
                {
                    st.d(sh) = -BIG;
                }
            }
        }
        st.d_mask.setConstant(1.0);
        st.m.setZero();
    };

    fill_ux_stage(qp.first, D::nslack_first);
    fill_box_stage(qp.first, true);
    for (int k = 1; k < N; ++k)
    {
        fill_ux_stage(qp.path[k - 1], D::nslack_path);
        fill_box_stage(qp.path[k - 1], false);
    }

    // terminal: box + ineq + eq + lin
    {
        const int nux = D::nx;
        MatD R(nux, nux);
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                R(i, j) = 0.3 * rng.uniform();
            }
        }
        const MatD Hux = R.transpose() * R;
        qp.term.hess.setZero();
        for (int i = 0; i < nux; ++i)
        {
            for (int j = 0; j < nux; ++j)
            {
                qp.term.hess(i, j) = Hux(i, j) + (i == j ? 1.0 : 0.0);
            }
        }
        for (int i = 0; i < nux; ++i)
        {
            qp.term.grad(i) = 0.2 * rng.uniform();
        }
        const auto& lay = D::lay_term;
        // terminal rows: [bx (nbx_t), ineq (ng_t), eq (ne_t), lin (nl_t)]
        const int nbx_t = D::nbx_t;
        const int ineq_r = lay.row_off(detail::g_ineq);
        const int eq_r = lay.row_off(detail::g_eq);
        const int lin_r = lay.row_off(detail::g_lin);
        for (int r = 0; r < D::nrow_term; ++r)
        {
            const int g = lay.group_of(r);
            const int sl = lay.side_lo(r);
            const int sh = lay.side_hi(r);
            if (g == detail::g_bx)
            {
                qp.term.DC(r, D::idxb_term[r]) = 1.0;
                if (sl >= 0)
                {
                    qp.term.d(sl) = -BIG;
                }
                if (sh >= 0)
                {
                    qp.term.d(sh) = -BIG;
                }
            }
            else if (g == detail::g_ineq)
            {
                // g_t(x) <= 0 ; wide so it is non-binding (d_hi = -BIG)
                qp.term.DC(r, 0) = 1.0;
                qp.term.DC(r, 1) = 1.0;
                if (sh >= 0)
                {
                    qp.term.d(sh) = -BIG;
                }
            }
            else if (g == detail::g_eq)
            {
                // e_t(x) == 0 ; active equality
                qp.term.DC(r, 0) = 1.0;
                qp.term.DC(r, 1) = -1.0;
                if (sl >= 0)
                {
                    qp.term.d(sl) = 0.0;
                }
                if (sh >= 0)
                {
                    qp.term.d(sh) = 0.0;
                }
            }
            else if (g == detail::g_lin)
            {
                // -BIG <= 0.5 (x0 + x1) <= BIG ; non-binding
                qp.term.DC(r, 0) = 0.5;
                qp.term.DC(r, 1) = 0.5;
                if (sl >= 0)
                {
                    qp.term.d(sl) = -BIG;
                }
                if (sh >= 0)
                {
                    qp.term.d(sh) = -BIG;
                }
            }
        }
        (void)ineq_r;
        (void)eq_r;
        (void)lin_r;
        (void)nbx_t;
        qp.term.d_mask.setConstant(1.0);
        qp.term.m.setZero();
    }
}

template <class P, int NH = Eigen::Dynamic>
void case4_terminal(const char* name, int N, const VecD& x0)
{
    const std::string p = name;
    Qp<P, NH> qp(N);
    QpSol<P, NH> sol(N);
    fill_terminal(qp, N, x0);

    HpipmQpSolver<P, NH> solver;
    const Status st = solver.solve(qp, sol);
    check(st == Status::kSolved,
          p + ": status kSolved (got " +
              std::to_string(static_cast<int>(st)) + ")");
    if (st != Status::kSolved)
    {
        return;
    }

    using D = QpDim<P>;
    const auto& lay = D::lay_term;
    const int eq_r = lay.row_off(detail::g_eq);

    // terminal equality is active: e_t = x0 - x1 == 0 at the solution
    const VecD xN = sol.ux_term.head(D::nx);
    check_close_scalar(xN(0) - xN(1), 0.0, 1e-7, p + ": terminal equality");

    // dense oracle with the terminal equality active
    Stg<P, NH> sg{qp, N};
    const std::vector<ActiveRow> act = {{N, eq_r, 1}};
    OracleOut<P, NH> ref = dense_oracle(sg, x0, act, {});
    check(ref.ok, p + ": dense oracle (active terminal eq) factorised");
    if (ref.ok)
    {
        check_close(sol.ux_term, ref.z[N], 1e-5, p + ": terminal z_N");
        for (int k = 0; k <= N; ++k)
        {
            VecD got;
            if (k == 0)
            {
                got = sol.ux_first;
            }
            else if (k == N)
            {
                got = sol.ux_term;
            }
            else
            {
                got = sol.ux_path[k - 1];
            }
            check_close(got, ref.z[k], 1e-5,
                        p + ": z_k " + std::to_string(k));
        }
        for (int k = 0; k < N; ++k)
        {
            check_close(sol.pi[k], ref.pi[k], 1e-5,
                        p + ": pi_k " + std::to_string(k));
        }
    }

    check_kkt_residuals(solver, qp, sol, p);
    QpRes<P, NH> res(N);
    solver.compute_residuals(qp, sol, res);
    std::printf("  %s: N=%d obj=%.6f res_g=%.1e res_b=%.1e\n", name, N,
                res.obj, res.res_g_max, res.res_b_max);
}

// ---------------------------------------------------------------------
// Case 5: edge cases -> status codes.
// ---------------------------------------------------------------------

// A box with lo > hi on one state -> the feasible set is empty -> the step
// length collapses to zero -> kMinStep.
template <class P, int NH = Eigen::Dynamic>
void case5_infeasible(const char* name, int N)
{
    using D = QpDim<P>;
    const std::string p = name;
    Qp<P, NH> qp(N);
    QpSol<P, NH> sol(N);
    fill_lqr(qp, N, VecD::Zero(D::nx));
    // Make the first-stage control box contradictory: lo = +2, hi = -2.
    // The control box is hard (no slack), so the feasible set is empty and
    // the step length collapses to zero.
    if (D::nbu > 0)
    {
        const auto& lay = D::lay_first;
        const int r_bu = lay.row_off(detail::g_bu);
        const int sl = lay.side_lo(r_bu);
        const int sh = lay.side_hi(r_bu);
        if (sl >= 0)
        {
            qp.first.d(sl) = 2.0;  // lo = +2
        }
        if (sh >= 0)
        {
            qp.first.d(sh) = 2.0;  // d_hi = -hi = +2  =>  hi = -2
        }
    }
    HpipmQpSolver<P, NH> solver;
    const Status st = solver.solve(qp, sol);
    // An infeasible QP must NOT be reported as solved; the interior-point
    // method either collapses the step (kMinStep), runs out of iterations
    // (kMaxIterations), or fails a factorisation (kQpFailure).
    const bool failed = (st == Status::kMinStep || st == Status::kMaxIterations ||
                         st == Status::kQpFailure);
    check(failed, p + ": infeasible QP reports a failure status (got " +
                      std::to_string(static_cast<int>(st)) + ")");
    std::printf("  %s: status = %d\n", name, static_cast<int>(st));
}

// A non-PSD Hessian -> Cholesky stays singular -> kQpFailure.
template <class P, int NH = Eigen::Dynamic>
void case5_nonpsd(const char* name, int N)
{
    using D = QpDim<P>;
    const std::string p = name;
    Qp<P, NH> qp(N);
    QpSol<P, NH> sol(N);
    fill_lqr(qp, N, VecD::Zero(D::nx));
    // overwrite the (u;x) Hessian with a negative definite one
    auto neg = [&](auto& st, int nvar, int nslack)
    {
        const int nux = nvar - nslack;
        st.hess.setZero();
        for (int i = 0; i < nux; ++i)
        {
            st.hess(i, i) = -1.0;
        }
        for (int i = 0; i < nslack; ++i)
        {
            st.hess(nux + i, nux + i) = 1.0;
        }
    };
    neg(qp.first, D::nvar_first, D::nslack_first);
    for (int k = 1; k < N; ++k)
    {
        neg(qp.path[k - 1], D::nvar_path, D::nslack_path);
    }
    neg(qp.term, D::nvar_term, D::nslack_term);

    HpipmQpSolver<P, NH> solver;
    const Status st = solver.solve(qp, sol);
    check(st == Status::kQpFailure,
          p + ": non-PSD Hessian -> kQpFailure (got " +
              std::to_string(static_cast<int>(st)) + ")");
    std::printf("  %s: status = %d\n", name, static_cast<int>(st));
}

// A well-posed QP with the iteration cap set to 1 -> exactly one IPM
// iteration -> kMaxIterations.
template <class P, int NH = Eigen::Dynamic>
void case5_maxiter(const char* name, int N, const VecD& x0)
{
    const std::string p = name;
    Qp<P, NH> qp(N);
    QpSol<P, NH> sol(N);
    fill_lqr(qp, N, x0);

    HpipmOptions opts;
    opts.iter_max = 1;
    opts.stat_max = 2;
    HpipmQpSolver<P, NH> solver(opts);
    const Status st = solver.solve(qp, sol);
    check(st == Status::kMaxIterations,
          p + ": iter_max = 1 -> kMaxIterations (got " +
              std::to_string(static_cast<int>(st)) + ")");
    check(solver.statistics().iter == 1, p + ": iter == 1");
    std::printf("  %s: status = %d iter = %d\n", name,
                static_cast<int>(st), solver.statistics().iter);
}

}  // namespace

int run_suite_1i_tests()
{
    failures = 0;
    using DI = DoubleIntegrator;
    using MS = MassSpring;
    using TC = TermConstr;
    constexpr int NH = Eigen::Dynamic;

    // (1) LQR-like QP, hard constraints only
    {
        VecD x0_di(2);
        x0_di << 1.0, 0.5;
        case1_lqr<DI, NH>("DI LQR N=2", 2, x0_di);
        case1_lqr<DI, NH>("DI LQR N=3", 3, x0_di);
        VecD x0_ms(8);
        x0_ms << 2.5, 2.5, 0, 0, 0, 0, 0, 0;
        case1_lqr<MS, NH>("MS LQR N=2", 2, x0_ms);
    }

    // (2) soft rows + slacks (velocity cap, weight 100)
    {
        VecD x0_di(2);
        x0_di << 1.0, 0.5;
        VecD jg(3);
        jg << 0.0, 0.0, 1.0;  // g = v - vmax  (u; x0; x1)
        case2_soft<DI, NH>("DI soft N=1", 1, x0_di, jg, 100.0);
        case2_soft<DI, NH>("DI soft N=2", 2, x0_di, jg, 100.0);
    }

    // (3) soft box rows (phase 3a): idxs_rev slack columns in the box DC
    {
        case3_softbox<SoftBox, NH>("SB soft box (hi violated)", false);
        case3_softbox<SoftBox, NH>("SB soft box (lo violated)", true);
    }

    // (4) terminal ineq / eq / lin + terminal box
    {
        VecD x0_tc(2);
        x0_tc << 1.0, -1.0;
        case4_terminal<TC, NH>("TC term N=1", 1, x0_tc);
        case4_terminal<TC, NH>("TC term N=2", 2, x0_tc);
    }

    // (5) edge cases -> status codes
    {
        VecD x0_di(2);
        x0_di << 1.0, 0.5;
        case5_infeasible<DI, NH>("DI infeasible", 1);
        case5_nonpsd<DI, NH>("DI non-PSD", 2);
        case5_maxiter<DI, NH>("DI maxiter", 2, x0_di);
    }

    if (failures == 0)
    {
        std::printf("All suite (1i) checks passed.\n");
    }
    return failures;
}

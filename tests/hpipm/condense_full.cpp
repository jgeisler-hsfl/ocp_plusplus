// Phase 1 (parity) block 1b + 1d test: condense_full / expand_full_sol.
//
// Part 1 (1b): assembles a dense QP from a random staged QP and compares
// every field against an independent reference built by explicit flow-map
// substitution (H = sum E_k' H_k E_k, g = sum E_k' g_k, C rows built per
// stage), cross-validating the L-recursion and the row assembly.
//
// Part 2 (1d): full pipeline on a feasible random QP — staged solve with
// HpipmQpSolver vs. condense_full + HpipmDenseQpSolver + expand_full_sol.
// Checks: staged KKT residuals at the expanded dense solution (pi
// recovery), agreement of (x, u, s), pi, lam, t and the objective between
// both solutions.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "ocp/solvers/hpipm/condense.hpp"
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

// ---------------------------------------------------------------------
// Dual-variable normalization for the staged vs pipeline comparison
//
// The primal is unique (PD Hessian), so it is compared directly.  The duals
// are not fully unique:
//   - masked sides (d_mask == 0): no constraint, lam / t are free; the two
//     solvers keep different arbitrary values there;
//   - degenerate two-sided rows (lo bound == hi bound, both sides active,
//     e.g. pin rows): KKT fixes only (lam_hi - lam_lo), not the split.
// normalize_t / normalize_lam zero the free entries (and collapse a
// degenerate pair to its difference on the lo side) so that only the
// determined parts are compared.
// ---------------------------------------------------------------------

template <class Dual>
Dual normalize_t(const Dual& t, const Dual& d_mask)
{
    return t.cwiseProduct(d_mask);
}

template <class Dual>
Dual normalize_lam(const Dual& lam, const Dual& d_mask, const Dual& d,
                   const Dual& t_ref, const detail::QpLayout& lay)
{
    Dual out = lam;
    const int nside = (int)lam.size();
    for (int i = 0; i < nside; ++i)
    {
        if (d_mask(i) < 0.5)
        {
            out(i) = 0.0;
        }
    }
    for (int r = 0; r < lay.nrow(); ++r)
    {
        const int sl = lay.side_lo(r);
        const int sh = lay.side_hi(r);
        if (sl < 0)
        {
            continue;
        }
        const bool degen = (std::fabs((double)d(sl) - (double)d(sh)) < 1e-12)
                           && (double)t_ref(sl) < 1e-6
                           && (double)t_ref(sh) < 1e-6;
        if (degen)
        {
            out(sl) = lam(sh) - lam(sl);
            out(sh) = 0.0;
        }
    }
    return out;
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
// Synthetic problems (copied from dense_qp_dim.cpp, soft rows enabled)
// ---------------------------------------------------------------------

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

struct TinyFixed : TinyFree
{
    static constexpr bool fixed_initial_state = true;
};

// ---------------------------------------------------------------------
// Fill a staged QP with seeded random, structurally valid data
// ---------------------------------------------------------------------

template <class P, int NH>
void fill_random_qp(Qp<P, NH>& qp, Rng& rng)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    const int N = qp.N;

    using BAt = Eigen::Matrix<S, D::nx, D::nu + D::nx>;
    using Bv = Eigen::Matrix<S, D::nx, 1>;

    auto fill_stage = [&](auto& hess, auto& grad, BAt* BA, Bv* b, auto& dc,
                          auto& d, auto& d_mask, auto& m, bool has_dyn,
                          int nvar, int nrow, const detail::QpLayout& lay,
                          auto const& idxs_lo, auto const& idxs_hi)
    {
        // PSD Hessian on the (u;x) part: A'A + eps*I
        const int nvx = nvar - lay.nslack;
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic> A =
            Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>::Random(nvx, nvx);
        hess.topLeftCorner(nvx, nvx) =
            A.transpose() * A
            + 0.1 * Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>::Identity(
                  nvx, nvx);
        for (int j = 0; j < lay.nslack; ++j)
        {
            hess(nvx + j, nvx + j) = 0.5 + rng.pos();  // slack weights
        }
        for (int j = 0; j < nvar; ++j)
        {
            grad(j) = rng.uniform();
        }
        if (has_dyn)
        {
            for (int i = 0; i < D::nx; ++i)
            {
                for (int j = 0; j < D::nu + D::nx; ++j)
                {
                    (*BA)(i, j) = rng.uniform();
                }
                (*b)(i) = rng.uniform();
            }
        }

        // DC: box/pin rows are unit vectors; general rows random
        std::vector<int> idxb_box;
        if (lay == D::lay_first)
        {
            idxb_box.assign(D::idxb_first.begin(), D::idxb_first.end());
        }
        else if (lay == D::lay_path)
        {
            idxb_box.assign(D::idxb_path.begin(), D::idxb_path.end());
        }
        else
        {
            idxb_box.assign(D::idxb_term.begin(), D::idxb_term.end());
        }
        // The slack columns of DC must follow the HPIPM OCP unit convention
        // (+1 for the lo-side slack, -1 for the hi-side slack, 0 elsewhere):
        // the staged solver hardcodes this coupling (stage_residuals,
        // usx_build_side_dc) and condense_full copies these columns into the
        // dense C.  Fill only the (u;x) part randomly here; the slack part is
        // set below.
        const int nux = nvar - lay.nslack;
        for (int r = 0; r < nrow; ++r)
        {
            dc.row(r).setZero();
            const int grp = lay.group_of(r);
            if (grp == detail::g_pin)
            {
                dc(r, D::idx_x0[r - lay.row_off(detail::g_pin)]) = 1.0;
            }
            else if (grp == detail::g_bx || grp == detail::g_bu)
            {
                // idxb_box is ordered [state box; control box], so the
                // control-box rows are offset by the state-box count.
                const int loc = r - lay.row_off(grp);
                const int off = (grp == detail::g_bu)
                    ? ((lay == D::lay_first) ? D::nbx_first : D::nbx)
                    : 0;
                dc(r, idxb_box[off + loc]) = 1.0;
            }
            else
            {
                for (int j = 0; j < nux; ++j)
                {
                    dc(r, j) = rng.uniform();
                }
            }
            if (idxs_lo[r] >= 0)
            {
                dc(r, idxs_lo[r]) = S(1.0);
            }
            if (idxs_hi[r] >= 0)
            {
                dc(r, idxs_hi[r]) = S(-1.0);
            }
        }

        // d / d_mask / m per side layout. General (ineq/eq/lin) lo sides are
        // masked (~30 %) to exercise the existing-but-inactive side path.
        const int lo_sz = lay.lo_size();
        for (int r = 0; r < nrow; ++r)
        {
            const int sl = lay.side_lo(r);
            if (sl >= 0)
            {
                const int grp = lay.group_of(r);
                const bool gen = (grp == detail::g_ineq || grp == detail::g_eq
                                  || grp == detail::g_lin);
                const bool masked = gen && (rng.pos() < 0.3);
                d(sl) = rng.uniform();
                d_mask(sl) = masked ? 0.0 : 1.0;
                m(sl) = rng.uniform();
            }
            d(lay.side_hi(r)) = rng.uniform();
            d_mask(lay.side_hi(r)) = 1.0;
            m(lay.side_hi(r)) = rng.uniform();
        }
        for (int j = 0; j < lay.nslack; ++j)
        {
            const int s = lo_sz + nrow + j;
            d(s) = 0.0;
            d_mask(s) = 1.0;
            m(s) = rng.uniform();
        }
    };

    fill_stage(qp.first.hess, qp.first.grad, &qp.first.BA, &qp.first.b,
               qp.first.DC, qp.first.d, qp.first.d_mask, qp.first.m, true,
               D::nvar_first, D::nrow_first, D::lay_first,
               D::idxs_lo_first, D::idxs_hi_first);
    for (int k = 1; k < N; ++k)
    {
        fill_stage(qp.path[k - 1].hess, qp.path[k - 1].grad,
                   &qp.path[k - 1].BA, &qp.path[k - 1].b,
                   qp.path[k - 1].DC, qp.path[k - 1].d,
                   qp.path[k - 1].d_mask, qp.path[k - 1].m, true,
                   D::nvar_path, D::nrow_path, D::lay_path,
                   D::idxs_lo_path, D::idxs_hi_path);
    }
    fill_stage(qp.term.hess, qp.term.grad, nullptr, nullptr, qp.term.DC,
               qp.term.d, qp.term.d_mask, qp.term.m, false, D::nvar_term,
               D::nrow_term, D::lay_term, D::idxs_lo_term, D::idxs_hi_term);
}

// ---------------------------------------------------------------------
// Independent reference: explicit flow map + E_k' H_k E_k substitution
// ---------------------------------------------------------------------

template <class P, int NH>
void ref_dense(const Qp<P, NH>& qp, DenseQp<P, NH>& out)
{
    using D = DenseQpDim<P>;
    using SD = QpDim<P>;
    using S = typename P::scalar_t;
    using M = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
    using V = Eigen::Matrix<S, Eigen::Dynamic, 1>;

    const int N = qp.N;
    out.resize(N);
    const int nv = out.nv;
    const int nvh = D::nx + N * D::nu;

    // 1) flow map (independent copy of the recurrence)
    std::vector<M> G(N + 1, M::Zero(nvh, D::nx));
    std::vector<V> gam(N + 1, V::Zero(D::nx));
    G[0].topLeftCorner(D::nx, D::nx).setIdentity();
    for (int k = 0; k < N; ++k)
    {
        M BA;
        V b;
        if (k == 0)
        {
            BA = qp.first.BA;
            b = qp.first.b;
        }
        else
        {
            BA = qp.path[k - 1].BA;
            b = qp.path[k - 1].b;
        }
        const M A = BA.block(0, D::nu, D::nx, D::nx);
        const M B = BA.block(0, 0, D::nx, D::nu);
        // G[k](v, d) = dx_k[d] / dv  =>  G[k+1].rows = G[k].rows * A'
        G[k + 1].topLeftCorner(D::nx, D::nx) =
            G[k].topLeftCorner(D::nx, D::nx) * A.transpose();
        if (k > 0)
        {
            G[k + 1].block(D::nx, 0, k * D::nu, D::nx) =
                G[k].block(D::nx, 0, k * D::nu, D::nx) * A.transpose();
        }
        G[k + 1].block(D::nx + k * D::nu, 0, D::nu, D::nx) = B.transpose();
        gam[k + 1] = A * gam[k] + b;
    }

    // 2) H = sum E_k' H_k E_k, g = sum E_k' (H_k c_k + g_k) (full
    //    substitution; c_k = [0; gamma_k; 0] is the per-stage constant from
    //    the flow map, so the x-row gradient is H_k,x * gamma_k + g_k,x)
    M H = M::Zero(nv, nv);
    V g(nv);
    g.setZero();
    for (int k = 0; k <= N; ++k)
    {
        const bool term = (k == N);
        const int nvar = term ? SD::nvar_term
                      : (k == 0 ? SD::nvar_first : SD::nvar_path);
        M E = M::Zero(nvar, nv);
        // u_k (absent at the terminal)
        if (!term)
        {
            E.block(0, D::u_off(k), D::nu, D::nu).setIdentity();
        }
        // x_k
        const int xrow = term ? 0 : D::nu;
        if (k == 0)
        {
            E.block(xrow, 0, D::nx, D::nx).setIdentity();
        }
        else
        {
            E.block(xrow, 0, D::nx, nvh) = G[k].transpose();
        }
        // s_k
        const int nsl = term ? SD::nslack_term
                      : (k == 0 ? SD::nslack_first : SD::nslack_path);
        const int srow = nvar - nsl;
        for (int j = 0; j < nsl; ++j)
        {
            E(srow + j, D::s_off(k, N) + j) = 1.0;
        }
        M Hk;
        V gk;
        if (term)
        {
            Hk = qp.term.hess;
            gk = qp.term.grad;
        }
        else if (k == 0)
        {
            Hk = qp.first.hess;
            gk = qp.first.grad;
        }
        else
        {
            Hk = qp.path[k - 1].hess;
            gk = qp.path[k - 1].grad;
        }
        // full substitution: y_k = E v + c_k with c_k = [0; gamma_k; 0]
        V ck(nvar);
        ck.setZero();
        ck.segment(xrow, D::nx) = gam[k];
        H.noalias() += E.transpose() * Hk * E;
        g.noalias() += E.transpose() * (Hk * ck + gk);
    }
    out.H = H;
    out.g = g;

    // 3) general rows: dense order = stage 0 (soft bx | soft bu |
    //    ineq|eq|lin), then per k = 1..N: (bx|soft bu|ineq|eq|lin)
    auto emit_gen = [&](const auto& dc, const auto& ds, const auto& dms,
                        const auto& ms, const detail::QpLayout& lay, int k,
                        int xoff, int nvar, auto const& idxs_lo,
                        auto const& idxs_hi, int& rg)
    {
        const int nsl = nvar - xoff - D::nx;
        const int nbox = out.nb;
        const int ngen = out.ng;
        for (int r = 0; r < lay.nrow(); ++r)
        {
            const int grp = lay.group_of(r);
            // Rows that stay box rows in the dense QP: pin rows, and hard
            // (non-soft) box rows.  Soft bx / bu rows carry a slack column
            // and are emitted as general rows.
            const bool soft = (idxs_lo[r] >= 0 || idxs_hi[r] >= 0);
            if (grp == detail::g_pin
                || (grp == detail::g_bu && !soft)
                || (k == 0 && grp == detail::g_bx && !soft))
            {
                continue;  // stays a box row in the dense QP
            }
            out.C_lo.row(rg).setZero();
            const auto dcx = dc.row(r).segment(xoff, D::nx);
            out.C_lo.row(rg).segment(0, nvh) = dcx * G[k].transpose();
            if (k < N)
            {
                out.C_lo.row(rg).segment(D::u_off(k), D::nu) =
                    dc.row(r).head(D::nu);
            }
            out.C_hi.row(rg) = out.C_lo.row(rg);
            if (nsl > 0)
            {
                // independent construction: a dense slack column carries the
                // lo slack of this row iff the staged DC entry is +1 (goes
                // into C_lo only) and the hi slack iff it is -1 (C_hi only)
                for (int j = 0; j < nsl; ++j)
                {
                    const double val = dc.row(r)(nvar - nsl + j);
                    const int col = D::s_off(k, N) + j;
                    if (val > S(0.5))
                    {
                        out.C_lo(rg, col) = S(1.0);
                    }
                    else if (val < S(-0.5))
                    {
                        out.C_hi(rg, col) = val;
                    }
                }
            }
            const S sh = dcx.dot(gam[k]);
            const int sl = lay.side_lo(r);
            out.d(nbox + rg) = (sl >= 0) ? (ds(sl) - sh) : 0.0;
            out.d_mask(nbox + rg) = (sl >= 0) ? dms(sl) : 0.0;
            out.m(nbox + rg) = (sl >= 0) ? ms(sl) : 0.0;
            out.d(nbox + ngen + nbox + rg) = ds(lay.side_hi(r)) + sh;
            out.d_mask(nbox + ngen + nbox + rg) = dms(lay.side_hi(r));
            out.m(nbox + ngen + nbox + rg) = ms(lay.side_hi(r));
            ++rg;
        }
    };

    int rg = 0;
    emit_gen(qp.first.DC, qp.first.d, qp.first.d_mask, qp.first.m,
             SD::lay_first, 0, D::nu, SD::nvar_first, SD::idxs_lo_first,
             SD::idxs_hi_first, rg);
    for (int k = 1; k < N; ++k)
    {
        emit_gen(qp.path[k - 1].DC, qp.path[k - 1].d, qp.path[k - 1].d_mask,
                 qp.path[k - 1].m, SD::lay_path, k, D::nu, SD::nvar_path,
                 SD::idxs_lo_path, SD::idxs_hi_path, rg);
    }
    emit_gen(qp.term.DC, qp.term.d, qp.term.d_mask, qp.term.m, SD::lay_term,
             N, 0, SD::nvar_term, SD::idxs_lo_term, SD::idxs_hi_term, rg);
    assert(rg == out.ng);

    // 4) box rows: dense order = stage 0 (pin | hard bx | hard bu), path
    //    hard bu k = 1..N-1, slack non-negativity k = 0..N.  Soft box rows
    //    are general rows (emitted in part 3).
    auto copy_box_sides = [&](int rb, const auto& ds, const auto& dms,
                              const auto& ms, const detail::QpLayout& lay,
                              int r)
    {
        const int nbox = out.nb;
        const int ngen = out.ng;
        out.d(rb) = ds(lay.side_lo(r));
        out.d_mask(rb) = dms(lay.side_lo(r));
        out.m(rb) = ms(lay.side_lo(r));
        out.d(rb + nbox + ngen) = ds(lay.side_hi(r));
        out.d_mask(rb + nbox + ngen) = dms(lay.side_hi(r));
        out.m(rb + nbox + ngen) = ms(lay.side_hi(r));
    };

    int rb = 0;
    {
        const auto& st = qp.first;
        const auto& lay = SD::lay_first;
        for (int j = 0; j < lay.rows[detail::g_pin]; ++j)
        {
            const int r = lay.row_off(detail::g_pin) + j;
            out.idxb[rb] = D::x0_off() + j;
            copy_box_sides(rb, st.d, st.d_mask, st.m, lay, r);
            ++rb;
        }
        for (int j = 0; j < lay.rows[detail::g_bx]; ++j)
        {
            const int r = lay.row_off(detail::g_bx) + j;
            if (SD::idxs_lo_first[r] >= 0 || SD::idxs_hi_first[r] >= 0)
            {
                continue;  // soft state box -> general row
            }
            out.idxb[rb] = D::x0_off() + P::state_box_idx[j];
            copy_box_sides(rb, st.d, st.d_mask, st.m, lay, r);
            ++rb;
        }
        for (int j = 0; j < lay.rows[detail::g_bu]; ++j)
        {
            const int r = lay.row_off(detail::g_bu) + j;
            if (SD::idxs_lo_first[r] >= 0 || SD::idxs_hi_first[r] >= 0)
            {
                continue;  // soft control box -> general row
            }
            out.idxb[rb] = D::u_off(0) + P::control_box_idx[j];
            copy_box_sides(rb, st.d, st.d_mask, st.m, lay, r);
            ++rb;
        }
    }
    for (int k = 1; k < N; ++k)
    {
        const auto& st = qp.path[k - 1];
        const auto& lay = SD::lay_path;
        for (int j = 0; j < lay.rows[detail::g_bu]; ++j)
        {
            const int r = lay.row_off(detail::g_bu) + j;
            if (SD::idxs_lo_path[r] >= 0 || SD::idxs_hi_path[r] >= 0)
            {
                continue;  // soft control box -> general row
            }
            out.idxb[rb] = D::u_off(k) + P::control_box_idx[j];
            copy_box_sides(rb, st.d, st.d_mask, st.m, lay, r);
            ++rb;
        }
    }
    for (int k = 0; k <= N; ++k)
    {
        const int nsl = (k == 0) ? SD::nslack_first
                     : (k == N) ? SD::nslack_term : SD::nslack_path;
        const int nbox = out.nb;
        const int ngen = out.ng;
        for (int j = 0; j < nsl; ++j)
        {
            out.idxb[rb] = D::s_off(k, N) + j;
            out.d(rb) = 0.0;
            out.d_mask(rb) = 1.0;
            out.m(rb) = 0.0;
            out.d(rb + nbox + ngen) = 0.0;
            out.d_mask(rb + nbox + ngen) = 0.0;
            out.m(rb + nbox + ngen) = 0.0;
            ++rb;
        }
    }
    assert(rb == out.nb);
}

// ---------------------------------------------------------------------
// Compare condense_full against the reference
// ---------------------------------------------------------------------

/// Make the random QP of fill_random_qp actually solvable:
///  - zero the complementarity target (production QPs have m = 0)
///  - widen every active bound so that the zero-input trajectory
///    (x_0 = 0, u_k = 0, s_k = 0) is strictly interior, which also makes
///    v = 0 an interior point of the condensed problem
template <class P, int NH>
void make_feasible(Qp<P, NH>& qp, Rng& rng)
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;
    const int N = qp.N;
    const int nx = D::nx, nu = D::nu;

    qp.first.m.setZero();
    for (int k = 1; k < N; ++k)
    {
        qp.path[k - 1].m.setZero();
    }
    qp.term.m.setZero();

    Eigen::VectorXd x(nx);
    x.setZero();

    auto widen_stage = [&](auto& stage, const detail::QpLayout& lay, int xoff)
    {
        const int nvar = (int)stage.DC.cols();
        Eigen::VectorXd z(nvar);
        z.setZero();
        z.segment(xoff, nx) = x;
        for (int r = 0; r < lay.nrow(); ++r)
        {
            const int grp = lay.group_of(r);
            const S val = stage.DC.row(r).dot(z);
            const int sl = lay.side_lo(r);
            const int sh = lay.side_hi(r);
            if (grp == detail::g_pin)
            {
                // keep a real pin (lo == hi) at the zero trajectory
                stage.d(sl) = S(0.0);
                stage.d(sh) = S(0.0);
            }
            else
            {
                if (sl >= 0 && stage.d_mask(sl) > S(0.5))
                {
                    stage.d(sl) = val - (S(0.5) + rng.pos());
                }
                if (stage.d_mask(sh) > S(0.5))
                {
                    stage.d(sh) = -(val + (S(0.5) + rng.pos()));
                }
            }
        }
    };

    widen_stage(qp.first, D::lay_first, nu);
    for (int k = 0; k < N; ++k)
    {
        const auto& BA = (k == 0) ? qp.first.BA : qp.path[k - 1].BA;
        const auto& b = (k == 0) ? qp.first.b : qp.path[k - 1].b;
        x = BA.block(0, nu, nx, nx) * x + b;
        if (k + 1 < N)
        {
            widen_stage(qp.path[k], D::lay_path, nu);
        }
        else
        {
            widen_stage(qp.term, D::lay_term, 0);
        }
    }
}

// ---------------------------------------------------------------------
// Part 2 (block 1d): full pipeline
//   staged QP -> condense -> dense solve -> expand  vs  staged solve
// ---------------------------------------------------------------------

template <class P, int NH>
void check_pipeline(const char* name, int N)
{
    using S = typename P::scalar_t;
    const std::string p = name;

    Qp<P, NH> qp(N);
    Rng rng;
    fill_random_qp(qp, rng);
    make_feasible(qp, rng);

    // reference: staged solve (cold start)
    HpipmQpSolver<P, NH> s_solver;
    QpSol<P, NH> sol_s;
    sol_s.resize(N);
    check(s_solver.solve(qp, sol_s) == Status::kSolved,
          p + " staged status");
    QpRes<P, NH> res_s;
    res_s.resize(N);
    s_solver.compute_residuals(qp, sol_s, res_s);

    // pipeline: condense -> dense solve -> expand
    DenseQp<P, NH> dense;
    condense_full(qp, dense);
    HpipmDenseQpSolver<P, NH> d_solver;
    DenseQpSol<P, NH> sol_d;
    sol_d.resize(N);
    check(d_solver.solve(dense, sol_d) == Status::kSolved,
          p + " dense status");

    QpSol<P, NH> sol_e;
    expand_full_sol(qp, dense, sol_d, sol_e);

    std::printf("%s staged: %d iter, dense: %d iter\n", name,
                s_solver.statistics().iter, d_solver.statistics().iter);

    // (a) staged KKT residuals at the expanded dense solution; this is
    //     the pi-recovery check of expand_full_sol
    QpRes<P, NH> res_e;
    res_e.resize(N);
    s_solver.compute_residuals(qp, sol_e, res_e);
    check(res_e.res_g_max <= S(1e-5), p + " expanded res_g_max");
    check(res_e.res_b_max <= S(1e-9), p + " expanded res_b_max");
    check(res_e.res_d_max <= S(1e-7), p + " expanded res_d_max");
    check(res_e.res_m_max <= S(1e-5), p + " expanded res_m_max");

    // (b) agreement with the staged solution (both solve the same QP;
    //     the primal is unique since the Hessian is PD; the dual is unique
    //     except on masked sides and degenerate rows, see normalize_*)
    const auto& dmask_f = qp.first.d_mask;
    check_close(sol_e.ux_first, sol_s.ux_first, 1e-5, p + " ux_first");
    check_close(normalize_t(sol_e.t_first, dmask_f),
                normalize_t(sol_s.t_first, dmask_f), 1e-5, p + " t_first");
    check_close(normalize_lam(sol_e.lam_first, dmask_f, qp.first.d,
                              sol_s.t_first, QpDim<P>::lay_first),
                normalize_lam(sol_s.lam_first, dmask_f, qp.first.d,
                              sol_s.t_first, QpDim<P>::lay_first),
                1e-3, p + " lam_first");
    for (int k = 0; k < N - 1; ++k)
    {
        const auto& dmask_p = qp.path[k].d_mask;
        check_close(sol_e.ux_path[k], sol_s.ux_path[k], 1e-5,
                    p + " ux_path[" + std::to_string(k) + "]");
        check_close(normalize_t(sol_e.t_path[k], dmask_p),
                    normalize_t(sol_s.t_path[k], dmask_p), 1e-5,
                    p + " t_path[" + std::to_string(k) + "]");
        check_close(normalize_lam(sol_e.lam_path[k], dmask_p, qp.path[k].d,
                                  sol_s.t_path[k], QpDim<P>::lay_path),
                    normalize_lam(sol_s.lam_path[k], dmask_p, qp.path[k].d,
                                  sol_s.t_path[k], QpDim<P>::lay_path),
                    1e-3, p + " lam_path[" + std::to_string(k) + "]");
    }
    const auto& dmask_t = qp.term.d_mask;
    check_close(sol_e.ux_term, sol_s.ux_term, 1e-5, p + " ux_term");
    check_close(normalize_t(sol_e.t_term, dmask_t),
                normalize_t(sol_s.t_term, dmask_t), 1e-5, p + " t_term");
    check_close(normalize_lam(sol_e.lam_term, dmask_t, qp.term.d,
                              sol_s.t_term, QpDim<P>::lay_term),
                normalize_lam(sol_s.lam_term, dmask_t, qp.term.d,
                              sol_s.t_term, QpDim<P>::lay_term),
                1e-3, p + " lam_term");
    for (int k = 0; k < N; ++k)
    {
        check_close(sol_e.pi[k], sol_s.pi[k], 1e-4,
                    p + " pi[" + std::to_string(k) + "]");
    }
    check(res_e.obj - res_s.obj <= S(1e-4), p + " obj agreement");
}

template <class P, int NH>
void check_condense(const char* name, int N)
{
    Qp<P, NH> qp(N);
    Rng rng;
    fill_random_qp(qp, rng);

    DenseQp<P, NH> dense;
    condense_full(qp, dense);
    DenseQp<P, NH> ref;
    ref_dense(qp, ref);

    const std::string p = name;
    check_close(dense.H, ref.H, 1e-11, p + " H");
    check_close(dense.g, ref.g, 1e-11, p + " g");
    check_close(dense.C_lo, ref.C_lo, 1e-12, p + " C_lo");
    check_close(dense.C_hi, ref.C_hi, 1e-12, p + " C_hi");
    check_close(dense.d, ref.d, 1e-12, p + " d");
    check_close(dense.d_mask, ref.d_mask, 0.0, p + " d_mask");
    check_close(dense.m, ref.m, 0.0, p + " m");
    check(dense.idxb == ref.idxb, p + " idxb");
}

}  // namespace

int main()
{
    check_condense<TinyFree, Eigen::Dynamic>("TinyFree N=1 ", 1);
    check_condense<TinyFree, Eigen::Dynamic>("TinyFree N=4 ", 4);
    check_condense<TinyFixed, Eigen::Dynamic>("TinyFixed N=1 ", 1);
    check_condense<TinyFixed, Eigen::Dynamic>("TinyFixed N=4 ", 4);
    check_condense<TinyFixed, 3>("TinyFixed NH=3", 3);
    check_condense<DoubleIntegrator, Eigen::Dynamic>("DI N=5     ", 5);
    check_condense<MassSpring, 4>("MS NH=4    ", 4);

    // Part 2 (block 1d): full pipeline vs staged solve
    check_pipeline<TinyFree, Eigen::Dynamic>("pipe free N=3 ", 3);
    check_pipeline<TinyFixed, Eigen::Dynamic>("pipe fixed N=3", 3);
    check_pipeline<TinyFixed, 3>("pipe fixed NH=3", 3);

    if (failures == 0)
    {
        std::printf("All condense_full checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}

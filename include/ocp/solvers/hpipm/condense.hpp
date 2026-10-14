// ocp/solvers/hpipm/condense.hpp
//
// Full condensing: staged Qp -> dense QP and back (parity plan Phase 1,
// docs/plans/SQP_HPIPM_PARITY_PHASE1_PLAN.md).
//
// condense_full(qp, dense):
//   The staged QP's dynamics x_{k+1} = BA_k (u_k; x_k) + b_k (k = 0..N-1)
//   are eliminated by substitution, expressing every state step x_k (k >= 1)
//   in the dense variable vector
//
//     v = [x_0; u_0; ...; u_{N-1}; s_0; ...; s_N]
//
//     x_k = G_k' [x_0; u_0; ...; u_{k-1}] + gamma_k
//
//   G_k stores G_k(v, d) = dx_k[d] / dv, so the transpose acts on the
//   dense variable vector.
//
//   The condensed problem is a plain dense QP (no equality rows): the
//   objective is the staged objective with the flow map substituted (direct
//   block assembly, H = sum E_k' H_k E_k and g = sum E_k' (H_k c_k + g_k)
//   with c_k the per-stage flow-map constant), the general rows are the
//   staged ineq/eq/lin rows plus the converted state-box rows (their d
//   offsets shifted by the flow-map constant, see DenseQpDim), and the
//   slacks stay ordinary variables with a diagonal Hessian block.
//
// expand_full_sol(qp, dense, dsol, sol):
//   Primal: u_k / x_0 / slacks read off v; states by forward simulation
//   (the QP dynamics, step semantics).
//   Dynamics multipliers (ocp++ sign convention, hpipm.hpp
//   compute_residuals where res_g_k = ... + BA_k' pi_k - pin(pi_{k-1}),
//   res_b = BA z + b - x_{k+1}): stationarity of x_k gives
//     pi_{k-1} = (res_g_k^{no-pi})_x + A_k' pi_k      (k = N-1 .. 1)
//     pi_{N-1} = (res_g_N^{no-pi})_x
//   where (res_g_k^{no-pi})_x = (H_k z_k + g_k)_x + DC_k,x'(lam_hi - lam_lo)
//   (masked).  lam / t are mapped from the dense side layout back to the
//   staged side layouts (dense row order documented in dense_qp.hpp).
//
// HPIPM reference: acados/external/hpipm/cond/x_cond.c, x_cond_aux.c
// (COND_LHS / COND_RHS / COND_EXPAND_SOL).

#pragma once

#include <cassert>
#include <vector>

#include <Eigen/Dense>

#include "dense_qp.hpp"

namespace ocp
{

// =========================================================================
//  condense_full
// =========================================================================

/// Condense the staged QP `in` (horizon N) into the dense QP `out`.
///
/// The dense row / side layouts, variable order and extent formulas are
/// defined in dense_qp.hpp (DenseQpDim).  Stage-0 rows map directly (x_0 is
/// a dense variable, pin rows included); path/terminal state-box rows and
/// all general rows go through the flow map with the d-constant shift.
template <class P, int NH>
void condense_full(const Qp<P, NH>& in, DenseQp<P, NH>& out)
{
    using D = DenseQpDim<P>;
    using SD = QpDim<P>;
    using S = typename P::scalar_t;
    using M = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
    using V = Eigen::Matrix<S, Eigen::Dynamic, 1>;
    using namespace ocp::detail;

    const int N = in.N;
    out.resize(N);
    out.H.setZero();
    out.g.setZero();

    const int nx = D::nx;
    const int nu = D::nu;
    const int nvh = nx + N * nu;  // hard block [x_0; u_0; ...; u_{N-1}]

    // ------------------------------------------------------------------
    // 1) Flow map: x_k = G[k]' v_hard + gamma[k], k = 0..N
    //    (G[k] is (nvh x nx), G[k](v, d) = dx_k[d] / dv)
    // ------------------------------------------------------------------
    std::vector<M> G(N + 1, M::Zero(nvh, nx));
    std::vector<V> gamma(N + 1, V::Zero(nx));
    G[0].topLeftCorner(nx, nx).setIdentity();
    for (int k = 0; k < N; ++k)
    {
        M A;
        M B;
        V b;
        if (k == 0)
        {
            A = in.first.BA.block(0, nu, nx, nx);
            B = in.first.BA.block(0, 0, nx, nu);
            b = in.first.b;
        }
        else
        {
            A = in.path[k - 1].BA.block(0, nu, nx, nx);
            B = in.path[k - 1].BA.block(0, 0, nx, nu);
            b = in.path[k - 1].b;
        }
        // G[k] stores G[k](v, d) = dx_k[d] / dv, so the recurrence is
        // G[k+1].rows = G[k].rows * A' (rows x_0 / u_j (j < k)); row u_k: B'
        G[k + 1].topLeftCorner(nx, nx) =
            G[k].topLeftCorner(nx, nx) * A.transpose();
        if (k > 0)
        {
            G[k + 1].block(nx, 0, k * nu, nx) =
                G[k].block(nx, 0, k * nu, nx) * A.transpose();
        }
        G[k + 1].block(nx + k * nu, 0, nu, nx) = B.transpose();
        gamma[k + 1] = A * gamma[k] + b;
    }

    // ------------------------------------------------------------------
    // 2) H and g: direct flow-map substitution
    //
    // Each staged objective 0.5 z_k' H_k z_k + g_k' z_k is expressed in the
    // dense variables by substituting x_k = G[k]' v_hard + gamma[k]
    // (k >= 1) and x_0 = v_hard[0:nx].  Only the hard block
    // [x_0; u_0; ...; u_{N-1}] participates in H / g here; the slacks enter
    // only through their own diagonal Hessian block (added below via
    // add_slacks) and the constraint rows, never through the (u;x) Hessian
    // (zero off-diagonal in the staged model).  For a stage whose hard part
    // is y_k = E_k v_hard + c_k (E_k embeds u_k and x_k, c_k = [0; gamma_k])
    // the contribution is  H += E_k' H_ux E_k,  g += E_k' (H_ux c_k + g_ux).
    // ------------------------------------------------------------------
    M H = M::Zero(nvh, nvh);
    V g(nvh);
    g.setZero();

    for (int k = 0; k < N; ++k)
    {
        M H_uu;
        M H_ux;
        M H_xx;
        V gk_u;
        V gk_x;
        if (k == 0)
        {
            H_uu = in.first.hess.topLeftCorner(nu, nu);
            H_ux = in.first.hess.block(0, nu, nu, nx);
            H_xx = in.first.hess.block(nu, nu, nx, nx);
            gk_u = in.first.grad.head(nu);
            gk_x = in.first.grad.segment(nu, nx);
        }
        else
        {
            H_uu = in.path[k - 1].hess.topLeftCorner(nu, nu);
            H_ux = in.path[k - 1].hess.block(0, nu, nu, nx);
            H_xx = in.path[k - 1].hess.block(nu, nu, nx, nx);
            gk_u = in.path[k - 1].grad.head(nu);
            gk_x = in.path[k - 1].grad.segment(nu, nx);
        }

        const int u_off = D::u_off(k);
        const int nact = nx + k * nu;  // active cols of G[k]
        const M Gk_a = G[k].topLeftCorner(nact, nx);
        const V& gm = gamma[k];

        // u-u block (u_off == nact)
        H.block(u_off, u_off, nu, nu).noalias() += H_uu;
        // u-x cross blocks (G[k] is zero beyond its first nact rows)
        const M xu = H_ux * Gk_a.transpose();  // (nu x nact)
        H.block(u_off, 0, nu, nact).noalias() += xu;
        H.block(0, u_off, nact, nu).noalias() += xu.transpose();
        // x-x block (top-left nact x nact)
        H.topLeftCorner(nact, nact).noalias() +=
            Gk_a * (H_xx * Gk_a.transpose());

        // linear part: gradient of the step-quadratic at the flow point
        g.segment(u_off, nu).noalias() += H_ux * gm + gk_u;
        g.head(nact).noalias() += Gk_a * (H_xx * gm + gk_x);
    }

    // Terminal: x_N = G[N]' v_hard + gamma[N] (no control at the terminal)
    {
        const auto& st = in.term;
        const M H_xx = st.hess.topLeftCorner(nx, nx);
        const V gk_x = st.grad.head(nx);
        H.noalias() += G[N] * (H_xx * G[N].transpose());
        g.noalias() += G[N] * (H_xx * gamma[N] + gk_x);
    }

    out.H.topLeftCorner(nvh, nvh) = H;
    out.g.head(nvh) = g;

    // Slack diagonal blocks + linear parts (slacks are dense variables;
    // their indices live in the full nv-space, beyond the hard block).
    auto add_slacks = [&](const auto& st, int nvar, int nsl, int k)
    {
        const int s0 = D::s_off(k, N);
        for (int j = 0; j < nsl; ++j)
        {
            out.H(s0 + j, s0 + j) = st.hess(nvar - nsl + j, nvar - nsl + j);
            out.g(s0 + j) = st.grad(nvar - nsl + j);
        }
    };
    add_slacks(in.first, SD::nvar_first, SD::nslack_first, 0);
    for (int k = 1; k < N; ++k)
    {
        add_slacks(in.path[k - 1], SD::nvar_path, SD::nslack_path, k);
    }
    add_slacks(in.term, SD::nvar_term, SD::nslack_term, N);

    // ------------------------------------------------------------------
    // 3) Dense rows: C, idxb, d, d_mask, m
    //
    // Dense row order (must stay in sync with expand_full_sol and with the
    // DenseQpDim header).  Soft box rows (state box / control box rows whose
    // staged DC row carries a slack column) become general rows because a
    // dense box row has a single idxb variable and cannot hold the slack
    // coupling.  Only hard box rows (no soft index) stay dense box rows.
    //   box rows:  stage-0 (pin | hard bx | hard bu), then path k = 1..N-1
    //              hard bu rows, then slack non-negativity rows (v order,
    //              k = 0..N)
    //   gen rows:  stage-0 (soft bx | soft bu | ineq | eq | lin), then per
    //              k = 1..N: (bx | soft bu | ineq | eq | lin)
    // ------------------------------------------------------------------
    const int nb = out.nb;
    const int ng = out.ng;
    int rb = 0;  // dense box row
    int rg = 0;  // dense general row

    // ------------------------------------------------------------------
    // Stage 0: direct mapping (x_0 is a dense variable; no flow shift)
    // ------------------------------------------------------------------
    {
        const auto& st = in.first;
        const auto& lay = SD::lay_first;

        // Soft box rows carry a slack column in their DC row, which a dense
        // box row (single idxb variable) cannot express: they are emitted as
        // general rows.  The remaining box rows (pin, hard bx / bu) stay box
        // rows.  A side that is present but masked in the staged model
        // (infinite bound) stays masked here.
        auto is_soft = [&](int r)
        {
            return SD::idxs_lo_first[r] >= 0 || SD::idxs_hi_first[r] >= 0;
        };
        auto copy_box = [&](int r)
        {
            out.d(rb) = st.d(lay.side_lo(r));
            out.d_mask(rb) = st.d_mask(lay.side_lo(r));
            out.m(rb) = st.m(lay.side_lo(r));
            out.d(rb + nb + ng) = st.d(lay.side_hi(r));
            out.d_mask(rb + nb + ng) = st.d_mask(lay.side_hi(r));
            out.m(rb + nb + ng) = st.m(lay.side_hi(r));
            ++rb;
        };
        for (int j = 0; j < lay.rows[g_pin]; ++j)
        {
            out.idxb[rb] = D::x0_off() + j;
            copy_box(lay.row_off(g_pin) + j);
        }
        for (int j = 0; j < lay.rows[g_bx]; ++j)
        {
            const int r = lay.row_off(g_bx) + j;
            if (is_soft(r))
            {
                continue;
            }
            out.idxb[rb] = D::x0_off() + P::state_box_idx[j];
            copy_box(r);
        }
        for (int j = 0; j < lay.rows[g_bu]; ++j)
        {
            const int r = lay.row_off(g_bu) + j;
            if (is_soft(r))
            {
                continue;
            }
            out.idxb[rb] = D::u_off(0) + P::control_box_idx[j];
            copy_box(r);
        }

        // General rows: soft bx | soft bu | ineq | eq | lin
        const int grps[5] = {g_bx, g_bu, g_ineq, g_eq, g_lin};
        for (int gg = 0; gg < 5; ++gg)
        {
            const int grp = grps[gg];
            for (int j = 0; j < lay.rows[grp]; ++j)
            {
                const int r = lay.row_off(grp) + j;
                const bool boxish = (grp == g_bx || grp == g_bu);
                if (boxish && !is_soft(r))
                {
                    continue;
                }
                const int rd = nb + rg;
                // Coefficient access (st.DC may be a compile-time 0-row
                // matrix; Eigen row-block .segment/.head then fail the
                // vector-only static assert even though this loop never
                // runs).  The C rows start zeroed (out.resize) and each
                // row is written exactly once, so no explicit row reset.
                for (int i = 0; i < nx; ++i)
                {
                    out.C_lo(rg, i) = st.DC(r, nu + i);
                    out.C_hi(rg, i) = st.DC(r, nu + i);
                }
                for (int i = 0; i < nu; ++i)
                {
                    out.C_lo(rg, D::u_off(0) + i) = st.DC(r, i);
                    out.C_hi(rg, D::u_off(0) + i) = st.DC(r, i);
                }
                if (SD::nslack_first > 0)
                {
                    const int sl_off = SD::nvar_first - SD::nslack_first;
                    for (int i = 0; i < SD::nslack_first; ++i)
                    {
                        out.C_lo(rg, D::s_off(0, N) + i) =
                            st.DC(r, sl_off + i);
                        out.C_hi(rg, D::s_off(0, N) + i) =
                            st.DC(r, sl_off + i);
                    }
                    if (SD::idxs_lo_first[r] >= 0)
                    {
                        out.C_hi(rg, D::s_off(0, N) +
                                         (SD::idxs_lo_first[r] - sl_off)) = 0.0;
                    }
                    if (SD::idxs_hi_first[r] >= 0)
                    {
                        out.C_lo(rg, D::s_off(0, N) +
                                         (SD::idxs_hi_first[r] - sl_off)) = 0.0;
                    }
                }
                const int sl = lay.side_lo(r);
                out.d(rd) = (sl >= 0) ? st.d(sl) : 0.0;
                out.d_mask(rd) = (sl >= 0) ? st.d_mask(sl) : 0.0;
                out.m(rd) = (sl >= 0) ? st.m(sl) : 0.0;
                out.d(rd + nb + ng) = st.d(lay.side_hi(r));
                out.d_mask(rd + nb + ng) = st.d_mask(lay.side_hi(r));
                out.m(rd + nb + ng) = st.m(lay.side_hi(r));
                ++rg;
            }
        }
    }

    // ------------------------------------------------------------------
    // Path (k = 1..N-1) and terminal (k = N): flow-map rows
    // ------------------------------------------------------------------
    auto emit_stage = [&](const auto& st, const QpLayout& lay, int k,
                          int xoff, int nvar, auto const& idxs_lo,
                          auto const& idxs_hi)
    {
        const int nsl = nvar - xoff - nx;
        const M& Gk = G[k];
        const V& gmv = gamma[k];

        // General rows: bx (converted) | soft bu | ineq | eq | lin.  Hard
        // control-box rows (no soft index) stay dense box rows below.
        const int grps[5] = {g_bx, g_bu, g_ineq, g_eq, g_lin};
        for (int gg = 0; gg < 5; ++gg)
        {
            const int grp = grps[gg];
            for (int j = 0; j < lay.rows[grp]; ++j)
            {
                const int r = lay.row_off(grp) + j;
                if (grp == g_bu && idxs_lo[r] < 0 && idxs_hi[r] < 0)
                {
                    continue;
                }
                const int rd = nb + rg;
                // Coefficient access (same reason as the stage-0 block:
                // st.DC may be a compile-time 0-row matrix).  The C rows
                // start zeroed (out.resize) and each row is written
                // exactly once, so no explicit row reset.
                V dcx(nx);
                for (int i = 0; i < nx; ++i)
                {
                    dcx(i) = st.DC(r, xoff + i);
                }
                const auto cprod = Gk * dcx;  // nvh x 1
                for (int i = 0; i < nvh; ++i)
                {
                    out.C_lo(rg, i) = cprod(i);
                    out.C_hi(rg, i) = cprod(i);
                }
                if (xoff > 0)
                {
                    for (int i = 0; i < nu; ++i)
                    {
                        out.C_lo(rg, D::u_off(k) + i) = st.DC(r, i);
                        out.C_hi(rg, D::u_off(k) + i) = st.DC(r, i);
                    }
                }
                if (nsl > 0)
                {
                    const int sl_off = nvar - nsl;
                    for (int i = 0; i < nsl; ++i)
                    {
                        out.C_lo(rg, D::s_off(k, N) + i) =
                            st.DC(r, sl_off + i);
                        out.C_hi(rg, D::s_off(k, N) + i) =
                            st.DC(r, sl_off + i);
                    }
                    if (idxs_lo[r] >= 0)
                    {
                        out.C_hi(rg, D::s_off(k, N)
                                      + (idxs_lo[r] - sl_off)) = 0.0;
                    }
                    if (idxs_hi[r] >= 0)
                    {
                        out.C_lo(rg, D::s_off(k, N)
                                       + (idxs_hi[r] - sl_off)) = 0.0;
                    }
                }
                const S sh = dcx.dot(gmv);
                const int sl = lay.side_lo(r);
                out.d(rd) = (sl >= 0) ? (st.d(sl) - sh) : 0.0;
                out.d_mask(rd) = (sl >= 0) ? st.d_mask(sl) : 0.0;
                out.m(rd) = (sl >= 0) ? st.m(sl) : 0.0;
                out.d(rd + nb + ng) = st.d(lay.side_hi(r)) + sh;
                out.d_mask(rd + nb + ng) = st.d_mask(lay.side_hi(r));
                out.m(rd + nb + ng) = st.m(lay.side_hi(r));
                ++rg;
            }
        }

        // Box rows: hard control box (terminal has none); soft bu rows were
        // emitted as general rows above.
        for (int j = 0; j < lay.rows[g_bu]; ++j)
        {
            const int r = lay.row_off(g_bu) + j;
            if (idxs_lo[r] >= 0 || idxs_hi[r] >= 0)
            {
                continue;
            }
            out.idxb[rb] = D::u_off(k) + P::control_box_idx[j];
            out.d(rb) = st.d(lay.side_lo(r));
            out.d_mask(rb) = st.d_mask(lay.side_lo(r));
            out.m(rb) = st.m(lay.side_lo(r));
            out.d(rb + nb + ng) = st.d(lay.side_hi(r));
            out.d_mask(rb + nb + ng) = st.d_mask(lay.side_hi(r));
            out.m(rb + nb + ng) = st.m(lay.side_hi(r));
            ++rb;
        }
    };
    for (int k = 1; k < N; ++k)
    {
        emit_stage(in.path[k - 1], SD::lay_path, k, nu, SD::nvar_path,
                   SD::idxs_lo_path, SD::idxs_hi_path);
    }
    emit_stage(in.term, SD::lay_term, N, 0, SD::nvar_term, SD::idxs_lo_term,
               SD::idxs_hi_term);

    // Slack non-negativity rows (lo side active, hi side masked)
    for (int k = 0; k <= N; ++k)
    {
        const int nsl = (k == 0) ? SD::nslack_first
                     : (k == N) ? SD::nslack_term : SD::nslack_path;
        for (int j = 0; j < nsl; ++j)
        {
            out.idxb[rb] = D::s_off(k, N) + j;
            out.d(rb) = 0.0;
            out.d_mask(rb) = 1.0;
            out.m(rb) = 0.0;
            out.d(rb + nb + ng) = 0.0;
            out.d_mask(rb + nb + ng) = 0.0;
            out.m(rb + nb + ng) = 0.0;
            ++rb;
        }
    }

    assert(rb == nb && rg == ng);
}

// =========================================================================
//  expand_full_sol
// =========================================================================

/// Expand a dense-QP solution back to the staged layout.  `staged` must be
/// the same staged QP that was condensed (the dynamics BA / b and the stage
/// Hessians / gradients / Jacobians are needed for the primal simulation and
/// the dynamics-multiplier recovery).
/// Expand a dense-QP solution back to the staged layout.
///
/// `include_b`: when true (the normal solution) the states are recovered by
/// the inhomogeneous forward simulation `x_{k+1} = BA [u_k; x_k] + b_k`.
/// For a sensitivity STEP of a converged QP pass false: the seeded KKT
/// enforces the homogeneous dynamics `x_{k+1} = BA [u_k; x_k]` (the seed has
/// no `b` field), so the affine offset must not be added.
template <class P, int NH>
void expand_full_sol(const Qp<P, NH>& staged, const DenseQp<P, NH>& dense,
                     const DenseQpSol<P, NH>& dsol, QpSol<P, NH>& sol,
                     bool include_b = true)
{
    using D = DenseQpDim<P>;
    using SD = QpDim<P>;
    using S = typename P::scalar_t;
    using M = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
    using V = Eigen::Matrix<S, Eigen::Dynamic, 1>;
    using namespace ocp::detail;

    const int N = dense.N;
    sol.resize(N);

    const int nx = D::nx;
    const int nu = D::nu;
    const auto& v = dsol.v;

    // ------------------------------------------------------------------
    // Primal: u_k / x_0 / slacks from v; states by forward simulation
    // (step semantics: x_{k+1} = BA_k (u_k; x_k) + b_k)
    // ------------------------------------------------------------------
    {
        auto& uxf = sol.ux_first;
        uxf.head(nu) = v.segment(D::u_off(0), nu);
        uxf.segment(nu, nx) = v.head(nx);
        if (SD::nslack_first > 0)
        {
            uxf.segment(nu + nx, SD::nslack_first) =
                v.segment(D::s_off(0, N), SD::nslack_first);
        }
    }
    V x = v.head(nx);  // x_0 (step)
    for (int k = 0; k < N; ++k)
    {
        M BA;
        V b;
        if (k == 0)
        {
            BA = staged.first.BA;
            b = staged.first.b;
        }
        else
        {
            BA = staged.path[k - 1].BA;
            b = staged.path[k - 1].b;
        }
        V uz(nu + nx);
        uz.head(nu) = v.segment(D::u_off(k), nu);
        uz.tail(nx) = x;
        x = BA * uz;
        if (include_b)
        {
            x += b;
        }
        if (k + 1 < N)
        {
            auto& uxp = sol.ux_path[k];
            uxp.head(nu) = v.segment(D::u_off(k + 1), nu);
            uxp.segment(nu, nx) = x;
            if (SD::nslack_path > 0)
            {
                uxp.segment(nu + nx, SD::nslack_path) =
                    v.segment(D::s_off(k + 1, N), SD::nslack_path);
            }
        }
        else
        {
            sol.ux_term.head(nx) = x;
            if (SD::nslack_term > 0)
            {
                sol.ux_term.tail(SD::nslack_term) =
                    v.segment(D::s_off(N, N), SD::nslack_term);
            }
        }
    }

    // ------------------------------------------------------------------
    // lam / t: dense sides -> staged sides (inverse of condense_full's
    // row order; see the DenseQpDim header)
    // ------------------------------------------------------------------
    const int nb = dense.nb;
    const int ng = dense.ng;
    const auto& dl = dsol.lam;
    const auto& dt = dsol.t;

    auto map_sides = [&](int rd, auto& lam_st, auto& t_st,
                         const QpLayout& lay, int r_st)
    {
        const int rh = rd + nb + ng;
        lam_st(lay.side_hi(r_st)) = dl(rh);
        t_st(lay.side_hi(r_st)) = dt(rh);
        const int sl = lay.side_lo(r_st);
        if (sl >= 0)
        {
            lam_st(sl) = dl(rd);
            t_st(sl) = dt(rd);
        }
    };

    // Dense box-row bases (condense_full's write order):
    //   [stage-0 box (pin | hard bx | hard bu)]
    //   [path hard bu, k = 1..N-1 (nbox_path per stage)]
    //   [slack rows k = 0..N]
    const int bu_base = D::nbox_first;
    const int slack_base = D::nb_total(N) - D::nslack_total(N);

    // Stage 0
    {
        const auto& lay = SD::lay_first;
        auto is_soft = [&](int r)
        {
            return SD::idxs_lo_first[r] >= 0 || SD::idxs_hi_first[r] >= 0;
        };
        const int n_pin = lay.rows[g_pin];
        // Pin rows: dense box rows 0 .. n_pin - 1
        for (int j = 0; j < n_pin; ++j)
        {
            map_sides(j, sol.lam_first, sol.t_first, lay, j);
        }
        // Hard bx / hard bu: dense box rows from n_pin
        int jb = n_pin;
        for (int j = 0; j < lay.rows[g_bx]; ++j)
        {
            const int r = lay.row_off(g_bx) + j;
            if (is_soft(r))
            {
                continue;
            }
            map_sides(jb++, sol.lam_first, sol.t_first, lay, r);
        }
        for (int j = 0; j < lay.rows[g_bu]; ++j)
        {
            const int r = lay.row_off(g_bu) + j;
            if (is_soft(r))
            {
                continue;
            }
            map_sides(jb++, sol.lam_first, sol.t_first, lay, r);
        }
        // General rows: soft bx | soft bu | ineq | eq | lin; dense gen
        // index = local index
        const int grps[5] = {g_bx, g_bu, g_ineq, g_eq, g_lin};
        int lg = 0;
        for (int gg = 0; gg < 5; ++gg)
        {
            const int grp = grps[gg];
            for (int j = 0; j < lay.rows[grp]; ++j)
            {
                const int r = lay.row_off(grp) + j;
                const bool boxish = (grp == g_bx || grp == g_bu);
                if (boxish && !is_soft(r))
                {
                    continue;
                }
                map_sides(nb + lg, sol.lam_first, sol.t_first, lay, r);
                ++lg;
            }
        }
        // Slack rows (stage 0)
        for (int j = 0; j < SD::nslack_first; ++j)
        {
            const int ss = lay.lo_size() + lay.nrow() + j;
            sol.lam_first(ss) = dl(slack_base + j);
            sol.t_first(ss) = dt(slack_base + j);
        }
    }

    // Path (k = 1..N-1) and terminal (k = N)
    auto map_stage = [&](auto& lam_st, auto& t_st, const QpLayout& lay, int k,
                         auto const& idxs_lo, auto const& idxs_hi)
    {
        // General rows: bx | soft bu | ineq | eq | lin
        const int grps[5] = {g_bx, g_bu, g_ineq, g_eq, g_lin};
        int lg = 0;
        for (int gg = 0; gg < 5; ++gg)
        {
            const int grp = grps[gg];
            for (int j = 0; j < lay.rows[grp]; ++j)
            {
                const int r = lay.row_off(grp) + j;
                if (grp == g_bu && idxs_lo[r] < 0 && idxs_hi[r] < 0)
                {
                    continue;
                }
                map_sides(nb + D::ngen_first + (k - 1) * D::ngen_path + lg,
                          lam_st, t_st, lay, r);
                ++lg;
            }
        }
        // Hard control box rows (terminal has none)
        int jb = 0;
        for (int j = 0; j < lay.rows[g_bu]; ++j)
        {
            const int r = lay.row_off(g_bu) + j;
            if (idxs_lo[r] >= 0 || idxs_hi[r] >= 0)
            {
                continue;
            }
            map_sides(bu_base + (k - 1) * D::nbox_path + jb, lam_st, t_st,
                      lay, r);
            ++jb;
        }
        // Slack rows (stage k)
        const int sbase = slack_base + (D::s_off(k, N) - D::s_base(N));
        for (int j = 0; j < lay.nslack; ++j)
        {
            const int ss = lay.lo_size() + lay.nrow() + j;
            lam_st(ss) = dl(sbase + j);
            t_st(ss) = dt(sbase + j);
        }
    };
    for (int k = 1; k <= N; ++k)
    {
        if (k == N)
        {
            map_stage(sol.lam_term, sol.t_term, SD::lay_term, N,
                      SD::idxs_lo_term, SD::idxs_hi_term);
        }
        else
        {
            map_stage(sol.lam_path[k - 1], sol.t_path[k - 1], SD::lay_path,
                      k, SD::idxs_lo_path, SD::idxs_hi_path);
        }
    }

    // ------------------------------------------------------------------
    // Dynamics multipliers: terminal seed + backward sweep
    //
    //   (res_g_k^{no-pi})_x = (H_k z_k + g_k)_x + DC_k,x'(lam_hi - lam_lo)
    //   (masked);  pi_{k-1} = (res_g_k^{no-pi})_x + A_k' pi_k
    // ------------------------------------------------------------------
    auto res_x_ux = [&](const auto& st, const V& z, const V& lam_m,
                        const QpLayout& lay, int nrow, int xoff)
    {
        V w(nrow);
        for (int r = 0; r < nrow; ++r)
        {
            const int sl = lay.side_lo(r);
            w(r) = lam_m(lay.side_hi(r))
                   - (sl >= 0 ? lam_m(sl) : 0.0);
        }
        const V Hz = st.hess * z;
        // .eval(): the returned expression must not reference the
        // lambda-local Hz / w (lifetime ends on return)
        return (Hz.segment(xoff, nx) + st.grad.segment(xoff, nx)
                + st.DC.block(0, xoff, nrow, nx).transpose() * w)
               .eval();
    };

    {
        const auto& st = staged.term;
        V z = sol.ux_term;
        const V lam_m = staged.term.d_mask.cwiseProduct(sol.lam_term);
        sol.pi[N - 1] =
            res_x_ux(st, z, lam_m, SD::lay_term, SD::nrow_term, 0);
    }
    for (int k = N - 1; k >= 1; --k)
    {
        const auto& st = staged.path[k - 1];
        V z = sol.ux_path[k - 1];
        const V lam_m = st.d_mask.cwiseProduct(sol.lam_path[k - 1]);
        sol.pi[k - 1] =
            res_x_ux(st, z, lam_m, SD::lay_path, SD::nrow_path, nu)
            + st.BA.block(0, nu, nx, nx).transpose() * sol.pi[k];
    }
}

}  // namespace ocp

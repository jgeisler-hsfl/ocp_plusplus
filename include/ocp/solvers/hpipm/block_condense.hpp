// ocp/solvers/hpipm/block_condense.hpp
//
// Partial (block) condensing: staged Qp -> staged block-Qp (parity plan
// Phase 2, block 2c, docs/plans/SQP_HPIPM_PARITY_PHASE2_PLAN.md).
//
// condense_block(in, out, NB):
//   Groups the staged QP's N stages into N2 = N / NB blocks of NB stages
//   each (block ii covers k0 = ii * NB ... k0 + NB - 1).  Within each block
//   the NB - 1 internal states x_{k0+1} ... x_{k0+NB-1} are eliminated by
//   substitution, producing a block stage with variable vector
//
//     (u_{k0}; ...; u_{k0+NB-1}; x_{k0}; s)
//
//   and block dynamics
//
//     x_{k0+NB} = BA_blk (u_blk; x_{k0}) + b_blk.
//
//   The block general rows are the original ineq / eq / lin rows plus the
//   internal-state box rows (which become two-sided general rows after
//   condensation), all flow-mapped; the block-start state box (x_{k0}) and
//   all control box rows stay box rows; a fixed x_0 keeps its pin rows in
//   the first block.  The slacks of the NB original stages are
//   concatenated into the block slack vector and keep a diagonal Hessian
//   block.  The original terminal stage is copied verbatim as the
//   block-QP terminal.
//
//   NB == 1 reproduces the staged QP exactly (N2 = N, one stage per block).
//
// expand_block_sol(staged, block, bsol, sol):
//   Inverse mapping for solutions (parity plan block 2d): u / s are sliced
//   from the block stage vectors, x_{k0} is read from the block, and the
//   internal states x_{k0+1} ... x_{k0+NB-1} are forward-simulated from
//   the staged dynamics.  lam / t are copied row group by row group
//   (inverse of condense_block's row-emit order; the flow map is a
//   feasible-set bijection, so multipliers and slacks transfer verbatim).
//   pi_blk[ii] is the staged dynamics multiplier of the block's last
//   stage (k0 + NB - 1) (the composite-constraint multiplier carries the
//   last stage of the chain); the internal pi's are recovered by a
//   per-stage stationarity backward sweep.
// expand_block_step(staged, bstep, sol)  (5h):
//   Inverse mapping for the sensitivity STEP of a converged block QP:
//   u / s are sliced from the block step vectors, dx_{k0} from the block,
//   and the internal state increments are forward-simulated HOMOGENEOUSLY
//   (no affine offset b: the seeded KKT enforces the step dynamics).
//   dlam / dt transfer verbatim (copy_block_duals).  bstep.pi[ii] is the
//   staged dynamics multiplier of the block's last stage (k0 + NB - 1);
//   the internal dpi's come from the LINEARIZED stationarity backward
//   sweep (no grad term; the solver-internal reg-primal correction is
//   O(reg) ~ 1e-15 and is omitted).

//
// HPIPM reference: acados/external/hpipm/cond/x_part_cond.c
// (PART_COND_QP_COND_LHS / _RHS, per-block COND_BAT / COND_RSQ / COND_DCT),
// reusing the Phase-1 flow-map machinery (condense.hpp).

#pragma once

#include <cassert>
#include <vector>

#include <Eigen/Dense>

#include "qp.hpp"

namespace ocp
{

/// Condense the staged QP `in` (horizon N, N % NB == 0) into the staged
/// block QP `out` (horizon N2 = N / NB), using BlockQpDim<P, NB> for the
/// block-stage structural dimensions.
template <class P, int NH, int NB>
void condense_block(const Qp<P, NH>& in,
                    Qp<P, Eigen::Dynamic, BlockQpDim<P, NB>>& out)
{
    using B = BlockQpDim<P, NB>;
    using SD = QpDim<P>;
    using S = typename P::scalar_t;
    using M = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
    using V = Eigen::Matrix<S, Eigen::Dynamic, 1>;
    using namespace ocp::detail;

    const int N = in.N;
    assert(NB >= 1);
    assert(N % NB == 0);
    const int N2 = N / NB;
    out.resize(N2);

    const int nx = SD::nx;
    const int nu = SD::nu;
    const int nvH = nx + NB * nu;  // hard vector, h-order [x_k0; u_blk]

    // ---------------------------------------------------------------
    // Per original stage: dynamic-Eigen view of the staged data (the
    // first / path stage types differ in size, so copy into dynamic
    // storage for uniform access).
    // ---------------------------------------------------------------
    struct Stage
    {
        M BA, hess, DC;
        V b, grad, d, d_mask, m;
        QpLayout lay;
        int nvar, nslack;
    };
    auto stage_at = [&](int k) -> Stage
    {
        Stage st;
        if (k == 0)
        {
            st.BA = in.first.BA;
            st.b = in.first.b;
            st.hess = in.first.hess;
            st.grad = in.first.grad;
            st.DC = in.first.DC;
            st.d = in.first.d;
            st.d_mask = in.first.d_mask;
            st.m = in.first.m;
            st.lay = SD::lay_first;
            st.nvar = SD::nvar_first;
            st.nslack = SD::nslack_first;
        }
        else
        {
            st.BA = in.path[k - 1].BA;
            st.b = in.path[k - 1].b;
            st.hess = in.path[k - 1].hess;
            st.grad = in.path[k - 1].grad;
            st.DC = in.path[k - 1].DC;
            st.d = in.path[k - 1].d;
            st.d_mask = in.path[k - 1].d_mask;
            st.m = in.path[k - 1].m;
            st.lay = SD::lay_path;
            st.nvar = SD::nvar_path;
            st.nslack = SD::nslack_path;
        }
        return st;
    };

    // Permute a hard-vector coefficient from h-order [x_k0; u_blk] to the
    // block variable order (u_blk; x_k0).
    auto permute_hard = [&](const V& hord)
    {
        V bo(nvH);
        bo.segment(0, NB * nu) = hord.segment(nx, NB * nu);
        bo.segment(NB * nu, nx) = hord.segment(0, nx);
        return bo;
    };

    for (int ii = 0; ii < N2; ++ii)
    {
        const int k0 = ii * NB;
        const bool isFirst = (ii == 0);

        std::vector<Stage> st(NB);
        for (int j = 0; j < NB; ++j)
        {
            st[j] = stage_at(k0 + j);
        }

        // -----------------------------------------------------------
        // Flow map: x_{k0+j} = G[j]' (x_k0; u_{k0}; ...; u_{k0+j-1})
        //             + gamma[j],  j = 0..NB
        // (G[j] stores G[j](v, d) = dx_{k0+j}[d] / dv; only the first
        //  nx + j*nu rows are nonzero)
        // -----------------------------------------------------------
        std::vector<M> G(NB + 1, M::Zero(nvH, nx));
        std::vector<V> gamma(NB + 1, V::Zero(nx));
        G[0].topLeftCorner(nx, nx).setIdentity();
        for (int j = 0; j < NB; ++j)
        {
            const M A = st[j].BA.block(0, nu, nx, nx);
            const M Bm = st[j].BA.block(0, 0, nx, nu);
            const V b = st[j].b;
            G[j + 1].topLeftCorner(nx, nx) =
                G[j].topLeftCorner(nx, nx) * A.transpose();
            if (j > 0)
            {
                G[j + 1].block(nx, 0, j * nu, nx) =
                    G[j].block(nx, 0, j * nu, nx) * A.transpose();
            }
            G[j + 1].block(nx + j * nu, 0, nu, nx) = Bm.transpose();
            gamma[j + 1] = A * gamma[j] + b;
        }

        auto fill = [&](auto& blk, const QpLayout& blay)
        {
            blk.setZero();

            // -------------------------------------------------------
            // Block dynamics: x_{k0+NB} = [B_blk | A_blk] (u_blk; x_k0)
            //   + b_blk.  A_blk = G[NB](x part), B_j = G[NB](u_j part).
            // -------------------------------------------------------
            blk.b = gamma[NB];
            blk.BA.block(0, NB * nu, nx, nx) =
                G[NB].topLeftCorner(nx, nx).transpose();
            for (int j = 0; j < NB; ++j)
            {
                blk.BA.block(0, j * nu, nx, nu) =
                    G[NB].block(nx + j * nu, 0, nu, nx).transpose();
            }

            // -------------------------------------------------------
            // Hessian / gradient over (u_blk; x_k0): flow-map
            // substitution, accumulated in h-order [x_k0; u_blk] and
            // permuted to the block order (u_blk; x_k0).
            // -------------------------------------------------------
            M Hh = M::Zero(nvH, nvH);
            V gh(nvH);
            gh.setZero();
            for (int j = 0; j < NB; ++j)
            {
                const M H_uu = st[j].hess.topLeftCorner(nu, nu);
                const M H_ux = st[j].hess.block(0, nu, nu, nx);
                const M H_xx = st[j].hess.block(nu, nu, nx, nx);
                const V gk_u = st[j].grad.head(nu);
                const V gk_x = st[j].grad.segment(nu, nx);

                const int u_off = nx + j * nu;
                const int nact = nx + j * nu;
                const M Gk_a = G[j].topLeftCorner(nact, nx);
                const V& gm = gamma[j];

                Hh.block(u_off, u_off, nu, nu).noalias() += H_uu;
                const M xu = H_ux * Gk_a.transpose();  // (nu x nact)
                Hh.block(u_off, 0, nu, nact).noalias() += xu;
                Hh.block(0, u_off, nact, nu).noalias() += xu.transpose();
                Hh.topLeftCorner(nact, nact).noalias() +=
                    Gk_a * (H_xx * Gk_a.transpose());

                gh.segment(u_off, nu).noalias() += H_ux * gm + gk_u;
                gh.head(nact).noalias() += Gk_a * (H_xx * gm + gk_x);
            }
            blk.hess.block(0, 0, NB * nu, NB * nu) =
                Hh.block(nx, nx, NB * nu, NB * nu);
            blk.hess.block(0, NB * nu, NB * nu, nx) =
                Hh.block(nx, 0, NB * nu, nx);
            blk.hess.block(NB * nu, 0, nx, NB * nu) =
                Hh.block(0, nx, nx, NB * nu);
            blk.hess.block(NB * nu, NB * nu, nx, nx) =
                Hh.block(0, 0, nx, nx);
            blk.grad.segment(0, NB * nu) = gh.segment(nx, NB * nu);
            blk.grad.segment(NB * nu, nx) = gh.segment(0, nx);

            // Slack diagonal Hessian + linear part (per original stage).
            for (int j = 0; j < NB; ++j)
            {
                const int nsl = st[j].nslack;
                const int sblk = NB * nu + nx
                                 + (isFirst ? B::stage_offset_first(j)
                                            : B::stage_offset_path(j));
                for (int i = 0; i < nsl; ++i)
                {
                    blk.hess(sblk + i, sblk + i) =
                        st[j].hess(st[j].nvar - nsl + i,
                                   st[j].nvar - nsl + i);
                    blk.grad(sblk + i) = st[j].grad(st[j].nvar - nsl + i);
                }
            }

            // Slack-side (non-negativity) entries: carried over from the
            // original stages (d = 0, d_mask = 1 per the driver convention).
            {
                const int side_off = blay.lo_size() + blay.nrow();
                for (int j = 0; j < NB; ++j)
                {
                    const int nsl = st[j].nslack;
                    const int sb = isFirst ? B::stage_offset_first(j)
                                           : B::stage_offset_path(j);
                    const int slo0 = st[j].lay.lo_size() + st[j].lay.nrow();
                    for (int i = 0; i < nsl; ++i)
                    {
                        blk.d(side_off + sb + i) = st[j].d(slo0 + i);
                        blk.d_mask(side_off + sb + i) = st[j].d_mask(slo0 + i);
                        blk.m(side_off + sb + i) = st[j].m(slo0 + i);
                    }
                }
            }

            // -------------------------------------------------------
            // DC rows + d / d_mask / m (iterate original rows and map
            // them onto the block row layout).
            // -------------------------------------------------------
            auto fill_s_part = [&](int br, int j, int r, const Stage& sj)
            {
                const int nsl = sj.nslack;
                const int sblk = NB * nu + nx
                                 + (isFirst ? B::stage_offset_first(j)
                                            : B::stage_offset_path(j));
                for (int i = 0; i < nsl; ++i)
                {
                    blk.DC(br, sblk + i) = sj.DC(r, nu + nx + i);
                }
            };

            // Flow-mapped general row of stage j (k = k0 + j):
            // x_k = G[j]' h + gamma[j]; the row's x-part is substituted,
            // its u-part stays on the j-th control, d shifts by
            // dcx * gamma[j] (Phase-1 convention).
            auto emit_flow = [&](int br, int j, int r, const Stage& sj)
            {
                const int nact = nx + j * nu;
                V dcx(nx);
                for (int c = 0; c < nx; ++c)
                {
                    dcx(c) = sj.DC(r, nu + c);
                }
                V hord(nvH);
                hord.setZero();
                hord.head(nact) = G[j].topLeftCorner(nact, nx) * dcx;
                for (int c = 0; c < nu; ++c)
                {
                    hord(nx + j * nu + c) = sj.DC(r, c);  // u_k part
                }
                const V bo = permute_hard(hord);
                for (int c = 0; c < nvH; ++c)
                {
                    blk.DC(br, c) = bo(c);
                }
                fill_s_part(br, j, r, sj);
                const S sh = dcx.dot(gamma[j]);
                const int olo = sj.lay.side_lo(r);
                const int ohi = sj.lay.side_hi(r);
                const int blo = blay.side_lo(br);
                const int bhi = blay.side_hi(br);
                if (olo >= 0 && blo >= 0)
                {
                    blk.d(blo) = sj.d(olo) - sh;
                    blk.d_mask(blo) = sj.d_mask(olo);
                    blk.m(blo) = sj.m(olo);
                }
                if (bhi >= 0)
                {
                    blk.d(bhi) = sj.d(ohi) + sh;
                    blk.d_mask(bhi) = sj.d_mask(ohi);
                    blk.m(bhi) = sj.m(ohi);
                }
            };

            // Direct (non flow-mapped) row: unit vector on the
            // constrained hard variable; d / d_mask / m copied as-is.
            auto emit_direct = [&](int br, int hcol, int j, int r,
                                   const Stage& sj)
            {
                blk.DC(br, hcol) = S(1.0);
                fill_s_part(br, j, r, sj);
                const int olo = sj.lay.side_lo(r);
                const int ohi = sj.lay.side_hi(r);
                const int blo = blay.side_lo(br);
                const int bhi = blay.side_hi(br);
                if (olo >= 0 && blo >= 0)
                {
                    blk.d(blo) = sj.d(olo);
                    blk.d_mask(blo) = sj.d_mask(olo);
                    blk.m(blo) = sj.m(olo);
                }
                if (bhi >= 0)
                {
                    blk.d(bhi) = sj.d(ohi);
                    blk.d_mask(bhi) = sj.d_mask(ohi);
                    blk.m(bhi) = sj.m(ohi);
                }
            };

            // Pin rows (first block, fixed x_0 only): x_k0 unit select.
            if (isFirst && SD::fixed_initial_state)
            {
                const Stage& s0 = st[0];
                for (int i = 0; i < s0.lay.rows[g_pin]; ++i)
                {
                    const int r = s0.lay.row_off(g_pin) + i;
                    const int br = blay.row_off(g_pin) + i;
                    emit_direct(br, NB * nu + i, 0, r, s0);
                }
            }

            for (int j = 0; j < NB; ++j)
            {
                const Stage& sj = st[j];
                // Control box rows (direct, u_{k0+j} select)
                for (int i = 0; i < sj.lay.rows[g_bu]; ++i)
                {
                    const int r = sj.lay.row_off(g_bu) + i;
                    const int br = blay.row_off(g_bu) + j * SD::nbu + i;
                    emit_direct(br, j * nu + P::control_box_idx[i], j, r,
                                sj);
                }
                // State box rows: j == 0 -> block-start state box (direct,
                // x_k0 select); j > 0 -> internal state box (flow-mapped,
                // in the g_lin group after the NB * nl linear rows)
                for (int i = 0; i < sj.lay.rows[g_bx]; ++i)
                {
                    const int r = sj.lay.row_off(g_bx) + i;
                    if (j == 0)
                    {
                        const int br = blay.row_off(g_bx) + i;
                        emit_direct(br, NB * nu + P::state_box_idx[i], j, r,
                                    sj);
                    }
                    else
                    {
                        const int br = blay.row_off(g_lin) + NB * SD::nl
                                       + (j - 1) * SD::nbx + i;
                        emit_flow(br, j, r, sj);
                    }
                }
                // ineq / eq / lin rows (flow-mapped)
                for (int i = 0; i < sj.lay.rows[g_ineq]; ++i)
                {
                    const int r = sj.lay.row_off(g_ineq) + i;
                    const int br = blay.row_off(g_ineq) + j * SD::ng + i;
                    emit_flow(br, j, r, sj);
                }
                for (int i = 0; i < sj.lay.rows[g_eq]; ++i)
                {
                    const int r = sj.lay.row_off(g_eq) + i;
                    const int br = blay.row_off(g_eq) + j * SD::ne + i;
                    emit_flow(br, j, r, sj);
                }
                for (int i = 0; i < sj.lay.rows[g_lin]; ++i)
                {
                    const int r = sj.lay.row_off(g_lin) + i;
                    const int br = blay.row_off(g_lin) + j * SD::nl + i;
                    emit_flow(br, j, r, sj);
                }
            }
        };

        if (isFirst)
        {
            fill(out.first, B::lay_first);
        }
        else
        {
            fill(out.path[ii - 1], B::lay_path);
        }
    }

    // Terminal: verbatim copy (BlockQpDim terminal dims == QpDim)
    out.term.hess = in.term.hess;
    out.term.grad = in.term.grad;
    out.term.DC = in.term.DC;
    out.term.d = in.term.d;
    out.term.d_mask = in.term.d_mask;
    out.term.m = in.term.m;
}

// =========================================================================
//  copy_block_duals
// =========================================================================

/// Copy the block lam / t (side multipliers and slacks) into the staged
/// layout, row group by row group (inverse of condense_block's row-emit
/// order; the flow map is a feasible-set bijection, so multipliers and
/// slacks transfer verbatim).  Shared by `expand_block_sol` (absolute
/// solution) and `expand_block_step` (sensitivity step: dlam / dt).
template <class B, class SD, int NB, class BsolT, class SolT>
void copy_block_duals(const BsolT& bsol, SolT& sol)
{
    using namespace ocp::detail;
    const int N2 = bsol.N;

    auto map_row = [&](int br, const QpLayout& blay, auto& lam_b, auto& t_b,
                       int k, int r_st)
    {
        const int bh = blay.side_hi(br);
        const int bl = blay.side_lo(br);
        if (k == 0)
        {
            const QpLayout& lay = SD::lay_first;
            auto& lam_st = sol.lam_first;
            auto& t_st = sol.t_first;
            lam_st(lay.side_hi(r_st)) = lam_b(bh);
            t_st(lay.side_hi(r_st)) = t_b(bh);
            if (bl >= 0)
            {
                lam_st(lay.side_lo(r_st)) = lam_b(bl);
                t_st(lay.side_lo(r_st)) = t_b(bl);
            }
        }
        else
        {
            const QpLayout& lay = SD::lay_path;
            auto& lam_st = sol.lam_path[k - 1];
            auto& t_st = sol.t_path[k - 1];
            lam_st(lay.side_hi(r_st)) = lam_b(bh);
            t_st(lay.side_hi(r_st)) = t_b(bh);
            if (bl >= 0)
            {
                lam_st(lay.side_lo(r_st)) = lam_b(bl);
                t_st(lay.side_lo(r_st)) = t_b(bl);
            }
        }
    };

    auto map_rows_of = [&](const QpLayout& blay, auto& lam_b, auto& t_b,
                           int k0)
    {
        for (int br = 0; br < blay.nrow(); ++br)
        {
            const int g = blay.group_of(br);
            const int loc = br - blay.row_off(g);
            int jj = 0, g_st = g, local = loc;
            switch (g)
            {
            case g_bu:
                jj = loc / SD::nbu;
                local = loc % SD::nbu;
                break;
            case g_ineq:
                jj = loc / SD::ng;
                local = loc % SD::ng;
                break;
            case g_eq:
                jj = loc / SD::ne;
                local = loc % SD::ne;
                break;
            case g_lin:
                if (loc < NB * SD::nl)
                {
                    jj = loc / SD::nl;
                    local = loc % SD::nl;
                }
                else
                {
                    // internal state box row -> staged state-box row
                    const int lb = loc - NB * SD::nl;
                    jj = lb / SD::nbx + 1;
                    g_st = g_bx;
                    local = lb % SD::nbx;
                }
                break;
            default:  // g_pin / g_bx: block-start row, stage j = 0
                break;
            }
            // r_st must be the ABSOLUTE row index in the staged stage
            // layout (row_off depends on the groups present in that type)
            const int k = k0 + jj;
            const QpLayout& lay_st = (k == 0) ? SD::lay_first
                                              : SD::lay_path;
            const int r_st = lay_st.row_off(g_st) + local;
            map_row(br, blay, lam_b, t_b, k, r_st);
        }
    };

    // Slack-side entries (non-negativity sides of the slack variables)
    auto copy_slack_sides = [&](const QpLayout& blay, auto& lam_b, auto& t_b,
                                int k0)
    {
        const int bso = blay.lo_size() + blay.nrow();
        for (int j = 0; j < NB; ++j)
        {
            const int k = k0 + j;
            const int nsl = (k == 0) ? SD::nslack_first : SD::nslack_path;
            if (nsl == 0)
            {
                continue;
            }
            const int soff = (k0 == 0) ? B::stage_offset_first(j)
                                       : B::stage_offset_path(j);
            const QpLayout& lay_st = (k == 0) ? SD::lay_first
                                              : SD::lay_path;
            const int sso = lay_st.lo_size() + lay_st.nrow();
            for (int i = 0; i < nsl; ++i)
            {
                if (k == 0)
                {
                    sol.lam_first(sso + i) = lam_b(bso + soff + i);
                    sol.t_first(sso + i) = t_b(bso + soff + i);
                }
                else
                {
                    sol.lam_path[k - 1](sso + i) = lam_b(bso + soff + i);
                    sol.t_path[k - 1](sso + i) = t_b(bso + soff + i);
                }
            }
        }
    };

    for (int ii = 0; ii < N2; ++ii)
    {
        const int k0 = ii * NB;
        if (ii == 0)
        {
            map_rows_of(B::lay_first, bsol.lam_first, bsol.t_first, k0);
            copy_slack_sides(B::lay_first, bsol.lam_first, bsol.t_first, k0);
        }
        else
        {
            map_rows_of(B::lay_path, bsol.lam_path[ii - 1],
                        bsol.t_path[ii - 1], k0);
            copy_slack_sides(B::lay_path, bsol.lam_path[ii - 1],
                             bsol.t_path[ii - 1], k0);
        }
    }
    sol.lam_term = bsol.lam_term;
    sol.t_term = bsol.t_term;
}

// =========================================================================
//  expand_block_sol
// =========================================================================

/// Expand a block-condensed-QP solution back into the staged `QpSol`
/// layout.  `staged` must be the staged QP that was passed to
/// `condense_block` (its dynamics BA / b and the per-stage Hessians /
/// gradients / Jacobians drive the primal forward simulation and the
/// dynamics-multiplier recovery); `block` / `bsol` must be the block QP
/// and the solution of it.
///
/// Primal: the controls and slacks are read directly from the block stage
/// vectors; x_{k0} comes from the block; the internal states
/// x_{k0+1} ... x_{k0+NB-1} are forward-simulated from the staged
/// dynamics.
///
/// Dual: lam / t are copied row group by row group (inverse of
/// condense_block's row-emit order; the flow map is a feasible-set
/// bijection, so multipliers and slacks transfer verbatim).  The block
/// dynamics multiplier pi_blk[ii] is the staged multiplier of the
/// block's last stage (k0 + NB - 1) (the composite-constraint multiplier
/// is the multiplier of the last stage of the chain); the internal
/// multipliers pi_{k0} ... pi_{k0+NB-2} come from a per-stage
/// stationarity backward sweep (same sweep as expand_full_sol).
///
/// NB == 1 is a trivial mapping (N2 = N).
template <class P, int NH, int NH2, int NB>
void expand_block_sol(const Qp<P, NH>& staged,
                      const Qp<P, NH2, BlockQpDim<P, NB>>& block,
                      const QpSol<P, NH2, BlockQpDim<P, NB>>& bsol,
                      QpSol<P, NH>& sol)
{
    using B = BlockQpDim<P, NB>;
    using SD = QpDim<P>;
    using S = typename P::scalar_t;
    using M = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
    using V = Eigen::Matrix<S, Eigen::Dynamic, 1>;
    using namespace ocp::detail;

    const int N = staged.N;
    assert(NB >= 1);
    assert(N % NB == 0);
    const int N2 = block.N;
    assert(N2 == N / NB && bsol.N == N2);
    sol.resize(N);

    const int nx = SD::nx;
    const int nu = SD::nu;

    // ---------------------------------------------------------------
    // Per original stage: dynamic-Eigen view of the staged data (first /
    // path stage types differ in size).
    // ---------------------------------------------------------------
    struct Stage
    {
        M BA, hess, DC;
        V b, grad, d_mask;
        QpLayout lay;
    };
    auto stage_at = [&](int k) -> Stage
    {
        Stage s;
        if (k == 0)
        {
            s.BA = staged.first.BA;
            s.b = staged.first.b;
            s.hess = staged.first.hess;
            s.grad = staged.first.grad;
            s.DC = staged.first.DC;
            s.d_mask = staged.first.d_mask;
            s.lay = SD::lay_first;
        }
        else
        {
            s.BA = staged.path[k - 1].BA;
            s.b = staged.path[k - 1].b;
            s.hess = staged.path[k - 1].hess;
            s.grad = staged.path[k - 1].grad;
            s.DC = staged.path[k - 1].DC;
            s.d_mask = staged.path[k - 1].d_mask;
            s.lay = SD::lay_path;
        }
        return s;
    };

    // ---------------------------------------------------------------
    // Primal: u / s sliced from the block vectors; internal states by
    // forward simulation of the staged dynamics.
    // ---------------------------------------------------------------
    for (int ii = 0; ii < N2; ++ii)
    {
        const int k0 = ii * NB;
        const int nvar = (ii == 0) ? B::nvar_first : B::nvar_path;
        V ub(nvar);
        if (ii == 0)
        {
            ub = bsol.ux_first;
        }
        else
        {
            ub = bsol.ux_path[ii - 1];
        }

        std::vector<V> xs(NB + 1);
        xs[0] = ub.segment(NB * nu, nx);
        for (int j = 0; j < NB; ++j)
        {
            const Stage sj = stage_at(k0 + j);
            V uz(nu + nx);
            uz.head(nu) = ub.segment(j * nu, nu);
            uz.tail(nx) = xs[j];
            xs[j + 1] = sj.BA * uz + sj.b;
        }

        for (int j = 0; j < NB; ++j)
        {
            const int k = k0 + j;
            const int nsl = (k == 0) ? SD::nslack_first : SD::nslack_path;
            const int soff = (ii == 0) ? B::stage_offset_first(j)
                                       : B::stage_offset_path(j);
            if (k == 0)
            {
                auto& uxf = sol.ux_first;
                uxf.head(nu) = ub.segment(j * nu, nu);
                uxf.segment(nu, nx) = xs[j];
                if (nsl > 0)
                {
                    uxf.segment(nu + nx, nsl) =
                        ub.segment(NB * nu + nx + soff, nsl);
                }
            }
            else
            {
                auto& uxp = sol.ux_path[k - 1];
                uxp.head(nu) = ub.segment(j * nu, nu);
                uxp.segment(nu, nx) = xs[j];
                if (nsl > 0)
                {
                    uxp.segment(nu + nx, nsl) =
                        ub.segment(NB * nu + nx + soff, nsl);
                }
            }
        }
    }
    sol.ux_term = bsol.ux_term;

    // Dual: dlam / dt row group by row group (verbatim;
    // copy_block_duals, 5h).
    copy_block_duals<B, SD, NB>(bsol, sol);

    // ---------------------------------------------------------------
    // Dynamics multipliers: pi_blk[ii] = pi_{k0 + NB - 1} (the
    // composite-constraint multiplier is the multiplier of the last
    // stage of the chain); internal pi's by the per-stage stationarity
    // backward sweep.
    // ---------------------------------------------------------------
    auto res_x_ux = [&](const Stage& s, const V& z, const V& lam_m, int nrow)
    {
        V w(nrow);
        for (int r = 0; r < nrow; ++r)
        {
            const int sl = s.lay.side_lo(r);
            w(r) = lam_m(s.lay.side_hi(r))
                   - (sl >= 0 ? lam_m(sl) : 0.0);
        }
        const V Hz = s.hess * z;
        // .eval(): the returned expression must not reference the
        // lambda-local Hz / w (lifetime ends on return)
        return (Hz.segment(nu, nx) + s.grad.segment(nu, nx)
                + s.DC.block(0, nu, nrow, nx).transpose() * w)
               .eval();
    };

    for (int ii = 0; ii < N2; ++ii)
    {
        const int k0 = ii * NB;
        const int klast = k0 + NB - 1;
        sol.pi[klast] = bsol.pi[ii];
        for (int k = klast; k >= k0 + 1; --k)
        {
            const Stage s = stage_at(k);
            const V z = sol.ux_path[k - 1];
            const V lam_m = s.d_mask.cwiseProduct(sol.lam_path[k - 1]);
            sol.pi[k - 1] =
                res_x_ux(s, z, lam_m, SD::nrow_path)
                + s.BA.block(0, nu, nx, nx).transpose() * sol.pi[k];
        }
    }
}
// =========================================================================
//  expand_block_step
// =========================================================================

/// Expand a block-condensed-QP sensitivity STEP (the linearized-KKT step
/// dv / dt / dlam / dpi of a converged block QP, as returned by
/// `HpipmQpSolver<..., BlockQpDim>::sens_forward` / `sens_adj`) back into
/// the staged `QpSol` layout.  `staged` must be the staged QP whose LHS was
/// condensed (its BA / hess / DC drive the primal forward simulation and
/// the dynamics-multiplier sweep); for parameter sensitivities this is the
/// SEED QP (same LHS as the base, whose `b` field is the p-derivative of
/// the dynamics), mirroring `expand_full_sol(..., include_b = true)`.
///
/// Primal: the controls and slacks are read directly from the block step
/// vectors; dx_{k0} comes from the block; the internal state increments
/// dx_{k0+1} ... dx_{k0+NB-1} are forward-simulated from the step
/// dynamics.  With `include_b` the affine seed offset is carried,
/// `dx_{k+1} = BA [du_k; dx_k] + b_k` (the seed `b` = d(b_k)/dp of the
/// condensed stage QP); with `include_b == false` the simulation is
/// homogeneous (the x_0 sensitivity path, whose seed carries no dynamics
/// offset — the dx_0 perturbation enters through the pin rows).
///
/// Dual: dlam / dt are copied row group by row group (copy_block_duals;
/// the flow map is a feasible-set bijection, so the step transfers
/// verbatim).  The block dynamics multiplier bstep.pi[ii] is the staged
/// multiplier of the block's last stage (k0 + NB - 1); the internal
/// multipliers dpi_{k0} ... dpi_{k0+NB-2} come from the LINEARIZED
/// stationarity backward sweep (no grad term; the solver-internal
/// reg-primal correction is O(reg) ~ 1e-15 and is omitted):
///   dpi_{k-1} = (H_k dz_k)_x + DC_k,x' (dlam_hi - dlam_lo) + A_k' dpi_k
///
/// NB == 1 is a trivial mapping (N2 = N).
template <class P, int NH, int NH2, int NB>
void expand_block_step(const Qp<P, NH>& staged,
                       const QpSol<P, NH2, BlockQpDim<P, NB>>& bstep,
                       QpSol<P, NH>& sol, bool include_b = false)
{
    using B = BlockQpDim<P, NB>;
    using SD = QpDim<P>;
    using S = typename P::scalar_t;
    using M = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
    using V = Eigen::Matrix<S, Eigen::Dynamic, 1>;
    using namespace ocp::detail;

    const int N = staged.N;
    assert(NB >= 1);
    assert(N % NB == 0);
    const int N2 = bstep.N;
    assert(N2 == N / NB);
    sol.resize(N);

    const int nx = SD::nx;
    const int nu = SD::nu;

    // Per original stage: dynamic-Eigen view of the staged data (first /
    // path stage types differ in size).
    struct Stage
    {
        M BA, hess, DC;
        V b, d_mask;
        QpLayout lay;
    };
    auto stage_at = [&](int k) -> Stage
    {
        Stage s;
        if (k == 0)
        {
            s.BA = staged.first.BA;
            s.hess = staged.first.hess;
            s.DC = staged.first.DC;
            s.b = staged.first.b;
            s.d_mask = staged.first.d_mask;
            s.lay = SD::lay_first;
        }
        else
        {
            s.BA = staged.path[k - 1].BA;
            s.hess = staged.path[k - 1].hess;
            s.DC = staged.path[k - 1].DC;
            s.b = staged.path[k - 1].b;
            s.d_mask = staged.path[k - 1].d_mask;
            s.lay = SD::lay_path;
        }
        return s;
    };

    // ---------------------------------------------------------------
    // Primal: du / ds sliced from the block vectors; internal dx by
    // forward simulation of the step dynamics (with the seed b offset
    // when include_b).
    // ---------------------------------------------------------------
    for (int ii = 0; ii < N2; ++ii)
    {
        const int k0 = ii * NB;
        const int nvar = (ii == 0) ? B::nvar_first : B::nvar_path;
        V ub(nvar);
        if (ii == 0)
        {
            ub = bstep.ux_first;
        }
        else
        {
            ub = bstep.ux_path[ii - 1];
        }

        std::vector<V> xs(NB + 1);
        xs[0] = ub.segment(NB * nu, nx);
        for (int j = 0; j < NB; ++j)
        {
            const Stage sj = stage_at(k0 + j);
            V uz(nu + nx);
            uz.head(nu) = ub.segment(j * nu, nu);
            uz.tail(nx) = xs[j];
            xs[j + 1] = sj.BA * uz;
            if (include_b)
            {
                xs[j + 1] += sj.b;
            }
        }

        for (int j = 0; j < NB; ++j)
        {
            const int k = k0 + j;
            const int nsl = (k == 0) ? SD::nslack_first : SD::nslack_path;
            const int soff = (ii == 0) ? B::stage_offset_first(j)
                                        : B::stage_offset_path(j);
            if (k == 0)
            {
                auto& uxf = sol.ux_first;
                uxf.head(nu) = ub.segment(j * nu, nu);
                uxf.segment(nu, nx) = xs[j];
                if (nsl > 0)
                {
                    uxf.segment(nu + nx, nsl) =
                        ub.segment(NB * nu + nx + soff, nsl);
                }
            }
            else
            {
                auto& uxp = sol.ux_path[k - 1];
                uxp.head(nu) = ub.segment(j * nu, nu);
                uxp.segment(nu, nx) = xs[j];
                if (nsl > 0)
                {
                    uxp.segment(nu + nx, nsl) =
                        ub.segment(NB * nu + nx + soff, nsl);
                }
            }
        }
    }
    sol.ux_term = bstep.ux_term;

    // Dual: dlam / dt row group by row group (verbatim).
    copy_block_duals<B, SD, NB>(bstep, sol);

    // ---------------------------------------------------------------
    // Dynamics multipliers: pi_blk[ii] = dpi_{k0 + NB - 1}; internal
    // dpi's by the linearized stationarity backward sweep.
    // ---------------------------------------------------------------
    auto step_res_x = [&](const Stage& s, const V& z, const V& lam_m,
                          int nrow)
    {
        V w(nrow);
        for (int r = 0; r < nrow; ++r)
        {
            const int sl = s.lay.side_lo(r);
            w(r) = lam_m(s.lay.side_hi(r))
                   - (sl >= 0 ? lam_m(sl) : 0.0);
        }
        const V Hz = s.hess * z;
        // .eval(): the returned expression must not reference the
        // lambda-local Hz / w (lifetime ends on return)
        return (Hz.segment(nu, nx)
                + s.DC.block(0, nu, nrow, nx).transpose() * w)
               .eval();
    };

    for (int ii = 0; ii < N2; ++ii)
    {
        const int k0 = ii * NB;
        const int klast = k0 + NB - 1;
        sol.pi[klast] = bstep.pi[ii];
        for (int k = klast; k >= k0 + 1; --k)
        {
            const Stage s = stage_at(k);
            const V z = sol.ux_path[k - 1];
            const V lam_m = s.d_mask.cwiseProduct(sol.lam_path[k - 1]);
            sol.pi[k - 1] =
                step_res_x(s, z, lam_m, SD::nrow_path)
                + s.BA.block(0, nu, nx, nx).transpose() * sol.pi[k];
        }
    }
}

}  // namespace ocp

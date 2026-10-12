// Phase 2 (parity) block 2c test: condense_block (partial / block
// condensing, docs/plans/SQP_HPIPM_PARITY_PHASE2_PLAN.md).
//
// Part 1 (NB = 1 identity): the block QP (N2 = N) must reproduce the
// staged QP exactly, field by field (trivial flow map, no d shift).
//
// Part 2 (NB > 1, data-level gates): every check evaluates the block QP
// against a forward simulation of the original staged dynamics (no code
// shared with condense_block):
//   (a) block dynamics: x_{k0+NB} == BA_blk (u_blk; x_k0) + b_blk
//   (b) objective: the block quadratic form equals the sum of the staged
//       stage costs at the forward-simulated states
//   (c) constraint rows: blk.DC(r) z + blk.d(side) ==
//       staged.DC(r') zk + staged.d(side') for every block row (the
//       flow-map d-shift cancels); d_mask / m copied verbatim
//   (d) terminal stage copied verbatim
//
// Part 3 (end-to-end): staged solve vs block solve — same primal (u, x at
// block starts, x_N, slacks) and objective; for N2 = 1 also against the
// Phase-1 full-dense pipeline.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "ocp/solvers/hpipm/block_condense.hpp"
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
    if (a.size() == 0)
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

// ---------------------------------------------------------------------
// Synthetic problems (copied from condense_full.cpp, soft rows enabled)
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
// (copied from condense_full.cpp)
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

        // d / d_mask / m per side layout.  General (ineq/eq/lin) lo sides
        // are masked (~30 %) to exercise the existing-but-inactive path.
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
// Make the random QP of fill_random_qp actually solvable:
//  - zero the complementarity target (production QPs have m = 0)
//  - widen every active bound so that the zero-input trajectory
//    (x_0 = 0, u_k = 0, s_k = 0) is strictly interior
// (copied from condense_full.cpp)
// ---------------------------------------------------------------------

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
        Eigen::MatrixXd BA;
        Eigen::VectorXd b;
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
// Part 1: NB = 1 identity — block QP == staged QP, field by field
// ---------------------------------------------------------------------

template <class P, int NH>
void check_nb1_identity(const char* name, int N)
{
    const std::string p = name;
    Qp<P, NH> qp(N);
    Rng rng;
    fill_random_qp(qp, rng);

    Qp<P, Eigen::Dynamic, BlockQpDim<P, 1>> bq;
    condense_block<P, NH, 1>(qp, bq);
    check(bq.N == N, p + " horizon");

    check_close(bq.first.hess, qp.first.hess, 0.0, p + " first.hess");
    check_close(bq.first.grad, qp.first.grad, 0.0, p + " first.grad");
    check_close(bq.first.BA, qp.first.BA, 0.0, p + " first.BA");
    check_close(bq.first.b, qp.first.b, 0.0, p + " first.b");
    check_close(bq.first.DC, qp.first.DC, 0.0, p + " first.DC");
    check_close(bq.first.d, qp.first.d, 0.0, p + " first.d");
    check_close(bq.first.d_mask, qp.first.d_mask, 0.0, p + " first.d_mask");
    check_close(bq.first.m, qp.first.m, 0.0, p + " first.m");
    for (int k = 1; k < N; ++k)
    {
        const auto& bf = bq.path[k - 1];
        const auto& sf = qp.path[k - 1];
        check_close(bf.hess, sf.hess, 0.0, p + " path.hess[" + std::to_string(k) + "]");
        check_close(bf.grad, sf.grad, 0.0, p + " path.grad[" + std::to_string(k) + "]");
        check_close(bf.BA, sf.BA, 0.0, p + " path.BA[" + std::to_string(k) + "]");
        check_close(bf.b, sf.b, 0.0, p + " path.b[" + std::to_string(k) + "]");
        check_close(bf.DC, sf.DC, 0.0, p + " path.DC[" + std::to_string(k) + "]");
        check_close(bf.d, sf.d, 0.0, p + " path.d[" + std::to_string(k) + "]");
        check_close(bf.d_mask, sf.d_mask, 0.0, p + " path.d_mask[" + std::to_string(k) + "]");
        check_close(bf.m, sf.m, 0.0, p + " path.m[" + std::to_string(k) + "]");
    }
    check_close(bq.term.hess, qp.term.hess, 0.0, p + " term.hess");
    check_close(bq.term.grad, qp.term.grad, 0.0, p + " term.grad");
    check_close(bq.term.DC, qp.term.DC, 0.0, p + " term.DC");
    check_close(bq.term.d, qp.term.d, 0.0, p + " term.d");
    check_close(bq.term.d_mask, qp.term.d_mask, 0.0, p + " term.d_mask");
    check_close(bq.term.m, qp.term.m, 0.0, p + " term.m");
}

// ---------------------------------------------------------------------
// Part 2: NB > 1, data-level gates (forward-sim based, no shared code)
// ---------------------------------------------------------------------

template <class P, int NH, int NB>
void check_block_data(const char* name, int N)
{
    using B = BlockQpDim<P, NB>;
    using SD = QpDim<P>;
    using S = typename P::scalar_t;
    const std::string p = name;
    const int N2 = N / NB;
    const int nx = SD::nx, nu = SD::nu;
    const int nvH = NB * nu + nx;

    Qp<P, NH> qp(N);
    Rng rng;
    fill_random_qp(qp, rng);

    Qp<P, Eigen::Dynamic, B> bq;
    condense_block<P, NH, NB>(qp, bq);
    check(bq.N == N2, p + " N2");

    // (d) terminal copied verbatim
    check_close(bq.term.hess, qp.term.hess, 0.0, p + " term.hess");
    check_close(bq.term.grad, qp.term.grad, 0.0, p + " term.grad");
    check_close(bq.term.DC, qp.term.DC, 0.0, p + " term.DC");
    check_close(bq.term.d, qp.term.d, 0.0, p + " term.d");
    check_close(bq.term.d_mask, qp.term.d_mask, 0.0, p + " term.d_mask");
    check_close(bq.term.m, qp.term.m, 0.0, p + " term.m");

    // dynamic view of original stage k (0..N-1)
    struct SSt
    {
        Eigen::MatrixXd BA, DC, H;
        Eigen::VectorXd b, g, d, dm, m;
        detail::QpLayout lay;
        int nvar, nsl;
    };
    auto sst = [&](int k) -> SSt
    {
        SSt s;
        if (k == 0)
        {
            s.BA = qp.first.BA;
            s.b = qp.first.b;
            s.DC = qp.first.DC;
            s.d = qp.first.d;
            s.dm = qp.first.d_mask;
            s.m = qp.first.m;
            s.H = qp.first.hess;
            s.g = qp.first.grad;
            s.lay = SD::lay_first;
            s.nvar = SD::nvar_first;
            s.nsl = SD::nslack_first;
        }
        else
        {
            s.BA = qp.path[k - 1].BA;
            s.b = qp.path[k - 1].b;
            s.DC = qp.path[k - 1].DC;
            s.d = qp.path[k - 1].d;
            s.dm = qp.path[k - 1].d_mask;
            s.m = qp.path[k - 1].m;
            s.H = qp.path[k - 1].hess;
            s.g = qp.path[k - 1].grad;
            s.lay = SD::lay_path;
            s.nvar = SD::nvar_path;
            s.nsl = SD::nslack_path;
        }
        return s;
    };

    // first / path block stages are different types; process each via a
    // generic lambda
    auto run_block = [&](auto& blk, int ii)
    {
        const int k0 = ii * NB;
        const int nvar = (ii == 0) ? B::nvar_first : B::nvar_path;
        const detail::QpLayout& blay = (ii == 0) ? B::lay_first
                                                 : B::lay_path;

        // block slack offset of the j-th original stage within the block
        auto soff = [&](int j)
        {
            return (ii == 0) ? B::stage_offset_first(j)
                             : B::stage_offset_path(j);
        };

        // map a block row to (block-local stage j, staged group, row)
        auto map_row = [&](int r, int& jj, int& g, int& r_orig)
        {
            g = blay.group_of(r);
            const int loc = r - blay.row_off(g);
            switch (g)
            {
            case detail::g_pin:
            case detail::g_bx:
                jj = 0;
                r_orig = loc;
                return;
            case detail::g_bu:
                jj = loc / SD::nbu;
                r_orig = loc % SD::nbu;
                return;
            case detail::g_ineq:
                jj = loc / SD::ng;
                r_orig = loc % SD::ng;
                return;
            case detail::g_eq:
                jj = loc / SD::ne;
                r_orig = loc % SD::ne;
                return;
            default:  // g_lin: [NB*nl lin rows] ++ [(NB-1)*nbx int. box]
                if (loc < NB * SD::nl)
                {
                    jj = loc / SD::nl;
                    r_orig = loc % SD::nl;
                }
                else
                {
                    const int lb = loc - NB * SD::nl;
                    jj = lb / SD::nbx + 1;
                    r_orig = lb % SD::nbx;
                    g = detail::g_bx;
                }
                return;
            }
        };

        for (int rep = 0; rep < 3; ++rep)
        {
            Eigen::VectorXd z(nvar);
            for (int i = 0; i < nvar; ++i)
            {
                z(i) = rng.uniform();
            }
            const Eigen::VectorXd x0 = z.segment(NB * nu, nx);

            // forward-simulate the block's states (independent of the
            // condense flow map)
            std::vector<Eigen::VectorXd> xs(NB + 1, x0);
            for (int j = 0; j < NB; ++j)
            {
                const SSt s = sst(k0 + j);
                Eigen::VectorXd uz(nu + nx);
                uz.head(nu) = z.segment(j * nu, nu);
                uz.tail(nx) = xs[j];
                xs[j + 1] = s.BA * uz + s.b;
            }

            // (a) block dynamics
            Eigen::VectorXd zh(nvH);
            zh.head(NB * nu) = z.head(NB * nu);
            zh.tail(nx) = x0;
            check_close(blk.BA * zh + blk.b, xs[NB], 1e-12,
                        p + " BA block " + std::to_string(ii));

            // (b) objective: block quadratic form vs staged stage costs
            const double cost_blk = S(0.5) * z.dot(blk.hess * z)
                                    + blk.grad.dot(z);
            double cost_st = 0.0;
            for (int j = 0; j < NB; ++j)
            {
                const SSt s = sst(k0 + j);
                Eigen::VectorXd zk(nu + nx + s.nsl);
                zk.head(nu) = z.segment(j * nu, nu);
                zk.segment(nu, nx) = xs[j];
                zk.tail(s.nsl) = z.segment(NB * nu + nx + soff(j), s.nsl);
                const double quad = zk.dot(s.H * zk);
                const double lin = s.g.dot(zk);
                cost_st += S(0.5) * quad + lin;
            }
            // The QP objective drops a z-independent constant. Compare
            // F(z) - F(0) with the block quadratic form; F(0) is the staged
            // cost at the zero step (u = 0, x_k0 = 0, s = 0), forward-
            // simulated independently.
            std::vector<Eigen::VectorXd> xz(NB + 1);
            xz[0] = Eigen::VectorXd::Zero(nx);
            for (int j2 = 0; j2 < NB; ++j2)
            {
                const SSt s2 = sst(k0 + j2);
                Eigen::VectorXd uz(nu + nx);
                uz.setZero();
                uz.tail(nx) = xz[j2];
                xz[j2 + 1] = s2.BA * uz + s2.b;
            }
            double cost_st0 = 0.0;
            for (int j2 = 0; j2 < NB; ++j2)
            {
                const SSt s2 = sst(k0 + j2);
                Eigen::VectorXd zk0(nu + nx + s2.nsl);
                zk0.setZero();
                zk0.segment(nu, nx) = xz[j2];
                cost_st0 += S(0.5) * zk0.dot(s2.H * zk0) + s2.g.dot(zk0);
            }
            const double dref = cost_st - cost_st0;
            const double herr = std::fabs(cost_blk - dref);
            check(herr <= 1e-9 * std::max(1.0, std::fabs(dref)),
                  p + " H/g block " + std::to_string(ii)
                  + " (blk=" + std::to_string(cost_blk)
                  + " ref=" + std::to_string(dref) + ")");

            // (c) constraint rows + offsets:
            // blk.DC(r) z + blk.d(side) == staged.DC(r') zk + staged.d(side')
            for (int r = 0; r < blk.DC.rows(); ++r)
            {
                int jj, g, r_orig;
                map_row(r, jj, g, r_orig);
                const int k = k0 + jj;
                const SSt s = sst(k);
                const int r_st = s.lay.row_off(g) + r_orig;
                // staged variable vector z_k = (u_k; x_k; s_k) at the current
                // iterate (x_k forward-simulated from x_k0).
                Eigen::VectorXd zk(nu + nx + s.nsl);
                zk.head(nu) = z.segment(jj * nu, nu);
                zk.segment(nu, nx) = xs[jj];
                zk.tail(s.nsl) = z.segment(NB * nu + nx + soff(jj), s.nsl);
                const double v_blk = blk.DC.row(r).dot(z);
                const double v_st = s.DC.row(r_st).dot(zk);
                const int blo = blay.side_lo(r);
                const int bhi = blay.side_hi(r);
                const int slo = s.lay.side_lo(r_st);
                const int shi = s.lay.side_hi(r_st);
                if (blo >= 0)
                {
                    // lo residual is d - v (qp.hpp side-layout docs)
                    check(std::fabs((blk.d(blo) - v_blk)
                                    - (s.d(slo) - v_st)) <= 1e-9,
                          p + " row lo blk " + std::to_string(ii)
                          + " row " + std::to_string(r));
                    check(blk.d_mask(blo) == s.dm(slo),
                          p + " row d_mask lo blk " + std::to_string(ii));
                    check(blk.m(blo) == s.m(slo),
                          p + " row m lo blk " + std::to_string(ii));
                }
                check(std::fabs(v_blk + blk.d(bhi)
                                - (v_st + s.d(shi))) <= 1e-9,
                      p + " row hi blk " + std::to_string(ii)
                      + " row " + std::to_string(r));
                check(blk.d_mask(bhi) == s.dm(shi),
                      p + " row d_mask hi blk " + std::to_string(ii));
                check(blk.m(bhi) == s.m(shi),
                      p + " row m hi blk " + std::to_string(ii));
            }
        }
    };

    for (int ii = 0; ii < N2; ++ii)
    {
        if (ii == 0)
        {
            run_block(bq.first, 0);
        }
        else
        {
            run_block(bq.path[ii - 1], ii);
        }
    }
}

// ---------------------------------------------------------------------
// Part 3: end-to-end — staged solve vs block solve (and vs dense for N2=1)
// ---------------------------------------------------------------------

template <class P, int NB>
void check_block_solve(const char* name, int N)
{
    using B = BlockQpDim<P, NB>;
    using SD = QpDim<P>;
    using S = typename P::scalar_t;
    const std::string p = name;
    const int N2 = N / NB;
    const int nx = SD::nx, nu = SD::nu;

    Qp<P, Eigen::Dynamic> qp(N);
    Rng rng;
    fill_random_qp(qp, rng);
    make_feasible(qp, rng);

    // reference: staged solve
    HpipmQpSolver<P, Eigen::Dynamic> s_solver;
    QpSol<P, Eigen::Dynamic> sol_s(N);
    check(s_solver.solve(qp, sol_s) == Status::kSolved, p + " staged status");
    QpRes<P, Eigen::Dynamic> res_s(N);
    s_solver.compute_residuals(qp, sol_s, res_s);

    // pipeline: condense_block -> staged solve on the block QP
    Qp<P, Eigen::Dynamic, B> bq;
    condense_block<P, Eigen::Dynamic, NB>(qp, bq);
    HpipmQpSolver<P, Eigen::Dynamic, B> b_solver;
    QpSol<P, Eigen::Dynamic, B> sol_b(N2);
    check(b_solver.solve(bq, sol_b) == Status::kSolved, p + " block status");
    QpRes<P, Eigen::Dynamic, B> res_b(N2);
    b_solver.compute_residuals(bq, sol_b, res_b);

    std::printf("%s staged: %d iter, block: %d iter\n", name,
                s_solver.statistics().iter, b_solver.statistics().iter);
    check(res_b.res_g_max <= S(1e-5), p + " block res_g_max");
    check(res_b.res_b_max <= S(1e-9), p + " block res_b_max");
    check(res_b.res_d_max <= S(1e-7), p + " block res_d_max");
    check(res_b.res_m_max <= S(1e-5), p + " block res_m_max");

    // objective agreement: both solves reach the same optimum of the same
    // underlying OCP. Compute the true (staged) objective for each solution.
    auto ocp_cost = [&](const std::vector<Eigen::VectorXd>& uu,
                        const std::vector<Eigen::VectorXd>& xx,
                        const std::vector<Eigen::VectorXd>& ss,
                        const Eigen::VectorXd& xN,
                        const Eigen::VectorXd& sN) -> double
    {
        double c = 0.0;
        for (int k = 0; k < N; ++k)
        {
            const int nsl = (k == 0) ? SD::nslack_first : SD::nslack_path;
            Eigen::VectorXd zk(nu + nx + nsl);
            zk.head(nu) = uu[k];
            zk.segment(nu, nx) = xx[k];
            if (nsl > 0)
            {
                zk.tail(nsl) = ss[k];
            }
            if (k == 0)
            {
                c += S(0.5) * zk.dot(qp.first.hess * zk) + qp.first.grad.dot(zk);
            }
            else
            {
                c += S(0.5) * zk.dot(qp.path[k - 1].hess * zk)
                     + qp.path[k - 1].grad.dot(zk);
            }
        }
        Eigen::VectorXd zt(nx + SD::nslack_term);
        zt.head(nx) = xN;
        if (SD::nslack_term > 0)
        {
            zt.tail(SD::nslack_term) = sN;
        }
        c += S(0.5) * zt.dot(qp.term.hess * zt) + qp.term.grad.dot(zt);
        return c;
    };

    // staged-solution cost
    std::vector<Eigen::VectorXd> uu_s(N), xx_s(N), ss_s(N);
    for (int k = 0; k < N; ++k)
    {
        const int nsl = (k == 0) ? SD::nslack_first : SD::nslack_path;
        if (k == 0)
        {
            uu_s[k] = sol_s.ux_first.head(nu);
            xx_s[k] = sol_s.ux_first.segment(nu, nx);
            ss_s[k] = sol_s.ux_first.tail(nsl);
        }
        else
        {
            uu_s[k] = sol_s.ux_path[k - 1].head(nu);
            xx_s[k] = sol_s.ux_path[k - 1].segment(nu, nx);
            ss_s[k] = sol_s.ux_path[k - 1].tail(nsl);
        }
    }
    const double cost_staged =
        ocp_cost(uu_s, xx_s, ss_s, sol_s.ux_term.head(nx),
                 sol_s.ux_term.tail(SD::nslack_term));

    // block-solution cost: forward-simulate internal states per block
    std::vector<Eigen::VectorXd> uu_b(N), xx_b(N), ss_b(N);
    for (int ii = 0; ii < N2; ++ii)
    {
        const int k0 = ii * NB;
        Eigen::VectorXd ub((ii == 0) ? B::nvar_first : B::nvar_path);
        if (ii == 0)
        {
            ub = sol_b.ux_first;
        }
        else
        {
            ub = sol_b.ux_path[ii - 1];
        }
        std::vector<Eigen::VectorXd> xb(NB + 1);
        xb[0] = ub.segment(NB * nu, nx);
        for (int j = 0; j < NB; ++j)
        {
            const int k = k0 + j;
            uu_b[k] = ub.head(NB * nu).segment(j * nu, nu);
            xx_b[k] = xb[j];
            const int nsl = (k == 0) ? SD::nslack_first : SD::nslack_path;
            const int soff = (ii == 0) ? B::stage_offset_first(j)
                                       : B::stage_offset_path(j);
            ss_b[k] = ub.segment(NB * nu + nx + soff, nsl);
            Eigen::VectorXd uz(nu + nx);
            uz.head(nu) = uu_b[k];
            uz.tail(nx) = xb[j];
            if (k == 0)
            {
                xb[j + 1] = qp.first.BA * uz + qp.first.b;
            }
            else
            {
                xb[j + 1] = qp.path[k - 1].BA * uz + qp.path[k - 1].b;
            }
        }
    }
    const double cost_block =
        ocp_cost(uu_b, xx_b, ss_b, sol_b.ux_term.head(nx),
                 sol_b.ux_term.tail(SD::nslack_term));

    check(std::fabs(cost_block - cost_staged)
              <= 1e-4 * std::max(1.0, std::fabs(cost_staged)),
          p + " obj agreement (blk=" + std::to_string(cost_block)
          + " st=" + std::to_string(cost_staged) + ")");

    // primal agreement: for each block ii (0..N2-1), read off u / x_k0 / slacks
    for (int ii = 0; ii < N2; ++ii)
    {
        // block-stage solution vector (u_blk; x_k0; s_blk); first / path
        // stages are different types, so copy into a dynamic vector
        const int nvar = (ii == 0) ? B::nvar_first : B::nvar_path;
        Eigen::VectorXd ub(nvar);
        if (ii == 0)
        {
            ub = sol_b.ux_first;
        }
        else
        {
            ub = sol_b.ux_path[ii - 1];
        }
        const int k0 = ii * NB;
        // controls
        for (int j = 0; j < NB; ++j)
        {
            const int k = k0 + j;
            Eigen::VectorXd us(nu);
            if (k == 0)
            {
                us = sol_s.ux_first.head(nu);
            }
            else
            {
                us = sol_s.ux_path[k - 1].head(nu);
            }
            check_close(us, ub.head(nu).segment(j * nu, nu), 1e-5,
                        p + " u[" + std::to_string(k) + "]");
        }
        // block-start state x_{k0}
        Eigen::VectorXd xs0(nx);
        if (k0 == 0)
        {
            xs0 = sol_s.ux_first.segment(nu, nx);
        }
        else
        {
            xs0 = sol_s.ux_path[k0 - 1].segment(nu, nx);
        }
        check_close(xs0, ub.segment(NB * nu, nx), 1e-5,
                    p + " x[" + std::to_string(k0) + "]");
        // slacks
        for (int j = 0; j < NB; ++j)
        {
            const int k = k0 + j;
            const int nsl = (k == 0) ? SD::nslack_first : SD::nslack_path;
            Eigen::VectorXd ss(nsl);
            if (k == 0)
            {
                ss = sol_s.ux_first.tail(nsl);
            }
            else
            {
                ss = sol_s.ux_path[k - 1].tail(nsl);
            }
            const int soff = (ii == 0) ? B::stage_offset_first(j)
                                       : B::stage_offset_path(j);
            check_close(ss, ub.segment(NB * nu + nx + soff, nsl), 1e-5,
                        p + " s[" + std::to_string(k) + "]");
        }
    }
    // terminal state
    check_close(sol_s.ux_term.head(nx), sol_b.ux_term.head(nx), 1e-5,
                p + " xN");
    check_close(sol_s.ux_term.tail(SD::nslack_term),
                sol_b.ux_term.tail(SD::nslack_term), 1e-5, p + " sN");

    // N2 = 1: cross-check against the Phase-1 full-dense pipeline
    if (N2 == 1)
    {
        using D = DenseQpDim<P>;
        DenseQp<P, Eigen::Dynamic> dq;
        condense_full<P, Eigen::Dynamic>(qp, dq);
        HpipmDenseQpSolver<P, Eigen::Dynamic> d_solver;
        DenseQpSol<P, Eigen::Dynamic> sol_d;
        sol_d.resize(N);
        check(d_solver.solve(dq, sol_d) == Status::kSolved, p + " dense status");
        const auto& v = sol_d.v;
        const auto& ub = sol_b.ux_first;
        check_close(v.head(nx), ub.segment(NB * nu, nx), 1e-5,
                    p + " dense x0");
        for (int k = 0; k < N; ++k)
        {
            check_close(v.segment(D::u_off(k), nu), ub.head(NB * nu).segment(k * nu, nu),
                        1e-5, p + " dense u[" + std::to_string(k) + "]");
        }
        int soff = 0;
        for (int k = 0; k < N; ++k)
        {
            const int nsl = (k == 0) ? SD::nslack_first : SD::nslack_path;
            check_close(v.segment(D::s_off(k, N), nsl),
                        ub.segment(N * nu + nx + soff, nsl), 1e-5,
                        p + " dense s[" + std::to_string(k) + "]");
            soff += nsl;
        }
        check_close(v.segment(D::s_off(N, N), SD::nslack_term),
                    sol_b.ux_term.tail(SD::nslack_term), 1e-5, p + " dense sN");
    }
}

}  // namespace

int main()
{
    // Part 1: NB = 1 identity
    check_nb1_identity<TinyFree, Eigen::Dynamic>("nb1 free  N=1 ", 1);
    check_nb1_identity<TinyFree, Eigen::Dynamic>("nb1 free  N=4 ", 4);
    check_nb1_identity<TinyFixed, Eigen::Dynamic>("nb1 fixed N=1 ", 1);
    check_nb1_identity<TinyFixed, Eigen::Dynamic>("nb1 fixed N=4 ", 4);
    check_nb1_identity<DoubleIntegrator, Eigen::Dynamic>("nb1 DI    N=3 ", 3);
    check_nb1_identity<MassSpring, Eigen::Dynamic>("nb1 MS    N=2 ", 2);

    // Part 2: NB > 1 data gates
    check_block_data<TinyFree, Eigen::Dynamic, 2>("d2 free  N=6 ", 6);
    check_block_data<TinyFree, Eigen::Dynamic, 3>("d3 free  N=6 ", 6);
    check_block_data<TinyFixed, Eigen::Dynamic, 2>("d2 fixed N=6 ", 6);
    check_block_data<DoubleIntegrator, Eigen::Dynamic, 2>("d2 DI    N=6 ", 6);
    check_block_data<DoubleIntegrator, Eigen::Dynamic, 3>("d3 DI    N=6 ", 6);
    check_block_data<MassSpring, Eigen::Dynamic, 2>("d2 MS    N=4 ", 4);
    check_block_data<MassSpring, Eigen::Dynamic, 2>("d2 MS    N=2 ", 2);
    check_block_data<MassSpring, Eigen::Dynamic, 4>("d4 MS    N=4 ", 4);

    // Part 3: end-to-end solve agreement
    check_block_solve<TinyFree, 2>("s2 free  N=4 ", 4);   // N2 = 2
    check_block_solve<TinyFree, 4>("s4 free  N=4 ", 4);   // N2 = 1 (+ dense)
    check_block_solve<TinyFixed, 2>("s2 fixed N=4 ", 4);
    check_block_solve<DoubleIntegrator, 2>("s2 DI    N=4 ", 4);
    check_block_solve<DoubleIntegrator, 4>("s4 DI    N=4 ", 4);
    check_block_solve<MassSpring, 2>("s2 MS    N=4 ", 4);
    check_block_solve<MassSpring, 4>("s4 MS    N=4 ", 4);

    if (failures == 0)
    {
        std::printf("All condense_block checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}

// Phase 2 (parity) block 2b test: BlockQpDim (partial-condensation block
// stage dimensions).
//   (a) NB = 1 reproduces QpDim exactly (degenerate block),
//   (b) NB = 2 extents, idxb / idxs and side mappings are structurally
//       correct (slack-column bijection),
//   (c) block extents agree with a faithful transcription of acados'
//       PART_COND_QP_COMPUTE_DIM,
//   (d) the staged Qp / QpSol / QpRes containers instantiate and size
//       correctly with a block Dim.

#include <array>
#include <cstdio>
#include <vector>

#include "ocp/solvers/hpipm/qp.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"
#include "../../examples/mass_spring/mass_spring.hpp"

namespace
{

using namespace ocp;

int failures = 0;

void check(bool cond, const char* msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

/// Synthetic problem: soft rows in every group, free x_0, so the idxb /
/// idxs offset formulas are fully exercised in the block context.
struct BlockRich
{
    using scalar_t = double;
    static constexpr int nx = 2, nu = 2;
    static constexpr bool fixed_initial_state = false;
    static constexpr int nbx = 2, nbu = 1, ng = 1, ne = 1, nl = 1;
    static constexpr int nbx_t = 1, ng_t = 1, ne_t = 0, nl_t = 1;
    static constexpr int nbx_soft = 1, nbu_soft = 1, ng_soft = 1, ne_soft = 1,
        nl_soft = 1;
    static constexpr int nbx_t_soft = 1, ng_t_soft = 1, ne_t_soft = 0,
        nl_t_soft = 1;
    static constexpr std::array<int, 2> state_box_idx = {0, 1};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 1> terminal_state_box_idx = {0};
    static constexpr std::array<int, 1> state_box_soft_idx = {1};
    static constexpr std::array<int, 1> control_box_soft_idx = {0};
    static constexpr std::array<int, 1> ineq_soft_idx = {0};
    static constexpr std::array<int, 1> eq_soft_idx = {0};
    static constexpr std::array<int, 1> lin_soft_idx = {0};
    static constexpr std::array<int, 1> terminal_state_box_soft_idx = {0};
    static constexpr std::array<int, 1> terminal_ineq_soft_idx = {0};
    static constexpr std::array<int, 0> terminal_eq_soft_idx = {};
    static constexpr std::array<int, 1> terminal_lin_soft_idx = {0};
};

/// Same, with the initial state fixed (pin rows replace the state box).
struct BlockRichFixed : BlockRich
{
    static constexpr bool fixed_initial_state = true;
};

/// hi side: always present, one per row, same order as rows; lo side:
/// present iff two-sided, inside the lo block.
void check_side_mapping(const detail::QpLayout& lay)
{
    const int nrow = lay.nrow();
    const int lo = lay.lo_size();
    for (int r = 0; r < nrow; ++r)
    {
        const int g = lay.group_of(r);
        check(g >= 0, "group_of in range");
        check(lay.side_hi(r) == lo + r, "side_hi == lo block + r");
        if (g == detail::g_ineq)
        {
            check(lay.side_lo(r) == -1, "ineq has no lo side");
        }
        else
        {
            const int lo_i = lay.side_lo(r);
            check(lo_i >= 0 && lo_i < lo, "side_lo in lo block");
        }
    }
}

/// Every slack column in [nv0, nv0 + nslack) must be referenced by exactly
/// one (row, side) pair, and only soft rows may reference it.
void check_slack_bijection(const char* what,
                           const int* idxs_lo, const int* idxs_hi, int nrow,
                           int nv0, int nslack)
{
    std::vector<char> used(nslack, 0);
    int count = 0;
    for (int r = 0; r < nrow; ++r)
    {
        const int cols[2] = {idxs_lo[r], idxs_hi[r]};
        for (int s = 0; s < 2; ++s)
        {
            const int col = cols[s];
            if (col < 0)
            {
                continue;
            }
            check(col >= nv0 && col < nv0 + nslack,
                  (std::string(what) + " slack col in range").c_str());
            const int i = col - nv0;
            check(used[i] == 0, (std::string(what) + " slack used once").c_str());
            used[i] = 1;
            ++count;
        }
    }
    check(count == nslack, (std::string(what) + " all slacks used").c_str());
}

/// (a) A block of size 1 must be identical to the plain staged QpDim.
template <class P>
void check_nb1_identity(const char* name)
{
    using S = QpDim<P>;
    using B = BlockQpDim<P, 1>;

    check(B::nx == S::nx && B::nu == S::nu, "NB=1 nx/nu");
    check(B::nbx == S::nbx && B::nbu == S::nbu, "NB=1 nbx/nbu");
    check(B::ng == S::ng && B::ne == S::ne && B::nl == S::nl, "NB=1 ng/ne/nl");
    check(B::nbx_t == S::nbx_t && B::ng_t == S::ng_t && B::ne_t == S::ne_t
              && B::nl_t == S::nl_t,
          "NB=1 terminal counts");
    check(B::fixed_initial_state == S::fixed_initial_state, "NB=1 fixed");
    check(B::nslack_first == S::nslack_first
              && B::nslack_path == S::nslack_path
              && B::nslack_term == S::nslack_term,
          "NB=1 nslack");
    check(B::nvar_first == S::nvar_first && B::nvar_path == S::nvar_path
              && B::nvar_term == S::nvar_term,
          "NB=1 nvar");
    check(B::nrow_first == S::nrow_first && B::nrow_path == S::nrow_path
              && B::nrow_term == S::nrow_term,
          "NB=1 nrow");
    check(B::nside_first == S::nside_first && B::nside_path == S::nside_path
              && B::nside_term == S::nside_term,
          "NB=1 nside");
    check(B::lay_first.rows == S::lay_first.rows
              && B::lay_first.nslack == S::lay_first.nslack,
          "NB=1 lay_first");
    check(B::lay_path.rows == S::lay_path.rows
              && B::lay_path.nslack == S::lay_path.nslack,
          "NB=1 lay_path");
    check(B::lay_term.rows == S::lay_term.rows
              && B::lay_term.nslack == S::lay_term.nslack,
          "NB=1 lay_term");
    check(B::idxb_first == S::idxb_first, "NB=1 idxb_first");
    check(B::idxb_path == S::idxb_path, "NB=1 idxb_path");
    check(B::idxb_term == S::idxb_term, "NB=1 idxb_term");
    check(B::idx_x0 == S::idx_x0, "NB=1 idx_x0");
    check(B::nbound_first == S::nbound_first
              && B::nbound_path == S::nbound_path
              && B::nbound_term == S::nbound_term,
          "NB=1 nbound");
    check(B::idxs_lo_first == S::idxs_lo_first
              && B::idxs_hi_first == S::idxs_hi_first,
          "NB=1 idxs_first");
    check(B::idxs_lo_path == S::idxs_lo_path
              && B::idxs_hi_path == S::idxs_hi_path,
          "NB=1 idxs_path");
    check(B::idxs_lo_term == S::idxs_lo_term
              && B::idxs_hi_term == S::idxs_hi_term,
          "NB=1 idxs_term");
    std::printf("NB=1 identity   %s: OK\n", name);
}

/// (b) NB = 2 extents and index structures, hand-computed.
template <class P>
void check_nb2(const char* name)
{
    using S = QpDim<P>;
    using B = BlockQpDim<P, 2>;
    const int NB = 2;

    check(B::nu == NB * P::nu, "NB=2 nu");
    check(B::nbx == P::nbx, "NB=2 nbx (block-start state only)");
    check(B::nbu == NB * P::nbu, "NB=2 nbu");
    check(B::ng == NB * P::ng, "NB=2 ng");
    check(B::ne == NB * P::ne, "NB=2 ne");
    check(B::nl == NB * P::nl + (NB - 1) * P::nbx, "NB=2 nl (lin + int. box)");
    check(B::nslack_first == S::nslack_first + (NB - 1) * S::nslack_path,
          "NB=2 nslack_first");
    check(B::nslack_path == NB * S::nslack_path, "NB=2 nslack_path");
    check(B::nslack_term == S::nslack_term, "NB=2 nslack_term");

    check(B::nvar_first == B::nu + B::nx + B::nslack_first, "NB=2 nvar_first");
    check(B::nvar_path == B::nu + B::nx + B::nslack_path, "NB=2 nvar_path");
    check(B::nvar_term == B::nx + B::nslack_term, "NB=2 nvar_term");
    check(B::nrow_first == B::lay_first.nrow(), "NB=2 nrow_first");
    check(B::nrow_path == B::lay_path.nrow(), "NB=2 nrow_path");
    check(B::nside_first == B::lay_first.nside(), "NB=2 nside_first");
    check(B::nside_path == B::lay_path.nside(), "NB=2 nside_path");

    const int pin = P::fixed_initial_state ? P::nx : 0;
    const int nbxf = P::fixed_initial_state ? 0 : P::nbx;
    const int ngen = NB * (P::ng + P::ne + P::nl) + (NB - 1) * P::nbx;
    check(B::nrow_first == pin + nbxf + B::nbu + ngen, "NB=2 nrow_first exp");
    check(B::nrow_path == P::nbx + B::nbu + ngen, "NB=2 nrow_path exp");
    check(B::nrow_term == S::nrow_term && B::nside_term == S::nside_term
              && B::nvar_term == S::nvar_term,
          "NB=2 terminal is an exact copy");

    check_side_mapping(B::lay_first);
    check_side_mapping(B::lay_path);

    // idxb: state box -> x (column offset nu); control j -> column
    // j * nu + control_box_idx[i].
    for (int j = 0; j < B::nbx_first; ++j)
    {
        check(B::idxb_first[j] == B::nu + P::state_box_idx[j],
              "NB=2 idxb_first bx");
    }
    for (int j = 0; j < P::nbx; ++j)
    {
        check(B::idxb_path[j] == B::nu + P::state_box_idx[j],
              "NB=2 idxb_path bx");
    }
    for (int j = 0; j < NB; ++j)
    {
        for (int i = 0; i < P::nbu; ++i)
        {
            const int row = j * P::nbu + i;
            const int col = j * P::nu + P::control_box_idx[i];
            check(B::idxb_first[B::nbx_first + row] == col,
                  "NB=2 idxb_first bu");
            check(B::idxb_path[P::nbx + row] == col, "NB=2 idxb_path bu");
        }
    }
    for (int j = 0; j < (int)B::idx_x0.size(); ++j)
    {
        check(B::idx_x0[j] == B::nu + j, "NB=2 idx_x0 pins x_0 at col nu+j");
    }

    // idxs: every slack column referenced exactly once (bijection).
    check_slack_bijection("NB2 first", B::idxs_lo_first.data(),
                          B::idxs_hi_first.data(), B::nrow_first,
                          B::nu + B::nx, B::nslack_first);
    check_slack_bijection("NB2 path", B::idxs_lo_path.data(),
                          B::idxs_hi_path.data(), B::nrow_path,
                          B::nu + B::nx, B::nslack_path);

    std::printf("NB=2 extents    %s: nvar_f/p/t=%d/%d/%d nrow_f/p=%d/%d  OK\n",
                name, B::nvar_first, B::nvar_path, B::nvar_term, B::nrow_first,
                B::nrow_path);
}

/// (c) Faithful transcription of acados' PART_COND_QP_COMPUTE_DIM
/// (hpipm/cond/x_part_cond.c) for one condensed block, over the original
/// per-stage count arrays (stage 0..N-1, terminal at N).
struct AcadosBlock
{
    int nx2, nu2, nbx2, nbu2, nb2, ng2, ns2;
};

AcadosBlock acados_ref_block(int ii, const std::vector<int>& block_size,
                             const std::vector<int>& nx,
                             const std::vector<int>& nu,
                             const std::vector<int>& nbx,
                             const std::vector<int>& nbu,
                             const std::vector<int>& ng,
                             const std::vector<int>& ns)
{
    AcadosBlock r{};
    int N_tmp = 0;
    for (int k = 0; k < ii; ++k)
    {
        N_tmp += block_size[k];
    }
    r.nx2 = nx[N_tmp + 0];
    r.nu2 = nu[N_tmp + 0];
    r.nbx2 = nbx[N_tmp + 0];
    r.nbu2 = nbu[N_tmp + 0];
    r.nb2 = nbx[N_tmp + 0] + nbu[N_tmp + 0];
    r.ng2 = ng[N_tmp + 0];
    r.ns2 = ns[N_tmp + 0];
    for (int jj = 1; jj < block_size[ii]; ++jj)
    {
        r.nu2 += nu[N_tmp + jj];
        r.nbu2 += nbu[N_tmp + jj];
        r.nb2 += nbu[N_tmp + jj];
        r.ng2 += ng[N_tmp + jj] + nbx[N_tmp + jj];
        r.ns2 += ns[N_tmp + jj];
    }
    return r;
}

template <class P, int NB>
void check_acados_cross(const char* name, int N)
{
    using S = QpDim<P>;
    using B = BlockQpDim<P, NB>;
    const int N2 = N / NB;

    // Original per-stage counts. acados lumps ineq + eq + lin into one
    // "ng"; "ns" is the per-stage slack count. A fixed x_0 has no stage-0
    // state box (pins instead), so nbx[0] = 0 for fixed problems.
    const int ng_per = P::ng + P::ne + P::nl;
    std::vector<int> nx(N, P::nx);
    std::vector<int> nu(N, P::nu);
    std::vector<int> nbx(N, P::nbx);
    std::vector<int> nbu(N, P::nbu);
    std::vector<int> ng(N, ng_per);
    std::vector<int> ns(N, S::nslack_path);
    if (P::fixed_initial_state)
    {
        nbx[0] = 0;
    }
    ns[0] = S::nslack_first;

    std::vector<int> block_size(N2, NB);

    // Block 0 (first) and block 1 (path): the two distinct stage types.
    for (int ii : {0, 1})
    {
        const AcadosBlock r =
            acados_ref_block(ii, block_size, nx, nu, nbx, nbu, ng, ns);
        const detail::QpLayout& lay = (ii == 0) ? B::lay_first : B::lay_path;
        const int bx = lay.rows[detail::g_bx];
        const int bu = lay.rows[detail::g_bu];
        check(r.nx2 == B::nx, "acados nx2");
        check(r.nu2 == B::nu, "acados nu2");
        check(r.nbx2 == bx, "acados nbx2");
        check(r.nbu2 == bu, "acados nbu2");
        check(r.nb2 == bx + bu, "acados nb2");
        check(r.ng2 == B::ng + B::ne + B::nl, "acados ng2 == ng+ne+nl");
        check(r.ns2 == (ii == 0 ? B::nslack_first : B::nslack_path),
              "acados ns2");
    }
    std::printf("acados cross    %s N=%d NB=%d: OK\n", name, N, NB);
}

/// (d) Staged containers instantiate and size with a block Dim.
template <class P, int NB>
void check_container(const char* name)
{
    using D = BlockQpDim<P, NB>;
    const int N2 = 2;  // two blocks

    Qp<P, Eigen::Dynamic, D> qp;
    qp.resize(N2);
    check(qp.N == N2, "block Qp::N");
    check(qp.first.hess.rows() == D::nvar_first, "block Qp hess_first");
    check(qp.first.DC.rows() == D::nrow_first, "block Qp DC_first rows");
    check(qp.first.BA.rows() == D::nx
              && qp.first.BA.cols() == D::nu + D::nx,
          "block Qp BA_first");
    for (int k = 1; k < N2; ++k)
    {
        check(qp.path[k - 1].hess.rows() == D::nvar_path, "block Qp hess_path");
        check(qp.path[k - 1].DC.rows() == D::nrow_path, "block Qp DC_path");
    }
    check(qp.term.DC.rows() == D::nrow_term, "block Qp DC_term");

    QpSol<P, Eigen::Dynamic, D> sol;
    sol.resize(N2);
    check(sol.ux_first.rows() == D::nvar_first, "block QpSol ux_first");
    check(sol.pi.size() == N2, "block QpSol pi");
    check(sol.lam_first.rows() == D::nside_first, "block QpSol lam_first");
    check(sol.lam_term.rows() == D::nside_term, "block QpSol lam_term");

    QpRes<P, Eigen::Dynamic, D> res;
    res.resize(N2);
    check(res.res_g_first.rows() == D::nvar_first, "block QpRes res_g_first");
    check(res.res_b.size() == N2, "block QpRes res_b");
    check(res.res_d_term.rows() == D::nside_term, "block QpRes res_d_term");

    std::printf("containers      %s (NB=%d, N2=%d): OK\n", name, NB, N2);
}

}  // namespace

int main()
{
    check_nb1_identity<DoubleIntegrator>("DoubleIntegrator");
    check_nb1_identity<MassSpring>("MassSpring");
    check_nb1_identity<BlockRich>("BlockRich (free, all soft)");
    check_nb1_identity<BlockRichFixed>("BlockRichFixed (fixed)");

    check_nb2<DoubleIntegrator>("DoubleIntegrator");
    check_nb2<MassSpring>("MassSpring");
    check_nb2<BlockRich>("BlockRich (free, all soft)");
    check_nb2<BlockRichFixed>("BlockRichFixed (fixed)");

    check_acados_cross<DoubleIntegrator, 2>("DoubleIntegrator", 4);
    check_acados_cross<DoubleIntegrator, 2>("DoubleIntegrator", 8);
    check_acados_cross<MassSpring, 2>("MassSpring", 4);
    check_acados_cross<BlockRich, 2>("BlockRich", 8);

    check_container<BlockRich, 2>("BlockRich");
    check_container<DoubleIntegrator, 2>("DoubleIntegrator");

    if (failures == 0)
    {
        std::printf("All BlockQpDim checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}

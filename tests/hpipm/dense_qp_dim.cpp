// Phase 1 (parity) block 1a test: DenseQpDim extent formulas, dense
// variable offsets, and the DenseQp / DenseQpSol / DenseQpRes container
// contracts (dynamic and fixed horizon).

#include <cstdio>
#include <string>

#include "ocp/solvers/hpipm/dense_qp.hpp"

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

/// Synthetic problem, small extents for hand-computed checks: x_0 free.
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

/// Same, with the initial state fixed (pin rows replace the state box).
struct TinyFixed : TinyFree
{
    static constexpr bool fixed_initial_state = true;
};

/// Hand-computed extent / offset checks for one synthetic problem.
template <class P>
void check_tiny(const char* name, bool fixed)
{
    using D = DenseQpDim<P>;

    static_assert(D::nx == 2 && D::nu == 2, "");
    check(D::kFixed == fixed, "kFixed flag");

    // slack counts (hand-computed from QpDim formulas)
    const int nsl_first =
        (fixed ? 0 : 2 * 1) + 0 + 1 + 0 + 0;  // nbx_soft = 1, ng_soft = 1
    const int nsl_path = 2 * 1 + 0 + 1 + 0 + 0;
    const int nsl_term = 2 * 1 + 0 + 0 + 0;
    check(D::nslack_total(1) == nsl_first + nsl_term, "nslack_total(1)");
    check(D::nslack_total(3) == nsl_first + 2 * nsl_path + nsl_term,
          "nslack_total(3)");

    // N = 3: hand-computed
    //   nslack_total(3) = nsl_first + 2*nsl_path + nsl_term
    //   (free: 3 + 6 + 2 = 11 -> nv = 19;  fixed: 1 + 6 + 2 = 9 -> nv = 17)
    //   ng (free)  = 3 + 2*4 + 3 = 14 (slack-independent)
    //   ng (fixed) = 2 + 2*4 + 3 = 13
    const int nbox_first = (fixed ? 2 : 0) + (fixed ? 0 : 2 - 1) + 1;
    const int ngen_first = 1 + 0 + 1 + (fixed ? 0 : 1);
    const int ngen_path = 2 + 1 + 0 + 1;
    const int ngen_term = 1 + 1 + 1 + 0;
    const int nsl3 = nsl_first + 2 * nsl_path + nsl_term;
    const int ng3 = ngen_first + 2 * ngen_path + ngen_term;
    check(D::nbox_first == nbox_first, "nbox_first");
    check(D::ngen_first == ngen_first, "ngen_first");
    check(D::ngen_path == ngen_path, "ngen_path");
    check(D::ngen_term == ngen_term, "ngen_term");
    check(D::nv_total(3) == 2 + 6 + nsl3, "nv_total(3)");
    if (!fixed)
    {
        check(D::nv_total(3) == 19, "nv_total(3) = 19 (free)");
    }
    else
    {
        check(D::nv_total(3) == 17, "nv_total(3) = 17 (fixed)");
    }
    check(D::nb_total(3) == nbox_first + 2 + nsl3, "nb_total(3)");
    check(D::ng_total(3) == ng3, "ng_total(3)");
    check(D::nside_total(3) == 2 * (nbox_first + 2 + nsl3 + ng3),
          "nside_total(3)");

    // N = 1: degenerate horizon (single dynamics step)
    check(D::nv_total(1) == 2 + 2 + nsl_first + nsl_term, "nv_total(1)");
    check(D::nb_total(1) == nbox_first + nsl_first + nsl_term,
          "nb_total(1)");
    check(D::ng_total(1) == ngen_first + ngen_term, "ng_total(1)");

    // offsets: v = [x_0 (2); u_0 (2); u_1 (2); u_2 (2); s (11)]
    check(D::x0_off() == 0, "x0_off");
    check(D::u_off(0) == 2, "u_off(0)");
    check(D::u_off(2) == 6, "u_off(2)");
    check(D::s_base(3) == 8, "s_base(3)");
    check(D::s_off(0, 3) == 8, "s_off(0,3)");
    check(D::s_off(1, 3) == 8 + nsl_first, "s_off(1,3)");
    check(D::s_off(2, 3) == 8 + nsl_first + nsl_path, "s_off(2,3)");
    check(D::s_off(3, 3) == 8 + nsl_first + 2 * nsl_path, "s_off(3,3)");
    check(D::s_off(3, 3) + nsl_term == D::nv_total(3), "s_off(3,3) + nsl == nv");

    std::printf("%s: nv(N=3)=%d nb=%d ng=%d nside=%d  OK\n", name,
                D::nv_total(3), D::nb_total(3), D::ng_total(3),
                D::nside_total(3));
}

/// Extent formulas cross-checked against explicit sums for the example
/// problems (guard against typos in the composed formulas).
template <class P>
void check_example_extents(const char* name)
{
    using D = DenseQpDim<P>;
    using S = QpDim<P>;

    for (int N : {1, 3, 5, 10})
    {
        const int nsl = S::nslack_first + (N - 1) * S::nslack_path
                        + S::nslack_term;
        const int nv = D::nx + N * D::nu + nsl;
        const int nb = D::nbox_first + (N - 1) * D::nbox_path + nsl;
        const int ng = D::ngen_first + (N - 1) * D::ngen_path + D::ngen_term;
        check(D::nv_total(N) == nv, "nv formula (example)");
        check(D::nb_total(N) == nb, "nb formula (example)");
        check(D::ng_total(N) == ng, "ng formula (example)");
        check(D::nside_total(N) == 2 * (nb + ng), "nside formula (example)");
    }
    std::printf("%s: extents OK for N = 1, 3, 5, 10\n", name);
}

/// Container contracts: resize / sizes / zeroing, dynamic + fixed horizon.
template <class P>
void check_containers(const char* name)
{
    using D = DenseQpDim<P>;
    const int N = 5;

    // dynamic horizon
    DenseQp<P> qp;
    qp.resize(N);
    check(qp.N == N, "DenseQp::N");
    check(qp.nv == D::nv_total(N), "DenseQp::nv");
    check(qp.nb == D::nb_total(N), "DenseQp::nb");
    check(qp.ng == D::ng_total(N), "DenseQp::ng");
    check(qp.nside == D::nside_total(N), "DenseQp::nside");
    check(qp.H.rows() == qp.nv && qp.H.cols() == qp.nv, "H size");
    check(qp.g.size() == qp.nv, "g size");
    check(qp.C_lo.rows() == qp.ng && qp.C_lo.cols() == qp.nv, "C_lo size");
    check(qp.C_hi.rows() == qp.ng && qp.C_hi.cols() == qp.nv, "C_hi size");
    check((int)qp.idxb.size() == qp.nb, "idxb size");
    check(qp.d.size() == qp.nside, "d size");
    check(qp.d_mask.size() == qp.nside, "d_mask size");
    check(qp.m.size() == qp.nside, "m size");
    check(qp.H.cwiseAbs().maxCoeff() == 0.0, "H zeroed");

    // resize to a different horizon reallocates
    qp.resize(3);
    check(qp.nv == D::nv_total(3), "re-resize nv");
    check(qp.H.rows() == qp.nv, "re-resize H size");

    DenseQpSol<P> sol;
    sol.resize(N);
    check(sol.nv == D::nv_total(N), "DenseQpSol::nv");
    check(sol.nside == D::nside_total(N), "DenseQpSol::nside");
    check(sol.v.size() == sol.nv, "sol.v size");
    check(sol.lam.size() == sol.nside, "sol.lam size");
    check(sol.t.size() == sol.nside, "sol.t size");

    DenseQpRes<P> res;
    res.resize(N);
    check(res.nv == D::nv_total(N), "DenseQpRes::nv");
    check(res.res_g.size() == res.nv, "res.res_g size");
    check(res.res_d.size() == res.nside, "res.res_d size");
    check(res.res_m.size() == res.nside, "res.res_m size");

    // fixed horizon NH = 5: compile-time extents
    DenseQp<P, 5> qpf;
    check(qpf.N == 5, "fixed DenseQp::N");
    check(qpf.nv == D::nv_total(5), "fixed DenseQp::nv");
    check(qpf.H.rows() == D::nv_total(5) && qpf.H.cols() == D::nv_total(5),
          "fixed H size");
    check(qpf.C_lo.rows() == D::ng_total(5), "fixed C_lo rows");
    check(qpf.C_lo.cols() == D::nv_total(5), "fixed C_lo cols");
    check(qpf.C_hi.rows() == D::ng_total(5), "fixed C_hi rows");
    check(qpf.C_hi.cols() == D::nv_total(5), "fixed C_hi cols");
    check(qpf.idxb.size() == D::nb_total(5), "fixed idxb size");
    check(qpf.d.size() == D::nside_total(5), "fixed d size");

    DenseQpSol<P, 5> solf;
    check(solf.v.size() == D::nv_total(5), "fixed sol.v size");
    DenseQpRes<P, 5> resf;
    check(resf.res_g.size() == D::nv_total(5), "fixed res.res_g size");

    // default construction (N = NH) already at the right extents
    DenseQp<P, 5> qpf2;
    check(qpf2.nv == D::nv_total(5), "fixed default-ctor nv");

    std::printf("%s: containers OK (N = %d dynamic, NH = 5 fixed)\n", name, N);
}

}  // namespace

int main()
{
    check_tiny<TinyFree>("TinyFree (x_0 free)", false);
    check_tiny<TinyFixed>("TinyFixed (x_0 fixed)", true);

    check_example_extents<DoubleIntegrator>("DoubleIntegrator");
    check_example_extents<MassSpring>("MassSpring");
    check_example_extents<TinyFree>("TinyFree");
    check_example_extents<TinyFixed>("TinyFixed");

    check_containers<TinyFree>("TinyFree");
    check_containers<TinyFixed>("TinyFixed");
    check_containers<DoubleIntegrator>("DoubleIntegrator");
    check_containers<MassSpring>("MassSpring");

    if (failures == 0)
    {
        std::printf("All DenseQpDim checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}

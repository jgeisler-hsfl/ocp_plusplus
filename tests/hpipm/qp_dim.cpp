// Phase 0 sanity test: verify QpDim stage-type layouts (first / path / term),
// side mappings, and index arrays for both example problems.

#include <cstdio>
#include <string>
#include <vector>

#include "ocp/solvers/hpipm/qp.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"
#include "../../examples/mass_spring/mass_spring.hpp"

namespace
{

/// Synthetic problem: soft rows in every group, first AND terminal, so the
/// idxs_lo/idxs_hi offset formulas are fully exercised.
struct RichSoft
{
    using scalar_t = double;
    static constexpr int nx = 3, nu = 2;
    static constexpr bool fixed_initial_state = true;
    static constexpr int nbx = 2, nbu = 1, ng = 2, ne = 1, nl = 1;
    static constexpr int nbx_t = 2, ng_t = 1, ne_t = 1, nl_t = 1;
    static constexpr int nbx_soft = 2, nbu_soft = 1, ng_soft = 1, ne_soft = 1,
        nl_soft = 1;
    static constexpr int nbx_t_soft = 2, ng_t_soft = 1, ne_t_soft = 1,
        nl_t_soft = 1;
    static constexpr std::array<int, 2> state_box_idx = {0, 1};
    static constexpr std::array<int, 1> control_box_idx = {0};
    static constexpr std::array<int, 2> terminal_state_box_idx = {0, 1};
    static constexpr std::array<int, 2> state_box_soft_idx = {0, 1};
    static constexpr std::array<int, 1> control_box_soft_idx = {0};
    static constexpr std::array<int, 1> ineq_soft_idx = {1};
    static constexpr std::array<int, 1> eq_soft_idx = {0};
    static constexpr std::array<int, 1> lin_soft_idx = {0};
    static constexpr std::array<int, 2> terminal_state_box_soft_idx = {0, 1};
    static constexpr std::array<int, 1> terminal_ineq_soft_idx = {0};
    static constexpr std::array<int, 1> terminal_eq_soft_idx = {0};
    static constexpr std::array<int, 1> terminal_lin_soft_idx = {0};
};

int failures = 0;

void check(bool cond, const char* msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

void check_side_mapping(const ocp::detail::QpLayout& lay)
{
    using ocp::detail::g_ineq;
    const int nrow = lay.nrow();
    const int lo = lay.lo_size();

    for (int r = 0; r < nrow; ++r)
    {
        const int g = lay.group_of(r);
        check(g >= 0, "group_of in range");

        const int lo_i = lay.side_lo(r);
        const int hi_i = lay.side_hi(r);

        // hi side: always present, one per row, same order as rows
        check(hi_i == lo + r, "side_hi == lo block + r");

        // lo side: present iff two-sided, inside the lo block
        if (g == g_ineq)
        {
            check(lo_i == -1, "ineq has no lo side");
        }
        else
        {
            check(lo_i >= 0 && lo_i < lo, "side_lo in lo block");
        }
    }
}

/// Every slack column in [nv0, nv0+nslack) must be referenced by exactly
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

template <class P>
void check_dims(const char* name)
{
    using D = ocp::QpDim<P>;

    // ---- row counts -------------------------------------------------
    check(D::nrow_path == D::nbx + D::nbu + D::ng + D::ne + D::nl,
          "nrow_path formula");
    check(D::nrow_term == D::nbx_t + D::ng_t + D::ne_t + D::nl_t,
          "nrow_term formula");
    const int nbx_first = P::fixed_initial_state ? 0 : D::nbx;
    const int npin = P::fixed_initial_state ? D::nx : 0;
    check(D::nrow_first == npin + nbx_first + D::nbu + D::ng + D::ne + D::nl,
          "nrow_first formula");

    // ---- side counts ------------------------------------------------
    check(D::nside_path
          == 2 * (D::nbx + D::nbu + D::ne + D::nl) + D::ng + D::nslack_path,
          "nside_path formula");
    check(D::nside_term
          == 2 * (D::nbx_t + D::ne_t + D::nl_t) + D::ng_t + D::nslack_term,
          "nside_term formula");
    check(D::nside_first
          == 2 * (npin + nbx_first + D::nbu + D::ne + D::nl) + D::ng
          + D::nslack_first,
          "nside_first formula");

    // ---- variable counts --------------------------------------------
    check(D::nvar_first == D::nu + D::nx + D::nslack_first, "nvar_first");
    check(D::nvar_path == D::nu + D::nx + D::nslack_path, "nvar_path");
    check(D::nvar_term == D::nx + D::nslack_term, "nvar_term");

    // ---- side mappings ----------------------------------------------
    check_side_mapping(D::lay_first);
    check_side_mapping(D::lay_path);
    check_side_mapping(D::lay_term);

    // ---- index arrays ------------------------------------------------
    static_assert(D::idxb_first.size() == D::nbx_first + D::nbu, "");
    static_assert(D::idxb_path.size() == D::nbx + D::nbu, "");
    static_assert(D::idxb_term.size() == D::nbx_t, "");
    static_assert(D::idx_x0.size() == (P::fixed_initial_state ? D::nx : 0), "");
    static_assert(D::idxs_lo_first.size() == D::nrow_first, "");
    static_assert(D::idxs_hi_first.size() == D::nrow_first, "");
    static_assert(D::idxs_lo_path.size() == D::nrow_path, "");
    static_assert(D::idxs_hi_path.size() == D::nrow_path, "");
    static_assert(D::idxs_lo_term.size() == D::nrow_term, "");
    static_assert(D::idxs_hi_term.size() == D::nrow_term, "");

    for (int j = 0; j < D::nbx_first; ++j)
    {
        check(D::idxb_first[j] == D::nu + P::state_box_idx[j], "idxb_first bx");
    }
    for (int j = 0; j < D::nbu; ++j)
    {
        check(D::idxb_first[D::nbx_first + j] == P::control_box_idx[j],
              "idxb_first bu");
        check(D::idxb_path[D::nbx + j] == P::control_box_idx[j], "idxb_path bu");
    }
    for (int j = 0; j < D::nbx; ++j)
    {
        check(D::idxb_path[j] == D::nu + P::state_box_idx[j], "idxb_path bx");
    }
    for (int j = 0; j < D::nbx_t; ++j)
    {
        check(D::idxb_term[j] == P::terminal_state_box_idx[j], "idxb_term");
    }
    for (int j = 0; j < (int)D::idx_x0.size(); ++j)
    {
        check(D::idx_x0[j] == D::nu + j, "idx_x0 identity into x_0");
    }

    // soft-slack column mapping: soft rows map into the slack block,
    // non-soft rows are -1; lo/hi columns are distinct
    const int nv0_first = D::nu + D::nx;
    for (int r = 0; r < D::nrow_first; ++r)
    {
        const int lo = D::idxs_lo_first[r];
        const int hi = D::idxs_hi_first[r];
        const bool soft = (lo >= 0) || (hi >= 0);
        if (soft)
        {
            check(lo >= nv0_first || lo == -1, "idxs_lo_first in slack block");
            check(hi >= nv0_first || hi == -1, "idxs_hi_first in slack block");
            if (lo >= 0 && hi >= 0)
            {
                check(lo != hi, "idxs_first lo != hi");
            }
        }
        else
        {
            check(lo == -1 && hi == -1, "idxs_first non-soft = -1");
        }
    }
    const int nv0_path = D::nu + D::nx;
    for (int r = 0; r < D::nrow_path; ++r)
    {
        const int lo = D::idxs_lo_path[r];
        const int hi = D::idxs_hi_path[r];
        check(lo == -1 || lo >= nv0_path, "idxs_lo_path in slack block");
        check(hi == -1 || hi >= nv0_path, "idxs_hi_path in slack block");
    }
    const int nv0_term = D::nx;
    for (int r = 0; r < D::nrow_term; ++r)
    {
        const int lo = D::idxs_lo_term[r];
        const int hi = D::idxs_hi_term[r];
        check(lo == -1 || lo >= nv0_term, "idxs_lo_term in slack block");
        check(hi == -1 || hi >= nv0_term, "idxs_hi_term in slack block");
    }

    // every slack column referenced exactly once (bijection)
    check_slack_bijection("first", D::idxs_lo_first.data(),
                          D::idxs_hi_first.data(), D::nrow_first,
                          nv0_first, D::nslack_first);
    check_slack_bijection("path", D::idxs_lo_path.data(),
                          D::idxs_hi_path.data(), D::nrow_path, nv0_path,
                          D::nslack_path);
    check_slack_bijection("term", D::idxs_lo_term.data(),
                          D::idxs_hi_term.data(), D::nrow_term, nv0_term,
                          D::nslack_term);

    std::printf("%s: nvar(first/path/term)=%d/%d/%d  nrow=%d/%d/%d  "
                "nside=%d/%d/%d  nslack=%d/%d/%d  OK\n",
                name, D::nvar_first, D::nvar_path, D::nvar_term, D::nrow_first,
                D::nrow_path, D::nrow_term, D::nside_first, D::nside_path,
                D::nside_term, D::nslack_first, D::nslack_path,
                D::nslack_term);
}

}  // namespace

int main()
{
    check_dims<DoubleIntegrator>("DoubleIntegrator");
    check_dims<MassSpring>("MassSpring");
    check_dims<RichSoft>("RichSoft (synthetic, all soft groups)");

    // Qp, QpSol, QpRes: dynamic-horizon construction + resize.
    {
        using D = ocp::QpDim<DoubleIntegrator>;
        ocp::Qp<DoubleIntegrator> qp;
        qp.resize(10);
        check(qp.N == 10, "Qp::N");
        check(qp.path.size() == 9, "Qp path size = N-1");

        ocp::Qp<DoubleIntegrator> qp_ctor(10);
        check(qp_ctor.N == 10 && qp_ctor.path.size() == 9, "Qp ctor");

        ocp::QpSol<DoubleIntegrator> sol;
        sol.resize(10);
        check(sol.ux_path.size() == 9, "QpSol ux_path size");
        check(sol.pi.size() == 10, "QpSol pi size");
        check(sol.lam_path.size() == 9, "QpSol lam_path size");
        check(sol.t_term.size() == D::nside_term, "QpSol t_term size");

        ocp::QpSol<DoubleIntegrator> sol_ctor(10);
        check(sol_ctor.N == 10 && sol_ctor.pi.size() == 10, "QpSol ctor");

        ocp::QpRes<DoubleIntegrator> res;
        res.resize(10);
        check(res.res_g_path.size() == 9, "QpRes res_g_path size");
        check(res.res_b.size() == 10, "QpRes res_b size");
        check(res.res_d_term.size() == D::nside_term, "QpRes res_d_term size");

        ocp::QpRes<DoubleIntegrator> res_ctor(10);
        check(res_ctor.N == 10 && res_ctor.res_b.size() == 10, "QpRes ctor");
    }

    // Fixed-horizon mode (NH = 5): compile-time extents, no allocation.
    {
        ocp::Qp<DoubleIntegrator, 5> qp_f;
        qp_f.resize(5);
        check(qp_f.N == 5, "fixed Qp::N");
        check(qp_f.path.size() == 4, "fixed Qp path size = NH-1");

        ocp::QpSol<DoubleIntegrator, 5> sol_f;
        sol_f.resize(5);
        check(sol_f.pi.size() == 5, "fixed QpSol pi size = NH");
        check(sol_f.lam_path.size() == 4, "fixed QpSol lam_path size");

        ocp::QpRes<DoubleIntegrator, 5> res_f;
        res_f.resize(5);
        check(res_f.res_g_path.size() == 4, "fixed QpRes res_g_path size");
        check(res_f.res_b.size() == 5, "fixed QpRes res_b size");
    }

    // Check Status enum has the new codes.
    static_assert(ocp::Status::kQpFailure != ocp::Status::kSolved, "");
    static_assert(ocp::Status::kMinStep != ocp::Status::kQpFailure, "");
    static_assert(ocp::Status::kUnbounded != ocp::Status::kMinStep, "");
    static_assert(ocp::Status::kNanDetected != ocp::Status::kUnbounded, "");
    static_assert(ocp::Status::kTimeout != ocp::Status::kNanDetected, "");

    if (failures == 0)
    {
        std::printf("All QpDim checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}

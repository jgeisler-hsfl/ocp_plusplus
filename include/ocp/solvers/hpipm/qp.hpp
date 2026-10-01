// ocp/solvers/hpipm/qp.hpp
//
// Staged QP data model for the HPIPM-based solver.
//
// Mirrors HPIPM's flat OCP-QP interface (hpipm_d_ocp_qp_dim.h,
// hpipm_d_ocp_qp.h, x_ocp_qp_res.c). HPIPM carries per-stage dimension
// arrays; since ocp++ stages are uniform (the problem data is constant
// over k except for the terminal), the horizon splits into three
// compile-time STAGE TYPES, each with its own row/side layout:
//
//   first (k = 0)     : initial stage. When x_0 is fixed (the usual
//                       case), the nx states are pinned by equality rows
//                       (lo == hi == x_0, HPIPM's `idxe` analogue) and the
//                       state-box rows are dropped (redundant); the stage
//                       may additionally carry control box, inequality,
//                       equality and linear rows.
//   path  (k = 1..N-1): interior stages; state box + control box +
//                       ineq + eq + lin rows.
//   term  (k = N)     : terminal; state box + ineq + eq + lin rows; no
//                       control, no dynamics.
//
// The staged QP solved by the solver (z_k = (u_k; x_k; s_k) at k = 0..N-1,
// z_N = (x_N; s_N)):
//
//   min   sum_k  0.5 z_k' hess_k z_k + grad_k' z_k
//   s.t.  x_{k+1} = BA_k (u_k; x_k) + b_k,              k = 0..N-1
//         per-row sides (side layout below), row value v:
//           lo side:   v - d <= t        (d holds the lower bound `lo`)
//           hi side:  -v - d <= t       (d holds `-hi`)
//         slack sides: s >= 0           (lo-type side, d = 0)
//   where v = z[idxb[r]] for box/pin rows, v = (DC row r) * z for the
//   general rows; soft rows carry a +1 (lo side) / -1 (hi side) slack
//   column in DC. Equality rows (eq group and pin rows) have lo == hi,
//   both sides active.
//
// Naming translation table (ocp++ -> HPIPM/acados):
//   nvar_*  -> nv          decision vars per stage type (u; x; s);
//                          HPIPM's terminal also has nu = 0, as here
//   nrow_*  -> nb + ng     constraint rows per stage type (HPIPM keeps
//                          every row two-sided and masks absent sides;
//                          we drop one-sided ineq rows' lo side)
//   nside_* -> nct        one-sided sides; length of d/d_mask/m/lam/t
//                          (HPIPM's nct = 2*(nb+ng+ns) counts both sides
//                          of every row incl. slacks)
//   ng, ne, nl -> ng       general-constraint rows (we keep ineq, eq, lin
//                          as separate groups)
//   nslack -> ns          HPIPM uses a slack pair per soft row; we use
//                          one slack variable per soft SIDE
//   BA    -> BAbt         HPIPM stores the dynamics matrix transposed,
//                          with b as the last column; we store [B|A] + b
//   DC    -> DCt          HPIPM stores the general-constraint Jacobian
//                          transposed, (nu+nx) x ng per stage (variables x
//                          constraints; box rows via idxb, slacks via
//                          idxs_rev). We store the natural orientation
//                          (nrow x nvar, constraints x variables) covering
//                          all row groups incl. box rows and slack columns.
//   idxb   -> idxb       box-row -> variable index (per stage type)
//   idx_x0 -> idxe       equality-pinned variable indices (HPIPM calls
//                        this idxe); here: the fixed x_0 components
//   idxs_lo_*/idxs_hi_* -> idxs / idxs_rev  soft row -> slack column
//
// Variable order per stage type:
//   first/path: (u; x; s)  -- u : nu, x : nx, s : nslack_<type>
//   term:       (x; s)     -- x : nx, s : nslack_term
// Slacks are solver-internal, one per soft constraint SIDE:
//   two-sided soft rows (state box, control box, equality, linear) get two
//   slacks (one per side); one-sided soft rows (inequality) get one.
// Per-stage-type slack order (groups in row-group order, zero-sized groups
// contribute nothing):
//   [bx lo; bx hi; bu lo; bu hi; ineq hi; eq lo; eq hi; lin lo; lin hi]
// with p indexing the group's *_soft_idx order. A soft row's penalty weight
// (one per row, from the problem's spec / *_constr_soft_penalty) applies to
// all of its slack variables: hess diagonal = w, grad slack part = 0
// (HPIPM's slack objective is 0.5 s'Zs + r_s' s, so Z = w; verified in
// x_ocp_qp_res.c and COND_SLACKS_FACT). Unused slack slots at a stage type
// still need a positive Hessian diagonal (e.g. 1.0) so the stage Hessian
// stays nonsingular.
//
// Row layout per stage type (groups in order; count 0 = absent group):
//   [pin; bx; bu; ineq; eq; lin]
//   first: pin (fixed only, nx rows, identity into x_0), bx (0 when fixed),
//          bu, ineq, eq, lin
//   path:  bx, bu, ineq, eq, lin
//   term:  bx (terminal box), ineq (terminal), eq (terminal), lin (terminal)
//
// Side layout per stage type (entries of d / d_mask / m / lam / t / res_d /
// res_m, nside_<type> total):
//   [lo sides: one per two-sided row, group order]
//   [hi sides: one per row, group order]
//   [slack sides: nslack_<type>, lo-type, d = 0]
// d holds bound offsets: lo sides store `lo`, hi sides store `-hi`
// (HPIPM convention; residuals are d + t - v (lo) and d + t + v (hi), so
// feasibility is lo <= v <= -d_hi, v = 0 at the linearization point for
// general rows). Pin rows: d_lo = x_0, d_hi = -x_0 (runtime values from
// initial_state()).
//
// Assembly notes (phase 2):
// - First stage with fixed x_0: stage ineq/lin rows whose (linearized)
//   Jacobian w.r.t. the decision variables (u_0; s_0) is identically zero
//   are degenerate; check D != 0 before relying on them.
// - Terminal stage has no u and no BA/b (zero dynamics).
//
// HPIPM reference:
//   acados/external/hpipm/include/hpipm_d_ocp_qp_dim.h
//   acados/external/hpipm/ocp_qp/x_ocp_qp.c        (CREATE, layout)
//   acados/external/hpipm/ocp_qp/x_ocp_qp_sol.c    (ux/pi/lam/t sizes)
//   acados/external/hpipm/ocp_qp/x_ocp_qp_res.c    (residual definitions)
//   acados/docs/algorithm/05-hpipm-qp-solver.md

#pragma once

#include <array>
#include <cassert>
#include <cstddef>

#include <Eigen/Dense>

#include "../../problem.hpp"

namespace ocp
{

// =========================================================================
//  Stage-type layout (compile-time)
// =========================================================================

namespace detail
{

/// Row groups of a stage type, in row order.
enum QpGroup : int
{
    g_pin = 0,  // initial-state pinning rows (first stage, fixed x_0 only)
    g_bx,       // state box (two-sided)
    g_bu,       // control box (two-sided)
    g_ineq,     // inequality g <= 0 (one-sided, hi only)
    g_eq,       // equality e == 0 (two-sided, lo == hi)
    g_lin,      // linear two-sided
    n_groups
};

constexpr bool is_two_sided(int g)
{
    return g != g_ineq;
}

/// Row/side layout of one stage type: per-group row counts + slack count.
struct QpLayout
{
    std::array<int, n_groups> rows;  // per-group row counts (0 = absent)
    int nslack;                      // slack variables of this stage type

    /// Total constraint rows.
    constexpr int nrow() const
    {
        int s = 0;
        for (int g = 0; g < n_groups; ++g)
        {
            s += rows[g];
        }
        return s;
    }

    /// lo sides: one per two-sided row, in group order.
    constexpr int lo_size() const
    {
        int s = 0;
        for (int g = 0; g < n_groups; ++g)
        {
            if (is_two_sided(g))
            {
                s += rows[g];
            }
        }
        return s;
    }

    /// Total sides: lo block + hi block (one per row) + slack sides.
    constexpr int nside() const
    {
        return lo_size() + nrow() + nslack;
    }

    /// First row index of group g.
    constexpr int row_off(int g) const
    {
        int s = 0;
        for (int h = 0; h < g; ++h)
        {
            s += rows[h];
        }
        return s;
    }

    /// Group of row r, or -1 if r is out of range.
    constexpr int group_of(int r) const
    {
        for (int g = 0; g < n_groups; ++g)
        {
            if (r < row_off(g) + rows[g])
            {
                return g;
            }
        }
        return -1;
    }

    /// Side index of the lo side of row r; -1 if the row has no lo side
    /// (ineq rows) or r is out of range.
    constexpr int side_lo(int r) const
    {
        const int g = group_of(r);
        if (g < 0 || !is_two_sided(g))
        {
            return -1;
        }
        int s = 0;
        for (int h = 0; h < g; ++h)
        {
            if (is_two_sided(h))
            {
                s += rows[h];
            }
        }
        return s + (r - row_off(g));
    }

    /// Side index of the hi side of row r; -1 if r is out of range.
    /// (hi block: one entry per row, same order as rows)
    constexpr int side_hi(int r) const
    {
        const int g = group_of(r);
        if (g < 0)
        {
            return -1;
        }
        return lo_size() + r;
    }
};

}  // namespace detail

// =========================================================================
//  QpDim
// =========================================================================

/// Compile-time structural dimensions of the staged QP derived from a
/// concrete ocp::Problem type.
///
/// All extents are constants; each stage TYPE (first / path / term) has its
/// own row count (nrow_*), side count (nside_*) and variable count
/// (nvar_*). See the header comment for the stage-type definitions.
template <class P>
struct QpDim
{
    // ---------------------------------------------------------------
    // Core dimensions
    // ---------------------------------------------------------------
    static constexpr int nx = P::nx;  // states
    static constexpr int nu = P::nu;  // controls

    // ---------------------------------------------------------------
    // Active row counts (stage k = 0..N-1, terminal k = N)
    // ---------------------------------------------------------------
    static constexpr int nbx  = P::nbx;  // state box rows (two-sided)
    static constexpr int nbu  = P::nbu;  // control box rows (two-sided)
    static constexpr int ng   = P::ng;   // ineq rows g <= 0 (one-sided, hi)
    static constexpr int ne   = P::ne;   // equality rows (two-sided)
    static constexpr int nl   = P::nl;   // linear rows (two-sided)
    static constexpr int nbx_t = P::nbx_t;  // terminal state box rows
    static constexpr int ng_t  = P::ng_t;
    static constexpr int ne_t  = P::ne_t;
    static constexpr int nl_t  = P::nl_t;

    // ---------------------------------------------------------------
    // Slack variables (one per soft side)
    // ---------------------------------------------------------------
    /// Slacks at the first stage: stage soft rows, minus the state-box
    /// ones when x_0 is fixed (the state box is replaced by pin rows).
    static constexpr int nslack_first =
        (P::fixed_initial_state ? 0 : 2 * P::nbx_soft) + 2 * P::nbu_soft
        + P::ng_soft + 2 * P::ne_soft + 2 * P::nl_soft;
    /// Slacks at a path stage.
    static constexpr int nslack_path = 2 * P::nbx_soft + 2 * P::nbu_soft
                                      + P::ng_soft + 2 * P::ne_soft
                                      + 2 * P::nl_soft;
    /// Slacks at the terminal stage.
    static constexpr int nslack_term = 2 * P::nbx_t_soft + P::ng_t_soft
                                      + 2 * P::ne_t_soft + 2 * P::nl_t_soft;

    // ---------------------------------------------------------------
    // Stage-type layouts
    // ---------------------------------------------------------------
    static constexpr detail::QpLayout lay_first = {
        {P::fixed_initial_state ? nx : 0, P::fixed_initial_state ? 0 : nbx,
         nbu, ng, ne, nl},
        nslack_first};
    static constexpr detail::QpLayout lay_path = {
        {0, nbx, nbu, ng, ne, nl},
        nslack_path};
    static constexpr detail::QpLayout lay_term = {
        {0, nbx_t, 0, ng_t, ne_t, nl_t},
        nslack_term};

    // ---------------------------------------------------------------
    // Derived totals (per stage type)
    // ---------------------------------------------------------------
    /// Decision variables per stage type.
    static constexpr int nvar_first = nu + nx + nslack_first;  // (u; x; s)
    static constexpr int nvar_path  = nu + nx + nslack_path;   // (u; x; s)
    static constexpr int nvar_term  = nx + nslack_term;        // (x; s)

    /// Constraint rows per stage type.
    static constexpr int nrow_first = lay_first.nrow();
    static constexpr int nrow_path  = lay_path.nrow();
    static constexpr int nrow_term  = lay_term.nrow();

    /// One-sided sides per stage type: length of the stage-type
    /// d / d_mask / m / lam / t / res_d / res_m vectors.
    static constexpr int nside_first = lay_first.nside();
    static constexpr int nside_path  = lay_path.nside();
    static constexpr int nside_term  = lay_term.nside();

    // ---------------------------------------------------------------
    // Side mapping per stage type
    // ---------------------------------------------------------------
    static constexpr int side_lo_first(int r) { return lay_first.side_lo(r); }
    static constexpr int side_hi_first(int r) { return lay_first.side_hi(r); }
    static constexpr int side_lo_path(int r)  { return lay_path.side_lo(r); }
    static constexpr int side_hi_path(int r)  { return lay_path.side_hi(r); }
    static constexpr int side_lo_term(int r)  { return lay_term.side_lo(r); }
    static constexpr int side_hi_term(int r)  { return lay_term.side_hi(r); }

    // ---------------------------------------------------------------
    // Compile-time index arrays (HPIPM stand-ins)
    // ---------------------------------------------------------------

    /// Box-row -> variable index in the FIRST-stage vector (u; x; s):
    /// state box (absent when fixed) then control box.
    static constexpr int nbx_first = P::fixed_initial_state ? 0 : nbx;
    static constexpr std::array<int, nbx_first + nbu> make_idxb_first()
    {
        std::array<int, nbx_first + nbu> a{};
        for (int j = 0; j < nbx_first; ++j)
        {
            a[j] = nu + P::state_box_idx[j];
        }
        for (int j = 0; j < nbu; ++j)
        {
            a[nbx_first + j] = P::control_box_idx[j];
        }
        return a;
    }
    static constexpr std::array<int, nbx_first + nbu> idxb_first =
        make_idxb_first();

    /// Box-row -> variable index in a PATH-stage vector (u; x; s).
    static constexpr std::array<int, nbx + nbu> make_idxb_path()
    {
        std::array<int, nbx + nbu> a{};
        for (int j = 0; j < nbx; ++j)
        {
            a[j] = nu + P::state_box_idx[j];
        }
        for (int j = 0; j < nbu; ++j)
        {
            a[nbx + j] = P::control_box_idx[j];
        }
        return a;
    }
    static constexpr std::array<int, nbx + nbu> idxb_path = make_idxb_path();

    /// Box-row -> variable index in the TERMINAL vector (x; s): no +nu
    /// offset (no control at the terminal).
    static constexpr std::array<int, nbx_t> make_idxb_term()
    {
        std::array<int, nbx_t> a{};
        for (int j = 0; j < nbx_t; ++j)
        {
            a[j] = P::terminal_state_box_idx[j];
        }
        return a;
    }
    static constexpr std::array<int, nbx_t> idxb_term = make_idxb_term();

    /// Equality-pinned variable indices (HPIPM `idxe`): the fixed initial
    /// state components inside the first-stage vector (u; x; s); empty
    /// when x_0 is not fixed. Pin row j pins variable idx_x0[j].
    static constexpr std::array<int, P::fixed_initial_state ? nx : 0>
        make_idx_x0()
    {
        std::array<int, P::fixed_initial_state ? nx : 0> a{};
        for (int j = 0; j < (P::fixed_initial_state ? nx : 0); ++j)
        {
            a[j] = nu + j;
        }
        return a;
    }
    static constexpr std::array<int, P::fixed_initial_state ? nx : 0>
        idx_x0 = make_idx_x0();

    // Box-type bound-row counts (init_point's (row, varidx) pairs):
    // pin rows (first, fixed x_0 only) + state box + control box.
    static constexpr int nbound_first = (P::fixed_initial_state ? nx : 0)
                                        + nbx_first + nbu;
    static constexpr int nbound_path = nbx + nbu;
    static constexpr int nbound_term = nbx_t;

    // ---------------------------------------------------------------
    // Soft-slack column mapping (HPIPM `idxs` / `idxs_rev` analogue)
    // ---------------------------------------------------------------
    // For each row of a stage type: the column index in the stage's
    // variable vector z at which its lo-side / hi-side slack variable
    // lives, or -1 if the row is not soft (or has no such side).
    //
    // The slack block in z starts at column `nvar_<type> - nslack_<type>`
    // and follows the documented slack order:
    //   [bx lo; bx hi; bu lo; bu hi; ineq hi; eq lo; eq hi; lin lo; lin hi]
    // (absent groups contribute nothing; the first stage drops the bx
    // entries when x_0 is fixed).

    // --- first stage ------------------------------------------------
    static constexpr std::array<int, nrow_first> make_idxs_lo_first()
    {
        constexpr int nbx_soft = P::fixed_initial_state ? 0 : P::nbx_soft;
        constexpr int nbu_soft = P::nbu_soft;
        constexpr int ne_soft  = P::ne_soft;
        constexpr int nl_soft  = P::nl_soft;
        const int nv0 = nu + nx;
        std::array<int, nrow_first> a{};
        for (int r = 0; r < nrow_first; ++r)
        {
            a[r] = -1;
        }
        for (int i = 0; i < nbx_soft; ++i)
        {
            a[lay_first.row_off(detail::g_bx) + P::state_box_soft_idx[i]] =
                nv0 + i;
        }
        for (int i = 0; i < nbu_soft; ++i)
        {
            a[lay_first.row_off(detail::g_bu) + P::control_box_soft_idx[i]] =
                nv0 + 2 * nbx_soft + i;
        }
        for (int i = 0; i < ne_soft; ++i)
        {
            a[lay_first.row_off(detail::g_eq) + P::eq_soft_idx[i]] =
                nv0 + 2 * nbx_soft + 2 * nbu_soft + P::ng_soft + i;
        }
        for (int i = 0; i < nl_soft; ++i)
        {
            a[lay_first.row_off(detail::g_lin) + P::lin_soft_idx[i]] =
                nv0 + 2 * nbx_soft + 2 * nbu_soft + P::ng_soft + 2 * ne_soft
                + i;
        }
        return a;
    }
    static constexpr std::array<int, nrow_first> idxs_lo_first =
        make_idxs_lo_first();

    static constexpr std::array<int, nrow_first> make_idxs_hi_first()
    {
        constexpr int nbx_soft = P::fixed_initial_state ? 0 : P::nbx_soft;
        constexpr int nbu_soft = P::nbu_soft;
        constexpr int ng_soft  = P::ng_soft;
        constexpr int ne_soft  = P::ne_soft;
        constexpr int nl_soft  = P::nl_soft;
        const int nv0 = nu + nx;
        std::array<int, nrow_first> a{};
        for (int r = 0; r < nrow_first; ++r)
        {
            a[r] = -1;
        }
        for (int i = 0; i < nbx_soft; ++i)
        {
            a[lay_first.row_off(detail::g_bx) + P::state_box_soft_idx[i]] =
                nv0 + nbx_soft + i;
        }
        for (int i = 0; i < nbu_soft; ++i)
        {
            a[lay_first.row_off(detail::g_bu) + P::control_box_soft_idx[i]] =
                nv0 + 2 * nbx_soft + nbu_soft + i;
        }
        for (int i = 0; i < ng_soft; ++i)
        {
            a[lay_first.row_off(detail::g_ineq) + P::ineq_soft_idx[i]] =
                nv0 + 2 * nbx_soft + 2 * nbu_soft + i;
        }
        for (int i = 0; i < ne_soft; ++i)
        {
            a[lay_first.row_off(detail::g_eq) + P::eq_soft_idx[i]] =
                nv0 + 2 * nbx_soft + 2 * nbu_soft + ng_soft + ne_soft + i;
        }
        for (int i = 0; i < nl_soft; ++i)
        {
            a[lay_first.row_off(detail::g_lin) + P::lin_soft_idx[i]] =
                nv0 + 2 * nbx_soft + 2 * nbu_soft + ng_soft
                + 2 * ne_soft + nl_soft + i;
        }
        return a;
    }
    static constexpr std::array<int, nrow_first> idxs_hi_first =
        make_idxs_hi_first();

    // --- path stages -------------------------------------------------
    static constexpr std::array<int, nrow_path> make_idxs_lo_path()
    {
        constexpr int nbx_soft = P::nbx_soft;
        constexpr int nbu_soft = P::nbu_soft;
        constexpr int ne_soft  = P::ne_soft;
        constexpr int nl_soft  = P::nl_soft;
        const int nv0 = nu + nx;
        std::array<int, nrow_path> a{};
        for (int r = 0; r < nrow_path; ++r)
        {
            a[r] = -1;
        }
        for (int i = 0; i < nbx_soft; ++i)
        {
            a[lay_path.row_off(detail::g_bx) + P::state_box_soft_idx[i]] =
                nv0 + i;
        }
        for (int i = 0; i < nbu_soft; ++i)
        {
            a[lay_path.row_off(detail::g_bu) + P::control_box_soft_idx[i]] =
                nv0 + 2 * nbx_soft + i;
        }
        for (int i = 0; i < ne_soft; ++i)
        {
            a[lay_path.row_off(detail::g_eq) + P::eq_soft_idx[i]] =
                nv0 + 2 * nbx_soft + 2 * nbu_soft + P::ng_soft + i;
        }
        for (int i = 0; i < nl_soft; ++i)
        {
            a[lay_path.row_off(detail::g_lin) + P::lin_soft_idx[i]] =
                nv0 + 2 * nbx_soft + 2 * nbu_soft + P::ng_soft + 2 * ne_soft
                + i;
        }
        return a;
    }
    static constexpr std::array<int, nrow_path> idxs_lo_path =
        make_idxs_lo_path();

    static constexpr std::array<int, nrow_path> make_idxs_hi_path()
    {
        constexpr int nbx_soft = P::nbx_soft;
        constexpr int nbu_soft = P::nbu_soft;
        constexpr int ng_soft  = P::ng_soft;
        constexpr int ne_soft  = P::ne_soft;
        constexpr int nl_soft  = P::nl_soft;
        const int nv0 = nu + nx;
        std::array<int, nrow_path> a{};
        for (int r = 0; r < nrow_path; ++r)
        {
            a[r] = -1;
        }
        for (int i = 0; i < nbx_soft; ++i)
        {
            a[lay_path.row_off(detail::g_bx) + P::state_box_soft_idx[i]] =
                nv0 + nbx_soft + i;
        }
        for (int i = 0; i < nbu_soft; ++i)
        {
            a[lay_path.row_off(detail::g_bu) + P::control_box_soft_idx[i]] =
                nv0 + 2 * nbx_soft + nbu_soft + i;
        }
        for (int i = 0; i < ng_soft; ++i)
        {
            a[lay_path.row_off(detail::g_ineq) + P::ineq_soft_idx[i]] =
                nv0 + 2 * nbx_soft + 2 * nbu_soft + i;
        }
        for (int i = 0; i < ne_soft; ++i)
        {
            a[lay_path.row_off(detail::g_eq) + P::eq_soft_idx[i]] =
                nv0 + 2 * nbx_soft + 2 * nbu_soft + ng_soft + ne_soft + i;
        }
        for (int i = 0; i < nl_soft; ++i)
        {
            a[lay_path.row_off(detail::g_lin) + P::lin_soft_idx[i]] =
                nv0 + 2 * nbx_soft + 2 * nbu_soft + ng_soft
                + 2 * ne_soft + nl_soft + i;
        }
        return a;
    }
    static constexpr std::array<int, nrow_path> idxs_hi_path =
        make_idxs_hi_path();

    // --- terminal stage ----------------------------------------------
    static constexpr std::array<int, nrow_term> make_idxs_lo_term()
    {
        constexpr int nbx_t_soft = P::nbx_t_soft;
        constexpr int ne_t_soft  = P::ne_t_soft;
        constexpr int nl_t_soft  = P::nl_t_soft;
        const int nv0 = nx;  // terminal vector is (x; s)
        std::array<int, nrow_term> a{};
        for (int r = 0; r < nrow_term; ++r)
        {
            a[r] = -1;
        }
        for (int i = 0; i < nbx_t_soft; ++i)
        {
            a[lay_term.row_off(detail::g_bx)
                + P::terminal_state_box_soft_idx[i]] = nv0 + i;
        }
        for (int i = 0; i < ne_t_soft; ++i)
        {
            a[lay_term.row_off(detail::g_eq) + P::terminal_eq_soft_idx[i]] =
                nv0 + 2 * nbx_t_soft + P::ng_t_soft + i;
        }
        for (int i = 0; i < nl_t_soft; ++i)
        {
            a[lay_term.row_off(detail::g_lin) + P::terminal_lin_soft_idx[i]] =
                nv0 + 2 * nbx_t_soft + P::ng_t_soft + 2 * ne_t_soft + i;
        }
        return a;
    }
    static constexpr std::array<int, nrow_term> idxs_lo_term =
        make_idxs_lo_term();

    static constexpr std::array<int, nrow_term> make_idxs_hi_term()
    {
        constexpr int nbx_t_soft = P::nbx_t_soft;
        constexpr int ng_t_soft  = P::ng_t_soft;
        constexpr int ne_t_soft  = P::ne_t_soft;
        constexpr int nl_t_soft  = P::nl_t_soft;
        const int nv0 = nx;  // terminal vector is (x; s)
        std::array<int, nrow_term> a{};
        for (int r = 0; r < nrow_term; ++r)
        {
            a[r] = -1;
        }
        for (int i = 0; i < nbx_t_soft; ++i)
        {
            a[lay_term.row_off(detail::g_bx)
                + P::terminal_state_box_soft_idx[i]] = nv0 + nbx_t_soft + i;
        }
        for (int i = 0; i < ng_t_soft; ++i)
        {
            a[lay_term.row_off(detail::g_ineq) + P::terminal_ineq_soft_idx[i]] =
                nv0 + 2 * nbx_t_soft + i;
        }
        for (int i = 0; i < ne_t_soft; ++i)
        {
            a[lay_term.row_off(detail::g_eq) + P::terminal_eq_soft_idx[i]] =
                nv0 + 2 * nbx_t_soft + ng_t_soft + ne_t_soft + i;
        }
        for (int i = 0; i < nl_t_soft; ++i)
        {
            a[lay_term.row_off(detail::g_lin) + P::terminal_lin_soft_idx[i]] =
                nv0 + 2 * nbx_t_soft + ng_t_soft + 2 * ne_t_soft + nl_t_soft
                + i;
        }
        return a;
    }
    static constexpr std::array<int, nrow_term> idxs_hi_term =
        make_idxs_hi_term();
};

// =========================================================================
//  Stage data
// =========================================================================

/// First-stage (k = 0) QP data.
///
/// hess/grad/DC cover the variable vector (u_0; x_0; s_0); DC rows follow
/// the first-stage row layout (pin rows first when x_0 is fixed).
template <class P>
struct QpStageFirst
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    /// Hessian of the stage cost over (u_0; x_0; s_0).
    /// The (u;x) block is the user cost Hessian plus the dynamics /
    /// constraint HVP terms assembled by the SQP; the slack diagonal holds
    /// the soft-penalty weight w (0.5 * w * s^2 contract; Z = w,
    /// verified). Unused slack slots: any positive value (e.g. 1.0).
    /// Off-diagonal slack entries are zero.
    Eigen::Matrix<S, D::nvar_first, D::nvar_first> hess{};

    /// Linear term over (u_0; x_0; s_0): the permuted cost gradient; the
    /// slack part is zero (HPIPM's r_s = 0).
    Eigen::Matrix<S, D::nvar_first, 1> grad{};

    /// Dynamics: [B | A], rows = x_1, cols = (u_0; x_0).
    /// (HPIPM stores this transposed as BAbt with b as the last column.)
    Eigen::Matrix<S, D::nx, D::nu + D::nx> BA{};

    /// Dynamics offset: b_0 = f_0(x_0, u_0) - x_1 (residual at the current
    /// iterate).
    Eigen::Matrix<S, D::nx, 1> b{};

    /// Constraint Jacobian over (u_0; x_0; s_0): nrow_first rows in the
    /// first-stage row layout, DC[r, :] = d c_r / d z (natural orientation;
    /// HPIPM's DCt is its transpose, general constraints only). Pin rows:
    /// unit vector at the pinned variable (idx_x0). Box rows: unit vector at
    /// the constrained variable (idxb_first). General rows: Jacobian
    /// [dg/dx | dg/du] plus slack columns (+1 lo side / -1 hi side) for
    /// soft rows. Absent groups: zero rows.
    Eigen::Matrix<S, D::nrow_first, D::nvar_first> DC{};

    /// Bound offsets, nside_first entries in the first-stage side layout:
    /// lo sides store `lo`, hi sides store `-hi`, slack sides 0. For
    /// general rows the assembly folds in the constraint value at the
    /// current iterate (offset form, not distance, as in HPIPM's `d`).
    /// Pin rows: lo side = x_0, hi side = -x_0.
    Eigen::Matrix<S, D::nside_first, 1> d{};

    /// 1 = finite / active side, 0 = infinite / absent side.
    Eigen::Matrix<S, D::nside_first, 1> d_mask{};

    /// Complementarity RHS: lam * t = m (zero in v1; reserved for
    /// m_relax / funnel in later phases).
    Eigen::Matrix<S, D::nside_first, 1> m{};

    /// Zero all stage data (container contract for fixed-extent storage).
    void setZero()
    {
        hess.setZero();
        grad.setZero();
        BA.setZero();
        b.setZero();
        DC.setZero();
        d.setZero();
        d_mask.setZero();
        m.setZero();
    }
};

/// Path-stage (k = 1..N-1) QP data; identical shape to QpStageFirst when
/// x_0 is not fixed, smaller when it is.
template <class P>
struct QpStagePath
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    /// Hessian of the stage cost over (u; x; s). See QpStageFirst::hess.
    Eigen::Matrix<S, D::nvar_path, D::nvar_path> hess{};

    /// Linear term over (u; x; s): the permuted cost gradient; slack part 0.
    Eigen::Matrix<S, D::nvar_path, 1> grad{};

    /// Dynamics: [B | A], rows = x_{k+1}, cols = (u_k; x_k).
    Eigen::Matrix<S, D::nx, D::nu + D::nx> BA{};

    /// Dynamics offset: b_k = f_k(x_k, u_k) - x_{k+1}.
    Eigen::Matrix<S, D::nx, 1> b{};

    /// Constraint Jacobian over (u; x; s): nrow_path rows in the
    /// path-stage row layout. See QpStageFirst::DC.
    Eigen::Matrix<S, D::nrow_path, D::nvar_path> DC{};

    /// Bound offsets, path-stage side layout. See QpStageFirst::d.
    Eigen::Matrix<S, D::nside_path, 1> d{};

    /// 1 = finite / active side, 0 = infinite / absent side.
    Eigen::Matrix<S, D::nside_path, 1> d_mask{};

    /// Complementarity RHS: lam * t = m (zero in v1).
    Eigen::Matrix<S, D::nside_path, 1> m{};

    /// Zero all stage data (container contract for fixed-extent storage).
    void setZero()
    {
        hess.setZero();
        grad.setZero();
        BA.setZero();
        b.setZero();
        DC.setZero();
        d.setZero();
        d_mask.setZero();
        m.setZero();
    }
};

/// Terminal (k = N) QP data; variable vector (x_N; s_N), no dynamics.
template <class P>
struct QpStageTerm
{
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    /// Hessian of the terminal cost over (x_N; s_N). See
    /// QpStageFirst::hess.
    Eigen::Matrix<S, D::nvar_term, D::nvar_term> hess{};

    /// Linear term over (x_N; s_N): the permuted terminal cost gradient;
    /// slack part 0.
    Eigen::Matrix<S, D::nvar_term, 1> grad{};

    /// Constraint Jacobian over (x_N; s_N): nrow_term rows in the
    /// terminal row layout (state box, ineq, eq, lin; no control box).
    /// See QpStageFirst::DC.
    Eigen::Matrix<S, D::nrow_term, D::nvar_term> DC{};

    /// Bound offsets, terminal side layout. See QpStageFirst::d.
    Eigen::Matrix<S, D::nside_term, 1> d{};

    /// 1 = finite / active side, 0 = infinite / absent side.
    Eigen::Matrix<S, D::nside_term, 1> d_mask{};

    /// Complementarity RHS: lam * t = m (zero in v1).
    Eigen::Matrix<S, D::nside_term, 1> m{};

    /// Zero all stage data (container contract for fixed-extent storage).
    void setZero()
    {
        hess.setZero();
        grad.setZero();
        DC.setZero();
        d.setZero();
        d_mask.setZero();
        m.setZero();
    }
};

// =========================================================================
//  Qp
// =========================================================================

/// A complete staged QP instance (N + 1 stages, k = 0..N).
///
/// NH = Eigen::Dynamic (default): runtime horizon, `resize(int)` allocates.
/// NH >= 1: compile-time horizon; path storage is fixed-extent (std::array)
/// and `resize(int)` only asserts that the argument matches NH. N = 0
/// (terminal-only problem) is not supported: at least one dynamics step is
/// required.
template <class P, int NH = Eigen::Dynamic>
struct Qp
{
    static_assert(NH == Eigen::Dynamic || NH >= 1,
        "ocp::Qp: the horizon must be at least 1 (one dynamics step) or "
        "Eigen::Dynamic.");

    using D = QpDim<P>;

    int N = (NH == Eigen::Dynamic) ? 0 : NH;
    QpStageFirst<P> first;  // k = 0
    Trajectory<QpStagePath<P>, detail::traj_extent<NH, -1>()> path;  // k = 1..N-1
    QpStageTerm<P> term;    // k = N

    Qp() = default;
    /// Dynamic horizon: allocate for `n_stages` stages; fixed horizon:
    /// assert that `n_stages` matches NH.
    explicit Qp(int n_stages)
    {
        resize(n_stages);
    }

    void resize(int n_stages)
    {
        if constexpr (NH == Eigen::Dynamic)
        {
            assert(n_stages >= 1);
            N = n_stages;
            path.resize(N - 1);
        }
        else
        {
            assert(n_stages == NH);
            N = NH;
        }
    }
};

// =========================================================================
//  QpSol
// =========================================================================

/// Solution of a staged QP.
///
/// Mixed convention (matching HPIPM):
///   - ux_*  : primal STEP (delta) over the stage-type variable vector
///   - pi    : ABSOLUTE dynamics multipliers (k = 0..N-1)
///   - lam_* : ABSOLUTE constraint multipliers, one per side
///   - t_*   : ABSOLUTE IPM slacks, same layout as lam
template <class P, int NH = Eigen::Dynamic>
struct QpSol
{
    static_assert(NH == Eigen::Dynamic || NH >= 1,
        "ocp::QpSol: the horizon must be at least 1 (one dynamics step) or "
        "Eigen::Dynamic.");

    using D = QpDim<P>;
    using S = typename P::scalar_t;

    using ux_first_t  = Eigen::Matrix<S, D::nvar_first, 1>;
    using ux_path_t   = Eigen::Matrix<S, D::nvar_path, 1>;
    using ux_term_t   = Eigen::Matrix<S, D::nvar_term, 1>;
    using pi_t        = Eigen::Matrix<S, D::nx, 1>;
    using lam_first_t = Eigen::Matrix<S, D::nside_first, 1>;
    using lam_path_t  = Eigen::Matrix<S, D::nside_path, 1>;
    using lam_term_t  = Eigen::Matrix<S, D::nside_term, 1>;

    int N = (NH == Eigen::Dynamic) ? 0 : NH;

    // primal steps
    ux_first_t ux_first;  // k = 0
    Trajectory<ux_path_t, detail::traj_extent<NH, -1>()> ux_path;  // k = 1..N-1
    ux_term_t ux_term;    // k = N

    // absolute dynamics multipliers (N entries, k = 0..N-1)
    Trajectory<pi_t, detail::traj_extent<NH, 0>()> pi;

    // absolute constraint multipliers / IPM slacks
    lam_first_t lam_first;
    Trajectory<lam_path_t, detail::traj_extent<NH, -1>()> lam_path;
    lam_term_t lam_term;
    lam_first_t t_first;
    Trajectory<lam_path_t, detail::traj_extent<NH, -1>()> t_path;
    lam_term_t t_term;

    QpSol() = default;
    /// Dynamic horizon: allocate for `n_stages` stages; fixed horizon:
    /// assert that `n_stages` matches NH.
    explicit QpSol(int n_stages)
    {
        resize(n_stages);
    }

    void resize(int n_stages)
    {
        if constexpr (NH == Eigen::Dynamic)
        {
            assert(n_stages >= 1);
            N = n_stages;
            ux_path.resize(N - 1);
            pi.resize(N);
            lam_path.resize(N - 1);
            t_path.resize(N - 1);
        }
        else
        {
            assert(n_stages == NH);
            N = NH;
        }
    }
};

// =========================================================================
//  QpRes
// =========================================================================

/// KKT residuals of a staged QP (HPIPM `d_ocp_qp_res` analogue).
///
/// Computed by the solver's `compute_residuals` each IPM iteration.
template <class P, int NH = Eigen::Dynamic>
struct QpRes
{
    static_assert(NH == Eigen::Dynamic || NH >= 1,
        "ocp::QpRes: the horizon must be at least 1 (one dynamics step) or "
        "Eigen::Dynamic.");

    using D = QpDim<P>;
    using S = typename P::scalar_t;

    using res_g_first_t = Eigen::Matrix<S, D::nvar_first, 1>;
    using res_g_path_t  = Eigen::Matrix<S, D::nvar_path, 1>;
    using res_g_term_t  = Eigen::Matrix<S, D::nvar_term, 1>;
    using res_b_t       = Eigen::Matrix<S, D::nx, 1>;
    using res_d_first_t = Eigen::Matrix<S, D::nside_first, 1>;
    using res_d_path_t  = Eigen::Matrix<S, D::nside_path, 1>;
    using res_d_term_t  = Eigen::Matrix<S, D::nside_term, 1>;

    int N = (NH == Eigen::Dynamic) ? 0 : NH;

    /// Stationarity residual (per stage-type variable vector)
    res_g_first_t res_g_first;
    Trajectory<res_g_path_t, detail::traj_extent<NH, -1>()> res_g_path;
    res_g_term_t res_g_term;

    /// Dynamics (equality) residual (N stages, k = 0..N-1)
    Trajectory<res_b_t, detail::traj_extent<NH, 0>()> res_b;

    /// Inequality feasibility residual (per stage type)
    res_d_first_t res_d_first;
    Trajectory<res_d_path_t, detail::traj_extent<NH, -1>()> res_d_path;
    res_d_term_t res_d_term;

    /// Complementarity residual (per stage type)
    res_d_first_t res_m_first;
    Trajectory<res_d_path_t, detail::traj_extent<NH, -1>()> res_m_path;
    res_d_term_t res_m_term;

    /// Inf-norm maxima across all stages
    S res_g_max = 0, res_b_max = 0, res_d_max = 0, res_m_max = 0;

    /// res_mu = ||res_m||_1 / nc_mask  (masked mean of complementarity)
    S res_mu_sum = 0;
    S res_mu     = 0;

    /// Primal objective value (accumulated during residual computation)
    S obj = 0;
    /// Dual gap
    S dual_gap = 0;

    QpRes() = default;
    /// Dynamic horizon: allocate for `n_stages` stages; fixed horizon:
    /// assert that `n_stages` matches NH.
    explicit QpRes(int n_stages)
    {
        resize(n_stages);
    }

    void resize(int n_stages)
    {
        if constexpr (NH == Eigen::Dynamic)
        {
            assert(n_stages >= 1);
            N = n_stages;
            res_g_path.resize(N - 1);
            res_b.resize(N);
            res_d_path.resize(N - 1);
            res_m_path.resize(N - 1);
        }
        else
        {
            assert(n_stages == NH);
            N = NH;
        }
    }
};

}  // namespace ocp

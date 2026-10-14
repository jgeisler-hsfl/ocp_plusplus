// ocp/problem.hpp
//
// Problem interface of the ocp++ framework (first draft).
//
// A *problem* is the mathematical definition of a discrete-time optimal
// control problem (OCP)
//
//     min_{x, u}   sum_{k=0}^{N-1} L_k(x_k, u_k)  +  L_N(x_N)
//     subject to   x_{k+1} = f_k(x_k, u_k),       k = 0..N-1
//                  box, linear and general (in)equality constraints,
//
// decoupled from any particular solver algorithm.
//
// Usage
// -----
// 1. Declare the structural dimensions as a small tag struct:
//
//        struct MyDims {
//            static constexpr int nx = 2, nu = 1;
//            static constexpr int ng = 1, ne = 0, nl = 1;
//            static constexpr int ng_t = 0, ne_t = 1, nl_t = 0;
//            static constexpr bool fixed_initial_state = true;  // false: x_0 is a
//                                                               // decision variable
//                                                               // (e.g. MHE)
//            static constexpr bool has_dynamics_hess = true;      // true: exact
//                                                                // Lagrangian Hessian
//                                                                // (acados EXACT);
//                                                                // false: Gauss-Newton
//                                                                // QP Hessian, dynamics
//                                                                // only via the
//                                                                // linearized BA (acados
//                                                                // GAUSS_NEWTON)
//            static constexpr bool has_constr_hess   = true;      // true: constraint
//                                                                // Hessian terms
//                                                                // included;
//                                                                // false: omitted
//                                                                // (valid if the maps
//                                                                // are linear)
//            static constexpr int np = 0;                 // sensitivity
//                                                          // parameter-vector
//                                                          // length (0: none)
//            static constexpr std::array<int, 1> state_box_idx   = {0};
//            static constexpr std::array<int, 1> control_box_idx = {0};
//            static constexpr std::array<int, 2> terminal_state_box_idx = {0, 1};
//            static constexpr std::array<int, 1> ineq_soft_idx   = {0};  // g row 0 is soft;
//            static constexpr std::array<int, 0> eq_soft_idx     = {};   // likewise for lin_soft_idx,
//            //                                                           // state_box_soft_idx, control_box_soft_idx,
//            //                                                           // terminal_{ineq,eq,lin}_soft_idx,
//            //                                                           // terminal_state_box_soft_idx
//        };
//
//    The box index arrays select which states/controls each box-constraint
//    row bounds (const, stage-invariant); their size *is* the number of
//    active box rows (nbx/nbu/nbx_t, derived by ocp::Problem). Box bounds
//    are stored per active row only. The `*_soft_idx` arrays mark which
//    constraint rows of each group are *soft* (may be violated by a solver
//    slack); their size *is* the number of soft rows in the group (an
//    empty set means the whole group is hard).
//
// 2. Implement the concrete problem by inheriting from
//    ocp::Problem<MyDims> and providing the model functions
//    (signatures in the "Interface contract" section below).
//
// The dimensions live at compile time (fixed-size Eigen objects, static
// allocation, no virtual dispatch). The horizon N is owned by the
// Solution: a runtime quantity by default, optionally fixed at compile
// time via the Solution's horizon template parameter. All model calls are
// made on the *concrete* problem type, so the derived implementations hide
// the base-class stubs and dispatch statically.

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

namespace ocp {

// =========================================================================
//  Trajectory
// =========================================================================

/// A horizon-extended, stage-wise quantity (e.g. the state trajectory).
///
/// A trajectory is a *semantic* container: that an object is a sequence of
/// stage-wise values is part of its type, while the underlying storage is
/// an implementation detail that may change without touching the problem
/// interface.
///
/// `NSlots` is the number of stage slots:
///   - `Eigen::Dynamic` (default): runtime stage count, owned by
///     `ocp::Solution`; storage is a heap-allocated `std::vector`, resized
///     on demand.
///   - a non-negative integer: the stage count is a compile-time constant;
///     storage is a value-initialized (zeroed) `std::array`, no heap
///     allocation. The sized constructor and `resize` become consistency
///     checks (the argument must equal `NSlots`).
///
/// Extent conventions for a horizon N (see ARCHITECTURE.md, sec. 5):
///
///   state trajectory        N + 1 stages   (k = 0..N)
///   control trajectory      N     stages   (k = 0..N-1)
///   dynamics residual       N     stages   (k = 0..N-1)
///   stage constraints        N     stages   (k = 0..N-1)
///   terminal constraints    1     stage    (k = N)
namespace detail
{
/// Fixed-extent trajectory storage (compile-time stage count, stack).
template <class T, int NSlots>
struct TrajectoryStorage
{
    static_assert(NSlots >= 0,
        "ocp::Trajectory: a fixed stage count must be non-negative.");
    std::array<T, NSlots> slots;

    // explicit zeroing (GCC's -Wmaybe-uninitialized does not track the
    // aggregate value-init of `slots{}` through the inheritance chain)
    TrajectoryStorage()
    {
        for (auto& s : slots)
        {
            s.setZero();
        }
    }
};

/// Dynamic-extent trajectory storage (runtime stage count, heap).
template <class T>
struct TrajectoryStorage<T, Eigen::Dynamic>
{
    std::vector<T> slots;
};

/// Stage count of a trajectory for horizon `NH` and extent offset
/// (1 for x-like quantities, 0 for u-like); Dynamic stays Dynamic.
template <int NH, int Offset>
constexpr int traj_extent()
{
    return (NH == Eigen::Dynamic) ? Eigen::Dynamic : NH + Offset;
}

/// Detection of an optional `np` member in a Dims tag (5i). Dims tags that
/// do not declare `np` are treated as np = 0 (no parameter vector; the
/// `*_sens_p` entries are optional and never called).
template <class D, class = void>
struct has_np : std::false_type
{
};
template <class D>
struct has_np<D, std::void_t<decltype(D::np)>> : std::true_type
{
};

/// Parameter-vector length of a Dims tag (0 when `np` is not declared).
template <class D>
constexpr int dims_np()
{
    if constexpr (has_np<D>::value)
    {
        return D::np;
    }
    else
    {
        return 0;
    }
}
}  // namespace detail

template <class T, int NSlots = Eigen::Dynamic>
class Trajectory : private detail::TrajectoryStorage<T, NSlots>
{
    using Storage = detail::TrajectoryStorage<T, NSlots>;

public:
    Trajectory() = default;
    explicit Trajectory(std::size_t n_stages)
    {
        resize(n_stages);
    }

    std::size_t size() const noexcept { return Storage::slots.size(); }
    bool empty() const noexcept { return size() == 0; }

    /// (Re)allocate for `n_stages` stages.
    ///
    /// Dynamic mode: reallocates (destroys previous values).
    /// Fixed mode: no allocation happens; the argument must equal the
    /// compile-time stage count.
    void resize(std::size_t n_stages)
    {
        if constexpr (NSlots != Eigen::Dynamic)
        {
            assert(n_stages == std::size_t(NSlots));
        }
        else
        {
            Storage::slots.resize(n_stages);
        }
    }

    T& operator[](std::size_t k) { return Storage::slots[k]; }
    const T& operator[](std::size_t k) const { return Storage::slots[k]; }

    T* data() { return Storage::slots.data(); }
    const T* data() const { return Storage::slots.data(); }
};

// =========================================================================
//  Constraint specs
// =========================================================================

/// Per-row slack penalty weights for soft constraint rows.
///
/// Each soft row i contributes a slack s[i] >= 0 that relaxes the row from
/// the outside. The solver adds to the objective
///
///     0.5 * quad[i] * s[i]^2  +  lin[i] * s[i]
///
/// for each soft side of row i. `quad` is the Hessian (second-order) weight
/// and `lin` the gradient (first-order) weight. This mirrors acados'
/// per-slack `Z` (diagonal Hessian weight) and `z` (gradient weight) in
/// `ocp_nlp_cost_common`; the standard soft constraint uses `quad = w`,
/// `lin = 0`. A full (non-diagonal) Hessian weight is not supported, in
/// line with the dense HPIPM QP which carries a diagonal slack Hessian.
template <class Scalar, int N>
struct SoftPenalty
{
    Eigen::Matrix<Scalar, N, 1> quad;  // Hessian weight: 0.5 * quad * s^2
    Eigen::Matrix<Scalar, N, 1> lin;   // gradient weight: lin * s (default 0)

    SoftPenalty()
    {
        quad.setZero();
        lin.setZero();
    }

    /// Quadratic-only convenience (the common case): lin defaults to 0.
    SoftPenalty(const Eigen::Matrix<Scalar, N, 1>& w)
        : quad(w), lin(0)
    {
    }

    SoftPenalty(const Eigen::Matrix<Scalar, N, 1>& w,
                const Eigen::Matrix<Scalar, N, 1>& r)
        : quad(w), lin(r)
    {
    }
};

/// Two-sided bounds on `Dim` active constraint rows, with per-row slack
/// (soft-constraint) penalty weights.
///
/// For box constraints, `Dim` is the number of *active* rows: row i
/// bounds state/control component `state_box_idx[i]` (or the
/// corresponding control / terminal index set), a compile-time constant
/// from the problem's `Dims` tag.
///
/// Which rows are *soft* is a compile-time decision made by the problem's
/// `Dims`: a row is soft iff its index appears in the group's `*_soft_idx`
/// set. A soft row may be violated by a non-negative slack that relaxes
/// the row from the outside; `penalty.quad[i]` and `penalty.lin[i]` are
/// the second- and first-order weights of the slack penalty the solver
/// adds to the stage cost for that row (per side, see `SoftPenalty`). The
/// numerical treatment of soft rows is the solver's business; the problem
/// only declares them.
template <class Scalar, int Dim>
struct BoxSpec
{
    Eigen::Matrix<Scalar, Dim, 1> lo{};      // lower bounds
    Eigen::Matrix<Scalar, Dim, 1> hi{};      // upper bounds
    SoftPenalty<Scalar, Dim>      penalty;   // per-row slack penalty weights
};

/// Linear two-sided stage constraint  lo <= A x + B u <= hi  (stage k);
/// an additive constant, if any, is absorbed into the bounds.
template <class Scalar, int NRow, int Nx, int Nu>
struct LinearSpec
{
    Eigen::Matrix<Scalar, NRow, Nx> A{};
    Eigen::Matrix<Scalar, NRow, Nu> B{};
    BoxSpec<Scalar, NRow>           bounds{};
};

/// Linear two-sided terminal constraint  lo <= A x <= hi  (no control);
/// an additive constant, if any, is absorbed into the bounds.
template <class Scalar, int NRow, int Nx>
struct TerminalLinearSpec
{
    Eigen::Matrix<Scalar, NRow, Nx> A{};
    BoxSpec<Scalar, NRow>           bounds{};
};

/// Parameter-vector (p) derivatives of a box spec's bounds (sensitivity
/// analysis, `*_sens_p` interface entries).  One column per parameter
/// component: `dlo_dp(i, j) = d(lo_i)/dp_j`.
template <class Scalar, int NRow, int NParam>
struct BoxSensP
{
    Eigen::Matrix<Scalar, NRow, NParam> dlo_dp{};
    Eigen::Matrix<Scalar, NRow, NParam> dhi_dp{};
};

/// Parameter-vector (p) derivatives of a linear spec's coefficients and
/// bounds (sensitivity analysis, `*_sens_p` interface entries).
///
/// The coefficient p-derivatives are column-stacked: for the parameter
/// component p_j, column `j` is the (NRow x Nx) matrix dA/dp_j flattened in
/// row-major order, i.e. `dA_dp(r * Nx + i, j) = dA(r, i)/dp_j` (likewise
/// dB_dp with Nu).  `NParam == 0` (no parameter vector) makes all fields
/// zero-size.
template <class Scalar, int NRow, int Nx, int Nu, int NParam>
struct LinearSensP
{
    Eigen::Matrix<Scalar, NRow * Nx, NParam> dA_dp{};
    Eigen::Matrix<Scalar, NRow * Nu, NParam> dB_dp{};
    Eigen::Matrix<Scalar, NRow, NParam>      dlo_dp{};
    Eigen::Matrix<Scalar, NRow, NParam>      dhi_dp{};
};

// =========================================================================
//  Problem
// =========================================================================

/// Template interface of a discrete-time OCP.
///
/// The concrete problem supplies its structural dimensions via the
/// `Dims` tag and implements the model functions listed below. The base
/// class re-exposes the dimensions as `static constexpr` members, derives
/// all fixed-size Eigen types from them, and provides one stub per
/// interface function: if a function is missing or
/// missigned in the derived class, the inherited stub is instantiated and
/// compilation fails with a descriptive message.
///
/// Design rules
/// ------------
/// - Structural dimensions (nx, nu, constraint rows) are compile-time
///   quantities; the horizon N is a runtime quantity owned by the
///   Solution (see `ocp::Solution` below), not by the problem.
/// - All model functions are `const` and pure: they read the user-managed
///   parameter state of the concrete problem but take no parameter
///   argument (parameters are not part of the interface, see
///   ARCHITECTURE.md sec. 2 "Separation of parameters").
/// - Stage index k: 0..N-1 for dynamics, stage cost and stage constraints;
///   the terminal stage k = N is served by the dedicated terminal_*
///   functions, which depend on the state only (there is no u_N).
/// - In combined (x, u) vectors and matrices the state is listed first:
///   gradients have layout [g_x; g_u], Hessians the block layout
///   [H_xx H_xu; H_ux H_uu].
/// - All evaluation/derivative calls are made on the *concrete* problem
///   type (static dispatch, no virtual machinery).
///
/// Interface contract (each function below must be implemented by the
/// concrete problem unless the corresponding dimension is zero, in which
/// case solvers never call it and the stub may stay unimplemented):
///
///   initial state (fixed x_0, not a decision variable; required only
///   when Dims::fixed_initial_state is true)
///     state_t initial_state() const;
///
///   dynamics (k = 0..N-1); fused evaluation entries
///     state_t dynamics_next_state(int k, const state_t& x,
///                                 const control_t& u) const;
///     void dynamics_value_jac(int k, const state_t& x, const control_t& u,
///                             state_t& x_next, dyn_df_dx_t& df_dx,
///                             dyn_df_du_t& df_du) const;
///     void dynamics_value_jac_hess(int k, const state_t& x,
///                                  const control_t& u, const state_t& lam,
///                                  state_t& x_next, dyn_df_dx_t& df_dx,
///                                  dyn_df_du_t& df_du,
///                                  dyn_hess_t& hess) const;
///       // hess v = sum_i lam_i D^2 f_i v, [x; u] layout
///
///   stage cost (k = 0..N-1); fused evaluation entries
///     Scalar stage_cost_value(int k, const state_t& x,
///                             const control_t& u) const;
///     void stage_cost_value_grad(int k, const state_t& x,
///                                const control_t& u,
///                                Scalar& value, stage_grad_t& grad) const;
///     void stage_cost_value_grad_hess(int k, const state_t& x,
///                                     const control_t& u,
///                                     Scalar& value, stage_grad_t& grad,
///                                     stage_hess_t& hess) const;
///
///   terminal cost (stage N)
///     Scalar terminal_cost_value(const state_t& x) const;
///     void terminal_cost_value_grad(const state_t& x, Scalar& value,
///                                   term_grad_t& grad) const;
///     void terminal_cost_value_grad_hess(const state_t& x, Scalar& value,
///                                        term_grad_t& grad,
///                                        term_hess_t& hess) const;
///
///   stage constraints (k = 0..N-1); fused evaluation entries
///     ineq_t stage_inequality_value(int k, const state_t& x,
///                                   const control_t& u) const;  // g <= 0
///     void stage_inequality_value_jac(int k, const state_t& x,
///                                     const control_t& u, ineq_t& g,
///                                     ineq_dg_dx_t& g_dx,
///                                     ineq_dg_du_t& g_du) const;
///     void stage_inequality_value_jac_hess(int k, const state_t& x,
///                                          const control_t& u,
///                                          const ineq_t& lam, ineq_t& g,
///                                          ineq_dg_dx_t& g_dx,
///                                          ineq_dg_du_t& g_du,
///                                          constr_hess_t& hess) const;
///     eq_t stage_equality_value(int k, const state_t& x,
///                               const control_t& u) const;       // e == 0
///     void stage_equality_value_jac(int k, const state_t& x,
///                                   const control_t& u, eq_t& e,
///                                   eq_de_dx_t& e_dx,
///                                   eq_de_du_t& e_du) const;
///     void stage_equality_value_jac_hess(int k, const state_t& x,
///                                        const control_t& u,
///                                        const eq_t& lam, eq_t& e,
///                                        eq_de_dx_t& e_dx,
///                                        eq_de_du_t& e_du,
///                                        constr_hess_t& hess) const;
///     stage_linear_t stage_linear_constr(int k) const;
///     state_box_t stage_state_box_constr(int k) const;        // rows: state_box_idx
///     control_box_t stage_control_box_constr(int k) const;    // rows: control_box_idx
///     ineq_pen_t stage_inequality_constr_soft_penalty(int k) const;  // rows: ineq_soft_idx
///     eq_pen_t   stage_equality_constr_soft_penalty(int k) const;    // rows: eq_soft_idx
///
///   terminal constraints (stage N, state only); fused evaluation entries
///     term_state_box_t terminal_state_box_constr() const;  // rows: terminal_state_box_idx
///     ineq_term_t terminal_inequality_value(const state_t& x) const;  // g <= 0
///     void terminal_inequality_value_jac(const state_t& x, ineq_term_t& g,
///                                        ineq_term_dg_dx_t& g_dx) const;
///     void terminal_inequality_value_jac_hess(const state_t& x,
///                                             const ineq_term_t& lam,
///                                             ineq_term_t& g,
///                                             ineq_term_dg_dx_t& g_dx,
///                                             term_constr_hess_t& hess) const;
///     eq_term_t terminal_equality_value(const state_t& x) const;      // e == 0
///     void terminal_equality_value_jac(const state_t& x, eq_term_t& e,
///                                      eq_term_de_dx_t& e_dx) const;
///     void terminal_equality_value_jac_hess(const state_t& x,
///                                           const eq_term_t& lam,
///                                           eq_term_t& e,
///                                           eq_term_de_dx_t& e_dx,
///                                           term_constr_hess_t& hess) const;
///     term_linear_t terminal_linear_constr() const;
///     ineq_term_pen_t terminal_inequality_constr_soft_penalty() const; // rows: terminal_ineq_soft_idx
///     eq_term_pen_t   terminal_equality_constr_soft_penalty() const;   // rows: terminal_eq_soft_idx
///
///   parameter-vector sensitivity (only when Dims::np > 0; all entries
///   evaluated at the current model state, one column per p component,
///   Jacobian-type entries column-stacked in p, see the type aliases):
///     void dynamics_sens_p(int k, const state_t& x, const control_t& u,
///                          dyn_df_dp_t& f_p, dyn_dA_dp_t& A_p,
///                          dyn_dB_dp_t& B_p) const;
///     void stage_cost_sens_p(int k, const state_t& x, const control_t& u,
///                            param_t& L_p, stage_grad_dp_t& grad_p) const;
///     void terminal_cost_sens_p(const state_t& x, param_t& L_p,
///                               term_grad_dp_t& grad_p) const;
    ///     void stage_inequality_sens_p(int k, const state_t& x,
    ///                                  const control_t& u, ineq_dg_dp_t& g_p,
    ///                                  ineq_dg_dx_dp_t& gdx_p,
    ///                                  ineq_dg_du_dp_t& gdu_p) const;
    ///     void stage_equality_sens_p(int k, const state_t& x,
    ///                                const control_t& u, eq_de_dp_t& e_p,
    ///                                eq_de_dx_dp_t& edx_p,
    ///                                eq_de_du_dp_t& edu_p) const;
    ///     void terminal_inequality_sens_p(const state_t& x,
    ///                                     ineq_term_dg_dp_t& g_p,
    ///                                     ineq_term_dg_dx_dp_t& gdx_p) const;
    ///     void terminal_equality_sens_p(const state_t& x,
    ///                                   eq_term_de_dp_t& e_p,
    ///                                   eq_term_de_dx_dp_t& edx_p) const;
///     stage_linear_sensp_t   stage_linear_sens_p(int k) const;
///     term_linear_sensp_t    terminal_linear_sens_p() const;
///     stage_state_box_sensp_t  stage_state_box_sens_p(int k) const;
///     stage_control_box_sensp_t stage_control_box_sens_p(int k) const;
///     term_state_box_sensp_t   terminal_state_box_sens_p() const;
///     x0_dp_t initial_state_sens_p() const;
///
/// Notes
/// -----
/// - Second-order information is user-supplied: the stage/terminal
///   Hessian may be the exact Hessian of the cost, a Gauss-Newton
///   approximation (residual Jacobian transposed times itself) or any
///   other valid approximation; the interface does not distinguish
///   (ARCHITECTURE.md sec. 6).
/// - When `Dims::fixed_initial_state` is true, the initial state x_0 is a
///   fixed given, provided by `initial_state()`, and is not a decision
///   variable (the stage-0 state box need not pin it). When it is false
///   (e.g. moving-horizon estimation), x_0 is a decision variable and
///   `initial_state()` is not part of the interface.
/// - Soft constraint rows are selected per group at compile time by the
///   `*_soft_idx` index sets in `Dims` (their size is the number of soft
///   rows in that group). Each soft row i carries a `SoftPenalty` with a
///   quadratic (Hessian) weight `quad[i]` and a linear (gradient) weight
///   `lin[i]`; the solver adds 0.5*quad*s^2 + lin*s per soft side (acados
///   `Zl/Zu` + `zl/zu`; the standard soft constraint is quad = w, lin = 0).
///   Nonlinear (in)equality groups report their penalties via the
///   `*_constr_soft_penalty` methods (returning a `SoftPenalty`); box and
///   linear groups carry them in the returned spec's `penalty` field. The
///   slack variables themselves are solver-internal.
/// - The `*_value_jac_hess` entries return the multiplier-contracted
///   Hessian in one call: `hess v = sum_i lam_i D^2(·)_i v`, where `lam`
///   is the net KKT multiplier of the corresponding group (e.g.
///   `lambda_dyn[k]` for dynamics). Model-side Hessians use the `[x; u]`
///   layout; the solver permutes into its internal `(u; x)` layout.
/// - Two independent flags, `Dims::has_dynamics_hess` and
///   `Dims::has_constr_hess`, tell the solver whether the
///   `dynamics_value_jac_hess` / constraint `*_value_jac_hess` entries
///   exist. `has_dynamics_hess` selects the QP Hessian model: `true`
///   assembles the exact Lagrangian Hessian (cost + dynamics Hessian,
///   acados `hessian_approx = EXACT`); `false` assembles the
///   Gauss-Newton Hessian, where the dynamics enter the QP only through
///   the linearized BA matrix (acados `hessian_approx = GAUSS_NEWTON` /
///   historical `ROSEN`). The two models coincide when the dynamics map is
///   linear (zero dynamics Hessian). `has_constr_hess` = false omits the
///   constraint Hessian terms (valid exactly when the constraint maps are
///   linear, i.e. their Hessians are identically zero). The solver exposes
///   two independent runtime gates, `AcadosOptions::compute_hess_dyn` /
///   `compute_hess_constr` (mirroring acados' `exact_hess_dyn` /
///   `exact_hess_constr`), that drop the multiplier-weighted dynamics /
///   constraint Hessian terms at runtime without recompiling.
/// - Parameter-vector sensitivity (`np > 0`): the problem declares a
///   compile-time parameter-vector length `np` in `Dims` and implements
///   the fused `*_sens_p` derivative entries listed below. They provide
///   the p-derivatives of the model values, cost gradients and constraint
///   Jacobians/bounds required by NLP-level sensitivity analysis (see
///   AcadosSolver::sens_param). When `np == 0` the entries are optional
///   and are never called.
/// - Extensions anticipated beyond this draft (not yet part of the
///   contract): algebraic (eliminated) variables.

template <class Dims, class Scalar = double>
class Problem
{
public:
    // ---------------------------------------------------------------
    // Structural dimensions (compile time)
    // ---------------------------------------------------------------
    static constexpr int nx   = Dims::nx;    // states
    static constexpr int nu   = Dims::nu;    // controls
    static constexpr int ng   = Dims::ng;    // stage nonlinear inequalities
    static constexpr int ne   = Dims::ne;    // stage nonlinear equalities
    static constexpr int nl   = Dims::nl;    // stage linear two-sided rows
    static constexpr int ng_t = Dims::ng_t;  // terminal nonlinear inequalities
    static constexpr int ne_t = Dims::ne_t;  // terminal nonlinear equalities
    static constexpr int nl_t = Dims::nl_t;  // terminal linear two-sided rows
    static constexpr int np   = detail::dims_np<Dims>();  // parameter-vector
                                             // NLP-level parameter sensitivity
                                             // (0: no parameter vector, the
                                             // *_sens_p entries are optional
                                             // and are never called)

    // true: x_0 is a fixed given via initial_state(); false: x_0 is a
    // decision variable (e.g. MHE) and initial_state() is not available
    static constexpr bool fixed_initial_state = Dims::fixed_initial_state;

    // true: dynamics_value_jac_hess is implemented; the solver assembles
    // the exact Lagrangian QP Hessian (cost Hessian + multiplier-contracted
    // dynamics Hessian; acados hessian_approx = EXACT). false: the Hessian
    // is not provided; the solver assembles the Gauss-Newton QP Hessian,
    // where the dynamics enter only through the linearized BA matrix
    // (acados hessian_approx = GAUSS_NEWTON; the two models coincide for a
    // linear dynamics map, where the dynamics Hessian term is zero anyway).
    // The term may additionally be dropped at runtime via
    // AcadosOptions::compute_hess_dyn (acados exact_hess_dyn).
    static constexpr bool has_dynamics_hess = Dims::has_dynamics_hess;

    // true: the constraint *_value_jac_hess methods are implemented; the
    // solver adds the multiplier-contracted constraint Hessians to the QP
    // Hessian (EXACT). false: no constraint Hessians; the constraint
    // Hessian terms are omitted (identically zero when the constraint maps
    // are linear). The terms may additionally be dropped at runtime via
    // AcadosOptions::compute_hess_constr (acados exact_hess_constr).
    static constexpr bool has_constr_hess = Dims::has_constr_hess;

    // active box-constraint rows: the const index sets select which
    // state/control components are bounded; their size *is* the number of
    // active box rows, and box specs store one bound entry per row only
    static constexpr int nbx   = Dims::state_box_idx.size();
    static constexpr int nbu   = Dims::control_box_idx.size();
    static constexpr int nbx_t = Dims::terminal_state_box_idx.size();
    static constexpr std::array<int, nbx>   state_box_idx   = Dims::state_box_idx;
    static constexpr std::array<int, nbu>   control_box_idx = Dims::control_box_idx;
    static constexpr std::array<int, nbx_t> terminal_state_box_idx = Dims::terminal_state_box_idx;

    // soft constraint rows: the const index sets select which rows of each
    // group may be violated by a solver slack; their size *is* the number
    // of soft rows (empty set => the whole group is hard)
    static constexpr int ng_soft    = Dims::ineq_soft_idx.size();
    static constexpr int ne_soft    = Dims::eq_soft_idx.size();
    static constexpr int nl_soft    = Dims::lin_soft_idx.size();
    static constexpr int nbx_soft   = Dims::state_box_soft_idx.size();
    static constexpr int nbu_soft   = Dims::control_box_soft_idx.size();
    static constexpr int ng_t_soft  = Dims::terminal_ineq_soft_idx.size();
    static constexpr int ne_t_soft  = Dims::terminal_eq_soft_idx.size();
    static constexpr int nl_t_soft  = Dims::terminal_lin_soft_idx.size();
    static constexpr int nbx_t_soft = Dims::terminal_state_box_soft_idx.size();
    static constexpr std::array<int, ng_soft>    ineq_soft_idx    = Dims::ineq_soft_idx;
    static constexpr std::array<int, ne_soft>    eq_soft_idx      = Dims::eq_soft_idx;
    static constexpr std::array<int, nl_soft>    lin_soft_idx     = Dims::lin_soft_idx;
    static constexpr std::array<int, ng_t_soft>  terminal_ineq_soft_idx =
        Dims::terminal_ineq_soft_idx;
    static constexpr std::array<int, ne_t_soft>  terminal_eq_soft_idx =
        Dims::terminal_eq_soft_idx;
    static constexpr std::array<int, nl_t_soft>  terminal_lin_soft_idx =
        Dims::terminal_lin_soft_idx;
    static constexpr std::array<int, nbx_soft>   state_box_soft_idx   =
        Dims::state_box_soft_idx;
    static constexpr std::array<int, nbu_soft>   control_box_soft_idx =
        Dims::control_box_soft_idx;
    static constexpr std::array<int, nbx_t_soft> terminal_state_box_soft_idx =
        Dims::terminal_state_box_soft_idx;

    // ---------------------------------------------------------------
    // Fixed-size vector / matrix types
    // ---------------------------------------------------------------
    using scalar_t = Scalar;

    using state_t   = Eigen::Matrix<Scalar, nx, 1>;
    using control_t = Eigen::Matrix<Scalar, nu, 1>;
    using ineq_t    = Eigen::Matrix<Scalar, ng, 1>;
    using eq_t      = Eigen::Matrix<Scalar, ne, 1>;
    using lin_t     = Eigen::Matrix<Scalar, nl, 1>;

    using ineq_term_t = Eigen::Matrix<Scalar, ng_t, 1>;
    using eq_term_t   = Eigen::Matrix<Scalar, ne_t, 1>;
    using lin_term_t  = Eigen::Matrix<Scalar, nl_t, 1>;

    // combined (x, u) quantities, layout [x; u]
    using stage_grad_t = Eigen::Matrix<Scalar, nx + nu, 1>;
    using stage_hess_t = Eigen::Matrix<Scalar, nx + nu, nx + nu>;
    using term_grad_t  = Eigen::Matrix<Scalar, nx, 1>;
    using term_hess_t  = Eigen::Matrix<Scalar, nx, nx>;

    // dynamics Jacobian blocks  (f_k: (x, u) -> x_{k+1})
    using dyn_df_dx_t = Eigen::Matrix<Scalar, nx, nx>;
    using dyn_df_du_t = Eigen::Matrix<Scalar, nx, nu>;

    // multiplier-contracted Hessians, [x; u] layout (D^2 L contracted with
    // the group multiplier); symmetric by contract
    using dyn_hess_t         = Eigen::Matrix<Scalar, nx + nu, nx + nu>;
    using constr_hess_t      = Eigen::Matrix<Scalar, nx + nu, nx + nu>;
    using term_constr_hess_t = Eigen::Matrix<Scalar, nx, nx>;

    // constraint Jacobians
    using ineq_dg_dx_t = Eigen::Matrix<Scalar, ng, nx>;
    using ineq_dg_du_t = Eigen::Matrix<Scalar, ng, nu>;
    using eq_de_dx_t   = Eigen::Matrix<Scalar, ne, nx>;
    using eq_de_du_t   = Eigen::Matrix<Scalar, ne, nu>;
    using ineq_term_dg_dx_t = Eigen::Matrix<Scalar, ng_t, nx>;
    using eq_term_de_dx_t   = Eigen::Matrix<Scalar, ne_t, nx>;

    // soft-row penalty weights (one {quad, lin} pair per soft row,
    // *_soft_idx order); see SoftPenalty
    using ineq_pen_t      = SoftPenalty<Scalar, ng_soft>;
    using eq_pen_t        = SoftPenalty<Scalar, ne_soft>;
    using ineq_term_pen_t = SoftPenalty<Scalar, ng_t_soft>;
    using eq_term_pen_t   = SoftPenalty<Scalar, ne_t_soft>;

    // constraint specs
    using state_box_t      = BoxSpec<Scalar, nbx>;
    using control_box_t    = BoxSpec<Scalar, nbu>;
    using term_state_box_t = BoxSpec<Scalar, nbx_t>;
    using stage_linear_t    = LinearSpec<Scalar, nl, nx, nu>;
    using term_linear_t    = TerminalLinearSpec<Scalar, nl_t, nx>;

    // parameter vector and its p-derivatives (NLP-level parameter
    // sensitivity; only used when np > 0).  Derivative matrices are
    // column-stacked in p: column j holds the (flattened) derivative
    // w.r.t. p_j, i.e. for a constraint Jacobian J of size (nrow x nvar),
    // dJ_dp(r * nvar + i, j) = dJ(r, i)/dp_j.
    using param_t                = Eigen::Matrix<Scalar, np, 1>;
    using dyn_df_dp_t            = Eigen::Matrix<Scalar, nx, np>;
    using dyn_dA_dp_t            = Eigen::Matrix<Scalar, nx * nx, np>;
    using dyn_dB_dp_t            = Eigen::Matrix<Scalar, nx * nu, np>;
    using stage_grad_dp_t        = Eigen::Matrix<Scalar, nx + nu, np>;
    using term_grad_dp_t         = Eigen::Matrix<Scalar, nx, np>;
    using ineq_dg_dx_dp_t        = Eigen::Matrix<Scalar, ng * nx, np>;
    using ineq_dg_du_dp_t        = Eigen::Matrix<Scalar, ng * nu, np>;
    using eq_de_dx_dp_t          = Eigen::Matrix<Scalar, ne * nx, np>;
    using eq_de_du_dp_t          = Eigen::Matrix<Scalar, ne * nu, np>;
    using ineq_term_dg_dx_dp_t   = Eigen::Matrix<Scalar, ng_t * nx, np>;
    using eq_term_de_dx_dp_t     = Eigen::Matrix<Scalar, ne_t * nx, np>;
    using ineq_dg_dp_t           = Eigen::Matrix<Scalar, ng, np>;
    using eq_de_dp_t             = Eigen::Matrix<Scalar, ne, np>;
    using ineq_term_dg_dp_t      = Eigen::Matrix<Scalar, ng_t, np>;
    using eq_term_de_dp_t        = Eigen::Matrix<Scalar, ne_t, np>;
    using x0_dp_t                = Eigen::Matrix<Scalar, nx, np>;
    using stage_state_box_sensp_t  = BoxSensP<Scalar, nbx, np>;
    using stage_control_box_sensp_t = BoxSensP<Scalar, nbu, np>;
    using term_state_box_sensp_t   = BoxSensP<Scalar, nbx_t, np>;
    using stage_linear_sensp_t     = LinearSensP<Scalar, nl, nx, nu, np>;
    using term_linear_sensp_t      = LinearSensP<Scalar, nl_t, nx, 0, np>;

    // The horizon N is *not* a property of the problem: it is a per-solve
    // quantity owned by the Solution (see below), whose trajectories have
    // the corresponding extents. Model functions are parameterised by the
    // stage index k and are independent of N.
    Problem() = default;

    // ---------------------------------------------------------------
    // Interface stubs
    //
    // Each stub carries the exact signature of the interface function.
    // A concrete problem hides the stub by declaring a member of the same
    // name and signature; calls on the concrete type then dispatch
    // statically to the real implementation. If a function is missing (or
    // misspelled / missigned), the inherited stub is instantiated and
    // compilation fails at the call site.
    // ---------------------------------------------------------------

    // initial state (fixed x_0; absent when !fixed_initial_state)
    state_t initial_state() const
    {
        if constexpr (fixed_initial_state)
        {
            static_assert(false,
                "ocp::Problem interface: implement "
                "'state_t initial_state() const' in the concrete problem.");
        }
        else
        {
            static_assert(false,
                "ocp::Problem: Dims::fixed_initial_state is false, so "
                "initial_state() is not part of the interface (the initial "
                "state is a decision variable).");
        }
        // unreachable; the static_assert above fires first
        return {};
    }

    // dynamics (k = 0..N-1)
    state_t dynamics_next_state(int /*k*/, const state_t& /*x*/,
                                const control_t& /*u*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'state_t dynamics_next_state(int k, const state_t&, const control_t&) const' "
            "in the concrete problem (or set nx to 0).");
        // unreachable; the static_assert above fires first
        return {};
    }
    void dynamics_value_jac(int /*k*/, const state_t& /*x*/,
                            const control_t& /*u*/,
                            state_t& /*x_next*/,
                            dyn_df_dx_t& /*df_dx*/,
                            dyn_df_du_t& /*df_du*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void dynamics_value_jac(int k, const state_t&, const control_t&, "
            "state_t& x_next, dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const' "
            "in the concrete problem.");
    }
    void dynamics_value_jac_hess(int /*k*/, const state_t& /*x*/,
                                 const control_t& /*u*/,
                                 const state_t& /*lam*/,
                                 state_t& /*x_next*/,
                                 dyn_df_dx_t& /*df_dx*/,
                                 dyn_df_du_t& /*df_du*/,
                                 dyn_hess_t& /*hess*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void dynamics_value_jac_hess(int k, const state_t&, "
            "const control_t&, const state_t& lam, state_t& x_next, "
            "dyn_df_dx_t& df_dx, dyn_df_du_t& df_du, dyn_hess_t& hess) const' "
            "in the concrete problem (multiplier-contracted Hessian, "
            "[x; u] layout; hess v = sum_i lam_i D^2 f_i v).");
    }

    // stage cost (k = 0..N-1)
    Scalar stage_cost_value(int /*k*/, const state_t& /*x*/,
                            const control_t& /*u*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'Scalar stage_cost_value(int k, const state_t&, const control_t&) const' "
            "in the concrete problem.");
        // unreachable; the static_assert above fires first
        return {};
    }
    void stage_cost_value_grad(int /*k*/, const state_t& /*x*/,
                               const control_t& /*u*/,
                               Scalar& /*value*/,
                               stage_grad_t& /*grad*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void stage_cost_value_grad(int k, const state_t&, const control_t&, "
            "Scalar& value, stage_grad_t& grad) const' in the concrete problem "
            "(layout [g_x; g_u]).");
    }
    void stage_cost_value_grad_hess(int /*k*/, const state_t& /*x*/,
                                    const control_t& /*u*/,
                                    Scalar& /*value*/,
                                    stage_grad_t& /*grad*/,
                                    stage_hess_t& /*hess*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void stage_cost_value_grad_hess(int k, const state_t&, "
            "const control_t&, Scalar& value, stage_grad_t& grad, "
            "stage_hess_t& hess) const' in the concrete problem "
            "(block layout [H_xx H_xu; H_ux H_uu], exact or Gauss-Newton, "
            "user's choice).");
    }

    // terminal cost (stage N, state only)
    Scalar terminal_cost_value(const state_t& /*x*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'Scalar terminal_cost_value(const state_t&) const' in the concrete problem.");
        // unreachable; the static_assert above fires first
        return {};
    }
    void terminal_cost_value_grad(const state_t& /*x*/,
                                  Scalar& /*value*/,
                                  term_grad_t& /*grad*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void terminal_cost_value_grad(const state_t&, Scalar& value, "
            "term_grad_t& grad) const' in the concrete problem.");
    }
    void terminal_cost_value_grad_hess(const state_t& /*x*/,
                                       Scalar& /*value*/,
                                       term_grad_t& /*grad*/,
                                       term_hess_t& /*hess*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void terminal_cost_value_grad_hess(const state_t&, Scalar& value, "
            "term_grad_t& grad, term_hess_t& hess) const' in the concrete "
            "problem (block layout [H_xx], exact or Gauss-Newton, user's "
            "choice).");
    }

    // stage constraints (k = 0..N-1)
    ineq_t stage_inequality_value(int /*k*/, const state_t& /*x*/,
                                  const control_t& /*u*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'ineq_t stage_inequality_value(int k, const state_t&, "
            "const control_t&) const' in the concrete problem "
            "(g(x,u) <= 0), or set ng to 0.");
        // unreachable; the static_assert above fires first
        return {};
    }
    void stage_inequality_value_jac(int /*k*/, const state_t& /*x*/,
                                    const control_t& /*u*/,
                                    ineq_t& /*g*/,
                                    ineq_dg_dx_t& /*g_dx*/,
                                    ineq_dg_du_t& /*g_du*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void stage_inequality_value_jac(int k, const state_t&, "
            "const control_t&, ineq_t& g, ineq_dg_dx_t& g_dx, "
            "ineq_dg_du_t& g_du) const' in the concrete problem.");
    }
    void stage_inequality_value_jac_hess(int /*k*/, const state_t& /*x*/,
                                         const control_t& /*u*/,
                                         const ineq_t& /*lam*/,
                                         ineq_t& /*g*/,
                                         ineq_dg_dx_t& /*g_dx*/,
                                         ineq_dg_du_t& /*g_du*/,
                                         constr_hess_t& /*hess*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void stage_inequality_value_jac_hess(int k, const state_t&, "
            "const control_t&, const ineq_t& lam, ineq_t& g, "
            "ineq_dg_dx_t& g_dx, ineq_dg_du_t& g_du, "
            "constr_hess_t& hess) const' in the concrete problem "
            "(multiplier-contracted Hessian, [x; u] layout), or set ng to 0.");
    }
    eq_t stage_equality_value(int /*k*/, const state_t& /*x*/,
                              const control_t& /*u*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'eq_t stage_equality_value(int k, const state_t&, "
            "const control_t&) const' in the concrete problem "
            "(e(x,u) == 0), or set ne to 0.");
        // unreachable; the static_assert above fires first
        return {};
    }
    void stage_equality_value_jac(int /*k*/, const state_t& /*x*/,
                                  const control_t& /*u*/,
                                  eq_t& /*e*/,
                                  eq_de_dx_t& /*e_dx*/,
                                  eq_de_du_t& /*e_du*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void stage_equality_value_jac(int k, const state_t&, "
            "const control_t&, eq_t& e, eq_de_dx_t& e_dx, "
            "eq_de_du_t& e_du) const' in the concrete problem.");
    }
    void stage_equality_value_jac_hess(int /*k*/, const state_t& /*x*/,
                                       const control_t& /*u*/,
                                       const eq_t& /*lam*/,
                                       eq_t& /*e*/,
                                       eq_de_dx_t& /*e_dx*/,
                                       eq_de_du_t& /*e_du*/,
                                       constr_hess_t& /*hess*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void stage_equality_value_jac_hess(int k, const state_t&, "
            "const control_t&, const eq_t& lam, eq_t& e, eq_de_dx_t& e_dx, "
            "eq_de_du_t& e_du, constr_hess_t& hess) const' in the "
            "concrete problem (multiplier-contracted Hessian, "
            "[x; u] layout), or set ne to 0.");
    }
    stage_linear_t stage_linear_constr(int /*k*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'stage_linear_t stage_linear_constr(int k) const' in the concrete problem "
            "(A, B with lo <= A x + B u <= hi), or set nl to 0.");
        // unreachable; the static_assert above fires first
        return {};
    }
    state_box_t stage_state_box_constr(int /*k*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'state_box_t stage_state_box_constr(int k) const' in the concrete problem.");
        // unreachable; the static_assert above fires first
        return {};
    }
    control_box_t stage_control_box_constr(int /*k*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'control_box_t stage_control_box_constr(int k) const' in the concrete problem.");
        // unreachable; the static_assert above fires first
        return {};
    }
    ineq_pen_t stage_inequality_constr_soft_penalty(int /*k*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'ineq_pen_t stage_inequality_constr_soft_penalty(int k) const' in the "
            "concrete problem (one weight per soft row, in ineq_soft_idx order), "
            "or leave ineq_soft_idx empty.");
        // unreachable; the static_assert above fires first
        return {};
    }
    eq_pen_t stage_equality_constr_soft_penalty(int /*k*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'eq_pen_t stage_equality_constr_soft_penalty(int k) const' in the "
            "concrete problem (one weight per soft row, in eq_soft_idx order), "
            "or leave eq_soft_idx empty.");
        // unreachable; the static_assert above fires first
        return {};
    }

    // terminal constraints (stage N, state only)
    term_state_box_t terminal_state_box_constr() const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'term_state_box_t terminal_state_box_constr() const' in the concrete problem.");
        // unreachable; the static_assert above fires first
        return {};
    }
    ineq_term_t terminal_inequality_value(const state_t& /*x*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'ineq_term_t terminal_inequality_value(const state_t&) const' "
            "(g(x) <= 0) in the concrete problem, or set ng_t to 0.");
        // unreachable; the static_assert above fires first
        return {};
    }
    void terminal_inequality_value_jac(const state_t& /*x*/,
                                       ineq_term_t& /*g*/,
                                       ineq_term_dg_dx_t& /*g_dx*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void terminal_inequality_value_jac(const state_t&, "
            "ineq_term_t& g, ineq_term_dg_dx_t& g_dx) const' in the "
            "concrete problem.");
    }
    void terminal_inequality_value_jac_hess(const state_t& /*x*/,
                                           const ineq_term_t& /*lam*/,
                                           ineq_term_t& /*g*/,
                                           ineq_term_dg_dx_t& /*g_dx*/,
                                           term_constr_hess_t& /*hess*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void terminal_inequality_value_jac_hess(const state_t&, "
            "const ineq_term_t& lam, ineq_term_t& g, ineq_term_dg_dx_t& g_dx, "
            "term_constr_hess_t& hess) const' in the concrete problem "
            "(multiplier-contracted Hessian, state only), or set ng_t to 0.");
    }
    eq_term_t terminal_equality_value(const state_t& /*x*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'eq_term_t terminal_equality_value(const state_t&) const' "
            "(e(x) == 0) in the concrete problem, or set ne_t to 0.");
        // unreachable; the static_assert above fires first
        return {};
    }
    void terminal_equality_value_jac(const state_t& /*x*/,
                                     eq_term_t& /*e*/,
                                     eq_term_de_dx_t& /*e_dx*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void terminal_equality_value_jac(const state_t&, "
            "eq_term_t& e, eq_term_de_dx_t& e_dx) const' in the "
            "concrete problem.");
    }
    void terminal_equality_value_jac_hess(const state_t& /*x*/,
                                         const eq_term_t& /*lam*/,
                                         eq_term_t& /*e*/,
                                         eq_term_de_dx_t& /*e_dx*/,
                                         term_constr_hess_t& /*hess*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void terminal_equality_value_jac_hess(const state_t&, "
            "const eq_term_t& lam, eq_term_t& e, eq_term_de_dx_t& e_dx, "
            "term_constr_hess_t& hess) const' in the concrete problem "
            "(multiplier-contracted Hessian, state only), or set ne_t to 0.");
    }
    term_linear_t terminal_linear_constr() const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'term_linear_t terminal_linear_constr() const' in the concrete problem "
            "(A with lo <= A x <= hi), or set nl_t to 0.");
        // unreachable; the static_assert above fires first
        return {};
    }
    ineq_term_pen_t terminal_inequality_constr_soft_penalty() const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'ineq_term_pen_t terminal_inequality_constr_soft_penalty() const' in the "
            "concrete problem (one weight per soft row, in terminal_ineq_soft_idx "
            "order), or leave terminal_ineq_soft_idx empty.");
        // unreachable; the static_assert above fires first
        return {};
    }
    eq_term_pen_t terminal_equality_constr_soft_penalty() const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'eq_term_pen_t terminal_equality_constr_soft_penalty() const' in the "
            "concrete problem (one weight per soft row, in terminal_eq_soft_idx "
            "order), or leave terminal_eq_soft_idx empty.");
        // unreachable; the static_assert above fires first
        return {};
    }

    // ---------------------------------------------------------------
    // Parameter-vector sensitivity (np > 0)
    //
    // p-derivatives of all model quantities, evaluated at the current
    // model state. Required by the solver only when np > 0; the solver
    // gates every call additionally on the group dimension, so entries
    // of zero-size groups (e.g. ne == 0) may stay unimplemented.
    // ---------------------------------------------------------------

    void dynamics_sens_p(int /*k*/, const state_t& /*x*/,
                         const control_t& /*u*/,
                         dyn_df_dp_t& /*f_p*/,
                         dyn_dA_dp_t& /*A_p*/,
                         dyn_dB_dp_t& /*B_p*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void dynamics_sens_p(int k, const state_t&, const control_t&, "
            "dyn_df_dp_t& f_p, dyn_dA_dp_t& A_p, dyn_dB_dp_t& B_p) const' "
            "in the concrete problem (requires Dims::np > 0).");
    }
    void stage_cost_sens_p(int /*k*/, const state_t& /*x*/,
                           const control_t& /*u*/,
                           param_t& /*L_p*/,
                           stage_grad_dp_t& /*grad_p*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void stage_cost_sens_p(int k, const state_t&, const control_t&, "
            "param_t& L_p, stage_grad_dp_t& grad_p) const' in the concrete "
            "problem (requires Dims::np > 0).");
    }
    void terminal_cost_sens_p(const state_t& /*x*/,
                              param_t& /*L_p*/,
                              term_grad_dp_t& /*grad_p*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void terminal_cost_sens_p(const state_t&, param_t& L_p, "
            "term_grad_dp_t& grad_p) const' in the concrete problem "
            "(requires Dims::np > 0).");
    }
    void stage_inequality_sens_p(int /*k*/, const state_t& /*x*/,
                                 const control_t& /*u*/,
                                 ineq_dg_dp_t& /*g_p*/,
                                 ineq_dg_dx_dp_t& /*gdx_p*/,
                                 ineq_dg_du_dp_t& /*gdu_p*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void stage_inequality_sens_p(int k, const state_t&, "
            "const control_t&, ineq_dg_dp_t& g_p, ineq_dg_dx_dp_t& gdx_p, "
            "ineq_dg_du_dp_t& gdu_p) const' in the concrete problem "
            "(requires Dims::np > 0), or set ng to 0.");
    }
    void stage_equality_sens_p(int /*k*/, const state_t& /*x*/,
                               const control_t& /*u*/,
                               eq_de_dp_t& /*e_p*/,
                               eq_de_dx_dp_t& /*edx_p*/,
                               eq_de_du_dp_t& /*edu_p*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void stage_equality_sens_p(int k, const state_t&, "
            "const control_t&, eq_de_dp_t& e_p, eq_de_dx_dp_t& edx_p, "
            "eq_de_du_dp_t& edu_p) const' in the concrete problem "
            "(requires Dims::np > 0), or set ne to 0.");
    }
    void terminal_inequality_sens_p(const state_t& /*x*/,
                                    ineq_term_dg_dp_t& /*g_p*/,
                                    ineq_term_dg_dx_dp_t& /*gdx_p*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void terminal_inequality_sens_p(const state_t&, "
            "ineq_term_dg_dp_t& g_p, ineq_term_dg_dx_dp_t& gdx_p) const' "
            "in the concrete problem "
            "(requires Dims::np > 0), or set ng_t to 0.");
    }
    void terminal_equality_sens_p(const state_t& /*x*/,
                                  eq_term_de_dp_t& /*e_p*/,
                                  eq_term_de_dx_dp_t& /*edx_p*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void terminal_equality_sens_p(const state_t&, "
            "eq_term_de_dp_t& e_p, eq_term_de_dx_dp_t& edx_p) const' "
            "in the concrete problem "
            "(requires Dims::np > 0), or set ne_t to 0.");
    }
    stage_linear_sensp_t stage_linear_sens_p(int /*k*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'LinearSensP<scalar_t, nl, nx, nu, np> stage_linear_sens_p(int k) const' "
            "in the concrete problem (requires Dims::np > 0), or set nl to 0.");
        // unreachable; the static_assert above fires first
        return {};
    }
    term_linear_sensp_t terminal_linear_sens_p() const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'LinearSensP<scalar_t, nl_t, nx, 0, np> terminal_linear_sens_p() const' "
            "in the concrete problem (requires Dims::np > 0), or set nl_t to 0.");
        // unreachable; the static_assert above fires first
        return {};
    }
    stage_state_box_sensp_t stage_state_box_sens_p(int /*k*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'BoxSensP<scalar_t, nbx, np> stage_state_box_sens_p(int k) const' in "
            "the concrete problem (requires Dims::np > 0), or set "
            "state_box_idx to an empty set.");
        // unreachable; the static_assert above fires first
        return {};
    }
    stage_control_box_sensp_t stage_control_box_sens_p(int /*k*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'BoxSensP<scalar_t, nbu, np> stage_control_box_sens_p(int k) const' in "
            "the concrete problem (requires Dims::np > 0), or set "
            "control_box_idx to an empty set.");
        // unreachable; the static_assert above fires first
        return {};
    }
    term_state_box_sensp_t terminal_state_box_sens_p() const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'BoxSensP<scalar_t, nbx_t, np> terminal_state_box_sens_p() const' in "
            "the concrete problem (requires Dims::np > 0), or set "
            "terminal_state_box_idx to an empty set.");
        // unreachable; the static_assert above fires first
        return {};
    }
    x0_dp_t initial_state_sens_p() const
    {
        if constexpr (fixed_initial_state)
        {
            static_assert(false,
                "ocp::Problem interface: implement "
                "'x0_dp_t initial_state_sens_p() const' in the concrete "
                "problem (requires Dims::np > 0 and "
                "Dims::fixed_initial_state).");
        }
        else
        {
            static_assert(false,
                "ocp::Problem: Dims::fixed_initial_state is false or Dims::np is "
                "0, so initial_state_sens_p() is not part of the interface.");
        }
        // unreachable; the static_assert above fires first
        return {};
    }
};

// =========================================================================
//  Solution
// =========================================================================

/// Solve status shared by all solver algorithms.
enum class Status
{
    kUnset,          // not solved yet
    kSolved,         // converged to an (approximate) KKT point
    kMaxIterations,  // iteration limit reached without convergence
    kInfeasible,     // (in)feasibility / unboundedness detected
    kQpFailure,      // inner QP solve failed
    kMinStep,        // step length fell below minimum threshold
    kUnbounded,      // problem detected as unbounded
    kNanDetected,    // NaN encountered during iteration
    kAborted,        // internal solver error
    kTimeout         // wall-clock timeout reached
};

/// Solver-independent solution (and warm-start) container.
///
/// The same object is used both as the initial iterate supplied to a
/// solver (primal variables and, optionally, multipliers) and as the
/// result the solver fills in. Solver-internal quantities (slack
/// variables, barrier/regularization parameters, multiplier orderings,
/// ...) are *not* stored here; each solver keeps its own
/// (ARCHITECTURE.md sec. 2 "Separation of internal variables").
///
/// `NH` is the horizon: `Eigen::Dynamic` (default) keeps it a runtime
/// quantity, set via the constructor or `resize` (heap-allocated
/// trajectories). A non-negative integer fixes it at compile time: the
/// trajectories become fixed-extent (stack-allocated, zeroed), `N` is
/// initialized from `NH`, and the constructor argument / `resize` must
/// equal `NH`.
template <class ProblemDerived, int NH = Eigen::Dynamic>
struct Solution
{
    static_assert(NH == Eigen::Dynamic || NH >= 0,
        "ocp::Solution: the horizon must be a non-negative integer or "
        "Eigen::Dynamic.");

    using P           = ProblemDerived;
    using scalar_t    = typename P::scalar_t;
    using state_t     = typename P::state_t;
    using control_t   = typename P::control_t;
    using ineq_t      = typename P::ineq_t;
    using eq_t        = typename P::eq_t;
    using lin_t       = typename P::lin_t;
    using ineq_term_t = typename P::ineq_term_t;
    using eq_term_t   = typename P::eq_term_t;
    using lin_term_t  = typename P::lin_term_t;

    using box_state_lam_t   = Eigen::Matrix<scalar_t, 2 * P::nbx, 1>;  // [lower; upper]
    using box_control_lam_t = Eigen::Matrix<scalar_t, 2 * P::nbu, 1>;

    // trajectory types: fixed-extent when the horizon is compile-time
    using x_traj_t         = Trajectory<state_t,         detail::traj_extent<NH, 1>()>;
    using u_traj_t         = Trajectory<control_t,       detail::traj_extent<NH, 0>()>;
    using lam_dyn_traj_t   = Trajectory<state_t,         detail::traj_extent<NH, 0>()>;
    using lam_ineq_traj_t  = Trajectory<ineq_t,          detail::traj_extent<NH, 0>()>;
    using lam_eq_traj_t    = Trajectory<eq_t,            detail::traj_extent<NH, 0>()>;
    using lam_lin_traj_t   = Trajectory<lin_t,           detail::traj_extent<NH, 0>()>;
    using lam_box_x_traj_t = Trajectory<box_state_lam_t, detail::traj_extent<NH, 1>()>;
    using lam_box_u_traj_t = Trajectory<box_control_lam_t, detail::traj_extent<NH, 0>()>;

    Solution() : N(NH == Eigen::Dynamic ? 0 : NH) {}
    explicit Solution(int n_stages) { resize(n_stages); }

    int N = 0;  // horizon (definitive; the trajectories have its extents)

    // primal variables
    x_traj_t   x;  // N + 1
    u_traj_t   u;  // N

    // KKT multipliers (warm-startable; zero until a solver fills them)
    lam_dyn_traj_t   lambda_dyn;          // N,    dynamics
    lam_ineq_traj_t  lambda_ineq_stage;   // N
    lam_eq_traj_t    lambda_eq_stage;     // N
    lam_lin_traj_t   lambda_lin_stage;    // N
    ineq_term_t      lambda_ineq_term;    // terminal
    eq_term_t        lambda_eq_term;      // terminal
    lin_term_t       lambda_lin_term;     // terminal
    lam_box_x_traj_t lambda_box_state;    // N + 1
    lam_box_u_traj_t lambda_box_control;  // N

    scalar_t cost_value = 0;  // objective at (x, u)
    Status   status     = Status::kUnset;

    /// (Re)allocate all trajectories for a horizon of `n_stages` stages.
    /// Fixed-horizon mode: no allocation happens; `n_stages` must equal
    /// the compile-time horizon `NH` (checked in debug builds).
    void resize(int n_stages)
    {
        if constexpr (NH == Eigen::Dynamic)
        {
            N = n_stages;
            x.resize(N + 1);
            u.resize(N);
            lambda_dyn.resize(N);
            lambda_ineq_stage.resize(N);
            lambda_eq_stage.resize(N);
            lambda_lin_stage.resize(N);
            lambda_box_state.resize(N + 1);
            lambda_box_control.resize(N);
        }
        else
        {
            assert(n_stages == NH);
            N = NH;
        }
    }
};

}  // namespace ocp

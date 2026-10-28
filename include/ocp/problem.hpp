// ocp/problem.hpp
//
// Problem interface of the ocp++ framework.
//
// A *problem* is the mathematical definition of a discrete-time optimal
// control problem (OCP), decoupled from any solver algorithm:
//
//     min_{x, u}   sum_{k=0}^{N-1} L_k(x_k, u_k) + L_N(x_N)
//     subject to   x_{k+1} = f_k(x_k, u_k),      k = 0..N-1
//                  box, linear and general (in)equality constraints
//
// A concrete problem inherits from ocp::Problem<Dims>, where Dims is a
// tag struct carrying the structural dimensions and index sets:
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
//            static constexpr std::array<int, 1> ineq_soft_idx = {0};
//            static constexpr std::array<int, 0> eq_soft_idx   = {};
//            // ... plus lin_soft_idx, state_box_soft_idx,
//            //     control_box_soft_idx, terminal_{ineq,eq,lin}_soft_idx,
//            //     terminal_state_box_soft_idx
//        };
//
// The box index arrays select which state/control components each box row
// bounds; their size is the number of active box rows (nbx, nbu, nbx_t),
// which ocp::Problem derives. The `*_soft_idx` arrays mark which rows of
// each constraint group are soft (relaxable by a solver slack); their
// size is the number of soft rows in the group (empty = hard).
//
// Dims flags:
//  - fixed_initial_state: true - x_0 is a fixed given supplied by
//    initial_state(); false - x_0 is a decision variable (e.g. MHE).
//  - has_dynamics_hess: true - the solver assembles the exact Lagrangian
//    QP Hessian (cost plus multiplier-contracted dynamics Hessian);
//    false - Gauss-Newton Hessian, where the dynamics enter the QP only
//    through the linearized BA matrix.
//  - has_constr_hess: true - the multiplier-contracted constraint
//    Hessians are added to the QP Hessian; false - omitted (valid when
//    the constraint maps are linear).
//
// All model calls dispatch statically on the concrete problem type. The
// base class re-exposes the dimensions, derives all fixed-size Eigen
// types, and carries one stub per interface function (see the interface
// contract on ocp::Problem): a missing or missigned override fails at
// the call site via static_assert. The horizon N is a per-solve quantity
// owned by ocp::Solution, not by the problem.

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
/// Extent conventions for a horizon N (see docs/ARCHITECTURE.md, sec. 5):
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
/// the outside. For each soft side of row i the solver adds
///
///     0.5 * quad[i] * s[i]^2  +  lin[i] * s[i]
///
/// to the objective, where `quad` is the Hessian (second-order) weight and
/// `lin` the gradient (first-order) weight. The standard soft constraint
/// uses `quad = w`, `lin = 0`. Only a diagonal (per-side) Hessian weight
/// is supported, since the QP solver assumes a diagonal slack Hessian.
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
// NOTE: Eigen 3.3 `Matrix()` is a user-provided default constructor, so
// value-initialization (`T{}` / `{}` NSDMI) does NOT zero the coefficients.
// All Eigen members in the spec/sens structs below are therefore
// initialized with an explicit `::Zero()` NSDMI; never rely on `{}`.
template <class Scalar, int Dim>
struct BoxSpec
{
    Eigen::Matrix<Scalar, Dim, 1> lo =
        Eigen::Matrix<Scalar, Dim, 1>::Zero();  // lower bounds
    Eigen::Matrix<Scalar, Dim, 1> hi =
        Eigen::Matrix<Scalar, Dim, 1>::Zero();  // upper bounds
    SoftPenalty<Scalar, Dim>      penalty;      // per-row slack penalty
};

/// Linear two-sided stage constraint  lo <= A x + B u <= hi  (stage k);
/// an additive constant, if any, is absorbed into the bounds.
template <class Scalar, int NRow, int Nx, int Nu>
struct LinearSpec
{
    Eigen::Matrix<Scalar, NRow, Nx> A =
        Eigen::Matrix<Scalar, NRow, Nx>::Zero();
    Eigen::Matrix<Scalar, NRow, Nu> B =
        Eigen::Matrix<Scalar, NRow, Nu>::Zero();
    BoxSpec<Scalar, NRow>           bounds{};
};

/// Linear two-sided terminal constraint  lo <= A x <= hi  (no control);
/// an additive constant, if any, is absorbed into the bounds.
template <class Scalar, int NRow, int Nx>
struct TerminalLinearSpec
{
    Eigen::Matrix<Scalar, NRow, Nx> A =
        Eigen::Matrix<Scalar, NRow, Nx>::Zero();
    BoxSpec<Scalar, NRow>           bounds{};
};

/// Parameter-vector (p) derivatives of a box spec's bounds (sensitivity
/// analysis, `*_sens_p` interface entries).  One column per parameter
/// component: `dlo_dp(i, j) = d(lo_i)/dp_j`.
template <class Scalar, int NRow, int NParam>
struct BoxSensP
{
    Eigen::Matrix<Scalar, NRow, NParam> dlo_dp =
        Eigen::Matrix<Scalar, NRow, NParam>::Zero();
    Eigen::Matrix<Scalar, NRow, NParam> dhi_dp =
        Eigen::Matrix<Scalar, NRow, NParam>::Zero();
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
    Eigen::Matrix<Scalar, NRow * Nx, NParam> dA_dp =
        Eigen::Matrix<Scalar, NRow * Nx, NParam>::Zero();
    Eigen::Matrix<Scalar, NRow * Nu, NParam> dB_dp =
        Eigen::Matrix<Scalar, NRow * Nu, NParam>::Zero();
    Eigen::Matrix<Scalar, NRow, NParam> dlo_dp =
        Eigen::Matrix<Scalar, NRow, NParam>::Zero();
    Eigen::Matrix<Scalar, NRow, NParam> dhi_dp =
        Eigen::Matrix<Scalar, NRow, NParam>::Zero();
};

// =========================================================================
//  Problem
// =========================================================================

/// Template interface of a discrete-time OCP.
///
/// The concrete problem supplies its structural dimensions via the
/// `Dims` tag and implements the model functions listed in the interface
/// contract below. The base class re-exposes the dimensions as
/// `static constexpr` members, derives all fixed-size Eigen types from
/// them, and provides one stub per interface function; if a function is
/// missing or missigned in the derived class, the inherited stub is
/// instantiated and compilation fails with a descriptive message.
///
/// Conventions
/// -----------
///  - Stage index k is 0..N-1 for dynamics, stage cost and stage
///    constraints; the terminal stage k = N is served by the dedicated
///    `terminal_*` functions, which depend on the state only (there is
///    no u_N).
///  - In combined (x, u) vectors and matrices the state is listed first:
///    gradients have layout [g_x; g_u], Hessians the block layout
///    [H_xx H_xu; H_ux H_uu].
///  - All model functions are `const` and take no parameter argument;
///    parameters are user-managed state of the concrete class.
///  - The `*_value_jac_hess` entries return the multiplier-contracted
///    Hessian in one call: `hess * v = sum_i lam_i * D^2(·)_i * v`,
///    where `lam` is the net KKT multiplier of the corresponding group
///    (e.g. `lambda_dyn[k]` for dynamics).
///  - Second-order information is user-supplied (exact, Gauss-Newton or
///    any valid approximation); `Dims::has_dynamics_hess` and
///    `Dims::has_constr_hess` tell the solver which `*_value_jac_hess`
///    entries exist and which QP Hessian model to assemble.
///
/// Interface contract (implement each function below unless the
/// corresponding dimension is zero, in which case solvers never call it
/// and the stub may stay unimplemented):
///
///   initial state (fixed x_0; required only when Dims::fixed_initial_state)
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
///   terminal cost (stage N, state only); fused evaluation entries
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
///     state_box_t stage_state_box_constr(int k) const;
///     control_box_t stage_control_box_constr(int k) const;
///     ineq_pen_t stage_inequality_constr_soft_penalty(int k) const;
///     eq_pen_t   stage_equality_constr_soft_penalty(int k) const;
///
///   terminal constraints (stage N, state only); fused evaluation entries
///     term_state_box_t terminal_state_box_constr() const;
///     ineq_term_t terminal_inequality_value(const state_t& x) const;  // g <= 0
///     void terminal_inequality_value_jac(const state_t& x, ineq_term_t& g,
///                                        ineq_term_dg_dx_t& g_dx) const;
///     void terminal_inequality_value_jac_hess(const state_t& x,
///                                             const ineq_term_t& lam,
///                                             ineq_term_t& g,
///                                             ineq_term_dg_dx_t& g_dx,
///                                             term_constr_hess_t& hess) const;
///     eq_term_t terminal_equality_value(const state_t& x) const;       // e == 0
///     void terminal_equality_value_jac(const state_t& x, eq_term_t& e,
///                                      eq_term_de_dx_t& e_dx) const;
///     void terminal_equality_value_jac_hess(const state_t& x,
///                                           const eq_term_t& lam,
///                                           eq_term_t& e,
///                                           eq_term_de_dx_t& e_dx,
///                                           term_constr_hess_t& hess) const;
///     term_linear_t terminal_linear_constr() const;
///     ineq_term_pen_t terminal_inequality_constr_soft_penalty() const;
///     eq_term_pen_t   terminal_equality_constr_soft_penalty() const;
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
///   (docs/ARCHITECTURE.md sec. 6).
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
///   the fused `*_sens_p` derivative entries listed above. They provide
///   the p-derivatives of the model values, cost gradients and constraint
///   Jacobians/bounds required by NLP-level sensitivity analysis (see
///   AcadosSolver::sens_param). When `np == 0` the entries are optional
///   and are never called.
/// - Extensions anticipated beyond this draft (not yet part of the
///   contract): algebraic (eliminated) variables.
///
/// See also:
///  - docs/ARCHITECTURE.md: problem/solver separation, parameter handling.
///  - acados correspondence: `has_dynamics_hess` / `has_constr_hess`
///    mirror acados `hessian_approx` EXACT / GAUSS_NEWTON and the runtime
///    gates `exact_hess_dyn` / `exact_hess_constr`; `SoftPenalty` mirrors
///    the per-slack `Z` (Hessian) and `z` (gradient) weights.

template <class Dims, class Scalar = double>
class Problem
{
public:
    /// The Dims tag this problem was instantiated with; re-exposed so
    /// wrapper classes (e.g. ScaledProblem) can name the base without
    /// the user repeating the tag.
    using dims_t = Dims;

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
    // dynamics Hessian). false: the Hessian is not provided; the solver
    // assembles the Gauss-Newton QP Hessian, where the dynamics enter only
    // through the linearized BA matrix (the two models coincide for a
    // linear dynamics map, where the dynamics Hessian term is zero anyway).
    // The term may additionally be dropped at runtime via
    // AcadosOptions::compute_hess_dyn.
    static constexpr bool has_dynamics_hess = Dims::has_dynamics_hess;

    // true: the constraint *_value_jac_hess methods are implemented; the
    // solver adds the multiplier-contracted constraint Hessians to the QP
    // Hessian. false: no constraint Hessians; the constraint Hessian terms
    // are omitted (identically zero when the constraint maps are linear).
    // The terms may additionally be dropped at runtime via
    // AcadosOptions::compute_hess_constr.
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

    /// Fixed initial state (required only when Dims::fixed_initial_state
    /// is true; absent from the interface when it is false).
    ///
    /// @return x_0, the fixed initial state (nx)
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
    /// Discrete dynamics x_{k+1} = f_k(x, u).
    ///
    /// @param  k stage index, 0..N-1
    /// @param  x state at stage k (nx)
    /// @param  u control at stage k (nu)
    /// @return x_next, state at stage k+1 (nx)
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
    /// Fused evaluation of the dynamics value and Jacobian.
    ///
    /// @param  k stage index, 0..N-1
    /// @param  x state at stage k (nx)
    /// @param  u control at stage k (nu)
    /// @param[out] x_next state at stage k+1 (nx)
    /// @param[out] df_dx  Jacobian df/dx (nx x nx)
    /// @param[out] df_du  Jacobian df/du (nx x nu)
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
    /// Fused evaluation of the dynamics value, Jacobian and
    /// multiplier-contracted Hessian (`hess * v = sum_i lam_i D^2 f_i v`).
    ///
    /// @param  k stage index, 0..N-1
    /// @param  x state at stage k (nx)
    /// @param  u control at stage k (nu)
    /// @param  lam dynamics multiplier at stage k (nx)
    /// @param[out] x_next state at stage k+1 (nx)
    /// @param[out] df_dx  Jacobian df/dx (nx x nx)
    /// @param[out] df_du  Jacobian df/du (nx x nu)
    /// @param[out] hess   contracted Hessian (nx+nu x nx+nu), [x; u] layout
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
    /// Stage cost L_k(x, u).
    ///
    /// @param  k stage index, 0..N-1
    /// @param  x state at stage k (nx)
    /// @param  u control at stage k (nu)
    /// @return L_k(x, u)
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
    /// Fused evaluation of the stage cost value and gradient.
    ///
    /// @param  k stage index, 0..N-1
    /// @param  x state at stage k (nx)
    /// @param  u control at stage k (nu)
    /// @param[out] value L_k(x, u)
    /// @param[out] grad  gradient, layout [g_x; g_u] (nx + nu)
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
    /// Fused evaluation of the stage cost value, gradient and Hessian.
    ///
    /// @param  k stage index, 0..N-1
    /// @param  x state at stage k (nx)
    /// @param  u control at stage k (nu)
    /// @param[out] value L_k(x, u)
    /// @param[out] grad  gradient, layout [g_x; g_u] (nx + nu)
    /// @param[out] hess  Hessian, block layout [H_xx H_xu; H_ux H_uu]
    ///                   (nx+nu x nx+nu); exact or Gauss-Newton (user's choice)
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
    /// Terminal cost L_N(x).
    ///
    /// @param  x state at the terminal stage (nx)
    /// @return L_N(x)
    Scalar terminal_cost_value(const state_t& /*x*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'Scalar terminal_cost_value(const state_t&) const' in the concrete problem.");
        // unreachable; the static_assert above fires first
        return {};
    }
    /// Fused evaluation of the terminal cost value and gradient.
    ///
    /// @param  x state at the terminal stage (nx)
    /// @param[out] value L_N(x)
    /// @param[out] grad  gradient dL_N/dx (nx)
    void terminal_cost_value_grad(const state_t& /*x*/,
                                  Scalar& /*value*/,
                                  term_grad_t& /*grad*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'void terminal_cost_value_grad(const state_t&, Scalar& value, "
            "term_grad_t& grad) const' in the concrete problem.");
    }
    /// Fused evaluation of the terminal cost value, gradient and Hessian.
    ///
    /// @param  x state at the terminal stage (nx)
    /// @param[out] value L_N(x)
    /// @param[out] grad  gradient dL_N/dx (nx)
    /// @param[out] hess  Hessian D^2 L_N (nx x nx); exact or Gauss-Newton
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
    /// Stage inequality constraint g_k(x, u) <= 0.
    ///
    /// @param  k stage index, 0..N-1
    /// @param  x state at stage k (nx)
    /// @param  u control at stage k (nu)
    /// @return g, inequality values (ng)
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
    /// Fused evaluation of the inequality value and Jacobian.
    ///
    /// @param  k stage index, 0..N-1
    /// @param  x state at stage k (nx)
    /// @param  u control at stage k (nu)
    /// @param[out] g    inequality values (ng)
    /// @param[out] g_dx Jacobian dg/dx (ng x nx)
    /// @param[out] g_du Jacobian dg/du (ng x nu)
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
    /// Fused evaluation of the inequality value, Jacobian and
    /// multiplier-contracted Hessian (`hess * v = sum_i lam_i D^2 g_i v`).
    ///
    /// @param  k stage index, 0..N-1
    /// @param  x state at stage k (nx)
    /// @param  u control at stage k (nu)
    /// @param  lam inequality multipliers at stage k (ng)
    /// @param[out] g    inequality values (ng)
    /// @param[out] g_dx Jacobian dg/dx (ng x nx)
    /// @param[out] g_du Jacobian dg/du (ng x nu)
    /// @param[out] hess contracted Hessian (nx+nu x nx+nu), [x; u] layout
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
    /// Stage equality constraint e_k(x, u) == 0.
    ///
    /// @param  k stage index, 0..N-1
    /// @param  x state at stage k (nx)
    /// @param  u control at stage k (nu)
    /// @return e, equality values (ne)
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
    /// Fused evaluation of the equality value and Jacobian.
    ///
    /// @param  k stage index, 0..N-1
    /// @param  x state at stage k (nx)
    /// @param  u control at stage k (nu)
    /// @param[out] e    equality values (ne)
    /// @param[out] e_dx Jacobian de/dx (ne x nx)
    /// @param[out] e_du Jacobian de/du (ne x nu)
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
    /// Fused evaluation of the equality value, Jacobian and
    /// multiplier-contracted Hessian (`hess * v = sum_i lam_i D^2 e_i v`).
    ///
    /// @param  k stage index, 0..N-1
    /// @param  x state at stage k (nx)
    /// @param  u control at stage k (nu)
    /// @param  lam equality multipliers at stage k (ne)
    /// @param[out] e    equality values (ne)
    /// @param[out] e_dx Jacobian de/dx (ne x nx)
    /// @param[out] e_du Jacobian de/du (ne x nu)
    /// @param[out] hess contracted Hessian (nx+nu x nx+nu), [x; u] layout
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
    /// Linear two-sided constraint lo <= A_k x + B_k u <= hi at stage k.
    ///
    /// @param  k stage index, 0..N-1
    /// @return spec: A (nl x nx), B (nl x nu), two-sided bounds and
    ///         per-row soft-penalty weights
    stage_linear_t stage_linear_constr(int /*k*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'stage_linear_t stage_linear_constr(int k) const' in the concrete problem "
            "(A, B with lo <= A x + B u <= hi), or set nl to 0.");
        // unreachable; the static_assert above fires first
        return {};
    }
    /// State box constraint at stage k: lo <= x[state_box_idx] <= hi.
    ///
    /// @param  k stage index, 0..N-1
    /// @return spec: lo/hi (one entry per active row, state_box_idx order)
    ///         and per-row soft-penalty weights
    state_box_t stage_state_box_constr(int /*k*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'state_box_t stage_state_box_constr(int k) const' in the concrete problem.");
        // unreachable; the static_assert above fires first
        return {};
    }
    /// Control box constraint at stage k: lo <= u[control_box_idx] <= hi.
    ///
    /// @param  k stage index, 0..N-1
    /// @return spec: lo/hi (one entry per active row, control_box_idx
    ///         order) and per-row soft-penalty weights
    control_box_t stage_control_box_constr(int /*k*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'control_box_t stage_control_box_constr(int k) const' in the concrete problem.");
        // unreachable; the static_assert above fires first
        return {};
    }
    /// Per-row slack-penalty weights of the soft inequality rows.
    ///
    /// @param  k stage index, 0..N-1
    /// @return one {quad, lin} pair per soft row, in ineq_soft_idx order
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
    /// Per-row slack-penalty weights of the soft equality rows.
    ///
    /// @param  k stage index, 0..N-1
    /// @return one {quad, lin} pair per soft row, in eq_soft_idx order
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
    /// Terminal state box constraint: lo <= x[terminal_state_box_idx] <= hi.
    ///
    /// @return spec: lo/hi (one entry per active row,
    ///         terminal_state_box_idx order) and per-row soft-penalty
    ///         weights
    term_state_box_t terminal_state_box_constr() const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'term_state_box_t terminal_state_box_constr() const' in the concrete problem.");
        // unreachable; the static_assert above fires first
        return {};
    }
    /// Terminal inequality constraint g(x) <= 0.
    ///
    /// @param  x state at the terminal stage (nx)
    /// @return g, inequality values (ng_t)
    ineq_term_t terminal_inequality_value(const state_t& /*x*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'ineq_term_t terminal_inequality_value(const state_t&) const' "
            "(g(x) <= 0) in the concrete problem, or set ng_t to 0.");
        // unreachable; the static_assert above fires first
        return {};
    }
    /// Fused evaluation of the terminal inequality value and Jacobian.
    ///
    /// @param  x state at the terminal stage (nx)
    /// @param[out] g    inequality values (ng_t)
    /// @param[out] g_dx Jacobian dg/dx (ng_t x nx)
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
    /// Fused evaluation of the terminal inequality value, Jacobian and
    /// multiplier-contracted Hessian (`hess * v = sum_i lam_i D^2 g_i v`).
    ///
    /// @param  x state at the terminal stage (nx)
    /// @param  lam terminal inequality multipliers (ng_t)
    /// @param[out] g    inequality values (ng_t)
    /// @param[out] g_dx Jacobian dg/dx (ng_t x nx)
    /// @param[out] hess contracted Hessian (nx x nx)
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
    /// Terminal equality constraint e(x) == 0.
    ///
    /// @param  x state at the terminal stage (nx)
    /// @return e, equality values (ne_t)
    eq_term_t terminal_equality_value(const state_t& /*x*/) const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'eq_term_t terminal_equality_value(const state_t&) const' "
            "(e(x) == 0) in the concrete problem, or set ne_t to 0.");
        // unreachable; the static_assert above fires first
        return {};
    }
    /// Fused evaluation of the terminal equality value and Jacobian.
    ///
    /// @param  x state at the terminal stage (nx)
    /// @param[out] e    equality values (ne_t)
    /// @param[out] e_dx Jacobian de/dx (ne_t x nx)
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
    /// Fused evaluation of the terminal equality value, Jacobian and
    /// multiplier-contracted Hessian (`hess * v = sum_i lam_i D^2 e_i v`).
    ///
    /// @param  x state at the terminal stage (nx)
    /// @param  lam terminal equality multipliers (ne_t)
    /// @param[out] e    equality values (ne_t)
    /// @param[out] e_dx Jacobian de/dx (ne_t x nx)
    /// @param[out] hess contracted Hessian (nx x nx)
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
    /// Linear two-sided terminal constraint lo <= A x <= hi.
    ///
    /// @return spec: A (nl_t x nx), two-sided bounds and per-row
    ///         soft-penalty weights
    term_linear_t terminal_linear_constr() const
    {
        static_assert(false,
            "ocp::Problem interface: implement "
            "'term_linear_t terminal_linear_constr() const' in the concrete problem "
            "(A with lo <= A x <= hi), or set nl_t to 0.");
        // unreachable; the static_assert above fires first
        return {};
    }
    /// Per-row slack-penalty weights of the soft terminal inequality rows.
    ///
    /// @return one {quad, lin} pair per soft row, in
    ///         terminal_ineq_soft_idx order
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
    /// Per-row slack-penalty weights of the soft terminal equality rows.
    ///
    /// @return one {quad, lin} pair per soft row, in terminal_eq_soft_idx
    ///         order
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
    kTimeout,        // wall-clock timeout reached
    kNotInitialized  // solve called before init()
};

/// Human-readable name of a solve status (the enumerator spelling,
/// e.g. "kSolved"), for logging and diagnostics. A value outside the
/// enum range maps to "unknown".
constexpr const char* to_string(Status s)
{
    switch (s)
    {
        case Status::kUnset:          return "kUnset";
        case Status::kSolved:         return "kSolved";
        case Status::kMaxIterations:  return "kMaxIterations";
        case Status::kInfeasible:     return "kInfeasible";
        case Status::kQpFailure:      return "kQpFailure";
        case Status::kMinStep:        return "kMinStep";
        case Status::kUnbounded:      return "kUnbounded";
        case Status::kNanDetected:    return "kNanDetected";
        case Status::kAborted:        return "kAborted";
        case Status::kTimeout:        return "kTimeout";
        case Status::kNotInitialized: return "kNotInitialized";
    }
    return "unknown";
}

/// How to fill the new last stage after a one-stage receding-horizon shift
/// (see `Solution::shift`).
enum class ShiftFill
{
    kRollover,  // x[N], u[N-1] keep their previous values (warm-start guess)
    kZero       // new last-stage x[N], u[N-1] are zeroed
};

/// Solver-independent solution (and warm-start) container.
///
/// The same object is used both as the initial iterate supplied to a
/// solver (primal variables and, optionally, multipliers) and as the
/// result the solver fills in. Solver-internal quantities (slack
/// variables, barrier/regularization parameters, multiplier orderings,
/// ...) are *not* stored here; each solver keeps its own
/// (docs/ARCHITECTURE.md sec. 2 "Separation of internal variables").
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

    /// Advance this solution one stage for the next MPC step (receding
    /// horizon shift), in place, usable as the warm start of the next
    /// `solve()`.
    ///
    /// Every stage quantity moves one index up (stage k inherits the old
    /// stage k + 1), the new initial state `x0_new` is written to `x[0]`,
    /// and the new last stage is filled per `fill`:
    ///
    ///   quantity                           rule
    ///   ---------------------------------  ---------------------------------
    ///   x[k] (k = 0..N-1)                  x[k] = x[k+1]
    ///   x[0]                               = x0_new (after the shift)
    ///   x[N]                               keep (kRollover) / 0 (kZero)
    ///   u[k] (k = 0..N-2)                  u[k] = u[k+1]
    ///   u[N-1]                             keep (kRollover) / 0 (kZero)
    ///   lambda_dyn[k] (k = 0..N-2)         = lambda_dyn[k+1]
    ///   stage ineq/eq/lin multipliers      = [k+1] for k = 0..N-2,
    ///                                        zeroed at k = N-1
    ///   lambda_box_state[k] (k = 0..N-1)   = lambda_box_state[k+1]
    ///   lambda_box_control[k] (k = 0..N-2) = lambda_box_control[k+1],
    ///                                        zeroed at k = N-1
    ///   terminal multipliers               unchanged (the terminal stage
    ///                                        stays terminal)
    ///   cost_value, status                 reset (0, kUnset)
    ///
    /// `x0_new` is the measured state of the new horizon; when
    /// `Dims::fixed_initial_state` is true the solver still pins x_0 from
    /// `Problem::initial_state()`, which the caller must update separately.
    ///
    /// @param x0_new  new initial state (written to x[0]).
    /// @param fill    how to fill the new last stage (default kRollover).
    void shift(const state_t& x0_new, ShiftFill fill = ShiftFill::kRollover)
    {
        assert(N >= 1);

        // states up one index (x[N] is left alone for the fill below)
        for (int k = 0; k < N; ++k)
        {
            x[k] = x[k + 1];
        }
        x[0] = x0_new;
        if (fill == ShiftFill::kZero)
        {
            x[N].setZero();
        }

        // controls up one index (u[N-1] is the new last stage)
        for (int k = 0; k + 1 < N; ++k)
        {
            u[k] = u[k + 1];
        }
        if (fill == ShiftFill::kZero)
        {
            u[N - 1].setZero();
        }

        // stage multipliers: up one index, new last stage zeroed
        for (int k = 0; k + 1 < N; ++k)
        {
            lambda_dyn[k] = lambda_dyn[k + 1];
            lambda_ineq_stage[k] = lambda_ineq_stage[k + 1];
            lambda_eq_stage[k] = lambda_eq_stage[k + 1];
            lambda_lin_stage[k] = lambda_lin_stage[k + 1];
            lambda_box_control[k] = lambda_box_control[k + 1];
        }
        lambda_dyn[N - 1].setZero();
        lambda_ineq_stage[N - 1].setZero();
        lambda_eq_stage[N - 1].setZero();
        lambda_lin_stage[N - 1].setZero();
        lambda_box_control[N - 1].setZero();

        // state box multipliers (N + 1 stages): up one index; the terminal
        // entry lambda_box_state[N] stays (terminal multipliers unchanged).
        for (int k = 0; k < N; ++k)
        {
            lambda_box_state[k] = lambda_box_state[k + 1];
        }

        cost_value = 0;
        status = Status::kUnset;
    }
};

}  // namespace ocp

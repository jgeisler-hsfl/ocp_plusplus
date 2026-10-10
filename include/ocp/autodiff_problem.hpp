// ocp/autodiff_problem.hpp
//
// CRTP mix-in providing the Problem<Dims> cost / nonlinear-constraint
// derivative entries from the scalar-templated value functions alone, via
// forward-mode AD on the vendored multidual scalar (thirdparty/cppduals,
// Eigen-3 glue in integrators/duals_eigen3.hpp).
//
// Problem-level counterpart of AutoDiffDae (dae_autodiff.hpp): the dynamics
// derivatives stay on the integrator level (the ODE f / DAE F routes); this
// mix-in covers the six cost / constraint value functions
//
//   stage cost,         stage inequality,   stage equality,
//   terminal cost,      terminal inequality, terminal equality,
//
// each of which the user writes once, scalar-templated (plain arithmetic /
// Eigen cwise operations on D, mirroring the ODE f / DAE F convention).
//
// The value functions accept any Eigen expression (the solver passes
// expressions, e.g. trial states x + step); write them
// expression-templated:
//
//   template <class XExpr, class UExpr>
//   auto stage_cost_value(int k, const XExpr& x, const UExpr& u) const
//       -> Eigen::Matrix<typename XExpr::Scalar, ng, 1>
//   {
//       using D = typename XExpr::Scalar;
//       ...
//   }
//
// (Non-template, double-only functions may use concrete Matrix<double, ...>
// parameters; Eigen's converting constructor then handles expressions.)
//
// The same methods serve the value pass (the solver deduces D = scalar_t)
// and the AD derivative passes below.
//
// The mix-in provides the concrete derivative entries (exactly the
// Problem<Dims> signatures); allocation-free, fixed-size stack buffers:
//
//   costs:
//     *_value_grad      -- one dual<scalar_t, nIn> forward pass; the value
//                          is carried for free;
//     *_value_grad_hess -- one dual<dual<scalar_t, nIn>, nIn> forward pass
//                          (the outer dimension carries the full Hessian
//                          basis): value + full gradient + full Hessian in
//                          a single evaluation, cheaper than nIn separate
//                          nested-dual HVP passes;
//   constraints:
//     *_value_jac       -- one dual<scalar_t, nIn> pass (value + both
//                          Jacobian blocks);
//     *_value_jac_hess  -- one triple-dual pass; the multiplier lam is
//                          contracted only at extraction:
//                          hess v = sum_i lam_i D^2(·)_i v, [x; u] layout.
//
// nIn = nx + nu for the stage functions (z = [x; u]), nIn = nx for the
// terminal ones (z = [x]).
//
// The mix-in deliberately re-declares none of the Problem<Dims> type
// aliases (it refers to them as P::... below, P = Problem<Dims>), so a
// concrete class can inherit both Problem<Dims> (or a ContinuousProblem
// adapter) and AutoDiffProblem without any name-lookup ambiguity.
//
// Per-function user-vs-AD choice: an entry hand-written in the concrete
// class hides both the mix-in's AD version and the Problem<Dims> stub (the
// solver dispatches on the concrete type). Entries left to AD must be
// pulled into scope with a using-declaration; without it the Problem stub
// and the mix-in entry of the same name are both visible and the call is
// ambiguous:
//
//   struct MyProb : ContinuousProblem<Dims, Ode, Integ>,
//                   ocp::AutoDiffProblem<MyProb, Dims>
//   {
//       // ... value functions (expression-templated as above) ...
//
//       using ocp::AutoDiffProblem<MyProb, Dims>::stage_cost_value_grad;
//       using ocp::AutoDiffProblem<MyProb, Dims>::stage_cost_value_grad_hess;
//       // ... one using-declaration per AD-provided derivative entry; a
//       // hand-written entry replaces the corresponding using-declaration.
//   };
//
// The WithConstrHess switch mirrors AutoDiffDae::WithHess:
//   WithConstrHess == true (default): the four constraint
//     *_value_jac_hess entries are provided; pair with
//     Dims::has_constr_hess = true for the exact-Newton constraint Hessian
//     model.
//   WithConstrHess == false: the constraint *_value_jac_hess entries are
//     not provided; pair with Dims::has_constr_hess = false (Gauss-Newton
//     constraint Hessian model) or hand-write the entries.
//   The cost entries and the constraint Jacobian entries are unaffected.
//
// A value function that is not dual-evaluable (e.g. hand-written for
// double only) cannot drive an AD derivative: the corresponding mix-in
// entry static_asserts at the call site. For such a function, hand-write
// the value and all its derivatives and drop the using-declarations.

#pragma once

#include <type_traits>

#include "ocp/integrators/duals_eigen3.hpp"
#include "ocp/problem.hpp"

namespace ocp
{

namespace detail
{

// SFINAE result-type aliases for the six value functions at dual scalar D.
// Each alias fails (SFINAE-friendly) when the call is ill-formed, which
// covers both a missing entry and a double-only (non-templated) one:
// Eigen's converting Matrix constructor is not a valid implicit path from
// Matrix<D> arguments to a Matrix<double> parameter.

template <class Derived, class Dims, class D>
using stage_cost_value_result_t =
    std::decay_t<decltype(std::declval<const Derived&>().stage_cost_value(
        std::declval<int>(),
        std::declval<const Eigen::Matrix<D, Problem<Dims>::nx, 1>&>(),
        std::declval<const Eigen::Matrix<D, Problem<Dims>::nu, 1>&>()))>;

template <class Derived, class Dims, class D>
using stage_inequality_value_result_t =
    std::decay_t<decltype(std::declval<const Derived&>().stage_inequality_value(
        std::declval<int>(),
        std::declval<const Eigen::Matrix<D, Problem<Dims>::nx, 1>&>(),
        std::declval<const Eigen::Matrix<D, Problem<Dims>::nu, 1>&>()))>;

template <class Derived, class Dims, class D>
using stage_equality_value_result_t =
    std::decay_t<decltype(std::declval<const Derived&>().stage_equality_value(
        std::declval<int>(),
        std::declval<const Eigen::Matrix<D, Problem<Dims>::nx, 1>&>(),
        std::declval<const Eigen::Matrix<D, Problem<Dims>::nu, 1>&>()))>;

template <class Derived, class Dims, class D>
using terminal_cost_value_result_t =
    std::decay_t<decltype(std::declval<const Derived&>().terminal_cost_value(
        std::declval<const Eigen::Matrix<D, Problem<Dims>::nx, 1>&>()))>;

template <class Derived, class Dims, class D>
using terminal_inequality_value_result_t =
    std::decay_t<decltype(
        std::declval<const Derived&>().terminal_inequality_value(
            std::declval<const Eigen::Matrix<D, Problem<Dims>::nx, 1>&>()))>;

template <class Derived, class Dims, class D>
using terminal_equality_value_result_t =
    std::decay_t<decltype(
        std::declval<const Derived&>().terminal_equality_value(
            std::declval<const Eigen::Matrix<D, Problem<Dims>::nx, 1>&>()))>;

// SFINAE probes: true when the value function is dual-evaluable at D AND
// returns the D-typed value (the return-type check guards against Eigen's
// unconstrained converting constructor). Each probe is only instantiated
// when queried.

template <class Derived, class Dims, class D, class = std::void_t<>>
struct stage_cost_value_evaluable_at : std::false_type
{
};

template <class Derived, class Dims, class D>
struct stage_cost_value_evaluable_at<
    Derived, Dims, D, std::void_t<stage_cost_value_result_t<Derived, Dims, D>>>
    : std::is_same<stage_cost_value_result_t<Derived, Dims, D>, D>
{
};

template <class Derived, class Dims, class D, class = std::void_t<>>
struct stage_inequality_value_evaluable_at : std::false_type
{
};

template <class Derived, class Dims, class D>
struct stage_inequality_value_evaluable_at<
    Derived, Dims, D,
    std::void_t<stage_inequality_value_result_t<Derived, Dims, D>>>
    : std::is_same<stage_inequality_value_result_t<Derived, Dims, D>,
                   Eigen::Matrix<D, Problem<Dims>::ng, 1>>
{
};

template <class Derived, class Dims, class D, class = std::void_t<>>
struct stage_equality_value_evaluable_at : std::false_type
{
};

template <class Derived, class Dims, class D>
struct stage_equality_value_evaluable_at<
    Derived, Dims, D,
    std::void_t<stage_equality_value_result_t<Derived, Dims, D>>>
    : std::is_same<stage_equality_value_result_t<Derived, Dims, D>,
                   Eigen::Matrix<D, Problem<Dims>::ne, 1>>
{
};

template <class Derived, class Dims, class D, class = std::void_t<>>
struct terminal_cost_value_evaluable_at : std::false_type
{
};

template <class Derived, class Dims, class D>
struct terminal_cost_value_evaluable_at<
    Derived, Dims, D,
    std::void_t<terminal_cost_value_result_t<Derived, Dims, D>>>
    : std::is_same<terminal_cost_value_result_t<Derived, Dims, D>, D>
{
};

template <class Derived, class Dims, class D, class = std::void_t<>>
struct terminal_inequality_value_evaluable_at : std::false_type
{
};

template <class Derived, class Dims, class D>
struct terminal_inequality_value_evaluable_at<
    Derived, Dims, D,
    std::void_t<terminal_inequality_value_result_t<Derived, Dims, D>>>
    : std::is_same<terminal_inequality_value_result_t<Derived, Dims, D>,
                   Eigen::Matrix<D, Problem<Dims>::ng_t, 1>>
{
};

template <class Derived, class Dims, class D, class = std::void_t<>>
struct terminal_equality_value_evaluable_at : std::false_type
{
};

template <class Derived, class Dims, class D>
struct terminal_equality_value_evaluable_at<
    Derived, Dims, D,
    std::void_t<terminal_equality_value_result_t<Derived, Dims, D>>>
    : std::is_same<terminal_equality_value_result_t<Derived, Dims, D>,
                   Eigen::Matrix<D, Problem<Dims>::ne_t, 1>>
{
};

}  // namespace detail

/// True when `Derived::stage_cost_value(k, x, u)` is dual-evaluable at D
/// (see the file-level contract in autodiff_problem.hpp).
template <class Derived, class Dims, class D>
constexpr bool stage_cost_value_evaluable_at =
    detail::stage_cost_value_evaluable_at<Derived, Dims, D>::value;

/// True when `Derived::stage_inequality_value(k, x, u)` is dual-evaluable
/// at D.
template <class Derived, class Dims, class D>
constexpr bool stage_inequality_value_evaluable_at =
    detail::stage_inequality_value_evaluable_at<Derived, Dims, D>::value;

/// True when `Derived::stage_equality_value(k, x, u)` is dual-evaluable at
/// D.
template <class Derived, class Dims, class D>
constexpr bool stage_equality_value_evaluable_at =
    detail::stage_equality_value_evaluable_at<Derived, Dims, D>::value;

/// True when `Derived::terminal_cost_value(x)` is dual-evaluable at D.
template <class Derived, class Dims, class D>
constexpr bool terminal_cost_value_evaluable_at =
    detail::terminal_cost_value_evaluable_at<Derived, Dims, D>::value;

/// True when `Derived::terminal_inequality_value(x)` is dual-evaluable at
/// D.
template <class Derived, class Dims, class D>
constexpr bool terminal_inequality_value_evaluable_at =
    detail::terminal_inequality_value_evaluable_at<Derived, Dims, D>::value;

/// True when `Derived::terminal_equality_value(x)` is dual-evaluable at D.
template <class Derived, class Dims, class D>
constexpr bool terminal_equality_value_evaluable_at =
    detail::terminal_equality_value_evaluable_at<Derived, Dims, D>::value;

// =========================================================================
//  AutoDiffProblem
// =========================================================================

/// CRTP mix-in providing the Problem<Dims> cost / nonlinear-constraint
/// derivative entries from the scalar-templated value functions (see the
/// file-level contract above). Inherit it alongside the dynamics adapter
/// (e.g. ContinuousProblem); it does not inherit Problem<Dims> itself (no
/// diamond) and re-declares none of Problem's type aliases.
///
/// `WithConstrHess` mirrors AutoDiffDae::WithHess: the primary template is
/// the Hessian-suppressed variant (no constraint `*_value_jac_hess`); the
/// `true` specialization below adds them.
template <class Derived, class Dims, bool WithConstrHess = true>
class AutoDiffProblem
{
public:
    using P = Problem<Dims>;

    // stage input count (z = [x; u]) and terminal input count (z = [x])
    static constexpr int nInS = P::nx + P::nu;
    static constexpr int nInT = P::nx;

    // Dual1: first-order basis over nIn inputs (value + full first
    // derivatives in one pass). Dual3: first-order inner basis PLUS the
    // full Hessian basis in the outer dimension (value + full first +
    // full second derivatives in one pass).
    using stage_dual1_t = duals::dual<typename P::scalar_t, nInS>;
    using stage_dual3_t = duals::dual<stage_dual1_t, nInS>;
    using term_dual1_t  = duals::dual<typename P::scalar_t, nInT>;
    using term_dual3_t  = duals::dual<term_dual1_t, nInT>;

    /// True when this specialization provides the constraint
    /// `*_value_jac_hess` entries (false on the primary template).
    static constexpr bool provides_constr_hess = WithConstrHess;

    /// The concrete problem (CRTP accessor).
    const Derived& self() const
    {
        return static_cast<const Derived&>(*this);
    }

    // ---------------------------------------------------------------
    //  stage cost
    // ---------------------------------------------------------------

    /// AD gradient of the stage cost: one `dual<scalar_t, nInS>` pass over
    /// z = [x; u]; the value is carried for free.
    void stage_cost_value_grad(int k, const typename P::state_t& x,
                               const typename P::control_t& u,
                               typename P::scalar_t& value,
                               typename P::stage_grad_t& grad) const
    {
        static_assert(
            stage_cost_value_evaluable_at<Derived, Dims, stage_dual1_t>,
            "AutoDiffProblem::stage_cost_value_grad: "
            "Derived::stage_cost_value must be scalar-templated "
            "(dual-evaluable) for the AD grad; hand-write "
            "stage_cost_value_grad instead (and drop the using-declaration).");
        Eigen::Matrix<stage_dual1_t, nInS, 1> zbar;
        seedS1(x, u, zbar);
        const Eigen::Matrix<stage_dual1_t, P::nx, 1> xd = zbar.head(P::nx);
        const Eigen::Matrix<stage_dual1_t, P::nu, 1> ud = zbar.tail(P::nu);
        const stage_dual1_t c = self().stage_cost_value(k, xd, ud);
        value = c.rpart();
        for (int j = 0; j < nInS; ++j)
        {
            grad(j) = c.dpart(j);
        }
    }

    /// AD value + gradient + full Hessian of the stage cost in one
    /// `dual<dual<scalar_t, nInS>, nInS>` pass: input k carries the inner
    /// basis e_k (gradient) and the outer basis e_k (Hessian row), so
    /// grad(j) = c.rpart().dpart(j) and hess(a, b) = c.dpart(a).dpart(b).
    void stage_cost_value_grad_hess(int k, const typename P::state_t& x,
                                    const typename P::control_t& u,
                                    typename P::scalar_t& value,
                                    typename P::stage_grad_t& grad,
                                    typename P::stage_hess_t& hess) const
    {
        static_assert(
            stage_cost_value_evaluable_at<Derived, Dims, stage_dual3_t>,
            "AutoDiffProblem::stage_cost_value_grad_hess: "
            "Derived::stage_cost_value must be scalar-templated "
            "(dual-evaluable) for the AD Hessian; hand-write "
            "stage_cost_value_grad_hess instead (and drop the "
            "using-declaration).");
        Eigen::Matrix<stage_dual3_t, nInS, 1> zbar;
        seedS3(x, u, zbar);
        const Eigen::Matrix<stage_dual3_t, P::nx, 1> xd = zbar.head(P::nx);
        const Eigen::Matrix<stage_dual3_t, P::nu, 1> ud = zbar.tail(P::nu);
        const stage_dual3_t c = self().stage_cost_value(k, xd, ud);
        value = c.rpart().rpart();
        for (int j = 0; j < nInS; ++j)
        {
            grad(j) = c.rpart().dpart(j);
        }
        for (int a = 0; a < nInS; ++a)
        {
            for (int b = 0; b < nInS; ++b)
            {
                hess(a, b) = c.dpart(a).dpart(b);
            }
        }
    }

    // ---------------------------------------------------------------
    //  stage constraints (Jacobian entries; Hessian entries in the
    //  WithConstrHess = true specialization)
    // ---------------------------------------------------------------

    /// AD value + Jacobian of the stage inequality: one Dual1 pass.
    void stage_inequality_value_jac(int k, const typename P::state_t& x,
                                    const typename P::control_t& u,
                                    typename P::ineq_t& g,
                                    typename P::ineq_dg_dx_t& g_dx,
                                    typename P::ineq_dg_du_t& g_du) const
    {
        static_assert(
            stage_inequality_value_evaluable_at<Derived, Dims, stage_dual1_t>,
            "AutoDiffProblem::stage_inequality_value_jac: "
            "Derived::stage_inequality_value must be scalar-templated "
            "(dual-evaluable) for the AD Jacobian; hand-write "
            "stage_inequality_value_jac instead (and drop the "
            "using-declaration).");
        Eigen::Matrix<stage_dual1_t, nInS, 1> zbar;
        seedS1(x, u, zbar);
        const Eigen::Matrix<stage_dual1_t, P::nx, 1> xd = zbar.head(P::nx);
        const Eigen::Matrix<stage_dual1_t, P::nu, 1> ud = zbar.tail(P::nu);
        const Eigen::Matrix<stage_dual1_t, P::ng, 1> gv =
            self().stage_inequality_value(k, xd, ud);
        for (int i = 0; i < P::ng; ++i)
        {
            g(i) = gv(i).rpart();
            for (int j = 0; j < P::nx; ++j)
            {
                g_dx(i, j) = gv(i).dpart(j);
            }
            for (int j = 0; j < P::nu; ++j)
            {
                g_du(i, j) = gv(i).dpart(P::nx + j);
            }
        }
    }

    /// AD value + Jacobian of the stage equality: one Dual1 pass.
    void stage_equality_value_jac(int k, const typename P::state_t& x,
                                  const typename P::control_t& u,
                                  typename P::eq_t& e,
                                  typename P::eq_de_dx_t& e_dx,
                                  typename P::eq_de_du_t& e_du) const
    {
        static_assert(
            stage_equality_value_evaluable_at<Derived, Dims, stage_dual1_t>,
            "AutoDiffProblem::stage_equality_value_jac: "
            "Derived::stage_equality_value must be scalar-templated "
            "(dual-evaluable) for the AD Jacobian; hand-write "
            "stage_equality_value_jac instead (and drop the "
            "using-declaration).");
        Eigen::Matrix<stage_dual1_t, nInS, 1> zbar;
        seedS1(x, u, zbar);
        const Eigen::Matrix<stage_dual1_t, P::nx, 1> xd = zbar.head(P::nx);
        const Eigen::Matrix<stage_dual1_t, P::nu, 1> ud = zbar.tail(P::nu);
        const Eigen::Matrix<stage_dual1_t, P::ne, 1> ev =
            self().stage_equality_value(k, xd, ud);
        for (int i = 0; i < P::ne; ++i)
        {
            e(i) = ev(i).rpart();
            for (int j = 0; j < P::nx; ++j)
            {
                e_dx(i, j) = ev(i).dpart(j);
            }
            for (int j = 0; j < P::nu; ++j)
            {
                e_du(i, j) = ev(i).dpart(P::nx + j);
            }
        }
    }

    // ---------------------------------------------------------------
    //  terminal cost
    // ---------------------------------------------------------------

    /// AD gradient of the terminal cost: one `dual<scalar_t, nInT>` pass
    /// over z = [x]; the value is carried for free.
    void terminal_cost_value_grad(const typename P::state_t& x,
                                  typename P::scalar_t& value,
                                  typename P::term_grad_t& grad) const
    {
        static_assert(
            terminal_cost_value_evaluable_at<Derived, Dims, term_dual1_t>,
            "AutoDiffProblem::terminal_cost_value_grad: "
            "Derived::terminal_cost_value must be scalar-templated "
            "(dual-evaluable) for the AD grad; hand-write "
            "terminal_cost_value_grad instead (and drop the using-declaration).");
        Eigen::Matrix<term_dual1_t, nInT, 1> zbar;
        seedT1(x, zbar);
        const term_dual1_t c = self().terminal_cost_value(zbar);
        value = c.rpart();
        for (int j = 0; j < nInT; ++j)
        {
            grad(j) = c.dpart(j);
        }
    }

    /// AD value + gradient + full Hessian of the terminal cost in one
    /// `dual<dual<scalar_t, nInT>, nInT>` pass (see
    /// stage_cost_value_grad_hess for the seeding / extraction scheme).
    void terminal_cost_value_grad_hess(const typename P::state_t& x,
                                       typename P::scalar_t& value,
                                       typename P::term_grad_t& grad,
                                       typename P::term_hess_t& hess) const
    {
        static_assert(
            terminal_cost_value_evaluable_at<Derived, Dims, term_dual3_t>,
            "AutoDiffProblem::terminal_cost_value_grad_hess: "
            "Derived::terminal_cost_value must be scalar-templated "
            "(dual-evaluable) for the AD Hessian; hand-write "
            "terminal_cost_value_grad_hess instead (and drop the "
            "using-declaration).");
        Eigen::Matrix<term_dual3_t, nInT, 1> zbar;
        seedT3(x, zbar);
        const term_dual3_t c = self().terminal_cost_value(zbar);
        value = c.rpart().rpart();
        for (int j = 0; j < nInT; ++j)
        {
            grad(j) = c.rpart().dpart(j);
        }
        for (int a = 0; a < nInT; ++a)
        {
            for (int b = 0; b < nInT; ++b)
            {
                hess(a, b) = c.dpart(a).dpart(b);
            }
        }
    }

    // ---------------------------------------------------------------
    //  terminal constraints (Jacobian entries; Hessian entries in the
    //  WithConstrHess = true specialization)
    // ---------------------------------------------------------------

    /// AD value + Jacobian of the terminal inequality: one Dual1 pass.
    void terminal_inequality_value_jac(const typename P::state_t& x,
                                       typename P::ineq_term_t& g,
                                       typename P::ineq_term_dg_dx_t& g_dx) const
    {
        static_assert(
            terminal_inequality_value_evaluable_at<Derived, Dims, term_dual1_t>,
            "AutoDiffProblem::terminal_inequality_value_jac: "
            "Derived::terminal_inequality_value must be scalar-templated "
            "(dual-evaluable) for the AD Jacobian; hand-write "
            "terminal_inequality_value_jac instead (and drop the "
            "using-declaration).");
        Eigen::Matrix<term_dual1_t, nInT, 1> zbar;
        seedT1(x, zbar);
        const Eigen::Matrix<term_dual1_t, P::ng_t, 1> gv =
            self().terminal_inequality_value(zbar);
        for (int i = 0; i < P::ng_t; ++i)
        {
            g(i) = gv(i).rpart();
            for (int j = 0; j < P::nx; ++j)
            {
                g_dx(i, j) = gv(i).dpart(j);
            }
        }
    }

    /// AD value + Jacobian of the terminal equality: one Dual1 pass.
    void terminal_equality_value_jac(const typename P::state_t& x,
                                     typename P::eq_term_t& e,
                                     typename P::eq_term_de_dx_t& e_dx) const
    {
        static_assert(
            terminal_equality_value_evaluable_at<Derived, Dims, term_dual1_t>,
            "AutoDiffProblem::terminal_equality_value_jac: "
            "Derived::terminal_equality_value must be scalar-templated "
            "(dual-evaluable) for the AD Jacobian; hand-write "
            "terminal_equality_value_jac instead (and drop the "
            "using-declaration).");
        Eigen::Matrix<term_dual1_t, nInT, 1> zbar;
        seedT1(x, zbar);
        const Eigen::Matrix<term_dual1_t, P::ne_t, 1> ev =
            self().terminal_equality_value(zbar);
        for (int i = 0; i < P::ne_t; ++i)
        {
            e(i) = ev(i).rpart();
            for (int j = 0; j < P::nx; ++j)
            {
                e_dx(i, j) = ev(i).dpart(j);
            }
        }
    }

public:
    // Seed helpers (public so the WithConstrHess = true specialization can
    // reuse them through the base; implementation detail, not part of the
    // user-facing contract).

    // Seed z = [x; u] with the first-order basis (input k carries e_k).
    static void seedS1(const typename P::state_t& x,
                       const typename P::control_t& u,
                       Eigen::Matrix<stage_dual1_t, nInS, 1>& zbar)
    {
        for (int k = 0; k < P::nx; ++k)
        {
            zbar(k) = stage_dual1_t::variable(x(k), k);
        }
        for (int k = 0; k < P::nu; ++k)
        {
            zbar(P::nx + k) = stage_dual1_t::variable(u(k), P::nx + k);
        }
    }

    // Seed z = [x; u] with the first- AND second-order bases: the inner
    // e_k in the outer rpart (gradient) and the outer e_k in dpart k
    // (full Hessian basis); all other partials stay zero.
    static void seedS3(const typename P::state_t& x,
                       const typename P::control_t& u,
                       Eigen::Matrix<stage_dual3_t, nInS, 1>& zbar)
    {
        for (int k = 0; k < P::nx; ++k)
        {
            zbar(k).set_rpart(stage_dual1_t::variable(x(k), k));
            zbar(k).set_dpart(stage_dual1_t(1.0), k);
        }
        for (int k = 0; k < P::nu; ++k)
        {
            zbar(P::nx + k).set_rpart(
                stage_dual1_t::variable(u(k), P::nx + k));
            zbar(P::nx + k).set_dpart(stage_dual1_t(1.0), P::nx + k);
        }
    }

    // Seed z = [x] with the first-order basis.
    static void seedT1(const typename P::state_t& x,
                       Eigen::Matrix<term_dual1_t, nInT, 1>& zbar)
    {
        for (int k = 0; k < P::nx; ++k)
        {
            zbar(k) = term_dual1_t::variable(x(k), k);
        }
    }

    // Seed z = [x] with the first- AND second-order bases (see seedS3).
    static void seedT3(const typename P::state_t& x,
                       Eigen::Matrix<term_dual3_t, nInT, 1>& zbar)
    {
        for (int k = 0; k < P::nx; ++k)
        {
            zbar(k).set_rpart(term_dual1_t::variable(x(k), k));
            zbar(k).set_dpart(term_dual1_t(1.0), k);
        }
    }
};

/// `WithConstrHess == true` specialization: adds the fused bilinear
/// Hessian-vector entries of the four constraint groups from a single
/// triple-dual pass (the multiplier is contracted only at extraction), so
/// `provides_constr_hess` is true.
template <class Derived, class Dims>
class AutoDiffProblem<Derived, Dims, true>
    : public AutoDiffProblem<Derived, Dims, false>
{
public:
    using Base = AutoDiffProblem<Derived, Dims, false>;
    using P = typename Base::P;

    using stage_dual3_t = typename Base::stage_dual3_t;
    using term_dual3_t  = typename Base::term_dual3_t;

    static constexpr int nInS = Base::nInS;
    static constexpr int nInT = Base::nInT;

    static constexpr bool provides_constr_hess = true;

    /// AD value + Jacobian + multiplier-contracted Hessian of the stage
    /// inequality in one triple-dual pass:
    /// hess v = sum_i lam_i D^2 g_i v, [x; u] layout.
    void stage_inequality_value_jac_hess(
        int k, const typename P::state_t& x, const typename P::control_t& u,
        const typename P::ineq_t& lam, typename P::ineq_t& g,
        typename P::ineq_dg_dx_t& g_dx, typename P::ineq_dg_du_t& g_du,
        typename P::constr_hess_t& hess) const
    {
        static_assert(
            stage_inequality_value_evaluable_at<Derived, Dims, stage_dual3_t>,
            "AutoDiffProblem::stage_inequality_value_jac_hess: "
            "Derived::stage_inequality_value must be scalar-templated "
            "(dual-evaluable) for the AD Hessian; hand-write "
            "stage_inequality_value_jac_hess instead (and drop the "
            "using-declaration).");
        Eigen::Matrix<stage_dual3_t, nInS, 1> zbar;
        Base::seedS3(x, u, zbar);
        const Eigen::Matrix<stage_dual3_t, P::nx, 1> xd = zbar.head(P::nx);
        const Eigen::Matrix<stage_dual3_t, P::nu, 1> ud = zbar.tail(P::nu);
        const Eigen::Matrix<stage_dual3_t, P::ng, 1> gv =
            this->self().stage_inequality_value(k, xd, ud);
        hess.setZero();
        for (int i = 0; i < P::ng; ++i)
        {
            g(i) = gv(i).rpart().rpart();
            for (int j = 0; j < P::nx; ++j)
            {
                g_dx(i, j) = gv(i).rpart().dpart(j);
            }
            for (int j = 0; j < P::nu; ++j)
            {
                g_du(i, j) = gv(i).rpart().dpart(P::nx + j);
            }
            const typename P::scalar_t li = lam(i);
            if (li != 0.0)
            {
                for (int a = 0; a < nInS; ++a)
                {
                    for (int b = 0; b < nInS; ++b)
                    {
                        hess(a, b) += li * gv(i).dpart(a).dpart(b);
                    }
                }
            }
        }
    }

    /// AD value + Jacobian + multiplier-contracted Hessian of the stage
    /// equality in one triple-dual pass (see
    /// stage_inequality_value_jac_hess for the scheme).
    void stage_equality_value_jac_hess(
        int k, const typename P::state_t& x, const typename P::control_t& u,
        const typename P::eq_t& lam, typename P::eq_t& e,
        typename P::eq_de_dx_t& e_dx, typename P::eq_de_du_t& e_du,
        typename P::constr_hess_t& hess) const
    {
        static_assert(
            stage_equality_value_evaluable_at<Derived, Dims, stage_dual3_t>,
            "AutoDiffProblem::stage_equality_value_jac_hess: "
            "Derived::stage_equality_value must be scalar-templated "
            "(dual-evaluable) for the AD Hessian; hand-write "
            "stage_equality_value_jac_hess instead (and drop the "
            "using-declaration).");
        Eigen::Matrix<stage_dual3_t, nInS, 1> zbar;
        Base::seedS3(x, u, zbar);
        const Eigen::Matrix<stage_dual3_t, P::nx, 1> xd = zbar.head(P::nx);
        const Eigen::Matrix<stage_dual3_t, P::nu, 1> ud = zbar.tail(P::nu);
        const Eigen::Matrix<stage_dual3_t, P::ne, 1> ev =
            this->self().stage_equality_value(k, xd, ud);
        hess.setZero();
        for (int i = 0; i < P::ne; ++i)
        {
            e(i) = ev(i).rpart().rpart();
            for (int j = 0; j < P::nx; ++j)
            {
                e_dx(i, j) = ev(i).rpart().dpart(j);
            }
            for (int j = 0; j < P::nu; ++j)
            {
                e_du(i, j) = ev(i).rpart().dpart(P::nx + j);
            }
            const typename P::scalar_t li = lam(i);
            if (li != 0.0)
            {
                for (int a = 0; a < nInS; ++a)
                {
                    for (int b = 0; b < nInS; ++b)
                    {
                        hess(a, b) += li * ev(i).dpart(a).dpart(b);
                    }
                }
            }
        }
    }

    /// AD value + Jacobian + multiplier-contracted Hessian of the
    /// terminal inequality in one triple-dual pass (state only).
    void terminal_inequality_value_jac_hess(
        const typename P::state_t& x, const typename P::ineq_term_t& lam,
        typename P::ineq_term_t& g, typename P::ineq_term_dg_dx_t& g_dx,
        typename P::term_constr_hess_t& hess) const
    {
        static_assert(
            terminal_inequality_value_evaluable_at<Derived, Dims, term_dual3_t>,
            "AutoDiffProblem::terminal_inequality_value_jac_hess: "
            "Derived::terminal_inequality_value must be scalar-templated "
            "(dual-evaluable) for the AD Hessian; hand-write "
            "terminal_inequality_value_jac_hess instead (and drop the "
            "using-declaration).");
        Eigen::Matrix<term_dual3_t, nInT, 1> zbar;
        Base::seedT3(x, zbar);
        const Eigen::Matrix<term_dual3_t, P::ng_t, 1> gv =
            this->self().terminal_inequality_value(zbar);
        hess.setZero();
        for (int i = 0; i < P::ng_t; ++i)
        {
            g(i) = gv(i).rpart().rpart();
            for (int j = 0; j < P::nx; ++j)
            {
                g_dx(i, j) = gv(i).rpart().dpart(j);
            }
            const typename P::scalar_t li = lam(i);
            if (li != 0.0)
            {
                for (int a = 0; a < nInT; ++a)
                {
                    for (int b = 0; b < nInT; ++b)
                    {
                        hess(a, b) += li * gv(i).dpart(a).dpart(b);
                    }
                }
            }
        }
    }

    /// AD value + Jacobian + multiplier-contracted Hessian of the
    /// terminal equality in one triple-dual pass (state only).
    void terminal_equality_value_jac_hess(
        const typename P::state_t& x, const typename P::eq_term_t& lam,
        typename P::eq_term_t& e, typename P::eq_term_de_dx_t& e_dx,
        typename P::term_constr_hess_t& hess) const
    {
        static_assert(
            terminal_equality_value_evaluable_at<Derived, Dims, term_dual3_t>,
            "AutoDiffProblem::terminal_equality_value_jac_hess: "
            "Derived::terminal_equality_value must be scalar-templated "
            "(dual-evaluable) for the AD Hessian; hand-write "
            "terminal_equality_value_jac_hess instead (and drop the "
            "using-declaration).");
        Eigen::Matrix<term_dual3_t, nInT, 1> zbar;
        Base::seedT3(x, zbar);
        const Eigen::Matrix<term_dual3_t, P::ne_t, 1> ev =
            this->self().terminal_equality_value(zbar);
        hess.setZero();
        for (int i = 0; i < P::ne_t; ++i)
        {
            e(i) = ev(i).rpart().rpart();
            for (int j = 0; j < P::nx; ++j)
            {
                e_dx(i, j) = ev(i).rpart().dpart(j);
            }
            const typename P::scalar_t li = lam(i);
            if (li != 0.0)
            {
                for (int a = 0; a < nInT; ++a)
                {
                    for (int b = 0; b < nInT; ++b)
                    {
                        hess(a, b) += li * ev(i).dpart(a).dpart(b);
                    }
                }
            }
        }
    }
};

}  // namespace ocp

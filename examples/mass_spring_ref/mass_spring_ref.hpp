// Problem matching the acados mass_spring QP example exactly
// (acados/examples/c/no_interface_examples/mass_spring_model/mass_spring_qp.c),
// used as the 2i reference-diff target.
//
//   x = [q; v]  (q: 4 positions, v: 4 velocities),  u: 3 forces
//
//   continuous time (Ts = 0.5):
//        q_dot = v
//        v_dot = T q + Bc u,   T = tridiag(1, -2, 1),  Bc = [0; I3]
//
//   discrete time:  x_{k+1} = A x_k + B u_k + b,
//        A = expm(Ts Ac),  B = (A - I) Ac^{-1} Bc,  Ac = [0  I; T  0],
//        b = 0.1 * ones(8)
//
//   stage cost:     0.5 * x'Q x + 0.5 * u'R u + q' x + r' u
//                   Q = I_8, R = 2 I_3, q = 0.1 * ones(8), r = 0.2 * ones(3)
//   terminal cost:  0.5 * x'Q x + q' x
//
//   box:  -0.5 <= u_i <= 0.5,  -4 <= x_i <= 4  (stages 1..N),  x_0 fixed
//   terminal equality (Term variant):  x_N(0:3) = 0
//
// Two concrete problems share the MassSpringRefData system matrices / costs:
//   MassSpringRefBox  : ne_t = 0 (box-only reference)
//   MassSpringRefTerm : ne_t = 4 (terminal-equality reference)

#pragma once

#include "ocp/problem.hpp"

#include <unsupported/Eigen/MatrixFunctions>

// Shared (D
//   x = [q; v]  (q: 4 positions, v: 4 velocities),  u: 3 forces
//
//   continuous time (Ts = 0.5):
//        q_dot = v
//        v_dot = T q + Bc u,   T = tridiag(1, -2, 1),  Bc = [0; I3]
//
//   discrete time:  x_{k+1} = A x_k + B u_k + b,
//        A = expm(Ts Ac),  B = (A - I) Ac^{-1} Bc,  Ac = [0  I; T  0],
//        b = 0.1 * ones(8)
//
//   stage cost:     0.5 * x'Q x + 0.5 * u'R u + q' x + r' u
//                   Q = I_8, R = 2 I_3, q = 0.1 * ones(8), r = 0.2 * ones(3)
//   terminal cost:  0.5 * x'Q x + q' x
//
//   box:  -0.5 <= u_i <= 0.5,  -4 <= x_i <= 4  (stages 1..N),  x_0 fixed
//   terminal equality (Term variant):  x_N(0:3) = 0
//
// Two concrete problems share the shared MassSpringRefData (system matrices,
// costs); the only difference between them is ne_t (terminal equality).

#pragma once

#include "ocp/problem.hpp"

#include <unsupported/Eigen/MatrixFunctions>

struct MassSpringRefBoxDims
{
    static constexpr int nx = 8;
    static constexpr int nu = 3;
    static constexpr int ng = 0;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 0;
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = true;  // x_0 = [2.5, 2.5, 0, ...]'
    static constexpr bool has_dynamics_hess_prod = true;  // linear map (HVP = 0)
    static constexpr bool has_constr_hess_prod = true;  // linear (vacuous)
    static constexpr std::array<int, 8> state_box_idx = {0, 1, 2, 3, 4, 5, 6, 7};
    static constexpr std::array<int, 3> control_box_idx = {0, 1, 2};
    static constexpr std::array<int, 8> terminal_state_box_idx = {0, 1, 2, 3, 4, 5, 6, 7};
    static constexpr std::array<int, 0> ineq_soft_idx = {};
    static constexpr std::array<int, 0> eq_soft_idx = {};
    static constexpr std::array<int, 0> lin_soft_idx = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {};
    static constexpr std::array<int, 0> state_box_soft_idx = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

struct MassSpringRefTermDims
{
    static constexpr int nx = 8;
    static constexpr int nu = 3;
    static constexpr int ng = 0;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 4;  // x_N(0:3) = 0
    static constexpr int nl_t = 0;
    static constexpr bool fixed_initial_state = true;  // x_0 = [2.5, 2.5, 0, ...]'
    static constexpr bool has_dynamics_hess_prod = true;  // linear map (HVP = 0)
    static constexpr bool has_constr_hess_prod = true;  // linear (vacuous)
    static constexpr std::array<int, 8> state_box_idx = {0, 1, 2, 3, 4, 5, 6, 7};
    static constexpr std::array<int, 3> control_box_idx = {0, 1, 2};
    static constexpr std::array<int, 8> terminal_state_box_idx = {0, 1, 2, 3, 4, 5, 6, 7};
    static constexpr std::array<int, 0> ineq_soft_idx = {};
    static constexpr std::array<int, 0> eq_soft_idx = {};
    static constexpr std::array<int, 0> lin_soft_idx = {};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_eq_soft_idx = {};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {};
    static constexpr std::array<int, 0> state_box_soft_idx = {};
    static constexpr std::array<int, 0> control_box_soft_idx = {};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {};
};

// Dims-independent system data and cost kernels shared by both problems.
// All matrices are fixed 8-state / 3-control (the acados mass-spring dims).
struct MassSpringRefData
{
    static constexpr int nx = 8;
    static constexpr int nu = 3;
    static constexpr int nmass = nx / 2;

    using Mat8 = Eigen::Matrix<double, nx, nx>;
    using Mat83 = Eigen::Matrix<double, nx, nu>;
    using Vec8 = Eigen::Matrix<double, nx, 1>;
    using Vec3 = Eigen::Matrix<double, nu, 1>;

    double Ts = 0.5;
    Mat8 A{};
    Mat83 B{};
    Vec8 b{};

    MassSpringRefData()
    {
        // continuous-time system  Ac = [0  I; T  0],  Bc = [0; I_nu]
        Mat8 T = Mat8::Zero();
        for (int i = 0; i < nmass; ++i)
        {
            T(i, i) = -2.0;
            if (i + 1 < nmass)
            {
                T(i, i + 1) = 1.0;
                T(i + 1, i) = 1.0;
            }
        }
        Mat8 Ac = Mat8::Zero();
        Ac.block(0, nmass, nmass, nmass) =
            Mat8::Identity().topLeftCorner(nmass, nmass);
        Ac.block(nmass, 0, nmass, nmass) = T;
        Mat83 Bc = Mat83::Zero();
        for (int i = 0; i < nu; ++i)
        {
            Bc(nmass + i, i) = 1.0;
        }

        const Mat8 E = (Ts * Ac).exp();
        A = E;
        B = (E - Mat8::Identity()) * Ac.inverse() * Bc;
        b.setConstant(0.1);
    }

    Vec8 x0() const
    {
        Vec8 x0;
        x0.setZero();
        x0(0) = 2.5;
        x0(1) = 2.5;
        return x0;
    }

    Vec8 next(const Vec8& x, const Vec3& u) const
    {
        return A * x + B * u + b;
    }

    double stage_cost(const Vec8& x, const Vec3& u) const
    {
        return 0.5 * (x.dot(x) + 2.0 * u.dot(u)) + 0.1 * x.sum()
             + 0.2 * u.sum();
    }

    Vec8 stage_grad_x(const Vec8& x) const
    {
        return x + 0.1 * Vec8::Constant(1.0);
    }

    Vec3 stage_grad_u(const Vec3& u) const
    {
        return 2.0 * u + 0.2 * Vec3::Constant(1.0);
    }

    double term_cost(const Vec8& x) const
    {
        return 0.5 * x.dot(x) + 0.1 * x.sum();
    }

    Vec8 term_grad(const Vec8& x) const
    {
        return x + 0.1 * Vec8::Constant(1.0);
    }
};

class MassSpringRefBox : public ocp::Problem<MassSpringRefBoxDims>
{
public:
    MassSpringRefBox() = default;

    state_t initial_state() const
    {
        return data_.x0();
    }

    state_t dynamics_next_state(int, const state_t& x, const control_t& u) const
    {
        return data_.next(x, u);
    }

    void dynamics_jacobian(int, const state_t&, const control_t&,
                           dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
    {
        df_dx = data_.A;
        df_du = data_.B;
    }

    void dynamics_hess_prod(int, const state_t&, const control_t&,
                            const state_t&, const state_t&, const control_t&,
                            state_t& hv_x, control_t& hv_u) const
    {
        // linear dynamics map: Hessian is zero
        hv_x.setZero();
        hv_u.setZero();
    }

    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return data_.stage_cost(x, u);
    }

    stage_grad_t stage_cost_gradient(int, const state_t& x,
                                     const control_t& u) const
    {
        stage_grad_t g;
        g.head<nx>() = data_.stage_grad_x(x);
        g.tail<nu>() = data_.stage_grad_u(u);
        return g;
    }

    stage_hess_t stage_cost_hessian(int, const state_t&, const control_t&) const
    {
        stage_hess_t H;
        H.setZero();
        H.block(0, 0, nx, nx) = dyn_df_dx_t::Identity();
        H.block(nx, nx, nu, nu) =
            2.0 * Eigen::Matrix<double, nu, nu>::Identity();
        return H;
    }

    double terminal_cost_value(const state_t& x) const
    {
        return data_.term_cost(x);
    }

    term_grad_t terminal_cost_gradient(const state_t& x) const
    {
        return data_.term_grad(x);
    }

    term_hess_t terminal_cost_hessian(const state_t&) const
    {
        return term_hess_t::Identity();
    }

    state_box_t stage_state_box_constr(int) const
    {
        state_box_t spec;
        spec.lo.setConstant(-4.0);
        spec.hi.setConstant(4.0);
        spec.soft_penalty.setZero();
        return spec;
    }

    control_box_t stage_control_box_constr(int) const
    {
        control_box_t spec;
        spec.lo.setConstant(-0.5);
        spec.hi.setConstant(0.5);
        spec.soft_penalty.setZero();
        return spec;
    }

    term_state_box_t terminal_state_box_constr() const
    {
        term_state_box_t spec;
        spec.lo.setConstant(-4.0);
        spec.hi.setConstant(4.0);
        spec.soft_penalty.setZero();
        return spec;
    }

private:
    MassSpringRefData data_;
};

class MassSpringRefTerm : public ocp::Problem<MassSpringRefTermDims>
{
public:
    MassSpringRefTerm() = default;

    state_t initial_state() const
    {
        return data_.x0();
    }

    state_t dynamics_next_state(int, const state_t& x, const control_t& u) const
    {
        return data_.next(x, u);
    }

    void dynamics_jacobian(int, const state_t&, const control_t&,
                           dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
    {
        df_dx = data_.A;
        df_du = data_.B;
    }

    void dynamics_hess_prod(int, const state_t&, const control_t&,
                            const state_t&, const state_t&, const control_t&,
                            state_t& hv_x, control_t& hv_u) const
    {
        // linear dynamics map: Hessian is zero
        hv_x.setZero();
        hv_u.setZero();
    }

    double stage_cost_value(int, const state_t& x, const control_t& u) const
    {
        return data_.stage_cost(x, u);
    }

    stage_grad_t stage_cost_gradient(int, const state_t& x,
                                     const control_t& u) const
    {
        stage_grad_t g;
        g.head<nx>() = data_.stage_grad_x(x);
        g.tail<nu>() = data_.stage_grad_u(u);
        return g;
    }

    stage_hess_t stage_cost_hessian(int, const state_t&, const control_t&) const
    {
        stage_hess_t H;
        H.setZero();
        H.block(0, 0, nx, nx) = dyn_df_dx_t::Identity();
        H.block(nx, nx, nu, nu) =
            2.0 * Eigen::Matrix<double, nu, nu>::Identity();
        return H;
    }

    double terminal_cost_value(const state_t& x) const
    {
        return data_.term_cost(x);
    }

    term_grad_t terminal_cost_gradient(const state_t& x) const
    {
        return data_.term_grad(x);
    }

    term_hess_t terminal_cost_hessian(const state_t&) const
    {
        return term_hess_t::Identity();
    }

    state_box_t stage_state_box_constr(int) const
    {
        state_box_t spec;
        spec.lo.setConstant(-4.0);
        spec.hi.setConstant(4.0);
        spec.soft_penalty.setZero();
        return spec;
    }

    control_box_t stage_control_box_constr(int) const
    {
        control_box_t spec;
        spec.lo.setConstant(-0.5);
        spec.hi.setConstant(0.5);
        spec.soft_penalty.setZero();
        return spec;
    }

    term_state_box_t terminal_state_box_constr() const
    {
        term_state_box_t spec;
        spec.lo.setConstant(-4.0);
        spec.hi.setConstant(4.0);
        spec.soft_penalty.setZero();
        return spec;
    }

    eq_term_t terminal_equality_constr(const state_t& x) const
    {
        return x.head<ne_t>();
    }

    void terminal_equality_constr_jacobian(const state_t&,
                                           eq_term_de_dx_t& e_dx) const
    {
        e_dx.setZero();
        for (int i = 0; i < ne_t; ++i)
        {
            e_dx(i, i) = 1.0;
        }
    }

    void terminal_equality_constr_hess_prod(const state_t&, const eq_term_t&,
                                            const state_t& v, state_t& hv) const
    {
        // linear constraint map: Hessian is zero
        hv.setZero();
        (void)v;
    }

private:
    MassSpringRefData data_;
};

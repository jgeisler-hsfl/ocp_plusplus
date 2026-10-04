#!/usr/bin/env python3
"""acados2ocp_pp.py -- generate an ocp++ problem class from acados_ocp_nlp.json.

Reads the acados code-generation JSON (the standard `acados_ocp_nlp.json`
artifact produced by *either* the MATLAB or the Python `acados_toolbox`
interface) and emits a self-contained ocp++ `ContinuousProblem` subclass
plus a driver and a CMake target snippet:

    <out>/<Name>.hpp          Dims + Ode + integrator alias + Problem
    <out>/<Name>_capi.hpp     extern-C wrapper around the CasADi DAE residual
    <out>/<name>_main.cpp     solve + CSV driver

The CasADi-generated C residual (`<model>_impl_dae_fun.c`, listed under
`external_function_files_model`) is NOT regenerated here; it is the shared
dynamics backend and is compiled into the target exactly as the hand-written
port does.  This script only emits the C++ *wrapper / problem class*.

Phase-1 scope (validated on masses_chain); anything outside it raises
`Unsupported`:
  * cost_type LINEAR_LS (stage and terminal)
  * box constraints only: stage state (idxbx), stage control (idxbu),
    terminal state (idxbx_e)
  * fixed initial state (idxbx_0 with lbx_0 == ubx_0)
  * integrator IRK, collocation GAUSS_LEGENDRE, 2..4 stages, >=1 sub-step
  * dynamics backend casadi (implicit DAE), nz == 0
No nonlinear / general / linear path or terminal constraints, no soft
constraints, no discrete dynamics, no algebraic states.

Usage:
    python3 tools/acados2ocp_pp.py \
        --json examples/masses_chain/acados_codegen/acados_ocp_nlp.json \
        --name MassesChainGen --out examples/masses_chain/generated
"""

import argparse
import json
import os
import re
import sys


class Unsupported(Exception):
    pass


def fail(msg):
    raise Unsupported(msg)


# ---------------------------------------------------------------------------
#  JSON extraction + validation
# ---------------------------------------------------------------------------

def _mat(val, rows, cols):
    """Normalise a JSON matrix to a list of `rows` row-lists of `cols` floats.

    Accepts either a 2-D list (list of row lists, row-major -- the acados
    convention) or a flat row-major list.  A missing / null value yields an
    empty list of rows.
    """
    if val is None:
        return [[] for _ in range(rows)]
    if rows == 0:
        return []
    if isinstance(val[0], list):          # already row-major 2-D
        if len(val) != rows:
            fail(f"matrix has {len(val)} rows, expected {rows}")
        out = [[float(v) for v in r] for r in val]
    else:                                 # flat row-major
        if len(val) != rows * cols:
            fail(f"matrix has {len(val)} entries, expected {rows * cols}")
        out = [list(val[i * cols:(i + 1) * cols]) for i in range(rows)]
    for r in out:
        if len(r) != cols:
            fail(f"matrix row has {len(r)} cols, expected {cols}")
    return out


def _allclose(a, b, tol=1e-14):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def load_spec(path):
    with open(path) as fh:
        d = json.load(fh)

    dim = d["dims"]
    cos = d["cost"]
    con = d["constraints"]
    so = d["solver_options"]
    mod = d["model"]

    spec = {}
    spec["hash"] = d.get("hash", "")
    spec["model_name"] = mod["name"]          # e.g. "masses_chain"
    spec["problem_name"] = d.get("name", mod["name"])

    # ---- dimensions -------------------------------------------------------
    nx = dim["nx"]
    nu = dim["nu"]
    nz = dim["nz"]
    if nz != 0:
        fail(f"nz = {nz} (algebraic states) are not supported in phase 1")
    spec["nx"], spec["nu"] = nx, nu

    # constraint counts (acados dims keys); all must be 0 for phase 1
    for key in ("ng", "nh", "ns", "nphi", "nr"):
        for suf in ("", "_0", "_e"):
            v = dim.get(key + suf, 0)
            if v != 0:
                fail(f"dims.{key}{suf} = {v}: non-box constraints are "
                     f"not supported in phase 1")
    spec["ng"] = spec["ne"] = spec["nl"] = 0
    spec["ng_t"] = spec["ne_t"] = spec["nl_t"] = 0

    # ---- box constraints --------------------------------------------------
    spec["state_box_idx"] = [int(v) for v in con["idxbx"]]
    spec["control_box_idx"] = [int(v) for v in con["idxbu"]]
    spec["terminal_state_box_idx"] = [int(v) for v in con["idxbx_e"]]
    spec["lbx"] = list(con["lbx"])
    spec["ubx"] = list(con["ubx"])
    spec["lbu"] = list(con["lbu"])
    spec["ubu"] = list(con["ubu"])
    spec["lbx_e"] = list(con["lbx_e"])
    spec["ubx_e"] = list(con["ubx_e"])
    # consistency: idxbx length matches lbx/ubx length
    assert len(spec["state_box_idx"]) == len(spec["lbx"]), "idxbx/lbx size"
    assert len(spec["control_box_idx"]) == len(spec["lbu"]), "idxbu/lbu size"
    assert len(spec["terminal_state_box_idx"]) == len(spec["lbx_e"])

    # ---- fixed initial state ----------------------------------------------
    idxbx_0 = [int(v) for v in con["idxbx_0"]]
    lbx_0, ubx_0 = list(con["lbx_0"]), list(con["ubx_0"])
    if idxbx_0:
        if not _allclose(lbx_0, ubx_0):
            fail("idxbx_0 with lbx_0 != ubx_0: a free (MHE) initial state "
                 "is not supported in phase 1")
        spec["fixed_initial_state"] = True
        spec["x0"] = lbx_0            # the fixed initial state vector
    else:
        spec["fixed_initial_state"] = False
        spec["x0"] = None

    # ---- cost (LINEAR_LS) -------------------------------------------------
    for label, k in (("stage", "cost_type"), ("terminal", "cost_type_e")):
        if cos[k] != "LINEAR_LS":
            fail(f"cost_type_{label} = {cos[k]!r} != LINEAR_LS")
    ny = dim["ny"]
    spec["ny"] = ny
    spec["Vx"] = _mat(cos["Vx"], ny, nx)
    spec["Vu"] = _mat(cos["Vu"], ny, nu)
    spec["W"] = _mat(cos["W"], ny, ny)
    spec["yref"] = list(cos["yref"])
    ny_e = dim["ny_e"]
    spec["ny_e"] = ny_e
    spec["Vx_e"] = _mat(cos["Vx_e"], ny_e, nx)
    spec["Vu_e"] = _mat(cos.get("Vu_e"), ny_e, nu) if nu else []
    spec["W_e"] = _mat(cos["W_e"], ny_e, ny_e)
    spec["yref_e"] = list(cos["yref_e"])

    # ---- integrator / solver options --------------------------------------
    itype = so["integrator_type"]
    if itype != "IRK":
        fail(f"integrator_type = {itype!r}, only IRK is supported")
    coll = so["collocation_type"]
    if coll != "GAUSS_LEGENDRE":
        fail(f"collocation_type = {coll!r}, only GAUSS_LEGENDRE is supported")
    nst = so["sim_method_num_stages"]
    nst = nst[0] if isinstance(nst, list) else nst
    nstep = so["sim_method_num_steps"]
    nstep = nstep[0] if isinstance(nstep, list) else nstep
    newton_iter = so["sim_method_newton_iter"]
    newton_tol = so["sim_method_newton_tol"]
    if nst not in (2, 3, 4):
        fail(f"sim_method_num_stages = {nst}, only 2..4 supported")
    if nstep < 1:
        fail(f"sim_method_num_steps = {nstep}, must be >= 1")
    spec["num_stages"] = nst
    spec["num_steps"] = nstep
    spec["newton_iter"] = newton_iter
    spec["newton_tol"] = newton_tol
    spec["hessian_approx"] = so.get("hessian_approx", "GAUSS_NEWTON")
    spec["max_iter"] = int(so.get("nlp_solver_max_iter", 100))
    spec["tol_stat"] = float(so.get("nlp_solver_tol_stat", 1e-6))
    spec["tol_eq"] = float(so.get("nlp_solver_tol_eq", 1e-6))
    spec["tol_ineq"] = float(so.get("nlp_solver_tol_ineq", 1e-6))
    spec["tol_comp"] = float(so.get("nlp_solver_tol_comp", 1e-6))

    # ---- dynamics backend -------------------------------------------------
    if mod.get("dyn_ext_fun_type") != "casadi":
        fail(f"dyn_ext_fun_type = {mod.get('dyn_ext_fun_type')!r}, "
             f"only 'casadi' supported")
    files = d.get("external_function_files_model") or []
    if not files:
        fail("no external_function_files_model: no casadi DAE residual")
    # the residual C file is the first (the *_fun.c, not the *_jac*.c)
    fun_c = None
    for f in files:
        b = os.path.basename(f)
        if b.endswith("_fun.c"):
            fun_c = b
            break
    if fun_c is None:
        fail(f"no '..._fun.c' among external_function_files_model: {files}")
    spec["c_dae_fun"] = os.path.splitext(fun_c)[0]   # e.g. masses_chain_impl_dae_fun
    spec["c_model_dir"] = os.path.dirname(files[0]) or "."
    spec["parameter_values"] = d.get("parameter_values", [0.0])

    # ---- horizon ----------------------------------------------------------
    N = dim["N"]
    steps = so.get("time_steps")
    if steps:
        spec["dt"] = float(steps[0])
        if N * spec["dt"] and not _allclose([spec["dt"]] * N, list(map(float, steps)), tol=1e-9):
            fail("non-uniform time_steps are not supported in phase 1")
    else:
        spec["dt"] = so["Tsim"] / N
    spec["N"] = N
    return spec


# ---------------------------------------------------------------------------
#  C++ formatting helpers
# ---------------------------------------------------------------------------

def cnum(v):
    """A float as a C++ literal (always with a decimal or exponent)."""
    s = repr(float(v))
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s


def cvec(values):
    """A 1-D Eigen vector initialiser: `v << a, b, c;` body (no trailing ;)."""
    return ", ".join(cnum(v) for v in values)


def cmat(m, R, C):
    """Emit a matrix as a C++ flat array body in *column-major* order.

    Column-major (Eigen's default) is used so the data can be mapped with a
    plain ``Eigen::Map<const Eigen::Matrix<double, R, C>>`` for any shape,
    including column vectors (``RowMajor`` is an illegal layout for a
    1-column matrix).  Eight values per line; every line but the last carries
    a trailing comma so the initialiser stays a single valid list.
    """
    flat = [cnum(m[i][j]) for j in range(C) for i in range(R)]
    n = len(flat)
    out = []
    for i in range(0, n, 8):
        chunk = flat[i:i + 8]
        line = "    " + ", ".join(chunk)
        if i + 8 < n:
            line += ","
        out.append(line)
    return "\n".join(out)


def carr(values):
    """A `std::array<int, N> = {...};` initialiser body."""
    return ", ".join(str(int(v)) for v in values)


# ---------------------------------------------------------------------------
#  Emitters
# ---------------------------------------------------------------------------

def emit_capi(spec, name, json_path):
    cfun = spec["c_dae_fun"]
    base = cfun
    return f"""// {name}_capi.hpp
// GENERATED by tools/acados2ocp_pp.py from {os.path.basename(json_path)} (hash {spec['hash']}).
// Do not edit by hand.
//
// extern-C ABI for the CasADi-generated implicit-DAE residual
//     {cfun}(x, xdot, u, z, p) = 0  <=>  xdot = f_expl(x, u)
// The residual's sign convention in xdot differs between acados codegen
// paths ("f - xdot" vs "xdot - f"); the Ode calibrates it at construction
// (see the c_ member) and solves F(x, 0, u) = 0 for xdot accordingly.
// Phase 1: nz = 0, np = 1 (p forwarded, ignored by this model).

#pragma once

namespace ocp
{{

namespace detail
{{

extern "C"
{{
    int {cfun}(const double** arg, double** res,
               int* iw, double* w, int mem);
}}

/// Evaluate the implicit-DAE residual. With xdot = 0 this is f_expl(x, u).
inline void {name}_residual(const double* x, const double* xdot,
                            const double* u, double p, double* F) noexcept
{{
    const double* arg[5] = {{ x, xdot, u, /* z (nz=0) */ nullptr, &p }};
    double* res[1] = {{ F }};
    (void){cfun}(arg, res, nullptr, nullptr, 0);
}}

}}  // namespace detail

}}  // namespace ocp
"""


def emit_header(spec, name, json_path):
    nx, nu = spec["nx"], spec["nu"]
    ny, ny_e = spec["ny"], spec["ny_e"]
    nst = spec["num_stages"]
    nstep = spec["num_steps"]
    gl_tag = {2: "GaussLegendre2Tag", 3: "GaussLegendre3Tag",
              4: "GaussLegendre4Tag"}[nst]
    cfun = spec["c_dae_fun"]
    pval = cnum(spec["parameter_values"][0]) if spec["parameter_values"] else "0.0"
    fixed = spec["fixed_initial_state"]
    dt = cnum(spec["dt"])

    # Dims
    dims = f"""
struct {name}Dims
{{
    static constexpr int nx = {nx};
    static constexpr int nu = {nu};
    static constexpr int ng = 0;
    static constexpr int ne = 0;
    static constexpr int nl = 0;
    static constexpr int ng_t = 0;
    static constexpr int ne_t = 0;
    static constexpr int nl_t = 0;

    static constexpr bool fixed_initial_state = {str(fixed).lower()};
    static constexpr bool has_dynamics_hess_prod = false;  // cost-only QP Hessian
    static constexpr bool has_constr_hess_prod = false;    // only (linear) box rows

    static constexpr std::array<int, {len(spec['state_box_idx'])}> state_box_idx =
        {{ {carr(spec['state_box_idx'])} }};
    static constexpr std::array<int, {len(spec['control_box_idx'])}> control_box_idx =
        {{ {carr(spec['control_box_idx'])} }};
    static constexpr std::array<int, {len(spec['terminal_state_box_idx'])}> terminal_state_box_idx =
        {{ {carr(spec['terminal_state_box_idx'])} }};

    static constexpr std::array<int, 0> ineq_soft_idx = {{}};
    static constexpr std::array<int, 0> eq_soft_idx = {{}};
    static constexpr std::array<int, 0> lin_soft_idx = {{}};
    static constexpr std::array<int, 0> terminal_ineq_soft_idx = {{}};
    static constexpr std::array<int, 0> terminal_eq_soft_idx = {{}};
    static constexpr std::array<int, 0> terminal_lin_soft_idx = {{}};
    static constexpr std::array<int, 0> state_box_soft_idx = {{}};
    static constexpr std::array<int, 0> control_box_soft_idx = {{}};
    static constexpr std::array<int, 0> terminal_state_box_soft_idx = {{}};
}};
"""

    # ODE (capi residual + central FD Jacobian)
    ode = f"""
struct {name}Ode
{{
    using P = ocp::Problem<{name}Dims>;
    using state_t   = typename P::state_t;
    using control_t = typename P::control_t;
    using dyn_df_dx_t = typename P::dyn_df_dx_t;
    using dyn_df_du_t = typename P::dyn_df_du_t;

    double p_ = {pval};  // forwarded to the C residual (unused here)

    // Per-component coefficient c_i of xdot in the implicit-DAE residual
    // F(x, xdot, u). For an ODE the residual is linear in xdot with a
    // constant diagonal dF/dxdot = c_i * I; c_i = +1 (CasADi "xdot - f") or
    // -1 (acados "f - xdot"). Calibrated once at construction so f() is
    // correct regardless of which sign the codegen residual uses.
    // The raw difference F1 - F0 carries ulp-level noise, so each c_i is
    // snapped to exactly +/-1 (exact for ODE residuals, keeps f()
    // bit-identical to a hand-written port of the same convention).
    state_t c_ = state_t::Zero();

    {name}Ode()
    {{
        state_t xref, zero, one, F0, F1;
        control_t uref;
        for (int i = 0; i < P::nx; ++i) xref(i) = 0.05 * (i + 1);
        uref.setConstant(0.05);
        zero.setZero();
        one.setConstant(1.0);
        detail::{name}_residual(xref.data(), zero.data(), uref.data(), p_, F0.data());
        detail::{name}_residual(xref.data(), one.data(), uref.data(), p_, F1.data());
        for (int i = 0; i < P::nx; ++i)
            c_(i) = (F1(i) - F0(i) >= 0.0) ? 1.0 : -1.0;
    }}

    // xdot = f(x, u, t) = -c^{-1} * F(x, 0, u)  (solve the residual for xdot).
    state_t f(const state_t& x, const control_t& u, double /*t*/) const
    {{
        state_t F;
        const state_t zero = state_t::Zero();
        detail::{name}_residual(x.data(), zero.data(), u.data(), p_, F.data());
        state_t r;
        for (int i = 0; i < P::nx; ++i) r(i) = -F(i) / c_(i);
        return r;
    }}

    // Central finite differences of f(x, u) w.r.t. x and u (phase 1).
    void jacobian(const state_t& x, const control_t& u, double t,
                  dyn_df_dx_t& df_dx, dyn_df_du_t& df_du) const
    {{
        const double h = 1.0e-6;
        for (int j = 0; j < P::nx; ++j)
        {{
            state_t xp = x, xm = x;
            xp(j) += h;
            xm(j) -= h;
            df_dx.col(j) = (f(xp, u, t) - f(xm, u, t)) / (2.0 * h);
        }}
        for (int j = 0; j < P::nu; ++j)
        {{
            control_t up = u, um = u;
            up(j) += h;
            um(j) -= h;
            df_du.col(j) = (f(x, up, t) - f(x, um, t)) / (2.0 * h);
        }}
    }}
}};
"""

    # integrator alias
    integ = f"""
using {name}Integ =
    ocp::ImplicitRkIntegrator<{name}Dims, {name}Ode, {nst},
                              ocp::{gl_tag}, {nstep}>;
"""

    # Inline (C++17 implicit-inline) constexpr data arrays for the cost matrices.
    # Row-major flat storage; the ctor maps them into Eigen row-major views.
    data_arrays = []
    for attr, mat, R, C in (("Vx_", spec["Vx"], ny, nx),
                            ("Vu_", spec["Vu"], ny, nu),
                            ("W_", spec["W"], ny, ny),
                            ("Vxe_", spec["Vx_e"], ny_e, nx),
                            ("W_e_", spec["W_e"], ny_e, ny_e)):
        data_arrays.append(
            f"    static constexpr double {attr}_data[{R * C}] = {{\n"
            f"{cmat(mat, R, C)}\n    }};")
    data_arrays_block = "\n".join(data_arrays)

    # initial-state / reference initialisers
    x0_line = (f"        x0_ << {cvec(spec['x0'])};"
               if fixed else "        x0_.setZero();")
    yref_body = ", ".join(cnum(v) for v in spec["yref"])
    yref_e_body = ", ".join(cnum(v) for v in spec["yref_e"])
    # Box `lo`/`hi` initialiser lines.  A group with no active rows (empty
    # index set) leaves the default zero/empty BoxSpec untouched, so the
    # `<<` assignment is omitted for it (an empty `spec.lo << ;` is invalid).
    lbx_assign = f"        spec.lo << {cvec(spec['lbx'])};\n" if spec["lbx"] else ""
    ubx_assign = f"        spec.hi << {cvec(spec['ubx'])};\n" if spec["ubx"] else ""
    lbu_assign = f"        spec.lo << {cvec(spec['lbu'])};\n" if spec["lbu"] else ""
    ubu_assign = f"        spec.hi << {cvec(spec['ubu'])};\n" if spec["ubu"] else ""
    lbx_e_body = cvec(spec["lbx_e"]) if spec["lbx_e"] else "0.0"
    ubx_e_body = cvec(spec["ubx_e"]) if spec["ubx_e"] else "0.0"

    # terminal-box methods are only emitted when nbx_t > 0
    term_box = ""
    if spec["terminal_state_box_idx"]:
        term_box = f"""
    term_state_box_t terminal_state_box_constr() const
    {{
        term_state_box_t spec;
        spec.lo << {lbx_e_body};
        spec.hi << {ubx_e_body};
        spec.soft_penalty.setZero();
        return spec;
    }}"""

    prob = f"""
class {name}
    : public ocp::ContinuousProblem<{name}Dims, {name}Ode, {name}Integ>
{{
    using Base =
        ocp::ContinuousProblem<{name}Dims, {name}Ode, {name}Integ>;

public:
    {name}() : Base({name}Ode{{}}, {dt})
    {{
        {x0_line}
        yref_ << {yref_body};
        yref_e_ << {yref_e_body};

        Vx_  = Eigen::Map<const Eigen::Matrix<double, {ny}, {nx}>>(Vx__data);
        Vu_  = Eigen::Map<const Eigen::Matrix<double, {ny}, {nu}>>(Vu__data);
        W_   = Eigen::Map<const Eigen::Matrix<double, {ny}, {ny}>>(W__data);
        Vxe_ = Eigen::Map<const Eigen::Matrix<double, {ny_e}, {nx}>>(Vxe__data);
        We_  = Eigen::Map<const Eigen::Matrix<double, {ny_e}, {ny_e}>>(W_e__data);
    }}

{('    state_t initial_state() const { return x0_; }' if fixed else '')}

    // cost reference (first nx rows of the stage yref); exposed so a driver
    // can warm-start the interior states with it (as the acados example does).
    state_t cost_reference() const
    {{
        return yref_.head<{nx}>();
    }}

    // ---------------------------------------------------------------
    //  stage cost  L = 0.5 * (Vx x + Vu u - yref)^T W (Vx x + Vu u - yref)
    // ---------------------------------------------------------------
    double stage_cost_value(int /*k*/, const state_t& x,
                            const control_t& u) const
    {{
        const Eigen::Matrix<double, {ny}, 1> y = Vx_ * x + Vu_ * u - yref_;
        return 0.5 * y.dot(W_ * y);
    }}

    stage_grad_t stage_cost_gradient(int /*k*/, const state_t& x,
                                     const control_t& u) const
    {{
        const Eigen::Matrix<double, {ny}, 1> y = Vx_ * x + Vu_ * u - yref_;
        const Eigen::Matrix<double, {ny}, 1> g = W_ * y;
        stage_grad_t r;
        r.head<{nx}>() = Vx_.transpose() * g;
        r.tail<{nu}>() = Vu_.transpose() * g;
        return r;
    }}

    stage_hess_t stage_cost_hessian(int /*k*/, const state_t& /*x*/,
                                    const control_t& /*u*/) const
    {{
        stage_hess_t H;
        H.topLeftCorner({nx}, {nx}) = Vx_.transpose() * W_ * Vx_;
        H.topRightCorner({nx}, {nu}) = Vx_.transpose() * W_ * Vu_;
        H.bottomLeftCorner({nu}, {nx}) = Vu_.transpose() * W_ * Vx_;
        H.bottomRightCorner({nu}, {nu}) = Vu_.transpose() * W_ * Vu_;
        return H;
    }}

    // ---------------------------------------------------------------
    //  terminal cost  L_N = 0.5 * (Vxe x - yref_e)^T We (Vxe x - yref_e)
    // ---------------------------------------------------------------
    double terminal_cost_value(const state_t& x) const
    {{
        const Eigen::Matrix<double, {ny_e}, 1> y = Vxe_ * x - yref_e_;
        return 0.5 * y.dot(We_ * y);
    }}

    term_grad_t terminal_cost_gradient(const state_t& x) const
    {{
        const Eigen::Matrix<double, {ny_e}, 1> y = Vxe_ * x - yref_e_;
        return Vxe_.transpose() * (We_ * y);
    }}

    term_hess_t terminal_cost_hessian(const state_t& /*x*/) const
    {{
        return Vxe_.transpose() * We_ * Vxe_;
    }}

    // ---------------------------------------------------------------
    //  box constraints (the only active groups; ng = ne = nl = 0)
    // ---------------------------------------------------------------
    state_box_t stage_state_box_constr(int /*k*/) const
    {{
        state_box_t spec;
{lbx_assign}{ubx_assign}        spec.soft_penalty.setZero();
        return spec;
    }}

    control_box_t stage_control_box_constr(int /*k*/) const
    {{
        control_box_t spec;
{lbu_assign}{ubu_assign}        spec.soft_penalty.setZero();
        return spec;
    }}{term_box}

private:
    {data_arrays_block}

    state_t x0_{{}};
    Eigen::Matrix<double, {ny}, {nx}> Vx_{{}};
    Eigen::Matrix<double, {ny}, {nu}> Vu_{{}};
    Eigen::Matrix<double, {ny}, {ny}> W_{{}};
    Eigen::Matrix<double, {ny_e}, {nx}> Vxe_{{}};
    Eigen::Matrix<double, {ny_e}, {ny_e}> We_{{}};
    Eigen::Matrix<double, {ny}, 1> yref_{{}};
    Eigen::Matrix<double, {ny_e}, 1> yref_e_{{}};
}};
"""

    return f"""// {name}.hpp
// GENERATED by tools/acados2ocp_pp.py from {os.path.basename(json_path)} (hash {spec['hash']}).
// Do not edit by hand; regenerate instead.
//
// ocp++ ConcreteProblem for the acados OCP "{spec['problem_name']}".
//   state  nx = {nx}, control nu = {nu}, horizon N = {spec['N']}, dt = {spec['dt']}
//   integrator: IRK {gl_tag} ({nst} stages x {nstep} sub-steps, {spec['newton_iter']} Newton iters)
//   cost: LINEAR_LS (stage {ny}-dim, terminal {ny_e}-dim)
//   Hessian model: objective-only (see has_dynamics_hess_prod); acados uses
//                  hessian_approx = {spec['hessian_approx']}.

#pragma once

#include "ocp/integrators/continuous_problem.hpp"
#include "ocp/integrators/rk_implicit.hpp"

#include "{name}_capi.hpp"

#include <Eigen/Dense>

namespace ocp
{{
{dims}
{ode}
{integ}
{prob}

}}  // namespace ocp
"""


def emit_main(spec, name, json_path):
    N = spec["N"]
    nx = spec["nx"]
    nu = spec["nu"]
    fixed = spec["fixed_initial_state"]
    max_iter = spec["max_iter"]
    tol_stat = spec["tol_stat"]
    tol_eq = spec["tol_eq"]
    tol_ineq = spec["tol_ineq"]
    tol_comp = spec["tol_comp"]
    x0_assign = ("    sol.x[0] = problem.initial_state();"
                  if fixed else "    sol.x[0].setZero();")
    return f"""// {name.lower()}_main.cpp
// GENERATED by tools/acados2ocp_pp.py from {os.path.basename(json_path)}.
// Driver: warm-start from the cost reference, solve, dump ocp_pp.csv in the
// same layout as the acados reference (ref_acados.csv) for comparison.

#include <cstdio>

#include "ocp/solvers/acados/sqp.hpp"
#include "{name}.hpp"

using namespace ocp;

int main()
{{
    {name} problem;

    SqpOptions opts;
    opts.print_level = 1;
    opts.max_iter = {max_iter};
    opts.tol_stat = {cnum(tol_stat)};
    opts.tol_eq = {cnum(tol_eq)};
    opts.tol_ineq = {cnum(tol_ineq)};
    opts.tol_comp = {cnum(tol_comp)};

    SqpSolver<{name}> solver(opts);
    Solution<{name}> sol({N});
    {x0_assign}
    for (int k = 0; k < {N}; ++k)
    {{
        sol.u[k].setZero();
        sol.x[k + 1] = problem.cost_reference();
    }}

    const Status st = solver.solve(problem, sol);
    std::printf("{name} (generated): status=%d cost=%.12f (%d SQP iters)\\n",
                static_cast<int>(sol.status), sol.cost_value,
                solver.statistics().iter);
    if (st != Status::kSolved)
    {{
        std::printf("{name} (generated): did not reach kSolved (status=%d)\\n",
                    static_cast<int>(st));
    }}

    // CSV dump, same layout as the acados reference (ref_acados.csv).
    const char* csv_name = "ocp_pp.csv";
    std::FILE* f = std::fopen(csv_name, "w");
    if (f == nullptr)
    {{
        std::printf("FAIL: could not open %s\\n", csv_name);
        return 1;
    }}
    const bool solved = (sol.status == Status::kSolved);
    std::fprintf(f, "# {name} ocp_pp\\n");
    std::fprintf(f, "# nx={nx} nu={nu} N={N}\\n");
    std::fprintf(f, "# status=%d cost=%.15g\\n",
                 solved ? 0 : static_cast<int>(sol.status), sol.cost_value);
    std::fprintf(f, "stage");
    for (int i = 0; i < {nx}; ++i)
    {{
        std::fprintf(f, ",x%d", i);
    }}
    for (int i = 0; i < {nu}; ++i)
    {{
        std::fprintf(f, ",u%d", i);
    }}
    std::fprintf(f, "\\n");
    for (int k = 0; k <= {N}; ++k)
    {{
        std::fprintf(f, "%d", k);
        for (int i = 0; i < {nx}; ++i)
        {{
            std::fprintf(f, ",%.15g", static_cast<double>(sol.x[k](i)));
        }}
        if (k < {N})
        {{
            for (int i = 0; i < {nu}; ++i)
            {{
                std::fprintf(f, ",%.15g", static_cast<double>(sol.u[k](i)));
            }}
        }}
        std::fprintf(f, "\\n");
    }}
    std::fclose(f);
    std::printf("wrote %s\\n", csv_name);
    return 0;
}}
"""


def emit_cmake(spec, name, outdir, csrc=None):
    cfun = spec["c_dae_fun"]
    if csrc is None:
        csrc = f"{outdir}/{spec['c_model_dir']}/{cfun}.c"
    return f"""# GENERATED by tools/acados2ocp_pp.py -- add to CMakeLists.txt
# (enable_language(C) must already be in effect for the target's C residual.)
set({name}_MODEL_C {csrc})
set_source_files_properties(${{{name}_MODEL_C}} PROPERTIES
  COMPILE_OPTIONS "-O2;-w" LANGUAGE C)
add_executable({name.lower()}_gen
               {outdir}/{name.lower()}_main.cpp ${{{name}_MODEL_C}})
target_include_directories({name.lower()}_gen
                           PRIVATE include {outdir} ${{EIGEN3_INCLUDE_DIR}})
target_compile_options({name.lower()}_gen PRIVATE ${{OCP_WARN_FLAGS}})
"""


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def pascal(s):
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[^a-zA-Z0-9]+", s)
                   if p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True, help="path to acados_ocp_nlp.json")
    ap.add_argument("--name", default=None,
                    help="C++ class prefix (default: PascalCase of model name)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--cmake", action="store_true",
                    help="also write a CMake target snippet to <out>/CMake_snippet.txt")
    ap.add_argument("--csrc", default=None,
                    help="path to the CasADi DAE residual .c (for the CMake "
                         "snippet; defaults to <out>/<model_dir>/<cfun>.c)")
    args = ap.parse_args(argv)

    spec = load_spec(args.json)
    name = args.name or pascal(spec["model_name"])
    os.makedirs(args.out, exist_ok=True)

    files = {
        f"{name}.hpp": emit_header(spec, name, args.json),
        f"{name}_capi.hpp": emit_capi(spec, name, args.json),
        f"{name.lower()}_main.cpp": emit_main(spec, name, args.json),
    }
    for fn, body in files.items():
        path = os.path.join(args.out, fn)
        with open(path, "w") as fh:
            fh.write(body)
        print(f"wrote {path}")

    if args.cmake:
        cm = emit_cmake(spec, name, args.out, args.csrc)
        cm_path = os.path.join(args.out, "CMake_snippet.txt")
        with open(cm_path, "w") as fh:
            fh.write(cm)
        print(f"wrote {cm_path}")
        print("\n" + cm)
    else:
        print("\n# CMake snippet (add to your CMakeLists.txt):")
        print(emit_cmake(spec, name, args.out, args.csrc))
    return 0


if __name__ == "__main__":
    sys.exit(main())

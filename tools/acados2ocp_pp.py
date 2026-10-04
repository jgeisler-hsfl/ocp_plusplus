#!/usr/bin/env python3
"""acados2ocp_pp.py -- generate an ocp++ problem class from acados_ocp_nlp.json.

Reads the acados code-generation JSON (the standard `acados_ocp_nlp.json`
artifact produced by *either* the MATLAB or the Python `acados_toolbox`
interface) and emits a self-contained ocp++ `ContinuousProblem` subclass
plus a driver and a CMake target snippet:

    <out>/<Name>.hpp          Dims + Ode + integrator alias + Problem
    <out>/<Name>_capi.hpp     extern-C wrappers + CasADi workspaces
    <out>/<name>_main.cpp     solve + CSV driver

The CasADi-generated C value functions listed under `external_function_files_*`
are NOT regenerated here; they are linked into the target and wrapped by the
generated capi header.  Their argument layout is read from the
`/* <sym>:(i0[n],...)->(o0[n]) */` ABI comment at the top of each file, so no
hand-maintained signature table is needed.  Jacobians and Hessians of
nonlinear costs / constraints are central finite differences of the value
functions (consistent with the dynamics-Jacobian treatment).

Supported feature set (validated on masses_chain, pendulum_on_cart, unicycle,
p5probe, p5probe_b):
  * cost_type LINEAR_LS / NONLINEAR_LS / EXTERNAL for the stage-0, path and
    terminal scope
  * integrator IRK (GAUSS_LEGENDRE / RADAU_IIA, 2..4 stages) or
    ERK (K1..K4), >= 1 sub-step
  * dynamics backend casadi (explicit ODE for ERK, implicit DAE for IRK),
    nz == 0, np <= 1
  * box constraints: stage state (incl. stage-0 idxbx_0), stage control,
    terminal state
  * free or fixed initial state (idxbx_0 with lbx_0 == ubx_0 => fixed)
  * linear two-sided stage / terminal constraints (C/D/lg/ug)
  * nonlinear (h) stage / terminal constraints; rows are split into
    inequalities (lo and hi sides) and equalities (lo == hi)
  * soft constraints: h rows use Zl / Zu per side (exact acados mapping);
    box / linear rows require Zl == Zu (ocp++ carries one weight per row)
  * terminal time-varying functions (t = N * dt is passed to the C functions)

Not supported (raises Unsupported):
  * nz > 0 (algebraic states), np > 1 (multiple parameters)
  * sparse casadi arguments / outputs
  * stage-0-specific constraint rows (ng_0 / nh_0) or stage-0 soft rows
  * stage-0-specific control box (idxbu_0), terminal x0 box (idxbxe_0)
  * phi soft rows (idxsphi)
  * finite slack bounds lsh / ush (ignored; the slack penalty keeps the
    slacks small in practice)

Usage:
    python3 tools/acados2ocp_pp.py \
        --json examples/masses_chain/acados_codegen/acados_ocp_nlp.json \
        --name MassesChainGen --out examples/masses_chain/generated
"""

import argparse
import json
import math
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

    Accepts either a 2-D list (row-major -- the acados convention) or a flat
    row-major list.  A missing / null value yields an empty list of rows.
    """
    if val is None:
        return [[] for _ in range(rows)]
    if rows == 0:
        return []
    if isinstance(val[0], list):          # already row-major 2-D
        if len(val) != rows:
            fail(f"matrix has {len(val)} rows, expected {rows}")
        out = [[float(v) for v in r] for r in val]
    else:                                  # flat row-major
        if len(val) != rows * cols:
            fail(f"matrix has {len(val)} entries, expected {rows * cols}")
        out = [list(val[i * cols:(i + 1) * cols]) for i in range(rows)]
    for r in out:
        if len(r) != cols:
            fail(f"matrix row has {len(r)} cols, expected {cols}")
    return out


def _allclose(a, b, tol=1e-14):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def _vec(val, n, label):
    """Normalise a JSON vector to a list of n floats (None -> all zeros)."""
    if val is None:
        return [0.0] * n
    if len(val) != n:
        fail(f"{label} has {len(val)} entries, expected {n}")
    return [float(v) for v in val]


def _split_h(lh, uh):
    """Split h rows with bounds (l, u) into ocp++ rows.

    Returns (ineq, eq): ineq is a list of (h_row, 'lo'|'hi') where 'lo' rows
    carry g = l - h <= 0 and 'hi' rows carry g = h - u <= 0; eq is a list of
    h_row indices carrying e = h - l == 0 (l == u).  Fully free rows are
    dropped.
    """
    ineq, eq = [], []
    for i in range(len(lh)):
        l, u = lh[i], uh[i]
        has_lo = not (math.isinf(l) and l < 0)
        has_hi = not (math.isinf(u) and u > 0)
        if not has_lo and not has_hi:
            continue
        if has_lo and has_hi and abs(l - u) <= 1e-12 * max(1.0, abs(l), abs(u)):
            eq.append(i)
        else:
            if has_lo:
                ineq.append((i, "lo"))
            if has_hi:
                ineq.append((i, "hi"))
    return ineq, eq


def parse_c_sig(cpath, sym):
    """Parse the casadi ABI comment of `sym` from the C source file.

    Returns (inputs, outputs) where each side is a list of (name, size) with
    size None for dense (unannotated) arguments.
    """
    with open(cpath) as fh:
        txt = fh.read()
    m = re.search(r"/\*\s*" + re.escape(sym) + r"\s*:\s*\(([^)]*)\)\s*->"
                  r"\s*\(([^)]*)\)\s*\*/", txt)
    if not m:
        fail(f"no casadi ABI comment for '{sym}' in {cpath}")

    def parse_side(s):
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            mm = re.match(r"[io]\d+\[([^\]]*)\]$", part)
            if mm:
                inner = mm.group(1)
                if "nz" in inner or "," in inner:
                    fail(f"sparse ABI argument {part!r} in {cpath} is "
                         f"not supported")
                out.append((part, 0 if inner == "" else int(inner)))
            else:
                if not re.match(r"[io]\d+$", part):
                    fail(f"unrecognized ABI token {part!r} in {cpath}")
                out.append((part, None))   # dense
        return out

    return parse_side(m.group(1)), parse_side(m.group(2))


def plan_args(tokens, kind, nx, nu, np_, terminal):
    """Map casadi input tokens onto the canonical slots
    (x, [xdot], u, z, p, t) in argument order.

    Returns a list of (symbol, actual_size) covering the whole argument list.
    Trailing slots (z, p, t) appear as a prefix; a slot whose token does not
    fit is treated as absent.  Dense tokens map to their structural size.
    """
    u_size = 0 if terminal else nu
    plan = []
    i = 0

    def consume(sym, allowed):
        nonlocal i
        if i >= len(tokens):
            fail(f"casadi signature: ran out of args (expected {sym})")
        name, tsz = tokens[i]
        if tsz not in allowed:
            fail(f"casadi signature: arg {name} size {tsz} not allowed for "
                 f"{sym} (expected one of {sorted(allowed, key=lambda v: (v is None, v))})")
        plan.append((sym, tsz))
        i += 1

    consume("x", {None, nx})
    if kind == "dae":
        consume("xdot", {None, nx})
    # u: dense (-> u_size), explicit u_size, or [0]
    if i < len(tokens):
        name, tsz = tokens[i]
        if tsz is None or tsz in (0, u_size):
            plan.append(("u", u_size if tsz is None else tsz))
            i += 1
        elif u_size > 0:
            fail(f"casadi signature: expected u (size {u_size}), got "
                 f"{name} size {tsz}")
    # z, p, t in order; each slot may be absent.
    allowed = {
        "z": {0},
        "p": ({None, np_} if np_ > 0 else {0, None}),
        "t": {0, 1},
    }
    for sym in ("z", "p", "t"):
        if i >= len(tokens):
            break
        _, tsz = tokens[i]
        if tsz in allowed[sym]:
            plan.append((sym, tsz))
            i += 1
    if i != len(tokens):
        fail(f"casadi signature: unmapped args {tokens[i:]}")
    # normalise dense -> structural size
    norm = []
    for sym, tsz in plan:
        if tsz is None:
            actual = {"u": u_size, "p": np_}.get(sym, 0)
        else:
            actual = tsz
        norm.append((sym, actual))
    return norm


def _find_cfile(files, suffix, label):
    for f in files or []:
        b = os.path.basename(f)
        if b.endswith(suffix) and not re.search(r"_jac|_hess|_vde", b):
            return f
    fail(f"no value function '{suffix}' for {label} in the external "
         f"function files: {files}")


def load_spec(path):
    with open(path) as fh:
        d = json.load(fh)

    base = os.path.dirname(os.path.abspath(path))

    def resolve(f):
        return (f if os.path.isabs(f)
                else os.path.normpath(os.path.join(base, f)))

    dim = d["dims"]
    cos = d["cost"]
    con = d["constraints"]
    so = d["solver_options"]
    mod = d["model"]

    spec = {}
    spec["hash"] = d.get("hash", "")
    spec["model_name"] = mod["name"]
    spec["problem_name"] = d.get("name", mod["name"])

    # ---- dimensions -------------------------------------------------------
    nx, nu = dim["nx"], dim["nu"]
    nz = dim.get("nz", 0)
    np_ = dim.get("np", 0)
    if nz != 0:
        fail(f"nz = {nz}: algebraic states are not supported")
    if np_ > 1:
        fail(f"np = {np_}: more than one parameter is not supported")
    spec["nx"], spec["nu"], spec["np"] = nx, nu, np_
    spec["N"] = dim["N"]

    # ---- integrator / solver options ---------------------------------------
    itype = so["integrator_type"]
    coll = so.get("collocation_type")
    nst_list = so["sim_method_num_stages"]
    nstep_list = so["sim_method_num_steps"]
    nst = nst_list[0] if isinstance(nst_list, list) else nst_list
    nstep = nstep_list[0] if isinstance(nstep_list, list) else nstep_list
    if isinstance(nst_list, list) and len(set(nst_list)) != 1:
        fail(f"non-uniform sim_method_num_stages {nst_list} not supported")
    if isinstance(nstep_list, list) and len(set(nstep_list)) != 1:
        fail(f"non-uniform sim_method_num_steps {nstep_list} not supported")
    if itype == "IRK":
        if coll not in ("GAUSS_LEGENDRE", "RADAU_IIA"):
            fail(f"collocation_type {coll!r}: only GAUSS_LEGENDRE or "
                 f"RADAU_IIA supported")
        if nst not in (2, 3, 4):
            fail(f"IRK stages {nst}: only 2..4 supported")
        tag = ("GaussLegendre%dTag" % nst if coll == "GAUSS_LEGENDRE"
               else "RadauIia%dTag" % nst)
        integ_class = "ImplicitRkIntegrator"
    elif itype == "ERK":
        if nst not in (1, 2, 3, 4):
            fail(f"ERK stages {nst}: only 1..4 supported")
        tag = f"K{nst}Tag"
        integ_class = "ExplicitRkIntegrator"
    else:
        fail(f"integrator_type {itype!r}: only IRK or ERK supported")
    if nstep < 1:
        fail(f"sim_method_num_steps = {nstep}, must be >= 1")
    spec["integ_kind"] = itype
    spec["num_stages"] = nst
    spec["num_steps"] = nstep
    spec["gl_tag"] = tag
    spec["integ_class"] = integ_class
    spec["newton_iter"] = so.get("sim_method_newton_iter", 10)
    spec["newton_tol"] = so.get("sim_method_newton_tol", 1e-8)
    spec["hessian_approx"] = so.get("hessian_approx", "GAUSS_NEWTON")
    spec["max_iter"] = int(so.get("nlp_solver_max_iter", 100))
    spec["tol_stat"] = float(so.get("nlp_solver_tol_stat", 1e-6))
    spec["tol_eq"] = float(so.get("nlp_solver_tol_eq", 1e-6))
    spec["tol_ineq"] = float(so.get("nlp_solver_tol_ineq", 1e-6))
    spec["tol_comp"] = float(so.get("nlp_solver_tol_comp", 1e-6))

    # ---- dynamics backend --------------------------------------------------
    if mod.get("dyn_ext_fun_type") != "casadi":
        fail(f"dyn_ext_fun_type = {mod.get('dyn_ext_fun_type')!r}, "
             f"only 'casadi' supported")
    model_files = [resolve(f) for f in (d.get("external_function_files_model") or [])]
    want = "impl_dae_fun.c" if itype == "IRK" else "expl_ode_fun.c"
    spec["dyn_c"] = _find_cfile(model_files, want, "dynamics")
    spec["parameter_values"] = d.get("parameter_values", []) or [0.0]

    # ---- horizon ------------------------------------------------------------
    N = spec["N"]
    steps = so.get("time_steps")
    if steps:
        spec["dt"] = float(steps[0])
        if not _allclose([spec["dt"]] * N, [float(v) for v in steps],
                         tol=1e-9):
            fail("non-uniform time_steps are not supported")
    else:
        spec["dt"] = so["Tsim"] / N
    spec["t_end"] = N * spec["dt"]

    # ---- box constraints ------------------------------------------------------
    idxbx = [int(v) for v in con.get("idxbx") or []]
    idxbu = [int(v) for v in con.get("idxbu") or []]
    idxbx_e = [int(v) for v in con.get("idxbx_e") or []]
    lbx = _vec(con.get("lbx"), len(idxbx), "lbx")
    ubx = _vec(con.get("ubx"), len(idxbx), "ubx")
    lbu = _vec(con.get("lbu"), len(idxbu), "lbu")
    ubu = _vec(con.get("ubu"), len(idxbu), "ubu")
    lbx_e = _vec(con.get("lbx_e"), len(idxbx_e), "lbx_e")
    ubx_e = _vec(con.get("ubx_e"), len(idxbx_e), "ubx_e")

    idxbx_0 = [int(v) for v in con.get("idxbx_0") or []]
    lbx_0 = _vec(con.get("lbx_0"), len(idxbx_0), "lbx_0")
    ubx_0 = _vec(con.get("ubx_0"), len(idxbx_0), "ubx_0")
    if con.get("idxbu_0"):
        fail("stage-0 control box (idxbu_0) is not supported")
    if con.get("idxbxe_0") and idxbx_e:
        fail("idxbxe_0 (terminal-box / x0 coupling) is not supported")
    # idxbxe_0 with an empty terminal box is vestigial (codegen artefact)

    fixed = bool(idxbx_0) and _allclose(lbx_0, ubx_0)
    spec["fixed_initial_state"] = fixed
    spec["x0"] = list(lbx_0) if fixed else None
    # state-box rows: union of path and stage-0 rows (the generated box
    # method branches on k; a row present in only one of the two is +-inf
    # on the other).
    union = sorted(set(idxbx) | set(idxbx_0))
    spec["state_box_union"] = union
    spec["state_box_stage0"] = {r: (l, u)
                                for r, l, u in zip(idxbx_0, lbx_0, ubx_0)}
    spec["state_box_path"] = {r: (l, u)
                              for r, l, u in zip(idxbx, lbx, ubx)}
    spec["control_box"] = list(zip(idxbu, lbu, ubu))
    spec["terminal_state_box"] = list(zip(idxbx_e, lbx_e, ubx_e))

    # ---- linear constraints (acados "g" channel) ------------------------------
    nl = dim.get("ng", 0)
    if nl:
        spec["lin"] = {
            "A": _mat(con["C"], nl, nx),
            "B": _mat(con["D"], nl, nu),
            "lo": _vec(con["lg"], nl, "lg"),
            "hi": _vec(con["ug"], nl, "ug"),
        }
    else:
        spec["lin"] = None
    nl_t = dim.get("ng_e", 0)
    if nl_t:
        spec["lin_t"] = {
            "A": _mat(con["C_e"], nl_t, nx),
            "lo": _vec(con["lg_e"], nl_t, "lg_e"),
            "hi": _vec(con["ug_e"], nl_t, "ug_e"),
        }
    else:
        spec["lin_t"] = None
    spec["nl"], spec["nl_t"] = nl, nl_t

    # ---- nonlinear (h) constraints (acados "h" channel) -----------------------
    def h_block(nh_n, key, cfiles):
        if nh_n == 0:
            return None
        lh = _vec(con.get("lh" + key), nh_n, "lh" + key)
        uh = _vec(con.get("uh" + key), nh_n, "uh" + key)
        ineq, eq = _split_h(lh, uh)
        cfile = None
        if ineq or eq:
            cfile = _find_cfile(cfiles, f"constr_h{key}_fun.c",
                                f"h constraints{key}")
        return {"lh": lh, "uh": uh, "ineq": ineq, "eq": eq, "cfile": cfile}

    # newer acados JSONs list all OCP value functions under a single key;
    # fall back to the cost / constraint split for older ones.
    ocp_files = d.get("external_function_files_ocp")
    if ocp_files is None:
        ocp_files = list(d.get("external_function_files_cost") or []) + \
            list(d.get("external_function_files_constraints") or [])
    ocp_files = [resolve(f) for f in ocp_files]
    spec["h"] = h_block(dim.get("nh", 0), "", ocp_files)
    spec["h_t"] = h_block(dim.get("nh_e", 0), "_e", ocp_files)
    spec["ng"] = len(spec["h"]["ineq"]) if spec["h"] else 0
    spec["ne"] = len(spec["h"]["eq"]) if spec["h"] else 0
    spec["ng_t"] = len(spec["h_t"]["ineq"]) if spec["h_t"] else 0
    spec["ne_t"] = len(spec["h_t"]["eq"]) if spec["h_t"] else 0
    if dim.get("nh_0", 0) or dim.get("ng_0", 0):
        fail("stage-0-specific constraint rows (nh_0 / ng_0) are not "
             "supported")

    # ---- soft constraints -------------------------------------------------------
    idxsbx = [int(v) for v in con.get("idxsbx") or []]
    idxsbu = [int(v) for v in con.get("idxsbu") or []]
    idxsg = [int(v) for v in con.get("idxsg") or []]
    idxsh = [int(v) for v in con.get("idxsh") or []]
    idxsphi = [int(v) for v in con.get("idxsphi") or []]
    if idxsphi:
        fail("phi soft constraints (idxsphi) are not supported")
    for key in ("idxsbx_0", "idxsbu_0", "idxsg_0", "idxsh_0",
                "idxsphi_0"):
        if con.get(key):
            fail(f"stage-0 soft rows ({key}) are not supported")
    Zl = _vec(cos.get("Zl"), dim.get("ns", 0), "Zl")
    Zu = _vec(cos.get("Zu"), dim.get("ns", 0), "Zu")

    def sym_w(j, label):
        if abs(Zl[j] - Zu[j]) > 1e-12:
            fail(f"asymmetric soft penalty Zl={Zl[j]} != Zu={Zu[j]} for "
                 f"{label}: ocp++ carries one weight per row")
        return Zl[j]

    off = 0
    state_box_soft = {}
    for j, r in enumerate(idxsbx):
        state_box_soft[union.index(idxbx[r])] = sym_w(off + j, "box x row")
    off += len(idxsbx)
    control_box_soft = {}
    for j, r in enumerate(idxsbu):
        control_box_soft[r] = sym_w(off + j, "box u row")
    off += len(idxsbu)
    lin_soft = {}
    for j, r in enumerate(idxsg):
        lin_soft[r] = sym_w(off + j, "linear row")
    off += len(idxsg)
    # h rows: the lo side gets Zl, the hi side gets Zu (exact acados mapping,
    # one slack per side).
    ineq_soft_w = []
    for i, side in (spec["h"]["ineq"] if spec["h"] else []):
        if i in idxsh:
            j = idxsh.index(i)
            ineq_soft_w.append(Zl[off + j] if side == "lo"
                               else Zu[off + j])
        else:
            ineq_soft_w.append(None)
    eq_soft_w = []
    for i in (spec["h"]["eq"] if spec["h"] else []):
        if i in idxsh:
            eq_soft_w.append(sym_w(off + idxsh.index(i), "h eq row"))
        else:
            eq_soft_w.append(None)
    off += len(idxsh)

    idxsbx_e = [int(v) for v in con.get("idxsbx_e") or []]
    idxsg_e = [int(v) for v in con.get("idxsg_e") or []]
    idxsh_e = [int(v) for v in con.get("idxsh_e") or []]
    idxsphi_e = [int(v) for v in con.get("idxsphi_e") or []]
    if idxsphi_e:
        fail("phi soft constraints (idxsphi_e) are not supported")
    Zl_e = _vec(cos.get("Zl_e"), dim.get("ns_e", 0), "Zl_e")
    Zu_e = _vec(cos.get("Zu_e"), dim.get("ns_e", 0), "Zu_e")

    def sym_w_e(j, label):
        if abs(Zl_e[j] - Zu_e[j]) > 1e-12:
            fail(f"asymmetric soft penalty Zl_e={Zl_e[j]} != Zu_e={Zu_e[j]}"
                 f" for {label}: ocp++ carries one weight per row")
        return Zl_e[j]

    off_e = 0
    terminal_state_box_soft = {}
    for j, r in enumerate(idxsbx_e):
        terminal_state_box_soft[r] = sym_w_e(off_e + j, "terminal box row")
    off_e += len(idxsbx_e)
    terminal_lin_soft = {}
    for j, r in enumerate(idxsg_e):
        terminal_lin_soft[r] = sym_w_e(off_e + j, "terminal linear row")
    off_e += len(idxsg_e)
    t_ineq_soft_w = []
    for i, side in (spec["h_t"]["ineq"] if spec["h_t"] else []):
        if i in idxsh_e:
            j = idxsh_e.index(i)
            t_ineq_soft_w.append(Zl_e[off_e + j] if side == "lo"
                                 else Zu_e[off_e + j])
        else:
            t_ineq_soft_w.append(None)
    t_eq_soft_w = []
    for i in (spec["h_t"]["eq"] if spec["h_t"] else []):
        if i in idxsh_e:
            t_eq_soft_w.append(sym_w_e(off_e + idxsh_e.index(i),
                                       "terminal h eq row"))
        else:
            t_eq_soft_w.append(None)

    spec["state_box_soft"] = [state_box_soft.get(r) for r in union]
    spec["control_box_soft"] = [control_box_soft.get(r) for r in idxbu]
    spec["lin_soft"] = [lin_soft.get(r) for r in range(nl)]
    if spec["h"]:
        spec["h"]["ineq_soft_w"] = ineq_soft_w
        spec["h"]["eq_soft_w"] = eq_soft_w
    if spec["h_t"]:
        spec["h_t"]["ineq_soft_w"] = t_ineq_soft_w
        spec["h_t"]["eq_soft_w"] = t_eq_soft_w
    spec["terminal_state_box_soft"] = \
        [terminal_state_box_soft.get(r) for r in idxbx_e]
    spec["terminal_lin_soft"] = \
        [terminal_lin_soft.get(r) for r in range(nl_t)]

    # ---- costs ------------------------------------------------------------------
    cost_files = ocp_files

    def cost_scope(suffix, label):
        ctype = cos.get("cost_type" + suffix)
        if ctype is None:
            return None
        if ctype not in ("LINEAR_LS", "NONLINEAR_LS", "EXTERNAL"):
            fail(f"cost_type{suffix} = {ctype!r} unsupported")
        s = {"type": ctype, "suffix": suffix, "label": label}
        if ctype == "LINEAR_LS":
            ny = dim.get("ny" + suffix, 0)
            if ny == 0:
                fail(f"cost_type{suffix} = LINEAR_LS but ny{suffix} = 0")
            s["ny"] = ny
            s["Vx"] = _mat(cos["Vx" + suffix], ny, nx)
            s["Vu"] = _mat(cos.get("Vu" + suffix), ny, nu)
            s["W"] = _mat(cos["W" + suffix], ny, ny)
            s["yref"] = _vec(cos["yref" + suffix], ny, "yref" + suffix)
        elif ctype == "NONLINEAR_LS":
            ny = dim.get("ny" + suffix, 0)
            if ny == 0:
                fail(f"cost_type{suffix} = NONLINEAR_LS but ny{suffix} = 0")
            s["ny"] = ny
            s["W"] = _mat(cos["W" + suffix], ny, ny)
            s["yref"] = _vec(cos["yref" + suffix], ny, "yref" + suffix)
            s["cfile"] = _find_cfile(cost_files, f"cost_y{suffix}_fun.c",
                                     label)
        else:  # EXTERNAL
            s["cfile"] = _find_cfile(cost_files,
                                     f"cost_ext_cost{suffix}_fun.c", label)
        return s

    spec["cost_stage"] = cost_scope("", "path cost")
    spec["cost_stage0"] = cost_scope("_0", "stage-0 cost")
    spec["cost_term"] = cost_scope("_e", "terminal cost")
    if spec["cost_stage"] is None:
        fail("no cost_type in JSON")

    # ---- collect C value functions ------------------------------------------------
    cfuns = []
    seen = set()

    def add_cfun(cfile, sym, role, terminal):
        if sym in seen:
            return
        seen.add(sym)
        ins, outs = parse_c_sig(cfile, sym)
        if len(outs) != 1:
            fail(f"casadi output of {sym} is not a single vector")
        # Only the DAE residual carries an xdot slot; cost/constraint
        # functions never take xdot, regardless of integrator kind.
        plan_kind = "dae" if (role == "ode" and itype == "IRK") else "ode"
        plan = plan_args(ins, plan_kind, nx, nu, np_, terminal)
        cfuns.append({"sym": sym, "path": cfile, "role": role,
                      "plan": plan, "terminal": terminal})

    add_cfun(spec["dyn_c"],
             os.path.splitext(os.path.basename(spec["dyn_c"]))[0],
             "ode", False)
    for key, role, terminal in (("cost_stage", "cost", False),
                                ("cost_stage0", "cost0", False),
                                ("cost_term", "cost_e", True),
                                ("h", "h", False),
                                ("h_t", "h_e", True)):
        block = spec[key]
        if block and block.get("cfile"):
            add_cfun(block["cfile"],
                     os.path.splitext(os.path.basename(block["cfile"]))[0],
                     role, terminal)
    spec["cfuns"] = cfuns
    spec["c_files"] = sorted({f["path"] for f in cfuns})
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
    including column vectors.  Eight values per line; every line but the
    last carries a trailing comma so the initialiser stays a single valid
    list.
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


def wvec(values):
    """Comma-joined weight literals (non-None entries only)."""
    return ", ".join(cnum(v) for v in values if v is not None)


INF = "inf_"
_ROLE_NAMES = {"ode": "Dyn", "cost": "Cost", "cost0": "Cost0",
               "cost_e": "CostE", "h": "H", "h_e": "HE"}


# ---------------------------------------------------------------------------
#  Emitters
# ---------------------------------------------------------------------------

def _arg_entry(sym, sz):
    """C++ expression for the argument-array entry of a plan slot."""
    if sym == "x":
        return "x"
    if sym == "xdot":
        return "xdot"
    if sym == "u":
        return "u" if sz > 0 else "nullptr"
    if sym == "z":
        return "nullptr"
    if sym == "p":
        return "(const double*)&p" if sz == 1 else "nullptr"
    if sym == "t":
        return "(const double*)&t" if sz == 1 else "nullptr"
    fail(f"unknown plan slot {sym!r}")


def emit_capi(spec, name, json_path):
    lines = [f"""// {name}_capi.hpp
// GENERATED by tools/acados2ocp_pp.py from {os.path.basename(json_path)} (hash {spec['hash']}).
// Do not edit by hand.
//
// extern-C ABI + CasADi workspaces for the generated value functions.
// Argument slots follow the casadi ABI comment in each .c file; the
// canonical order is (x, [xdot], u, z, p, t) with nz = 0, np <= 1.

#pragma once

#include <cstddef>
#include <vector>

namespace ocp
{{

namespace detail
{{

extern "C"
{{
"""]
    for f in spec["cfuns"]:
        lines.append(f"""    int {f['sym']}(const double** arg, double** res,
                   int* iw, double* w, int mem);
    int {f['sym']}_work(int* sz_arg, int* sz_res, int* sz_iw, int* sz_w);
""")
    lines.append("}\n")

    for f in spec["cfuns"]:
        sym = f["sym"]
        role = _ROLE_NAMES[f["role"]]
        plan = f["plan"]
        entries = ", ".join(_arg_entry(s, sz) for s, sz in plan)
        voids = []
        if not any(s == "u" and sz > 0 for s, sz in plan):
            voids.append("u")
        if not any(s == "xdot" for s, sz in plan):
            voids.append("xdot")
        if not any(s == "p" and sz == 1 for s, sz in plan):
            voids.append("p")
        if not any(s == "t" and sz == 1 for s, sz in plan):
            voids.append("t")
        void_body = " ".join(f"(void){v};" for v in voids)
        arg_desc = ", ".join(f"{s}[{sz}]" for s, sz in plan)
        lines.append(f"""
/// CasADi workspace + evaluator for {sym} (args: {arg_desc}).
struct {name}{role}
{{
    mutable std::vector<int> iw_;
    mutable std::vector<double> w_;

    {name}{role}()
    {{
        int sz_iw = 0, sz_w = 0;
        (void){sym}_work(nullptr, nullptr, &sz_iw, &sz_w);
        iw_.resize(std::size_t(sz_iw));
        w_.resize(std::size_t(sz_w));
    }}

    void eval(const double* x, const double* u, const double* xdot,
              double p, double t, double* out) const
    {{
        {void_body}
        const double* arg[{len(plan)}] = {{ {entries} }};
        double* res[1] = {{ out }};
        (void){sym}(arg, res,
                    iw_.empty() ? nullptr : iw_.data(),
                    w_.empty() ? nullptr : w_.data(), 0);
    }}
}};
""")
    lines.append("""
}  // namespace detail

}  // namespace ocp
""")
    return "".join(lines)


def _soft_idx_array(label, values):
    idx = [i for i, v in enumerate(values) if v is not None]
    return (f"    static constexpr std::array<int, {len(idx)}> {label} = "
            f"{{ {carr(idx)} }};")


def emit_header(spec, name, json_path):
    nx, nu = spec["nx"], spec["nu"]
    nst = spec["num_stages"]
    nstep = spec["num_steps"]
    fixed = spec["fixed_initial_state"]
    dt = cnum(spec["dt"])
    dt_expr = repr(spec["dt"]) if isinstance(spec["dt"], float) else str(spec["dt"])
    t_end = cnum(spec["t_end"])
    pval = cnum(spec["parameter_values"][0])
    integ_inc = ("rk_explicit" if spec["integ_kind"] == "ERK"
                 else "rk_implicit")

    ng = len(spec["h"]["ineq"]) if spec["h"] else 0
    ne = len(spec["h"]["eq"]) if spec["h"] else 0
    ng_t = len(spec["h_t"]["ineq"]) if spec["h_t"] else 0
    ne_t = len(spec["h_t"]["eq"]) if spec["h_t"] else 0
    nl, nl_t = spec["nl"], spec["nl_t"]

    # ------------------------------------------------------------------ Dims
    soft_arrays = "\n".join([
        _soft_idx_array("ineq_soft_idx",
                        spec["h"]["ineq_soft_w"] if spec["h"] else []),
        _soft_idx_array("eq_soft_idx",
                        spec["h"]["eq_soft_w"] if spec["h"] else []),
        _soft_idx_array("lin_soft_idx", spec["lin_soft"]),
        _soft_idx_array("terminal_ineq_soft_idx",
                        spec["h_t"]["ineq_soft_w"] if spec["h_t"] else []),
        _soft_idx_array("terminal_eq_soft_idx",
                        spec["h_t"]["eq_soft_w"] if spec["h_t"] else []),
        _soft_idx_array("terminal_lin_soft_idx", spec["terminal_lin_soft"]),
        _soft_idx_array("state_box_soft_idx", spec["state_box_soft"]),
        _soft_idx_array("control_box_soft_idx", spec["control_box_soft"]),
        _soft_idx_array("terminal_state_box_soft_idx",
                        spec["terminal_state_box_soft"]),
    ])
    dims = f"""
struct {name}Dims
{{
    static constexpr int nx = {nx};
    static constexpr int nu = {nu};
    static constexpr int ng = {ng};
    static constexpr int ne = {ne};
    static constexpr int nl = {nl};
    static constexpr int ng_t = {ng_t};
    static constexpr int ne_t = {ne_t};
    static constexpr int nl_t = {nl_t};

    static constexpr bool fixed_initial_state = {str(fixed).lower()};
    static constexpr bool has_dynamics_hess_prod = false;  // Gauss-Newton QP Hessian
    static constexpr bool has_constr_hess_prod = false;

    static constexpr std::array<int, {len(spec['state_box_union'])}> state_box_idx =
        {{ {carr(spec['state_box_union'])} }};
    static constexpr std::array<int, {len(spec['control_box'])}> control_box_idx =
        {{ {carr([r for r, _, _ in spec['control_box']])} }};
    static constexpr std::array<int, {len(spec['terminal_state_box'])}> terminal_state_box_idx =
        {{ {carr([r for r, _, _ in spec['terminal_state_box']])} }};

{soft_arrays}
}};
"""

    # ------------------------------------------------------------------ Ode
    p_use = "p_"
    # The Ode struct always carries a p_ member; the problem-class helpers
    # only do when np > 0 (otherwise they pass 0.0).
    p_cls = "p_" if spec["np"] > 0 else "0.0"
    if spec["integ_kind"] == "IRK":
        f_body = f"""    // xdot = f(x, u, t) = -c^(-1) * F(x, 0, u)  (solve the residual for
    // xdot; the c_ calibration makes this exact for either residual sign).
    state_t f(const state_t& x, const control_t& u, double t) const
    {{
        state_t F;
        const state_t zero = state_t::Zero();
        dyn_.eval(x.data(), u.data(), zero.data(), {p_use}, t, F.data());
        state_t r;
        for (int i = 0; i < P::nx; ++i) r(i) = -F(i) / c_(i);
        return r;
    }}"""
    else:
        f_body = f"""    // xdot = f_expl(x, u, t) directly from the explicit ODE function.
    state_t f(const state_t& x, const control_t& u, double t) const
    {{
        state_t r;
        dyn_.eval(x.data(), u.data(), nullptr, {p_use}, t, r.data());
        return r;
    }}"""

    calib = ""
    if spec["integ_kind"] == "IRK":
        calib = f"""
    // Per-component coefficient c_i of xdot in the implicit-DAE residual
    // F(x, xdot, u). For an ODE the residual is linear in xdot with a
    // constant diagonal dF/dxdot = c_i * I; c_i = +1 (CasADi "xdot - f") or
    // -1 (acados "f - xdot"). Calibrated once at construction so f() is
    // correct regardless of which sign the codegen residual uses. The raw
    // difference F1 - F0 carries ulp-level noise, so each c_i is snapped to
    // exactly +/-1 (exact for ODE residuals).
    state_t c_ = state_t::Zero();

    {name}Ode()
    {{
        state_t xref, zero, one, F0, F1;
        control_t uref;
        for (int i = 0; i < P::nx; ++i) xref(i) = 0.05 * (i + 1);
        uref.setConstant(0.05);
        zero.setZero();
        one.setConstant(1.0);
        dyn_.eval(xref.data(), uref.data(), zero.data(), {p_use}, 0.0,
                  F0.data());
        dyn_.eval(xref.data(), uref.data(), one.data(), {p_use}, 0.0,
                  F1.data());
        for (int i = 0; i < P::nx; ++i)
            c_(i) = (F1(i) - F0(i) >= 0.0) ? 1.0 : -1.0;
    }}
"""

    ode = f"""
struct {name}Ode
{{
    using P = ocp::Problem<{name}Dims>;
    using state_t   = typename P::state_t;
    using control_t = typename P::control_t;
    using dyn_df_dx_t = typename P::dyn_df_dx_t;
    using dyn_df_du_t = typename P::dyn_df_du_t;

    detail::{name}Dyn dyn_;
    double {p_use} = {pval};
{calib}
    {f_body}

    // Central finite differences of f(x, u, t) w.r.t. x and u.
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

    integ = f"""
using {name}Integ =
    ocp::{spec['integ_class']}<{name}Dims, {name}Ode, {nst},
                              ocp::{spec['gl_tag']}, {nstep}>;
"""

    # ------------------------------------------------------------------ Problem
    members = []
    if spec["np"] > 0:
        members.append(f"    double p_ = {pval};")
    ctor_lines = []
    data_blocks = []
    helpers = []
    cost_methods = []

    if fixed:
        members.append("    state_t x0_{};")
        ctor_lines.append(f"        x0_ << {cvec(spec['x0'])};")

    # ---- cost scopes --------------------------------------------------------
    def cost_scope_body(scope, prefix, terminal):
        """Return (methods, members, ctor lines, data blocks, helpers)
        for one cost scope; prefix is 'cp' / 'c0' / 'ct'."""
        ct = scope["type"]
        meth, mem, ctor, data, help_ = [], [], [], [], []
        if ct == "LINEAR_LS":
            ny = scope["ny"]
            Vx = scope["Vx"]
            W = scope["W"]
            yref = scope["yref"]
            data.append(
                f"    static constexpr double {prefix}Vx_data[{ny * nx}] = {{\n"
                f"{cmat(Vx, ny, nx)}\n    }};")
            data.append(
                f"    static constexpr double {prefix}W_data[{ny * ny}] = {{\n"
                f"{cmat(W, ny, ny)}\n    }};")
            mem.append(f"    Eigen::Matrix<double, {ny}, {nx}> {prefix}Vx_{{}};")
            mem.append(f"    Eigen::Matrix<double, {ny}, {ny}> {prefix}W_{{}};")
            mem.append(f"    Eigen::Matrix<double, {ny}, 1> {prefix}yref_{{}};")
            ctor.append(f"        {prefix}Vx_ = Eigen::Map<const "
                        f"Eigen::Matrix<double, {ny}, {nx}>>"
                        f"({prefix}Vx_data);")
            ctor.append(f"        {prefix}W_ = Eigen::Map<const "
                        f"Eigen::Matrix<double, {ny}, {ny}>>"
                        f"({prefix}W_data);")
            ctor.append(f"        {prefix}yref_ << {cvec(yref)};")
            if not terminal and nu > 0:
                Vu = scope["Vu"] or [[0.0] * nu for _ in range(ny)]
                data.append(
                    f"    static constexpr double {prefix}Vu_data"
                    f"[{ny * nu}] = {{\n{cmat(Vu, ny, nu)}\n    }};")
                mem.append(
                    f"    Eigen::Matrix<double, {ny}, {nu}> {prefix}Vu_{{}};")
                ctor.append(f"        {prefix}Vu_ = Eigen::Map<const "
                            f"Eigen::Matrix<double, {ny}, {nu}>>"
                            f"({prefix}Vu_data);")
                yexpr = f"{prefix}Vx_ * x + {prefix}Vu_ * u - {prefix}yref_"
                grad_ret = (f"            stage_grad_t r;\n"
                            f"            r.head<{nx}>() = {prefix}Vx_."
                            f"transpose() * g;\n"
                            f"            r.tail<{nu}>() = {prefix}Vu_."
                            f"transpose() * g;\n"
                            f"            return r;")
                hess_body = (f"            stage_hess_t H;\n"
                             f"            H.topLeftCorner({nx}, {nx}) = "
                             f"{prefix}Vx_.transpose() * {prefix}W_ * "
                             f"{prefix}Vx_;\n"
                             f"            H.topRightCorner({nx}, {nu}) = "
                             f"{prefix}Vx_.transpose() * {prefix}W_ * "
                             f"{prefix}Vu_;\n"
                             f"            H.bottomLeftCorner({nu}, {nx}) = "
                             f"{prefix}Vu_.transpose() * {prefix}W_ * "
                             f"{prefix}Vx_;\n"
                             f"            H.bottomRightCorner({nu}, {nu}) = "
                             f"{prefix}Vu_.transpose() * {prefix}W_ * "
                             f"{prefix}Vu_;\n"
                             f"            return H;")
            else:
                yexpr = f"{prefix}Vx_ * x - {prefix}yref_"
                grad_ret = (f"            return {prefix}Vx_.transpose() "
                            f"* g;")
                hess_body = (f"            return {prefix}Vx_.transpose() "
                             f"* {prefix}W_ * {prefix}Vx_;")
            if terminal:
                meth.append(f"""    double {prefix}_value(const state_t& x, double /*t*/) const
    {{
        const Eigen::Matrix<double, {ny}, 1> y = {yexpr};
        return 0.5 * y.dot({prefix}W_ * y);
    }}

    term_grad_t {prefix}_grad(const state_t& x, double /*t*/) const
    {{
        const Eigen::Matrix<double, {ny}, 1> y = {yexpr};
        const Eigen::Matrix<double, {ny}, 1> g = {prefix}W_ * y;
{grad_ret}
    }}

    term_hess_t {prefix}_hess(const state_t& /*x*/, double /*t*/) const
    {{
{hess_body}
    }}""")
            else:
                meth.append(f"""    double {prefix}_value(const state_t& x, const control_t& u,
                          double /*t*/) const
    {{
        const Eigen::Matrix<double, {ny}, 1> y = {yexpr};
        return 0.5 * y.dot({prefix}W_ * y);
    }}

    stage_grad_t {prefix}_grad(const state_t& x, const control_t& u,
                               double /*t*/) const
    {{
        const Eigen::Matrix<double, {ny}, 1> y = {yexpr};
        const Eigen::Matrix<double, {ny}, 1> g = {prefix}W_ * y;
{grad_ret}
    }}

    stage_hess_t {prefix}_hess(const state_t& /*x*/, const control_t& /*u*/,
                               double /*t*/) const
    {{
{hess_body}
    }}""")
        elif ct == "NONLINEAR_LS":
            ny = scope["ny"]
            W = scope["W"]
            yref = scope["yref"]
            role = "CostE" if terminal else ("Cost0" if prefix == "c0"
                                             else "Cost")
            data.append(
                f"    static constexpr double {prefix}W_data[{ny * ny}] = {{\n"
                f"{cmat(W, ny, ny)}\n    }};")
            mem.append(f"    Eigen::Matrix<double, {ny}, {ny}> {prefix}W_{{}};")
            mem.append(f"    Eigen::Matrix<double, {ny}, 1> {prefix}yref_{{}};")
            mem.append(f"    detail::{name}{role} ws_{prefix}_;")
            ctor.append(f"        {prefix}W_ = Eigen::Map<const "
                        f"Eigen::Matrix<double, {ny}, {ny}>>"
                        f"({prefix}W_data);")
            ctor.append(f"        {prefix}yref_ << {cvec(yref)};")
            if terminal:
                meth.append(f"""    double {prefix}_value(const state_t& x, double t) const
    {{
        const Eigen::Matrix<double, {ny}, 1> r = {prefix}_y(x, t) - {prefix}yref_;
        return 0.5 * r.dot({prefix}W_ * r);
    }}

    term_grad_t {prefix}_grad(const state_t& x, double t) const
    {{
        const Eigen::Matrix<double, {ny}, {nx}> J = {prefix}_jy(x, t);
        const Eigen::Matrix<double, {ny}, 1> g =
            {prefix}W_ * ({prefix}_y(x, t) - {prefix}yref_);
        return J.transpose() * g;
    }}

    term_hess_t {prefix}_hess(const state_t& x, double t) const
    {{
        const Eigen::Matrix<double, {ny}, {nx}> J = {prefix}_jy(x, t);
        return J.transpose() * {prefix}W_ * J;
    }}""")
                help_.append(f"""    Eigen::Matrix<double, {ny}, 1> {prefix}_y(const state_t& x,
                                              double t) const
    {{
        Eigen::Matrix<double, {ny}, 1> y;
        ws_{prefix}_.eval(x.data(), nullptr, nullptr, {p_cls}, t, y.data());
        return y;
    }}

    Eigen::Matrix<double, {ny}, {nx}> {prefix}_jy(const state_t& x,
                                                  double t) const
    {{
        const double h = 1.0e-6;
        Eigen::Matrix<double, {ny}, {nx}> J;
        for (int j = 0; j < {nx}; ++j)
        {{
            state_t xp = x, xm = x;
            xp(j) += h;
            xm(j) -= h;
            J.col(j) = ({prefix}_y(xp, t) - {prefix}_y(xm, t)) / (2.0 * h);
        }}
        return J;
    }}""")
            else:
                meth.append(f"""    double {prefix}_value(const state_t& x, const control_t& u,
                          double t) const
    {{
        const Eigen::Matrix<double, {ny}, 1> r =
            {prefix}_y(x, u, t) - {prefix}yref_;
        return 0.5 * r.dot({prefix}W_ * r);
    }}

    stage_grad_t {prefix}_grad(const state_t& x, const control_t& u,
                               double t) const
    {{
        const Eigen::Matrix<double, {ny}, {nx + nu}> J =
            {prefix}_jy(x, u, t);
        const Eigen::Matrix<double, {ny}, 1> g =
            {prefix}W_ * ({prefix}_y(x, u, t) - {prefix}yref_);
        stage_grad_t r;
        r.head<{nx}>() = J.topLeftCorner({ny}, {nx}).transpose() * g;
        r.tail<{nu}>() = J.bottomRightCorner({ny}, {nu}).transpose() * g;
        return r;
    }}

    stage_hess_t {prefix}_hess(const state_t& x, const control_t& u,
                               double t) const
    {{
        const Eigen::Matrix<double, {ny}, {nx + nu}> J =
            {prefix}_jy(x, u, t);
        const auto Jx = J.topLeftCorner({ny}, {nx});
        const auto Ju = J.bottomRightCorner({ny}, {nu});
        stage_hess_t H;
        H.topLeftCorner({nx}, {nx}) = Jx.transpose() * {prefix}W_ * Jx;
        H.topRightCorner({nx}, {nu}) = Jx.transpose() * {prefix}W_ * Ju;
        H.bottomLeftCorner({nu}, {nx}) = Ju.transpose() * {prefix}W_ * Jx;
        H.bottomRightCorner({nu}, {nu}) = Ju.transpose() * {prefix}W_ * Ju;
        return H;
    }}""")
                help_.append(f"""    Eigen::Matrix<double, {ny}, 1> {prefix}_y(const state_t& x,
                                              const control_t& u,
                                              double t) const
    {{
        Eigen::Matrix<double, {ny}, 1> y;
        ws_{prefix}_.eval(x.data(), u.data(), nullptr, {p_cls}, t,
                          y.data());
        return y;
    }}

    Eigen::Matrix<double, {ny}, {nx + nu}> {prefix}_jy(const state_t& x,
                                                       const control_t& u,
                                                       double t) const
    {{
        const double h = 1.0e-6;
        Eigen::Matrix<double, {ny}, {nx + nu}> J;
        for (int j = 0; j < {nx}; ++j)
        {{
            state_t xp = x, xm = x;
            xp(j) += h;
            xm(j) -= h;
            J.col(j) = ({prefix}_y(xp, u, t) - {prefix}_y(xm, u, t))
                       / (2.0 * h);
        }}
        for (int j = 0; j < {nu}; ++j)
        {{
            control_t up = u, um = u;
            up(j) += h;
            um(j) -= h;
            J.col({nx} + j) =
                ({prefix}_y(x, up, t) - {prefix}_y(x, um, t)) / (2.0 * h);
        }}
        return J;
    }}""")
        else:  # EXTERNAL
            role = "CostE" if terminal else ("Cost0" if prefix == "c0"
                                             else "Cost")
            mem.append(f"    detail::{name}{role} ws_{prefix}_;")
            if terminal:
                meth.append(f"""    double {prefix}_value(const state_t& x, double t) const
    {{
        double c = 0.0;
        ws_{prefix}_.eval(x.data(), nullptr, nullptr, {p_cls}, t, &c);
        return c;
    }}

    term_grad_t {prefix}_grad(const state_t& x, double t) const
    {{
        const double h = 1.0e-6;
        term_grad_t g;
        for (int j = 0; j < {nx}; ++j)
        {{
            state_t xp = x, xm = x;
            xp(j) += h;
            xm(j) -= h;
            g(j) = ({prefix}_value(xp, t) - {prefix}_value(xm, t))
                   / (2.0 * h);
        }}
        return g;
    }}

    term_hess_t {prefix}_hess(const state_t& x, double t) const
    {{
        const double h = 1.0e-4;
        term_hess_t H;
        for (int i = 0; i < {nx}; ++i)
            for (int j = i; j < {nx}; ++j)
            {{
                state_t xpp = x, xpm = x, xmp = x, xmm = x;
                xpp(i) += h; xpp(j) += h;
                xpm(i) += h; xpm(j) -= h;
                xmp(i) -= h; xmp(j) += h;
                xmm(i) -= h; xmm(j) -= h;
                H(i, j) = ({prefix}_value(xpp, t) - {prefix}_value(xpm, t)
                           - {prefix}_value(xmp, t)
                           + {prefix}_value(xmm, t)) / (4.0 * h * h);
                if (i != j) H(j, i) = H(i, j);
            }}
        return H;
    }}""")
            else:
                meth.append(f"""    double {prefix}_value(const state_t& x, const control_t& u,
                          double t) const
    {{
        double c = 0.0;
        ws_{prefix}_.eval(x.data(), u.data(), nullptr, {p_cls}, t, &c);
        return c;
    }}

    stage_grad_t {prefix}_grad(const state_t& x, const control_t& u,
                               double t) const
    {{
        const double h = 1.0e-6;
        stage_grad_t g;
        for (int j = 0; j < {nx}; ++j)
        {{
            state_t xp = x, xm = x;
            xp(j) += h;
            xm(j) -= h;
            g(j) = ({prefix}_value(xp, u, t) - {prefix}_value(xm, u, t))
                   / (2.0 * h);
        }}
        for (int j = 0; j < {nu}; ++j)
        {{
            control_t up = u, um = u;
            up(j) += h;
            um(j) -= h;
            g({nx} + j) =
                ({prefix}_value(x, up, t) - {prefix}_value(x, um, t))
                / (2.0 * h);
        }}
        return g;
    }}

    stage_hess_t {prefix}_hess(const state_t& x, const control_t& u,
                               double t) const
    {{
        const double h = 1.0e-4;
        stage_hess_t H;
        for (int i = 0; i < {nx}; ++i)
            for (int j = i; j < {nx}; ++j)
            {{
                state_t xpp = x, xpm = x, xmp = x, xmm = x;
                xpp(i) += h; xpp(j) += h;
                xpm(i) += h; xpm(j) -= h;
                xmp(i) -= h; xmp(j) += h;
                xmm(i) -= h; xmm(j) -= h;
                H(i, j) = ({prefix}_value(xpp, u, t)
                           - {prefix}_value(xpm, u, t)
                           - {prefix}_value(xmp, u, t)
                           + {prefix}_value(xmm, u, t)) / (4.0 * h * h);
                if (i != j) H(j, i) = H(i, j);
            }}
        for (int i = 0; i < {nx}; ++i)
            for (int j = 0; j < {nu}; ++j)
            {{
                state_t xpp = x, xpm = x, xmp = x, xmm = x;
                control_t upp = u, upm = u, ump = u, umm = u;
                xpp(i) += h; upp(j) += h;
                xpm(i) += h; upm(j) -= h;
                xmp(i) -= h; ump(j) += h;
                xmm(i) -= h; umm(j) -= h;
                H(i, {nx} + j) =
                    ({prefix}_value(xpp, upp, t) - {prefix}_value(xpm, upm, t)
                     - {prefix}_value(xmp, ump, t)
                     + {prefix}_value(xmm, umm, t)) / (4.0 * h * h);
                H({nx} + j, i) = H(i, {nx} + j);
            }}
        for (int i = 0; i < {nu}; ++i)
            for (int j = i; j < {nu}; ++j)
            {{
                state_t z = x;
                control_t upp = u, upm = u, ump = u, umm = u;
                upp(i) += h; upp(j) += h;
                upm(i) += h; upm(j) -= h;
                ump(i) -= h; ump(j) += h;
                umm(i) -= h; umm(j) -= h;
                H({nx} + i, {nx} + j) =
                    ({prefix}_value(z, upp, t) - {prefix}_value(z, upm, t)
                     - {prefix}_value(z, ump, t)
                     + {prefix}_value(z, umm, t)) / (4.0 * h * h);
                if (i != j)
                    H({nx} + j, {nx} + i) = H({nx} + i, {nx} + j);
            }}
        return H;
    }}""")
        return meth, mem, ctor, data, help_

    cp_m, cp_mem, cp_ctor, cp_data, cp_help = cost_scope_body(
        spec["cost_stage"], "cp", False)
    cost_methods.extend(cp_m)
    members.extend(cp_mem)
    ctor_lines.extend(cp_ctor)
    data_blocks.extend(cp_data)
    helpers.extend(cp_help)
    if spec["cost_stage0"] is not None:
        c0_m, c0_mem, c0_ctor, c0_data, c0_help = cost_scope_body(
            spec["cost_stage0"], "c0", False)
        cost_methods.extend(c0_m)
        members.extend(c0_mem)
        ctor_lines.extend(c0_ctor)
        data_blocks.extend(c0_data)
        helpers.extend(c0_help)
    ct_m, ct_mem, ct_ctor, ct_data, ct_help = cost_scope_body(
        spec["cost_term"], "ct", True)
    cost_methods.extend(ct_m)
    members.extend(ct_mem)
    ctor_lines.extend(ct_ctor)
    data_blocks.extend(ct_data)
    helpers.extend(ct_help)

    # ---- stage / terminal dispatch ----------------------------------------
    if spec["cost_stage0"] is not None:
        stage_dispatch = f"""    double stage_cost_value(int k, const state_t& x,
                            const control_t& u) const
    {{
        if (k == 0)
            return c0_value(x, u, 0.0);
        return cp_value(x, u, {dt_expr} * k);
    }}

    stage_grad_t stage_cost_gradient(int k, const state_t& x,
                                     const control_t& u) const
    {{
        if (k == 0)
            return c0_grad(x, u, 0.0);
        return cp_grad(x, u, {dt_expr} * k);
    }}

    stage_hess_t stage_cost_hessian(int k, const state_t& x,
                                    const control_t& u) const
    {{
        if (k == 0)
            return c0_hess(x, u, 0.0);
        return cp_hess(x, u, {dt_expr} * k);
    }}"""
    else:
        stage_dispatch = f"""    double stage_cost_value(int k, const state_t& x,
                            const control_t& u) const
    {{
        return cp_value(x, u, {dt_expr} * k);
    }}

    stage_grad_t stage_cost_gradient(int k, const state_t& x,
                                     const control_t& u) const
    {{
        return cp_grad(x, u, {dt_expr} * k);
    }}

    stage_hess_t stage_cost_hessian(int k, const state_t& x,
                                    const control_t& u) const
    {{
        return cp_hess(x, u, {dt_expr} * k);
    }}"""

    term_dispatch = f"""    double terminal_cost_value(const state_t& x) const
    {{
        return ct_value(x, {t_end});
    }}

    term_grad_t terminal_cost_gradient(const state_t& x) const
    {{
        return ct_grad(x, {t_end});
    }}

    term_hess_t terminal_cost_hessian(const state_t& x) const
    {{
        return ct_hess(x, {t_end});
    }}"""

    path = spec["cost_stage"]
    if path["type"] == "LINEAR_LS" and path["ny"] >= nx:
        cost_ref = f"""    state_t cost_reference() const
    {{
        return cpyref_.head<{nx}>();
    }}"""
    else:
        cost_ref = """    state_t cost_reference() const
    {
        state_t r;
        r.setZero();
        return r;
    }"""

    # ---- constraint methods ---------------------------------------------------
    constr = []
    h = spec["h"]
    nh = (len(h["ineq"]) + len(h["eq"])) if h else 0

    if h and (h["ineq"] or h["eq"]):
        if h["cfile"]:
            members.append(f"    detail::{name}H ws_h_;")
            helpers.append(f"""    Eigen::Matrix<double, {nh}, 1> hval(const state_t& x,
                                         const control_t& u,
                                         double t) const
    {{
        Eigen::Matrix<double, {nh}, 1> h;
        ws_h_.eval(x.data(), u.data(), nullptr, {p_cls}, t, h.data());
        return h;
    }}

    Eigen::Matrix<double, {nh}, {nx + nu}> hjac(const state_t& x,
                                                const control_t& u,
                                                double t) const
    {{
        const double h = 1.0e-6;
        Eigen::Matrix<double, {nh}, {nx + nu}> J;
        for (int j = 0; j < {nx}; ++j)
        {{
            state_t xp = x, xm = x;
            xp(j) += h;
            xm(j) -= h;
            J.col(j) = (hval(xp, u, t) - hval(xm, u, t)) / (2.0 * h);
        }}
        for (int j = 0; j < {nu}; ++j)
        {{
            control_t up = u, um = u;
            up(j) += h;
            um(j) -= h;
            J.col({nx} + j) = (hval(x, up, t) - hval(x, um, t))
                              / (2.0 * h);
        }}
        return J;
    }}""")
        if h["ineq"]:
            val_lines = "\n".join(
                (f"        g({r}) = {cnum(h['lh'][i])} - h({i});"
                 if side == "lo" else
                 f"        g({r}) = h({i}) - {cnum(h['uh'][i])};")
                for r, (i, side) in enumerate(h["ineq"]))
            jac_lines = "\n".join(
                f"        g_dx.row({r}) = "
                f"{'-' if side == 'lo' else ''}"
                f"Jh.row({i}).leftCols<{nx}>();\n"
                f"        g_du.row({r}) = "
                f"{'-' if side == 'lo' else ''}"
                f"Jh.row({i}).rightCols<{nu}>();"
                for r, (i, side) in enumerate(h["ineq"]))
            constr.append(f"""    ineq_t stage_inequality_constr(int k, const state_t& x,
                                   const control_t& u) const
    {{
        ineq_t g;
        const auto h = hval(x, u, {dt_expr} * k);
{val_lines}
        return g;
    }}

    void stage_inequality_constr_jacobian(int k, const state_t& x,
                                          const control_t& u,
                                          ineq_dg_dx_t& g_dx,
                                          ineq_dg_du_t& g_du) const
    {{
        const auto Jh = hjac(x, u, {dt_expr} * k);
{jac_lines}
    }}""")
            if any(w is not None for w in h.get("ineq_soft_w", [])):
                constr.append(
                    f"    ineq_pen_t stage_inequality_constr_soft_penalty"
                    f"(int /*k*/) const\n"
                    f"    {{\n"
                    f"        ineq_pen_t w;\n"
                    f"        w << {wvec(h['ineq_soft_w'])};\n"
                    f"        return w;\n"
                    f"    }}\n")
        if h["eq"]:
            val_lines = "\n".join(
                f"        e({r}) = h({i}) - {cnum(h['lh'][i])};"
                for r, i in enumerate(h["eq"]))
            jac_lines = "\n".join(
                f"        e_dx.row({r}) = Jh.row({i}).leftCols<{nx}>();\n"
                f"        e_du.row({r}) = Jh.row({i}).rightCols<{nu}>();"
                for r, i in enumerate(h["eq"]))
            constr.append(f"""    eq_t stage_equality_constr(int k, const state_t& x,
                               const control_t& u) const
    {{
        eq_t e;
        const auto h = hval(x, u, {dt_expr} * k);
{val_lines}
        return e;
    }}

    void stage_equality_constr_jacobian(int k, const state_t& x,
                                        const control_t& u,
                                        eq_de_dx_t& e_dx,
                                        eq_de_du_t& e_du) const
    {{
        const auto Jh = hjac(x, u, {dt_expr} * k);
{jac_lines}
    }}""")
            if any(w is not None for w in h.get("eq_soft_w", [])):
                constr.append(
                    f"    eq_pen_t stage_equality_constr_soft_penalty"
                    f"(int /*k*/) const\n"
                    f"    {{\n"
                    f"        eq_pen_t w;\n"
                    f"        w << {wvec(h['eq_soft_w'])};\n"
                    f"        return w;\n"
                    f"    }}\n")

    if spec["lin"]:
        lin = spec["lin"]
        data_blocks.append(
            f"    static constexpr double linA_data[{nl * nx}] = {{\n"
            f"{cmat(lin['A'], nl, nx)}\n    }};")
        members.append(f"    Eigen::Matrix<double, {nl}, {nx}> linA_{{}};")
        ctor_lines.append(f"        linA_ = Eigen::Map<const "
                          f"Eigen::Matrix<double, {nl}, {nx}>>"
                          f"(linA_data);")
        if nu > 0:
            data_blocks.append(
                f"    static constexpr double linB_data[{nl * nu}] = {{\n"
                f"{cmat(lin['B'], nl, nu)}\n    }};")
            members.append(
                f"    Eigen::Matrix<double, {nl}, {nu}> linB_{{}};")
            ctor_lines.append(f"        linB_ = Eigen::Map<const "
                              f"Eigen::Matrix<double, {nl}, {nu}>>"
                              f"(linB_data);")
        soft_vals = ", ".join(
            cnum(spec["lin_soft"][r]) if spec["lin_soft"][r] is not None
            else "0.0" for r in range(nl))
        constr.append(f"""    stage_linear_t stage_linear_constr(int /*k*/) const
    {{
        stage_linear_t spec;
        spec.A = linA_;
{('        spec.B = linB_;\n' if nu > 0 else '')}        spec.bounds.lo << {cvec(lin['lo'])};
        spec.bounds.hi << {cvec(lin['hi'])};
        spec.bounds.soft_penalty << {soft_vals};
        return spec;
    }}""")

    h_t = spec["h_t"]
    nh_t = (len(h_t["ineq"]) + len(h_t["eq"])) if h_t else 0
    if h_t and (h_t["ineq"] or h_t["eq"]):
        if h_t["cfile"]:
            members.append(f"    detail::{name}HE ws_he_;")
            helpers.append(f"""    Eigen::Matrix<double, {nh_t}, 1> htval(const state_t& x) const
    {{
        Eigen::Matrix<double, {nh_t}, 1> h;
        ws_he_.eval(x.data(), nullptr, nullptr, {p_cls}, {t_end},
                    h.data());
        return h;
    }}

    Eigen::Matrix<double, {nh_t}, {nx}> htjac(const state_t& x) const
    {{
        const double h = 1.0e-6;
        Eigen::Matrix<double, {nh_t}, {nx}> J;
        for (int j = 0; j < {nx}; ++j)
        {{
            state_t xp = x, xm = x;
            xp(j) += h;
            xm(j) -= h;
            J.col(j) = (htval(xp) - htval(xm)) / (2.0 * h);
        }}
        return J;
    }}""")
        if h_t["ineq"]:
            val_lines = "\n".join(
                (f"        g({r}) = {cnum(h_t['lh'][i])} - h({i});"
                 if side == "lo" else
                 f"        g({r}) = h({i}) - {cnum(h_t['uh'][i])};")
                for r, (i, side) in enumerate(h_t["ineq"]))
            jac_lines = "\n".join(
                f"        g_dx.row({r}) = "
                f"{'-' if side == 'lo' else ''}"
                f"Jh.row({i}).leftCols<{nx}>();"
                for r, (i, side) in enumerate(h_t["ineq"]))
            constr.append(f"""    ineq_term_t terminal_inequality_constr(
        const state_t& x) const
    {{
        ineq_term_t g;
        const auto h = htval(x);
{val_lines}
        return g;
    }}

    void terminal_inequality_constr_jacobian(const state_t& x,
                                             ineq_term_dg_dx_t& g_dx) const
    {{
        const auto Jh = htjac(x);
{jac_lines}
    }}""")
            if any(w is not None for w in h_t.get("ineq_soft_w", [])):
                constr.append(
                    f"    ineq_term_pen_t "
                    f"terminal_inequality_constr_soft_penalty() const\n"
                    f"    {{\n"
                    f"        ineq_term_pen_t w;\n"
                    f"        w << {wvec(h_t['ineq_soft_w'])};\n"
                    f"        return w;\n"
                    f"    }}\n")
        if h_t["eq"]:
            val_lines = "\n".join(
                f"        e({r}) = h({i}) - {cnum(h_t['lh'][i])};"
                for r, i in enumerate(h_t["eq"]))
            jac_lines = "\n".join(
                f"        e_dx.row({r}) = Jh.row({i}).leftCols<{nx}>().eval();"
                for r, i in enumerate(h_t["eq"]))
            constr.append(f"""    eq_term_t terminal_equality_constr(
        const state_t& x) const
    {{
        eq_term_t e;
        const auto h = htval(x);
{val_lines}
        return e;
    }}

    void terminal_equality_constr_jacobian(const state_t& x,
                                           eq_term_de_dx_t& e_dx) const
    {{
        const auto Jh = htjac(x);
{jac_lines}
    }}""")
            if any(w is not None for w in h_t.get("eq_soft_w", [])):
                constr.append(
                    f"    eq_term_pen_t "
                    f"terminal_equality_constr_soft_penalty() const\n"
                    f"    {{\n"
                    f"        eq_term_pen_t w;\n"
                    f"        w << {wvec(h_t['eq_soft_w'])};\n"
                    f"        return w;\n"
                    f"    }}\n")

    if spec["lin_t"]:
        lt = spec["lin_t"]
        data_blocks.append(
            f"    static constexpr double lin_tA_data[{nl_t * nx}] = {{\n"
            f"{cmat(lt['A'], nl_t, nx)}\n    }};")
        members.append(
            f"    Eigen::Matrix<double, {nl_t}, {nx}> lin_tA_{{}};")
        ctor_lines.append(f"        lin_tA_ = Eigen::Map<const "
                          f"Eigen::Matrix<double, {nl_t}, {nx}>>"
                          f"(lin_tA_data);")
        soft_vals = ", ".join(
            cnum(spec["terminal_lin_soft"][r])
            if spec["terminal_lin_soft"][r] is not None
            else "0.0" for r in range(nl_t))
        constr.append(f"""    term_linear_t terminal_linear_constr() const
    {{
        term_linear_t spec;
        spec.A = lin_tA_;
        spec.bounds.lo << {cvec(lt['lo'])};
        spec.bounds.hi << {cvec(lt['hi'])};
        spec.bounds.soft_penalty << {soft_vals};
        return spec;
    }}""")

    # ---- box methods -----------------------------------------------------------
    union = spec["state_box_union"]
    if union:
        k0_assign = "\n".join(
            f"            spec.lo({union.index(r)}) = {cnum(l)};\n"
            f"            spec.hi({union.index(r)}) = {cnum(u)};"
            for r, (l, u) in sorted(spec["state_box_stage0"].items()))
        kp_assign = "\n".join(
            f"            spec.lo({union.index(r)}) = {cnum(l)};\n"
            f"            spec.hi({union.index(r)}) = {cnum(u)};"
            for r, (l, u) in sorted(spec["state_box_path"].items()))
        box_soft = ", ".join(
            cnum(v) if v is not None else "0.0" for v in
            spec["state_box_soft"])
        if fixed:
            box_method = f"""    state_box_t stage_state_box_constr(int /*k*/) const
    {{
        state_box_t spec;
        spec.lo.setConstant(-{INF});
        spec.hi.setConstant({INF});
{kp_assign}
        spec.soft_penalty << {box_soft};
        return spec;
    }}"""
        else:
            if k0_assign and kp_assign:
                k_block = f"""        if (k == 0)
        {{
{k0_assign}
        }}
        else
        {{
{kp_assign}
        }}"""
            elif k0_assign:
                k_block = f"""        if (k == 0)
        {{
{k0_assign}
        }}"""
            else:
                k_block = kp_assign
            box_method = f"""    state_box_t stage_state_box_constr(int k) const
    {{
        state_box_t spec;
        spec.lo.setConstant(-{INF});
        spec.hi.setConstant({INF});
{k_block}
        spec.soft_penalty << {box_soft};
        return spec;
    }}"""
        constr.append(box_method)

    if spec["control_box"]:
        cb_lo = "\n".join(
            f"        spec.lo({i}) = {cnum(l)};"
            for i, (_, l, u) in enumerate(spec["control_box"]))
        cb_hi = "\n".join(
            f"        spec.hi({i}) = {cnum(u)};"
            for i, (_, l, u) in enumerate(spec["control_box"]))
        cb_soft = ", ".join(
            cnum(v) if v is not None else "0.0"
            for v in spec["control_box_soft"])
        constr.append(f"""    control_box_t stage_control_box_constr(int /*k*/) const
    {{
        control_box_t spec;
{cb_lo}
{cb_hi}
        spec.soft_penalty << {cb_soft};
        return spec;
    }}""")

    if spec["terminal_state_box"]:
        tb_lo = "\n".join(
            f"        spec.lo({i}) = {cnum(l)};"
            for i, (_, l, u) in enumerate(spec["terminal_state_box"]))
        tb_hi = "\n".join(
            f"        spec.hi({i}) = {cnum(u)};"
            for i, (_, l, u) in enumerate(spec["terminal_state_box"]))
        tb_soft = ", ".join(
            cnum(v) if v is not None else "0.0"
            for v in spec["terminal_state_box_soft"])
        constr.append(f"""    term_state_box_t terminal_state_box_constr() const
    {{
        term_state_box_t spec;
{tb_lo}
{tb_hi}
        spec.soft_penalty << {tb_soft};
        return spec;
    }}""")

    # ---- assemble ----------------------------------------------------------------
    if not ctor_lines:
        ctor_lines.append("        (void)0;")
    ctor_body = "\n".join(ctor_lines)
    init_state = (f"""    state_t initial_state() const
    {{
        return x0_;
    }}
""" if fixed else "")

    class_block = f"""
class {name}
    : public ocp::ContinuousProblem<{name}Dims, {name}Ode, {name}Integ>
{{
    using Base =
        ocp::ContinuousProblem<{name}Dims, {name}Ode, {name}Integ>;

public:
    {name}() : Base({name}Ode{{}}, {dt})
    {{
{ctor_body}
    }}

{init_state}    static constexpr double {INF} =
        std::numeric_limits<double>::infinity();

{cost_ref}

{stage_dispatch}

{term_dispatch}

{chr(10).join(constr)}
{chr(10).join(cost_methods)}

private:
{chr(10).join(data_blocks)}

{chr(10).join(helpers)}

{chr(10).join(members)}
}};
"""

    return f"""// {name}.hpp
// GENERATED by tools/acados2ocp_pp.py from {os.path.basename(json_path)} (hash {spec['hash']}).
// Do not edit by hand; regenerate instead.
//
// ocp++ ConcreteProblem for the acados OCP "{spec['problem_name']}".
//   state  nx = {nx}, control nu = {nu}, horizon N = {spec['N']}, dt = {spec['dt']}
//   integrator: {spec['integ_kind']} {spec['gl_tag']} ({nst} stages x {nstep} sub-steps)
//   Hessian model: Gauss-Newton (cost Hessian only; dynamics enter the QP
//                  via the linearized BA matrix, see has_dynamics_hess_prod);
//                  acados uses hessian_approx = {spec['hessian_approx']}.

#pragma once

#include "ocp/integrators/continuous_problem.hpp"
#include "ocp/integrators/{integ_inc}.hpp"

#include "{name}_capi.hpp"

#include <Eigen/Dense>
#include <limits>

namespace ocp
{{
{dims}{ode}{integ}{class_block}
}}  // namespace ocp
"""


def emit_main(spec, name, json_path):
    N = spec["N"]
    nx = spec["nx"]
    nu = spec["nu"]
    fixed = spec["fixed_initial_state"]
    if fixed:
        x0_assign = "    sol.x[0] = problem.initial_state();"
    else:
        # warm-start x0 from the midpoint of the finite stage-0 box bounds
        vals = []
        for r in spec["state_box_union"]:
            if r in spec["state_box_stage0"]:
                l, u = spec["state_box_stage0"][r]
                if math.isfinite(l) and math.isfinite(u):
                    vals.append((l + u) / 2)
                elif math.isfinite(l):
                    vals.append(l)
                elif math.isfinite(u):
                    vals.append(u)
                else:
                    vals.append(0.0)
            else:
                vals.append(0.0)
        x0_assign = f"    sol.x[0] << {cvec(vals)};"
    return f"""// {name.lower()}_main.cpp
// GENERATED by tools/acados2ocp_pp.py from {os.path.basename(json_path)}.
// Driver: warm-start, solve, dump ocp_pp.csv in the same layout as the acados
// reference (ref_acados.csv) for comparison.

#include <cstdio>

#include "ocp/solvers/acados/sqp.hpp"
#include "{name}.hpp"

using namespace ocp;

int main()
{{
    {name} problem;

    SqpOptions opts;
    opts.print_level = 1;
    opts.max_iter = {spec['max_iter']};
    opts.tol_stat = {cnum(spec['tol_stat'])};
    opts.tol_eq = {cnum(spec['tol_eq'])};
    opts.tol_ineq = {cnum(spec['tol_ineq'])};
    opts.tol_comp = {cnum(spec['tol_comp'])};

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
                std::fprintf(f, ",%.15g",
                             static_cast<double>(sol.u[k](i)));
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
    files = [csrc] if csrc else spec["c_files"]
    file_lines = "\n".join(f'    "{f}"' for f in files)
    return f"""# GENERATED by tools/acados2ocp_pp.py -- add to CMakeLists.txt
# (enable_language(C) must already be in effect for the C value functions.)
set({name}_C
{file_lines})
set_source_files_properties(${{{name}_C}} PROPERTIES
  COMPILE_OPTIONS "-O2;-w" LANGUAGE C)
add_executable({name.lower()}_gen
               {outdir}/{name.lower()}_main.cpp ${{{name}_C}})
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
                    help="also write a CMake target snippet to "
                         "<out>/CMake_snippet.txt")
    ap.add_argument("--csrc", default=None,
                    help="path to a single C source for the CMake snippet "
                         "(defaults to the C files listed in the JSON)")
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

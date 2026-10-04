"""Pendulum-on-cart OCP: reproducible acados codegen for the ocp++ port.

This mirrors the acados python example `pendulum_on_cart/ocp/minimal_example_ocp_cmake.py`
(LINEAR_LS cost, IRK / Gauss-Legendre 4 stages, control box, fixed initial
state). It reuses the acados repo's `pendulum_model.export_pendulum_ode_model`
so the generated `acados_ocp_nlp.json` + C code is byte-for-byte the same as
what the acados Python interface produces.

The generated JSON is the input to tools/acados2ocp_pp.py, which emits the
ocp++ wrapper (PendulumGen.hpp + _capi.hpp + driver).

Run from the repo root (needs casadi <= 3.8.1 + the acados python interface on
PYTHONPATH, plus ACADOS_SOURCE_DIR / ACADOS_INSTALL_DIR):

    PYTHONPATH=/home/jgeisler/repos/acados/interfaces/acados_template \
    ACADOS_SOURCE_DIR=/home/jgeisler/repos/acados \
    ACADOS_INSTALL_DIR=/home/jgeisler/repos/acados \
    python3 examples/pendulum_on_cart/example_ocp.py
"""

import os

import numpy as np
import scipy.linalg

from acados_template import (AcadosOcp, AcadosOcpSolver,
                             ocp_get_default_cmake_builder)
from pendulum_model import export_pendulum_ode_model

HERE = os.path.dirname(os.path.abspath(__file__))

FMAX = 80
T_HORIZON = 1.0
N = 20


def create_ocp() -> AcadosOcp:
    ocp = AcadosOcp()
    model = export_pendulum_ode_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = T_HORIZON

    # cost: LINEAR_LS (stage + terminal)
    Q = 2 * np.diag([1e3, 1e3, 1e-2, 1e-2])
    R = 2 * np.diag([1e-2])
    ocp.cost.W_e = Q
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[4, 0] = 1.0
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # constraints: control box + fixed initial state
    ocp.constraints.lbu = np.array([-FMAX])
    ocp.constraints.ubu = np.array([+FMAX])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

    # solver: SQP + partial condensing (HPIPM), Gauss-Newton, IRK
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"

    ocp.code_gen_options.code_export_directory = os.path.join(HERE, "acados_codegen")
    ocp.code_gen_options.json_file = "acados_ocp_nlp.json"
    return ocp


def main():
    ocp = create_ocp()
    builder = ocp_get_default_cmake_builder()
    builder.options_on = ["BUILD_EXAMPLE"]
    AcadosOcpSolver(ocp, cmake_builder=builder, verbose=False)
    print("codegen + CMake build written to",
          ocp.code_gen_options.code_export_directory)


if __name__ == "__main__":
    main()

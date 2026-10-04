"""Unicycle OCP: reproducible acados codegen for the ocp++ port.

Mirrors the acados python example `unicycle/main.py` (LINEAR_LS cost,
IRK / Gauss-Legendre 2 stages, control box on F, fixed initial state).
Batch (open-loop) OCP: drive the unicycle from x0 = 0 to the target
(x, y) = (1, 1) by T = 2.0 s; the reference (1, 1, 0, 0, 0, 0, 0) is the
same yref the acados closed-loop example uses at every iteration.

It reuses the acados repo's `robot_model.export_robot_model` so the
generated `acados_ocp_nlp.json` + C code is byte-for-byte the same as what
the acados Python interface produces.

The generated JSON is the input to tools/acados2ocp_pp.py, which emits the
ocp++ wrapper (UnicycleGen.hpp + _capi.hpp + driver).

Run from the repo root (needs casadi <= 3.8.1 + the acados python interface on
PYTHONPATH, plus ACADOS_SOURCE_DIR / ACADOS_INSTALL_DIR):

    PYTHONPATH=/home/jgeisler/repos/acados/interfaces/acados_template \
              /home/jgeisler/repos/acados/examples/acados_python/unicycle \
    ACADOS_SOURCE_DIR=/home/jgeisler/repos/acados \
    ACADOS_INSTALL_DIR=/home/jgeisler/repos/acados \
    python3 examples/unicycle/example_ocp.py
"""

import os

import numpy as np
import scipy.linalg

from acados_template import (AcadosOcp, AcadosOcpSolver,
                             ocp_get_default_cmake_builder)
from robot_model import export_robot_model

HERE = os.path.dirname(os.path.abspath(__file__))

F_MAX = 10
T_HORIZON = 2.0
N = 50

# target (x, y) = (1, 1); same reference as the acados closed-loop example
YREF = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
YREF_E = np.array([1.0, 1.0, 0.0, 0.0, 0.0])


def create_ocp() -> AcadosOcp:
    ocp = AcadosOcp()
    model = export_robot_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = T_HORIZON

    # cost: LINEAR_LS (stage + terminal)
    Q = 2 * np.diag([1e3, 1e3, 1e-4, 1e-3, 1e-3])
    R = 2 * 5 * np.diag([1e-1, 1e-2])
    ocp.cost.W_e = Q
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx:nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref = YREF
    ocp.cost.yref_e = YREF_E

    # constraints: control box on F + fixed initial state
    ocp.constraints.lbu = np.array([-F_MAX])
    ocp.constraints.ubu = np.array([+F_MAX])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.x0 = np.zeros(nx)

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

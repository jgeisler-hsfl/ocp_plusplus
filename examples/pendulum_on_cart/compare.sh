#!/usr/bin/env bash
# compare.sh -- build both solvers, regenerate their solution CSVs, and report
# the cross-solver differences (cost, state/control gap) for pendulum_on_cart.
#
# Manual companion to tests/sqp_pendulum (which embeds the reference and is
# CI-runnable).  Run from the repo root:
#   examples/pendulum_on_cart/compare.sh
#
# Tolerances are loose on purpose: the OCP is non-convex with a bang-bang
# optimal control, and ocp++ uses a cost-only QP Hessian whereas acados uses
# Gauss-Newton, so the two SQP paths settle at slightly different local KKT
# points.  A sign-convention / dynamics port bug, by contrast, would show up
# as a large gap (max|dx| >> 0.25, wrong cost) and is what the check guards.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
cd "${HERE}"

# acados runtime libraries (the codegen links against a prebuilt acados).
ACADOS_LIB="${ACADOS_LIB:-/home/jgeisler/repos/acados/lib}"
export LD_LIBRARY_PATH="${ACADOS_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "==> acados reference"
cmake -S acados_codegen -B acados_codegen/build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLE=ON >/dev/null
cmake --build acados_codegen/build --target main_ocp_pendulum_c662f455 >/dev/null
( cd acados_codegen && ./build/main_ocp_pendulum_c662f455 ref_acados.csv >/dev/null )

echo "==> ocp++"
cmake -S "${ROOT}" -B "${ROOT}/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${ROOT}/build" --target pendulum_on_cart >/dev/null
( "${ROOT}/build/pendulum_on_cart" >/dev/null ) \
    || echo "  (ocp++ example exited nonzero; CSV still written)"

echo "==> comparison"
python3 - "${HERE}/acados_codegen/ref_acados.csv" "${HERE}/ocp_pp.csv" <<'PY'
import re, sys

def load(p):
    xs, us, cost = {}, {}, None
    for line in open(p):
        line = line.rstrip()
        if line.startswith('#'):
            m = re.search(r'cost=([0-9.eE+-]+)', line)
            if m:
                cost = float(m.group(1))
            continue
        if line.startswith('stage'):
            continue
        f = line.split(',')
        k = int(f[0])
        xs[k] = [float(v) for v in f[1:5]]
        if len(f) >= 6:
            us[k] = [float(v) for v in f[5:6]]
    return xs, us, cost

xa, ua, ca = load(sys.argv[1])
xo, uo, co = load(sys.argv[2])
N = max(xa)
dx = max(max(abs(a - b) for a, b in zip(xa[k], xo[k])) for k in range(N + 1))
du = max(max(abs(a - b) for a, b in zip(ua[k], uo[k])) for k in range(N))
rel = abs(ca - co) / abs(ca)
print(f"acados cost : {ca:.6f}")
print(f"ocp++  cost : {co:.6f}")
print(f"cost rel diff : {rel:.3e}")
print(f"max |dx|      : {dx:.3e}")
print(f"max |du|      : {du:.3e}")
ok = rel < 1e-3 and dx < 0.25 and du < 1.5
print("RESULT:", "PASS" if ok else "CHECK")
PY

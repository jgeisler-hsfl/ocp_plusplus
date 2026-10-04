#!/usr/bin/env bash
# compare.sh -- build both solvers, regenerate their solution CSVs, and report
# the cross-solver differences (cost, state/control gap) for masses_chain.
#
# This is the manual companion to tests/sqp_masses_chain (which embeds the
# reference and is CI-runnable).  Run from the repo root:
#   examples/masses_chain/compare.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
cd "${HERE}"

# acados runtime libraries (the codegen links against a prebuilt acados).
ACADOS_LIB="${ACADOS_LIB:-/home/jgeisler/repos/acados/lib}"
export LD_LIBRARY_PATH="${ACADOS_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "==> acados reference"
cmake -S acados_codegen -B acados_codegen/build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build acados_codegen/build --target main_ocp_masses_chain_21ef639b >/dev/null
( cd acados_codegen && ./build/main_ocp_masses_chain_21ef639b ref_acados.csv >/dev/null )

echo "==> ocp++"
cmake -S "${ROOT}" -B "${ROOT}/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${ROOT}/build" --target masses_chain >/dev/null
( "${ROOT}/build/masses_chain" >/dev/null )

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
        xs[k] = [float(v) for v in f[1:25]]
        if len(f) >= 28:
            us[k] = [float(v) for v in f[25:28]]
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
ok = rel < 1e-4 and dx < 2e-3 and du < 1.5e-2
print("RESULT:", "PASS" if ok else "CHECK")
PY

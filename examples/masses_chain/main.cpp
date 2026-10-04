// main.cpp
//
// Phase-1 hand-written ocp++ port of the acados masses_chain OCP. Solves the
// problem with the default SqpSolver (HPIPM QP + GLM regularizer + merit
// backtracking) and writes ocp_pp.csv in the same layout as the acados
// reference (ref_acados.csv) for side-by-side comparison.
//
// Warm start mirrors the MATLAB example: x[0] = fixed initial state (hanging),
// x[1..N] = x_ref (spread configuration), u = 0.

#include <cmath>
#include <cstdio>

#include "ocp/solvers/acados/sqp.hpp"

#include "masses_chain.hpp"

using namespace ocp;

constexpr int N = 40;  // T / dt = 8 / 0.2

Solution<MassesChain> warm_start(const MassesChain& p)
{
    Solution<MassesChain> sol(N);
    sol.x[0] = p.initial_state();
    // x[1..N] and u: seed with the reference configuration (acados init_x /
    // init_u); x[0] stays the fixed initial state.
    for (int k = 0; k < N; ++k)
    {
        sol.u[k].setZero();
        sol.x[k + 1] = p.cost_reference();
    }
    return sol;
}

int main()
{
    MassesChain problem;

    SqpOptions opts;
    opts.print_level = 1;
    // Mirror the acados codegen settings (acados_ocp_nlp.json,
    // nlp_solver_tol_* = 1e-6): with the default 1e-8 tolerances the SQP
    // stalls in kMinStep just short of kSolved on this weakly-convex OCP.
    opts.tol_stat = 1e-6;
    opts.tol_eq = 1e-6;
    opts.tol_ineq = 1e-6;
    opts.tol_comp = 1e-6;

    SqpSolver<MassesChain> solver(opts);
    Solution<MassesChain> sol = warm_start(problem);

    const Status st = solver.solve(problem, sol);

    std::printf("ocp++ masses_chain: status=%d  cost=%.12f  (%d SQP iters)\n",
                static_cast<int>(sol.status), sol.cost_value,
                solver.statistics().iter);
    if (st != Status::kSolved)
    {
        std::printf("ocp++ solve did not converge (status=%d)\n",
                    static_cast<int>(st));
    }

    // CSV dump, same layout as ref_acados.csv
    const char* csv_name = "ocp_pp.csv";
    std::FILE* f = std::fopen(csv_name, "w");
    if (f == nullptr)
    {
        std::printf("FAIL: could not open %s\n", csv_name);
        return 1;
    }
    const bool solved = (sol.status == Status::kSolved);
    std::fprintf(f, "# masses_chain ocp_pp\n");
    std::fprintf(f, "# nx=%d nu=%d N=%d\n", MassesChain::nx, MassesChain::nu, N);
    std::fprintf(f, "# status=%d cost=%.15g\n",
                 solved ? 0 : static_cast<int>(sol.status), sol.cost_value);
    std::fprintf(f, "stage");
    for (int i = 0; i < MassesChain::nx; ++i)
    {
        std::fprintf(f, ",x%d", i);
    }
    for (int i = 0; i < MassesChain::nu; ++i)
    {
        std::fprintf(f, ",u%d", i);
    }
    std::fprintf(f, "\n");
    for (int k = 0; k <= N; ++k)
    {
        std::fprintf(f, "%d", k);
        for (int i = 0; i < MassesChain::nx; ++i)
        {
            std::fprintf(f, ",%.15g", static_cast<double>(sol.x[k](i)));
        }
        if (k < N)
        {
            for (int i = 0; i < MassesChain::nu; ++i)
            {
                std::fprintf(f, ",%.15g", static_cast<double>(sol.u[k](i)));
            }
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    std::printf("wrote %s\n", csv_name);

    return solved ? 0 : 1;
}

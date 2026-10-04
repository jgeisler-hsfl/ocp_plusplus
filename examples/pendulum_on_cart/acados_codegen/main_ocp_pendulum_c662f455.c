/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_ocp_pendulum_c662f455.h"

// blasfeo
#include "blasfeo_d_aux_ext_dep.h"

#define NX     OCP_PENDULUM_C662F455_NX
#define NP     OCP_PENDULUM_C662F455_NP
#define NU     OCP_PENDULUM_C662F455_NU
#define NBX0   OCP_PENDULUM_C662F455_NBX0
#define NP_GLOBAL   OCP_PENDULUM_C662F455_NP_GLOBAL


int main(int argc, char* argv[])
{

    ocp_pendulum_c662f455_solver_capsule *acados_ocp_capsule = ocp_pendulum_c662f455_acados_create_capsule();
    // there is an opportunity to change the number of shooting intervals in C without new code generation
    int N = OCP_PENDULUM_C662F455_N;
    // allocate the array and fill it accordingly
    double* new_time_steps = NULL;
    int status = ocp_pendulum_c662f455_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);

    if (status)
    {
        printf("ocp_pendulum_c662f455_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    ocp_nlp_config *nlp_config = ocp_pendulum_c662f455_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = ocp_pendulum_c662f455_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = ocp_pendulum_c662f455_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = ocp_pendulum_c662f455_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = ocp_pendulum_c662f455_acados_get_nlp_solver(acados_ocp_capsule);
    void *nlp_opts = ocp_pendulum_c662f455_acados_get_nlp_opts(acados_ocp_capsule);
    // initial condition
    double lbx0[NBX0];
    double ubx0[NBX0];
    lbx0[0] = 0;
    ubx0[0] = 0;
    lbx0[1] = 3.141592653589793;
    ubx0[1] = 3.141592653589793;
    lbx0[2] = 0;
    ubx0[2] = 0;
    lbx0[3] = 0;
    ubx0[3] = 0;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);

    // initialization for state values
    double x_init[NX];
    x_init[0] = 0.0;
    x_init[1] = 0.0;
    x_init[2] = 0.0;
    x_init[3] = 0.0;

    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;

    // prepare evaluation
    int NTIMINGS = 1;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    double xtraj[NX * (N+1)];
    double utraj[NU * N];

    // solve ocp in loop
    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        // initialize solution
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
        }
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x_init);
        status = ocp_pendulum_c662f455_acados_solve(acados_ocp_capsule);
        ocp_nlp_get(nlp_solver, "time_tot", &elapsed_time);
        min_time = MIN(elapsed_time, min_time);
    }

    /* print solution and statistics */
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &xtraj[ii*NX]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &utraj[ii*NU]);

    /* --- CSV dump for cross-solver comparison (ref side) -------------------
     * Layout: one header block, then one row per stage k=0..N with x (NX) and
     * u (NU, blank for k=N). Cost is recomputed from x/u so the file is
     * self-contained. Path via argv[1] (default "ref_acados.csv"). */
    {
        const char* csv_path = (argc > 1) ? argv[1] : "ref_acados.csv";
        FILE* f = fopen(csv_path, "w");
        if (f)
        {
            // LINEAR_LS, yref = 0:  stage L_k = 0.5 * ||[x; u]||^2_W with
            // diag(W) = (2e3, 2e3, 2e-2, 2e-2, 2e-2); terminal 0.5 * ||x||^2_We
            // with diag(We) = (2e3, 2e3, 2e-2, 2e-2).
            double cost = 0.0;
            for (int k = 0; k < N; k++)
            {
                double c = 2.0e3 * xtraj[k*NX+0] * xtraj[k*NX+0]
                          + 2.0e3 * xtraj[k*NX+1] * xtraj[k*NX+1]
                          + 2.0e-2 * xtraj[k*NX+2] * xtraj[k*NX+2]
                          + 2.0e-2 * xtraj[k*NX+3] * xtraj[k*NX+3]
                          + 2.0e-2 * utraj[k*NU+0] * utraj[k*NU+0];
                cost += 0.5 * c;
            }
            double ce = 2.0e3 * xtraj[N*NX+0] * xtraj[N*NX+0]
                      + 2.0e3 * xtraj[N*NX+1] * xtraj[N*NX+1]
                      + 2.0e-2 * xtraj[N*NX+2] * xtraj[N*NX+2]
                      + 2.0e-2 * xtraj[N*NX+3] * xtraj[N*NX+3];
            cost += 0.5 * ce;

            fprintf(f, "# pendulum_on_cart acados reference\n");
            fprintf(f, "# nx=%d nu=%d N=%d\n", NX, NU, N);
            fprintf(f, "# status=%d cost=%.15g\n", status, cost);
            fprintf(f, "stage");
            for (int i = 0; i < NX; i++) fprintf(f, ",x%d", i);
            for (int i = 0; i < NU; i++) fprintf(f, ",u%d", i);
            fprintf(f, "\n");
            for (int k = 0; k <= N; k++)
            {
                fprintf(f, "%d", k);
                for (int i = 0; i < NX; i++) fprintf(f, ",%.15g", xtraj[k*NX+i]);
                if (k < N)
                    for (int i = 0; i < NU; i++)
                        fprintf(f, ",%.15g", utraj[k*NU+i]);
                fprintf(f, "\n");
            }
            fclose(f);
            printf("wrote %s (cost=%.12g, status=%d)\n", csv_path, cost, status);
        }
        else
        {
            printf("failed to open %s for writing\n", csv_path);
        }
    }

    printf("\n--- xtraj ---\n");
    d_print_exp_tran_mat( NX, N+1, xtraj, NX);
    printf("\n--- utraj ---\n");
    d_print_exp_tran_mat( NU, N, utraj, NU );
    // print_ocp_nlp_out(nlp_solver->dims, nlp_out);

    printf("\nsolved ocp %d times, solution printed above\n\n", NTIMINGS);

    if (status == ACADOS_SUCCESS)
    {
        printf("ocp_pendulum_c662f455_acados_solve(): SUCCESS!\n");
    }
    else
    {
        printf("ocp_pendulum_c662f455_acados_solve() failed with status %d.\n", status);
    }

    // get solution
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_solver, "sqp_iter", &sqp_iter);

    ocp_pendulum_c662f455_acados_print_stats(acados_ocp_capsule);

    printf("\nSolver info:\n");
    printf(" SQP iterations %2d\n minimum time for %d solve %f [ms]\n KKT %e\n",
           sqp_iter, NTIMINGS, min_time*1000, kkt_norm_inf);



    // free solver
    status = ocp_pendulum_c662f455_acados_free(acados_ocp_capsule);
    if (status) {
        printf("ocp_pendulum_c662f455_acados_free() returned status %d. \n", status);
    }
    // free solver capsule
    status = ocp_pendulum_c662f455_acados_free_capsule(acados_ocp_capsule);
    if (status) {
        printf("ocp_pendulum_c662f455_acados_free_capsule() returned status %d. \n", status);
    }

    return status;
}

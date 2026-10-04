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
#include "acados_solver_ocp_masses_chain_21ef639b.h"

// blasfeo
#include "blasfeo_d_aux_ext_dep.h"

#define NX     OCP_MASSES_CHAIN_21EF639B_NX
#define NP     OCP_MASSES_CHAIN_21EF639B_NP
#define NU     OCP_MASSES_CHAIN_21EF639B_NU
#define NBX0   OCP_MASSES_CHAIN_21EF639B_NBX0
#define NP_GLOBAL   OCP_MASSES_CHAIN_21EF639B_NP_GLOBAL


int main(int argc, char* argv[])
{

    ocp_masses_chain_21ef639b_solver_capsule *acados_ocp_capsule = ocp_masses_chain_21ef639b_acados_create_capsule();
    // there is an opportunity to change the number of shooting intervals in C without new code generation
    int N = OCP_MASSES_CHAIN_21EF639B_N;
    // allocate the array and fill it accordingly
    double* new_time_steps = NULL;
    int status = ocp_masses_chain_21ef639b_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);

    if (status)
    {
        printf("ocp_masses_chain_21ef639b_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    ocp_nlp_config *nlp_config = ocp_masses_chain_21ef639b_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = ocp_masses_chain_21ef639b_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = ocp_masses_chain_21ef639b_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = ocp_masses_chain_21ef639b_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = ocp_masses_chain_21ef639b_acados_get_nlp_solver(acados_ocp_capsule);
    void *nlp_opts = ocp_masses_chain_21ef639b_acados_get_nlp_opts(acados_ocp_capsule);
    // initial condition
    double lbx0[NBX0];
    double ubx0[NBX0];
    lbx0[0] = -0.000000000000000000000136855384628859;
    ubx0[0] = -0.000000000000000000000136855384628859;
    lbx0[1] = 0.3740500601637552;
    ubx0[1] = 0.3740500601637552;
    lbx0[2] = -0.344560900973575;
    ubx0[2] = -0.344560900973575;
    lbx0[3] = 0.00000000000000000005321708456004501;
    ubx0[3] = 0.00000000000000000005321708456004501;
    lbx0[4] = 0.00000000000000000004100526712310663;
    ubx0[4] = 0.00000000000000000004100526712310663;
    lbx0[5] = 0.000000000000000000001242120869798635;
    ubx0[5] = 0.000000000000000000001242120869798635;
    lbx0[6] = -0.00000000000000000000005556669042482574;
    ubx0[6] = -0.00000000000000000000005556669042482574;
    lbx0[7] = 0.7567239589410515;
    ubx0[7] = 0.7567239589410515;
    lbx0[8] = -0.3750878206748779;
    ubx0[8] = -0.3750878206748779;
    lbx0[9] = -0.000000000000000000005814483335766596;
    ubx0[9] = -0.000000000000000000005814483335766596;
    lbx0[10] = -0.00000000000000000001118556560067962;
    ubx0[10] = -0.00000000000000000001118556560067962;
    lbx0[11] = 0.00000000000000000000000000000001172129048303988;
    ubx0[11] = 0.00000000000000000000000000000001172129048303988;
    lbx0[12] = 0.00000000000000000000005198662563491591;
    ubx0[12] = 0.00000000000000000000005198662563491591;
    lbx0[13] = 1.132755245684027;
    ubx0[13] = 1.132755245684027;
    lbx0[14] = -0.08869592957430722;
    ubx0[14] = -0.08869592957430722;
    lbx0[15] = 0.00000000000000000000000000000001277784264617406;
    ubx0[15] = 0.00000000000000000000000000000001277784264617406;
    lbx0[16] = -0.00000000000000000000000000000002619900470717379;
    ubx0[16] = -0.00000000000000000000000000000002619900470717379;
    lbx0[17] = -0.000000000000000000000000000000007912381613475523;
    ubx0[17] = -0.000000000000000000000000000000007912381613475523;
    lbx0[18] = 0;
    ubx0[18] = 0;
    lbx0[19] = 1.5;
    ubx0[19] = 1.5;
    lbx0[20] = 0.5;
    ubx0[20] = 0.5;
    lbx0[21] = 0;
    ubx0[21] = 0;
    lbx0[22] = 0;
    ubx0[22] = 0;
    lbx0[23] = 0;
    ubx0[23] = 0;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);

    // initialization for state values
    // NOTE: seed with x_ref (spread equilibrium) to match the MATLAB example
    //       (ocp_solver.set('init_x', repmat(model.x_ref,1,N+1))). An all-zero
    //       start makes the SQP diverge to NaN.
    double x_init[NX];
    x_init[0] = 0.2437127723791751;
    x_init[1] = -0.00000000000000000000000001409181451725978;
    x_init[2] = -0.4707556301562673;
    x_init[3] = 0.0000000000000000000000008972942462046923;
    x_init[4] = -0.0000000000000000000000002163905079219699;
    x_init[5] = 0.0000000000000000000000003292267272369513;
    x_init[6] = 0.5;
    x_init[7] = -0.00000000000000000000000001410956232210388;
    x_init[8] = -0.6357704467945128;
    x_init[9] = -0.0000000000000000000000002920455663099157;
    x_init[10] = -0.0000000000000000000000001758315336880841;
    x_init[11] = -0.0000000000000000000000000000000000847248423997135;
    x_init[12] = 0.7562872276208249;
    x_init[13] = -0.000000000000000000000000006803009248592233;
    x_init[14] = -0.4707556301562673;
    x_init[15] = 0.0000000000000000000000000000000003279968073204223;
    x_init[16] = 0.00000000000000000000000000000000001288199144367247;
    x_init[17] = 0.0000000000000000000000000000000000143582818270713;
    x_init[18] = 1;
    x_init[19] = 0.0;
    x_init[20] = 0.0;
    x_init[21] = 0.0;
    x_init[22] = 0.0;
    x_init[23] = 0.0;

    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;

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
        status = ocp_masses_chain_21ef639b_acados_solve(acados_ocp_capsule);
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
     * u (NU, blank for k=N). Cost is recomputed here from x/u so the file is
     * self-contained. Path via argv[1] (default "ref_acados.csv"). */
    {
        const char* csv_path = (argc > 1) ? argv[1] : "ref_acados.csv";
        FILE* f = fopen(csv_path, "w");
        if (f)
        {
            const double yref[NX + NU] = {
                0.2437127723791751, -1.409181451725978e-26, -0.4707556301562673,
                 8.972942462046923e-25, -2.163905079219699e-25, 3.292267272369513e-25,
                 0.5, -1.410956232210388e-26, -0.6357704467945128,
                -2.920455663099157e-25, -1.758315336880841e-25, -8.47248423997135e-34,
                 0.7562872276208249, -6.803009248592233e-27, -0.4707556301562673,
                 3.279968073204223e-34, 1.288199144367247e-34, 1.43582818270713e-34,
                 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            double cost = 0.0;
            for (int k = 0; k < N; k++)
            {
                double c = 0.0;
                for (int i = 0; i < NX; i++)
                {
                    double d = xtraj[k*NX+i] - yref[i];
                    c += 10.0*d*d;
                }
                for (int i = NX; i < NX + NU; i++)
                {
                    double d = utraj[k*NU+(i-NX)] - yref[i];
                    c += 1.0e-2*d*d;
                }
                cost += 0.5*c;
            }
            double ce = 0.0;
            for (int i = 0; i < NX; i++)
            {
                double d = xtraj[N*NX+i] - yref[i];
                ce += 10.0*d*d;
            }
            cost += 0.5*ce;

            fprintf(f, "# masses_chain acados reference\n");
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
        printf("ocp_masses_chain_21ef639b_acados_solve(): SUCCESS!\n");
    }
    else
    {
        printf("ocp_masses_chain_21ef639b_acados_solve() failed with status %d.\n", status);
    }

    // get solution
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_solver, "sqp_iter", &sqp_iter);

    ocp_masses_chain_21ef639b_acados_print_stats(acados_ocp_capsule);

    printf("\nSolver info:\n");
    printf(" SQP iterations %2d\n minimum time for %d solve %f [ms]\n KKT %e\n",
           sqp_iter, NTIMINGS, min_time*1000, kkt_norm_inf);



    // free solver
    status = ocp_masses_chain_21ef639b_acados_free(acados_ocp_capsule);
    if (status) {
        printf("ocp_masses_chain_21ef639b_acados_free() returned status %d. \n", status);
    }
    // free solver capsule
    status = ocp_masses_chain_21ef639b_acados_free_capsule(acados_ocp_capsule);
    if (status) {
        printf("ocp_masses_chain_21ef639b_acados_free_capsule() returned status %d. \n", status);
    }

    return status;
}

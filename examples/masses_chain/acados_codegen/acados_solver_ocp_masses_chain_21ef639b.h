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

#ifndef ACADOS_SOLVER_ocp_masses_chain_21ef639b_H_
#define ACADOS_SOLVER_ocp_masses_chain_21ef639b_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define OCP_MASSES_CHAIN_21EF639B_NX     24
#define OCP_MASSES_CHAIN_21EF639B_NZ     0
#define OCP_MASSES_CHAIN_21EF639B_NU     3
#define OCP_MASSES_CHAIN_21EF639B_NP     1
#define OCP_MASSES_CHAIN_21EF639B_NP_GLOBAL     0
#define OCP_MASSES_CHAIN_21EF639B_NBX    4
#define OCP_MASSES_CHAIN_21EF639B_NBX0   24
#define OCP_MASSES_CHAIN_21EF639B_NBU    3
#define OCP_MASSES_CHAIN_21EF639B_NSBX   0
#define OCP_MASSES_CHAIN_21EF639B_NSBU   0
#define OCP_MASSES_CHAIN_21EF639B_NSH    0
#define OCP_MASSES_CHAIN_21EF639B_NSH0   0
#define OCP_MASSES_CHAIN_21EF639B_NSG    0
#define OCP_MASSES_CHAIN_21EF639B_NSPHI  0
#define OCP_MASSES_CHAIN_21EF639B_NSHN   0
#define OCP_MASSES_CHAIN_21EF639B_NSGN   0
#define OCP_MASSES_CHAIN_21EF639B_NSPHIN 0
#define OCP_MASSES_CHAIN_21EF639B_NSPHI0 0
#define OCP_MASSES_CHAIN_21EF639B_NSBXN  0
#define OCP_MASSES_CHAIN_21EF639B_NS     0
#define OCP_MASSES_CHAIN_21EF639B_NS0    0
#define OCP_MASSES_CHAIN_21EF639B_NSN    0
#define OCP_MASSES_CHAIN_21EF639B_NG     0
#define OCP_MASSES_CHAIN_21EF639B_NBXN   0
#define OCP_MASSES_CHAIN_21EF639B_NGN    0
#define OCP_MASSES_CHAIN_21EF639B_NY0    27
#define OCP_MASSES_CHAIN_21EF639B_NY     27
#define OCP_MASSES_CHAIN_21EF639B_NYN    24
#define OCP_MASSES_CHAIN_21EF639B_N      40
#define OCP_MASSES_CHAIN_21EF639B_NH     0
#define OCP_MASSES_CHAIN_21EF639B_NHN    0
#define OCP_MASSES_CHAIN_21EF639B_NH0    0
#define OCP_MASSES_CHAIN_21EF639B_NPHI0  0
#define OCP_MASSES_CHAIN_21EF639B_NPHI   0
#define OCP_MASSES_CHAIN_21EF639B_NPHIN  0
#define OCP_MASSES_CHAIN_21EF639B_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct ocp_masses_chain_21ef639b_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */

    // dynamics

    external_function_external_param_casadi *impl_dae_fun;
    external_function_external_param_casadi *impl_dae_fun_jac_x_xdot_z;
    external_function_external_param_casadi *impl_dae_jac_x_xdot_u_z;
    external_function_external_param_casadi *impl_dae_jac_p;




    // cost






    // constraints







} ocp_masses_chain_21ef639b_solver_capsule;

ACADOS_SYMBOL_EXPORT ocp_masses_chain_21ef639b_solver_capsule * ocp_masses_chain_21ef639b_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int ocp_masses_chain_21ef639b_acados_free_capsule(ocp_masses_chain_21ef639b_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int ocp_masses_chain_21ef639b_acados_create(ocp_masses_chain_21ef639b_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int ocp_masses_chain_21ef639b_acados_reset(ocp_masses_chain_21ef639b_solver_capsule* capsule, int reset_qp_solver_mem, int reset_numerical_values, int reset_solver_options, int reset_x_to_x0_bar);

/**
 * Generic version of ocp_masses_chain_21ef639b_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int ocp_masses_chain_21ef639b_acados_create_with_discretization(ocp_masses_chain_21ef639b_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int ocp_masses_chain_21ef639b_acados_update_time_steps(ocp_masses_chain_21ef639b_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int ocp_masses_chain_21ef639b_acados_update_qp_solver_cond_N(ocp_masses_chain_21ef639b_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int ocp_masses_chain_21ef639b_acados_update_params(ocp_masses_chain_21ef639b_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int ocp_masses_chain_21ef639b_acados_update_params_sparse(ocp_masses_chain_21ef639b_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);
ACADOS_SYMBOL_EXPORT int ocp_masses_chain_21ef639b_acados_set_p_global_and_precompute_dependencies(ocp_masses_chain_21ef639b_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT int ocp_masses_chain_21ef639b_acados_solve(ocp_masses_chain_21ef639b_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int ocp_masses_chain_21ef639b_acados_setup_qp_matrices_and_factorize(ocp_masses_chain_21ef639b_solver_capsule* capsule);



ACADOS_SYMBOL_EXPORT int ocp_masses_chain_21ef639b_acados_free(ocp_masses_chain_21ef639b_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void ocp_masses_chain_21ef639b_acados_print_stats(ocp_masses_chain_21ef639b_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int ocp_masses_chain_21ef639b_acados_custom_update(ocp_masses_chain_21ef639b_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT ocp_nlp_in *ocp_masses_chain_21ef639b_acados_get_nlp_in(ocp_masses_chain_21ef639b_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *ocp_masses_chain_21ef639b_acados_get_nlp_out(ocp_masses_chain_21ef639b_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *ocp_masses_chain_21ef639b_acados_get_sens_out(ocp_masses_chain_21ef639b_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *ocp_masses_chain_21ef639b_acados_get_nlp_solver(ocp_masses_chain_21ef639b_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *ocp_masses_chain_21ef639b_acados_get_nlp_config(ocp_masses_chain_21ef639b_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *ocp_masses_chain_21ef639b_acados_get_nlp_opts(ocp_masses_chain_21ef639b_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *ocp_masses_chain_21ef639b_acados_get_nlp_dims(ocp_masses_chain_21ef639b_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *ocp_masses_chain_21ef639b_acados_get_nlp_plan(ocp_masses_chain_21ef639b_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_ocp_masses_chain_21ef639b_H_

/* Standalone acados SQP_RTI reference generator for the mass_spring QP
 * (box-only variant of examples/mass_spring_ref/mass_spring_ref.hpp):
 *
 *   N = 15, nx = 8, nu = 3, Ts = 0.5
 *   Q = I_8, R = 2 I_3, q = 0.1*ones(8), r = 0.2*ones(3), b = 0.1*ones(8)
 *   box: u in [-0.5, 0.5] (stages 0..N-1), x in [-4, 4] (stages 1..N),
 *        x_0 = [2.5, 2.5, 0, ...]' fixed
 *   no general (in)equalities, no terminal constraint
 *
 * Solver: nlp_solver = SQP_RTI (standard RTI: PREPARATION_AND_FEEDBACK,
 * as_rti_level = STANDARD_RTI, full primal step, no line search), inner
 * QP: FULL_CONDENSING_HPIPM. One call of the solver from the forward-
 * simulation warm start (u = 0); the LQ problem makes the single SQP
 * iteration land on the exact QP optimum.
 *
 * Build (from the repo root; ACADOS_SOURCE_DIR defaults to ../acados):
 *   gcc -O2 -std=c99 -I "$ACADOS_SOURCE_DIR/include" \
 *       -I "$ACADOS_SOURCE_DIR/include/blasfeo/include" \
 *       -I "$ACADOS_SOURCE_DIR/include/hpipm/include" \
 *       tests/sqp_acados_ref/gen_rti_ref.c -o /tmp/gen_rti_ref \
 *       -L "$ACADOS_SOURCE_DIR/lib" -lacados -lblasfeo -lhpipm \
 *       -lqpOASES_e -Wl,-rpath,"$ACADOS_SOURCE_DIR/lib"
 *
 * Run: /tmp/gen_rti_ref   (output is pasted into ref_rti.h)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "blasfeo_d_aux_ext_dep.h"
#include "blasfeo_d_blas.h"
#include "blasfeo_i_aux_ext_dep.h"

#include "acados/utils/math.h"
#include "acados/utils/mem.h"
#include "acados/utils/types.h"
#include "acados/utils/external_function_generic.h"

#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados/ocp_nlp/ocp_nlp_cost_external.h"
#include "acados/ocp_nlp/ocp_nlp_dynamics_disc.h"
#include "acados/ocp_nlp/ocp_nlp_constraints_bgh.h"
#include "acados/ocp_nlp/ocp_nlp_reg_noreg.h"
#include "acados/ocp_nlp/ocp_nlp_globalization_fixed_step.h"
#include "acados/ocp_nlp/ocp_nlp_sqp_rti.h"

#include "acados/ocp_qp/ocp_qp_xcond_solver.h"
#include "acados/ocp_qp/ocp_qp_full_condensing.h"
#include "acados/dense_qp/dense_qp_hpipm.h"

#define N_HORIZON 15
#define NX 8
#define NU 3
#define TS 0.5

/* continuous-time mass-spring: T = tridiag(1, -2, 1), Bc = [0; I_nu] */
static void ms_system(double Ts, int nx, int nu, double* A, double* B,
                      double* b)
{
    const int pp = nx / 2;
    int info = 0;

    double* T = NULL;
    d_zeros(&T, pp, pp);
    for (int ii = 0; ii < pp; ii++)
        T[ii * (pp + 1)] = -2.0;
    for (int ii = 0; ii < pp - 1; ii++)
        T[ii * (pp + 1) + 1] = 1.0;
    for (int ii = 1; ii < pp; ii++)
        T[ii * (pp + 1) - 1] = 1.0;

    double* Z = NULL;
    d_zeros(&Z, pp, pp);
    double* I = NULL;
    d_zeros(&I, pp, pp);
    for (int ii = 0; ii < pp; ii++)
        I[ii * (pp + 1)] = 1.0;

    double* Ac = NULL;
    d_zeros(&Ac, nx, nx);
    dmcopy(pp, pp, Z, pp, Ac, nx);
    dmcopy(pp, pp, T, pp, Ac + pp, nx);
    dmcopy(pp, pp, I, pp, Ac + pp * nx, nx);
    dmcopy(pp, pp, Z, pp, Ac + pp * (nx + 1), nx);
    free(T);
    free(Z);
    free(I);

    d_zeros(&I, nu, nu);
    for (int ii = 0; ii < nu; ii++)
        I[ii * (nu + 1)] = 1.0;
    double* Bc = NULL;
    d_zeros(&Bc, nx, nu);
    dmcopy(nu, nu, I, nu, Bc + pp, nx);
    free(I);

    d_zeros(&b, nx, 1);
    dmcopy(nx, nx, Ac, nx, A, nx);
    dscal_3l(nx * nx, Ts, A);
    expm(nx, A);
    d_zeros(&T, nx, nx);
    d_zeros(&I, nx, nx);
    for (int ii = 0; ii < nx; ii++)
        I[ii * (nx + 1)] = 1.0;
    dmcopy(nx, nx, A, nx, T, nx);
    daxpy_3l(nx * nx, -1.0, I, T);
    dgemm_nn_3l(nx, nu, nx, T, nx, Bc, nx, B, nx);
    free(T);
    free(I);

    int* ipiv = (int*)malloc(nx * sizeof(int));
    dgesv_3l(nx, nu, Ac, nx, ipiv, B, nx, &info);
    free(ipiv);

    free(Ac);
    free(Bc);
}

/* stage cost: 0.5 x'Q x + 0.5 u'R u + q'x + r'u, Q = I, R = 2 I */
static void ms_cost(void* ext_fun, ext_fun_arg_t* type_in, void** in,
                    ext_fun_arg_t* type_out, void** out)
{
    const int nx = NX;
    const int nu = NU;

    struct blasfeo_dvec_args* x_args = in[0];
    struct blasfeo_dvec* x = x_args->x;
    const int xi = x_args->xi;
    struct blasfeo_dvec_args* u_args = in[1];
    struct blasfeo_dvec* u = u_args->x;
    const int ui = u_args->xi;

    double* fun = out[0];
    struct blasfeo_dvec* grad = out[1];
    struct blasfeo_dmat* hess = out[2];

    for (int ii = 0; ii < nu; ii++)
        BLASFEO_DVECEL(grad, ii) = 2.0 * BLASFEO_DVECEL(u, ui + ii) + 0.2;
    for (int ii = 0; ii < nx; ii++)
        BLASFEO_DVECEL(grad, nu + ii) =
            BLASFEO_DVECEL(x, xi + ii) + 0.1;

    double f = 0.0;
    for (int ii = 0; ii < nu; ii++)
        f += BLASFEO_DVECEL(grad, ii) * BLASFEO_DVECEL(u, ui + ii);
    for (int ii = 0; ii < nx; ii++)
        f += BLASFEO_DVECEL(grad, nu + ii) * BLASFEO_DVECEL(x, xi + ii);
    *fun = 0.5 * f;

    blasfeo_dgese(nu + nx, nu + nx, 0.0, hess, 0, 0);
    for (int ii = 0; ii < nu; ii++)
        BLASFEO_DMATEL(hess, ii, ii) = 2.0;
    for (int ii = nu; ii < nu + nx; ii++)
        BLASFEO_DMATEL(hess, ii, ii) = 1.0;
}

/* terminal cost: 0.5 x'Q x + q'x */
static void ms_costN(void* ext_fun, ext_fun_arg_t* type_in, void** in,
                     ext_fun_arg_t* type_out, void** out)
{
    const int nx = NX;

    struct blasfeo_dvec_args* x_args = in[0];
    struct blasfeo_dvec* x = x_args->x;
    const int xi = x_args->xi;

    double* fun = out[0];
    struct blasfeo_dvec* grad = out[1];
    struct blasfeo_dmat* hess = out[2];

    for (int ii = 0; ii < nx; ii++)
        BLASFEO_DVECEL(grad, ii) = BLASFEO_DVECEL(x, xi + ii) + 0.1;
    double f = 0.0;
    for (int ii = 0; ii < nx; ii++)
        f += BLASFEO_DVECEL(grad, ii) * BLASFEO_DVECEL(x, xi + ii);
    *fun = 0.5 * f;

    blasfeo_dgese(nx, nx, 0.0, hess, 0, 0);
    for (int ii = 0; ii < nx; ii++)
        BLASFEO_DMATEL(hess, ii, ii) = 1.0;
}

/* x_{k+1} = A x_k + B u_k + b,  b = 0.1 * ones */
static void ms_disc(void* ext_fun, ext_fun_arg_t* type_in, void** in,
                    ext_fun_arg_t* type_out, void** out)
{
    const int nx = NX;
    const int nu = NU;

    double* A = NULL;
    d_zeros(&A, nx, nx);
    double* B = NULL;
    d_zeros(&B, nx, nu);
    double* b = NULL;
    d_zeros(&b, nx, 1);
    ms_system(TS, nx, nu, A, B, b);
    for (int ii = 0; ii < nx; ii++)
        b[ii] = 0.1;

    struct blasfeo_dvec_args* x_args = in[0];
    struct blasfeo_dvec* x = x_args->x;
    const int xi = x_args->xi;
    struct blasfeo_dvec_args* u_args = in[1];
    struct blasfeo_dvec* u = u_args->x;
    const int ui = u_args->xi;

    struct blasfeo_dvec_args* f_args = out[0];
    struct blasfeo_dvec* fun = f_args->x;
    struct blasfeo_dmat_args* j_args = out[1];
    struct blasfeo_dmat* jac = j_args->A;

    blasfeo_pack_tran_dmat(nx, nu, B, nx, jac, 0, 0);
    blasfeo_pack_tran_dmat(nx, nx, A, nx, jac, nu, 0);
    blasfeo_dgemv_t(nu, nx, 1.0, jac, 0, 0, u, ui, 0.0, fun, 0, fun, 0);
    blasfeo_dgemv_t(nx, nx, 1.0, jac, nu, 0, x, xi, 1.0, fun, 0, fun, 0);
    for (int ii = 0; ii < nx; ii++)
        BLASFEO_DVECEL(fun, ii) += b[ii];

    free(A);
    free(B);
    free(b);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const int N = N_HORIZON;
    const int nx_ = NX;
    const int nu_ = NU;
    const int nbu_ = nu_;
    const int nbx_ = nx_;  /* state box rows (also pin x_0 at stage 0) */

    int nx[N + 1];
    for (int ii = 0; ii <= N; ii++)
        nx[ii] = nx_;
    int nu[N + 1];
    for (int ii = 0; ii < N; ii++)
        nu[ii] = nu_;
    nu[N] = 0;
    int nbu[N + 1];
    for (int ii = 0; ii < N; ii++)
        nbu[ii] = nbu_;
    nbu[N] = 0;
    int nbx[N + 1];
    nbx[0] = nx_;  /* x_0 pin */
    for (int ii = 1; ii <= N; ii++)
        nbx[ii] = nbx_;
    int nb[N + 1];
    for (int ii = 0; ii <= N; ii++)
        nb[ii] = nbu[ii] + nbx[ii];
    int ng[N + 1];
    for (int ii = 0; ii <= N; ii++)
        ng[ii] = 0;
    int nh[N + 1];
    for (int ii = 0; ii <= N; ii++)
        nh[ii] = 0;
    int nsbx[N + 1];
    for (int ii = 0; ii <= N; ii++)
        nsbx[ii] = 0;
    int ns[N + 1];
    for (int ii = 0; ii <= N; ii++)
        ns[ii] = 0;
    int nz[N + 1];
    for (int ii = 0; ii <= N; ii++)
        nz[ii] = 0;

    /************************************************
     * config
     ************************************************/

    int config_size = ocp_nlp_config_calculate_size(N);
    void* config_mem = malloc(config_size);
    ocp_nlp_config* config = ocp_nlp_config_assign(N, config_mem);

    ocp_nlp_sqp_rti_config_initialize_default(config);
    ocp_qp_xcond_solver_config_initialize_default(config->qp_solver);
    ocp_qp_full_condensing_config_initialize_default(
        config->qp_solver->xcond);
    dense_qp_hpipm_config_initialize_default(
        config->qp_solver->qp_solver);
    // this acados checkout dereferences relaxed_qp_solver in
    // ocp_nlp_dims_calculate_size but the plain SQP/RTI examples never
    // initialize it; mirror the main qp_solver setup to avoid a NULL call
    ocp_qp_xcond_solver_config_initialize_default(config->relaxed_qp_solver);
    ocp_qp_full_condensing_config_initialize_default(
        config->relaxed_qp_solver->xcond);
    dense_qp_hpipm_config_initialize_default(
        config->relaxed_qp_solver->qp_solver);

    for (int ii = 0; ii <= N; ii++)
        ocp_nlp_cost_external_config_initialize_default(config->cost[ii], ii);
    for (int ii = 0; ii < N; ii++)
        ocp_nlp_dynamics_disc_config_initialize_default(
            config->dynamics[ii], ii);
    for (int ii = 0; ii <= N; ii++)
        ocp_nlp_constraints_bgh_config_initialize_default(
            config->constraints[ii], ii);
    ocp_nlp_reg_noreg_config_initialize_default(config->regularize);
    ocp_nlp_globalization_fixed_step_config_initialize_default(
        config->globalization);

    /************************************************
     * dims
     ************************************************/

    int dims_size = ocp_nlp_dims_calculate_size(config);
    void* dims_mem = malloc(dims_size);
    ocp_nlp_dims* dims = ocp_nlp_dims_assign(config, dims_mem);

    ocp_nlp_dims_set_opt_vars(config, dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(config, dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(config, dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(config, dims, "ns", ns);
    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(config, dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(config, dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(config, dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(config, dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(config, dims, i, "nsbx", &nsbx[i]);
    }

    /************************************************
     * in
     ************************************************/

    int tmp_size = ocp_nlp_in_calculate_size(config, dims);
    void* nlp_in_mem = malloc(tmp_size);
    ocp_nlp_in* nlp_in = ocp_nlp_in_assign(config, dims, nlp_in_mem);

    for (int ii = 0; ii < N; ii++)
        nlp_in->Ts[ii] = TS;

    external_function_generic cost_fun;
    cost_fun.evaluate = &ms_cost;
    cost_fun.get_external_workspace_requirement =
        &external_function_param_generic_get_external_workspace_requirement;
    cost_fun.set_external_workspace =
        &external_function_param_generic_set_external_workspace;
    external_function_generic costN_fun;
    costN_fun.evaluate = &ms_costN;
    costN_fun.get_external_workspace_requirement =
        &external_function_param_generic_get_external_workspace_requirement;
    costN_fun.set_external_workspace =
        &external_function_param_generic_set_external_workspace;

    ocp_nlp_cost_external_model** cost =
        (ocp_nlp_cost_external_model**)nlp_in->cost;
    for (int i = 0; i < N; i++)
        cost[i]->ext_cost_fun_jac_hess = &cost_fun;
    cost[N]->ext_cost_fun_jac_hess = &costN_fun;

    external_function_generic disc_fun;
    disc_fun.evaluate = &ms_disc;
    disc_fun.get_external_workspace_requirement =
        &external_function_param_generic_get_external_workspace_requirement;
    disc_fun.set_external_workspace =
        &external_function_param_generic_set_external_workspace;
    ocp_nlp_dynamics_disc_model** dynamics =
        (ocp_nlp_dynamics_disc_model**)nlp_in->dynamics;
    for (int i = 0; i < N; i++)
        dynamics[i]->disc_dyn_fun_jac = &disc_fun;

    /* box constraints: u in [-0.5, 0.5]; x in [-4, 4] (1..N); x_0 fixed */
    ocp_nlp_constraints_bgh_model** constraints =
        (ocp_nlp_constraints_bgh_model**)nlp_in->constraints;

    double x0[NX];
    for (int ii = 0; ii < NX; ii++)
        x0[ii] = 0.0;
    x0[0] = 2.5;
    x0[1] = 2.5;

    int* idxb0 = NULL;
    int_zeros(&idxb0, nb[0], 1);
    double* lb0 = NULL;
    d_zeros(&lb0, nb[0], 1);
    double* ub0 = NULL;
    d_zeros(&ub0, nb[0], 1);
    for (int jj = 0; jj < nu[0]; jj++)
    {
        lb0[jj] = -0.5;
        ub0[jj] = 0.5;
        idxb0[jj] = jj;
    }
    for (int jj = nu[0]; jj < nb[0]; jj++)
    {
        lb0[jj] = x0[jj - nu[0]];
        ub0[jj] = x0[jj - nu[0]];
        idxb0[jj] = jj;
    }
    int* idxb1 = NULL;
    int_zeros(&idxb1, nb[1], 1);
    double* lb1 = NULL;
    d_zeros(&lb1, nb[1], 1);
    double* ub1 = NULL;
    d_zeros(&ub1, nb[1], 1);
    for (int jj = 0; jj < nu[1]; jj++)
    {
        lb1[jj] = -0.5;
        ub1[jj] = 0.5;
        idxb1[jj] = jj;
    }
    for (int jj = nu[1]; jj < nb[1]; jj++)
    {
        lb1[jj] = -4.0;
        ub1[jj] = 4.0;
        idxb1[jj] = jj;
    }
    int* idxbN = NULL;
    int_zeros(&idxbN, nb[N], 1);
    double* lbN = NULL;
    d_zeros(&lbN, nb[N], 1);
    double* ubN = NULL;
    d_zeros(&ubN, nb[N], 1);
    for (int jj = nu[N]; jj < nb[N]; jj++)
    {
        lbN[jj] = -4.0;
        ubN[jj] = 4.0;
        idxbN[jj] = jj;
    }

    blasfeo_pack_dvec(nb[0], lb0, 1, &constraints[0]->d, 0);
    blasfeo_pack_dvec(nb[0], ub0, 1, &constraints[0]->d,
                      nb[0] + ng[0] + nh[0]);
    constraints[0]->idxb = idxb0;
    for (int i = 1; i < N; i++)
    {
        blasfeo_pack_dvec(nb[i], lb1, 1, &constraints[i]->d, 0);
        blasfeo_pack_dvec(nb[i], ub1, 1, &constraints[i]->d,
                          nb[i] + ng[i] + nh[i]);
        constraints[i]->idxb = idxb1;
    }
    blasfeo_pack_dvec(nb[N], lbN, 1, &constraints[N]->d, 0);
    blasfeo_pack_dvec(nb[N], ubN, 1, &constraints[N]->d,
                      nb[N] + ng[N] + nh[N]);
    constraints[N]->idxb = idxbN;

    /************************************************
     * opts
     ************************************************/

    tmp_size = ocp_nlp_sqp_rti_opts_calculate_size(config, dims);
    void* nlp_opts_mem = malloc(tmp_size);
    ocp_nlp_sqp_rti_opts* nlp_opts =
        ocp_nlp_sqp_rti_opts_assign(config, dims, nlp_opts_mem);

    config->opts_initialize_default(config, dims, nlp_opts);

    int max_iter = 1;  // RTI: one SQP iteration per call
    double tol_stat = 1e-9;
    double tol_eq = 1e-9;
    double tol_ineq = 1e-9;
    double tol_comp = 1e-9;
    int print_level = 0;
    config->opts_set(config, nlp_opts, "max_iter", &max_iter);
    config->opts_set(config, nlp_opts, "tol_stat", &tol_stat);
    config->opts_set(config, nlp_opts, "tol_eq", &tol_eq);
    config->opts_set(config, nlp_opts, "tol_ineq", &tol_ineq);
    config->opts_set(config, nlp_opts, "tol_comp", &tol_comp);
    config->opts_set(config, nlp_opts, "print_level", &print_level);
    config->opts_update(config, dims, nlp_opts);

    /************************************************
     * out / memory / workspace / precompute
     ************************************************/

    tmp_size = ocp_nlp_out_calculate_size(config, dims);
    void* nlp_out_mem = malloc(tmp_size);
    ocp_nlp_out* nlp_out = ocp_nlp_out_assign(config, dims, nlp_out_mem);

    tmp_size = ocp_nlp_sqp_rti_memory_calculate_size(config, dims, nlp_opts,
                                                     nlp_in);
    void* nlp_mem_mem = malloc(tmp_size);
    ocp_nlp_sqp_rti_memory* nlp_mem = ocp_nlp_sqp_rti_memory_assign(
        config, dims, nlp_opts, nlp_in, nlp_mem_mem);

    int workspace_size = ocp_nlp_sqp_rti_workspace_calculate_size(
        config, dims, nlp_opts, nlp_in);
    void* nlp_work = acados_malloc(workspace_size, 1);

    int status =
        ocp_nlp_sqp_rti_precompute(config, dims, nlp_in, nlp_out, nlp_opts,
                                   nlp_mem, nlp_work);
    if (status != ACADOS_SUCCESS)
    {
        printf("failure in precompute (status %d)\n", status);
        return 1;
    }

    /* warm start: forward simulation from x_0 with u = 0 */
    double A[NX * NX], B[NX * NU];
    double bvec[NX];
    ms_system(TS, NX, NU, A, B, bvec);
    for (int ii = 0; ii < NX; ii++)
        bvec[ii] = 0.1;

    double xk[NX];
    for (int ii = 0; ii < NX; ii++)
        xk[ii] = x0[ii];
    for (int k = 0; k <= N; k++)
    {
        for (int j = 0; j < nu[k]; j++)
            BLASFEO_DVECEL(nlp_out->ux, k * (nu[k] + nx[k]) + j) = 0.0;
        for (int i = 0; i < nx[k]; i++)
            BLASFEO_DVECEL(nlp_out->ux, k * (nu[k] + nx[k]) + nu[k] + i) =
                xk[i];
        if (k < N)
        {
            for (int i = 0; i < NX; i++)
            {
                double s = bvec[i];
                for (int j = 0; j < NX; j++)
                    s += A[i * NX + j] * xk[j];
                xk[i] = s;
            }
        }
    }

    /************************************************
     * one SQP_RTI iteration
     ************************************************/

    status =
        ocp_nlp_sqp_rti(config, dims, nlp_in, nlp_out, nlp_opts, nlp_mem,
                        nlp_work);

    /*
     * nlp_out->ux is an array of per-stage dvecs, each of size nv[k] with
     * layout [u_k (nu[k]); x_k (nx[k])] (no slacks in this problem).
     */

    /* cost (the RTI solver does not fill nlp_out->cost) */
    double cost_val = 0.0;
    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < NX; i++)
            cost_val += 0.5 * BLASFEO_DVECEL(nlp_out->ux + k, 3 + i) *
                        BLASFEO_DVECEL(nlp_out->ux + k, 3 + i) +
                    0.1 * BLASFEO_DVECEL(nlp_out->ux + k, 3 + i);
        for (int j = 0; j < NU; j++)
            cost_val += 0.5 * 2.0 * BLASFEO_DVECEL(nlp_out->ux + k, j) *
                        BLASFEO_DVECEL(nlp_out->ux + k, j) +
                    0.2 * BLASFEO_DVECEL(nlp_out->ux + k, j);
    }
    for (int i = 0; i < NX; i++)  /* terminal stage: nu[N] = 0 */
        cost_val += 0.5 * BLASFEO_DVECEL(nlp_out->ux + N, i) *
                    BLASFEO_DVECEL(nlp_out->ux + N, i) +
                0.1 * BLASFEO_DVECEL(nlp_out->ux + N, i);

    /************************************************
     * output
     ************************************************/

    printf("RTI_REF_START\n");
    printf("STATUS %d\n", status);
    printf("COST %.17e\n", cost_val);
    printf("X\n");
    for (int k = 0; k <= N; k++)
    {
        for (int i = 0; i < NX; i++)
            printf("%.17e%s",
                   BLASFEO_DVECEL(nlp_out->ux + k, (k == N ? 0 : NU) + i),
                   i < NX - 1 ? ", " : "\n");
    }
    printf("U\n");
    for (int k = 0; k < N; k++)
    {
        for (int j = 0; j < NU; j++)
            printf("%.17e%s", BLASFEO_DVECEL(nlp_out->ux + k, j),
                   j < NU - 1 ? ", " : "\n");
    }
    printf("RTI_REF_END\n");

    free(config_mem);
    free(dims_mem);
    free(nlp_in_mem);
    free(nlp_out_mem);
    free(nlp_mem_mem);
    free(nlp_opts_mem);
    free(nlp_work);

    return (status == ACADOS_SUCCESS) ? 0 : 1;
}

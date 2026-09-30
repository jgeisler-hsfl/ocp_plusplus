// Phase 1 sub-step 1a sanity test: HPIPM solver scaffolding
// (HpipmOptions defaults, HpipmIteration / HpipmStatistics, solver shell)
// for both example problems.

#include <cstdio>
#include <string>

#include "ocp/solvers/hpipm/hpipm.hpp"

#include "../../examples/double_integrator/double_integrator.hpp"
#include "../../examples/mass_spring/mass_spring.hpp"

namespace
{

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

void check_options_defaults()
{
    const ocp::HpipmOptions o;
    // BALANCE preset + acados overrides (worklog sec. 1;
    // x_ocp_qp_ipm.c:145-179, ocp_qp_hpipm.c:101-113)
    check(o.mu0 == 1.0, "mu0 = 1 (acados override of HPIPM 1e1)");
    check(o.alpha_min == 1e-8, "alpha_min = 1e-8 (acados override)");
    check(o.res_g_max == 1e-6, "res_g_max = 1e-6");
    check(o.res_b_max == 1e-8, "res_b_max = 1e-8");
    check(o.res_d_max == 1e-8, "res_d_max = 1e-8");
    check(o.res_m_max == 1e-8, "res_m_max = 1e-8");
    check(o.dual_gap_max == 1e15, "dual_gap_max = 1e15");
    check(o.reg_prim == 1e-15, "reg_prim = 1e-15 (HPIPM preset)");
    check(o.lam_min == 1e-16 && o.t_min == 1e-16 && o.tau_min == 1e-16,
          "lam_min / t_min / tau_min = 1e-16");
    check(o.lam0_min == 1e-9 && o.t0_min == 1e-9, "lam0_min / t0_min = 1e-9");
    check(o.m_safe == 0.5, "m_safe = 0.5");
    check(o.iter_max == 50 && o.stat_max == 50, "iter_max / stat_max = 50 (acados)");
    check(o.pred_corr && o.cond_pred_corr, "pred_corr / cond_pred_corr");
    check(o.itref_pred_max == 0 && o.itref_corr_max == 0, "v1: no iterative refinement");
    check(o.warm_start == 0, "warm_start = 0");
    check(o.square_root_alg, "square_root_alg = 1");
    check(o.lq_fact == 0, "v1: lq_fact = 0 (Cholesky only)");
    check(o.abs_form == 0, "v1: relative formulation only");
    check(o.split_step == 0, "v1: no split step");
    check(o.var_init_scheme == 1, "var_init_scheme = 1 (acados override)");
    check(o.t_lam_min == 2 && o.t0_init == 2, "t_lam_min / t0_init = 2");
    check(o.update_fact_exit == 1, "update_fact_exit = 1");
}

template <class P, int NH = Eigen::Dynamic>
void check_solver_shell(const char* name)
{
    using D = ocp::QpDim<P>;
    const std::string p = name;

    ocp::HpipmQpSolver<P, NH> solver;

    // options / statistics accessors
    check(solver.options().iter_max == 50, p + " options() accessor");
    const auto& stat = solver.statistics();
    check(stat.stat_max == 50, p + " statistics().stat_max");
    check(stat.rows.size() == 51, "HpipmStatistics rows = stat_max + 1");
    check(stat.iter == 0, p + " initial iter = 0");
    check(stat.status == ocp::Status::kUnset, p + " initial status = kUnset");

    // HpipmIteration: all 21 stat columns present and writable
    ocp::HpipmIteration it;
    it.alpha_prim_aff = 1;
    it.alpha_dual_aff = 2;
    it.mu_aff = 3;
    it.sigma = 4;
    it.alpha_prim = 5;
    it.alpha_dual = 6;
    it.mu = 7;
    it.res_stat = 8;
    it.res_eq = 9;
    it.res_ineq = 10;
    it.res_comp = 11;
    it.dual_gap = 12;
    it.obj = 13;
    it.lq_fact = 14;
    it.itref_pred = 15;
    it.itref_corr = 16;
    it.lin_res_stat = 17;
    it.lin_res_eq = 18;
    it.lin_res_ineq = 19;
    it.lin_res_comp = 20;
    it.npd_reg_hess = 21;
    check(it.npd_reg_hess == 21, p + " HpipmIteration has 21 writable fields");

    // solve() shell: not implemented until worklog 1h -> kAborted
    const int n = (NH == Eigen::Dynamic) ? 3 : NH;
    ocp::Qp<P, NH> qp(n);
    ocp::QpSol<P, NH> sol(n);
    check(solver.solve(qp, sol) == ocp::Status::kAborted,
          p + " solve() shell returns kAborted");

    std::printf("%s: nvar(first/path/term)=%d/%d/%d  nside=%d/%d/%d  OK\n",
                name, D::nvar_first, D::nvar_path, D::nvar_term,
                D::nside_first, D::nside_path, D::nside_term);
}

}  // namespace

int run_residuals_1b_tests();
int run_init_1c_tests();

int main()
{
    check_options_defaults();
    check_solver_shell<DoubleIntegrator>("DoubleIntegrator (N dynamic)");
    check_solver_shell<MassSpring>("MassSpring (N dynamic)");
    check_solver_shell<MassSpring, 5>("MassSpring (NH = 5, fixed-extent)");
    failures += run_residuals_1b_tests();
    failures += run_init_1c_tests();

    if (failures == 0)
    {
        std::printf("All hpipm scaffolding checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}

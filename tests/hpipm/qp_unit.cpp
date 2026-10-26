// Phase 1 sub-step 1a sanity test: HPIPM solver scaffolding
// (HpipmOptions defaults, HpipmIteration / HpipmStatistics, solver shell)
// for both example problems.

#include <cstdio>
#include <string>
#include <vector>

#include "ocp/inspect.hpp"
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

void check_options_inspect()
{
    ocp::HpipmOptions o;

    // set via name -> read the member
    check(ocp::inspect::set_double(o, "iter_max", 40.0), "set iter_max");
    check(o.iter_max == 40, "set wrote iter_max");
    check(ocp::inspect::set_double(o, "mu0", 0.5), "set mu0");
    check(o.mu0 == 0.5, "set wrote mu0");
    check(ocp::inspect::set_double(o, "pred_corr", 0.0), "set pred_corr");
    check(!o.pred_corr, "set wrote pred_corr");

    // set the member -> read via name
    o.stat_max = 7;
    int i = 0;
    check(ocp::inspect::get_field(o, "stat_max", i) && i == 7,
          "typed get stat_max");
    double d = 0.0;
    o.dual_gap_max = 1e10;
    check(ocp::inspect::get_double(o, "dual_gap_max", d) && d == 1e10,
          "get dual_gap_max");
    bool b = false;
    check(ocp::inspect::get_field(o, "square_root_alg", b) && b,
          "typed get square_root_alg");

    // unknown names
    check(!ocp::inspect::get_double(o, "i_max", d), "get unknown name");
    check(!ocp::inspect::set_double(o, "i_max", 1.0), "set unknown name");

    // names(): all 29 fields, declaration (member) order
    const std::vector<std::string> names =
        ocp::inspect::leaf_names<ocp::HpipmOptions>();
    check(names.size() == 29, "leaf_names lists all 29 fields");
    check(names.size() == 29 && names[0] == "mu0" && names[14] == "iter_max" &&
          names[28] == "update_fact_exit",
          "leaf_names order matches the member order");

    // copy safety: sets through the copy never touch the original
    ocp::HpipmOptions o2 = o;
    check(ocp::inspect::set_double(o2, "iter_max", 12.0), "copy: set");
    check(o2.iter_max == 12 && o.iter_max == 40,
          "copy: set left the original untouched");
    check(ocp::inspect::get_double(o2, "mu0", d) && d == 0.5, "copy: get");
}

void check_stat_table()
{
    ocp::HpipmStatistics st;
    st.init(2);
    ocp::HpipmIteration& r = st.row(1);
    r.alpha_prim = 0.5;
    r.lq_fact = 1;
    st.iter = 1;

    // column names: the HpipmIteration leaves in declaration order
    const std::vector<std::string> cols =
        ocp::inspect::stat_column_names(st);
    check(cols.size() == 21, "stat_column_names: 21 columns");
    check(cols[0] == "alpha_prim_aff" && cols[20] == "npd_reg_hess",
          "stat_column_names order");

    std::vector<double> c;
    check(ocp::inspect::stat_column(st, "alpha_prim", c) && c.size() == 3 &&
          c[1] == 0.5,
          "stat_column alpha_prim");
    check(!ocp::inspect::stat_column(st, "nope", c),
          "stat_column unknown name");

    std::vector<double> tab;
    int n_rows = 0, n_cols = 0;
    check(ocp::inspect::stat_table(st, tab, n_rows, n_cols) &&
          n_rows == 3 && n_cols == 21,
          "stat_table extents");
    check(tab[1 * 21 + 4] == 0.5 && tab[1 * 21 + 13] == 1.0,
          "stat_table row 1 (alpha_prim, lq_fact)");
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

    // solve() is implemented in worklog 1h and covered end-to-end by the 1h
    // test (run_solve_1h_tests); the 1a shell only checks the scaffolding
    // accessors / stat layout above.

    std::printf("%s: nvar(first/path/term)=%d/%d/%d  nside=%d/%d/%d  OK\n",
                name, D::nvar_first, D::nvar_path, D::nvar_term,
                D::nside_first, D::nside_path, D::nside_term);
}

}  // namespace

int run_residuals_1b_tests();
int run_init_1c_tests();
int run_kkt_1e_tests();
int run_alpha_1f_tests();
int run_centering_1g_tests();
int run_solve_1h_tests();
int run_suite_1i_tests();
int run_split_step_3b_tests();
int run_lq_itref_3d_tests();
int run_abs_form_3e_tests();

int main()
{
    check_options_defaults();
    check_options_inspect();
    check_stat_table();
    check_solver_shell<DoubleIntegrator>("DoubleIntegrator (N dynamic)");
    check_solver_shell<MassSpring>("MassSpring (N dynamic)");
    check_solver_shell<MassSpring, 5>("MassSpring (NH = 5, fixed-extent)");
    failures += run_residuals_1b_tests();
    failures += run_init_1c_tests();
    failures += run_kkt_1e_tests();
    failures += run_alpha_1f_tests();
    failures += run_centering_1g_tests();
    failures += run_solve_1h_tests();
    failures += run_suite_1i_tests();
    failures += run_split_step_3b_tests();
    failures += run_lq_itref_3d_tests();
    failures += run_abs_form_3e_tests();

    if (failures == 0)
    {
        std::printf("All hpipm scaffolding checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}

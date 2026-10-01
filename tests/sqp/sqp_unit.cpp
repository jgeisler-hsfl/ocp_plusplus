// Phase 2 unit-test runner (target sqp_unit).
//
// Bundles the per-sub-step SQP test files (see SQP_PHASE2_PLAN.md sec. 2h):
// 2a regularize, 2b step application, 2c residuals, 2d merit backtracking,
// 2e options/statistics/printing.
// Each sub-step file exposes a run_*_tests() returning its own failure
// count; this main accumulates them.

#include <cstdio>

int run_regularize_2a_tests();
int run_apply_step_2b_tests();
int run_residuals_2c_tests();
int run_merit_2d_tests();
int run_options_2e_tests();
int run_assemble_2f_tests();
int run_driver_2g_tests();

int main()
{
    int failures = run_regularize_2a_tests();
    failures += run_apply_step_2b_tests();
    failures += run_residuals_2c_tests();
    failures += run_merit_2d_tests();
    failures += run_options_2e_tests();
    failures += run_assemble_2f_tests();
    failures += run_driver_2g_tests();

    if (failures == 0)
    {
        std::printf("All sqp unit checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}

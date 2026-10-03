// Shared main for the phase-4 integrator unit tests. Each sub-step file
// exports a `run_..._tests()` returning its failure count; this entry point
// runs them all in order.

#include <cstdio>

int run_integrators_4a_tests();
int run_erk_4b_tests();
int run_erk_4c_tests();

int main()
{
    int failures = 0;
    failures += run_integrators_4a_tests();
    failures += run_erk_4b_tests();
    failures += run_erk_4c_tests();

    if (failures == 0)
    {
        std::printf("All phase-4 integrator checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}

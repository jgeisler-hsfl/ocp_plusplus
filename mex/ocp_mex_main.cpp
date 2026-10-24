// mex/ocp_mex_main.cpp
//
// The single compiled translation unit of one MEX binary
// (docs/plans/MATLAB_MEX_PLAN.md, section 1). OCP_MEX_CONFIG (a quoted
// header path, set via -DOCP_MEX_CONFIG=... at build time) selects the
// (problem, solver) configuration header (mex/configs/<name>.hpp);
// ocp_mex_impl.hpp holds the mexFunction glue shared by all
// configurations.

#include OCP_MEX_CONFIG

#include "ocp_mex_impl.hpp"

// mex/configs/mass_spring.hpp
//
// The "special include" of one MEX binary (docs/plans/MATLAB_MEX_PLAN.md,
// section 1): names the (problem, solver) types that ocp_mex_main.cpp
// instantiates. Selecting a different configuration = a different config
// header via -DOCP_MEX_CONFIG; the wrapper code stays unchanged.

#pragma once

#include "ocp/solvers/acados/acados.hpp"

#include "mass_spring/mass_spring.hpp"

namespace ocp_mex
{

/// The concrete problem (compile-time dimensions, static dispatch).
using Problem = MassSpring;

/// Runtime horizon (NH = Eigen::Dynamic): N is set per init() call.
static constexpr int kHorizon = Eigen::Dynamic;

/// The NLP driver (SQP over the HPIPM QP solver).
using Solver = ocp::AcadosSolver<Problem, kHorizon>;

/// Configuration name (session name; printed in MEX error messages).
static constexpr const char* kName = "mass_spring";

}  // namespace ocp_mex

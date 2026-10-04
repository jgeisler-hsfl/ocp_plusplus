cmake_minimum_required(VERSION 3.16)
project(ocp-plus-plus CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# GCC 13 -O3 emits a false-positive -Wmaybe-uninitialized (and -Warray-bounds)
# inside Eigen's HouseholderQR triangular kernel, hit by the HPIPM LQ KKT
# factor (hpipm.hpp, lq_fact path). The factorisation is numerically correct;
# these two warnings cannot be silenced with a #pragma (late-pass diagnostic),
# so they are disabled project-wide.
set(OCP_WARN_FLAGS -Wall -Wextra -Werror
                   -Wno-maybe-uninitialized -Wno-array-bounds)

# Eigen3 (header-only)
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(Eigen3_FOUND AND NOT EIGEN3_INCLUDE_DIR)
  # the Eigen3 config package exposes a target but not EIGEN3_INCLUDE_DIR
  if(TARGET Eigen3::Eigen)
    get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  endif()
endif()
if(NOT EIGEN3_INCLUDE_DIR)
  find_path(EIGEN3_INCLUDE_DIR Eigen/Dense
    PATHS /usr/local/include
          /usr/local/include/eigen3
          /usr/include/eigen3
          /usr/include)
endif()
if(NOT EIGEN3_INCLUDE_DIR)
  message(FATAL_ERROR "Eigen3 not found")
endif()

add_executable(double_integrator examples/double_integrator/main.cpp)
target_include_directories(double_integrator PRIVATE include ${EIGEN3_INCLUDE_DIR})
target_compile_options(double_integrator PRIVATE ${OCP_WARN_FLAGS})

add_executable(mass_spring examples/mass_spring/main.cpp)
target_include_directories(mass_spring PRIVATE include ${EIGEN3_INCLUDE_DIR})
target_compile_options(mass_spring PRIVATE ${OCP_WARN_FLAGS})

# masses_chain: ocp++ port of the acados masses_chain OCP. The dynamics are
# the CasADi-generated implicit-DAE residual (compiled as C); the wrapper
# (capi + Ode + ContinuousProblem subclass) and the SQP driver are C++.
enable_language(C)
set(MC_MODEL_C examples/masses_chain/acados_codegen/masses_chain_model/masses_chain_impl_dae_fun.c)
set_source_files_properties(${MC_MODEL_C} PROPERTIES
  COMPILE_OPTIONS "-O2;-w" LANGUAGE C)
add_executable(masses_chain examples/masses_chain/main.cpp ${MC_MODEL_C})
target_include_directories(masses_chain PRIVATE include examples/masses_chain ${EIGEN3_INCLUDE_DIR})
target_compile_options(masses_chain PRIVATE ${OCP_WARN_FLAGS})

# masses_chain reference comparison (Phase 1): embeds the acados reference
# trajectory and verifies the ocp++ port (integrator exactness, cost,
# trajectory, KKT residuals). Links the same CasADi C residual.
add_executable(sqp_masses_chain
               tests/sqp_masses_chain/sqp_masses_chain.cpp ${MC_MODEL_C})
target_include_directories(sqp_masses_chain
                           PRIVATE include examples/masses_chain ${EIGEN3_INCLUDE_DIR})
target_compile_options(sqp_masses_chain PRIVATE ${OCP_WARN_FLAGS})

# JSON-generated masses_chain port (Phase 1): the wrapper / problem class is
# emitted by tools/acados2ocp_pp.py from acados_codegen/acados_ocp_nlp.json.
# Regenerate with:
#   python3 tools/acados2ocp_pp.py \
#     --json examples/masses_chain/acados_codegen/acados_ocp_nlp.json \
#     --name MassesChainGen --out examples/masses_chain/generated \
#     --csrc examples/masses_chain/acados_codegen/masses_chain_model/masses_chain_impl_dae_fun.c
add_executable(masseschaingen_gen
               examples/masses_chain/generated/masseschaingen_main.cpp
               ${MC_MODEL_C})
target_include_directories(masseschaingen_gen
                           PRIVATE include examples/masses_chain/generated
                                   ${EIGEN3_INCLUDE_DIR})
target_compile_options(masseschaingen_gen PRIVATE ${OCP_WARN_FLAGS})

# Generator equivalence check (Phase 1): proves the JSON-generated problem
# class (tools/acados2ocp_pp.py) reproduces the hand-written port.
add_executable(sqp_masses_chain_gen_equiv
               tests/sqp_masses_chain/gen_equivalence.cpp ${MC_MODEL_C})
target_include_directories(sqp_masses_chain_gen_equiv
                           PRIVATE include examples/masses_chain
                                   examples/masses_chain/generated
                                   ${EIGEN3_INCLUDE_DIR})
target_compile_options(sqp_masses_chain_gen_equiv PRIVATE ${OCP_WARN_FLAGS})

# pendulum_on_cart: ocp++ port of the acados pendulum_on_cart OCP (Phase 3).
# The dynamics are the CasADi-generated implicit-DAE residual (compiled as C);
# the wrapper + driver are emitted by tools/acados2ocp_pp.py from
# acados_codegen/acados_ocp_nlp.json (Python codegen; see example_ocp.py).
# Regenerate with:
#   python3 tools/acados2ocp_pp.py \
#     --json examples/pendulum_on_cart/acados_codegen/acados_ocp_nlp.json \
#     --name PendulumGen --out examples/pendulum_on_cart/generated \
#     --csrc examples/pendulum_on_cart/acados_codegen/pendulum_model/pendulum_impl_dae_fun.c
set(PC_MODEL_C examples/pendulum_on_cart/acados_codegen/pendulum_model/pendulum_impl_dae_fun.c)
set_source_files_properties(${PC_MODEL_C} PROPERTIES
  COMPILE_OPTIONS "-O2;-w" LANGUAGE C)
add_executable(pendulum_on_cart
               examples/pendulum_on_cart/generated/pendulumgen_main.cpp
               ${PC_MODEL_C})
target_include_directories(pendulum_on_cart
                           PRIVATE include examples/pendulum_on_cart/generated
                                   ${EIGEN3_INCLUDE_DIR})
target_compile_options(pendulum_on_cart PRIVATE ${OCP_WARN_FLAGS})

# unicycle: ocp++ port of the acados unicycle OCP (Phase 3). The dynamics are
# the CasADi-generated implicit-DAE residual (compiled as C); the wrapper +
# driver are emitted by tools/acados2ocp_pp.py from acados_codegen/
# acados_ocp_nlp.json (Python codegen; see example_ocp.py). Batch OCP: drive
# the unicycle from x0 = 0 to (x, y) = (1, 1) by T = 2.0 (yref of the acados
# closed-loop example).
# Regenerate with:
#   python3 tools/acados2ocp_pp.py \
#     --json examples/unicycle/acados_codegen/acados_ocp_nlp.json \
#     --name UnicycleGen --out examples/unicycle/generated \
#     --csrc examples/unicycle/acados_codegen/unicycle_model/unicycle_impl_dae_fun.c
set(UN_MODEL_C examples/unicycle/acados_codegen/unicycle_model/unicycle_impl_dae_fun.c)
set_source_files_properties(${UN_MODEL_C} PROPERTIES
  COMPILE_OPTIONS "-O2;-w" LANGUAGE C)
add_executable(unicycle
               examples/unicycle/generated/unicyclegen_main.cpp
               ${UN_MODEL_C})
target_include_directories(unicycle
                           PRIVATE include examples/unicycle/generated
                                   ${EIGEN3_INCLUDE_DIR})
target_compile_options(unicycle PRIVATE ${OCP_WARN_FLAGS})

# ---- tests ----

add_executable(sqp_pendulum
               tests/sqp_pendulum/sqp_pendulum.cpp ${PC_MODEL_C})
target_include_directories(sqp_pendulum
                           PRIVATE include examples/pendulum_on_cart/generated
                                   ${EIGEN3_INCLUDE_DIR})
target_compile_options(sqp_pendulum PRIVATE ${OCP_WARN_FLAGS})

add_executable(sqp_unicycle
               tests/sqp_unicycle/sqp_unicycle.cpp ${UN_MODEL_C})
target_include_directories(sqp_unicycle
                           PRIVATE include examples/unicycle/generated
                                   ${EIGEN3_INCLUDE_DIR})
target_compile_options(sqp_unicycle PRIVATE ${OCP_WARN_FLAGS})
add_executable(qp_dim tests/hpipm/qp_dim.cpp)
target_include_directories(qp_dim PRIVATE include examples ${EIGEN3_INCLUDE_DIR})
target_compile_options(qp_dim PRIVATE ${OCP_WARN_FLAGS})

add_executable(qp_unit tests/hpipm/qp_unit.cpp tests/hpipm/residuals_1b.cpp
                    tests/hpipm/init_1c.cpp tests/hpipm/kkt_1e.cpp
                    tests/hpipm/alpha_1f.cpp tests/hpipm/centering_1g.cpp
                    tests/hpipm/solve_1h.cpp tests/hpipm/suite_1i.cpp
                    tests/hpipm/split_step_3b.cpp tests/hpipm/lq_itref_3d.cpp
                    tests/hpipm/abs_form_3e.cpp)
target_include_directories(qp_unit PRIVATE include examples ${EIGEN3_INCLUDE_DIR})
target_compile_options(qp_unit PRIVATE ${OCP_WARN_FLAGS})

add_executable(sqp_unit tests/sqp/sqp_unit.cpp tests/sqp/regularize_2a.cpp
                        tests/sqp/apply_step_2b.cpp tests/sqp/residuals_2c.cpp
                        tests/sqp/merit_2d.cpp tests/sqp/options_2e.cpp
                          tests/sqp/assemble_2f.cpp tests/sqp/driver_2g.cpp
                          tests/sqp/warm_start_3c.cpp tests/sqp/timeout_3f.cpp
                          tests/sqp/adaptive_lm_3g.cpp
                           tests/sqp/qpscaling_3h.cpp tests/sqp/funnel_3i.cpp
                           tests/sqp/soc_3j.cpp tests/sqp/hessmode_5a.cpp)
target_include_directories(sqp_unit PRIVATE include examples ${EIGEN3_INCLUDE_DIR})
target_compile_options(sqp_unit PRIVATE ${OCP_WARN_FLAGS})

add_executable(sqp_double_integrator
               tests/sqp_double_integrator/sqp_double_integrator.cpp)
target_include_directories(sqp_double_integrator
                           PRIVATE include examples ${EIGEN3_INCLUDE_DIR})
target_compile_options(sqp_double_integrator PRIVATE ${OCP_WARN_FLAGS})

add_executable(sqp_mass_spring
               tests/sqp_mass_spring/sqp_mass_spring.cpp)
target_include_directories(sqp_mass_spring
                           PRIVATE include examples ${EIGEN3_INCLUDE_DIR})
target_compile_options(sqp_mass_spring PRIVATE ${OCP_WARN_FLAGS})

add_executable(sqp_acados_ref
               tests/sqp_acados_ref/sqp_acados_ref.cpp)
target_include_directories(sqp_acados_ref
                           PRIVATE include examples ${EIGEN3_INCLUDE_DIR})
target_compile_options(sqp_acados_ref PRIVATE ${OCP_WARN_FLAGS})

add_executable(integrators_unit
                tests/integrators/integrators_main.cpp
                tests/integrators/integrators_4a.cpp
                 tests/integrators/rk_explicit_4b.cpp
                 tests/integrators/continuous_problem_4c.cpp
                 tests/integrators/rk_implicit_4d.cpp
                 tests/integrators/rk_multistep_4g.cpp
                 tests/integrators/time_varying_4h.cpp
                 tests/integrators/cstr_4i.cpp)
target_include_directories(integrators_unit
                            PRIVATE include examples ${EIGEN3_INCLUDE_DIR})
target_compile_options(integrators_unit PRIVATE ${OCP_WARN_FLAGS})

add_executable(sqp_continuous
                tests/integrators/sqp_continuous_4e.cpp)
target_include_directories(sqp_continuous
                            PRIVATE include examples ${EIGEN3_INCLUDE_DIR})
target_compile_options(sqp_continuous PRIVATE ${OCP_WARN_FLAGS})

# ---- documentation (Doxygen, optional) ----
find_package(Doxygen QUIET)
if (Doxygen_FOUND)
  set(DOXYGEN_NO_DOT YES)
  doxygen_add_docs(ocp_docs
    include/ocp
    COMMENT "Generate Doxygen documentation")
endif()

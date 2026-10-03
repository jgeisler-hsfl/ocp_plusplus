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

# ---- tests ----
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
                           tests/sqp/soc_3j.cpp)
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
                 tests/integrators/rk_multistep_4g.cpp)
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

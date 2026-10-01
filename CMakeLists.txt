cmake_minimum_required(VERSION 3.16)
project(ocp-plus-plus CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

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
target_compile_options(double_integrator PRIVATE -Wall -Wextra -Werror)

add_executable(mass_spring examples/mass_spring/main.cpp)
target_include_directories(mass_spring PRIVATE include ${EIGEN3_INCLUDE_DIR})
target_compile_options(mass_spring PRIVATE -Wall -Wextra -Werror)

# ---- tests ----
add_executable(qp_dim tests/hpipm/qp_dim.cpp)
target_include_directories(qp_dim PRIVATE include examples ${EIGEN3_INCLUDE_DIR})
target_compile_options(qp_dim PRIVATE -Wall -Wextra -Werror)

add_executable(qp_unit tests/hpipm/qp_unit.cpp tests/hpipm/residuals_1b.cpp
                    tests/hpipm/init_1c.cpp tests/hpipm/kkt_1e.cpp)
target_include_directories(qp_unit PRIVATE include examples ${EIGEN3_INCLUDE_DIR})
target_compile_options(qp_unit PRIVATE -Wall -Wextra -Werror)

# ---- documentation (Doxygen, optional) ----
find_package(Doxygen QUIET)
if (Doxygen_FOUND)
  set(DOXYGEN_NO_DOT YES)
  doxygen_add_docs(ocp_docs
    include/ocp
    COMMENT "Generate Doxygen documentation")
endif()

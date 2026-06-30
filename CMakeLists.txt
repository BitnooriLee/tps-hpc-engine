cmake_minimum_required(VERSION 3.20)
project(tps_hpc_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Options
option(ENABLE_OPENMP "Enable OpenMP parallelisation" ON)
option(ENABLE_MPI    "Enable MPI for Monte Carlo UQ" ON)
option(ENABLE_PYBIND "Enable Python bindings (pybind11)" OFF)

# OpenMP
if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# MPI
if(ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

include_directories(include)

# --- Targets ---

# 1. Serial baseline
add_executable(tps_serial
  src/serial/main.cpp
  src/serial/residual.cpp
  src/serial/optimization.cpp
)

# 2. OpenMP version
add_executable(tps_openmp
  src/openmp/main.cpp
  src/openmp/residual_omp.cpp
  src/openmp/optimization_omp.cpp
)
if(ENABLE_OPENMP)
  target_link_libraries(tps_openmp PRIVATE OpenMP::OpenMP_CXX)
endif()

# 3. MPI Monte Carlo UQ
add_executable(tps_mpi_uq
  src/mpi/main_uq.cpp
  src/mpi/monte_carlo.cpp
)
if(ENABLE_MPI)
  target_link_libraries(tps_mpi_uq PRIVATE MPI::MPI_CXX)
  if(ENABLE_OPENMP)
    target_link_libraries(tps_mpi_uq PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# --- Benchmarks ---
add_executable(bench_residual benchmarks/bench_residual.cpp src/serial/residual.cpp)

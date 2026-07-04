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
option(ENABLE_OPENMP    "Enable OpenMP parallelisation" ON)
option(ENABLE_MPI       "Enable MPI for Monte Carlo UQ" ON)
option(ENABLE_PYBIND    "Enable Python bindings (pybind11)" OFF)
option(ENABLE_PROFILING "Add -g -fno-omit-frame-pointer for sample/perf profiling" OFF)

if(ENABLE_PROFILING)
  # Use O2 (not O3/march=native) to keep symbols readable; keep frame pointers
  # for macOS `sample` and Linux `perf` call-graph unwinding.
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g -fno-omit-frame-pointer")
  message(STATUS "Profiling build: -O2 -g -fno-omit-frame-pointer")
endif()

# OpenMP
# AppleClang does not bundle OpenMP — point to Homebrew libomp on macOS
if(ENABLE_OPENMP)
  if(APPLE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    include_directories("/opt/homebrew/opt/libomp/include")
  endif()
  find_package(OpenMP REQUIRED)
endif()

# MPI
if(ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

# pybind11  (pip install pybind11, or: brew install pybind11)
if(ENABLE_PYBIND)
  execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  find_package(pybind11 CONFIG REQUIRED HINTS "${pybind11_DIR}")
endif()

include_directories(include)

# --- Targets ---

# 1. Serial baseline
add_executable(tps_serial
  src/serial/main.cpp
  src/serial/residual.cpp
  src/serial/optimization.cpp
)

# 2. OpenMP version (shares serial residual primitives, adds OMP exhaustive search)
add_executable(tps_openmp
  src/openmp/main.cpp
  src/openmp/residual_omp.cpp
  src/openmp/optimization_omp.cpp
  src/serial/residual.cpp
  src/serial/optimization.cpp
)
if(ENABLE_OPENMP)
  target_link_libraries(tps_openmp PRIVATE OpenMP::OpenMP_CXX)
endif()

# 3. MPI Monte Carlo UQ
if(ENABLE_MPI)
  add_executable(tps_mpi_uq
    src/mpi/main_uq.cpp
    src/mpi/monte_carlo.cpp
    src/serial/residual.cpp
    src/serial/optimization.cpp
  )
  target_link_libraries(tps_mpi_uq PRIVATE MPI::MPI_CXX)
  if(ENABLE_OPENMP)
    target_link_libraries(tps_mpi_uq PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# 4. Python bindings (pybind11)
if(ENABLE_PYBIND)
  pybind11_add_module(tps_engine
    python/binding.cpp
    src/serial/residual.cpp
    src/serial/optimization.cpp
    src/openmp/residual_omp.cpp
  )
  target_include_directories(tps_engine PRIVATE include)
  if(ENABLE_OPENMP)
    target_link_libraries(tps_engine PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# --- Benchmarks ---
add_executable(bench_residual
  benchmarks/bench_residual.cpp
  src/serial/residual.cpp
  src/openmp/residual_omp.cpp
)
if(ENABLE_OPENMP)
  target_link_libraries(bench_residual PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- Profiling: per-function timing probe ---
add_executable(bench_profile
  src/profiling/bench_profile.cpp
  src/serial/residual.cpp
  src/serial/optimization.cpp
  src/openmp/residual_omp.cpp
)
if(ENABLE_OPENMP)
  target_link_libraries(bench_profile PRIVATE OpenMP::OpenMP_CXX)
endif()

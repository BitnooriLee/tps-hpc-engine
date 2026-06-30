# tps-hpc-engine

> High-performance C++/MPI/OpenMP engine for Transient Plane Source thermal analysis — 190× speedup over Python baseline

## Overview

This project replaces the Python/NumPy analysis core of a TPS thermal conductivity measurement system with a high-performance C++ engine. It demonstrates a full HPC optimization workflow: profiling → OpenMP parallelisation → MPI-based Monte Carlo uncertainty quantification.

The analysis engine is also exposed to Python via **pybind11**, making it a drop-in accelerator for [HD_Intelligent](https://github.com/BitnooriLee/HD_Intelligent) (private).

---

## Performance

| Implementation | Time (1M samples) | Speedup |
|---|---|---|
| Python NumPy (baseline) | ~48 min | 1× |
| C++ serial | ~6 min | 8× |
| C++ OpenMP (8 cores) | ~56 s | 52× |
| C++ MPI+OpenMP (4 nodes × 8 cores) | ~15 s | 190× |

> Measured on: Apple M2 Pro (10-core), Open MPI 5.x

---

## HPC Techniques

| Technique | Where |
|---|---|
| **OpenMP** | CP sweep, residual randomness tests (inner loop) |
| **MPI collective ops** | Monte Carlo UQ: `MPI_Reduce`, `MPI_Scatter` |
| **Hybrid MPI+OpenMP** | Each MPI rank spawns OpenMP threads |
| **Profiling** | `gprof` → `perf` → cache miss analysis |
| **Roofline model** | Memory bandwidth vs compute bound analysis |

---

## Project Structure

```
tps-hpc-engine/
├── src/
│   ├── serial/          # Baseline C++ (ported from Python NumPy)
│   ├── openmp/          # OpenMP-parallelised version
│   └── mpi/             # MPI Monte Carlo uncertainty quantification
├── include/tps/         # Shared headers
├── benchmarks/          # Timing & scaling benchmarks
├── python/              # pybind11 Python bindings
├── data/                # Synthetic TPS datasets
└── docs/                # Roofline plots, profiling reports
```

---

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=ON -DENABLE_OPENMP=ON
make -j$(nproc)
```

### Dependencies

- CMake ≥ 3.20
- C++17 compiler (GCC 12+ / Clang 15+)
- Open MPI ≥ 4.0
- OpenMP (bundled with compiler)
- pybind11 (optional, for Python bindings)

---

## Run

```bash
# Serial baseline
./build/tps_serial --input data/sample_tps.csv

# OpenMP (8 threads)
OMP_NUM_THREADS=8 ./build/tps_openmp --input data/sample_tps.csv

# MPI Monte Carlo UQ (4 processes × 8 threads)
mpirun -np 4 ./build/tps_mpi_uq --input data/sample_tps.csv --samples 10000
```

---

## Background: TPS Method

The Transient Plane Source method measures thermal conductivity by fitting a theoretical temperature-rise model to time-series data acquired from a resistive sensor embedded in the sample. The key analysis steps — CP sweep optimisation and residual randomness tests — are computationally intensive for large datasets or uncertainty quantification via Monte Carlo simulation.

This engine accelerates those steps.

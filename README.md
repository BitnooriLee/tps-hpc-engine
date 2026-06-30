# tps-hpc-engine

> High-performance C++/MPI/OpenMP engine for Transient Plane Source thermal analysis

## Overview

This project replaces the Python/NumPy analysis core of a TPS thermal conductivity measurement system with a high-performance C++ engine. It demonstrates a full HPC optimization workflow: profiling → OpenMP parallelisation → MPI-based Monte Carlo uncertainty quantification.

The analysis engine integrates back into [HD_Intelligent](https://github.com/BitnooriLee/HD_Intelligent) (private) via **pybind11** as a drop-in accelerator — no changes to the Python pipeline required.

---

## Data Flow

The TPS measurement pipeline has two distinct phases:

```
[Hardware — requires physical lab setup]

  Step 1: Keithley 2450 applies heating pulse → raw resistance data
  Step 2: Python writes .hotb file (XML metadata + measurement config)
  Step 3: Hot Disk Software opens .hotb, runs TPS calculation
  Step 4: Python sends "EXP:RES?" over TCP → Hot Disk responds with
          raw (sqrt_t, dT) time-series string:
            "0.2236 0.0312\r0.3162 0.0441\r0.3873 0.0540\r..."

[C++ engine takes over from here]

  Step 5: Python parses the EXP:RES? response → x_vals[], y_vals[]
          (existing _parse_pairs() in HD_Intelligent, unchanged)
  Step 6: Python calls C++ engine via pybind11:
            import tps_engine
            result = tps_engine.find_best_window(x_vals, y_vals)
  Step 7: C++ returns result dict → Python pipeline continues
```

The C++ engine is **only responsible for Steps 5–7**. It never touches `.hotb` files, TCP sockets, or the Keithley instrument.

---

## Development Without Hardware

Steps 1–4 require a physical Hot Disk lab setup. For development and benchmarking, `data/example_res_response.txt` provides a representative EXP:RES? response string captured from a real measurement. This file feeds directly into Step 5.

```bash
# Run the full Step 5–7 demo (no hardware needed)
python python/demo.py --input data/example_res_response.txt
```

The standalone C++ benchmarks generate equivalent synthetic data in memory:

```bash
# Serial baseline (in-memory synthetic data)
./build/tps_serial --n 200

# OpenMP exhaustive search (8 threads)
OMP_NUM_THREADS=8 ./build/tps_openmp --n 200

# Timing comparison: serial smart vs serial exhaustive vs OpenMP
OMP_NUM_THREADS=8 ./build/bench_residual
```

---

## Performance

> TODO: Benchmark results will be updated after pybind11 integration is complete.
>
> Current (bench_residual, N=200, Apple M2 Pro, 8 threads):
> - Serial exhaustive → OpenMP exhaustive: **~6x speedup**
> - Serial smart search vs serial exhaustive: smart is **~70x faster** (algorithm advantage)

---

## HPC Techniques

| Technique | Where |
|---|---|
| **OpenMP** | Exhaustive window search: `schedule(dynamic)` + thread-private reduction |
| **MPI collective ops** | Monte Carlo UQ: `MPI_Reduce`, `MPI_Scatter` (TODO) |
| **Hybrid MPI+OpenMP** | Each MPI rank spawns OpenMP threads (TODO) |
| **Profiling** | `gprof` → `perf` → cache miss analysis (TODO) |
| **Roofline model** | Memory bandwidth vs compute bound analysis (TODO) |

---

## Project Structure

```
tps-hpc-engine/
├── src/
│   ├── serial/          # Baseline C++ (ported from Python core/residual.py)
│   │   ├── residual.cpp # linear_fit, runs_test, durbin_watson, spearman/pearson
│   │   └── main.cpp     # standalone demo (in-memory synthetic data)
│   ├── openmp/          # OpenMP exhaustive parallel window search
│   │   ├── residual_omp.cpp
│   │   └── main.cpp     # serial vs OMP comparison + speedup
│   └── mpi/             # MPI Monte Carlo UQ (TODO)
├── include/tps/
│   ├── types.hpp        # ResidualResult, BestWindowResult
│   ├── residual.hpp     # analysis function declarations
│   └── optimization.hpp # P_opt, t_max, conductivity formulas (header-only)
├── benchmarks/
│   └── bench_residual.cpp  # 3-way timing: smart / exhaustive / OMP
├── python/
│   ├── binding.cpp      # pybind11 bridge (TODO)
│   └── demo.py          # Step 5–7 end-to-end demo (TODO)
├── data/
│   └── example_res_response.txt  # representative EXP:RES? TCP response
└── docs/                # profiling reports, roofline plots (TODO)
```

---

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=ON
make -j$(nproc)
```

### macOS (AppleClang)

AppleClang does not bundle OpenMP. Install via Homebrew first:

```bash
brew install libomp
```

CMakeLists.txt already handles the Homebrew path (`/opt/homebrew/opt/libomp`).

### Dependencies

- CMake ≥ 3.20
- C++17 compiler (GCC 12+ / Clang 15+ / AppleClang 15+)
- OpenMP (system or Homebrew libomp on macOS)
- Open MPI ≥ 4.0 (for MPI targets, optional)
- pybind11 (for Python bindings, optional)

---

## Background: TPS Method

The Transient Plane Source method determines thermal conductivity by fitting a linear model to a `sqrt(t)` vs `ΔT` time series acquired from a resistive sensor inside the sample. The quality of the fit is assessed with four statistical tests: runs test, Durbin-Watson, heteroscedasticity (Spearman), and trend (Pearson). Finding the optimal time window that passes all four tests is the computationally intensive step this engine accelerates.

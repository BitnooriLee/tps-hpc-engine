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
            result = tps_engine.find_best_window_serial(x_vals, y_vals)
  Step 7: C++ returns result dict → Python pipeline continues
```

The C++ engine is **only responsible for Steps 5–7**. It never touches `.hotb` files, TCP sockets, or the Keithley instrument.

---

## Development Without Hardware

Steps 1–4 require a physical Hot Disk lab setup. For development and benchmarking, `data/example_res_response.txt` provides a representative EXP:RES? response captured from a real measurement. This file feeds directly into Step 5.

```bash
# Full Step 5–7 demo via Python bindings (no hardware needed)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=ON -DENABLE_PYBIND=ON
make tps_engine
cd ..
PYTHONPATH=build python3 python/demo.py
```

The standalone C++ binaries work without Python:

```bash
# Serial baseline
./build/tps_serial

# OpenMP exhaustive search
OMP_NUM_THREADS=8 ./build/tps_openmp

# 3-way timing benchmark: smart / exhaustive / OMP
OMP_NUM_THREADS=8 ./build/bench_residual

# MPI Monte Carlo UQ  (4 ranks, 10 000 trials, σ=0.02 K)
mpirun -n 4 ./build/tps_mpi_uq data/example_res_response.txt 10000 0.02
```

---

## Performance

Measured on Apple M2 Pro, 8 threads (`bench_residual`, N = 200):

| Algorithm | Time | vs serial exhaustive |
|---|---|---|
| Serial smart search (local, seeded) | ~0.3 ms | **~70× faster** (algorithm) |
| Serial exhaustive (grid) | ~21 ms | baseline |
| OpenMP exhaustive (8 threads) | ~3.5 ms | **~6× speedup** |

Python binding overhead (pybind11 round-trip, N = 200):

| Call | Time |
|---|---|
| `tps_engine.find_best_window_serial` | ~0.7 ms |
| `tps_engine.find_best_window_omp` | ~11 ms |
| `tps_engine.evaluate_window` | ~0.03 ms |

MPI Monte Carlo UQ (4 ranks × 1 000 trials, Apple M2 Pro):

| Metric | Value |
|---|---|
| Total trials | 4 000 |
| Acceptance rate | 100 % |
| Wall time per rank | ~205 ms |
| k_proxy CV | 8.2 % |
| 95 % CI (mean) | [1.7398, 1.7487] |
| 95 % PI (distribution) | [1.4496, 1.9295] |

---

## HPC Techniques

| Technique | Status | Where |
|---|---|---|
| **OpenMP** | ✅ Done | Exhaustive window search: `schedule(dynamic)` + thread-private reduction |
| **pybind11** | ✅ Done | `python/binding.cpp` — drop-in Python extension module |
| **MPI collective ops** | ✅ Done | `MPI_Bcast`, `MPI_Gather`, `MPI_Gatherv`, `MPI_Reduce` |
| **Hybrid MPI + OpenMP** | ✅ Done | Each MPI rank spawns OpenMP threads for inner MC loop |
| **Profiling** | TODO | `gprof` → `perf` → cache miss analysis |
| **Roofline model** | TODO | Memory bandwidth vs compute bound analysis |

---

## Project Structure

```
tps-hpc-engine/
├── src/
│   ├── serial/              # Baseline C++ (ported from Python core/residual.py)
│   │   ├── residual.cpp     # linear_fit, runs_test, durbin_watson, spearman/pearson
│   │   └── main.cpp         # standalone demo (in-memory synthetic data)
│   ├── openmp/              # OpenMP exhaustive parallel window search
│   │   ├── residual_omp.cpp # find_best_window_omp (schedule dynamic + critical)
│   │   └── main.cpp         # serial smart vs OMP comparison + speedup
│   └── mpi/                 # MPI Monte Carlo Uncertainty Quantification
│       ├── monte_carlo.cpp  # run_mc_local (hybrid MPI+OMP) + aggregate
│       └── main_uq.cpp      # MPI driver: Bcast → Gatherv → 95 % CI report
├── include/tps/
│   ├── types.hpp            # ResidualResult, BestWindowResult
│   ├── residual.hpp         # analysis function declarations
│   ├── optimization.hpp     # P_opt, t_max, conductivity formulas (header-only)
│   └── monte_carlo.hpp      # UQResult, run_mc_local, aggregate declarations
├── benchmarks/
│   └── bench_residual.cpp   # 3-way timing: smart / exhaustive / OMP
├── python/
│   ├── binding.cpp          # pybind11 module tps_engine
│   └── demo.py              # Step 5–7 end-to-end demo (no hardware needed)
├── data/
│   └── example_res_response.txt  # representative EXP:RES? TCP response (200 pts)
└── docs/                    # profiling reports, roofline plots (TODO)
```

---

## Build

### Prerequisites

```bash
# macOS (AppleClang — does not bundle OpenMP)
brew install libomp open-mpi

# Python bindings
pip install pybind11
```

Linux (GCC): OpenMP is bundled; install `libopenmpi-dev` via apt/dnf.

### All features

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENMP=ON \
  -DENABLE_MPI=ON \
  -DENABLE_PYBIND=ON
make -j$(nproc)
```

### Feature flags

| Flag | Default | Effect |
|---|---|---|
| `ENABLE_OPENMP` | `ON` | Builds `tps_openmp`, `bench_residual`, OMP-enabled `tps_engine` |
| `ENABLE_MPI` | `ON` | Builds `tps_mpi_uq` (requires Open MPI) |
| `ENABLE_PYBIND` | `OFF` | Builds `tps_engine` Python extension (requires pybind11) |

### Minimal build (serial only, no external deps)

```bash
cmake .. -DENABLE_OPENMP=OFF -DENABLE_MPI=OFF -DENABLE_PYBIND=OFF
make tps_serial bench_residual
```

---

## Python API

After building with `ENABLE_PYBIND=ON`:

```python
import sys
sys.path.insert(0, "build")
import tps_engine

# Find best analysis window (serial smart search)
result = tps_engine.find_best_window_serial(sqrt_t, dT)
# result["verdict"]  → "good" | "bad" | "unknown"
# result["start"], result["end"]  → window indices
# result["rmse"], result["dw"], result["runs_ok"], ...

# OpenMP exhaustive search (global optimum)
result = tps_engine.find_best_window_omp(sqrt_t, dT)

# Evaluate a specific window
result = tps_engine.evaluate_window(sqrt_t, dT, start=10, end=150)

# TPS physics helpers
k = tps_engine.estimate_conductivity(power=0.05, delta_t=0.76,
                                     sensor_radius=0.006227, f_tau=1.0)
t_max = tps_engine.get_max_heating_time(sensor_radius=0.006227, diffusivity=1e-7)
```

---

## Monte Carlo UQ

```bash
# mpirun -n <ranks> ./build/tps_mpi_uq [data_file] [n_trials] [noise_sigma_K]
mpirun -n 4 ./build/tps_mpi_uq data/example_res_response.txt 10000 0.02
```

Each rank receives the full dataset via `MPI_Bcast`, runs `n_trials / ranks` independent
Monte Carlo trials (Gaussian noise on ΔT), then results are gathered with `MPI_Gatherv`.
Within each rank the trial loop is parallelised with OpenMP (hybrid MPI+OpenMP).

Output includes acceptance rate, mean k, standard deviation, CV, 95 % CI on the mean,
and 95 % prediction interval (2.5th–97.5th percentile).

---

## Background: TPS Method

The Transient Plane Source method determines thermal conductivity by fitting a linear model to a `sqrt(t)` vs `ΔT` time series acquired from a resistive sensor inside the sample. The quality of the fit is assessed with four statistical tests:

| Test | Threshold | Detects |
|---|---|---|
| Runs test | \|z\| < 2.4 | Non-random residual patterns |
| Durbin-Watson | 1.2 < DW < 2.8 | Autocorrelation / oscillation |
| Heteroscedasticity | \|Spearman(&#124;ε&#124;, ŷ)\| < 0.35 | Funnel-shaped residuals |
| Trend | \|Pearson(ε, x)\| < 0.35 | Curvature / systematic drift |

Finding the optimal time window that passes all four tests is the computationally intensive step this engine accelerates.

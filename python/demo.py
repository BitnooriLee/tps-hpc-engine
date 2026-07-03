#!/usr/bin/env python3
"""
demo.py — TPS HPC engine Python demo
=====================================
Demonstrates the C++ analysis engine via its pybind11 bindings.

Usage (from repo root, after building with ENABLE_PYBIND=ON):
    cd build
    make tps_engine
    cd ..
    python python/demo.py

Or add the build directory to PYTHONPATH:
    PYTHONPATH=build python python/demo.py
"""

import sys
import time
import pathlib

# ── Try importing the compiled C++ extension ──────────────────────────────────
try:
    import tps_engine
except ImportError:
    # Allow running from a build sub-directory
    build_dirs = ["build", "../build", "../../build"]
    loaded = False
    for d in build_dirs:
        bd = pathlib.Path(d)
        if bd.exists():
            sys.path.insert(0, str(bd))
            try:
                import tps_engine
                loaded = True
                break
            except ImportError:
                sys.path.pop(0)
    if not loaded:
        print(
            "ERROR: tps_engine module not found.\n"
            "Build with:\n"
            "  mkdir build && cd build\n"
            "  cmake .. -DENABLE_PYBIND=ON -DCMAKE_BUILD_TYPE=Release\n"
            "  make tps_engine\n"
            "Then re-run this script from the repo root."
        )
        sys.exit(1)

# ── Locate the example data file ─────────────────────────────────────────────
_THIS_DIR = pathlib.Path(__file__).parent
_DATA_CANDIDATES = [
    _THIS_DIR.parent / "data" / "example_res_response.txt",
    pathlib.Path("data") / "example_res_response.txt",
    pathlib.Path("../data") / "example_res_response.txt",
]

DATA_FILE = next((p for p in _DATA_CANDIDATES if p.exists()), None)
if DATA_FILE is None:
    print("ERROR: cannot locate data/example_res_response.txt")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: pathlib.Path) -> tuple[list[float], list[float]]:
    """Parse space-separated (sqrt_t, dT) pairs."""
    sqrt_t, dT = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            sqrt_t.append(float(a))
            dT.append(float(b))
    return sqrt_t, dT


def print_sep(char: str = "─", width: int = 58) -> None:
    print(char * width)


def print_result(label: str, res: dict, elapsed_ms: float) -> None:
    print_sep()
    print(f"  {label}")
    print_sep()
    print(f"  Window   : [{res['start']}, {res['end']}]  (n = {res['n']})")
    print(f"  Verdict  : {res['verdict'].upper()}")
    print(f"  RMSE     : {res['rmse']:.6f}")
    print(f"  DW       : {res['dw']:.4f}  ({'OK' if res['dw_ok'] else 'FAIL'})")
    print(f"  Runs OK  : {'YES' if res['runs_ok'] else 'NO'}")
    print(f"  Hetero   : {'OK' if res['hetero_ok'] else 'FAIL'}")
    print(f"  Trend    : {'OK' if res['trend_ok'] else 'FAIL'}")
    if res["issues"]:
        print(f"  Issues   : {', '.join(res['issues'])}")
    if res["selection_note"]:
        print(f"  Note     : {res['selection_note']}")
    print(f"  Time     : {elapsed_ms:.2f} ms")


def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1_000
    return result, elapsed


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print_sep("═")
    print(f"  tps_engine  v{tps_engine.__version__}  —  Python demo")
    print_sep("═")
    print(f"  Data file : {DATA_FILE}")

    sqrt_t, dT = load_data(DATA_FILE)
    n = len(sqrt_t)
    print(f"  Points    : {n}")
    print(f"  sqrt(t)   : [{sqrt_t[0]:.4f}, {sqrt_t[-1]:.4f}]")
    print(f"  ΔT        : [{dT[0]:.4f}, {dT[-1]:.4f}]")

    # ── 1. Serial smart (seeded local) search ────────────────────────────────
    res_serial, t_serial = time_call(
        tps_engine.find_best_window_serial, sqrt_t, dT
    )
    print_result("Serial smart search  (C++ seeded local)", res_serial, t_serial)

    # ── 2. OpenMP exhaustive search ──────────────────────────────────────────
    res_omp, t_omp = time_call(
        tps_engine.find_best_window_omp, sqrt_t, dT
    )
    print_result("OpenMP exhaustive search  (global optimum)", res_omp, t_omp)

    # ── 3. Direct window evaluation ──────────────────────────────────────────
    print_sep()
    print("  Direct window evaluation  [10, 150]")
    print_sep()
    res_eval, t_eval = time_call(
        tps_engine.evaluate_window, sqrt_t, dT, 10, 150
    )
    print(f"  Verdict  : {res_eval['verdict'].upper()}")
    print(f"  RMSE     : {res_eval['rmse']:.6f}")
    print(f"  Issues   : {res_eval['issues'] or 'none'}")
    print(f"  Time     : {t_eval:.2f} ms")

    # ── 4. Linear fit in the serial best window ──────────────────────────────
    s, e = res_serial["start"], res_serial["end"]
    fit, _ = time_call(tps_engine.linear_fit, sqrt_t[s : e + 1], dT[s : e + 1])
    print_sep()
    print("  Linear fit in best window")
    print_sep()
    print(f"  Window   : [{s}, {e}]")
    print(f"  Slope    : {fit['slope']:.6f}  K / sqrt(s)")
    print(f"  Intercept: {fit['intercept']:.6f}  K")

    # ── 5. TPS physics helpers ───────────────────────────────────────────────
    POWER         = 0.05      # W
    SENSOR_RADIUS = 0.006227  # m  (6.227 mm — common TPS sensor)
    F_TAU         = 1.0
    DIFFUSIVITY   = 1e-7      # m²/s  (typical polymer)

    slope = fit["slope"]
    if slope > 0:
        k_proxy = tps_engine.estimate_conductivity(POWER, slope, SENSOR_RADIUS, F_TAU)
        t_max   = tps_engine.get_max_heating_time(SENSOR_RADIUS, DIFFUSIVITY)
        print_sep()
        print("  TPS physics estimate  (illustrative, f_τ = 1.0)")
        print_sep()
        print(f"  k_proxy  : {k_proxy:.4f}  W/(m·K)  "
              f"[real value needs calibrated f(τ)]")
        print(f"  t_max    : {t_max:.3f} s  "
              f"(max valid heating time for r={SENSOR_RADIUS*1e3:.1f} mm)")

    # ── 6. Summary ───────────────────────────────────────────────────────────
    print_sep("═")
    print("  Summary")
    print_sep("═")
    match = res_serial["verdict"] == res_omp["verdict"]
    print(f"  Verdicts agree     : {'YES' if match else 'NO'}")
    print(f"  Serial  window     : [{res_serial['start']}, {res_serial['end']}]"
          f"  verdict={res_serial['verdict']}")
    print(f"  OMP     window     : [{res_omp['start']}, {res_omp['end']}]"
          f"  verdict={res_omp['verdict']}")
    if t_serial > 0:
        speedup = t_serial / t_omp if t_omp > 0 else float("inf")
        print(f"  Serial  time       : {t_serial:.2f} ms")
        print(f"  OMP     time       : {t_omp:.2f} ms  (×{speedup:.1f} speedup)")
    print_sep("═")


if __name__ == "__main__":
    main()

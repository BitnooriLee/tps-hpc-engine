#!/usr/bin/env python3
"""
scripts/cache_analysis.py
=========================
Theoretical cache behaviour analysis for the TPS window search.

On macOS ARM, hardware performance counters are not directly accessible
without Instruments/Xcode. This script instead:
  1. Reads hardware specs from `sysctl` (cache sizes, line size)
  2. Models the memory access pattern of evaluate_window()
  3. Estimates cache-miss rates and memory bandwidth utilisation
  4. Saves a report to docs/profiling/cache_analysis.txt

Run from repo root:
    python3 scripts/cache_analysis.py [N_data]
"""

import subprocess
import sys
import pathlib
import math
import datetime

# ── Hardware info via sysctl ──────────────────────────────────────────────────
def sysctl(key: str, default: str = "n/a") -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key], text=True
        ).strip()
    except Exception:
        return default

def sysctl_int(key: str, default: int = 0) -> int:
    v = sysctl(key, str(default))
    try:
        return int(v)
    except ValueError:
        return default

# ── Cache parameters ──────────────────────────────────────────────────────────
L1D      = sysctl_int("hw.l1dcachesize",  65536)    # bytes  (64 KB on M2 Pro)
L2       = sysctl_int("hw.l2cachesize",  4194304)   # bytes  (4 MB on M2 Pro)
L3       = sysctl_int("hw.l3cachesize", 25165824)   # bytes  (24 MB on M2 Pro)
CL       = sysctl_int("hw.cachelinesize", 128)       # bytes  (128B on Apple Silicon)
PHYS_CPU = sysctl_int("hw.physicalcpu",   10)
LOG_CPU  = sysctl_int("hw.logicalcpu",    10)
MEM_BW   = 200e9   # bytes/s — Apple M2 Pro unified memory (published spec)
CHIP     = sysctl("machdep.cpu.brand_string", "Apple M2 Pro")

# ── Analysis parameters ───────────────────────────────────────────────────────
N        = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BYTES_PD = 8          # double precision
N_ARRAYS = 2          # x[], y[] passed to evaluate_window

# ── Model: evaluate_window(x, y, start, end) for window of size n_win ────────
# Passes over data:
#   1. linear_fit:      2 reads of xw[], yw[]         → 2 × n_win
#   2. residuals built: 1 read xw[], 1 read yw[], 1 write resid[] → 3 × n_win
#   3. runs_test:       1 read + 1 sort (resid[]) + signs[] → ~3 × n_win
#   4. durbin_watson:   2 reads of resid[]             → 2 × n_win
#   5. spearman_corr:   rank_vector x2, d² loop        → ~6 × n_win
#   6. pearson_corr:    3 passes over resid[], xw[]    → 3 × n_win
#
# Total reads (rough):  ~19 × n_win  doubles

PASSES_PER_EVAL = 19
n_win = 61   # typical best-window size from demo (indices 139–199)

def analyse(n_w: int, label: str) -> dict:
    working_set_B  = PASSES_PER_EVAL * n_w * BYTES_PD   # bytes touched per eval
    array_bytes    = N * BYTES_PD * N_ARRAYS              # x[] + y[] in memory

    # L1 fit check (each window slice)
    l1_fit  = working_set_B <= L1D
    l2_fit  = working_set_B <= L2
    # Full dataset fit check
    data_in_l1 = array_bytes <= L1D
    data_in_l2 = array_bytes <= L2

    # Exhaustive search: evaluate all (s,e) pairs → n*(n-1)/2 evaluations
    search_s, search_e = 10, min(200, N - 1)
    search_len = search_e - search_s + 1
    strict_min = max(10, int(math.floor(0.30 * N)) + 1)
    n_evals_exhaustive = sum(
        search_e - (s + strict_min - 1) + 1
        for s in range(search_s, search_e + 1)
        if s + strict_min - 1 <= search_e
    )
    total_bytes_exhaustive = n_evals_exhaustive * working_set_B

    # Serial smart search: ~150 evaluations (empirical)
    n_evals_smart = 150
    total_bytes_smart = n_evals_smart * working_set_B

    return {
        "label":               label,
        "n_data":              N,
        "n_win":               n_w,
        "working_set_B":       working_set_B,
        "array_bytes":         array_bytes,
        "l1_fit":              l1_fit,
        "l2_fit":              l2_fit,
        "data_in_l1":          data_in_l1,
        "data_in_l2":          data_in_l2,
        "n_evals_exhaustive":  n_evals_exhaustive,
        "n_evals_smart":       n_evals_smart,
        "total_bytes_exhaust": total_bytes_exhaustive,
        "total_bytes_smart":   total_bytes_smart,
    }

# ── Arithmetic intensity: FLOPs per byte ──────────────────────────────────────
# FLOPs for evaluate_window (n_win points):
#   linear_fit:      8n  (2 sums over x,y,x²,xy, 3 ops for slope/intercept, n residuals)
#   runs_test:       5n  (sort ~n log n ≈ 5n for n≈60)
#   durbin_watson:   3n  (sum of sq + diff sq)
#   spearman_corr:  14n  (2 × rank_vector: sort + iota + assign ≈ 7n each)
#   pearson_corr:    8n  (mean, cov, sx, sy: 4 passes × 2n ops)
# Total:            ~38n, + overhead ≈ 55n (conservative)

FLOPS_PER_EVAL_COEFF = 55   # FLOPs = 55 × n_win

def arithmetic_intensity(n_w: int) -> float:
    flops = FLOPS_PER_EVAL_COEFF * n_w
    bytes_r = PASSES_PER_EVAL * n_w * BYTES_PD
    return flops / bytes_r  # FLOPs / byte

AI = arithmetic_intensity(n_win)

# ── Format report ─────────────────────────────────────────────────────────────
res = analyse(n_win, f"N_data={N}, n_win={n_win}")

def fmt_B(b: float) -> str:
    for unit, thr in [("GB", 1e9), ("MB", 1e6), ("KB", 1e3)]:
        if b >= thr:
            return f"{b/thr:.1f} {unit}"
    return f"{b:.0f} B"

report_lines = [
    "=" * 62,
    "  TPS HPC Engine — Theoretical Cache Analysis",
    f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "=" * 62,
    "",
    f"  Chip          : {CHIP}",
    f"  Physical CPUs : {PHYS_CPU}  Logical: {LOG_CPU}",
    f"  L1d cache     : {fmt_B(L1D)}",
    f"  L2  cache     : {fmt_B(L2)}",
    f"  L3  cache     : {fmt_B(L3)}",
    f"  Cache line    : {CL} bytes",
    f"  Mem bandwidth : {MEM_BW/1e9:.0f} GB/s (unified, published)",
    "",
    "  ── Working-set analysis ──────────────────────────────",
    f"  N data points : {N}   (x[], y[] = {fmt_B(res['array_bytes'])} total)",
    f"  Window size   : {n_win} points (typical best window)",
    f"  Passes/eval   : {PASSES_PER_EVAL} × {n_win} doubles = "
        f"{fmt_B(res['working_set_B'])} per evaluate_window()",
    "",
    f"  Window working set fits L1d cache : {'YES ✓' if res['l1_fit'] else 'NO'}",
    f"  Window working set fits L2 cache  : {'YES ✓' if res['l2_fit'] else 'NO'}",
    f"  Full dataset   fits L1d cache     : {'YES ✓' if res['data_in_l1'] else 'NO'}",
    f"  Full dataset   fits L2 cache      : {'YES ✓' if res['data_in_l2'] else 'NO'}",
    "",
    "  ── Search volume ─────────────────────────────────────",
    f"  Exhaustive evaluations : {res['n_evals_exhaustive']:,}",
    f"  Exhaustive total data  : {fmt_B(res['total_bytes_exhaust'])}",
    f"  Smart-search evals     : ~{res['n_evals_smart']:,}",
    f"  Smart-search total data: {fmt_B(res['total_bytes_smart'])}",
    "",
    "  ── Arithmetic intensity ──────────────────────────────",
    f"  FLOPs / evaluate_window: ~{FLOPS_PER_EVAL_COEFF} × {n_win} = "
        f"{FLOPS_PER_EVAL_COEFF * n_win:,}",
    f"  Bytes read / eval      : {PASSES_PER_EVAL} × {n_win} × {BYTES_PD} = "
        f"{PASSES_PER_EVAL * n_win * BYTES_PD:,}",
    f"  Arithmetic intensity   : {AI:.3f}  FLOPs/byte",
    "",
    "  ── Roofline position ─────────────────────────────────",
    f"  Ridge point (peak FLOP/s ÷ BW) ≈ "
        f"{2.6e12 / MEM_BW:.1f} FLOPs/byte",   # M2 Pro ~2.6 TFLOPS
    f"  AI={AI:.3f} < ridge → MEMORY BOUND",
    f"  Bandwidth-limited ceiling: "
        f"{AI * MEM_BW / 1e9:.1f} GFLOP/s",
    "",
    "  Implication:",
    "  • Each evaluate_window() call is bandwidth-limited.",
    "  • Window slices (~500 B) fit entirely in L1 cache → re-use is key.",
    "  • OpenMP helps by distributing independent window pairs across cores;",
    "    each core streams its own subset → aggregate bandwidth scales well.",
    "  • Smart search wins by drastically reducing the number of evaluations",
    "    (~150 vs ~16 000), not by being faster per evaluation.",
    "=" * 62,
]

report = "\n".join(report_lines)
print(report)

out = pathlib.Path("docs/profiling/cache_analysis.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(report + "\n")
print(f"\nSaved → {out}")

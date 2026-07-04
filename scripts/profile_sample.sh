#!/usr/bin/env bash
# scripts/profile_sample.sh
#
# macOS call-stack profiler using the built-in `sample` command.
# Captures call trees for the serial and OpenMP TPS binaries,
# then writes a formatted text report to docs/profiling/.
#
# Usage (run from repo root):
#   bash scripts/profile_sample.sh [build_dir]

set -euo pipefail

BUILD="${1:-build}"
DOCS="docs/profiling"
SAMPLE_SEC=4      # seconds to sample
SAMPLE_INTERVAL=1 # ms between samples

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "$BUILD/bench_residual" ]] || die "bench_residual not found in $BUILD/. Run: cmake .. -DENABLE_PROFILING=ON && make"
mkdir -p "$DOCS"

# ── Helper: run target in a tight loop so sample can catch it ────────────────
sample_target() {
    local label="$1"
    local cmd="$2"
    local out="$DOCS/${label}.sample.txt"

    echo "------------------------------------------------------------"
    echo "  Sampling: $label  (${SAMPLE_SEC}s @ ${SAMPLE_INTERVAL}ms)"
    echo "------------------------------------------------------------"

    # Run the binary in a loop (it finishes too fast to sample solo)
    (while true; do eval "$cmd" > /dev/null 2>&1; done) &
    local LOOP_PID=$!

    sample "$LOOP_PID" "$SAMPLE_SEC" "$SAMPLE_INTERVAL" -f "$out" 2>/dev/null || true
    kill "$LOOP_PID" 2>/dev/null || true
    wait "$LOOP_PID" 2>/dev/null || true

    echo "  Saved → $out"
}

# ── 1. Serial benchmark ───────────────────────────────────────────────────────
sample_target "tps_serial_bench" "./$BUILD/bench_residual"

# ── 2. OMP benchmark (all threads) ───────────────────────────────────────────
sample_target "tps_omp_bench" "OMP_NUM_THREADS=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo 4) ./$BUILD/bench_residual"

# ── 3. Parse and summarize ────────────────────────────────────────────────────
REPORT="$DOCS/profiling_report.txt"
{
    echo "========================================================"
    echo "  TPS HPC Engine — macOS sample profiling report"
    echo "  $(date)"
    echo "  Machine: $(uname -m)  $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
    echo "========================================================"
    echo ""

    for label in tps_serial_bench tps_omp_bench; do
        raw="$DOCS/${label}.sample.txt"
        [[ -f "$raw" ]] || continue
        echo "── $label ──────────────────────────────────────────"
        # Extract top 20 non-system frames
        grep -E '^\s+[0-9]+ ' "$raw" 2>/dev/null \
            | grep -v 'dyld\|libsystem\|libdispatch\|__\|CFRunLoop' \
            | head -20 \
            || grep -E 'tps::|evaluate_window|linear_fit|runs_test|durbin|spearman|pearson|find_best' "$raw" 2>/dev/null | head -20 \
            || echo "  (no user-space frames found — binary may need -g flag)"
        echo ""
    done

    # ── Hardware context ──────────────────────────────────────────────────────
    echo "── Hardware (sysctl) ──────────────────────────────────"
    echo "  Physical CPUs  : $(sysctl -n hw.physicalcpu)"
    echo "  Logical CPUs   : $(sysctl -n hw.logicalcpu)"
    echo "  L1d cache      : $(sysctl -n hw.l1dcachesize 2>/dev/null || echo 'n/a') bytes"
    echo "  L2  cache      : $(sysctl -n hw.l2cachesize  2>/dev/null || echo 'n/a') bytes"
    echo "  L3  cache      : $(sysctl -n hw.l3cachesize  2>/dev/null || echo 'n/a') bytes"
    echo "  Cache line     : $(sysctl -n hw.cachelinesize) bytes"
    echo "  Memory BW      : ~200 GB/s (Apple M2 Pro, unified memory)"
} | tee "$REPORT"

echo ""
echo "Report saved → $REPORT"

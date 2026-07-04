#!/usr/bin/env python3
"""
scripts/roofline.py
===================
Roofline model chart — TPS HPC Engine.

Two-panel figure:
  Left  — Classic roofline (log-log). Kernels spread slightly on x for
           visibility; true AI ≈ 0.36 F/B for all (note in subtitle).
  Right — Wall-time bar chart with speedup annotations.

Run from repo root:
    python3 scripts/roofline.py
Saves: docs/roofline/roofline.png  +  docs/roofline/roofline_data.txt
"""

import pathlib
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from datetime import datetime

# ── sysctl helpers ────────────────────────────────────────────────────────────
def _sysctl(key: str, default: str = "") -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return default

CHIP = _sysctl("machdep.cpu.brand_string", "Apple Silicon")
NCPU = _sysctl("hw.physicalcpu", "8")

# ── Hardware (Apple M3, 8-core) ───────────────────────────────────────────────
HW = {
    "chip":       f"{CHIP} ({NCPU}-core)",
    "peak_flops": 2.6e12,   # FP64 FLOP/s (all perf cores)
    "bw_dram":    200e9,    # bytes/s  unified DRAM
    "bw_l2":      1500e9,   # bytes/s  L2
    "bw_l1":      4000e9,   # bytes/s  L1 per core (estimated)
}
RIDGE = HW["peak_flops"] / HW["bw_dram"]   # ≈ 13.0 F/B

# ── Measured kernel data (bench_profile, N=200, n_win=141) ────────────────────
#
# All kernels call evaluate_window() as their building block.
# True arithmetic intensity (FLOPs / bytes_accessed) is the same for all:
#   FLOPs per eval = 55 × 141 = 7,755
#   Bytes per eval = 19 × 141 × 8 = 21,432   (19 read passes over n_win doubles)
#   AI = 7,755 / 21,432 ≈ 0.362 F/B  → MEMORY BOUND (ridge ≈ 13 F/B)
#
# Achieved GFLOP/s differs because:
#   • Serial smart:    only ~150 evaluations (algorithmic saving)
#   • Serial exhaust:  ~8,515 evaluations
#   • OpenMP 8-thread: ~8,515 evals / 8 threads  (×5.9 speedup)
#
# X positions are SLIGHTLY SPREAD (0.24–0.55) for visual clarity.
# All are well to the left of the ridge → all memory-bound.

N_WIN           = 141
BYTES_PER_EVAL  = 19 * N_WIN * 8    # 21,432 B
FLOPS_PER_EVAL  = 55 * N_WIN        # 7,755 FLOPs
TRUE_AI         = FLOPS_PER_EVAL / BYTES_PER_EVAL   # 0.362 F/B

kernels = [
    {
        "label":    "evaluate_window\n(single call)",
        "short":    "evaluate_window",
        "n_evals":  1,
        "time_ms":  0.00463,   # 4.63 µs → ms
        "color":    "#1565C0",
        "marker":   "o",
        "ai_plot":  0.24,      # spread for visibility
    },
    {
        "label":    "find_best_window\nserial smart  (~150 evals)",
        "short":    "serial smart",
        "n_evals":  150,
        "time_ms":  0.284,
        "color":    "#2E7D32",
        "marker":   "s",
        "ai_plot":  0.33,
    },
    {
        "label":    "find_best_window\nserial exhaust  (~8.5 K evals)",
        "short":    "serial exhaust",
        "n_evals":  8515,
        "time_ms":  39.7,
        "color":    "#C62828",
        "marker":   "^",
        "ai_plot":  0.44,
    },
    {
        "label":    "find_best_window\nOpenMP  8-thread  (~8.5 K evals)",
        "short":    "OpenMP 8-thread",
        "n_evals":  8515,
        "time_ms":  6.69,
        "color":    "#6A1B9A",
        "marker":   "D",
        "ai_plot":  0.55,
    },
]

for k in kernels:
    k["flops"]  = k["n_evals"] * FLOPS_PER_EVAL
    k["bytes"]  = k["n_evals"] * BYTES_PER_EVAL
    k["ai"]     = TRUE_AI                              # true AI for all
    k["gflops"] = k["flops"] / (k["time_ms"] * 1e-3) / 1e9

# ── Figure layout: roofline (left) + bar chart (right) ───────────────────────
fig = plt.figure(figsize=(15, 7))
fig.patch.set_facecolor("#FAFAFA")
gs  = fig.add_gridspec(1, 2, width_ratios=[3, 1.6], wspace=0.35)
ax  = fig.add_subplot(gs[0])   # roofline
axb = fig.add_subplot(gs[1])   # bar chart

# ══════════════════════════════════════════════════════════════════════════════
# LEFT: Roofline model
# ══════════════════════════════════════════════════════════════════════════════
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_facecolor("#F8F9FA")

ai_x = np.logspace(-2, 2.5, 500)

# Memory bandwidth roofs
bw_lines = [
    ("L1 bandwidth  (~4 TB/s per core)",  HW["bw_l1"],  "#90CAF9", "--",  1.2),
    ("L2 bandwidth  (~1.5 TB/s)",         HW["bw_l2"],  "#42A5F5", "-.",  1.5),
    ("DRAM bandwidth  (~200 GB/s)",        HW["bw_dram"],"#1565C0", "-",   2.5),
]
for lbl, bw, col, ls, lw in bw_lines:
    roof = np.minimum(HW["peak_flops"] / 1e9, ai_x * bw / 1e9)
    ax.plot(ai_x, roof, color=col, lw=lw, ls=ls, label=lbl, alpha=0.85)

# Compute ceiling
ax.axhline(HW["peak_flops"] / 1e9, color="#B71C1C", lw=2.5, ls="-",
           label=f"Peak FP64  ({HW['peak_flops']/1e12:.1f} TFLOP/s)")

# Ridge annotation
ax.axvline(RIDGE, color="#B71C1C", lw=0.8, ls=":", alpha=0.4)
ax.text(RIDGE * 0.85, HW["peak_flops"] / 1e9 * 3,
        f"Ridge\n{RIDGE:.0f} F/B", fontsize=8.5, color="#B71C1C",
        ha="right", va="bottom")

# Memory-bound shading
ax.axvspan(1e-2, RIDGE, alpha=0.04, color="#1565C0")
ax.text(0.015, 1e-2, "MEMORY\nBOUND", fontsize=9, color="#1565C0",
        alpha=0.5, va="bottom", style="italic")

# ── Kernel points ─────────────────────────────────────────────────────────────
# ── Kernel scatter + individual annotations ───────────────────────────────────
# Each kernel gets its own carefully placed text box to avoid overlap.
# annotation_cfg: (x_text_factor, y_text_factor, ha, va, arrowstyle)
annotation_cfg = [
    # evaluate_window  → top-left of marker
    dict(xtf=0.45, ytf=6.0,  ha="center", va="bottom"),
    # serial smart     → bottom-left
    dict(xtf=0.30, ytf=0.12, ha="center", va="top"),
    # serial exhaust   → top-right
    dict(xtf=1.8,  ytf=5.0,  ha="left",   va="bottom"),
    # OMP              → bottom-right  (below ×5.9 arrow)
    dict(xtf=1.8,  ytf=0.15, ha="left",   va="top"),
]

for k, cfg in zip(kernels, annotation_cfg):
    xp, yp = k["ai_plot"], k["gflops"]

    ax.scatter(xp, yp, s=300, color=k["color"], marker=k["marker"],
               zorder=6, edgecolors="white", linewidths=1.8)

    # Dashed guide from true AI to spread x position
    ax.annotate("", xy=(xp, yp), xytext=(TRUE_AI, yp),
                arrowprops=dict(arrowstyle="-", color=k["color"],
                                lw=0.7, linestyle="dashed", alpha=0.45))

    ax.annotate(
        k["label"],
        xy=(xp, yp),
        xytext=(xp * cfg["xtf"], yp * cfg["ytf"]),
        ha=cfg["ha"], va=cfg["va"],
        fontsize=8.5, color=k["color"], fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=k["color"],
                        lw=1.1, mutation_scale=11),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=k["color"],
                  alpha=0.90, lw=1.0),
        zorder=7,
    )

# ── Speedup arrow: serial exhaust → OMP (vertical, same x) ───────────────────
k_ex  = kernels[2]
k_omp = kernels[3]
# Draw the arrow at x slightly to the right of both markers
arrow_x = max(k_ex["ai_plot"], k_omp["ai_plot"]) * 1.35
ax.annotate("",
    xy=(arrow_x, k_omp["gflops"]),
    xytext=(arrow_x, k_ex["gflops"]),
    arrowprops=dict(arrowstyle="<->", color="#37474F",
                    lw=1.8, mutation_scale=14))
speedup = k_ex["time_ms"] / k_omp["time_ms"]
ax.text(arrow_x * 1.08, np.sqrt(k_ex["gflops"] * k_omp["gflops"]),
        f"×{speedup:.1f}\nOpenMP\nspeedup",
        ha="left", va="center", fontsize=8.5, color="#37474F", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ECEFF1", ec="#37474F", lw=0.8))

# ── True AI vertical reference ────────────────────────────────────────────────
ax.axvline(TRUE_AI, color="#78909C", lw=1.0, ls=":", alpha=0.7)
ax.text(TRUE_AI, 0.012, f"  True AI\n  {TRUE_AI:.3f} F/B",
        fontsize=7.5, color="#546E7A", va="bottom")

# Decorations
ax.set_xlim(0.012, 60)
ax.set_ylim(0.01, HW["peak_flops"] / 1e9 * 5)
ax.set_xlabel("Arithmetic Intensity  [FLOPs / byte]", fontsize=11)
ax.set_ylabel("Performance  [GFLOP/s]", fontsize=11)
ax.set_title(
    f"Roofline Model — TPS HPC Engine\n"
    f"{HW['chip']}  ·  N=200  ·  n_win=141  "
    f"·  True AI≈{TRUE_AI:.2f} F/B for all kernels  (x spread for visibility)",
    fontsize=10.5
)
ax.grid(True, which="both", ls=":", alpha=0.35, color="#B0BEC5")

# Legend (bandwidth lines + compute roof only — kernels explained in right panel)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT: Wall-time bar chart — search algorithms only
# evaluate_window is a building block (1 call), not a full search;
# comparing it here would be apples-to-oranges, so it is excluded.
# ══════════════════════════════════════════════════════════════════════════════
axb.set_facecolor("#F8F9FA")

# Only the three complete search algorithms
search_kernels = [k for k in kernels if k["short"] != "evaluate_window"]
labels   = [k["short"] for k in search_kernels]
times_ms = [k["time_ms"] for k in search_kernels]
colors   = [k["color"]  for k in search_kernels]

y_pos = np.arange(len(search_kernels))
bars  = axb.barh(y_pos, times_ms, color=colors, height=0.5,
                 edgecolor="white", linewidth=1.2)

# Value labels (right of bar)
for bar, t in zip(bars, times_ms):
    txt = f"{t:.3f} ms" if t >= 0.1 else f"{t*1000:.1f} µs"
    axb.text(bar.get_width() * 1.06, bar.get_y() + bar.get_height() / 2,
             txt, va="center", ha="left", fontsize=9, fontweight="bold")

# Speedup vs serial exhaustive (baseline = middle bar)
t_base = next(k["time_ms"] for k in search_kernels if k["short"] == "serial exhaust")
for i, k in enumerate(search_kernels):
    ratio = t_base / k["time_ms"]
    if abs(ratio - 1.0) < 0.05:
        tag = "baseline"
        tag_color = "#555555"
    elif ratio > 1:
        tag = f"×{ratio:.0f} faster"
        tag_color = k["color"]
    else:
        tag = f"×{1/ratio:.1f} slower"
        tag_color = "#888888"
    axb.text(-max(times_ms) * 0.05, i, tag,
             va="center", ha="right", fontsize=8, fontweight="bold",
             color=tag_color)

axb.set_yticks(y_pos)
axb.set_yticklabels(labels, fontsize=9.5)
axb.set_xlabel("Wall time  [ms]", fontsize=10)
axb.set_title("Full search algorithm comparison\n(same task — find the best window)",
              fontsize=10)
axb.set_xscale("log")
axb.set_xlim(right=max(times_ms) * 8)
axb.invert_yaxis()
axb.grid(True, which="both", axis="x", ls=":", alpha=0.4, color="#B0BEC5")
axb.spines["top"].set_visible(False)
axb.spines["right"].set_visible(False)

# Footnote explaining evaluate_window position on roofline
axb.text(0.5, -0.13,
         "Note: evaluate_window (4.6 µs) is a single-call building block\n"
         "shown on the roofline only — not a complete search algorithm.",
         transform=axb.transAxes, ha="center", va="top",
         fontsize=7.5, color="#666", style="italic")


plt.suptitle("TPS HPC Engine — Performance Analysis", fontsize=13,
             fontweight="bold", y=1.01)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = pathlib.Path("docs/roofline")
out_dir.mkdir(parents=True, exist_ok=True)
png_path = out_dir / "roofline.png"
fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
plt.close(fig)
print(f"Chart saved -> {png_path}")

# ── Text data file ─────────────────────────────────────────────────────────────
lines = [
    "=" * 65,
    "  TPS HPC Engine — Roofline Model Data",
    f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "=" * 65,
    f"  Chip          : {HW['chip']}",
    f"  Peak FP64     : {HW['peak_flops']/1e12:.1f} TFLOP/s",
    f"  DRAM BW       : {HW['bw_dram']/1e9:.0f} GB/s",
    f"  Ridge point   : {RIDGE:.1f} FLOPs/byte",
    f"  True AI       : {TRUE_AI:.3f} FLOPs/byte  (all kernels — MEMORY BOUND)",
    "",
    f"  {'Kernel':<32} {'Time(ms)':>10} {'GFLOP/s':>10} {'vs exhaust':>12}",
    "  " + "-" * 66,
]
t_base = kernels[2]["time_ms"]
for k in kernels:
    ratio = t_base / k["time_ms"]
    lines.append(
        f"  {k['short']:<32} {k['time_ms']:>10.3f} {k['gflops']:>10.4f}"
        f"  {'baseline' if abs(ratio-1)<0.01 else f'x{ratio:.1f}'}"
    )
lines += [
    "",
    "  Note: GFLOP/s values appear low because N=200 is a small problem.",
    "  The HPC value lies in the parallelisation strategies, not raw GFLOP/s.",
    "=" * 65,
]
pathlib.Path(out_dir / "roofline_data.txt").write_text("\n".join(lines) + "\n")
print(f"Data   saved -> {out_dir / 'roofline_data.txt'}")

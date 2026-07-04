// bench_profile.cpp
//
// Fine-grained per-function timing profiler.
// Measures each sub-routine of evaluate_window() independently so we can
// identify which statistical test dominates wall-time.
//
// Output: docs/profiling/function_timings.txt
//
// Build & run:
//   cmake .. -DENABLE_PROFILING=ON -DENABLE_OPENMP=ON && make bench_profile
//   ./bench_profile [N_data] [N_reps] [output_file]

#include <tps/residual.hpp>
#include <tps/types.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;
using Us    = std::chrono::duration<double, std::micro>;

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic data generator (mirrors serial/main.cpp)
// ─────────────────────────────────────────────────────────────────────────────
static void make_data(int n, std::vector<double>& x, std::vector<double>& y)
{
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.02);
    x.resize(n); y.resize(n);
    for (int i = 0; i < n; ++i) {
        x[i] = std::sqrt(static_cast<double>(i + 1));
        y[i] = 1.5 * x[i] + 0.8 + noise(rng);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark helper: run fn() N_reps times → return {mean_us, min_us, max_us}
// ─────────────────────────────────────────────────────────────────────────────
struct Timing { double mean_us, min_us, max_us, total_ms; };

template<typename Fn>
static Timing benchmark(Fn&& fn, int n_reps)
{
    std::vector<double> samples(n_reps);
    for (int i = 0; i < n_reps; ++i) {
        auto t0 = Clock::now();
        fn();
        samples[i] = Us(Clock::now() - t0).count();
    }
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    double mn  = *std::min_element(samples.begin(), samples.end());
    double mx  = *std::max_element(samples.begin(), samples.end());
    return {sum / n_reps, mn, mx, sum / 1000.0};
}

// ─────────────────────────────────────────────────────────────────────────────
static void print_row(std::ostream& os,
                      const std::string& name,
                      const Timing& t,
                      int n_data)
{
    // Estimate FLOPs per call (rough, see comments in roofline.py)
    // linear_fit: ~8n, runs_test: ~5n, durbin_watson: ~3n,
    // spearman: ~14n log n, pearson: ~8n, evaluate_window: all of the above
    static const std::map<std::string, double> flop_factor = {
        {"linear_fit",      8.0},
        {"runs_test",       5.0},
        {"durbin_watson",   3.0},
        {"spearman_corr",  14.0 /* + sort */},
        {"pearson_corr",    8.0},
        {"evaluate_window", 55.0},
        {"find_best_window_serial",   55.0 * 150},  // ~150 evaluations
        {"find_best_window_exhaustive", 55.0 * 16110}, // n*(n-1)/2 for n=180
        {"find_best_window_omp",      55.0 * 16110},
    };
    auto it = flop_factor.find(name);
    double flops = (it != flop_factor.end()) ? it->second * n_data : 0.0;
    double gflops = (t.mean_us > 0 && flops > 0)
                    ? flops / (t.mean_us * 1e-6) / 1e9 : 0.0;

    os << "  " << std::left  << std::setw(34) << name
       << std::right
       << std::setw(9)  << std::fixed << std::setprecision(2) << t.mean_us << " us"
       << std::setw(9)  << t.min_us  << " us"
       << std::setw(9)  << t.max_us  << " us";
    if (gflops > 0)
        os << "   " << std::setprecision(4) << gflops << " GFLOP/s";
    os << '\n';
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    int         N_data  = (argc > 1) ? std::atoi(argv[1]) : 200;
    int         N_reps  = (argc > 2) ? std::atoi(argv[2]) : 2000;
    std::string outfile = (argc > 3) ? argv[3] : "docs/profiling/function_timings.txt";

    std::vector<double> x, y;
    make_data(N_data, x, y);

    // Window for single-function benchmarks
    int ws = 10, we = std::min(N_data - 1, 150);
    std::vector<double> xw(x.begin()+ws, x.begin()+we+1);
    std::vector<double> yw(y.begin()+ws, y.begin()+we+1);
    int n_win = we - ws + 1;

    // Pre-compute residuals for stat-test benchmarks
    auto fit   = tps::linear_fit(xw, yw);
    auto& resid = fit.residuals;

    // ── Benchmark each primitive ─────────────────────────────────────────────
    auto t_linfit   = benchmark([&]{ tps::linear_fit(xw, yw); },                N_reps);
    auto t_runs     = benchmark([&]{ tps::runs_test(resid); },                   N_reps);
    auto t_dw       = benchmark([&]{ tps::durbin_watson(resid); },               N_reps);
    auto t_spear    = benchmark([&]{ tps::spearman_corr(xw, yw); },              N_reps);
    auto t_pears    = benchmark([&]{ tps::pearson_corr(resid, xw); },            N_reps);
    auto t_evalwin  = benchmark([&]{ tps::evaluate_window(x, y, ws, we); },      N_reps);
    auto t_serial   = benchmark([&]{ tps::find_best_window_serial(x, y); },      N_reps / 20);
    auto t_exhaust  = benchmark([&]{ tps::find_best_window_exhaustive(x, y); },  N_reps / 100);
    auto t_omp      = benchmark([&]{ tps::find_best_window_omp(x, y); },         N_reps / 100);

    // ── Format and output ────────────────────────────────────────────────────
    auto report = [&](std::ostream& os) {
        os << "========================================================\n";
        os << "  TPS HPC Engine — Per-function timing profile\n";
        os << "  N_data=" << N_data << "  win=[" << ws << "," << we << "] n=" << n_win << "\n";
        os << "  reps=" << N_reps << "  (search reps: " << N_reps/20 << " / " << N_reps/100 << ")\n";
        os << "========================================================\n";
        os << "  " << std::left << std::setw(34) << "Function"
           << std::right << std::setw(11) << "mean"
           << std::setw(11) << "min"
           << std::setw(11) << "max\n";
        os << "  " << std::string(75, '-') << '\n';

        print_row(os, "linear_fit",             t_linfit,  n_win);
        print_row(os, "runs_test",              t_runs,    n_win);
        print_row(os, "durbin_watson",          t_dw,      n_win);
        print_row(os, "spearman_corr",          t_spear,   n_win);
        print_row(os, "pearson_corr",           t_pears,   n_win);
        os << "  " << std::string(75, '-') << '\n';
        print_row(os, "evaluate_window",        t_evalwin, n_win);
        os << "  " << std::string(75, '-') << '\n';
        print_row(os, "find_best_window_serial",   t_serial,  N_data);
        print_row(os, "find_best_window_exhaustive", t_exhaust, N_data);
        print_row(os, "find_best_window_omp",       t_omp,     N_data);
        os << "========================================================\n";

        // Speedup summary
        os << "\n  Speedup summary\n";
        os << "  Serial smart vs exhaustive : x"
           << std::setprecision(1) << t_exhaust.mean_us / t_serial.mean_us << "\n";
        os << "  OpenMP vs serial exhaustive: x"
           << t_exhaust.mean_us / t_omp.mean_us << "\n";

        // Hot-path breakdown (% of evaluate_window)
        double ev = t_evalwin.mean_us;
        os << "\n  evaluate_window breakdown (% of total)\n";
        os << "  linear_fit   : " << std::setprecision(1) << 100.0*t_linfit.mean_us/ev  << " %\n";
        os << "  runs_test    : " << 100.0*t_runs.mean_us/ev    << " %\n";
        os << "  durbin_watson: " << 100.0*t_dw.mean_us/ev      << " %\n";
        os << "  spearman_corr: " << 100.0*t_spear.mean_us/ev   << " %\n";
        os << "  pearson_corr : " << 100.0*t_pears.mean_us/ev   << " %\n";
        os << "  overhead     : "
           << 100.0*(ev - t_linfit.mean_us - t_runs.mean_us - t_dw.mean_us
                        - t_spear.mean_us - t_pears.mean_us) / ev << " %\n";
    };

    report(std::cout);

    // Write to file (try outfile path; if fails, try relative from repo root)
    std::ofstream f(outfile);
    if (!f.is_open()) {
        f.open("../../" + outfile);
    }
    if (f.is_open()) {
        report(f);
        std::cout << "\nSaved → " << outfile << '\n';
    } else {
        std::cerr << "Warning: could not write " << outfile << '\n';
    }

    return 0;
}

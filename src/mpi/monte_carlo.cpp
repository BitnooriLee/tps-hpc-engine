// monte_carlo.cpp
//
// MPI Monte Carlo Uncertainty Quantification for TPS thermal analysis.
//
// HPC patterns used:
//   • Each MPI rank runs an independent subset of N_total trials  (data-parallel MC)
//   • Hybrid: optional OpenMP inner loop (one thread per trial on the node)
//   • Rank 0 gathers and aggregates via MPI_Gatherv

#include <tps/monte_carlo.hpp>
#include <tps/residual.hpp>
#include <tps/optimization.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace tps {
namespace uq {

// ─────────────────────────────────────────────────────────────────────────────
// run_mc_local
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> run_mc_local(
    const std::vector<double>& x,
    const std::vector<double>& y,
    int    n_local,
    double noise_sigma,
    double power,
    double sensor_radius,
    double f_tau,
    unsigned seed)
{
    // Pre-allocate worst-case; actual size ≤ n_local
    std::vector<double> k_samples;
    k_samples.reserve(n_local);

#ifdef _OPENMP
    // ── Hybrid: use OpenMP threads within this MPI rank ──────────────────────
    // Each thread needs its own RNG seeded differently to avoid correlations.
    int n_threads = omp_get_max_threads();
    std::vector<std::mt19937> rngs(n_threads);
    for (int t = 0; t < n_threads; ++t)
        rngs[t].seed(seed + static_cast<unsigned>(t) * 999983u);

    // Thread-local result vectors, merged after the loop
    std::vector<std::vector<double>> thread_k(n_threads);
    for (auto& v : thread_k) v.reserve(n_local / n_threads + 1);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n_local; ++i) {
        int tid = omp_get_thread_num();
        std::mt19937& rng = rngs[tid];
        std::normal_distribution<double> noise_dist(0.0, noise_sigma);

        // Perturb ΔT measurements
        std::vector<double> y_noisy(y.size());
        for (std::size_t j = 0; j < y.size(); ++j)
            y_noisy[j] = y[j] + noise_dist(rng);

        // Find best linear window in perturbed data
        auto bwr = find_best_window_serial(x, y_noisy);
        if (bwr.result.n < 10) continue;

        // Linear fit within best window
        int s = bwr.start, e = bwr.end;
        std::vector<double> xw(x.begin() + s, x.begin() + e + 1);
        std::vector<double> yw(y_noisy.begin() + s, y_noisy.begin() + e + 1);
        LinearFitResult fit = linear_fit(xw, yw);

        // slope = dΔT / d(sqrt(t)); k_proxy = P / (π^1.5 × r × slope × f_τ)
        if (fit.slope <= 0.0) continue;
        double k = estimate_conductivity(power, fit.slope, sensor_radius, f_tau);
        thread_k[tid].push_back(k);
    }

    // Merge per-thread vectors
    for (auto& v : thread_k)
        k_samples.insert(k_samples.end(), v.begin(), v.end());

#else
    // ── Serial fallback (no OpenMP) ───────────────────────────────────────────
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise_dist(0.0, noise_sigma);

    for (int i = 0; i < n_local; ++i) {
        std::vector<double> y_noisy(y.size());
        for (std::size_t j = 0; j < y.size(); ++j)
            y_noisy[j] = y[j] + noise_dist(rng);

        auto bwr = find_best_window_serial(x, y_noisy);
        if (bwr.result.n < 10) continue;

        int s = bwr.start, e = bwr.end;
        std::vector<double> xw(x.begin() + s, x.begin() + e + 1);
        std::vector<double> yw(y_noisy.begin() + s, y_noisy.begin() + e + 1);
        LinearFitResult fit = linear_fit(xw, yw);

        if (fit.slope <= 0.0) continue;
        double k = estimate_conductivity(power, fit.slope, sensor_radius, f_tau);
        k_samples.push_back(k);
    }
#endif

    return k_samples;
}

// ─────────────────────────────────────────────────────────────────────────────
// aggregate  (called only on rank 0, after MPI_Gatherv)
// ─────────────────────────────────────────────────────────────────────────────
UQResult aggregate(const std::vector<double>& k_samples, int n_attempted_total)
{
    UQResult res;
    res.n_attempted = n_attempted_total;
    res.n_accepted  = static_cast<int>(k_samples.size());
    if (res.n_accepted == 0) return res;

    // Mean
    double sum = std::accumulate(k_samples.begin(), k_samples.end(), 0.0);
    res.mean_k = sum / res.n_accepted;

    // Standard deviation (population, biased)
    double sq = 0.0;
    for (double k : k_samples) sq += (k - res.mean_k) * (k - res.mean_k);
    res.std_k  = std::sqrt(sq / res.n_accepted);
    res.cv_pct = (res.mean_k != 0.0) ? 100.0 * res.std_k / res.mean_k : 0.0;

    // 95 % CI on the mean (central-limit: ±1.96 × SE)
    double se  = res.std_k / std::sqrt(static_cast<double>(res.n_accepted));
    res.ci95_lo = res.mean_k - 1.96 * se;
    res.ci95_hi = res.mean_k + 1.96 * se;

    // 95 % prediction interval — 2.5th / 97.5th percentile
    std::vector<double> sorted = k_samples;
    std::sort(sorted.begin(), sorted.end());
    int lo = static_cast<int>(0.025 * sorted.size());
    int hi = static_cast<int>(0.975 * sorted.size());
    hi = std::min(hi, res.n_accepted - 1);
    res.p025 = sorted[lo];
    res.p975 = sorted[hi];

    return res;
}

}} // namespace tps::uq

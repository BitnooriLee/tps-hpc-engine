#pragma once
#include <vector>

namespace tps {
namespace uq {

// ─────────────────────────────────────────────────────────────────────────────
// Result of a Monte Carlo uncertainty quantification run
// ─────────────────────────────────────────────────────────────────────────────
struct UQResult {
    // Quantity of interest: k_proxy = P / (π^1.5 × r × slope × f_τ)
    double mean_k    = 0.0;   // mean across accepted samples
    double std_k     = 0.0;   // sample standard deviation
    double cv_pct    = 0.0;   // coefficient of variation [%]

    // 95 % confidence interval on the mean (normal approx: ±1.96 SE)
    double ci95_lo   = 0.0;
    double ci95_hi   = 0.0;

    // 95 % prediction interval (2.5th–97.5th percentile of the distribution)
    double p025      = 0.0;
    double p975      = 0.0;

    int n_attempted  = 0;     // MC trials attempted by this rank
    int n_accepted   = 0;     // trials that yielded a valid positive slope
};

// ─────────────────────────────────────────────────────────────────────────────
// Run n_local Monte Carlo trials on one MPI rank.
//
// Each trial:
//   1. Adds i.i.d. N(0, noise_sigma²) noise to y (ΔT measurements).
//   2. Calls find_best_window_serial on the perturbed data.
//   3. Re-fits a line in the best window → extracts slope.
//   4. Computes k_proxy = estimate_conductivity(power, slope, sensor_radius, f_tau).
//
// Returns a vector of accepted k_proxy values (length ≤ n_local).
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> run_mc_local(
    const std::vector<double>& x,        // sqrt(t) array
    const std::vector<double>& y,        // ΔT array
    int    n_local,                       // number of trials for this rank
    double noise_sigma,                   // Gaussian noise std dev on ΔT
    double power,                         // heating power [W]
    double sensor_radius,                 // TPS sensor radius [m]
    double f_tau,                         // dimensionless TPS correction factor
    unsigned seed                         // RNG seed (per-rank unique)
);

// ─────────────────────────────────────────────────────────────────────────────
// Aggregate all k samples (from all ranks, already gathered on rank 0).
// ─────────────────────────────────────────────────────────────────────────────
UQResult aggregate(const std::vector<double>& k_samples, int n_attempted_total);

}} // namespace tps::uq

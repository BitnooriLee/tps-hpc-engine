#pragma once
#include <tps/types.hpp>
#include <vector>

namespace tps {

// --- statistical primitives ---

struct LinearFitResult {
    double slope;
    double intercept;
    std::vector<double> residuals;
};

LinearFitResult linear_fit(const std::vector<double>& x,
                           const std::vector<double>& y);

struct RunsTestResult {
    int    runs;
    double expected;
    bool   is_random;
};
RunsTestResult runs_test(const std::vector<double>& residuals);

double durbin_watson(const std::vector<double>& residuals);
double spearman_corr(const std::vector<double>& x, const std::vector<double>& y);
double pearson_corr (const std::vector<double>& x, const std::vector<double>& y);

// --- residual evaluation ---

ResidualResult evaluate_window(const std::vector<double>& x,
                               const std::vector<double>& y,
                               int start, int end);

double score_result(const ResidualResult& r);

// --- best window search ---

// Serial: seeded local search (smart, fast — O(seeds × iterations))
BestWindowResult find_best_window_serial(const std::vector<double>& x,
                                         const std::vector<double>& y,
                                         double min_fraction      = 0.30,
                                         int    total_points_hint = -1);

// Serial: exhaustive grid search (baseline for fair OpenMP comparison)
BestWindowResult find_best_window_exhaustive(const std::vector<double>& x,
                                              const std::vector<double>& y,
                                              double min_fraction      = 0.30,
                                              int    total_points_hint = -1);

// OpenMP: exhaustive parallel grid search (global optimum)
BestWindowResult find_best_window_omp(const std::vector<double>& x,
                                       const std::vector<double>& y,
                                       double min_fraction      = 0.30,
                                       int    total_points_hint = -1);

}  // namespace tps

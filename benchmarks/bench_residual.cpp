//  bench_residual — compare serial vs OpenMP timing on synthetic data
//
//  Generates N (sqrt_t, dT) pairs with controlled noise, then runs both
//  find_best_window_serial and find_best_window_omp, printing wall-clock times.
//
//  TPS context: real measurements produce ~200 data points (the search window
//  is bounded to [10, 200] by default). Here N controls how many points we
//  generate and also acts as the total_points_hint, which determines the
//  minimum window fraction. Keep N in the range 100–500 for realistic results.
//
//  Usage:  ./bench_residual [N=200]

#include <tps/residual.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static std::pair<std::vector<double>,std::vector<double>> generate_tps_data(int n) {
    // Simulate: dT = slope * sqrt_t + intercept + noise
    std::vector<double> xs(n), ys(n);
    const double slope = 0.45, intercept = 0.01;
    for (int i = 0; i < n; ++i) {
        double t = 0.05 + i * 0.01;
        xs[i] = t;
        // small Gaussian-like noise via deterministic pseudo-random
        double noise = 0.001 * std::sin(i * 1.7 + 0.3);
        ys[i] = slope * t + intercept + noise;
    }
    return {xs, ys};
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 200;
    if (N > 500) {
        std::cerr << "Warning: N=" << N << " > 500. "
                  << "TPS measurements are ~200 pts; large N explodes search space. "
                  << "Clamping to 500.\n";
        N = 500;
    }

    auto [x, y] = generate_tps_data(N);
    std::cout << "Benchmark: N=" << N << " points\n";

#ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#endif

    // Warm-up
    tps::find_best_window_serial(x, y);

    const int REPS = 5;
    double smart_total = 0, exhaustive_total = 0, omp_total = 0;

    // Warm-up
    tps::find_best_window_serial(x, y);
    tps::find_best_window_exhaustive(x, y);
    tps::find_best_window_omp(x, y);

    for (int r = 0; r < REPS; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        tps::find_best_window_serial(x, y);
        auto t1 = std::chrono::high_resolution_clock::now();
        smart_total += std::chrono::duration<double, std::micro>(t1 - t0).count();

        auto t2 = std::chrono::high_resolution_clock::now();
        tps::find_best_window_exhaustive(x, y);
        auto t3 = std::chrono::high_resolution_clock::now();
        exhaustive_total += std::chrono::duration<double, std::micro>(t3 - t2).count();

        auto t4 = std::chrono::high_resolution_clock::now();
        tps::find_best_window_omp(x, y);
        auto t5 = std::chrono::high_resolution_clock::now();
        omp_total += std::chrono::duration<double, std::micro>(t5 - t4).count();
    }

    double smart_avg     = smart_total     / REPS;
    double exhaustive_avg = exhaustive_total / REPS;
    double omp_avg       = omp_total       / REPS;

    std::cout << "\nResults (avg over " << REPS << " runs):\n";
    std::cout << "  Serial smart (local search) : " << smart_avg      << " µs\n";
    std::cout << "  Serial exhaustive           : " << exhaustive_avg << " µs\n";
    std::cout << "  OpenMP exhaustive           : " << omp_avg        << " µs\n";
    std::cout << "\n  [Fair comparison] Exhaustive serial vs OpenMP:\n";
    if (omp_avg > 0)
        std::cout << "  Speedup (exhaustive → OMP)  : " << exhaustive_avg / omp_avg << "x\n";
    std::cout << "\n  [Algorithm advantage] Smart serial vs exhaustive serial:\n";
    if (exhaustive_avg > 0)
        std::cout << "  Smart search advantage      : " << exhaustive_avg / smart_avg << "x faster\n";

    return 0;
}

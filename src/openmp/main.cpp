//  tps_openmp — exhaustive parallel window search
//
//  Generates synthetic TPS data in memory, then runs serial and OpenMP
//  back-to-back and prints speedup.
//
//  Usage:  OMP_NUM_THREADS=8 ./tps_openmp [--n 200]

#include <tps/residual.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static std::pair<std::vector<double>, std::vector<double>>
generate_tps_data(int n) {
    std::vector<double> xs(n), ys(n);
    for (int i = 0; i < n; ++i) {
        double t = 0.05 + i * 0.01;
        xs[i] = t;
        ys[i] = 0.45 * t + 0.01 + 0.001 * std::sin(i * 1.7 + 0.3);
    }
    return {xs, ys};
}

static void print_result(const char* label, const tps::BestWindowResult& bw, double us) {
    const auto& r = bw.result;
    std::cout << "\n--- " << label << " ---\n";
    std::cout << "Best window : [" << bw.start << ", " << bw.end
              << "]  (n=" << r.n << ")\n";
    std::cout << "Verdict     : " << r.verdict << "\n";
    std::cout << "RMSE        : " << r.rmse << "\n";
    std::cout << "Elapsed     : " << us << " µs\n";
}

int main(int argc, char** argv) {
    int    n            = 200;
    double min_fraction = 0.30;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--n" && i+1 < argc)            n            = std::stoi(argv[++i]);
        if (arg == "--min-fraction" && i+1 < argc) min_fraction = std::stod(argv[++i]);
    }

    auto [x, y] = generate_tps_data(n);
    std::cout << "Generated " << n << " synthetic TPS points (in-memory)\n";

#ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP: NOT enabled (running serial fallback)\n";
#endif

    // Serial baseline
    auto t0 = std::chrono::high_resolution_clock::now();
    auto serial_bw = tps::find_best_window_serial(x, y, min_fraction);
    auto t1 = std::chrono::high_resolution_clock::now();
    double serial_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // OpenMP exhaustive
    auto t2 = std::chrono::high_resolution_clock::now();
    auto omp_bw = tps::find_best_window_omp(x, y, min_fraction);
    auto t3 = std::chrono::high_resolution_clock::now();
    double omp_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

    print_result("Serial (seeded local search)", serial_bw, serial_us);
    print_result("OpenMP (exhaustive grid search)", omp_bw, omp_us);

    std::cout << "\n--- Comparison ---\n";
    std::cout << "Serial      : " << serial_us  << " µs\n";
    std::cout << "OpenMP      : " << omp_us     << " µs\n";
    if (omp_us > 0)
        std::cout << "Speedup     : " << serial_us / omp_us << "x\n";

    return 0;
}

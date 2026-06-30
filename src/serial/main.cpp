//  tps_serial — serial baseline
//
//  Generates synthetic TPS (sqrt_t, dT) data in memory and runs the
//  best-window search. No file I/O — mirrors the real use case where
//  Python passes float arrays directly via pybind11 after parsing the
//  EXP:RES? TCP response from Hot Disk Software.
//
//  Usage:  ./tps_serial [--n 200] [--min-fraction 0.30]

#include <tps/residual.hpp>
#include <tps/optimization.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>


static std::pair<std::vector<double>, std::vector<double>>
generate_tps_data(int n) {
    // Simulate: dT ≈ slope × sqrt_t + intercept + small noise
    // Matches the linear model Hot Disk fits in the TPS method.
    std::vector<double> xs(n), ys(n);
    for (int i = 0; i < n; ++i) {
        double t   = 0.05 + i * 0.01;
        double noise = 0.001 * std::sin(i * 1.7 + 0.3);
        xs[i] = t;
        ys[i] = 0.45 * t + 0.01 + noise;
    }
    return {xs, ys};
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
    std::cout << "Generated " << n << " synthetic TPS points (in-memory)\n"
              << "(Real use: Python parses EXP:RES? response → passes arrays via pybind11)\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    tps::BestWindowResult bw = tps::find_best_window_serial(x, y, min_fraction);

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    const auto& r = bw.result;
    std::cout << "\n--- Serial Result ---\n";
    std::cout << "Best window : [" << bw.start << ", " << bw.end << "]"
              << "  (n=" << r.n << ")\n";
    std::cout << "Verdict     : " << r.verdict << "\n";
    std::cout << "RMSE        : " << r.rmse << "\n";
    std::cout << "Durbin-Watson: " << r.durbin_watson
              << "  (" << (r.dw_ok ? "ok" : "FAIL") << ")\n";
    std::cout << "Runs test   : runs=" << r.runs
              << "  expected=" << r.runs_expected
              << "  (" << (r.runs_ok ? "ok" : "FAIL") << ")\n";
    if (!r.issues.empty()) {
        std::cout << "Issues      : ";
        for (auto& s : r.issues) std::cout << s << "; ";
        std::cout << "\n";
    }
    if (!r.selection_note.empty())
        std::cout << "Note        : " << r.selection_note << "\n";

    std::cout << "\nElapsed     : " << elapsed_us << " µs\n";
    return 0;
}

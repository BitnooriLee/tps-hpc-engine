//  tps_openmp — exhaustive parallel window search
//
//  Usage:
//    OMP_NUM_THREADS=8 ./tps_openmp --input data/sample.csv
//
//  Runs both serial and OpenMP back-to-back and prints speedup.

#include <tps/residual.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static std::pair<std::vector<double>, std::vector<double>>
load_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<double> xs, ys;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line);
        double x, y;
        if (ss >> x >> y) { xs.push_back(x); ys.push_back(y); }
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
    std::string input_path;
    double min_fraction = 0.30;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if ((arg == "--input" || arg == "-i") && i+1 < argc) input_path = argv[++i];
        else if (arg == "--min-fraction" && i+1 < argc) min_fraction = std::stod(argv[++i]);
    }
    if (input_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --input <file.csv>\n";
        return 1;
    }

    auto [x, y] = load_csv(input_path);
    std::cout << "Loaded " << x.size() << " points from " << input_path << "\n";

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

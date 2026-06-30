//  tps_serial — serial baseline
//
//  Usage:
//    ./tps_serial --input data/sample.csv
//    ./tps_serial --input data/sample.csv --min-fraction 0.3
//
//  CSV format (no header):  sqrt_t,dT
//    0.123,0.045
//    0.234,0.089
//    ...
//
//  Output: best window [start, end], verdict, RMSE, wall-clock time (µs)

#include <tps/residual.hpp>
#include <tps/optimization.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " --input <file.csv> [--min-fraction 0.30]\n";
}

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

int main(int argc, char** argv) {
    std::string input_path;
    double min_fraction = 0.30;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if ((arg == "--input" || arg == "-i") && i+1 < argc)
            input_path = argv[++i];
        else if (arg == "--min-fraction" && i+1 < argc)
            min_fraction = std::stod(argv[++i]);
        else if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
    }

    if (input_path.empty()) { usage(argv[0]); return 1; }

    std::vector<double> x, y;
    try {
        auto [xx, yy] = load_csv(input_path);
        x = xx; y = yy;
    } catch (const std::exception& e) {
        std::cerr << "Error loading CSV: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Loaded " << x.size() << " points from " << input_path << "\n";

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

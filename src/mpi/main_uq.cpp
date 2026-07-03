// main_uq.cpp
//
// MPI driver for TPS Monte Carlo Uncertainty Quantification.
//
// Usage:
//   mpirun -n <ranks> ./tps_mpi_uq [data_file] [n_total] [noise_sigma]
//
// HPC workflow:
//   Rank 0: read data file → MPI_Bcast
//   All:    run_mc_local (each rank works on n_total/size trials)
//   Rank 0: MPI_Gatherv → aggregate → print UQ report

#include <tps/monte_carlo.hpp>
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ── TPS sensor / experiment parameters (override via build flags if needed) ──
static constexpr double POWER         = 0.05;      // heating power [W]
static constexpr double SENSOR_RADIUS = 0.006227;  // sensor radius [m]
static constexpr double F_TAU         = 1.0;       // dimensionless TPS factor

// ─────────────────────────────────────────────────────────────────────────────
static bool load_data(const std::string& path,
                      std::vector<double>& x,
                      std::vector<double>& y)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;
    double a, b;
    while (f >> a >> b) { x.push_back(a); y.push_back(b); }
    return !x.empty();
}

// ─────────────────────────────────────────────────────────────────────────────
// Repeat a single ASCII character to form a separator line.
static void print_separator(char c = '-', int w = 60) {
    std::cout << std::string(w, c) << '\n';
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ── Parse command-line args ──────────────────────────────────────────────
    std::string data_file  = (argc > 1) ? argv[1] : "data/example_res_response.txt";
    int         n_total    = (argc > 2) ? std::atoi(argv[2]) : 10000;
    double      noise_sigma = (argc > 3) ? std::atof(argv[3]) : 0.02;

    if (n_total < size) n_total = size;  // at least one trial per rank

    // ── Rank 0 reads data, broadcasts to all ranks ───────────────────────────
    std::vector<double> x, y;
    int data_size = 0;

    if (rank == 0) {
        if (!load_data(data_file, x, y)) {
            // Fallback: look one level up (when running from build/)
            if (!load_data("../" + data_file, x, y)) {
                std::cerr << "[Rank 0] ERROR: cannot open data file: "
                          << data_file << '\n';
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        data_size = static_cast<int>(x.size());
    }

    MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) { x.resize(data_size); y.resize(data_size); }
    MPI_Bcast(x.data(), data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y.data(), data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ── Print header (rank 0 only) ───────────────────────────────────────────
    if (rank == 0) {
        print_separator('=');
        std::cout << "  TPS Monte Carlo Uncertainty Quantification\n";
        print_separator('=');
        std::cout << "  Data file   : " << data_file
                  << "  (" << data_size << " points)\n"
                  << "  MPI ranks   : " << size << '\n'
                  << "  MC trials   : " << n_total << "  total  ("
                  << (n_total / size) << "–" << (n_total / size + 1) << " per rank)\n"
                  << "  Noise σ     : " << noise_sigma << " K\n"
                  << "  Power       : " << POWER << " W\n"
                  << "  Sensor r    : " << SENSOR_RADIUS * 1e3 << " mm\n";
        print_separator();
        std::cout << std::flush;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ── Each rank runs its share of MC trials ────────────────────────────────
    // Distribute trials as evenly as possible across ranks.
    int n_local = n_total / size + (rank < (n_total % size) ? 1 : 0);
    unsigned seed = 42u + static_cast<unsigned>(rank) * 999983u;

    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<double> local_k = tps::uq::run_mc_local(
        x, y, n_local, noise_sigma, POWER, SENSOR_RADIUS, F_TAU, seed);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    int local_accepted = static_cast<int>(local_k.size());

    // ── Print per-rank timing ────────────────────────────────────────────────
    for (int r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "  Rank " << std::setw(2) << rank
                      << "  trials=" << std::setw(6) << n_local
                      << "  accepted=" << std::setw(6) << local_accepted
                      << "  time=" << std::fixed << std::setprecision(1)
                      << elapsed_ms << " ms\n" << std::flush;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ── Gather sample counts, then all k values to rank 0 ───────────────────
    std::vector<int> counts(size, 0);
    MPI_Gather(&local_accepted, 1, MPI_INT,
               counts.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    std::vector<int> displs(size, 0);
    int total_accepted = 0;
    std::vector<double> all_k;

    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            displs[i]    = total_accepted;
            total_accepted += counts[i];
        }
        all_k.resize(total_accepted);
    }

    MPI_Gatherv(local_k.data(), local_accepted, MPI_DOUBLE,
                rank == 0 ? all_k.data()    : nullptr,
                rank == 0 ? counts.data()   : nullptr,
                rank == 0 ? displs.data()   : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ── Gather total-attempted counts ────────────────────────────────────────
    int total_attempted = 0;
    MPI_Reduce(&n_local, &total_attempted, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // ── Rank 0 aggregates and prints the UQ report ───────────────────────────
    if (rank == 0) {
        print_separator();
        tps::uq::UQResult res = tps::uq::aggregate(all_k, total_attempted);

        if (res.n_accepted == 0) {
            std::cout << "  WARNING: no valid samples — check data and noise_sigma.\n";
        } else {
            double accept_pct = 100.0 * res.n_accepted / res.n_attempted;
            std::cout << "\n  ── UQ Results ──────────────────────────────────────\n";
            std::cout << "  Acceptance rate : " << res.n_accepted << " / "
                      << res.n_attempted << "  ("
                      << std::fixed << std::setprecision(1) << accept_pct << " %)\n\n";

            std::cout << std::setprecision(6);
            std::cout << "  k_proxy mean    : " << res.mean_k    << "  [a.u.]\n";
            std::cout << "  k_proxy std     : " << res.std_k     << "  [a.u.]\n";
            std::cout << "  CV              : " << std::setprecision(2)
                                                << res.cv_pct    << " %\n\n";

            std::cout << std::setprecision(6);
            std::cout << "  95 % CI (mean)  : ["
                      << res.ci95_lo << ", " << res.ci95_hi << "]\n";
            std::cout << "  95 % PI (dist)  : ["
                      << res.p025    << ", " << res.p975    << "]\n";
        }
        print_separator('=');
    }

    MPI_Finalize();
    return 0;
}

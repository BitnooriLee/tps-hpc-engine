//  residual_omp.cpp
//
//  OpenMP version: exhaustive parallel window search.
//
//  Strategy vs serial:
//    Serial  — seeded local search, O(seeds × iterations × 6) evaluations
//    OpenMP  — full grid search over ALL (start, end) windows in parallel
//              → finds the global optimum, not just a local one
//              → each window evaluation is independent → embarrassingly parallel
//
//  The key HPC pattern here:
//    - outer loop over all (s, e) pairs partitioned with `schedule(dynamic)`
//    - thread-private best candidate, then reduction across threads

#include <tps/residual.hpp>
#include <omp.h>
#include <algorithm>
#include <climits>
#include <cmath>

namespace tps {

BestWindowResult find_best_window_omp(const std::vector<double>& x,
                                       const std::vector<double>& y,
                                       double min_fraction,
                                       int    total_points_hint) {
    BestWindowResult out;
    int n = static_cast<int>(x.size());

    if (n < 10 || n != static_cast<int>(y.size())) {
        out.start = 0; out.end = std::max(0, n-1);
        out.result.verdict = "unknown";
        out.result.reason  = "insufficient data";
        out.result.n       = n;
        return out;
    }

    int total_points   = std::max(n, total_points_hint < 0 ? n : total_points_hint);
    int strict_min_pts = std::max(10, static_cast<int>(std::floor(min_fraction * total_points)) + 1);
    int search_start   = std::min(10, n - 1);
    int search_end     = std::min(200, n - 1);
    if (search_end < search_start) { search_start = 0; search_end = n - 1; }
    int search_len = search_end - search_start + 1;
    if (search_len < strict_min_pts) {
        search_start = 0; search_end = n - 1;
        strict_min_pts = std::min(strict_min_pts, n);
    }

    // Enumerate all valid (s, e) pairs into a flat list for parallel dispatch
    std::vector<std::pair<int,int>> windows;
    windows.reserve(search_len * search_len / 2);
    for (int s = search_start; s <= search_end; ++s)
        for (int e = s + strict_min_pts - 1; e <= search_end; ++e)
            windows.push_back({s, e});

    const int nw = static_cast<int>(windows.size());

    // Thread-private best: store (score, start, end, result) per thread
    double        global_best_score = -1e9;
    int           global_best_s = search_start, global_best_e = search_end;
    ResidualResult global_best_result;

    #pragma omp parallel
    {
        double         local_best_score = -1e9;
        int            local_best_s = search_start, local_best_e = search_end;
        ResidualResult local_best_result;

        #pragma omp for schedule(dynamic, 32) nowait
        for (int i = 0; i < nw; ++i) {
            auto [s, e] = windows[i];
            ResidualResult r = evaluate_window(x, y, s, e);
            double sc = score_result(r);
            if (sc > local_best_score) {
                local_best_score  = sc;
                local_best_s      = s;
                local_best_e      = e;
                local_best_result = r;
            }
        }

        // Merge thread-private result into global (critical section)
        #pragma omp critical
        {
            if (local_best_score > global_best_score) {
                global_best_score  = local_best_score;
                global_best_s      = local_best_s;
                global_best_e      = local_best_e;
                global_best_result = local_best_result;
            }
        }
    }  // end parallel

    if (global_best_result.verdict.empty()) {
        global_best_result = evaluate_window(x, y, search_start, search_end);
        global_best_s = search_start; global_best_e = search_end;
    }
    if (global_best_result.verdict != "good")
        global_best_result.selection_note = "No good window found; returning least-bad window.";

    out.start  = global_best_s;
    out.end    = global_best_e;
    out.result = global_best_result;
    return out;
}

}  // namespace tps

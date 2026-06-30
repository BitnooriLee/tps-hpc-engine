#include <tps/residual.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace tps {

// ─────────────────────────────────────────────────────────────────────────────
// Constants (mirrors Python thresholds)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr double RUNS_Z_THRESHOLD  = 2.4;
static constexpr double DW_LOW            = 1.2;
static constexpr double DW_HIGH           = 2.8;
static constexpr double CORR_ABS_THRESH   = 0.35;
static constexpr double TREND_CORR_THRESH = 0.35;

// ─────────────────────────────────────────────────────────────────────────────
// linear_fit  (mirrors numpy.polyfit(x, y, 1))
// ─────────────────────────────────────────────────────────────────────────────
LinearFitResult linear_fit(const std::vector<double>& x,
                           const std::vector<double>& y) {
    LinearFitResult out{0.0, 0.0, {}};
    int n = static_cast<int>(x.size());
    if (n < 2 || n != static_cast<int>(y.size())) return out;

    double sum_x  = 0, sum_y  = 0, sum_xx = 0, sum_xy = 0;
    for (int i = 0; i < n; ++i) {
        sum_x  += x[i];
        sum_y  += y[i];
        sum_xx += x[i] * x[i];
        sum_xy += x[i] * y[i];
    }
    double denom = n * sum_xx - sum_x * sum_x;
    if (denom == 0.0) return out;

    out.slope     = (n * sum_xy - sum_x * sum_y) / denom;
    out.intercept = (sum_y - out.slope * sum_x) / n;

    out.residuals.resize(n);
    for (int i = 0; i < n; ++i)
        out.residuals[i] = y[i] - (out.slope * x[i] + out.intercept);

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// runs_test
// ─────────────────────────────────────────────────────────────────────────────
RunsTestResult runs_test(const std::vector<double>& residuals) {
    int n = static_cast<int>(residuals.size());
    RunsTestResult out{0, 0.0, true};
    if (n < 10) return out;

    std::vector<double> sorted_r = residuals;
    std::sort(sorted_r.begin(), sorted_r.end());
    double median = (n % 2 == 0)
        ? (sorted_r[n/2 - 1] + sorted_r[n/2]) / 2.0
        : sorted_r[n/2];

    std::vector<int> signs(n);
    for (int i = 0; i < n; ++i)
        signs[i] = (residuals[i] >= median) ? 1 : 0;

    int runs = 1;
    for (int i = 1; i < n; ++i)
        if (signs[i] != signs[i-1]) ++runs;

    int n1 = std::accumulate(signs.begin(), signs.end(), 0);
    int n2 = n - n1;

    if (n1 == 0 || n2 == 0) { out.runs = runs; out.is_random = false; return out; }

    double expected = 2.0 * n1 * n2 / n + 1.0;
    double var_runs = 2.0 * n1 * n2 * (2.0*n1*n2 - n) / ((double)n * n * (n - 1));
    double z = (var_runs > 0) ? (runs - expected) / std::sqrt(var_runs) : 0.0;

    out.runs       = runs;
    out.expected   = expected;
    out.is_random  = std::abs(z) < RUNS_Z_THRESHOLD;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// durbin_watson
// ─────────────────────────────────────────────────────────────────────────────
double durbin_watson(const std::vector<double>& r) {
    int n = static_cast<int>(r.size());
    if (n < 3) return 2.0;

    double res_sq = 0.0;
    for (double v : r) res_sq += v * v;
    if (res_sq == 0.0) return 2.0;

    double diff_sq = 0.0;
    for (int i = 1; i < n; ++i) diff_sq += (r[i] - r[i-1]) * (r[i] - r[i-1]);
    return diff_sq / res_sq;
}

// ─────────────────────────────────────────────────────────────────────────────
// spearman_corr  (via argsort ranks)
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<double> rank_vector(const std::vector<double>& v) {
    int n = static_cast<int>(v.size());
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return v[a] < v[b]; });
    std::vector<double> rank(n);
    for (int i = 0; i < n; ++i) rank[idx[i]] = i + 1.0;
    return rank;
}

double spearman_corr(const std::vector<double>& x, const std::vector<double>& y) {
    int n = static_cast<int>(x.size());
    if (n < 3) return 0.0;

    auto rx = rank_vector(x);
    auto ry = rank_vector(y);
    double d_sq = 0.0;
    for (int i = 0; i < n; ++i) d_sq += (rx[i] - ry[i]) * (rx[i] - ry[i]);
    double denom = (double)n * ((double)n*n - 1.0);
    return denom == 0.0 ? 0.0 : 1.0 - 6.0 * d_sq / denom;
}

// ─────────────────────────────────────────────────────────────────────────────
// pearson_corr
// ─────────────────────────────────────────────────────────────────────────────
double pearson_corr(const std::vector<double>& x, const std::vector<double>& y) {
    int n = static_cast<int>(x.size());
    if (n < 3) return 0.0;

    double mx = 0, my = 0;
    for (int i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;

    double cov = 0, sx = 0, sy = 0;
    for (int i = 0; i < n; ++i) {
        double dx = x[i] - mx, dy = y[i] - my;
        cov += dx * dy;
        sx  += dx * dx;
        sy  += dy * dy;
    }
    double denom = std::sqrt(sx * sy);
    if (denom == 0.0 || std::isnan(denom)) return 0.0;
    double r = cov / denom;
    return std::isnan(r) ? 0.0 : r;
}

// ─────────────────────────────────────────────────────────────────────────────
// evaluate_window  (inclusive [start, end] slice)
// ─────────────────────────────────────────────────────────────────────────────
ResidualResult evaluate_window(const std::vector<double>& x,
                               const std::vector<double>& y,
                               int start, int end) {
    ResidualResult out;
    int total = static_cast<int>(x.size());
    start = std::max(0, std::min(start, total - 1));
    end   = std::max(0, std::min(end,   total - 1));
    if (start > end) std::swap(start, end);

    int n = end - start + 1;
    out.n = n;

    if (n < 10 || total != static_cast<int>(y.size())) {
        out.verdict = "unknown";
        out.reason  = "insufficient data";
        return out;
    }

    std::vector<double> xw(x.begin()+start, x.begin()+end+1);
    std::vector<double> yw(y.begin()+start, y.begin()+end+1);

    auto fit     = linear_fit(xw, yw);
    auto& resid  = fit.residuals;

    // RMSE
    double mse = 0.0;
    for (double r : resid) mse += r * r;
    out.rmse = std::sqrt(mse / n);

    // Runs test
    auto rt       = runs_test(resid);
    out.runs          = rt.runs;
    out.runs_expected = rt.expected;
    out.runs_ok       = rt.is_random;

    // Durbin-Watson
    out.durbin_watson = durbin_watson(resid);
    out.dw_ok         = (out.durbin_watson > DW_LOW && out.durbin_watson < DW_HIGH);

    // Heteroscedasticity: Spearman(|resid|, fitted)
    std::vector<double> fitted(n), abs_res(n);
    for (int i = 0; i < n; ++i) {
        fitted[i]  = yw[i] - resid[i];
        abs_res[i] = std::abs(resid[i]);
    }
    double corr_abs_fitted = spearman_corr(abs_res, fitted);
    out.hetero_ok = std::abs(corr_abs_fitted) < CORR_ABS_THRESH;

    // Trend: Pearson(resid, x)
    double corr_res_x = pearson_corr(resid, xw);
    out.trend_ok = std::abs(corr_res_x) < TREND_CORR_THRESH;

    // Issues
    if (!out.runs_ok)   out.issues.push_back("non-random (runs test)");
    if (!out.dw_ok)     out.issues.push_back(out.durbin_watson < 1.5
                            ? "autocorrelation/pattern" : "oscillation");
    if (!out.hetero_ok) out.issues.push_back("heteroscedasticity (funnel)");
    if (!out.trend_ok)  out.issues.push_back("curvature/trend");

    out.verdict = out.issues.empty() ? "good" : "bad";
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// score_result  (higher is better)
// ─────────────────────────────────────────────────────────────────────────────
double score_result(const ResidualResult& r) {
    if (r.verdict == "unknown") return -1e9;
    double good   = (r.verdict == "good") ? 1.0 : 0.0;
    double issues = static_cast<double>(r.issues.size());
    return good * 10000.0 - r.rmse * 1000.0 - issues;
}

// ─────────────────────────────────────────────────────────────────────────────
// find_best_window_serial  (seeded local search — mirrors Python logic)
// ─────────────────────────────────────────────────────────────────────────────
BestWindowResult find_best_window_serial(const std::vector<double>& x,
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

    int total_points  = std::max(n, total_points_hint < 0 ? n : total_points_hint);
    int strict_min_pts = std::max(10, static_cast<int>(std::floor(min_fraction * total_points)) + 1);

    int search_start = std::min(10, n - 1);
    int search_end   = std::min(200, n - 1);
    if (search_end < search_start) { search_start = 0; search_end = n - 1; }
    int search_len = search_end - search_start + 1;
    if (search_len < strict_min_pts) {
        search_start = 0; search_end = n - 1;
        search_len   = n;
        strict_min_pts = std::min(strict_min_pts, search_len);
    }

    // Window clamping helper
    auto clamp_window = [&](int s, int e) -> std::pair<int,int> {
        s = std::max(search_start, std::min(s, search_end));
        e = std::max(search_start, std::min(e, search_end));
        if (s > e) std::swap(s, e);
        if (e - s + 1 < strict_min_pts) {
            e = std::min(search_end, s + strict_min_pts - 1);
            s = std::max(search_start, e - strict_min_pts + 1);
        }
        return {s, e};
    };

    // Cache: map (start, end) -> (score, result)
    struct PairHash {
        size_t operator()(const std::pair<int,int>& p) const {
            return std::hash<long long>()(((long long)p.first << 32) | (unsigned)p.second);
        }
    };
    std::unordered_map<std::pair<int,int>, std::pair<double, ResidualResult>, PairHash> cache;

    int evaluated = 0;
    const int MAX_CANDIDATES  = 5000;
    const int MAX_ITERATIONS  = 200;

    auto eval_window = [&](int s, int e) -> std::pair<double, ResidualResult> {
        auto [cs, ce] = clamp_window(s, e);
        auto key = std::make_pair(cs, ce);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        auto r = evaluate_window(x, y, cs, ce);
        double sc = score_result(r);
        cache[key] = {sc, r};
        ++evaluated;
        return {sc, r};
    };

    // Seeds: left, center, right, full
    int seed_len      = strict_min_pts;
    int center_start  = search_start + std::max(0, (search_len - seed_len) / 2);
    std::vector<std::pair<int,int>> seeds = {
        {search_start, search_start + seed_len - 1},
        {center_start, center_start + seed_len - 1},
        {search_end - seed_len + 1, search_end},
        {search_start, search_end},
    };

    double best_score = -1e9;
    int    best_s = search_start, best_e = search_end;
    ResidualResult best_result;
    bool   has_best = false;

    for (auto& [seed_s, seed_e] : seeds) {
        if (evaluated >= MAX_CANDIDATES) break;
        auto [s, e] = clamp_window(seed_s, seed_e);
        int step = std::max(1, (e - s + 1) / 4);

        auto [cur_score, cur_result] = eval_window(s, e);
        if (cur_score > best_score) {
            best_score = cur_score; best_s = s; best_e = e;
            best_result = cur_result; has_best = true;
        }

        int iterations = 0;
        while (step >= 1 && iterations < MAX_ITERATIONS && evaluated < MAX_CANDIDATES) {
            ++iterations;
            std::vector<std::pair<int,int>> cands = {
                {s-step, e-step}, {s+step, e+step},
                {s-step, e},      {s+step, e},
                {s, e-step},      {s, e+step},
            };

            double local_best_score = cur_score;
            int    local_s = s, local_e = e;
            ResidualResult local_result = cur_result;
            bool improved = false;

            for (auto& [cs, ce] : cands) {
                auto [ccs, cce] = clamp_window(cs, ce);
                if (ccs == s && cce == e) continue;
                auto [sc, res] = eval_window(ccs, cce);
                if (sc > local_best_score) {
                    local_best_score = sc;
                    local_s = ccs; local_e = cce;
                    local_result = res;
                    improved = true;
                }
            }
            if (improved) {
                cur_score = local_best_score;
                s = local_s; e = local_e;
                cur_result = local_result;
                if (cur_score > best_score) {
                    best_score = cur_score; best_s = s; best_e = e;
                    best_result = cur_result; has_best = true;
                }
            } else {
                step /= 2;
            }
        }
    }

    if (!has_best) {
        best_result = evaluate_window(x, y, search_start, search_end);
        best_s = search_start; best_e = search_end;
    }

    if (best_result.verdict != "good") {
        best_result.selection_note = "No good window found; returning least-bad window.";
    }

    out.start  = best_s;
    out.end    = best_e;
    out.result = best_result;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// find_best_window_exhaustive  (serial exhaustive grid — fair baseline for OMP)
// ─────────────────────────────────────────────────────────────────────────────
BestWindowResult find_best_window_exhaustive(const std::vector<double>& x,
                                              const std::vector<double>& y,
                                              double min_fraction,
                                              int    total_points_hint) {
    BestWindowResult out;
    int n = static_cast<int>(x.size());
    if (n < 10 || n != static_cast<int>(y.size())) {
        out.start = 0; out.end = std::max(0, n-1);
        out.result.verdict = "unknown"; out.result.reason = "insufficient data";
        return out;
    }

    int total_points   = std::max(n, total_points_hint < 0 ? n : total_points_hint);
    int strict_min_pts = std::max(10, static_cast<int>(std::floor(min_fraction * total_points)) + 1);
    int search_start   = std::min(10, n - 1);
    int search_end     = std::min(200, n - 1);
    if (search_end < search_start) { search_start = 0; search_end = n - 1; }
    if (search_end - search_start + 1 < strict_min_pts) {
        search_start = 0; search_end = n - 1;
        strict_min_pts = std::min(strict_min_pts, n);
    }

    double         best_score = -1e9;
    int            best_s = search_start, best_e = search_end;
    ResidualResult best_result;

    for (int s = search_start; s <= search_end; ++s) {
        for (int e = s + strict_min_pts - 1; e <= search_end; ++e) {
            auto r  = evaluate_window(x, y, s, e);
            double sc = score_result(r);
            if (sc > best_score) {
                best_score = sc; best_s = s; best_e = e; best_result = r;
            }
        }
    }

    if (best_result.verdict != "good")
        best_result.selection_note = "No good window found; returning least-bad window.";
    out.start = best_s; out.end = best_e; out.result = best_result;
    return out;
}

}  // namespace tps

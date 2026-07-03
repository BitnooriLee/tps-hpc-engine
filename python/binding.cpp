// binding.cpp
//
// pybind11 module: tps_engine
//
// Exposes the C++ TPS analysis engine to Python as a drop-in accelerator.
// The API is designed to match what HD_Intelligent currently calls via NumPy,
// so no changes to the Python pipeline are required — only the import path
// changes from the NumPy implementation to this compiled module.
//
// Build:
//   cmake .. -DENABLE_PYBIND=ON && make tps_engine
//   # or: pip install pybind11 && cmake .. -DENABLE_PYBIND=ON -DENABLE_MPI=OFF

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tps/residual.hpp>
#include <tps/optimization.hpp>

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: convert C++ structs → Python dicts
// ─────────────────────────────────────────────────────────────────────────────

static py::dict residual_result_to_dict(const tps::ResidualResult& r)
{
    py::dict d;
    d["verdict"]        = r.verdict;
    d["reason"]         = r.reason;
    d["selection_note"] = r.selection_note;
    d["rmse"]           = r.rmse;
    d["n"]              = r.n;
    d["runs"]           = r.runs;
    d["runs_expected"]  = r.runs_expected;
    d["runs_ok"]        = r.runs_ok;
    d["dw"]             = r.durbin_watson;
    d["dw_ok"]          = r.dw_ok;
    d["hetero_ok"]      = r.hetero_ok;
    d["trend_ok"]       = r.trend_ok;
    d["issues"]         = r.issues;
    return d;
}

static py::dict bwr_to_dict(const tps::BestWindowResult& bwr)
{
    py::dict d = residual_result_to_dict(bwr.result);
    d["start"] = bwr.start;
    d["end"]   = bwr.end;
    return d;
}

static py::dict linear_fit_to_dict(const tps::LinearFitResult& fit)
{
    py::dict d;
    d["slope"]     = fit.slope;
    d["intercept"] = fit.intercept;
    d["residuals"] = fit.residuals;
    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
// Module definition
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(tps_engine, m)
{
    m.doc() =
        "tps_engine — C++/OpenMP TPS analysis engine (pybind11 bindings)\n\n"
        "Drop-in accelerator for HD_Intelligent thermal conductivity pipeline.";

    // ── Window search ────────────────────────────────────────────────────────

    m.def("find_best_window_serial",
          [](const std::vector<double>& x,
             const std::vector<double>& y,
             double min_fraction) {
              return bwr_to_dict(tps::find_best_window_serial(x, y, min_fraction));
          },
          py::arg("x"), py::arg("y"), py::arg("min_fraction") = 0.30,
          R"(Serial seeded local search for the best TPS analysis window.

Returns a dict with keys:
  start, end          — window indices (inclusive)
  verdict             — "good" | "bad" | "unknown"
  rmse                — root mean squared error of linear fit
  n                   — number of points in window
  runs_ok, dw_ok, hetero_ok, trend_ok — individual test results
  dw                  — Durbin-Watson statistic
  issues              — list of failed test names
  selection_note      — non-empty when no "good" window was found
)");

    m.def("find_best_window_omp",
          [](const std::vector<double>& x,
             const std::vector<double>& y,
             double min_fraction) {
              return bwr_to_dict(tps::find_best_window_omp(x, y, min_fraction));
          },
          py::arg("x"), py::arg("y"), py::arg("min_fraction") = 0.30,
          R"(OpenMP exhaustive parallel search — finds the global optimum window.

Same return format as find_best_window_serial.
Number of threads controlled by OMP_NUM_THREADS environment variable.
)");

    // ── Single window evaluation ─────────────────────────────────────────────

    m.def("evaluate_window",
          [](const std::vector<double>& x,
             const std::vector<double>& y,
             int start, int end) {
              return residual_result_to_dict(tps::evaluate_window(x, y, start, end));
          },
          py::arg("x"), py::arg("y"), py::arg("start"), py::arg("end"),
          R"(Evaluate a single window [start, end] against all TPS quality criteria.

Runs: linear fit → runs test → Durbin-Watson → heteroscedasticity → trend check.
Returns a dict with the same keys as find_best_window_serial, minus start/end.
)");

    // ── Linear fit (exposed for scripting / inspection) ──────────────────────

    m.def("linear_fit",
          [](const std::vector<double>& x, const std::vector<double>& y) {
              return linear_fit_to_dict(tps::linear_fit(x, y));
          },
          py::arg("x"), py::arg("y"),
          "Least-squares linear fit. Returns {slope, intercept, residuals}.");

    // ── TPS physics helpers ──────────────────────────────────────────────────

    m.def("estimate_conductivity",
          &tps::estimate_conductivity,
          py::arg("power"), py::arg("delta_t"),
          py::arg("sensor_radius"), py::arg("f_tau"),
          R"(Estimate thermal conductivity.

k = power / (π^1.5 × sensor_radius × delta_t × f_tau)

Args:
  power         — heating power [W]
  delta_t       — temperature rise ΔT [K]  (or slope dΔT/d√t as proxy)
  sensor_radius — TPS sensor radius [m]
  f_tau         — dimensionless TPS correction factor

Returns:
  k [W/(m·K)]  (or a.u. if delta_t is used as a slope proxy)
)");

    m.def("get_max_heating_time",
          &tps::get_max_heating_time,
          py::arg("sensor_radius"), py::arg("diffusivity"),
          "t_max = 0.67 × r² / α  [s]");

    m.def("target_power",
          &tps::target_power,
          py::arg("conductivity"), py::arg("target_delta_t"),
          py::arg("sensor_radius"), py::arg("f_tau"),
          "Compute the heating power needed to achieve a target ΔT.");

    m.def("compute_optimal_power",
          [](double measured_delta_t, double target_delta_t,
             double used_power, double sensor_radius, double f_tau) {
              auto [k, p] = tps::compute_optimal_power(
                  measured_delta_t, target_delta_t,
                  used_power, sensor_radius, f_tau);
              return py::make_tuple(k, p);
          },
          py::arg("measured_delta_t"), py::arg("target_delta_t"),
          py::arg("used_power"), py::arg("sensor_radius"), py::arg("f_tau"),
          "Returns (estimated_k, optimal_power) tuple.");

    // ── Version info ─────────────────────────────────────────────────────────
    m.attr("__version__") = "0.1.0";
}

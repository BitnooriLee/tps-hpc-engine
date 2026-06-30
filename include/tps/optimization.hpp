#pragma once
#include <stdexcept>
#include <cmath>

namespace tps {

// t_max = 0.67 × r² / α
inline double get_max_heating_time(double sensor_radius, double diffusivity) {
    return 0.67 * sensor_radius * sensor_radius / diffusivity;
}

// k = P / (π^(3/2) × r × ΔT × f(τ))
inline double estimate_conductivity(double power, double delta_t,
                                    double sensor_radius, double f_tau) {
    double denom = std::pow(M_PI, 1.5) * sensor_radius * delta_t * f_tau;
    if (denom == 0.0) throw std::invalid_argument("denominator is zero");
    return power / denom;
}

// P_target = k × π^(3/2) × r × ΔT_target × f(τ)
inline double target_power(double conductivity, double target_delta_t,
                           double sensor_radius, double f_tau) {
    return conductivity * std::pow(M_PI, 1.5) * sensor_radius * target_delta_t * f_tau;
}

// Returns {estimated_k, optimal_power}
inline std::pair<double,double> compute_optimal_power(
        double measured_delta_t, double target_delta_t,
        double used_power, double sensor_radius, double f_tau) {
    if (measured_delta_t <= 0) throw std::invalid_argument("measured_delta_t must be > 0");
    if (target_delta_t   <= 0) throw std::invalid_argument("target_delta_t must be > 0");
    if (used_power       <= 0) throw std::invalid_argument("used_power must be > 0");
    double k   = estimate_conductivity(used_power, measured_delta_t, sensor_radius, f_tau);
    double p   = target_power(k, target_delta_t, sensor_radius, f_tau);
    return {k, p};
}

}  // namespace tps

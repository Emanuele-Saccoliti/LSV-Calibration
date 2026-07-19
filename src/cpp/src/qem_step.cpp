#include "lsv/qem_step.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lsv {
namespace {

constexpr double kSqrtTwo = 1.4142135623730950488;

double normal_cdf(const double value) {
  return 0.5 * std::erfc(-value / kSqrtTwo);
}

struct VarianceStep {
  double next_variance;
  double conditional_exponential_moment;
};

VarianceStep sample_variance(const double variance, const double normal,
                             const double dt,
                             const HestonParameters& parameters,
                             const double psi_threshold,
                             const double moment_argument) {
  const double exponential_kdt = std::exp(-parameters.kappa * dt);
  const double one_minus_exponential = 1.0 - exponential_kdt;
  const double mean = parameters.theta +
                      (variance - parameters.theta) * exponential_kdt;
  const double eta_squared = parameters.eta * parameters.eta;
  const double conditional_variance =
      variance * eta_squared * exponential_kdt * one_minus_exponential /
          parameters.kappa +
      parameters.theta * eta_squared * one_minus_exponential *
          one_minus_exponential / (2.0 * parameters.kappa);
  const double psi = conditional_variance / (mean * mean);
  if (psi <= psi_threshold) {
    const double inverse_psi = 2.0 / psi;
    const double b_squared =
        inverse_psi - 1.0 + std::sqrt(inverse_psi * (inverse_psi - 1.0));
    const double a = mean / (1.0 + b_squared);
    const double shifted_normal = std::sqrt(b_squared) + normal;
    const double denominator = 1.0 - 2.0 * moment_argument * a;
    if (denominator <= 0.0) {
      throw std::runtime_error("QE-M quadratic moment does not exist; refine steps");
    }
    return {a * shifted_normal * shifted_normal,
            std::exp(moment_argument * a * b_squared / denominator) /
                std::sqrt(denominator)};
  }
  const double probability_zero = (psi - 1.0) / (psi + 1.0);
  const double beta = (1.0 - probability_zero) / mean;
  if (beta <= moment_argument) {
    throw std::runtime_error("QE-M exponential moment does not exist; refine steps");
  }
  const double uniform = std::clamp(normal_cdf(normal), 1e-15, 1.0 - 1e-15);
  const double next_variance =
      uniform <= probability_zero
          ? 0.0
          : std::log((1.0 - probability_zero) / (1.0 - uniform)) / beta;
  return {next_variance,
          probability_zero + beta * (1.0 - probability_zero) /
                                 (beta - moment_argument)};
}

}  // namespace

QemState advance_qem(const QemState& state, const double variance_normal,
                     const double independent_normal, const double dt,
                     const double carry, const HestonParameters& parameters,
                     const double leverage, const double psi_threshold) {
  if (!std::isfinite(leverage) || leverage <= 0.0) {
    throw std::invalid_argument("leverage must be finite and positive");
  }
  constexpr double gamma_one = 0.5;
  constexpr double gamma_two = 0.5;
  const double correlated_scale = parameters.rho * leverage / parameters.eta;
  const double common = parameters.kappa * correlated_scale - 0.5 * leverage * leverage;
  const double k1 = gamma_one * dt * common - correlated_scale;
  const double k2 = gamma_two * dt * common + correlated_scale;
  const double uncorrelated_variance =
      leverage * leverage * (1.0 - parameters.rho * parameters.rho);
  const double k3 = gamma_one * dt * uncorrelated_variance;
  const double k4 = gamma_two * dt * uncorrelated_variance;
  const double moment_argument = k2 + 0.5 * k4;
  const VarianceStep transition = sample_variance(
      state.variance, variance_normal, dt, parameters, psi_threshold,
      moment_argument);
  const double conditional_moment =
      std::exp((k1 + 0.5 * k3) * state.variance) *
      transition.conditional_exponential_moment;
  if (!(conditional_moment > 0.0) || !std::isfinite(conditional_moment)) {
    throw std::runtime_error("invalid QE-M conditional martingale moment");
  }
  const double diffusion_variance =
      std::max(k3 * state.variance + k4 * transition.next_variance, 0.0);
  const double next_log_spot =
      state.log_spot + carry * dt - std::log(conditional_moment) +
      k1 * state.variance + k2 * transition.next_variance +
      std::sqrt(diffusion_variance) * independent_normal;
  return {next_log_spot, transition.next_variance};
}

}  // namespace lsv


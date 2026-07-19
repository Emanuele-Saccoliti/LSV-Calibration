#include "lsv/heston_qem.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "lsv/qem_step.hpp"

namespace lsv {
namespace {

void validate(const HestonParameters& parameters,
              const SimulationConfig& config) {
  const bool valid_parameters =
      std::isfinite(parameters.kappa) && parameters.kappa > 0.0 &&
      std::isfinite(parameters.theta) && parameters.theta > 0.0 &&
      std::isfinite(parameters.eta) && parameters.eta > 0.0 &&
      std::isfinite(parameters.rho) && parameters.rho > -1.0 &&
      parameters.rho < 1.0 && std::isfinite(parameters.v0) &&
      parameters.v0 >= 0.0;
  if (!valid_parameters) {
    throw std::invalid_argument("invalid Heston parameters");
  }
  const bool valid_config =
      std::isfinite(config.spot) && config.spot > 0.0 &&
      std::isfinite(config.maturity) && config.maturity > 0.0 &&
      std::isfinite(config.rate) && std::isfinite(config.dividend_yield) &&
      config.paths > 0U && config.steps > 0U &&
      std::isfinite(config.psi_threshold) && config.psi_threshold > 1.0 &&
      config.psi_threshold < 2.0;
  if (!valid_config) {
    throw std::invalid_argument("invalid Heston simulation configuration");
  }
  if (config.paths >
      std::numeric_limits<std::size_t>::max() / (config.steps + 1U)) {
    throw std::invalid_argument("requested path matrix is too large");
  }
}

}  // namespace

HestonPathResult simulate_heston_qem(const HestonParameters& parameters,
                                     const SimulationConfig& config) {
  validate(parameters, config);
  const std::size_t time_points = config.steps + 1U;
  HestonPathResult result{config.paths,
                          time_points,
                          std::vector<double>(config.paths * time_points),
                          std::vector<double>(config.paths * time_points)};
  const double dt = config.maturity / static_cast<double>(config.steps);
  std::mt19937_64 engine(config.seed);
  std::normal_distribution<double> normal_distribution(0.0, 1.0);
  const std::size_t path_increment = config.antithetic ? 2U : 1U;
  for (std::size_t first_path = 0; first_path < config.paths;
       first_path += path_increment) {
    const bool has_pair = config.antithetic && first_path + 1U < config.paths;
    const std::size_t first_row = first_path * time_points;
    const std::size_t second_row = (first_path + 1U) * time_points;
    result.spots[first_row] = config.spot;
    result.variances[first_row] = parameters.v0;
    if (has_pair) {
      result.spots[second_row] = config.spot;
      result.variances[second_row] = parameters.v0;
    }
    double first_log_spot = std::log(config.spot);
    double second_log_spot = first_log_spot;
    double first_variance = parameters.v0;
    double second_variance = parameters.v0;

    const auto advance = [&](double& log_spot, double& variance,
                             const double variance_normal,
                             const double spot_normal, const std::size_t row,
                             const std::size_t step) {
      const QemState next = advance_qem(
          {log_spot, variance}, variance_normal, spot_normal, dt,
          config.rate - config.dividend_yield, parameters, 1.0,
          config.psi_threshold);
      log_spot = next.log_spot;
      variance = next.variance;
      result.spots[row + step + 1U] = std::exp(log_spot);
      result.variances[row + step + 1U] = variance;
    };

    for (std::size_t step = 0; step < config.steps; ++step) {
      const double variance_normal = normal_distribution(engine);
      const double spot_normal = normal_distribution(engine);
      advance(first_log_spot, first_variance, variance_normal, spot_normal,
              first_row, step);
      if (has_pair) {
        advance(second_log_spot, second_variance, -variance_normal, -spot_normal,
                second_row, step);
      }
    }
  }
  return result;
}

}  // namespace lsv

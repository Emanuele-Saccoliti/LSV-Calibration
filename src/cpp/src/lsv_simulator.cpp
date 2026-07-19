#include "lsv/lsv_simulator.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

#include "lsv/qem_step.hpp"

namespace lsv {

HestonPathResult simulate_lsv_qem(const HestonParameters& parameters,
                                  const SimulationConfig& config,
                                  const LeverageSurface& leverage_surface) {
  if (config.paths == 0U || config.steps == 0U || config.spot <= 0.0 ||
      config.maturity <= 0.0) {
    throw std::invalid_argument("invalid LSV simulation configuration");
  }
  const std::size_t time_points = config.steps + 1U;
  HestonPathResult result{config.paths,
                          time_points,
                          std::vector<double>(config.paths * time_points),
                          std::vector<double>(config.paths * time_points)};
  const double dt = config.maturity / static_cast<double>(config.steps);
  const double carry = config.rate - config.dividend_yield;
  std::mt19937_64 engine(config.seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  const std::size_t increment = config.antithetic ? 2U : 1U;
  for (std::size_t first_path = 0; first_path < config.paths;
       first_path += increment) {
    const bool paired = config.antithetic && first_path + 1U < config.paths;
    QemState first{std::log(config.spot), parameters.v0};
    QemState second = first;
    const std::size_t first_row = first_path * time_points;
    const std::size_t second_row = (first_path + 1U) * time_points;
    result.spots[first_row] = config.spot;
    result.variances[first_row] = parameters.v0;
    if (paired) {
      result.spots[second_row] = config.spot;
      result.variances[second_row] = parameters.v0;
    }
    for (std::size_t step = 0; step < config.steps; ++step) {
      const double variance_normal = normal(engine);
      const double independent_normal = normal(engine);
      const double time = static_cast<double>(step) * dt;
      const double first_moneyness =
          first.log_spot - std::log(config.spot) - carry * time;
      const double first_leverage =
          leverage_surface.value(time, first_moneyness);
      first = advance_qem(first, variance_normal, independent_normal, dt, carry,
                          parameters, first_leverage, config.psi_threshold);
      result.spots[first_row + step + 1U] = std::exp(first.log_spot);
      result.variances[first_row + step + 1U] = first.variance;
      if (paired) {
        const double second_moneyness =
            second.log_spot - std::log(config.spot) - carry * time;
        const double second_leverage =
            leverage_surface.value(time, second_moneyness);
        second = advance_qem(second, -variance_normal, -independent_normal, dt,
                             carry, parameters, second_leverage,
                             config.psi_threshold);
        result.spots[second_row + step + 1U] = std::exp(second.log_spot);
        result.variances[second_row + step + 1U] = second.variance;
      }
    }
  }
  return result;
}

}  // namespace lsv


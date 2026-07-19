#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>

#include "lsv/heston_qem.hpp"
#include "lsv/types.hpp"

int main() {
  const lsv::HestonParameters parameters{1.7, 0.045, 0.5, -0.65, 0.04};
  const lsv::SimulationConfig config{100.0, 1.0, 0.03, 0.01,
                                     20000U, 64U, 123456U, true, 1.5};
  const auto first = lsv::simulate_heston_qem(parameters, config);
  const auto second = lsv::simulate_heston_qem(parameters, config);
  if (first.spots != second.spots || first.variances != second.variances) {
    std::cerr << "identical seeds did not produce identical paths\n";
    return 1;
  }
  if (*std::min_element(first.variances.begin(), first.variances.end()) < 0.0) {
    std::cerr << "QE produced a negative variance\n";
    return 1;
  }

  double terminal_variance_sum = 0.0;
  double discounted_spot_sum = 0.0;
  double discounted_spot_squared_sum = 0.0;
  for (std::size_t path = 0; path < config.paths; ++path) {
    const std::size_t terminal = path * first.time_points + config.steps;
    terminal_variance_sum += first.variances[terminal];
    const double discounted =
        first.spots[terminal] * std::exp(-config.rate * config.maturity);
    discounted_spot_sum += discounted;
    discounted_spot_squared_sum += discounted * discounted;
  }
  const double path_count = static_cast<double>(config.paths);
  const double mean_variance = terminal_variance_sum / path_count;
  const double expected_variance =
      parameters.theta + (parameters.v0 - parameters.theta) *
                             std::exp(-parameters.kappa * config.maturity);
  if (std::abs(mean_variance - expected_variance) > 8e-4) {
    std::cerr << "CIR mean mismatch: " << mean_variance << " versus "
              << expected_variance << '\n';
    return 1;
  }

  const double discounted_mean = discounted_spot_sum / path_count;
  const double sample_variance =
      (discounted_spot_squared_sum -
       path_count * discounted_mean * discounted_mean) /
      (path_count - 1.0);
  const double standard_error = std::sqrt(sample_variance / path_count);
  const double expected_discounted_spot =
      config.spot * std::exp(-config.dividend_yield * config.maturity);
  if (std::abs(discounted_mean - expected_discounted_spot) >
      4.0 * standard_error) {
    std::cerr << "discounted spot is outside four standard errors\n";
    return 1;
  }
  return 0;
}


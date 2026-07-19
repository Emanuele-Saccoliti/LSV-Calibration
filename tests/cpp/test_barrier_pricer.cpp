#include <cmath>
#include <iostream>
#include <vector>

#include "lsv/barrier_pricer.hpp"

int main() {
  constexpr std::size_t paths = 100U;
  constexpr std::size_t steps = 1U;
  const std::vector<double> spots(paths * (steps + 1U), 100.0);
  const std::vector<double> interval_variances(paths * steps, 0.04);
  const auto no_touch = lsv::price_barrier_paths(
      spots, interval_variances, paths, steps, 1.0, 0.02, 1.0,
      lsv::BarrierType::Double, lsv::BarrierPayoff::NoTouch,
      lsv::Monitoring::Continuous, 80.0, 120.0, 7U);
  const auto touch = lsv::price_barrier_paths(
      spots, interval_variances, paths, steps, 1.0, 0.02, 1.0,
      lsv::BarrierType::Double, lsv::BarrierPayoff::Touch,
      lsv::Monitoring::Continuous, 80.0, 120.0, 7U);
  const double discounted_payout = std::exp(-0.02);
  if (!(no_touch.estimate > 0.0 && no_touch.estimate < discounted_payout)) {
    std::cerr << "DNT price is outside payoff bounds\n";
    return 1;
  }
  if (std::abs(no_touch.estimate + touch.estimate - discounted_payout) > 1e-12) {
    std::cerr << "touch/no-touch parity failed\n";
    return 1;
  }
  return 0;
}


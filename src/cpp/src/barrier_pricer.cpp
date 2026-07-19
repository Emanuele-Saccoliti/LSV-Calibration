#include "lsv/barrier_pricer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "lsv/brownian_bridge.hpp"

namespace lsv {

BarrierPriceResult price_barrier_paths(
    std::span<const double> spots,
    std::span<const double> interval_variances, const std::size_t paths,
    const std::size_t steps, const double maturity, const double rate,
    const double payout, const BarrierType barrier_type,
    const BarrierPayoff payoff_type, const Monitoring monitoring,
    const double lower_barrier, const double upper_barrier,
    const std::uint64_t seed, const int image_terms) {
  if (paths == 0U || steps == 0U || spots.size() != paths * (steps + 1U) ||
      interval_variances.size() != paths * steps || maturity <= 0.0 ||
      payout < 0.0 || !std::isfinite(rate) || !std::isfinite(payout)) {
    throw std::invalid_argument("invalid barrier pricing inputs");
  }
  const bool uses_lower =
      barrier_type == BarrierType::Lower || barrier_type == BarrierType::Double;
  const bool uses_upper =
      barrier_type == BarrierType::Upper || barrier_type == BarrierType::Double;
  if ((uses_lower && (!(lower_barrier > 0.0) ||
                      lower_barrier >= spots.front())) ||
      (uses_upper && (!(upper_barrier > spots.front()))) ||
      (barrier_type == BarrierType::Double &&
       lower_barrier >= upper_barrier)) {
    throw std::invalid_argument("barriers must strictly bracket initial spot");
  }
  const double log_lower = uses_lower ? std::log(lower_barrier) : 0.0;
  const double log_upper = uses_upper ? std::log(upper_barrier) : 0.0;
  const double discount = std::exp(-rate * maturity);
  double sum = 0.0;
  double squared_sum = 0.0;
  for (std::size_t path = 0; path < paths; ++path) {
    const std::size_t spot_row = path * (steps + 1U);
    const std::size_t variance_row = path * steps;
    double survival = 1.0;
    for (std::size_t step = 0; step < steps; ++step) {
      const double start = spots[spot_row + step];
      const double end = spots[spot_row + step + 1U];
      const double variance = interval_variances[variance_row + step];
      if (!(start > 0.0) || !(end > 0.0) || variance < 0.0 ||
          !std::isfinite(variance)) {
        throw std::invalid_argument("paths and interval variances must be valid");
      }
      const double log_start = std::log(start);
      const double log_end = std::log(end);
      double interval_survival = 1.0;
      if (barrier_type == BarrierType::Lower) {
        interval_survival =
            monitoring == Monitoring::Continuous
                ? lower_bridge_survival(log_start, log_end, log_lower, variance)
                : (end > lower_barrier ? 1.0 : 0.0);
      } else if (barrier_type == BarrierType::Upper) {
        interval_survival =
            monitoring == Monitoring::Continuous
                ? upper_bridge_survival(log_start, log_end, log_upper, variance)
                : (end < upper_barrier ? 1.0 : 0.0);
      } else {
        interval_survival =
            monitoring == Monitoring::Continuous
                ? double_bridge_survival(log_start, log_end, log_lower,
                                         log_upper, variance, image_terms)
                : (end > lower_barrier && end < upper_barrier ? 1.0 : 0.0);
      }
      survival *= interval_survival;
      if (survival == 0.0) {
        break;
      }
    }
    const double probability =
        payoff_type == BarrierPayoff::NoTouch ? survival : 1.0 - survival;
    const double discounted_payoff = discount * payout * probability;
    sum += discounted_payoff;
    squared_sum += discounted_payoff * discounted_payoff;
  }
  const double count = static_cast<double>(paths);
  const double estimate = sum / count;
  const double sample_variance =
      paths > 1U
          ? std::max((squared_sum - count * estimate * estimate) / (count - 1.0),
                     0.0)
          : 0.0;
  const double standard_error = std::sqrt(sample_variance / count);
  constexpr double z95 = 1.959963984540054;
  return {estimate,
          standard_error,
          std::max(0.0, estimate - z95 * standard_error),
          std::min(discount * payout, estimate + z95 * standard_error),
          paths,
          steps,
          seed};
}

}  // namespace lsv


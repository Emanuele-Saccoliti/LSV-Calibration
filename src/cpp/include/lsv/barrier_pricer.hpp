#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace lsv {

enum class BarrierType { Lower, Upper, Double };
enum class BarrierPayoff { Touch, NoTouch };
enum class Monitoring { Discrete, Continuous };

struct BarrierPriceResult {
  double estimate;
  double standard_error;
  double confidence_interval_low;
  double confidence_interval_high;
  std::size_t paths;
  std::size_t steps;
  std::uint64_t seed;
};

[[nodiscard]] BarrierPriceResult price_barrier_paths(
    std::span<const double> spots, std::span<const double> interval_variances,
    std::size_t paths, std::size_t steps, double maturity, double rate,
    double payout, BarrierType barrier_type, BarrierPayoff payoff_type,
    Monitoring monitoring, double lower_barrier, double upper_barrier,
    std::uint64_t seed, int image_terms = 12);

}  // namespace lsv


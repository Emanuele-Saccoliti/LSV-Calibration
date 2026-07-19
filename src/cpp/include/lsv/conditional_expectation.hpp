#pragma once

#include <span>
#include <vector>

namespace lsv {

struct ConditionalExpectationResult {
  std::vector<double> conditional_variances;
  std::vector<double> effective_sample_sizes;
  std::vector<bool> low_density;
};

[[nodiscard]] ConditionalExpectationResult estimate_conditional_variance(
    std::span<const double> particle_coordinates,
    std::span<const double> particle_variances,
    std::span<const double> query_coordinates, double bandwidth,
    double denominator_floor, double minimum_effective_sample_size);

}  // namespace lsv


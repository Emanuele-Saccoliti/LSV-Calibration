#include "lsv/conditional_expectation.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace lsv {

ConditionalExpectationResult estimate_conditional_variance(
    std::span<const double> particle_coordinates,
    std::span<const double> particle_variances,
    std::span<const double> query_coordinates, const double bandwidth,
    const double denominator_floor,
    const double minimum_effective_sample_size) {
  if (particle_coordinates.empty() ||
      particle_coordinates.size() != particle_variances.size() ||
      !std::isfinite(bandwidth) || bandwidth <= 0.0 ||
      !std::isfinite(denominator_floor) || denominator_floor <= 0.0 ||
      minimum_effective_sample_size <= 0.0) {
    throw std::invalid_argument("invalid conditional-expectation inputs");
  }
  ConditionalExpectationResult result;
  result.conditional_variances.reserve(query_coordinates.size());
  result.effective_sample_sizes.reserve(query_coordinates.size());
  result.low_density.reserve(query_coordinates.size());
  for (const double query : query_coordinates) {
    double weight_sum = 0.0;
    double squared_weight_sum = 0.0;
    double weighted_variance_sum = 0.0;
    double nearest_distance = std::numeric_limits<double>::infinity();
    double nearest_variance = 0.0;
    for (std::size_t particle = 0; particle < particle_coordinates.size();
         ++particle) {
      const double distance = particle_coordinates[particle] - query;
      const double scaled_distance = distance / bandwidth;
      const double weight = std::exp(-0.5 * scaled_distance * scaled_distance);
      weight_sum += weight;
      squared_weight_sum += weight * weight;
      weighted_variance_sum += weight * particle_variances[particle];
      if (std::abs(distance) < nearest_distance) {
        nearest_distance = std::abs(distance);
        nearest_variance = particle_variances[particle];
      }
    }
    const double effective_sample_size =
        squared_weight_sum > 0.0
            ? weight_sum * weight_sum / squared_weight_sum
            : 0.0;
    const bool denominator_low = weight_sum < denominator_floor;
    const double estimate = denominator_low
                                ? nearest_variance
                                : weighted_variance_sum / weight_sum;
    result.conditional_variances.push_back(estimate);
    result.effective_sample_sizes.push_back(effective_sample_size);
    result.low_density.push_back(
        denominator_low || effective_sample_size < minimum_effective_sample_size);
  }
  return result;
}

}  // namespace lsv


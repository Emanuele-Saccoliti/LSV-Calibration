#include "lsv/brownian_bridge.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lsv {

double lower_bridge_survival(const double log_start, const double log_end,
                             const double log_barrier,
                             const double integrated_variance) {
  if (log_start <= log_barrier || log_end <= log_barrier) {
    return 0.0;
  }
  if (integrated_variance <= 0.0) {
    return 1.0;
  }
  const double crossing = std::exp(
      -2.0 * (log_start - log_barrier) * (log_end - log_barrier) /
      integrated_variance);
  return std::clamp(1.0 - crossing, 0.0, 1.0);
}

double upper_bridge_survival(const double log_start, const double log_end,
                             const double log_barrier,
                             const double integrated_variance) {
  if (log_start >= log_barrier || log_end >= log_barrier) {
    return 0.0;
  }
  if (integrated_variance <= 0.0) {
    return 1.0;
  }
  const double crossing = std::exp(
      -2.0 * (log_barrier - log_start) * (log_barrier - log_end) /
      integrated_variance);
  return std::clamp(1.0 - crossing, 0.0, 1.0);
}

double double_bridge_survival(const double log_start, const double log_end,
                              const double log_lower, const double log_upper,
                              const double integrated_variance,
                              const int image_terms) {
  if (!(log_lower < log_upper) || image_terms < 1) {
    throw std::invalid_argument("invalid double Brownian-bridge configuration");
  }
  if (log_start <= log_lower || log_start >= log_upper ||
      log_end <= log_lower || log_end >= log_upper) {
    return 0.0;
  }
  if (integrated_variance <= 0.0) {
    return 1.0;
  }
  const double width = log_upper - log_lower;
  const double direct = log_end - log_start;
  double survival = 0.0;
  for (int image = -image_terms; image <= image_terms; ++image) {
    const double shift = 2.0 * static_cast<double>(image) * width;
    const double first = direct + shift;
    const double reflected = log_end + log_start - 2.0 * log_lower + shift;
    survival +=
        std::exp(-(first * first - direct * direct) /
                 (2.0 * integrated_variance)) -
        std::exp(-(reflected * reflected - direct * direct) /
                 (2.0 * integrated_variance));
  }
  return std::clamp(survival, 0.0, 1.0);
}

}  // namespace lsv


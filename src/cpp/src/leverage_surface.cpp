#include "lsv/leverage_surface.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lsv {
namespace {

std::size_t lower_index(const std::vector<double>& grid, const double point) {
  if (point <= grid.front()) {
    return 0U;
  }
  if (point >= grid.back()) {
    return grid.size() - 2U;
  }
  return static_cast<std::size_t>(
      std::upper_bound(grid.begin(), grid.end(), point) - grid.begin() - 1);
}

}  // namespace

LeverageSurface::LeverageSurface(std::span<const double> times,
                                 std::span<const double> log_moneyness,
                                 std::span<const double> values)
    : times_(times.begin(), times.end()),
      log_moneyness_(log_moneyness.begin(), log_moneyness.end()),
      values_(values.begin(), values.end()) {
  if (times_.size() < 2U || log_moneyness_.size() < 2U ||
      values_.size() != times_.size() * log_moneyness_.size()) {
    throw std::invalid_argument("invalid leverage surface dimensions");
  }
  if (!std::is_sorted(times_.begin(), times_.end()) ||
      std::adjacent_find(times_.begin(), times_.end()) != times_.end() ||
      !std::is_sorted(log_moneyness_.begin(), log_moneyness_.end()) ||
      std::adjacent_find(log_moneyness_.begin(), log_moneyness_.end()) !=
          log_moneyness_.end()) {
    throw std::invalid_argument("leverage grids must be strictly increasing");
  }
  const auto invalid = [](const double value) {
    return !std::isfinite(value) || value <= 0.0;
  };
  if (std::any_of(times_.begin(), times_.end(), invalid) ||
      std::any_of(values_.begin(), values_.end(), invalid) ||
      std::any_of(log_moneyness_.begin(), log_moneyness_.end(),
                  [](const double value) { return !std::isfinite(value); })) {
    throw std::invalid_argument("leverage surface values must be finite and positive");
  }
}

double LeverageSurface::value(const double time,
                              const double log_moneyness) const {
  if (!std::isfinite(time) || !std::isfinite(log_moneyness)) {
    throw std::invalid_argument("leverage interpolation coordinates must be finite");
  }
  const double bounded_time = std::clamp(time, times_.front(), times_.back());
  const double bounded_strike = std::clamp(
      log_moneyness, log_moneyness_.front(), log_moneyness_.back());
  const std::size_t time_index = lower_index(times_, bounded_time);
  const std::size_t strike_index = lower_index(log_moneyness_, bounded_strike);
  const double time_weight =
      (bounded_time - times_[time_index]) /
      (times_[time_index + 1U] - times_[time_index]);
  const double strike_weight =
      (bounded_strike - log_moneyness_[strike_index]) /
      (log_moneyness_[strike_index + 1U] - log_moneyness_[strike_index]);
  const auto at = [&](const std::size_t row, const std::size_t column) {
    return values_[row * log_moneyness_.size() + column];
  };
  const double lower = (1.0 - strike_weight) * at(time_index, strike_index) +
                       strike_weight * at(time_index, strike_index + 1U);
  const double upper =
      (1.0 - strike_weight) * at(time_index + 1U, strike_index) +
      strike_weight * at(time_index + 1U, strike_index + 1U);
  return (1.0 - time_weight) * lower + time_weight * upper;
}

}  // namespace lsv


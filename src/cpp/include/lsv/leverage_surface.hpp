#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace lsv {

class LeverageSurface {
 public:
  LeverageSurface(std::span<const double> times,
                  std::span<const double> log_moneyness,
                  std::span<const double> values);

  [[nodiscard]] double value(double time, double log_moneyness) const;
  [[nodiscard]] std::size_t time_count() const { return times_.size(); }
  [[nodiscard]] std::size_t strike_count() const { return log_moneyness_.size(); }

 private:
  std::vector<double> times_;
  std::vector<double> log_moneyness_;
  std::vector<double> values_;
};

}  // namespace lsv


#include <cmath>
#include <iostream>
#include <vector>

#include "lsv/conditional_expectation.hpp"

int main() {
  const std::vector<double> coordinates{-0.2, -0.1, 0.0, 0.1, 0.2};
  const std::vector<double> variances(5U, 0.04);
  const std::vector<double> queries{0.0, 2.0};
  const auto result = lsv::estimate_conditional_variance(
      coordinates, variances, queries, 0.1, 1e-12, 2.0);
  if (std::abs(result.conditional_variances[0] - 0.04) > 1e-14) {
    std::cerr << "kernel estimator failed constant reproduction\n";
    return 1;
  }
  if (result.low_density[0] || !result.low_density[1]) {
    std::cerr << "low-density flags are incorrect\n";
    return 1;
  }
  return 0;
}


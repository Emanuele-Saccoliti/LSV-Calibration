#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lsv {

struct HestonParameters {
  double kappa;
  double theta;
  double eta;
  double rho;
  double v0;
};

struct SimulationConfig {
  double spot;
  double maturity;
  double rate;
  double dividend_yield;
  std::size_t paths;
  std::size_t steps;
  std::uint64_t seed;
  bool antithetic;
  double psi_threshold{1.5};
};

struct HestonPathResult {
  std::size_t paths;
  std::size_t time_points;
  std::vector<double> spots;
  std::vector<double> variances;
};

}  // namespace lsv


#include <iostream>
#include <vector>

#include "lsv/heston_qem.hpp"
#include "lsv/leverage_surface.hpp"
#include "lsv/lsv_simulator.hpp"

int main() {
  const lsv::HestonParameters parameters{1.5, 0.04, 0.3, -0.6, 0.04};
  const lsv::SimulationConfig config{100.0, 1.0, 0.02, 0.01,
                                     1000U, 8U, 91U, true, 1.5};
  const std::vector<double> times{0.125, 1.0};
  const std::vector<double> strikes{-0.5, 0.5};
  const std::vector<double> leverage(4U, 1.0);
  const lsv::LeverageSurface surface(times, strikes, leverage);
  const auto heston = lsv::simulate_heston_qem(parameters, config);
  const auto lsv = lsv::simulate_lsv_qem(parameters, config, surface);
  if (heston.spots != lsv.spots || heston.variances != lsv.variances) {
    std::cerr << "unit leverage does not reproduce Heston paths\n";
    return 1;
  }
  return 0;
}


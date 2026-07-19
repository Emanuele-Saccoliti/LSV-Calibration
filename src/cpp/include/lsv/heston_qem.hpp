#pragma once

#include "lsv/types.hpp"

namespace lsv {

// Simulate row-major path matrices with shape (paths, steps + 1).
// Implements Andersen QE variance sampling and conditional martingale
// correction for the log spot (QE-M). All rates and parameters are annualized.
[[nodiscard]] HestonPathResult simulate_heston_qem(
    const HestonParameters& parameters, const SimulationConfig& config);

}  // namespace lsv


#pragma once

#include "lsv/leverage_surface.hpp"
#include "lsv/types.hpp"

namespace lsv {

[[nodiscard]] HestonPathResult simulate_lsv_qem(
    const HestonParameters& parameters, const SimulationConfig& config,
    const LeverageSurface& leverage_surface);

}  // namespace lsv


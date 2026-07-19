#pragma once

#include "lsv/types.hpp"

namespace lsv {

struct QemState {
  double log_spot;
  double variance;
};

// One Andersen QE-M step. Leverage is frozen at the beginning of the step.
[[nodiscard]] QemState advance_qem(const QemState& state,
                                   double variance_normal,
                                   double independent_normal, double dt,
                                   double carry,
                                   const HestonParameters& parameters,
                                   double leverage, double psi_threshold);

}  // namespace lsv


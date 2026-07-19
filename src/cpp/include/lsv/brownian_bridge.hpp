#pragma once

namespace lsv {

[[nodiscard]] double lower_bridge_survival(double log_start, double log_end,
                                           double log_barrier,
                                           double integrated_variance);
[[nodiscard]] double upper_bridge_survival(double log_start, double log_end,
                                           double log_barrier,
                                           double integrated_variance);
[[nodiscard]] double double_bridge_survival(double log_start, double log_end,
                                            double log_lower,
                                            double log_upper,
                                            double integrated_variance,
                                            int image_terms = 12);

}  // namespace lsv


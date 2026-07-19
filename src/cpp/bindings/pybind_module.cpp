#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#include "lsv/heston_qem.hpp"
#include "lsv/barrier_pricer.hpp"
#include "lsv/conditional_expectation.hpp"
#include "lsv/leverage_surface.hpp"
#include "lsv/lsv_simulator.hpp"
#include "lsv/types.hpp"
#include "lsv/version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(lsv_cpp, module) {
  module.doc() = "C++ numerical backend for the LSV engine";
  module.attr("__version__") = std::string{lsv::version};
  module.def(
      "simulate_heston_qem",
      [](const double spot, const double maturity, const double rate,
         const double dividend_yield, const double kappa, const double theta,
         const double eta, const double rho, const double v0,
         const std::size_t paths, const std::size_t steps,
         const std::uint64_t seed, const bool antithetic,
         const double psi_threshold) {
        const lsv::HestonParameters parameters{kappa, theta, eta, rho, v0};
        const lsv::SimulationConfig config{spot,       maturity, rate,
                                           dividend_yield, paths, steps,
                                           seed,       antithetic,
                                           psi_threshold};
        lsv::HestonPathResult result;
        {
          py::gil_scoped_release release;
          result = lsv::simulate_heston_qem(parameters, config);
        }
        const auto path_count = static_cast<py::ssize_t>(result.paths);
        const auto time_count = static_cast<py::ssize_t>(result.time_points);
        py::array_t<double> spots({path_count, time_count});
        py::array_t<double> variances({path_count, time_count});
        std::copy(result.spots.begin(), result.spots.end(), spots.mutable_data());
        std::copy(result.variances.begin(), result.variances.end(),
                  variances.mutable_data());
        py::dict output;
        output["spots"] = std::move(spots);
        output["variances"] = std::move(variances);
        output["seed"] = seed;
        output["paths"] = paths;
        output["steps"] = steps;
        return output;
      },
      py::arg("spot"), py::arg("maturity"), py::arg("rate"),
      py::arg("dividend_yield"), py::arg("kappa"), py::arg("theta"),
      py::arg("eta"), py::arg("rho"), py::arg("v0"), py::arg("paths"),
      py::arg("steps"), py::arg("seed"), py::arg("antithetic") = false,
      py::arg("psi_threshold") = 1.5,
      R"doc(
Simulate Heston paths with Andersen QE-M.

Returns row-major NumPy arrays ``spots`` and ``variances`` with shape
``(paths, steps + 1)``. Rates and model parameters are annualized; seed and
path count are explicit. When antithetic is true, consecutive paths use
opposite Gaussian innovations.
)doc");
  module.def(
      "simulate_lsv",
      [](const double spot, const double maturity, const double rate,
         const double dividend_yield, const double kappa, const double theta,
         const double eta, const double rho, const double v0,
         const std::size_t paths, const std::size_t steps,
         const std::uint64_t seed, const bool antithetic,
         const py::array_t<double, py::array::c_style | py::array::forcecast>&
             leverage_times,
         const py::array_t<double, py::array::c_style | py::array::forcecast>&
             leverage_log_moneyness,
         const py::array_t<double, py::array::c_style | py::array::forcecast>&
             leverage_values,
         const double psi_threshold) {
        if (leverage_times.ndim() != 1 ||
            leverage_log_moneyness.ndim() != 1 ||
            leverage_values.ndim() != 2 ||
            leverage_values.shape(0) != leverage_times.shape(0) ||
            leverage_values.shape(1) != leverage_log_moneyness.shape(0)) {
          throw py::value_error("invalid leverage surface array dimensions");
        }
        const auto time_size = static_cast<std::size_t>(leverage_times.size());
        const auto strike_size =
            static_cast<std::size_t>(leverage_log_moneyness.size());
        const lsv::LeverageSurface surface(
            std::span<const double>(leverage_times.data(), time_size),
            std::span<const double>(leverage_log_moneyness.data(), strike_size),
            std::span<const double>(leverage_values.data(),
                                    time_size * strike_size));
        const lsv::HestonParameters parameters{kappa, theta, eta, rho, v0};
        const lsv::SimulationConfig config{spot,       maturity, rate,
                                           dividend_yield, paths, steps,
                                           seed,       antithetic,
                                           psi_threshold};
        lsv::HestonPathResult result;
        {
          py::gil_scoped_release release;
          result = lsv::simulate_lsv_qem(parameters, config, surface);
        }
        const auto path_count = static_cast<py::ssize_t>(result.paths);
        const auto time_count = static_cast<py::ssize_t>(result.time_points);
        py::array_t<double> spots({path_count, time_count});
        py::array_t<double> variances({path_count, time_count});
        std::copy(result.spots.begin(), result.spots.end(), spots.mutable_data());
        std::copy(result.variances.begin(), result.variances.end(),
                  variances.mutable_data());
        py::dict output;
        output["spots"] = std::move(spots);
        output["variances"] = std::move(variances);
        return output;
      },
      py::arg("spot"), py::arg("maturity"), py::arg("rate"),
      py::arg("dividend_yield"), py::arg("kappa"), py::arg("theta"),
      py::arg("eta"), py::arg("rho"), py::arg("v0"), py::arg("paths"),
      py::arg("steps"), py::arg("seed"), py::arg("antithetic"),
      py::arg("leverage_times"), py::arg("leverage_log_moneyness"),
      py::arg("leverage_values"), py::arg("psi_threshold") = 1.5,
      "Simulate LSV paths using frozen-step leverage and the shared QE-M kernel.");
  module.def(
      "estimate_conditional_variance",
      [](const py::array_t<double,
                           py::array::c_style | py::array::forcecast>& coordinates,
         const py::array_t<double,
                           py::array::c_style | py::array::forcecast>& variances,
         const py::array_t<double,
                           py::array::c_style | py::array::forcecast>& queries,
         const double bandwidth, const double denominator_floor,
         const double minimum_effective_sample_size) {
        if (coordinates.ndim() != 1 || variances.ndim() != 1 ||
            queries.ndim() != 1 || coordinates.size() != variances.size()) {
          throw py::value_error("conditional-expectation arrays must be vectors");
        }
        const auto particle_count = static_cast<std::size_t>(coordinates.size());
        const auto query_count = static_cast<std::size_t>(queries.size());
        const auto result = lsv::estimate_conditional_variance(
            std::span<const double>(coordinates.data(), particle_count),
            std::span<const double>(variances.data(), particle_count),
            std::span<const double>(queries.data(), query_count), bandwidth,
            denominator_floor, minimum_effective_sample_size);
        py::dict output;
        output["conditional_variances"] = result.conditional_variances;
        output["effective_sample_sizes"] = result.effective_sample_sizes;
        output["low_density"] = result.low_density;
        return output;
      },
      py::arg("particle_coordinates"), py::arg("particle_variances"),
      py::arg("query_coordinates"), py::arg("bandwidth"),
      py::arg("denominator_floor") = 1e-12,
      py::arg("minimum_effective_sample_size") = 20.0,
      "Estimate E[V|X=x] by Gaussian Nadaraya-Watson kernel regression.");
  module.def(
      "price_barrier",
      [](const py::array_t<double,
                           py::array::c_style | py::array::forcecast>& spots,
         const py::array_t<double,
                           py::array::c_style | py::array::forcecast>&
             interval_variances,
         const double maturity, const double rate, const double payout,
         const std::string& barrier_type, const std::string& payoff_type,
         const std::string& monitoring, const double lower_barrier,
         const double upper_barrier, const std::uint64_t seed,
         const int image_terms) {
        if (spots.ndim() != 2 || interval_variances.ndim() != 2 ||
            spots.shape(0) != interval_variances.shape(0) ||
            spots.shape(1) != interval_variances.shape(1) + 1) {
          throw py::value_error(
              "spots must have shape (paths, steps+1) and interval variances "
              "shape (paths, steps)");
        }
        lsv::BarrierType barrier;
        if (barrier_type == "lower") {
          barrier = lsv::BarrierType::Lower;
        } else if (barrier_type == "upper") {
          barrier = lsv::BarrierType::Upper;
        } else if (barrier_type == "double") {
          barrier = lsv::BarrierType::Double;
        } else {
          throw py::value_error("barrier_type must be lower, upper, or double");
        }
        lsv::BarrierPayoff payoff;
        if (payoff_type == "touch") {
          payoff = lsv::BarrierPayoff::Touch;
        } else if (payoff_type == "no_touch") {
          payoff = lsv::BarrierPayoff::NoTouch;
        } else {
          throw py::value_error("payoff_type must be touch or no_touch");
        }
        lsv::Monitoring monitoring_mode;
        if (monitoring == "continuous") {
          monitoring_mode = lsv::Monitoring::Continuous;
        } else if (monitoring == "discrete") {
          monitoring_mode = lsv::Monitoring::Discrete;
        } else {
          throw py::value_error("monitoring must be continuous or discrete");
        }
        const auto paths = static_cast<std::size_t>(spots.shape(0));
        const auto steps = static_cast<std::size_t>(interval_variances.shape(1));
        lsv::BarrierPriceResult result;
        {
          py::gil_scoped_release release;
          result = lsv::price_barrier_paths(
              std::span<const double>(spots.data(),
                                      static_cast<std::size_t>(spots.size())),
              std::span<const double>(interval_variances.data(),
                                      static_cast<std::size_t>(
                                          interval_variances.size())),
              paths, steps, maturity, rate, payout, barrier, payoff,
              monitoring_mode, lower_barrier, upper_barrier, seed, image_terms);
        }
        py::dict output;
        output["estimate"] = result.estimate;
        output["standard_error"] = result.standard_error;
        output["confidence_interval_low"] = result.confidence_interval_low;
        output["confidence_interval_high"] = result.confidence_interval_high;
        output["paths"] = result.paths;
        output["steps"] = result.steps;
        output["seed"] = result.seed;
        return output;
      },
      py::arg("spots"), py::arg("interval_variances"), py::arg("maturity"),
      py::arg("rate"), py::arg("payout"), py::arg("barrier_type"),
      py::arg("payoff_type"), py::arg("monitoring"),
      py::arg("lower_barrier") = 0.0, py::arg("upper_barrier") = 0.0,
      py::arg("seed") = 0U, py::arg("image_terms") = 12,
      "Price maturity-paid barrier claims from paths with bridge correction.");
}

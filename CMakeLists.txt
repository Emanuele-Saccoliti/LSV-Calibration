cmake_minimum_required(VERSION 3.20)
project(lsv_model VERSION 0.1.0 LANGUAGES CXX)

option(LSV_BUILD_PYTHON "Build the lsv_cpp Python extension" ON)
option(LSV_BUILD_TESTS "Build C++ tests" ON)

add_library(lsv_compile_options INTERFACE)
target_compile_features(lsv_compile_options INTERFACE cxx_std_20)
if(MSVC)
  target_compile_options(lsv_compile_options INTERFACE /W4 /WX)
else()
  target_compile_options(
    lsv_compile_options
    INTERFACE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
  )
endif()

add_library(
  lsv_core STATIC
  src/cpp/src/barrier_pricer.cpp
  src/cpp/src/brownian_bridge.cpp
  src/cpp/src/conditional_expectation.cpp
  src/cpp/src/heston_qem.cpp
  src/cpp/src/leverage_surface.cpp
  src/cpp/src/lsv_simulator.cpp
  src/cpp/src/qem_step.cpp
)
set_target_properties(lsv_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(lsv_core PUBLIC lsv_compile_options)
target_include_directories(lsv_core PUBLIC src/cpp/include)

if(LSV_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(lsv_cpp src/cpp/bindings/pybind_module.cpp)
  target_link_libraries(lsv_cpp PRIVATE lsv_core)
  target_include_directories(lsv_cpp PRIVATE src/cpp/include)
  install(TARGETS lsv_cpp LIBRARY DESTINATION . RUNTIME DESTINATION .)
endif()

if(LSV_BUILD_TESTS)
  enable_testing()
  add_executable(lsv_cpp_smoke tests/cpp/test_smoke.cpp)
  target_link_libraries(lsv_cpp_smoke PRIVATE lsv_compile_options)
  target_include_directories(lsv_cpp_smoke PRIVATE src/cpp/include)
  add_test(NAME lsv_cpp_smoke COMMAND lsv_cpp_smoke)

  add_executable(lsv_heston_qem_test tests/cpp/test_heston_qem.cpp)
  target_link_libraries(lsv_heston_qem_test PRIVATE lsv_core)
  add_test(NAME lsv_heston_qem_test COMMAND lsv_heston_qem_test)

  add_executable(
    lsv_conditional_expectation_test tests/cpp/test_conditional_expectation.cpp
  )
  target_link_libraries(lsv_conditional_expectation_test PRIVATE lsv_core)
  add_test(
    NAME lsv_conditional_expectation_test
    COMMAND lsv_conditional_expectation_test
  )

  add_executable(lsv_simulator_test tests/cpp/test_lsv_simulator.cpp)
  target_link_libraries(lsv_simulator_test PRIVATE lsv_core)
  add_test(NAME lsv_simulator_test COMMAND lsv_simulator_test)

  add_executable(lsv_barrier_pricer_test tests/cpp/test_barrier_pricer.cpp)
  target_link_libraries(lsv_barrier_pricer_test PRIVATE lsv_core)
  add_test(NAME lsv_barrier_pricer_test COMMAND lsv_barrier_pricer_test)
endif()

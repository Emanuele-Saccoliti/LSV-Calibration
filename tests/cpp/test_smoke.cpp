#include <iostream>

#include "lsv/version.hpp"

int main() {
  if (lsv::version != "0.1.0") {
    std::cerr << "unexpected LSV C++ version\n";
    return 1;
  }
  return 0;
}


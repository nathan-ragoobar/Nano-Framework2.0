//#include <fpm/fixed.hpp>  // For fpm::fixed_16_16
//#include <fpm/math.hpp>   // For fpm::cos
//#include <fpm/ios.hpp>    // For fpm::operator<<
#include "./fpm/fpm.hpp"
#include <iostream>       // For std::cin, std::cout

int main() {
    std::cout << "Please input a number: ";
    fpm::fixed_16_16 x;
    std::cin >> x;
    std::cout << "The cosine of " << x << " radians is: " << cos(x) << std::endl;
    return 0;
}
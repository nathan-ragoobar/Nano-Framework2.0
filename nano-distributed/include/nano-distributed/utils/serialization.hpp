#ifndef NANO_DISTRIBUTED_UTILS_SERIALIZATION_HPP
#define NANO_DISTRIBUTED_UTILS_SERIALIZATION_HPP

#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace nano_distributed {
namespace utils {

// Function to serialize a vector of any type
template <typename T>
std::string serialize(const std::vector<T>& data) {
    std::ostringstream oss;
    for (const auto& item : data) {
        oss << item << " ";
    }
    return oss.str();
}

// Function to deserialize a vector of any type
template <typename T>
std::vector<T> deserialize(const std::string& str) {
    std::istringstream iss(str);
    std::vector<T> data;
    T item;
    while (iss >> item) {
        data.push_back(item);
    }
    return data;
}

// Function to serialize a single object
template <typename T>
std::string serialize(const T& obj) {
    static_assert(std::is_trivially_copyable<T>::value, "Object must be trivially copyable");
    std::ostringstream oss;
    oss.write(reinterpret_cast<const char*>(&obj), sizeof(T));
    return oss.str();
}

// Function to deserialize a single object
template <typename T>
T deserialize(const std::string& str) {
    static_assert(std::is_trivially_copyable<T>::value, "Object must be trivially copyable");
    T obj;
    std::memcpy(&obj, str.data(), sizeof(T));
    return obj;
}

} // namespace utils
} // namespace nano_distributed

#endif // NANO_DISTRIBUTED_UTILS_SERIALIZATION_HPP
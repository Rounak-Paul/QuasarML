#pragma once
#include <cstddef>
#include <stdexcept>

namespace QuasarML {

/** Supported tensor data types; extend for more (float16, int8 etc). */
enum class DataType : int {
    FLOAT32,
    INT32,
    UINT32,
    FLOAT16,
    // TODO: float64, int8, etc.
};

inline size_t DataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return 4;
        case DataType::INT32:   return 4;
        case DataType::UINT32:  return 4;
        case DataType::FLOAT16: return 2;
        default: throw std::runtime_error("Unsupported DataType");
    }
}

inline const char* DataTypeGLSL(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "float";
        case DataType::INT32:   return "int";
        case DataType::UINT32:  return "uint";
        // Extend for float16
        default: throw std::runtime_error("Unsupported DataType for GLSL");
    }
}

} // namespace QuasarML

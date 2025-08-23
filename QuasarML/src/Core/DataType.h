#pragma once
#include <cstddef>
#include <stdexcept>

namespace QuasarML {

/** Supported tensor data types. */
enum class DataType : int {
    UINT32,
    INT32,
    FLOAT32,
    FLOAT64
};

/** Return size in bytes for each data type */
inline size_t DataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::UINT32:  return 4;
        case DataType::INT32:   return 4;
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT64: return 8;
        default: throw std::runtime_error("Unsupported DataType");
    }
}

/** Map data type to its GLSL type string */
inline const char* DataTypeGLSL(DataType dtype) {
    switch (dtype) {
        case DataType::UINT32:  return "uint";
        case DataType::INT32:   return "int";
        case DataType::FLOAT32: return "float";
        case DataType::FLOAT64: return "double"; // requires desktop GLSL 4.00+
        default: throw std::runtime_error("Unsupported DataType for GLSL");
    }
}

} // namespace QuasarML
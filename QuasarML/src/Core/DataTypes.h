#pragma once
#include <cstdint>
using u32 = uint32_t;

enum class DataType : u32 {
    F32 = 0,
    F16,
    I32,
    I16,
    I8,
    U32,
    U16,
    U8
};

constexpr u32 get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return 4;
        case DataType::F16: return 2;
        case DataType::I32: return 4;
        case DataType::I16: return 2;
        case DataType::I8:  return 1;
        case DataType::U32: return 4;
        case DataType::U16: return 2;
        case DataType::U8:  return 1;
        default: return 0;
    }
}

// Only declarations here
const char* dtype_to_glsl_type(DataType dtype);
const char* dtype_to_string(DataType dtype);
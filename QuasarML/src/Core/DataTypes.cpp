#include "DataTypes.h"

const char* dtype_to_glsl_type(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return "float";
        case DataType::F16: return "float16_t";
        case DataType::I32: return "int";
        case DataType::I16: return "int16_t";
        case DataType::I8:  return "int8_t";
        case DataType::U32: return "uint";
        case DataType::U16: return "uint16_t";
        case DataType::U8:  return "uint8_t";
        default: return "float";
    }
}

const char* dtype_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return "float32";
        case DataType::F16: return "float16";
        case DataType::I32: return "int32";
        case DataType::I16: return "int16";
        case DataType::I8:  return "int8";
        case DataType::U32: return "uint32";
        case DataType::U16: return "uint16";
        case DataType::U8:  return "uint8";
        default: return "unknown";
    }
}
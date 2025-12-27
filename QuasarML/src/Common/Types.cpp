#include "Types.h"

namespace QuasarML {

const char* dtype_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return "F32";
        case DataType::F16: return "F16";
        case DataType::I32: return "I32";
        case DataType::I16: return "I16";
        case DataType::I8:  return "I8";
        case DataType::U32: return "U32";
        case DataType::U16: return "U16";
        case DataType::U8:  return "U8";
    }
    return "Unknown";
}

const char* dtype_to_glsl(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return "float";
        case DataType::F16: return "float16_t";
        case DataType::I32: return "int";
        case DataType::I16: return "int16_t";
        case DataType::I8:  return "int8_t";
        case DataType::U32: return "uint";
        case DataType::U16: return "uint16_t";
        case DataType::U8:  return "uint8_t";
    }
    return "float";
}

}

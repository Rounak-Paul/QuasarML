#include "DataTypes.h"

const char* dtype_to_glsl_type(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return "float";
        // Many GLSL environments and shader compilers don't expose 8/16-bit scalar
        // types without enabling specific extensions. Use 32-bit types in
        // generated GLSL so kernels compile broadly. The runtime still interprets
        // buffers according to the tensor element size (host/device packing).
        case DataType::F16: return "float";  // promote half to float in shaders
        case DataType::I32: return "int";
        case DataType::I16: return "int";    // use int for 16-bit signed
        case DataType::I8:  return "int";    // use int for 8-bit signed
        case DataType::U32: return "uint";
        case DataType::U16: return "uint";   // use uint for 16-bit unsigned
        case DataType::U8:  return "uint";   // use uint for 8-bit unsigned
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
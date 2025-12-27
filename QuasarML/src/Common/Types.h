#pragma once

#include <cstdint>
#include <cstddef>

namespace QuasarML {

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;

using b8  = bool;
using b32 = int32_t;

static_assert(sizeof(u8)  == 1, "u8 must be 1 byte");
static_assert(sizeof(u16) == 2, "u16 must be 2 bytes");
static_assert(sizeof(u32) == 4, "u32 must be 4 bytes");
static_assert(sizeof(u64) == 8, "u64 must be 8 bytes");
static_assert(sizeof(i8)  == 1, "i8 must be 1 byte");
static_assert(sizeof(i16) == 2, "i16 must be 2 bytes");
static_assert(sizeof(i32) == 4, "i32 must be 4 bytes");
static_assert(sizeof(i64) == 8, "i64 must be 8 bytes");
static_assert(sizeof(f32) == 4, "f32 must be 4 bytes");
static_assert(sizeof(f64) == 8, "f64 must be 8 bytes");

constexpr u64 INVALID_ID_U64 = 0xFFFFFFFFFFFFFFFFULL;
constexpr u32 INVALID_ID_U32 = 0xFFFFFFFFU;
constexpr u16 INVALID_ID_U16 = 0xFFFFU;
constexpr u8  INVALID_ID_U8  = 0xFFU;

#ifdef QS_BUILD_DLL
    #ifdef _WIN32
        #define QS_API __declspec(dllexport)
    #else
        #define QS_API __attribute__((visibility("default")))
    #endif
#else
    #ifdef _WIN32
        #define QS_API __declspec(dllimport)
    #else
        #define QS_API
    #endif
#endif

#if defined(__clang__) || defined(__GNUC__)
    #define QS_INLINE __attribute__((always_inline)) inline
    #define QS_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
    #define QS_INLINE __forceinline
    #define QS_NOINLINE __declspec(noinline)
#else
    #define QS_INLINE inline
    #define QS_NOINLINE
#endif

template<typename T>
constexpr T qs_min(T a, T b) { return a < b ? a : b; }

template<typename T>
constexpr T qs_max(T a, T b) { return a > b ? a : b; }

template<typename T>
constexpr T qs_clamp(T val, T min_val, T max_val) {
    return qs_min(qs_max(val, min_val), max_val);
}

constexpr u64 align_up(u64 value, u64 alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

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

constexpr u32 dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return 4;
        case DataType::F16: return 2;
        case DataType::I32: return 4;
        case DataType::I16: return 2;
        case DataType::I8:  return 1;
        case DataType::U32: return 4;
        case DataType::U16: return 2;
        case DataType::U8:  return 1;
    }
    return 0;
}

const char* dtype_to_string(DataType dtype);
const char* dtype_to_glsl(DataType dtype);

}

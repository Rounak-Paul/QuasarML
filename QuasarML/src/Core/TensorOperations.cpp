#include "TensorOperations.h"
#include "Accelerator.h"
#include "Tensor.h"
#include "Kernel.h"
#include <cstring>
#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <random>
#include <cmath>
#include <fstream>

namespace QuasarML {

// forward declarations for helpers used by scalar fast-paths
static float half_to_float(uint16_t h);
static std::pair<std::array<u8,4>, u32> make_push_constant(const u8* buf, DataType dtype);

// helper: convert float -> IEEE-754 half (16-bit)
static uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mantissa = x & 0x007FFFFFu;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x00800000u) >> (1 - exp);
        if (mantissa & 0x00001000u) mantissa += 0x00002000u;
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    } else if (exp == 0xFF - (127 - 15)) {
        if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7C00u); // inf
        else return static_cast<uint16_t>(sign | 0x7C00u | (mantissa ? 0x0200u : 0)); // NaN
    } else {
        if (exp > 30) {
            // overflow -> inf
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mantissa >> 13));
        return h;
    }
}

// Generic CPU elementwise helper for equal-shaped tensors. Supports all DataType enum entries.
static std::shared_ptr<Tensor> cpu_elementwise_equal(Accelerator& acc,
                                                    std::shared_ptr<Tensor> a,
                                                    std::shared_ptr<Tensor> b,
                                                    DataType dtype,
                                                    const std::function<void(const void*, const void*, void*, u64)>& op_bytewise) {
    // op_bytewise operates on raw arrays of bytes where element size = get_dtype_size(dtype)
    u64 count = a->get_element_count();
    u32 esize = get_dtype_size(dtype);
    std::vector<u8> a_buf(count * esize);
    std::vector<u8> b_buf(count * esize);
    std::vector<u8> out_buf(count * esize);
    // notify accelerator that a CPU fallback path is executing
    acc.notify_cpu_fallback();
    a->download_data(a_buf.data());
    b->download_data(b_buf.data());
    // call op which should handle element conversions internally
    op_bytewise(a_buf.data(), b_buf.data(), out_buf.data(), count);
    return acc.create_tensor(out_buf.data(), a->get_shape(), dtype, /*device_only=*/false);
}

// Generic CPU broadcast helper for binary ops (+ - * /) across all datatypes.
static std::shared_ptr<Tensor> cpu_broadcast_binary(Accelerator& acc,
                                                   std::shared_ptr<Tensor> a,
                                                   std::shared_ptr<Tensor> b,
                                                   const std::vector<u32>& out_shape,
                                                   DataType dtype,
                                                   char op) {
    // element size
    u32 esz = get_dtype_size(dtype);
    u64 out_count = calculate_element_count(out_shape);

    u64 a_count = a->get_element_count();
    u64 b_count = b->get_element_count();
    std::vector<u8> a_buf(a_count * esz);
    std::vector<u8> b_buf(b_count * esz);
    std::vector<u8> out_buf(out_count * esz);
    // CPU fallback execution instrumentation
    acc.notify_cpu_fallback();
    a->download_data(a_buf.data());
    b->download_data(b_buf.data());

    auto flat_from_idx = [](const std::vector<u32>& shape, const std::vector<u32>& idx)->u64{
        u64 flat = 0; u64 stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            flat += static_cast<u64>(idx[i]) * stride;
            stride *= shape[i];
        }
        return flat;
    };

    std::vector<u32> idx(out_shape.size());
    std::vector<u32> a_shape = a->get_shape();
    std::vector<u32> b_shape = b->get_shape();

    for (u64 i = 0; i < out_count; ++i) {
        // unravel into multi-dim index
        u64 rem = i; u64 denom = out_count;
        for (size_t d = 0; d < out_shape.size(); ++d) {
            denom /= out_shape[d];
            idx[d] = static_cast<u32>(rem / denom);
            rem = rem % denom;
        }

        // map to a/b coords (right align shapes)
        std::vector<u32> a_idx(a_shape.size()), b_idx(b_shape.size());
        for (int d = 0; d < static_cast<int>(a_shape.size()); ++d) {
            int od = static_cast<int>(out_shape.size()) - static_cast<int>(a_shape.size()) + d;
            if (od < 0) a_idx[d] = 0; else a_idx[d] = (a_shape[d] == 1) ? 0 : idx[od];
        }
        for (int d = 0; d < static_cast<int>(b_shape.size()); ++d) {
            int od = static_cast<int>(out_shape.size()) - static_cast<int>(b_shape.size()) + d;
            if (od < 0) b_idx[d] = 0; else b_idx[d] = (b_shape[d] == 1) ? 0 : idx[od];
        }

        u64 a_flat = flat_from_idx(a_shape, a_idx);
        u64 b_flat = flat_from_idx(b_shape, b_idx);

        const u8* ap = a_buf.data() + a_flat * esz;
        const u8* bp = b_buf.data() + b_flat * esz;
        u8* optr = out_buf.data() + i * esz;

        // perform per-dtype operation
        switch (dtype) {
            case DataType::F32: {
                float va; float vb; std::memcpy(&va, ap, 4); std::memcpy(&vb, bp, 4);
                float vr = 0.0f;
                if (op == '+') vr = va + vb; else if (op == '-') vr = va - vb; else if (op == '*') vr = va * vb; else if (op == '/') vr = va / vb;
                std::memcpy(optr, &vr, 4);
                break;
            }
            case DataType::F16: {
                uint16_t ha, hb; std::memcpy(&ha, ap, 2); std::memcpy(&hb, bp, 2);
                float fa = half_to_float(ha); float fb = half_to_float(hb);
                float fr = 0.0f;
                if (op == '+') fr = fa + fb; else if (op == '-') fr = fa - fb; else if (op == '*') fr = fa * fb; else if (op == '/') fr = fa / fb;
                uint16_t hr = float_to_half(fr);
                std::memcpy(optr, &hr, 2);
                break;
            }
            case DataType::I32: {
                int32_t va; int32_t vb; std::memcpy(&va, ap, 4); std::memcpy(&vb, bp, 4);
                int32_t vr = 0;
                if (op == '+') vr = va + vb; else if (op == '-') vr = va - vb; else if (op == '*') vr = va * vb; else if (op == '/') vr = va / vb;
                std::memcpy(optr, &vr, 4);
                break;
            }
            case DataType::U32: {
                uint32_t va; uint32_t vb; std::memcpy(&va, ap, 4); std::memcpy(&vb, bp, 4);
                uint32_t vr = 0;
                if (op == '+') vr = va + vb; else if (op == '-') vr = va - vb; else if (op == '*') vr = va * vb; else if (op == '/') vr = va / vb;
                std::memcpy(optr, &vr, 4);
                break;
            }
            case DataType::I16: {
                int16_t va; int16_t vb; std::memcpy(&va, ap, 2); std::memcpy(&vb, bp, 2);
                int16_t vr = 0;
                if (op == '+') vr = static_cast<int16_t>(va + vb); else if (op == '-') vr = static_cast<int16_t>(va - vb); else if (op == '*') vr = static_cast<int16_t>(va * vb); else if (op == '/') vr = static_cast<int16_t>(va / vb);
                std::memcpy(optr, &vr, 2);
                break;
            }
            case DataType::U16: {
                uint16_t va; uint16_t vb; std::memcpy(&va, ap, 2); std::memcpy(&vb, bp, 2);
                uint16_t vr = 0;
                if (op == '+') vr = static_cast<uint16_t>(va + vb); else if (op == '-') vr = static_cast<uint16_t>(va - vb); else if (op == '*') vr = static_cast<uint16_t>(va * vb); else if (op == '/') vr = static_cast<uint16_t>(va / vb);
                std::memcpy(optr, &vr, 2);
                break;
            }
            case DataType::I8: {
                int8_t va; int8_t vb; std::memcpy(&va, ap, 1); std::memcpy(&vb, bp, 1);
                int8_t vr = 0;
                if (op == '+') vr = static_cast<int8_t>(va + vb); else if (op == '-') vr = static_cast<int8_t>(va - vb); else if (op == '*') vr = static_cast<int8_t>(va * vb); else if (op == '/') vr = static_cast<int8_t>(va / vb);
                std::memcpy(optr, &vr, 1);
                break;
            }
            case DataType::U8: {
                uint8_t va; uint8_t vb; std::memcpy(&va, ap, 1); std::memcpy(&vb, bp, 1);
                uint8_t vr = 0;
                if (op == '+') vr = static_cast<uint8_t>(va + vb); else if (op == '-') vr = static_cast<uint8_t>(va - vb); else if (op == '*') vr = static_cast<uint8_t>(va * vb); else if (op == '/') vr = static_cast<uint8_t>(va / vb);
                std::memcpy(optr, &vr, 1);
                break;
            }
            default:
                break;
        }
    }

    return acc.create_tensor(out_buf.data(), out_shape, dtype, /*device_only=*/false);
}

std::shared_ptr<Tensor> TensorOperations::add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a || !b) throw std::invalid_argument("Tensor pointers cannot be null");
    if (!a->is_valid() || !b->is_valid()) throw std::invalid_argument("All tensors must be valid");

    // scalar tensors are handled via the broadcasting elementwise kernels on-device

    if (!a->is_shape_compatible(*b)) {
        // try general broadcasting
        auto out_shape = compute_broadcast_shape(a->get_shape(), b->get_shape());
        if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");

        DataType dtype = a->get_dtype();
        // If accelerator is in CPU mode, perform CPU broadcasting implementation
        if (!_accelerator.use_gpu()) {
            // CPU broadcast implementation supporting all datatypes
            return cpu_broadcast_binary(_accelerator, a, b, out_shape, dtype, '+');
        }

        auto result = _accelerator.create_tensor(out_shape, dtype);

        // prepare meta buffer: layout vec of uints as described in generator
        u32 max_rank = static_cast<u32>(std::max(a->get_rank(), b->get_rank()));
        std::vector<u32> meta;
        meta.reserve(4 + max_rank * 6);
        meta.push_back(max_rank);
        meta.push_back(static_cast<u32>(a->get_rank()));
        meta.push_back(static_cast<u32>(b->get_rank()));
        meta.push_back(static_cast<u32>(out_shape.size()));

        auto pad_dims = [&](const std::vector<u32>& s) {
            std::vector<u32> padded(max_rank, 1);
            for (size_t i = 0; i < s.size(); ++i) padded[max_rank - s.size() + i] = s[i];
            return padded;
        };

        auto a_p = pad_dims(a->get_shape());
        auto b_p = pad_dims(b->get_shape());
        auto out_p = pad_dims(out_shape);

        // push dims
        for (u32 v : a_p) meta.push_back(v);
        for (u32 v : b_p) meta.push_back(v);
        for (u32 v : out_p) meta.push_back(v);

        // compute strides (padded)
        auto a_strides = compute_strides_padded(a->get_shape(), max_rank);
        auto b_strides = compute_strides_padded(b->get_shape(), max_rank);
        auto out_strides = compute_strides_padded(out_shape, max_rank);
        for (u32 v : a_strides) meta.push_back(static_cast<u32>(v));
        for (u32 v : b_strides) meta.push_back(static_cast<u32>(v));
        for (u32 v : out_strides) meta.push_back(static_cast<u32>(v));

    // create device-only tensor for meta (keep meta on GPU)
    auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, true);

        std::string kernel_name = get_kernel_name_for_dtype("add_broadcast", dtype);
        std::string glsl_source = generate_elementwise_kernel_source("data_a[a_index] + data_b[b_index]", dtype, false, true, max_rank);
        auto kernel = get_or_create_kernel(kernel_name, glsl_source, 4);
        u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(calculate_element_count(out_shape)));
        _accelerator.execute(kernel, {a, b, result, meta_tensor}, dispatch_size);
        return result;
    }

    if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");

    if (!_accelerator.use_gpu()) {
        // CPU generic elementwise for equal-shaped tensors (supports all datatypes)
            // Provide a wrapper that operates on raw byte arrays for the tensor's dtype
            auto op_wrapper = [this, a](const void* A, const void* B, void* O, u64 cnt) {
                DataType dt = a->get_dtype();
                u32 esz = get_dtype_size(dt);
                for (u64 i = 0; i < cnt; ++i) {
                    const u8* ap = reinterpret_cast<const u8*>(A) + i * esz;
                    const u8* bp = reinterpret_cast<const u8*>(B) + i * esz;
                    u8* outp = reinterpret_cast<u8*>(O) + i * esz;
                    switch (dt) {
                        case DataType::F32: {
                            float va, vb; std::memcpy(&va, ap, 4); std::memcpy(&vb, bp, 4);
                            float r = va + vb; std::memcpy(outp, &r, 4);
                            break;
                        }
                        case DataType::F16: {
                            uint16_t ha, hb; std::memcpy(&ha, ap, 2); std::memcpy(&hb, bp, 2);
                            float fa = half_to_float(ha); float fb = half_to_float(hb);
                            float fr = fa + fb; uint16_t hr = float_to_half(fr);
                            std::memcpy(outp, &hr, 2);
                            break;
                        }
                        case DataType::I32: {
                            int32_t va, vb; std::memcpy(&va, ap, 4); std::memcpy(&vb, bp, 4);
                            int32_t r = va + vb; std::memcpy(outp, &r, 4);
                            break;
                        }
                        case DataType::U32: {
                            uint32_t va, vb; std::memcpy(&va, ap, 4); std::memcpy(&vb, bp, 4);
                            uint32_t r = va + vb; std::memcpy(outp, &r, 4);
                            break;
                        }
                        case DataType::I16: {
                            int16_t va, vb; std::memcpy(&va, ap, 2); std::memcpy(&vb, bp, 2);
                            int16_t r = static_cast<int16_t>(va + vb); std::memcpy(outp, &r, 2);
                            break;
                        }
                        case DataType::U16: {
                            uint16_t va, vb; std::memcpy(&va, ap, 2); std::memcpy(&vb, bp, 2);
                            uint16_t r = static_cast<uint16_t>(va + vb); std::memcpy(outp, &r, 2);
                            break;
                        }
                        case DataType::I8: {
                            int8_t va, vb; std::memcpy(&va, ap, 1); std::memcpy(&vb, bp, 1);
                            int8_t r = static_cast<int8_t>(va + vb); std::memcpy(outp, &r, 1);
                            break;
                        }
                        case DataType::U8: {
                            uint8_t va, vb; std::memcpy(&va, ap, 1); std::memcpy(&vb, bp, 1);
                            uint8_t r = static_cast<uint8_t>(va + vb); std::memcpy(outp, &r, 1);
                            break;
                        }
                        default:
                            break;
                    }
                }
            };
        return cpu_elementwise_equal(_accelerator, a, b, a->get_dtype(), op_wrapper);
    }

    auto result = _accelerator.create_tensor(a->get_shape(), a->get_dtype());
    std::string kernel_name = get_kernel_name_for_dtype("add", a->get_dtype());
    std::string glsl_source = generate_elementwise_kernel_source("data_a[index] + data_b[index]", a->get_dtype());
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3);
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    _accelerator.execute(kernel, {a, b, result}, dispatch_size);
    return result;
}


std::shared_ptr<Tensor> TensorOperations::sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a || !b) throw std::invalid_argument("Tensor pointers cannot be null");
    if (!a->is_valid() || !b->is_valid()) throw std::invalid_argument("All tensors must be valid");

    // scalar tensors are handled via the broadcasting elementwise kernels on-device

    if (!a->is_shape_compatible(*b)) {
        auto out_shape = compute_broadcast_shape(a->get_shape(), b->get_shape());
        if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");

        DataType dtype = a->get_dtype();
        if (!_accelerator.use_gpu()) {
            return cpu_broadcast_binary(_accelerator, a, b, out_shape, dtype, '-');
        }

        auto result = _accelerator.create_tensor(out_shape, dtype);

        u32 max_rank = static_cast<u32>(std::max(a->get_rank(), b->get_rank()));
        std::vector<u32> meta;
        meta.reserve(4 + max_rank * 6);
        meta.push_back(max_rank);
        meta.push_back(static_cast<u32>(a->get_rank()));
        meta.push_back(static_cast<u32>(b->get_rank()));
        meta.push_back(static_cast<u32>(out_shape.size()));

        auto pad_dims = [&](const std::vector<u32>& s) {
            std::vector<u32> padded(max_rank, 1);
            for (size_t i = 0; i < s.size(); ++i) padded[max_rank - s.size() + i] = s[i];
            return padded;
        };

        auto a_p = pad_dims(a->get_shape());
        auto b_p = pad_dims(b->get_shape());
        auto out_p = pad_dims(out_shape);

        for (u32 v : a_p) meta.push_back(v);
        for (u32 v : b_p) meta.push_back(v);
        for (u32 v : out_p) meta.push_back(v);

        auto a_strides = compute_strides_padded(a->get_shape(), max_rank);
        auto b_strides = compute_strides_padded(b->get_shape(), max_rank);
        auto out_strides = compute_strides_padded(out_shape, max_rank);
        for (u32 v : a_strides) meta.push_back(static_cast<u32>(v));
        for (u32 v : b_strides) meta.push_back(static_cast<u32>(v));
        for (u32 v : out_strides) meta.push_back(static_cast<u32>(v));

    auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, true);

        std::string kernel_name = get_kernel_name_for_dtype("sub_broadcast", dtype);
        std::string glsl_source = generate_elementwise_kernel_source("data_a[a_index] - data_b[b_index]", dtype, false, true, max_rank);
        auto kernel = get_or_create_kernel(kernel_name, glsl_source, 4);
        u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(calculate_element_count(out_shape)));
        _accelerator.execute(kernel, {a, b, result, meta_tensor}, dispatch_size);
        return result;
    }

    if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");

    if (!_accelerator.use_gpu()) {
            auto op_wrapper = [this, a](const void* A, const void* B, void* O, u64 cnt) {
                DataType dt = a->get_dtype(); u32 esz = get_dtype_size(dt);
                for (u64 i=0;i<cnt;++i) {
                    const u8* ap = reinterpret_cast<const u8*>(A) + i * esz;
                    const u8* bp = reinterpret_cast<const u8*>(B) + i * esz;
                    u8* op = reinterpret_cast<u8*>(O) + i * esz;
                    switch (dt) {
                        case DataType::F32: { float va; float vb; std::memcpy(&va, ap, 4); std::memcpy(&vb, bp, 4); float r = va - vb; std::memcpy(op, &r, 4); break; }
                        case DataType::F16: { uint16_t ha,hb; std::memcpy(&ha,ap,2); std::memcpy(&hb,bp,2); float fr = half_to_float(ha) - half_to_float(hb); uint16_t hr = float_to_half(fr); std::memcpy(op,&hr,2); break; }
                        case DataType::I32: { int32_t va; int32_t vb; std::memcpy(&va,ap,4); std::memcpy(&vb,bp,4); int32_t r = va - vb; std::memcpy(op,&r,4); break; }
                        case DataType::U32: { uint32_t va; uint32_t vb; std::memcpy(&va,ap,4); std::memcpy(&vb,bp,4); uint32_t r = va - vb; std::memcpy(op,&r,4); break; }
                        case DataType::I16: { int16_t va; int16_t vb; std::memcpy(&va,ap,2); std::memcpy(&vb,bp,2); int16_t r = static_cast<int16_t>(va - vb); std::memcpy(op,&r,2); break; }
                        case DataType::U16: { uint16_t va; uint16_t vb; std::memcpy(&va,ap,2); std::memcpy(&vb,bp,2); uint16_t r = static_cast<uint16_t>(va - vb); std::memcpy(op,&r,2); break; }
                        case DataType::I8: { int8_t va; int8_t vb; std::memcpy(&va,ap,1); std::memcpy(&vb,bp,1); int8_t r = static_cast<int8_t>(va - vb); std::memcpy(op,&r,1); break; }
                        case DataType::U8: { uint8_t va; uint8_t vb; std::memcpy(&va,ap,1); std::memcpy(&vb,bp,1); uint8_t r = static_cast<uint8_t>(va - vb); std::memcpy(op,&r,1); break; }
                        default: break;
                    }
                }
            };
        return cpu_elementwise_equal(_accelerator, a, b, a->get_dtype(), op_wrapper);
    }

    auto result = _accelerator.create_tensor(a->get_shape(), a->get_dtype());
    std::string kernel_name = get_kernel_name_for_dtype("sub", a->get_dtype());
    std::string glsl_source = generate_elementwise_kernel_source("data_a[index] - data_b[index]", a->get_dtype());
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3);
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    _accelerator.execute(kernel, {a, b, result}, dispatch_size);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a || !b) throw std::invalid_argument("Tensor pointers cannot be null");
    if (!a->is_valid() || !b->is_valid()) throw std::invalid_argument("All tensors must be valid");

    // scalar tensors are handled via the broadcasting elementwise kernels on-device

    if (!a->is_shape_compatible(*b)) {
        auto out_shape = compute_broadcast_shape(a->get_shape(), b->get_shape());
        if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");

        DataType dtype = a->get_dtype();
        if (!_accelerator.use_gpu()) {
            return cpu_broadcast_binary(_accelerator, a, b, out_shape, dtype, '*');
        }

        auto result = _accelerator.create_tensor(out_shape, dtype);

        u32 max_rank = static_cast<u32>(std::max(a->get_rank(), b->get_rank()));
        std::vector<u32> meta;
        meta.reserve(4 + max_rank * 6);
        meta.push_back(max_rank);
        meta.push_back(static_cast<u32>(a->get_rank()));
        meta.push_back(static_cast<u32>(b->get_rank()));
        meta.push_back(static_cast<u32>(out_shape.size()));

        auto pad_dims = [&](const std::vector<u32>& s) {
            std::vector<u32> padded(max_rank, 1);
            for (size_t i = 0; i < s.size(); ++i) padded[max_rank - s.size() + i] = s[i];
            return padded;
        };

        auto a_p = pad_dims(a->get_shape());
        auto b_p = pad_dims(b->get_shape());
        auto out_p = pad_dims(out_shape);

        for (u32 v : a_p) meta.push_back(v);
        for (u32 v : b_p) meta.push_back(v);
        for (u32 v : out_p) meta.push_back(v);

        auto a_strides = compute_strides_padded(a->get_shape(), max_rank);
        auto b_strides = compute_strides_padded(b->get_shape(), max_rank);
        auto out_strides = compute_strides_padded(out_shape, max_rank);
        for (u32 v : a_strides) meta.push_back(static_cast<u32>(v));
        for (u32 v : b_strides) meta.push_back(static_cast<u32>(v));
        for (u32 v : out_strides) meta.push_back(static_cast<u32>(v));

    auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, true);

        std::string kernel_name = get_kernel_name_for_dtype("mul_broadcast", dtype);
        std::string glsl_source = generate_elementwise_kernel_source("data_a[a_index] * data_b[b_index]", dtype, false, true, max_rank);
        auto kernel = get_or_create_kernel(kernel_name, glsl_source, 4);
        u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(calculate_element_count(out_shape)));
        _accelerator.execute(kernel, {a, b, result, meta_tensor}, dispatch_size);
        return result;
    }

    if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");

    if (!_accelerator.use_gpu()) {
        // instrumentation
        _accelerator.notify_cpu_fallback();
        if (a->get_dtype() != DataType::F32 || b->get_dtype() != DataType::F32) throw std::runtime_error("CPU fallback only supports F32");
        u64 count = a->get_element_count(); std::vector<float> a_buf(count), b_buf(count), out_buf(count);
        a->download_data(a_buf.data()); b->download_data(b_buf.data());
        for (u64 i=0;i<count;++i) out_buf[i]=a_buf[i]*b_buf[i];
        return _accelerator.create_tensor(out_buf.data(), a->get_shape(), a->get_dtype(), false);
    }

    auto result = _accelerator.create_tensor(a->get_shape(), a->get_dtype());
    std::string kernel_name = get_kernel_name_for_dtype("mul", a->get_dtype());
    std::string glsl_source = generate_elementwise_kernel_source("data_a[index] * data_b[index]", a->get_dtype());
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3);
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    _accelerator.execute(kernel, {a, b, result}, dispatch_size);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a || !b) throw std::invalid_argument("Tensor pointers cannot be null");
    if (!a->is_valid() || !b->is_valid()) throw std::invalid_argument("All tensors must be valid");

    // scalar tensors are handled via the broadcasting elementwise kernels on-device

    if (!a->is_shape_compatible(*b)) {
        auto out_shape = compute_broadcast_shape(a->get_shape(), b->get_shape());
        if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");

        DataType dtype = a->get_dtype();
        if (!_accelerator.use_gpu()) {
            return cpu_broadcast_binary(_accelerator, a, b, out_shape, dtype, '/');
        }

        auto result = _accelerator.create_tensor(out_shape, dtype);

        u32 max_rank = static_cast<u32>(std::max(a->get_rank(), b->get_rank()));
        std::vector<u32> meta;
        meta.reserve(4 + max_rank * 6);
        meta.push_back(max_rank);
        meta.push_back(static_cast<u32>(a->get_rank()));
        meta.push_back(static_cast<u32>(b->get_rank()));
        meta.push_back(static_cast<u32>(out_shape.size()));

        auto pad_dims = [&](const std::vector<u32>& s) {
            std::vector<u32> padded(max_rank, 1);
            for (size_t i = 0; i < s.size(); ++i) padded[max_rank - s.size() + i] = s[i];
            return padded;
        };

        auto a_p = pad_dims(a->get_shape());
        auto b_p = pad_dims(b->get_shape());
        auto out_p = pad_dims(out_shape);

        for (u32 v : a_p) meta.push_back(v);
        for (u32 v : b_p) meta.push_back(v);
        for (u32 v : out_p) meta.push_back(v);

        auto a_strides = compute_strides_padded(a->get_shape(), max_rank);
        auto b_strides = compute_strides_padded(b->get_shape(), max_rank);
        auto out_strides = compute_strides_padded(out_shape, max_rank);
        for (u32 v : a_strides) meta.push_back(static_cast<u32>(v));
        for (u32 v : b_strides) meta.push_back(static_cast<u32>(v));
        for (u32 v : out_strides) meta.push_back(static_cast<u32>(v));

    auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, true);

        std::string kernel_name = get_kernel_name_for_dtype("div_broadcast", dtype);
        std::string glsl_source = generate_elementwise_kernel_source("data_a[a_index] / data_b[b_index]", dtype, false, true, max_rank);
        auto kernel = get_or_create_kernel(kernel_name, glsl_source, 4);
        u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(calculate_element_count(out_shape)));
        _accelerator.execute(kernel, {a, b, result, meta_tensor}, dispatch_size);
        return result;
    }

    if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");

    auto result = _accelerator.create_tensor(a->get_shape(), a->get_dtype());
    std::string kernel_name = get_kernel_name_for_dtype("div", a->get_dtype());
    std::string glsl_source = generate_elementwise_kernel_source("data_a[index] / data_b[index]", a->get_dtype());
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3);
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    _accelerator.execute(kernel, {a, b, result}, dispatch_size);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::add_scalar(std::shared_ptr<Tensor> tensor, float scalar) {
    if (!tensor->is_valid()) {
        throw std::invalid_argument("Tensor must be valid");
    }
    
    DataType dtype = tensor->get_dtype();
    if (!_accelerator.use_gpu()) {
        // Generic scalar add on CPU for all data types
        auto op_scalar_add = [this, tensor, scalar](const void* A, const void* B, void* O, u64 cnt) {
            // B is unused; scalar captured
            DataType dt = tensor->get_dtype(); u32 esz = get_dtype_size(dt);
            for (u64 i=0;i<cnt;++i) {
                const u8* ap = reinterpret_cast<const u8*>(A) + i * esz;
                u8* op = reinterpret_cast<u8*>(O) + i * esz;
                switch (dt) {
                    case DataType::F32: { float v; std::memcpy(&v, ap, 4); float r = v + scalar; std::memcpy(op, &r, 4); break; }
                    case DataType::F16: { uint16_t h; std::memcpy(&h, ap, 2); float fv = half_to_float(h); float fr = fv + scalar; uint16_t hr = float_to_half(fr); std::memcpy(op, &hr, 2); break; }
                    case DataType::I32: { int32_t v; std::memcpy(&v, ap, 4); int32_t r = static_cast<int32_t>(v + static_cast<int32_t>(scalar)); std::memcpy(op, &r, 4); break; }
                    case DataType::U32: { uint32_t v; std::memcpy(&v, ap, 4); uint32_t r = static_cast<uint32_t>(v + static_cast<uint32_t>(scalar)); std::memcpy(op, &r, 4); break; }
                    case DataType::I16: { int16_t v; std::memcpy(&v, ap, 2); int16_t r = static_cast<int16_t>(v + static_cast<int16_t>(scalar)); std::memcpy(op, &r, 2); break; }
                    case DataType::U16: { uint16_t v; std::memcpy(&v, ap, 2); uint16_t r = static_cast<uint16_t>(v + static_cast<uint16_t>(scalar)); std::memcpy(op, &r, 2); break; }
                    case DataType::I8: { int8_t v; std::memcpy(&v, ap, 1); int8_t r = static_cast<int8_t>(v + static_cast<int8_t>(scalar)); std::memcpy(op, &r, 1); break; }
                    case DataType::U8: { uint8_t v; std::memcpy(&v, ap, 1); uint8_t r = static_cast<uint8_t>(v + static_cast<uint8_t>(scalar)); std::memcpy(op, &r, 1); break; }
                    default: break;
                }
            }
        };
        return cpu_elementwise_equal(_accelerator, tensor, tensor, dtype, op_scalar_add);
    }

    auto result = _accelerator.create_tensor(tensor->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("add_scalar", dtype);
    std::string glsl_source = generate_elementwise_kernel_source("data_in[index] + pc.scalar", dtype, true);
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2, sizeof(float));
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    
    auto pc = make_push_constant(reinterpret_cast<const u8*>(&scalar), tensor->get_dtype());
    _accelerator.execute(kernel, {tensor, result}, dispatch_size, 1, 1, pc.first.data());
    return result;
}

std::shared_ptr<Tensor> TensorOperations::mul_scalar(std::shared_ptr<Tensor> tensor, float scalar) {
    if (!tensor->is_valid()) {
        throw std::invalid_argument("Tensor must be valid");
    }
    
    DataType dtype = tensor->get_dtype();
    if (!_accelerator.use_gpu()) {
        auto op_scalar_mul = [this, tensor, scalar](const void* A, const void* B, void* O, u64 cnt) {
            DataType dt = tensor->get_dtype(); u32 esz = get_dtype_size(dt);
            for (u64 i=0;i<cnt;++i) {
                const u8* ap = reinterpret_cast<const u8*>(A) + i * esz;
                u8* op = reinterpret_cast<u8*>(O) + i * esz;
                switch (dt) {
                    case DataType::F32: { float v; std::memcpy(&v, ap, 4); float r = v * scalar; std::memcpy(op, &r, 4); break; }
                    case DataType::F16: { uint16_t h; std::memcpy(&h, ap, 2); float fv = half_to_float(h); float fr = fv * scalar; uint16_t hr = float_to_half(fr); std::memcpy(op, &hr, 2); break; }
                    case DataType::I32: { int32_t v; std::memcpy(&v, ap, 4); int32_t r = static_cast<int32_t>(v * static_cast<int32_t>(scalar)); std::memcpy(op, &r, 4); break; }
                    case DataType::U32: { uint32_t v; std::memcpy(&v, ap, 4); uint32_t r = static_cast<uint32_t>(v * static_cast<uint32_t>(scalar)); std::memcpy(op, &r, 4); break; }
                    case DataType::I16: { int16_t v; std::memcpy(&v, ap, 2); int16_t r = static_cast<int16_t>(v * static_cast<int16_t>(scalar)); std::memcpy(op, &r, 2); break; }
                    case DataType::U16: { uint16_t v; std::memcpy(&v, ap, 2); uint16_t r = static_cast<uint16_t>(v * static_cast<uint16_t>(scalar)); std::memcpy(op, &r, 2); break; }
                    case DataType::I8: { int8_t v; std::memcpy(&v, ap, 1); int8_t r = static_cast<int8_t>(v * static_cast<int8_t>(scalar)); std::memcpy(op, &r, 1); break; }
                    case DataType::U8: { uint8_t v; std::memcpy(&v, ap, 1); uint8_t r = static_cast<uint8_t>(v * static_cast<uint8_t>(scalar)); std::memcpy(op, &r, 1); break; }
                    default: break;
                }
            }
        };
        return cpu_elementwise_equal(_accelerator, tensor, tensor, dtype, op_scalar_mul);
    }

    auto result = _accelerator.create_tensor(tensor->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("mul_scalar", dtype);
    std::string glsl_source = generate_elementwise_kernel_source("data_in[index] * pc.scalar", dtype, true);
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2, sizeof(float));
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    
    auto pc2 = make_push_constant(reinterpret_cast<const u8*>(&scalar), tensor->get_dtype());
    _accelerator.execute(kernel, {tensor, result}, dispatch_size, 1, 1, pc2.first.data());
    return result;
}

std::shared_ptr<Tensor> TensorOperations::relu(std::shared_ptr<Tensor> tensor) {
    if (!tensor->is_valid()) {
        throw std::invalid_argument("Tensor must be valid");
    }
    
    DataType dtype = tensor->get_dtype();
    if (!_accelerator.use_gpu()) {
        auto op_relu = [this, tensor](const void* A, const void* B, void* O, u64 cnt) {
            DataType dt = tensor->get_dtype(); u32 esz = get_dtype_size(dt);
            for (u64 i=0;i<cnt;++i) {
                const u8* ap = reinterpret_cast<const u8*>(A) + i * esz;
                u8* op = reinterpret_cast<u8*>(O) + i * esz;
                switch (dt) {
                    case DataType::F32: { float v; std::memcpy(&v, ap, 4); float r = std::max(0.0f, v); std::memcpy(op, &r, 4); break; }
                    case DataType::F16: { uint16_t h; std::memcpy(&h, ap, 2); float fv = half_to_float(h); float fr = std::max(0.0f, fv); uint16_t hr = float_to_half(fr); std::memcpy(op, &hr, 2); break; }
                    case DataType::I32: { int32_t v; std::memcpy(&v, ap, 4); int32_t r = std::max(0, v); std::memcpy(op, &r, 4); break; }
                    case DataType::U32: { uint32_t v; std::memcpy(&v, ap, 4); uint32_t r = v; std::memcpy(op, &r, 4); break; }
                    case DataType::I16: { int16_t v; std::memcpy(&v, ap, 2); int16_t r = static_cast<int16_t>(std::max(0, static_cast<int>(v))); std::memcpy(op, &r, 2); break; }
                    case DataType::U16: { uint16_t v; std::memcpy(&v, ap, 2); uint16_t r = v; std::memcpy(op, &r, 2); break; }
                    case DataType::I8: { int8_t v; std::memcpy(&v, ap, 1); int8_t r = static_cast<int8_t>(std::max(0, static_cast<int>(v))); std::memcpy(op, &r, 1); break; }
                    case DataType::U8: { uint8_t v; std::memcpy(&v, ap, 1); uint8_t r = v; std::memcpy(op, &r, 1); break; }
                    default: break;
                }
            }
        };
        return cpu_elementwise_equal(_accelerator, tensor, tensor, dtype, op_relu);
    }

    auto result = _accelerator.create_tensor(tensor->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("relu", dtype);
    std::string glsl_source = generate_relu_kernel_source(dtype);
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2);
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    
    _accelerator.execute(kernel, {tensor, result}, dispatch_size);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    validate_tensor_shape_2d(a);
    validate_tensor_shape_2d(b);
    
    auto a_shape = a->get_shape();
    auto b_shape = b->get_shape();
    
    if (a_shape[1] != b_shape[0]) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    if (a->get_dtype() != b->get_dtype()) {
        throw std::invalid_argument("Both matrices must have the same data type");
    }
    
    DataType dtype = a->get_dtype();
    std::vector<u32> result_shape = {a_shape[0], b_shape[1]};
    
    if (!_accelerator.use_gpu()) {
        _accelerator.notify_cpu_fallback();
        if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32");
        u32 M = a_shape[0]; u32 K = a_shape[1]; u32 N = b_shape[1]; std::vector<float> A(M*K), B(K*N), C(M*N);
        a->download_data(A.data()); b->download_data(B.data()); for (u32 i=0;i<M;++i) for (u32 j=0;j<N;++j){ float s=0; for (u32 k=0;k<K;++k) s+=A[i*K+k]*B[k*N+j]; C[i*N+j]=s; }
        return _accelerator.create_tensor(C.data(), result_shape, dtype, false);
    }

    auto result = _accelerator.create_tensor(result_shape, dtype);
    
    bool use_small_kernel = (a_shape[0] < 64 || b_shape[1] < 64 || a_shape[1] < 64);
    std::string kernel_name = get_kernel_name_for_dtype(use_small_kernel ? "matmul_small" : "matmul", dtype);
    std::string glsl_source = use_small_kernel ? generate_matmul_small_kernel_source(dtype) : generate_matmul_kernel_source(dtype);
    
    struct MatMulPushConstants {
        u32 M, N, K;
    } push_data = {a_shape[0], b_shape[1], a_shape[1]};

    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3, sizeof(MatMulPushConstants));
    
    u32 dispatch_x, dispatch_y;
    if (use_small_kernel) {
        dispatch_x = (a_shape[0] + 15) / 16;
        dispatch_y = (b_shape[1] + 15) / 16;
    } else {
        u32 tile_size = 16;
        u32 block_size = 4;
        u32 effective_tile = tile_size * block_size;
        dispatch_x = (a_shape[0] + effective_tile - 1) / effective_tile;
        dispatch_y = (b_shape[1] + effective_tile - 1) / effective_tile;
    }
    
    _accelerator.execute(kernel, {a, b, result}, dispatch_x, dispatch_y, 1, &push_data);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::transpose(std::shared_ptr<Tensor> tensor) {
    validate_tensor_shape_2d(tensor);
    
    auto input_shape = tensor->get_shape();
    std::vector<u32> result_shape = {input_shape[1], input_shape[0]};
    
    DataType dtype = tensor->get_dtype();
    if (!_accelerator.use_gpu()) {
        _accelerator.notify_cpu_fallback();
        if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32");
        u32 rows = input_shape[0], cols = input_shape[1]; std::vector<float> in(rows*cols), out(rows*cols);
        tensor->download_data(in.data()); for (u32 r=0;r<rows;++r) for (u32 c=0;c<cols;++c) out[c*rows + r] = in[r*cols + c]; return _accelerator.create_tensor(out.data(), result_shape, dtype, false);
    }

    auto result = _accelerator.create_tensor(result_shape, dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("transpose", dtype);
    std::string glsl_source = generate_transpose_kernel_source(dtype);
    
    struct TransposePushConstants {
        u32 rows, cols;
    } push_data = {input_shape[0], input_shape[1]};

    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2, sizeof(TransposePushConstants));
    
    u32 dispatch_x = (input_shape[0] + 31) / 32;
    u32 dispatch_y = (input_shape[1] + 31) / 32;
    
    _accelerator.execute(kernel, {tensor, result}, dispatch_x, dispatch_y, 1, &push_data);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::sum_axis(std::shared_ptr<Tensor> tensor, u32 axis) {
    // General N-d support: GPU two-pass reduction (no CPU fallback)
    auto input_shape = tensor->get_shape();
    u32 rank = tensor->get_rank();
    if (axis >= rank) throw std::invalid_argument("Axis out of range");

    if (rank == 2) {
        validate_tensor_shape_2d(tensor);
        u32 expected_result_size = (axis == 0) ? input_shape[1] : input_shape[0];
        std::vector<u32> result_shape = {expected_result_size};
        DataType dtype = tensor->get_dtype();
            if (!_accelerator.use_gpu()) {
                if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32");
                u64 out_count = calculate_element_count(result_shape); std::vector<float> in(tensor->get_element_count()); tensor->download_data(in.data()); std::vector<float> out(out_count, 0.0f);
                // naive reduce over axis
                auto shape = tensor->get_shape(); std::vector<u32> idx(shape.size()); u64 total = tensor->get_element_count(); for (u64 i=0;i<total;++i){ // unravel
                    u64 rem=i, denom=total; for (size_t d=0; d<shape.size(); ++d){ denom/=shape[d]; idx[d]=static_cast<u32>(rem/denom); rem%=denom; }
                    // compute output index
                    u64 out_flat=0, out_stride=1; for (int d=static_cast<int>(shape.size())-1; d>=0; --d){ if (static_cast<u32>(d)!=axis){ out_flat += static_cast<u64>(idx[d])*out_stride; out_stride *= (d < static_cast<int>(result_shape.size()) ? result_shape[d - (d>axis?1:0)] : 1); } }
                    out[out_flat] += in[i]; }
                return _accelerator.create_tensor(out.data(), result_shape, dtype, false);
            }

        auto result = _accelerator.create_tensor(result_shape, dtype);
        std::string kernel_name = get_kernel_name_for_dtype("sum_axis", dtype);
        std::string glsl_source = generate_sum_axis_kernel_source(dtype);
        struct SumAxisPushConstants { u32 rows, cols, axis; } push_data = {input_shape[0], input_shape[1], axis};
        auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2, sizeof(SumAxisPushConstants));
        u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(expected_result_size);
        _accelerator.execute(kernel, {tensor, result}, dispatch_size, 1, 1, &push_data);
        return result;
    }
    DataType dtype = tensor->get_dtype();
    auto shape = tensor->get_shape();
    std::vector<u32> out_shape; for (u32 i=0;i<shape.size();++i) if (i!=axis) out_shape.push_back(shape[i]);
    if (out_shape.empty()) out_shape.push_back(1); // scalar result
    auto result = _accelerator.create_tensor(out_shape, dtype);

    u32 max_rank = static_cast<u32>(shape.size());
    std::vector<u32> meta; meta.reserve(2 + max_rank*4);
    meta.push_back(max_rank); meta.push_back(axis);
    auto pad_dims = [&](const std::vector<u32>& s) { std::vector<u32> padded(max_rank,1u); size_t off=max_rank-s.size(); for(size_t i=0;i<s.size();++i) padded[off+i]=s[i]; return padded; };
    auto in_p = pad_dims(shape); auto out_p = pad_dims(out_shape);
    auto in_str = compute_strides_padded(shape, max_rank); auto out_str = compute_strides_padded(out_shape, max_rank);
    for (u32 v:in_p) meta.push_back(v); for (u32 v:out_p) meta.push_back(v); for (u32 v:in_str) meta.push_back(v); for (u32 v:out_str) meta.push_back(v);
    auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, true);

    // two-pass reduction parameters
    u32 red_len = shape[axis];
    const u32 LOCAL = 256u;
    u32 group_size = LOCAL; // elements per group processed by first pass
    u32 group_count = (red_len + group_size - 1) / group_size;

    // partials layout: out_count * group_count
    u64 out_count = calculate_element_count(out_shape);
    u64 partials_count = static_cast<u64>(out_count) * static_cast<u64>(group_count);
    auto partials = _accelerator.create_tensor({static_cast<u32>(partials_count)}, dtype, true);

    // first pass kernel
    std::string k1_name = get_kernel_name_for_dtype("reduce_first_pass_sum", dtype);
    std::string k1_src = generate_reduce_axis_first_pass_kernel_source(dtype, "sum", LOCAL);
    auto k1_kernel = get_or_create_kernel(k1_name, k1_src, 3, sizeof(u32)*3);
    struct ReduceFirstPC { u32 group_size; u32 group_count; u32 axis; } reduce_first_pc{group_size, group_count, axis};
    u32 dispatch_x = static_cast<u32>(out_count);
    u32 dispatch_y = group_count;
    _accelerator.execute(k1_kernel, {tensor, partials, meta_tensor}, dispatch_x, dispatch_y, 1, &reduce_first_pc);

    // second pass kernel
    std::string k2_name = get_kernel_name_for_dtype("reduce_second_pass_sum", dtype);
    std::string k2_src = generate_reduce_axis_second_pass_kernel_source(dtype, "sum");
    auto k2_kernel = get_or_create_kernel(k2_name, k2_src, 2, sizeof(u32));
    struct ReduceSecondPC { u32 group_count; } reduce_second_pc{group_count};
    u32 dispatch2 = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(out_count));
    _accelerator.execute(k2_kernel, {partials, result}, dispatch2, 1, 1, &reduce_second_pc);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::mean_axis(std::shared_ptr<Tensor> tensor, u32 axis) {
    auto sumt = sum_axis(tensor, axis);
    u32 len = tensor->get_shape()[axis];
    float denom = static_cast<float>(len);
    auto denom_t = _accelerator.create_tensor(&denom, {1}, tensor->get_dtype());
    return div(sumt, denom_t);
}

std::shared_ptr<Tensor> TensorOperations::layer_norm(std::shared_ptr<Tensor> tensor, std::shared_ptr<Tensor> gamma, std::shared_ptr<Tensor> beta, float epsilon) {
    validate_tensor_shape_2d(tensor);
    auto shape = tensor->get_shape();
    u32 batch_size = shape[0];
    u32 feature_dim = shape[1];
    
    if (gamma->get_element_count() != feature_dim || beta->get_element_count() != feature_dim) {
        throw std::invalid_argument("gamma and beta must have same size as last dimension");
    }
    
    DataType dtype = tensor->get_dtype();
    if (!_accelerator.use_gpu()) {
        _accelerator.notify_cpu_fallback();
        if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32");
        
        std::vector<float> x_data(batch_size * feature_dim);
        std::vector<float> gamma_data(feature_dim);
        std::vector<float> beta_data(feature_dim);
        std::vector<float> output(batch_size * feature_dim);
        
        tensor->download_data(x_data.data());
        gamma->download_data(gamma_data.data());
        beta->download_data(beta_data.data());
        
        for (u32 i = 0; i < batch_size; i++) {
            float mean = 0.0f;
            for (u32 j = 0; j < feature_dim; j++) {
                mean += x_data[i * feature_dim + j];
            }
            mean /= feature_dim;
            
            float var = 0.0f;
            for (u32 j = 0; j < feature_dim; j++) {
                float diff = x_data[i * feature_dim + j] - mean;
                var += diff * diff;
            }
            var /= feature_dim;
            float std = std::sqrt(var + epsilon);
            
            for (u32 j = 0; j < feature_dim; j++) {
                float normalized = (x_data[i * feature_dim + j] - mean) / std;
                output[i * feature_dim + j] = gamma_data[j] * normalized + beta_data[j];
            }
        }
        
        return _accelerator.create_tensor(output.data(), shape, dtype, false);
    }
    
    auto result = _accelerator.create_tensor(shape, dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("layer_norm", dtype);
    std::string glsl_source = generate_layer_norm_kernel_source(dtype);
    
    struct LayerNormPushConstants {
        u32 batch_size;
        u32 feature_dim;
        float epsilon;
    } push_data = {batch_size, feature_dim, epsilon};
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 4, sizeof(LayerNormPushConstants));
    
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(batch_size);
    _accelerator.execute(kernel, {tensor, gamma, beta, result}, dispatch_size, 1, 1, &push_data);
    
    return result;
}

std::shared_ptr<Tensor> TensorOperations::min_axis(std::shared_ptr<Tensor> tensor, u32 axis) {
    auto input_shape = tensor->get_shape();
    u32 rank = tensor->get_rank();
    if (axis >= rank) throw std::invalid_argument("Axis out of range");
    if (rank == 2) {
        auto input_shape = tensor->get_shape();
        u32 expected_result_size = (axis == 0) ? input_shape[1] : input_shape[0];
        std::vector<u32> result_shape = {expected_result_size};
        DataType dtype = tensor->get_dtype();
        auto result = _accelerator.create_tensor(result_shape, dtype);
        std::string kernel_name = get_kernel_name_for_dtype("min_axis", dtype);
        std::string glsl = "#version 450\n"
            "layout(local_size_x = 256) in;\n"
            "layout(push_constant) uniform PushConstants { uint rows; uint cols; uint axis; } pc;\n"
            "layout(set = 0, binding = 0, std430) restrict readonly buffer Input { " + std::string(dtype_to_glsl_type(dtype)) + " data_in[]; } ;\n"
            "layout(set = 0, binding = 1, std430) restrict writeonly buffer Output { " + std::string(dtype_to_glsl_type(dtype)) + " data_out[]; } ;\n"
            "void main() { uint index = gl_GlobalInvocationID.x; if (pc.axis == 0) { if (index >= pc.cols) return; "
            + std::string(dtype_to_glsl_type(dtype)) + " best = data_in[index]; for (uint r=1;r<pc.rows;++r) { best = min(best, data_in[r*pc.cols + index]); } data_out[index] = best; } else { if (index >= pc.rows) return; "
            + std::string(dtype_to_glsl_type(dtype)) + " best = data_in[index*pc.cols + 0]; for (uint c=1;c<pc.cols;++c) { best = min(best, data_in[index*pc.cols + c]); } data_out[index] = best; } }\n";
        auto kernel = get_or_create_kernel(kernel_name, glsl, 2, sizeof(u32)*3);
        struct PC { u32 rows, cols, axis; } pc{input_shape[0], input_shape[1], axis};
        u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(expected_result_size);
        _accelerator.execute(kernel, {tensor, result}, dispatch, 1, 1, &pc);
        return result;
    }
    // two-pass min reduction (similar to sum path)
    DataType dtype = tensor->get_dtype(); auto shape = tensor->get_shape(); std::vector<u32> out_shape; for (u32 i=0;i<shape.size();++i) if (i!=axis) out_shape.push_back(shape[i]); auto result = _accelerator.create_tensor(out_shape, dtype);
    u32 max_rank = static_cast<u32>(shape.size()); std::vector<u32> meta; meta.reserve(2+max_rank*4); meta.push_back(max_rank); meta.push_back(axis);
    auto pad_dims = [&](const std::vector<u32>& s) { std::vector<u32> padded(max_rank,1u); size_t off=max_rank-s.size(); for(size_t i=0;i<s.size();++i) padded[off+i]=s[i]; return padded; };
    auto in_p = pad_dims(shape); auto out_p = pad_dims(out_shape); auto in_str = compute_strides_padded(shape, max_rank); auto out_str = compute_strides_padded(out_shape, max_rank);
    for (u32 v:in_p) meta.push_back(v); for (u32 v:out_p) meta.push_back(v); for (u32 v:in_str) meta.push_back(v); for (u32 v:out_str) meta.push_back(v);
    auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, true);
    u32 red_len = shape[axis];
    const u32 LOCAL = 256u;
    u32 group_size = LOCAL;
    u32 group_count = (red_len + group_size - 1)/group_size;
    u64 out_count = calculate_element_count(out_shape);
    u64 partials_count = out_count * static_cast<u64>(group_count);
    auto partials = _accelerator.create_tensor({static_cast<u32>(partials_count)}, dtype, true);

    std::string k1_name_min = get_kernel_name_for_dtype("reduce_first_pass_min", dtype);
    std::string k1_src_min = generate_reduce_axis_first_pass_kernel_source(dtype, "min", LOCAL);
    auto k1_kernel_min = get_or_create_kernel(k1_name_min, k1_src_min, 3, sizeof(u32)*3);
    struct ReduceFirstPCMin { u32 group_size; u32 group_count; u32 axis; } reduce_first_pc_min{group_size, group_count, axis};
    u32 dispatch_x_min = static_cast<u32>(out_count);
    u32 dispatch_y_min = group_count;
    _accelerator.execute(k1_kernel_min, {tensor, partials, meta_tensor}, dispatch_x_min, dispatch_y_min, 1, &reduce_first_pc_min);

    std::string k2_name_min = get_kernel_name_for_dtype("reduce_second_pass_min", dtype);
    std::string k2_src_min = generate_reduce_axis_second_pass_kernel_source(dtype, "min");
    auto k2_kernel_min = get_or_create_kernel(k2_name_min, k2_src_min, 2, sizeof(u32));
    struct ReduceSecondPCMin { u32 group_count; } reduce_second_pc_min{group_count};
    u32 dispatch2_min = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(out_count));
    _accelerator.execute(k2_kernel_min, {partials, result}, dispatch2_min, 1, 1, &reduce_second_pc_min);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::max_axis(std::shared_ptr<Tensor> tensor, u32 axis) {
    auto input_shape = tensor->get_shape();
    u32 rank = tensor->get_rank();
    if (axis >= rank) throw std::invalid_argument("Axis out of range");
    if (rank == 2) {
        auto input_shape = tensor->get_shape();
        u32 expected_result_size = (axis == 0) ? input_shape[1] : input_shape[0];
        std::vector<u32> result_shape = {expected_result_size};
        DataType dtype = tensor->get_dtype();
        auto result = _accelerator.create_tensor(result_shape, dtype);
        std::string kernel_name = get_kernel_name_for_dtype("max_axis", dtype);
        std::string glsl = "#version 450\n"
            "layout(local_size_x = 256) in;\n"
            "layout(push_constant) uniform PushConstants { uint rows; uint cols; uint axis; } pc;\n"
            "layout(set = 0, binding = 0, std430) restrict readonly buffer Input { " + std::string(dtype_to_glsl_type(dtype)) + " data_in[]; } ;\n"
            "layout(set = 0, binding = 1, std430) restrict writeonly buffer Output { " + std::string(dtype_to_glsl_type(dtype)) + " data_out[]; } ;\n"
            "void main() { uint index = gl_GlobalInvocationID.x; if (pc.axis == 0) { if (index >= pc.cols) return; "
            + std::string(dtype_to_glsl_type(dtype)) + " best = data_in[index]; for (uint r=1;r<pc.rows;++r) { best = max(best, data_in[r*pc.cols + index]); } data_out[index] = best; } else { if (index >= pc.rows) return; "
            + std::string(dtype_to_glsl_type(dtype)) + " best = data_in[index*pc.cols + 0]; for (uint c=1;c<pc.cols;++c) { best = max(best, data_in[index*pc.cols + c]); } data_out[index] = best; } }\n";
        auto kernel = get_or_create_kernel(kernel_name, glsl, 2, sizeof(u32)*3);
        struct PC { u32 rows, cols, axis; } pc{input_shape[0], input_shape[1], axis};
        u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(expected_result_size);
        _accelerator.execute(kernel, {tensor, result}, dispatch, 1, 1, &pc);
        return result;
    }
    // two-pass max reduction
    DataType dtype = tensor->get_dtype(); auto shape = tensor->get_shape(); std::vector<u32> out_shape; for (u32 i=0;i<shape.size();++i) if (i!=axis) out_shape.push_back(shape[i]); 
    if (out_shape.empty()) out_shape.push_back(1); // scalar result
    auto result = _accelerator.create_tensor(out_shape, dtype);
    u32 max_rank = static_cast<u32>(shape.size()); std::vector<u32> meta; meta.reserve(2+max_rank*4); meta.push_back(max_rank); meta.push_back(axis);
    auto pad_dims2 = [&](const std::vector<u32>& s) { std::vector<u32> padded(max_rank,1u); size_t off=max_rank-s.size(); for(size_t i=0;i<s.size();++i) padded[off+i]=s[i]; return padded; };
    auto in_p2 = pad_dims2(shape); auto out_p2 = pad_dims2(out_shape); auto in_str2 = compute_strides_padded(shape, max_rank); auto out_str2 = compute_strides_padded(out_shape, max_rank);
    for (u32 v:in_p2) meta.push_back(v); for (u32 v:out_p2) meta.push_back(v); for (u32 v:in_str2) meta.push_back(v); for (u32 v:out_str2) meta.push_back(v);
    auto meta_tensor2 = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, true);
    u32 red_len2 = shape[axis];
    const u32 LOCAL2 = 256u;
    u32 group_size2 = LOCAL2;
    u32 group_count2 = (red_len2 + group_size2 - 1)/group_size2;
    u64 out_count2 = calculate_element_count(out_shape);
    u64 partials_count2 = out_count2 * static_cast<u64>(group_count2);
    auto partials2 = _accelerator.create_tensor({static_cast<u32>(partials_count2)}, dtype, true);

    std::string k1_name_max = get_kernel_name_for_dtype("reduce_first_pass_max", dtype);
    std::string k1_src_max = generate_reduce_axis_first_pass_kernel_source(dtype, "max", LOCAL2);
    auto k1_kernel_max = get_or_create_kernel(k1_name_max, k1_src_max, 3, sizeof(u32)*3);
    struct ReduceFirstPCMax { u32 group_size; u32 group_count; u32 axis; } reduce_first_pc_max{group_size2, group_count2, axis};
    u32 dispatch_x_max = static_cast<u32>(out_count2);
    u32 dispatch_y_max = group_count2;
    _accelerator.execute(k1_kernel_max, {tensor, partials2, meta_tensor2}, dispatch_x_max, dispatch_y_max, 1, &reduce_first_pc_max);

    std::string k2_name_max = get_kernel_name_for_dtype("reduce_second_pass_max", dtype);
    std::string k2_src_max = generate_reduce_axis_second_pass_kernel_source(dtype, "max");
    auto k2_kernel_max = get_or_create_kernel(k2_name_max, k2_src_max, 2, sizeof(u32));
    struct ReduceSecondPCMax { u32 group_count; } reduce_second_pc_max{group_count2};
    u32 dispatch_final = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(out_count2));
    _accelerator.execute(k2_kernel_max, {partials2, result}, dispatch_final, 1, 1, &reduce_second_pc_max);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::pow(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a || !b) throw std::invalid_argument("Tensor pointers cannot be null");
    if (!a->is_valid() || !b->is_valid()) throw std::invalid_argument("All tensors must be valid");

    // scalar power will be handled by the broadcast/elementwise path below using on-device buffers

    // broadcast/combine paths: reuse elementwise framework
    if (!a->is_shape_compatible(*b)) {
        auto out_shape = compute_broadcast_shape(a->get_shape(), b->get_shape());
        if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");
        DataType dtype = a->get_dtype();
        auto result = _accelerator.create_tensor(out_shape, dtype);
        u32 max_rank = static_cast<u32>(std::max(a->get_rank(), b->get_rank()));
        std::vector<u32> meta;
        meta.reserve(4 + max_rank * 6);
        meta.push_back(max_rank);
        meta.push_back(static_cast<u32>(a->get_rank()));
        meta.push_back(static_cast<u32>(b->get_rank()));
        meta.push_back(static_cast<u32>(out_shape.size()));
        auto pad_dims = [&](const std::vector<u32>& s) {
            std::vector<u32> padded(max_rank, 1);
            for (size_t i = 0; i < s.size(); ++i) padded[max_rank - s.size() + i] = s[i];
            return padded;
        };
        auto a_p = pad_dims(a->get_shape());
        auto b_p = pad_dims(b->get_shape());
        auto out_p = pad_dims(out_shape);
        for (u32 v : a_p) meta.push_back(v);
        for (u32 v : b_p) meta.push_back(v);
        for (u32 v : out_p) meta.push_back(v);
        auto a_strides = compute_strides_padded(a->get_shape(), max_rank);
        auto b_strides = compute_strides_padded(b->get_shape(), max_rank);
        auto out_strides = compute_strides_padded(out_shape, max_rank);
        for (u32 v : a_strides) meta.push_back(static_cast<u32>(v));
        for (u32 v : b_strides) meta.push_back(static_cast<u32>(v));
        for (u32 v : out_strides) meta.push_back(static_cast<u32>(v));
    auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, true);
        std::string kernel_name = get_kernel_name_for_dtype("pow_broadcast", dtype);
        std::string glsl_source = generate_elementwise_kernel_source("pow(data_a[a_index], data_b[b_index])", dtype, false, true, max_rank);
        auto kernel = get_or_create_kernel(kernel_name, glsl_source, 4);
        u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(calculate_element_count(out_shape)));
        _accelerator.execute(kernel, {a, b, result, meta_tensor}, dispatch_size);
        return result;
    }

    if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");
    auto result = _accelerator.create_tensor(a->get_shape(), a->get_dtype());
    std::string kernel_name = get_kernel_name_for_dtype("pow", a->get_dtype());
    std::string glsl_source = generate_elementwise_kernel_source("pow(data_a[index], data_b[index])", a->get_dtype());
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3);
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    _accelerator.execute(kernel, {a, b, result}, dispatch_size);
    return result;
}

// New elementwise math ops
std::shared_ptr<Tensor> TensorOperations::exp(std::shared_ptr<Tensor> tensor) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    DataType dtype = tensor->get_dtype();
    if (!_accelerator.use_gpu()) {
        _accelerator.notify_cpu_fallback();
        if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32 for exp");
        u64 cnt = tensor->get_element_count(); std::vector<float> in(cnt), out(cnt); tensor->download_data(in.data()); for (u64 i=0;i<cnt;++i) out[i]=std::exp(in[i]); return _accelerator.create_tensor(out.data(), tensor->get_shape(), dtype, false);
    }
    auto result = _accelerator.create_tensor(tensor->get_shape(), dtype);
    std::string kernel_name = get_kernel_name_for_dtype("exp", dtype);
    std::string glsl = generate_activation_kernel_source(dtype, "exp(data_in[index])");
    auto kernel = get_or_create_kernel(kernel_name, glsl, 2);
    u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    _accelerator.execute(kernel, {tensor, result}, dispatch);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::log(std::shared_ptr<Tensor> tensor) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    DataType dtype = tensor->get_dtype();
    if (!_accelerator.use_gpu()) {
        _accelerator.notify_cpu_fallback();
        if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32 for log");
        u64 cnt = tensor->get_element_count(); std::vector<float> in(cnt), out(cnt); tensor->download_data(in.data()); for (u64 i=0;i<cnt;++i) out[i]=std::log(in[i]); return _accelerator.create_tensor(out.data(), tensor->get_shape(), dtype, false);
    }
    auto result = _accelerator.create_tensor(tensor->get_shape(), dtype);
    std::string kernel_name = get_kernel_name_for_dtype("log", dtype);
    std::string glsl = generate_unary_kernel_source("log(data_in[index])", dtype);
    auto kernel = get_or_create_kernel(kernel_name, glsl, 2);
    u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    _accelerator.execute(kernel, {tensor, result}, dispatch);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::sin(std::shared_ptr<Tensor> tensor) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    DataType dtype = tensor->get_dtype();
    if (!_accelerator.use_gpu()) {
        _accelerator.notify_cpu_fallback();
        if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32 for sin");
        u64 cnt = tensor->get_element_count(); std::vector<float> in(cnt), out(cnt); tensor->download_data(in.data()); for (u64 i=0;i<cnt;++i) out[i]=std::sin(in[i]); return _accelerator.create_tensor(out.data(), tensor->get_shape(), dtype, false);
    }
    auto result = _accelerator.create_tensor(tensor->get_shape(), dtype);
    std::string kernel_name = get_kernel_name_for_dtype("sin", dtype);
    std::string glsl = generate_unary_kernel_source("sin(data_in[index])", dtype);
    auto kernel = get_or_create_kernel(kernel_name, glsl, 2);
    u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    _accelerator.execute(kernel, {tensor, result}, dispatch);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::cos(std::shared_ptr<Tensor> tensor) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    DataType dtype = tensor->get_dtype();
    if (!_accelerator.use_gpu()) {
        _accelerator.notify_cpu_fallback();
        if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32 for cos");
        u64 cnt = tensor->get_element_count(); std::vector<float> in(cnt), out(cnt); tensor->download_data(in.data()); for (u64 i=0;i<cnt;++i) out[i]=std::cos(in[i]); return _accelerator.create_tensor(out.data(), tensor->get_shape(), dtype, false);
    }
    auto result = _accelerator.create_tensor(tensor->get_shape(), dtype);
    std::string kernel_name = get_kernel_name_for_dtype("cos", dtype);
    std::string glsl = generate_unary_kernel_source("cos(data_in[index])", dtype);
    auto kernel = get_or_create_kernel(kernel_name, glsl, 2);
    u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    _accelerator.execute(kernel, {tensor, result}, dispatch);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::sqrt(std::shared_ptr<Tensor> tensor) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    DataType dtype = tensor->get_dtype();
    if (!_accelerator.use_gpu()) {
        _accelerator.notify_cpu_fallback();
        if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32 for sqrt");
        u64 cnt = tensor->get_element_count(); std::vector<float> in(cnt), out(cnt); tensor->download_data(in.data()); for (u64 i=0;i<cnt;++i) out[i]=std::sqrt(in[i]); return _accelerator.create_tensor(out.data(), tensor->get_shape(), dtype, false);
    }
    auto result = _accelerator.create_tensor(tensor->get_shape(), dtype);
    std::string kernel_name = get_kernel_name_for_dtype("sqrt", dtype);
    std::string glsl = generate_unary_kernel_source("sqrt(data_in[index])", dtype);
    auto kernel = get_or_create_kernel(kernel_name, glsl, 2);
    u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    _accelerator.execute(kernel, {tensor, result}, dispatch);
    return result;
}

// dot convenience: alias to matmul for 1D/2D combinations where appropriate
std::shared_ptr<Tensor> TensorOperations::dot(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a || !b) throw std::invalid_argument("Tensors cannot be null");
    if (a->get_rank() == 1 && b->get_rank() == 1) {
        // inner product -> result scalar (shape {1}) implemented as matmul of (1,K) x (K,1)
        std::vector<u32> A_shape = {1, a->get_shape()[0]};
        std::vector<u32> B_shape = {b->get_shape()[0], 1};
        auto a2 = a->create_view_with_shape(A_shape);
        auto b2 = b->create_view_with_shape(B_shape);
        auto r = matmul(a2, b2);
        // produce scalar tensor (shape {}) keep as 1-element tensor
        return r;
    }
    if (a->get_rank() == 2 && b->get_rank() == 2) {
        return matmul(a, b);
    }
    // fallback: attempt matmul where shapes are compatible
    return matmul(a, b);
}

// Random generators - CPU-only, create host-visible tensors
std::shared_ptr<Tensor> TensorOperations::random_uniform(const std::vector<u32>& shape, DataType dtype, float low, float high) {
    if (dtype != DataType::F32) throw std::invalid_argument("random_uniform currently supports F32 only");
    u64 count = calculate_element_count(shape);
    std::vector<float> buf(count);
    std::random_device rd; std::mt19937 gen(rd()); std::uniform_real_distribution<float> dist(low, high);
    for (u64 i=0;i<count;++i) buf[i] = dist(gen);
    return _accelerator.create_tensor(buf.data(), shape, dtype, /*device_only=*/false);
}

std::shared_ptr<Tensor> TensorOperations::random_normal(const std::vector<u32>& shape, DataType dtype, float mean, float stddev) {
    if (dtype != DataType::F32) throw std::invalid_argument("random_normal currently supports F32 only");
    u64 count = calculate_element_count(shape);
    std::vector<float> buf(count);
    std::random_device rd; std::mt19937 gen(rd()); std::normal_distribution<float> dist(mean, stddev);
    for (u64 i=0;i<count;++i) buf[i] = dist(gen);
    return _accelerator.create_tensor(buf.data(), shape, dtype, /*device_only=*/false);
}

std::shared_ptr<Tensor> TensorOperations::bernoulli(const std::vector<u32>& shape, float p) {
    u64 count = calculate_element_count(shape);
    std::vector<u8> buf(count);
    std::random_device rd; std::mt19937 gen(rd()); std::bernoulli_distribution dist(p);
    for (u64 i=0;i<count;++i) buf[i] = static_cast<u8>(dist(gen) ? 1 : 0);
    return _accelerator.create_tensor(buf.data(), shape, DataType::U8, /*device_only=*/false);
}

// Simple binary format: header magic '.qsbin' (6 bytes), dtype (u32), rank (u32), dims..., raw bytes
void TensorOperations::save_tensor(std::shared_ptr<Tensor> tensor, const std::string& path) const {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Failed to open file for writing: " + path);
    const char magic[6] = {'.','q','s','b','i','n'}; ofs.write(magic, 6);
    u32 dt = static_cast<u32>(tensor->get_dtype()); ofs.write(reinterpret_cast<const char*>(&dt), sizeof(u32));
    u32 rank = static_cast<u32>(tensor->get_shape().size()); ofs.write(reinterpret_cast<const char*>(&rank), sizeof(u32));
    for (u32 d : tensor->get_shape()) ofs.write(reinterpret_cast<const char*>(&d), sizeof(u32));
    u64 bytes = tensor->get_size_bytes(); std::vector<u8> buf(bytes); tensor->download_data(buf.data()); ofs.write(reinterpret_cast<const char*>(buf.data()), bytes);
}

std::shared_ptr<Tensor> TensorOperations::load_tensor(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open file for reading: " + path);
    char magic[6]; ifs.read(magic,6); if (magic[0]!='.' || magic[1]!='q' || magic[2]!='s' || magic[3]!='b' || magic[4]!='i' || magic[5]!='n') throw std::runtime_error("Not a .qsbin file");
    u32 dt; ifs.read(reinterpret_cast<char*>(&dt), sizeof(u32)); DataType dtype = static_cast<DataType>(dt);
    u32 rank; ifs.read(reinterpret_cast<char*>(&rank), sizeof(u32)); std::vector<u32> shape(rank);
    for (u32 i=0;i<rank;++i) ifs.read(reinterpret_cast<char*>(&shape[i]), sizeof(u32));
    u64 bytes = calculate_element_count(shape) * get_dtype_size(dtype);
    std::vector<u8> buf(bytes); ifs.read(reinterpret_cast<char*>(buf.data()), bytes);
    // create host-visible tensor
    return _accelerator.create_tensor(buf.data(), shape, dtype, /*device_only=*/false);
}

// ============================================================================
// NEW BASIC TENSOR OPERATIONS
// ============================================================================

// Sigmoid activation: 1 / (1 + exp(-x))
std::shared_ptr<Tensor> TensorOperations::sigmoid(std::shared_ptr<Tensor> tensor) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    DataType dtype = tensor->get_dtype();
    std::string kernel_name = get_kernel_name_for_dtype("sigmoid", dtype);
    std::string glsl = generate_activation_kernel_source(dtype, "1.0 / (1.0 + exp(-data_in[index]))");
    auto kernel = get_or_create_kernel(kernel_name, glsl, 2);
    auto result = _accelerator.create_tensor(tensor->get_shape(), dtype);
    u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    _accelerator.execute(kernel, {tensor, result}, dispatch);
    return result;
}

// Tanh activation: tanh(x)
std::shared_ptr<Tensor> TensorOperations::tanh(std::shared_ptr<Tensor> tensor) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    DataType dtype = tensor->get_dtype();
    std::string kernel_name = get_kernel_name_for_dtype("tanh", dtype);
    std::string glsl = generate_activation_kernel_source(dtype, "tanh(data_in[index])");
    auto kernel = get_or_create_kernel(kernel_name, glsl, 2);
    auto result = _accelerator.create_tensor(tensor->get_shape(), dtype);
    u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    _accelerator.execute(kernel, {tensor, result}, dispatch);
    return result;
}

// Softmax: exp(x_i) / sum(exp(x_j)) along specified axis
std::shared_ptr<Tensor> TensorOperations::softmax(std::shared_ptr<Tensor> tensor, int axis) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    
    // Normalize axis
    int rank = static_cast<int>(tensor->get_rank());
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) throw std::invalid_argument("Axis out of range");
    
    u32 ax = static_cast<u32>(axis);
    
    // Softmax: numerically stable version
    // 1. Find max along axis (for numerical stability)
    auto max_vals = max_axis(tensor, ax);
    
    // 2. Subtract max (broadcasting)
    auto shifted = sub(tensor, max_vals);
    
    // 3. Compute exp
    auto exp_vals = exp(shifted);
    
    // 4. Sum along axis
    auto sum_exp = sum_axis(exp_vals, ax);
    
    // 5. Divide (broadcasting)
    auto result = div(exp_vals, sum_exp);
    
    return result;
}

// Absolute value
std::shared_ptr<Tensor> TensorOperations::abs(std::shared_ptr<Tensor> tensor) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    DataType dtype = tensor->get_dtype();
    std::string kernel_name = get_kernel_name_for_dtype("abs", dtype);
    std::string glsl = generate_activation_kernel_source(dtype, "abs(data_in[index])");
    auto kernel = get_or_create_kernel(kernel_name, glsl, 2);
    auto result = _accelerator.create_tensor(tensor->get_shape(), dtype);
    u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    _accelerator.execute(kernel, {tensor, result}, dispatch);
    return result;
}

// Negation
std::shared_ptr<Tensor> TensorOperations::neg(std::shared_ptr<Tensor> tensor) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    DataType dtype = tensor->get_dtype();
    std::string kernel_name = get_kernel_name_for_dtype("neg", dtype);
    std::string glsl = generate_activation_kernel_source(dtype, "-data_in[index]");
    auto kernel = get_or_create_kernel(kernel_name, glsl, 2);
    auto result = _accelerator.create_tensor(tensor->get_shape(), dtype);
    u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    _accelerator.execute(kernel, {tensor, result}, dispatch);
    return result;
}

// Clamp: constrain values between min and max
std::shared_ptr<Tensor> TensorOperations::clamp(std::shared_ptr<Tensor> tensor, float min_val, float max_val) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    if (min_val > max_val) throw std::invalid_argument("min_val must be <= max_val");
    
    DataType dtype = tensor->get_dtype();
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    std::string src = "#version 450\n";
    src += "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n";
    src += "layout(push_constant) uniform PushConstants { float min_v; float max_v; } pc;\n";
    src += "layout(set = 0, binding = 0, std430) restrict readonly buffer Input { " + std::string(glsl_type) + " data_in[]; } ;\n";
    src += "layout(set = 0, binding = 1, std430) restrict writeonly buffer Output { " + std::string(glsl_type) + " data_out[]; } ;\n";
    src += "void main() {\n";
    src += "    uint index = gl_GlobalInvocationID.x;\n";
    src += "    if (index >= data_in.length()) return;\n";
    src += "    " + std::string(glsl_type) + " val = data_in[index];\n";
    src += "    data_out[index] = clamp(val, " + std::string(glsl_type) + "(pc.min_v), " + std::string(glsl_type) + "(pc.max_v));\n";
    src += "}\n";
    
    std::string kernel_name = get_kernel_name_for_dtype("clamp", dtype);
    auto kernel = get_or_create_kernel(kernel_name, src, 2, sizeof(float) * 2);
    auto result = _accelerator.create_tensor(tensor->get_shape(), dtype);
    
    struct PushConst { float min_v; float max_v; } pc{min_val, max_val};
    u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    _accelerator.execute(kernel, {tensor, result}, dispatch, 1, 1, &pc);
    return result;
}

// Permute: arbitrary dimension reordering (generalized transpose)
std::shared_ptr<Tensor> TensorOperations::permute(std::shared_ptr<Tensor> tensor, const std::vector<u32>& dims) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    
    auto shape = tensor->get_shape();
    u32 rank = tensor->get_rank();
    
    if (dims.size() != rank) throw std::invalid_argument("dims must match tensor rank");
    
    // Validate dims: must be a permutation of 0..rank-1
    std::vector<bool> seen(rank, false);
    for (u32 d : dims) {
        if (d >= rank) throw std::invalid_argument("Invalid dimension in permutation");
        if (seen[d]) throw std::invalid_argument("Duplicate dimension in permutation");
        seen[d] = true;
    }
    
    // Compute output shape
    std::vector<u32> out_shape(rank);
    for (u32 i = 0; i < rank; ++i) {
        out_shape[i] = shape[dims[i]];
    }
    
    DataType dtype = tensor->get_dtype();
    auto result = _accelerator.create_tensor(out_shape, dtype);
    
    // Build metadata: [rank, dims..., in_shape..., out_shape..., in_strides..., out_strides...]
    auto in_strides = tensor->calculate_strides();
    auto out_strides = result->calculate_strides();
    
    std::vector<u32> meta;
    meta.push_back(rank);
    for (u32 d : dims) meta.push_back(d);
    for (u32 s : shape) meta.push_back(s);
    for (u32 s : out_shape) meta.push_back(s);
    for (auto s : in_strides) meta.push_back(static_cast<u32>(s));
    for (auto s : out_strides) meta.push_back(static_cast<u32>(s));
    
    auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, true);
    
    std::string kernel_name = get_kernel_name_for_dtype("permute", dtype);
    std::string glsl = generate_permute_kernel_source(dtype);
    auto kernel = get_or_create_kernel(kernel_name, glsl, 3);
    
    u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(result->get_element_count()));
    _accelerator.execute(kernel, {tensor, result, meta_tensor}, dispatch);
    
    return result;
}

// Concatenate: join tensors along an axis
std::shared_ptr<Tensor> TensorOperations::concatenate(const std::vector<std::shared_ptr<Tensor>>& tensors, u32 axis) {
    if (tensors.empty()) throw std::invalid_argument("Cannot concatenate empty tensor list");
    if (!tensors[0]) throw std::invalid_argument("Tensors cannot be null");
    
    u32 rank = tensors[0]->get_rank();
    if (axis >= rank) throw std::invalid_argument("Axis out of range");
    
    auto dtype = tensors[0]->get_dtype();
    auto first_shape = tensors[0]->get_shape();
    
    // Validate all tensors have compatible shapes
    u32 total_dim = 0;
    for (const auto& t : tensors) {
        if (!t || !t->is_valid()) throw std::invalid_argument("All tensors must be valid");
        if (t->get_dtype() != dtype) throw std::invalid_argument("All tensors must have same dtype");
        if (t->get_rank() != rank) throw std::invalid_argument("All tensors must have same rank");
        
        auto shape = t->get_shape();
        for (u32 i = 0; i < rank; ++i) {
            if (i != axis && shape[i] != first_shape[i]) {
                throw std::invalid_argument("Tensor shapes must match except on concat axis");
            }
        }
        total_dim += shape[axis];
    }
    
    // Compute output shape
    std::vector<u32> out_shape = first_shape;
    out_shape[axis] = total_dim;
    
    auto result = _accelerator.create_tensor(out_shape, dtype);
    
    // GPU kernel approach: support up to 8 input tensors
    if (tensors.size() > 8) {
        throw std::invalid_argument("Concatenate supports up to 8 tensors on GPU");
    }
    
    // Build metadata: [num_tensors, axis, rank, out_shape[rank], out_strides[rank], 
    //                   offset0...offset7, size0...size7, in_strides0[rank]...in_strides7[rank]]
    std::vector<u32> meta;
    meta.push_back(static_cast<u32>(tensors.size()));
    meta.push_back(axis);
    meta.push_back(rank);
    
    // Output shape
    for (u32 dim : out_shape) meta.push_back(dim);
    
    // Output strides
    auto out_strides = result->calculate_strides();
    for (auto s : out_strides) meta.push_back(static_cast<u32>(s));
    
    // Offsets: cumulative axis offsets for each tensor (pad to 8)
    u32 offset = 0;
    for (size_t i = 0; i < 8; ++i) {
        if (i < tensors.size()) {
            meta.push_back(offset);
            offset += tensors[i]->get_shape()[axis];
        } else {
            meta.push_back(0);
        }
    }
    
    // Sizes: axis size for each tensor (pad to 8)
    for (size_t i = 0; i < 8; ++i) {
        if (i < tensors.size()) {
            meta.push_back(tensors[i]->get_shape()[axis]);
        } else {
            meta.push_back(1); // dummy value
        }
    }
    
    // Input strides for each tensor (pad to 8 tensors)
    for (size_t i = 0; i < 8; ++i) {
        if (i < tensors.size()) {
            auto in_strides = tensors[i]->calculate_strides();
            for (auto s : in_strides) meta.push_back(static_cast<u32>(s));
        } else {
            // Pad with dummy strides
            for (u32 r = 0; r < rank; ++r) meta.push_back(1);
        }
    }
    
    auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, true);
    
    // Prepare tensor list with padding (fill unused slots with first tensor to avoid null buffers)
    std::vector<std::shared_ptr<Tensor>> padded_tensors = tensors;
    while (padded_tensors.size() < 8) {
        padded_tensors.push_back(tensors[0]);
    }
    
    // Execute kernel
    std::string kernel_name = get_kernel_name_for_dtype("concatenate", dtype);
    std::string glsl = generate_concatenate_kernel_source(dtype);
    auto kernel = get_or_create_kernel(kernel_name, glsl, 10); // 8 inputs + 1 output + 1 meta
    
    u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(result->get_element_count()));
    _accelerator.execute(kernel, {padded_tensors[0], padded_tensors[1], padded_tensors[2], padded_tensors[3],
                                   padded_tensors[4], padded_tensors[5], padded_tensors[6], padded_tensors[7],
                                   result, meta_tensor}, dispatch);
    
    return result;
}

// Split: divide tensor along an axis into multiple tensors
std::vector<std::shared_ptr<Tensor>> TensorOperations::split(std::shared_ptr<Tensor> tensor, u32 num_splits, u32 axis) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    
    auto shape = tensor->get_shape();
    u32 rank = tensor->get_rank();
    
    if (axis >= rank) throw std::invalid_argument("Axis out of range");
    if (num_splits == 0) throw std::invalid_argument("num_splits must be > 0");
    if (shape[axis] % num_splits != 0) {
        throw std::invalid_argument("Axis size must be divisible by num_splits");
    }
    
    u32 split_size = shape[axis] / num_splits;
    std::vector<std::shared_ptr<Tensor>> results;
    
    // Use slice to extract each split
    for (u32 i = 0; i < num_splits; ++i) {
        std::vector<u32> start(rank, 0);
        std::vector<u32> lengths = shape;
        
        start[axis] = i * split_size;
        lengths[axis] = split_size;
        
        results.push_back(slice(tensor, start, lengths));
    }
    
    return results;
}

// Squeeze: remove dimensions of size 1
std::shared_ptr<Tensor> TensorOperations::squeeze(std::shared_ptr<Tensor> tensor, int axis) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    
    auto shape = tensor->get_shape();
    std::vector<u32> new_shape;
    
    if (axis == -1) {
        // Remove all dimensions of size 1
        for (u32 dim : shape) {
            if (dim != 1) new_shape.push_back(dim);
        }
    } else {
        // Remove specific axis if it's size 1
        int rank = static_cast<int>(shape.size());
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw std::invalid_argument("Axis out of range");
        
        if (shape[axis] != 1) {
            throw std::invalid_argument("Cannot squeeze axis with size != 1");
        }
        
        for (u32 i = 0; i < shape.size(); ++i) {
            if (i != static_cast<u32>(axis)) new_shape.push_back(shape[i]);
        }
    }
    
    // Handle case where all dimensions were 1
    if (new_shape.empty()) new_shape.push_back(1);
    
    // Return a view with the new shape
    return tensor->create_view_with_shape(new_shape);
}

// Unsqueeze: add a dimension of size 1
std::shared_ptr<Tensor> TensorOperations::unsqueeze(std::shared_ptr<Tensor> tensor, u32 axis) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    
    auto shape = tensor->get_shape();
    u32 new_rank = static_cast<u32>(shape.size()) + 1;
    
    if (axis > shape.size()) throw std::invalid_argument("Axis out of range");
    
    std::vector<u32> new_shape;
    for (u32 i = 0; i < shape.size(); ++i) {
        if (i == axis) new_shape.push_back(1);
        new_shape.push_back(shape[i]);
    }
    if (axis == shape.size()) new_shape.push_back(1);
    
    // Return a view with the new shape
    return tensor->create_view_with_shape(new_shape);
}

std::shared_ptr<Tensor> TensorOperations::slice(std::shared_ptr<Tensor> tensor, const std::vector<u32>& start, const std::vector<u32>& lengths) {
    if (!tensor || !tensor->is_valid()) throw std::invalid_argument("Tensor must be valid");
    if (start.size() != tensor->get_shape().size() || lengths.size() != tensor->get_shape().size()) throw std::invalid_argument("Start and lengths must match tensor rank");
    // Attempt zero-copy view when the requested slice is a contiguous block in C-order
    u64 total_elems = tensor->get_element_count();
    u32 rank = tensor->get_rank();
    auto strides = tensor->calculate_strides();
    // compute result element count
    u64 res_count = 1;
    for (u32 l : lengths) res_count *= l;

    // Check if the slice can be represented as a contiguous sub-block in flat (C-order) memory.
    // Compute the starting flat index and the last flat index of the slice (using inclusive ranges)
    // and test whether the range [start_flat, last_flat] has exactly res_count elements.
    bool contiguous = true;
    u64 start_flat = 0;
    for (u32 d = 0; d < rank; ++d) {
        if (start[d] >= tensor->get_shape()[d]) throw std::out_of_range("Slice start out of range");
        start_flat += static_cast<u64>(start[d]) * strides[d];
    }
    u64 last_flat = 0;
    for (u32 d = 0; d < rank; ++d) {
        u32 last_idx = start[d] + lengths[d] - 1;
        last_flat += static_cast<u64>(last_idx) * strides[d];
    }
    // contiguous iff the flat range equals the number of elements requested
    if (last_flat - start_flat + 1 != res_count) contiguous = false;

    if (contiguous) {
        // create view with offset start_flat
        return tensor->create_view_with_shape_and_offset(std::vector<u32>(lengths.begin(), lengths.end()), start_flat);
    }

    // GPU gather: build meta buffer and dispatch a strided-gather kernel that maps each output index
    // to its input source using padded start offsets, out_dims and strides.
    DataType dtype = tensor->get_dtype();
    LOG_DEBUG("[slice] non-contiguous GPU gather: dtype={} rank={} res_count={} start_flat={}", dtype_to_string(dtype), rank, res_count, start_flat);
    // prepare padded meta: layout [ max_rank, start_padded[max_rank], out_dims_padded[max_rank], out_strides_padded[max_rank], in_strides_padded[max_rank] ]
    u32 max_rank = rank;
    auto pad_vals = [&](const std::vector<u32>& v) {
        std::vector<u32> padded(max_rank, 1u);
        size_t off = max_rank - v.size();
        for (size_t i = 0; i < v.size(); ++i) padded[off + i] = v[i];
        return padded;
    };

    auto start_p = pad_vals(start);
    auto out_p = pad_vals(std::vector<u32>(lengths.begin(), lengths.end()));
    auto out_str = compute_strides_padded(std::vector<u32>(lengths.begin(), lengths.end()), max_rank);
    auto in_str = compute_strides_padded(tensor->get_shape(), max_rank);

    std::vector<u32> meta;
    meta.reserve(1 + max_rank * 4);
    meta.push_back(max_rank);
    for (u32 v : start_p) meta.push_back(v);
    for (u32 v : out_p) meta.push_back(v);
    for (u32 v : out_str) meta.push_back(v);
    for (u32 v : in_str) meta.push_back(v);

    // log meta buffer contents at debug level
    {
        std::string meta_str;
        meta_str.reserve(meta.size() * 6);
        for (auto m : meta) { meta_str += std::to_string(m); meta_str += ' '; }
        LOG_DEBUG("[slice] meta({}): {}", meta.size(), meta_str);
    }

    auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, true);

    // create output tensor on device
    auto result = _accelerator.create_tensor(std::vector<u32>(lengths.begin(), lengths.end()), dtype);

    std::string kernel_name = get_kernel_name_for_dtype("gather_strided", dtype);
    std::string glsl = generate_strided_gather_kernel_source(dtype);
    auto kernel = get_or_create_kernel(kernel_name, glsl, 3);
    u32 dispatch = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(res_count));
    _accelerator.execute(kernel, {tensor, result, meta_tensor}, dispatch);
    return result;
}

// Generate a kernel that maps each output flat index to an input flat index using start offsets and strides
std::string TensorOperations::generate_strided_gather_kernel_source(DataType dtype) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    std::string src = "#version 450\n";
    src += "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n";
    src += "layout(set = 0, binding = 0, std430) restrict readonly buffer Input { " + std::string(glsl_type) + " data_in[]; } ;\n";
    src += "layout(set = 0, binding = 1, std430) restrict writeonly buffer Output { " + std::string(glsl_type) + " data_out[]; } ;\n";
    src += "layout(set = 0, binding = 2, std430) restrict readonly buffer Meta { uint shapes[]; } ;\n";
    src += "void main() { uint index = gl_GlobalInvocationID.x; if (index >= data_out.length()) return; uint max_rank = shapes[0]; uint off = 1u; uint start_off = off; uint out_dims_off = start_off + max_rank; uint out_strides_off = out_dims_off + max_rank; uint in_strides_off = out_strides_off + max_rank; uint rem = index; uint in_flat = 0u;\n";
    src += "    // reconstruct coordinates from output flat index using out_strides and map to input using start offsets and in_strides\n";
    src += "    for (uint d = 0u; d < max_rank; ++d) { uint dim = shapes[out_dims_off + d]; uint stride = shapes[out_strides_off + d]; uint coord = 0u; if (stride > 0u) { coord = (rem / stride) % dim; } uint s = shapes[start_off + d]; uint in_stride = shapes[in_strides_off + d]; uint in_coord = s + coord; if (in_stride > 0u) in_flat += in_coord * in_stride; }\n";
    src += "    data_out[index] = data_in[in_flat]; }\n";
    return src;
}

void TensorOperations::validate_tensor_op_compatibility(std::shared_ptr<Tensor> a, 
                                                       std::shared_ptr<Tensor> b) const {
    if (!a->is_valid() || !b->is_valid()) {
        throw std::invalid_argument("All tensors must be valid");
    }
    
    if (!a->is_shape_compatible(*b)) {
        throw std::invalid_argument("Tensors must have compatible shapes");
    }
    
    if (a->get_dtype() != b->get_dtype()) {
        throw std::invalid_argument("Tensors must have the same data type");
    }
}

void TensorOperations::validate_tensor_shape_2d(std::shared_ptr<Tensor> tensor) const {
    if (!tensor->is_valid()) {
        throw std::invalid_argument("Tensor must be valid");
    }
    
    if (tensor->get_rank() != 2) {
        throw std::invalid_argument("Tensor must be 2D");
    }
}

std::vector<u32> TensorOperations::compute_broadcast_shape(const std::vector<u32>& a, const std::vector<u32>& b) const {
    // General N-d broadcasting: align from trailing dimensions
    // Determine max rank
    size_t ra = a.size();
    size_t rb = b.size();
    size_t r = std::max(ra, rb);
    std::vector<u32> out(r, 1);

    for (size_t i = 0; i < r; ++i) {
        u32 da = (i < ra) ? a[ra - 1 - i] : 1;
        u32 db = (i < rb) ? b[rb - 1 - i] : 1;
        if (da == db || da == 1) out[r - 1 - i] = db;
        else if (db == 1) out[r - 1 - i] = da;
        else throw std::runtime_error("compute_broadcast_shape: incompatible dimensions");
    }
    return out;
}

std::shared_ptr<Kernel> TensorOperations::get_or_create_kernel(const std::string& name, 
                                                              const std::string& glsl_source, 
                                                              u32 num_tensors, u32 push_constant_size) {
    auto existing_kernel = _accelerator.get_kernel(name);
    if (existing_kernel) {
        return existing_kernel;
    }
    // Kernel source is intentionally not logged here to avoid noisy logs on success.
    // Shader compilation diagnostics (including GLSL) are emitted by the Vulkan backend
    // when a compilation error occurs.
    return _accelerator.create_kernel(name, glsl_source, num_tensors, push_constant_size);
}

std::string TensorOperations::generate_unary_kernel_source(const std::string& operation, DataType dtype) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    std::string source = "#version 450\n"
                        "layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;\n"
                        "layout(set = 0, binding = 0, std430) restrict readonly buffer Input {\n"
                        "    " + std::string(glsl_type) + " data_in[];\n"
                        "};\n"
                        "layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {\n"
                        "    " + std::string(glsl_type) + " data_out[];\n"
                        "};\n"
                        "void main() {\n"
                        "    uint index = gl_GlobalInvocationID.x;\n"
                        "    if (index >= data_in.length()) return;\n"
                        "    data_out[index] = " + std::string(glsl_type) + "(" + operation + ");\n"
                        "}\n";
    return source;
}

std::string TensorOperations::generate_elementwise_kernel_source(const std::string& operation, 
                                                                DataType dtype, 
                                                                bool is_scalar,
                                                                bool supports_broadcast,
                                                                u32 max_rank) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    std::string source = "#version 450\n"
                        "layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;\n";
    
    if (is_scalar) {
        const char* pc_type = "float";
        switch (dtype) {
            case DataType::F32: pc_type = "float"; break;
            case DataType::F16: pc_type = "float"; break;
            case DataType::I32: case DataType::I16: case DataType::I8: pc_type = "int"; break;
            case DataType::U32: case DataType::U16: case DataType::U8: pc_type = "uint"; break;
            default: pc_type = "float"; break;
        }
        source += std::string("layout(push_constant) uniform PushConstants { ") + pc_type + " scalar; } pc;\n"
                 "layout(set = 0, binding = 0, std430) restrict readonly buffer Input {\n"
                 "    " + std::string(glsl_type) + " data_in[];\n"
                 "};\n"
                 "layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {\n"
                 "    " + std::string(glsl_type) + " data_out[];\n"
                 "};\n"
                 "void main() {\n"
                 "    uint index = gl_GlobalInvocationID.x;\n"
                 "    if (index >= data_in.length()) return;\n"
                 "    data_out[index] = " + std::string(glsl_type) + "(" + operation + ");\n"
                 "}\n";
    } else {
        if (!supports_broadcast) {
            source += "layout(set = 0, binding = 0, std430) restrict readonly buffer InputA {\n"
                     "    " + std::string(glsl_type) + " data_a[];\n"
                     "};\n"
                     "layout(set = 0, binding = 1, std430) restrict readonly buffer InputB {\n"
                     "    " + std::string(glsl_type) + " data_b[];\n"
                     "};\n"
                     "layout(set = 0, binding = 2, std430) restrict writeonly buffer Output {\n"
                     "    " + std::string(glsl_type) + " data_out[];\n"
                     "};\n"
                     "void main() {\n"
                     "    uint index = gl_GlobalInvocationID.x;\n"
                     "    if (index >= data_a.length()) return;\n"
                     "    " + std::string(glsl_type) + " a = data_a[index];\n"
                     "    " + std::string(glsl_type) + " b = data_b[index];\n"
                     "    data_out[index] = " + operation + ";\n"
                     "}\n";
        } else {
            // Binding layout: 0 = A, 1 = B, 2 = Output, 3 = Meta (shapes+strides)
            source += "layout(set = 0, binding = 0, std430) restrict readonly buffer InputA {\n"
                     "    " + std::string(glsl_type) + " data_a[];\n"
                     "};\n"
                     "layout(set = 0, binding = 1, std430) restrict readonly buffer InputB {\n"
                     "    " + std::string(glsl_type) + " data_b[];\n"
                     "};\n"
                     "layout(set = 0, binding = 2, std430) restrict writeonly buffer Output {\n"
                     "    " + std::string(glsl_type) + " data_out[];\n"
                     "};\n"
                     "layout(set = 0, binding = 3, std430) restrict readonly buffer Meta {\n"
                     "    uint shapes[]; // layout: max_rank, a_rank, b_rank, out_rank, then padded dims (a), padded dims (b), padded dims (out), then padded strides (a), padded strides (b)\n"
                     "};\n"
                     "void main() {\n"
                     "    uint index = gl_GlobalInvocationID.x;\n"
                     "    // reconstruct multi-dim coords from flat index using out_shape strides\n"
                     "    uint max_rank = shapes[0];\n"
                     "    uint a_rank = shapes[1];\n"
                     "    uint b_rank = shapes[2];\n"
                     "    uint out_rank = shapes[3];\n"
                     "    uint offset = 4;\n"
                     "    // pointers into shapes arrays\n"
                     "    uint a_dims_off = offset;\n"
                     "    uint b_dims_off = a_dims_off + max_rank;\n"
                     "    uint out_dims_off = b_dims_off + max_rank;\n"
                     "    uint a_strides_off = out_dims_off + max_rank;\n"
                     "    uint b_strides_off = a_strides_off + max_rank;\n"
                     "    uint out_strides_off = b_strides_off + max_rank;\n"
                     "    uint coords_idx = index;\n"
                     "    uint a_index = 0u;\n"
                     "    uint b_index = 0u;\n"
                     "    for (uint d = 0; d < max_rank; ++d) {\n"
                     "        uint dim = shapes[out_dims_off + d];\n"
                     "        uint stride = shapes[out_strides_off + d];\n"
                     "        uint coord = 0u;\n"
                     "        if (stride > 0u) { coord = (coords_idx / stride) % dim; }\n"
                     "        uint a_dim = shapes[a_dims_off + d];\n"
                     "        uint b_dim = shapes[b_dims_off + d];\n"
                     "        uint a_stride = shapes[a_strides_off + d];\n"
                     "        uint b_stride = shapes[b_strides_off + d];\n"
                     "        uint a_coord = (a_dim == 1u) ? 0u : coord;\n"
                     "        uint b_coord = (b_dim == 1u) ? 0u : coord;\n"
                     "        if (a_stride > 0u) a_index += a_coord * a_stride;\n"
                     "        if (b_stride > 0u) b_index += b_coord * b_stride;\n"
                     "    }\n"
                     "    data_out[index] = " + operation + ";\n"
                     "}\n";
        }
    }

    return source;
}

// helper: convert IEEE-754 half (16-bit) to float
static float half_to_float(uint16_t h) {
    uint16_t h_exp = (h & 0x7C00u);
    uint16_t h_sig = (h & 0x03FFu);
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t f;
    if (h_exp == 0x7C00u) {
        // Inf/NaN
        f = sign | 0x7F800000u | (h_sig ? 0x200000u : 0u);
    } else if (h_exp != 0) {
        // normalized
        uint32_t exp = ((h_exp >> 10) + (127 - 15)) << 23;
        uint32_t sig = (h_sig << 13);
        f = sign | exp | sig;
    } else if (h_sig != 0) {
        // subnormal
        uint32_t sig = h_sig;
        uint32_t exp = 0;
        while ((sig & 0x0400u) == 0) {
            sig <<= 1;
            exp++;
        }
        sig &= 0x03FFu;
        uint32_t exp32 = (127 - 15 - exp + 1) << 23;
        uint32_t sig32 = sig << 13;
        f = sign | exp32 | sig32;
    } else {
        // zero
        f = sign;
    }
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

// helper: pack downloaded scalar bytes into a 4-byte push constant (promote as needed)
static std::pair<std::array<u8,4>, u32> make_push_constant(const u8* buf, DataType dtype) {
    std::array<u8,4> out{};
    u32 size = 4;
    switch (dtype) {
        case DataType::F32: {
            std::memcpy(out.data(), buf, 4);
            break;
        }
        case DataType::F16: {
            uint16_t h;
            std::memcpy(&h, buf, 2);
            float f = half_to_float(h);
            std::memcpy(out.data(), &f, 4);
            break;
        }
        case DataType::I32: {
            int32_t v;
            std::memcpy(&v, buf, 4);
            std::memcpy(out.data(), &v, 4);
            break;
        }
        case DataType::U32: {
            uint32_t v;
            std::memcpy(&v, buf, 4);
            std::memcpy(out.data(), &v, 4);
            break;
        }
        case DataType::I16: {
            int16_t v16;
            std::memcpy(&v16, buf, 2);
            int32_t v = static_cast<int32_t>(v16);
            std::memcpy(out.data(), &v, 4);
            break;
        }
        case DataType::U16: {
            uint16_t v16;
            std::memcpy(&v16, buf, 2);
            uint32_t v = static_cast<uint32_t>(v16);
            std::memcpy(out.data(), &v, 4);
            break;
        }
        case DataType::I8: {
            int8_t v8;
            std::memcpy(&v8, buf, 1);
            int32_t v = static_cast<int32_t>(v8);
            std::memcpy(out.data(), &v, 4);
            break;
        }
        case DataType::U8: {
            uint8_t v8;
            std::memcpy(&v8, buf, 1);
            uint32_t v = static_cast<uint32_t>(v8);
            std::memcpy(out.data(), &v, 4);
            break;
        }
        default: {
            float f = 0.0f;
            std::memcpy(out.data(), &f, 4);
            break;
        }
    }
    return {out, size};
}

std::vector<u32> TensorOperations::compute_strides_padded(const std::vector<u32>& shape, u32 rank) const {
    std::vector<u32> padded(rank, 1u);
    if (!shape.empty()) {
        size_t offset = rank - shape.size();
        for (size_t i = 0; i < shape.size(); ++i) padded[offset + i] = shape[i];
    }

    std::vector<u32> strides(rank, 0u);
    if (rank == 0) return strides;
    // compute C-order strides
    uint64_t acc = 1;
    for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
        strides[i] = static_cast<u32>(acc);
        acc *= padded[i];
    }
    return strides;
}

std::string TensorOperations::generate_relu_kernel_source(DataType dtype) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    return "#version 450\n"
           "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
           "layout(set = 0, binding = 0, std430) restrict readonly buffer Input {\n"
           "    " + std::string(glsl_type) + " data_in[];\n"
           "};\n"
           "layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {\n"
           "    " + std::string(glsl_type) + " data_out[];\n"
           "};\n"
           "void main() {\n"
           "    uint index = gl_GlobalInvocationID.x;\n"
           "    if (index >= data_in.length()) return;\n"
           "    data_out[index] = max(" + std::string(glsl_type) + "(0), data_in[index]);\n"
           "}\n";
}

std::string TensorOperations::generate_matmul_kernel_source(DataType dtype) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    return "#version 450\n"
           "#define TILE_SIZE 16\n"
           "#define BLOCK_SIZE 4\n"
           "layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;\n"
           "layout(push_constant) uniform PushConstants {\n"
           "    uint M; uint N; uint K;\n"
           "} pc;\n"
           "layout(set = 0, binding = 0, std430) restrict readonly buffer MatrixA {\n"
           "    " + std::string(glsl_type) + " data_a[];\n"
           "};\n"
           "layout(set = 0, binding = 1, std430) restrict readonly buffer MatrixB {\n"
           "    " + std::string(glsl_type) + " data_b[];\n"
           "};\n"
           "layout(set = 0, binding = 2, std430) restrict writeonly buffer Result {\n"
           "    " + std::string(glsl_type) + " data_result[];\n"
           "};\n"
           "shared " + std::string(glsl_type) + " tile_a[2][TILE_SIZE][TILE_SIZE + 1];\n"
           "shared " + std::string(glsl_type) + " tile_b[2][TILE_SIZE][TILE_SIZE + 1];\n"
           "void main() {\n"
           "    uint local_row = gl_LocalInvocationID.x;\n"
           "    uint local_col = gl_LocalInvocationID.y;\n"
           "    uint base_row = gl_WorkGroupID.x * (TILE_SIZE * BLOCK_SIZE);\n"
           "    uint base_col = gl_WorkGroupID.y * (TILE_SIZE * BLOCK_SIZE);\n"
           "    " + std::string(glsl_type) + " acc[BLOCK_SIZE][BLOCK_SIZE];\n"
           "    for (uint i = 0; i < BLOCK_SIZE; ++i) {\n"
           "        for (uint j = 0; j < BLOCK_SIZE; ++j) {\n"
           "            acc[i][j] = " + std::string(glsl_type) + "(0);\n"
           "        }\n"
           "    }\n"
           "    uint num_tiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;\n"
           "    for (uint t = 0; t < num_tiles; ++t) {\n"
           "        uint tile_k = t * TILE_SIZE;\n"
           "        for (uint bi = 0; bi < BLOCK_SIZE; ++bi) {\n"
           "            uint row = base_row + bi * TILE_SIZE + local_row;\n"
           "            uint col_k = tile_k + local_col;\n"
           "            if (row < pc.M && col_k < pc.K) {\n"
           "                tile_a[bi % 2][local_row][local_col] = data_a[row * pc.K + col_k];\n"
           "            } else {\n"
           "                tile_a[bi % 2][local_row][local_col] = " + std::string(glsl_type) + "(0);\n"
           "            }\n"
           "        }\n"
           "        for (uint bj = 0; bj < BLOCK_SIZE; ++bj) {\n"
           "            uint row_k = tile_k + local_row;\n"
           "            uint col = base_col + bj * TILE_SIZE + local_col;\n"
           "            if (row_k < pc.K && col < pc.N) {\n"
           "                tile_b[bj % 2][local_row][local_col] = data_b[row_k * pc.N + col];\n"
           "            } else {\n"
           "                tile_b[bj % 2][local_row][local_col] = " + std::string(glsl_type) + "(0);\n"
           "            }\n"
           "        }\n"
           "        barrier();\n"
           "        for (uint k = 0; k < TILE_SIZE; k += 4) {\n"
           "            " + std::string(glsl_type) + " a_reg[BLOCK_SIZE][4];\n"
           "            " + std::string(glsl_type) + " b_reg[BLOCK_SIZE][4];\n"
           "            for (uint bi = 0; bi < BLOCK_SIZE; ++bi) {\n"
           "                a_reg[bi][0] = tile_a[bi % 2][local_row][k];\n"
           "                a_reg[bi][1] = tile_a[bi % 2][local_row][k + 1];\n"
           "                a_reg[bi][2] = tile_a[bi % 2][local_row][k + 2];\n"
           "                a_reg[bi][3] = tile_a[bi % 2][local_row][k + 3];\n"
           "            }\n"
           "            for (uint bj = 0; bj < BLOCK_SIZE; ++bj) {\n"
           "                b_reg[bj][0] = tile_b[bj % 2][k][local_col];\n"
           "                b_reg[bj][1] = tile_b[bj % 2][k + 1][local_col];\n"
           "                b_reg[bj][2] = tile_b[bj % 2][k + 2][local_col];\n"
           "                b_reg[bj][3] = tile_b[bj % 2][k + 3][local_col];\n"
           "            }\n"
           "            for (uint bi = 0; bi < BLOCK_SIZE; ++bi) {\n"
           "                for (uint bj = 0; bj < BLOCK_SIZE; ++bj) {\n"
           "                    acc[bi][bj] += a_reg[bi][0] * b_reg[bj][0];\n"
           "                    acc[bi][bj] += a_reg[bi][1] * b_reg[bj][1];\n"
           "                    acc[bi][bj] += a_reg[bi][2] * b_reg[bj][2];\n"
           "                    acc[bi][bj] += a_reg[bi][3] * b_reg[bj][3];\n"
           "                }\n"
           "            }\n"
           "        }\n"
           "        barrier();\n"
           "    }\n"
           "    for (uint bi = 0; bi < BLOCK_SIZE; ++bi) {\n"
           "        for (uint bj = 0; bj < BLOCK_SIZE; ++bj) {\n"
           "            uint out_row = base_row + bi * TILE_SIZE + local_row;\n"
           "            uint out_col = base_col + bj * TILE_SIZE + local_col;\n"
           "            if (out_row < pc.M && out_col < pc.N) {\n"
           "                data_result[out_row * pc.N + out_col] = acc[bi][bj];\n"
           "            }\n"
           "        }\n"
           "    }\n"
           "}\n";
}

std::string TensorOperations::generate_transpose_kernel_source(DataType dtype) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    return "#version 450\n"
           "#define TILE_SIZE 32\n"
           "layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;\n"
           "shared " + std::string(glsl_type) + " tile[TILE_SIZE][TILE_SIZE+1];\n"
           "layout(push_constant) uniform PushConstants {\n"
           "    uint rows; uint cols;\n"
           "} pc;\n"
           "layout(set = 0, binding = 0, std430) restrict readonly buffer Input {\n"
           "    " + std::string(glsl_type) + " data_in[];\n"
           "};\n"
           "layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {\n"
           "    " + std::string(glsl_type) + " data_out[];\n"
           "};\n"
           "void main() {\n"
           "    uint row = gl_WorkGroupID.x * TILE_SIZE + gl_LocalInvocationID.x;\n"
           "    uint col = gl_WorkGroupID.y * TILE_SIZE + gl_LocalInvocationID.y;\n"
           "    if (row < pc.rows && col < pc.cols) {\n"
           "        tile[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = data_in[row * pc.cols + col];\n"
           "    }\n"
           "    barrier();\n"
           "    uint out_row = gl_WorkGroupID.y * TILE_SIZE + gl_LocalInvocationID.x;\n"
           "    uint out_col = gl_WorkGroupID.x * TILE_SIZE + gl_LocalInvocationID.y;\n"
           "    if (out_row < pc.cols && out_col < pc.rows) {\n"
           "        data_out[out_row * pc.rows + out_col] = tile[gl_LocalInvocationID.y][gl_LocalInvocationID.x];\n"
           "    }\n"
           "}\n";
}

std::string TensorOperations::generate_matmul_small_kernel_source(DataType dtype) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    return "#version 450\n"
           "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n"
           "layout(push_constant) uniform PushConstants {\n"
           "    uint M; uint N; uint K;\n"
           "} pc;\n"
           "layout(set = 0, binding = 0, std430) restrict readonly buffer MatrixA {\n"
           "    " + std::string(glsl_type) + " data_a[];\n"
           "};\n"
           "layout(set = 0, binding = 1, std430) restrict readonly buffer MatrixB {\n"
           "    " + std::string(glsl_type) + " data_b[];\n"
           "};\n"
           "layout(set = 0, binding = 2, std430) restrict writeonly buffer Result {\n"
           "    " + std::string(glsl_type) + " data_result[];\n"
           "};\n"
           "void main() {\n"
           "    uint row = gl_WorkGroupID.x * 16 + gl_LocalInvocationID.x;\n"
           "    uint col = gl_WorkGroupID.y * 16 + gl_LocalInvocationID.y;\n"
           "    if (row >= pc.M || col >= pc.N) return;\n"
           "    " + std::string(glsl_type) + " sum = " + std::string(glsl_type) + "(0);\n"
           "    for (uint k = 0; k < pc.K; ++k) {\n"
           "        sum += data_a[row * pc.K + k] * data_b[k * pc.N + col];\n"
           "    }\n"
           "    data_result[row * pc.N + col] = sum;\n"
           "}\n";
}

std::string TensorOperations::generate_sum_axis_kernel_source(DataType dtype) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    return "#version 450\n"
           "layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;\n"
           "layout(push_constant) uniform PushConstants {\n"
           "    uint rows; uint cols; uint axis;\n"
           "} pc;\n"
           "layout(set = 0, binding = 0, std430) restrict readonly buffer Input {\n"
           "    " + std::string(glsl_type) + " data_in[];\n"
           "};\n"
           "layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {\n"
           "    " + std::string(glsl_type) + " data_out[];\n"
           "};\n"
           "void main() {\n"
           "    uint index = gl_GlobalInvocationID.x;\n"
           "    if (pc.axis == 0) {\n"
           "        if (index >= pc.cols) return;\n"
           "        " + std::string(glsl_type) + " sum = " + std::string(glsl_type) + "(0);\n"
           "        for (uint row = 0; row < pc.rows; ++row) {\n"
           "            sum += data_in[row * pc.cols + index];\n"
           "        }\n"
           "        data_out[index] = sum;\n"
           "    } else {\n"
           "        if (index >= pc.rows) return;\n"
           "        " + std::string(glsl_type) + " sum = " + std::string(glsl_type) + "(0);\n"
           "        for (uint col = 0; col < pc.cols; ++col) {\n"
           "            sum += data_in[index * pc.cols + col];\n"
           "        }\n"
           "        data_out[index] = sum;\n"
           "    }\n"
           "}\n";
}

std::string TensorOperations::generate_reduce_axis_kernel_source(DataType dtype, const std::string& op) const {
    // op: "sum", "min", "max"
    const char* glsl_type = dtype_to_glsl_type(dtype);
    std::string body;
    if (op == "sum") {
        body = std::string("        ") + glsl_type + " acc = " + std::string(glsl_type) + "(0);\n"
               "        for (uint r = 0; r < red_size; ++r) { uint in_index = base + r * in_strides[axis]; acc += data_in[in_index]; }\n"
               "        data_out[index] = acc;\n";
    } else if (op == "min") {
        body = std::string("        ") + glsl_type + " acc = data_in[base];\n"
               "        for (uint r = 1; r < red_size; ++r) { uint in_index = base + r * in_strides[axis]; acc = min(acc, data_in[in_index]); }\n"
               "        data_out[index] = acc;\n";
    } else if (op == "max") {
        body = std::string("        ") + glsl_type + " acc = data_in[base];\n"
               "        for (uint r = 1; r < red_size; ++r) { uint in_index = base + r * in_strides[axis]; acc = max(acc, data_in[in_index]); }\n"
               "        data_out[index] = acc;\n";
    } else {
        throw std::runtime_error("Unsupported reduction op");
    }

    // meta layout (u32): max_rank, axis, then in_dims[padded], out_dims[padded], in_strides[padded], out_strides[padded]
    // Build the full GLSL source: we will read meta buffer to access padded dims and strides.
    std::string src = "#version 450\n";
    src += "layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;\n";
    src += "layout(set = 0, binding = 0, std430) restrict readonly buffer Input { " + std::string(glsl_type) + " data_in[]; } ;\n";
    src += "layout(set = 0, binding = 1, std430) restrict writeonly buffer Output { " + std::string(glsl_type) + " data_out[]; } ;\n";
    src += "layout(set = 0, binding = 2, std430) restrict readonly buffer Meta { uint shapes[]; } ;\n";
    src += "void main() { uint index = gl_GlobalInvocationID.x; uint max_rank = shapes[0]; uint axis = shapes[1]; uint off = 2u; uint in_dims_off = off; uint out_dims_off = in_dims_off + max_rank; uint in_strides_off = out_dims_off + max_rank; uint out_strides_off = in_strides_off + max_rank;\n";
    src += "    // reconstruct output multi-index (padded) and compute corresponding base input index (axis coord = 0)\n";
    src += "    uint rem = index; uint base = 0u; uint red_size = shapes[in_dims_off + axis]; uint out_stride = 0u;\n";
    src += "    // compute coords from highest to lowest padded dim\n";
    src += "    for (uint d = 0u; d < max_rank; ++d) {\n";
    src += "        uint dim = shapes[out_dims_off + d];\n";
    src += "        uint stride = shapes[out_strides_off + d];\n";
    src += "        uint coord = 0u;\n";
    src += "        if (stride > 0u) { coord = (rem / stride) % dim; }\n";
    src += "        // map coord to input dim (if input dim == 1 -> 0)\n";
    src += "        uint in_dim = shapes[in_dims_off + d];\n";
    src += "        uint in_stride = shapes[in_strides_off + d];\n";
    src += "        uint in_coord = (in_dim == 1u) ? 0u : coord;\n";
    src += "        if (d != axis) { if (in_stride > 0u) base += in_coord * in_stride; } else { /* axis coord left as 0 for base */ }\n";
    src += "    }\n";
    src += "    // perform reduction along axis: iterate red_size entries using in_strides[axis] to step\n";
    // produce the reduction body by injecting op-specific code; we rely on data_in[base + r * in_strides[axis]] accesses
    src += "    uint in_stride_axis = shapes[in_strides_off + axis];\n";
    if (op == "sum") {
        src += "    " + std::string(glsl_type) + " acc = " + std::string(glsl_type) + "(0);\n";
        src += "    for (uint r = 0u; r < red_size; ++r) { acc += data_in[base + r * in_stride_axis]; }\n";
        src += "    data_out[index] = acc;\n";
    } else if (op == "min") {
        src += "    " + std::string(glsl_type) + " acc = data_in[base];\n";
        src += "    for (uint r = 1u; r < red_size; ++r) { acc = min(acc, data_in[base + r * in_stride_axis]); }\n";
        src += "    data_out[index] = acc;\n";
    } else if (op == "max") {
        src += "    " + std::string(glsl_type) + " acc = data_in[base];\n";
        src += "    for (uint r = 1u; r < red_size; ++r) { acc = max(acc, data_in[base + r * in_stride_axis]); }\n";
        src += "    data_out[index] = acc;\n";
    }
    src += "}\n";
    return src;
}

std::string TensorOperations::get_kernel_name_for_dtype(const std::string& base_name, DataType dtype) const {
    return base_name + "_" + std::string(dtype_to_string(dtype));
}

// First-pass: per (out_index, group_id) workgroup reduces a slice of the reduction axis and writes partials
std::string TensorOperations::generate_reduce_axis_first_pass_kernel_source(DataType dtype, const std::string& op, u32 local_size) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    std::string LOCAL = std::to_string(local_size);
    std::string src = "#version 450\n";
    src += "layout(local_size_x = " + LOCAL + ", local_size_y = 1, local_size_z = 1) in;\n";
    src += "layout(set = 0, binding = 0, std430) restrict readonly buffer Input { " + std::string(glsl_type) + " data_in[]; } ;\n";
    src += "layout(set = 0, binding = 1, std430) restrict writeonly buffer Partials { " + std::string(glsl_type) + " partials[]; } ;\n";
    src += "layout(set = 0, binding = 2, std430) restrict readonly buffer Meta { uint shapes[]; } ;\n";
    src += "layout(push_constant) uniform PC { uint group_size; uint group_count; uint axis; } pc;\n";
    src += "shared " + std::string(glsl_type) + " sdata[" + LOCAL + "];\n";
    src += "void main() { uint out_index = gl_GlobalInvocationID.x; uint group_id = gl_GlobalInvocationID.y; uint local_id = gl_LocalInvocationID.x; uint max_rank = shapes[0]; uint off = 2u; uint in_dims_off = off; uint out_dims_off = in_dims_off + max_rank; uint in_strides_off = out_dims_off + max_rank; uint out_strides_off = in_strides_off + max_rank; uint rem = out_index; uint base = 0u;\n";
    src += "    for (uint d = 0u; d < max_rank; ++d) { uint dim = shapes[out_dims_off + d]; uint stride = shapes[out_strides_off + d]; uint coord = 0u; if (stride > 0u) { coord = (rem / stride) % dim; } uint in_dim = shapes[in_dims_off + d]; uint in_stride = shapes[in_strides_off + d]; uint in_coord = (in_dim == 1u) ? 0u : coord; if (d != pc.axis) { if (in_stride > 0u) base += in_coord * in_stride; } }\n";
    src += "    uint red_size = shapes[in_dims_off + pc.axis]; uint start = group_id * pc.group_size; uint end = start + pc.group_size; if (end > red_size) end = red_size;\n";
    // For sum we can start from zero; for min/max initialize from the first assigned element
    if (op == "sum") {
        src += "    " + std::string(glsl_type) + " acc = " + std::string(glsl_type) + "(0);\n";
        src += "    uint stride_axis = shapes[in_strides_off + pc.axis];\n";
        src += "    for (uint idx = start + local_id; idx < end; idx += " + LOCAL + ") { uint in_index = base + idx * stride_axis; acc += data_in[in_index]; }\n";
    } else if (op == "min" || op == "max") {
        src += "    uint stride_axis = shapes[in_strides_off + pc.axis];\n";
        // declare acc first so it's visible after the conditional
        src += "    " + std::string(glsl_type) + " acc;\n";
        src += "    uint first_idx = start + local_id;\n";
        src += "    if (first_idx < end) {\n";
        src += "        acc = data_in[base + first_idx * stride_axis];\n";
        // loop over remaining assigned indices for this thread
        src += "        for (uint idx = first_idx + " + LOCAL + "; idx < end; idx += " + LOCAL + ") { ";
        if (op == "min") src += " acc = min(acc, data_in[base + idx * stride_axis]); ";
        if (op == "max") src += " acc = max(acc, data_in[base + idx * stride_axis]); ";
        src += " }\n";
        src += "    } else {\n";
        // no local element assigned: use base element as a neutral-but-valid value (base is within reduction range)
        src += "        acc = data_in[base];\n";
        src += "    }\n";
    }
    src += "    sdata[local_id] = acc;\n    barrier();\n";
    src += "    for (uint s = " + std::to_string(local_size/2) + "u; s > 0u; s >>= 1u) { if (local_id < s) { sdata[local_id] = ";
    if (op == "sum") src += "sdata[local_id] + sdata[local_id + s];";
    if (op == "min") src += "min(sdata[local_id], sdata[local_id + s]);";
    if (op == "max") src += "max(sdata[local_id], sdata[local_id + s]);";
    src += " } barrier(); }\n";
    src += "    if (local_id == 0u) { uint out_pos = out_index * pc.group_count + group_id; partials[out_pos] = sdata[0]; }\n";
    src += "}\n";
    return src;
}

// Second-pass: reduce partials across group_count for each output index
std::string TensorOperations::generate_reduce_axis_second_pass_kernel_source(DataType dtype, const std::string& op) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    std::string src = "#version 450\n";
    src += "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n";
    src += "layout(set = 0, binding = 0, std430) restrict readonly buffer Partials { " + std::string(glsl_type) + " partials[]; } ;\n";
    src += "layout(set = 0, binding = 1, std430) restrict writeonly buffer Output { " + std::string(glsl_type) + " data_out[]; } ;\n";
    src += "layout(push_constant) uniform PC { uint group_count; } pc;\n";
    src += "void main() { uint index = gl_GlobalInvocationID.x; if (index >= data_out.length()) return; uint base = index * pc.group_count; ";
    if (op == "sum") {
        src += "    " + std::string(glsl_type) + " acc = " + std::string(glsl_type) + "(0); for (uint g = 0u; g < pc.group_count; ++g) acc += partials[base + g]; data_out[index] = acc; }\n";
    } else if (op == "min") {
        // initialize from first partial and reduce the rest
        src += "    " + std::string(glsl_type) + " acc = partials[base]; for (uint g = 1u; g < pc.group_count; ++g) acc = min(acc, partials[base + g]); data_out[index] = acc; }\n";
    } else if (op == "max") {
        src += "    " + std::string(glsl_type) + " acc = partials[base]; for (uint g = 1u; g < pc.group_count; ++g) acc = max(acc, partials[base + g]); data_out[index] = acc; }\n";
    }
    return src;
}

// Helper: generate simple activation kernel (sigmoid, tanh, abs, neg)
std::string TensorOperations::generate_activation_kernel_source(DataType dtype, const std::string& activation_func) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    std::string src = "#version 450\n";
    src += "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n";
    src += "layout(set = 0, binding = 0, std430) restrict readonly buffer Input {\n";
    src += "    " + std::string(glsl_type) + " data_in[];\n";
    src += "};\n";
    src += "layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {\n";
    src += "    " + std::string(glsl_type) + " data_out[];\n";
    src += "};\n";
    src += "void main() {\n";
    src += "    uint index = gl_GlobalInvocationID.x;\n";
    src += "    if (index >= data_in.length()) return;\n";
    src += "    data_out[index] = " + std::string(glsl_type) + "(" + activation_func + ");\n";
    src += "}\n";
    
    return src;
}

// Helper: generate permute kernel
std::string TensorOperations::generate_permute_kernel_source(DataType dtype) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    std::string src = "#version 450\n";
    src += "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n";
    src += "layout(set = 0, binding = 0, std430) restrict readonly buffer Input { " + std::string(glsl_type) + " data_in[]; } ;\n";
    src += "layout(set = 0, binding = 1, std430) restrict writeonly buffer Output { " + std::string(glsl_type) + " data_out[]; } ;\n";
    src += "layout(set = 0, binding = 2, std430) restrict readonly buffer Meta { uint data[]; } ;\n";
    src += "void main() {\n";
    src += "    uint out_idx = gl_GlobalInvocationID.x;\n";
    src += "    if (out_idx >= data_out.length()) return;\n";
    src += "    \n";
    src += "    // Meta layout: [rank, dims..., in_shape..., out_shape..., in_strides..., out_strides...]\n";
    src += "    uint rank = data[0];\n";
    src += "    uint dims_off = 1;\n";
    src += "    uint in_shape_off = dims_off + rank;\n";
    src += "    uint out_shape_off = in_shape_off + rank;\n";
    src += "    uint in_strides_off = out_shape_off + rank;\n";
    src += "    uint out_strides_off = in_strides_off + rank;\n";
    src += "    \n";
    src += "    // Decompose output index into coordinates\n";
    src += "    uint rem = out_idx;\n";
    src += "    uint coords[16]; // max rank support\n";
    src += "    for (uint i = 0; i < rank; ++i) {\n";
    src += "        uint stride = data[out_strides_off + i];\n";
    src += "        if (stride > 0) {\n";
    src += "            coords[i] = rem / stride;\n";
    src += "            rem = rem % stride;\n";
    src += "        } else {\n";
    src += "            coords[i] = 0;\n";
    src += "        }\n";
    src += "    }\n";
    src += "    \n";
    src += "    // Map coordinates through permutation to get input index\n";
    src += "    uint in_idx = 0;\n";
    src += "    for (uint i = 0; i < rank; ++i) {\n";
    src += "        uint out_dim = i;  // which output dimension we're looking at\n";
    src += "        uint in_dim = data[dims_off + i];  // which input dimension it came from\n";
    src += "        uint out_coord = coords[out_dim];  // coordinate in output space\n";
    src += "        // This output coordinate corresponds to input dimension in_dim\n";
    src += "        uint in_stride = data[in_strides_off + in_dim];\n";
    src += "        in_idx += out_coord * in_stride;\n";
    src += "    }\n";
    src += "    \n";
    src += "    data_out[out_idx] = data_in[in_idx];\n";
    src += "}\n";
    
    return src;
}

// Helper: generate concatenate kernel
std::string TensorOperations::generate_concatenate_kernel_source(DataType dtype) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    std::string src = "#version 450\n";
    src += "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n";
    src += "layout(set = 0, binding = 0, std430) restrict readonly buffer Input0 { " + std::string(glsl_type) + " data0[]; } ;\n";
    src += "layout(set = 0, binding = 1, std430) restrict readonly buffer Input1 { " + std::string(glsl_type) + " data1[]; } ;\n";
    src += "layout(set = 0, binding = 2, std430) restrict readonly buffer Input2 { " + std::string(glsl_type) + " data2[]; } ;\n";
    src += "layout(set = 0, binding = 3, std430) restrict readonly buffer Input3 { " + std::string(glsl_type) + " data3[]; } ;\n";
    src += "layout(set = 0, binding = 4, std430) restrict readonly buffer Input4 { " + std::string(glsl_type) + " data4[]; } ;\n";
    src += "layout(set = 0, binding = 5, std430) restrict readonly buffer Input5 { " + std::string(glsl_type) + " data5[]; } ;\n";
    src += "layout(set = 0, binding = 6, std430) restrict readonly buffer Input6 { " + std::string(glsl_type) + " data6[]; } ;\n";
    src += "layout(set = 0, binding = 7, std430) restrict readonly buffer Input7 { " + std::string(glsl_type) + " data7[]; } ;\n";
    src += "layout(set = 0, binding = 8, std430) restrict writeonly buffer Output { " + std::string(glsl_type) + " data_out[]; } ;\n";
    src += "layout(set = 0, binding = 9, std430) restrict readonly buffer Meta { uint data[]; } ;\n";
    src += "void main() {\n";
    src += "    uint out_idx = gl_GlobalInvocationID.x;\n";
    src += "    if (out_idx >= data_out.length()) return;\n";
    src += "    \n";
    src += "    // Meta layout: [num_tensors, axis, rank, out_shape[rank], out_strides[rank], \n";
    src += "    //                offset0...offset7, size0...size7, in_strides0[rank]...in_strides7[rank]]\n";
    src += "    uint num_tensors = data[0];\n";
    src += "    uint axis = data[1];\n";
    src += "    uint rank = data[2];\n";
    src += "    uint out_shape_off = 3;\n";
    src += "    uint out_strides_off = out_shape_off + rank;\n";
    src += "    uint offsets_off = out_strides_off + rank;\n";
    src += "    uint sizes_off = offsets_off + 8;\n";
    src += "    uint in_strides_base = sizes_off + 8;\n";
    src += "    \n";
    src += "    // Decompose output index into coordinates\n";
    src += "    uint coords[16];\n";
    src += "    uint rem = out_idx;\n";
    src += "    for (uint i = 0; i < rank; ++i) {\n";
    src += "        uint stride = data[out_strides_off + i];\n";
    src += "        if (stride > 0) {\n";
    src += "            coords[i] = rem / stride;\n";
    src += "            rem = rem % stride;\n";
    src += "        } else {\n";
    src += "            coords[i] = 0;\n";
    src += "        }\n";
    src += "    }\n";
    src += "    \n";
    src += "    // Determine which input tensor this output element comes from\n";
    src += "    uint axis_coord = coords[axis];\n";
    src += "    uint tensor_idx = 0;\n";
    src += "    uint adjusted_coord = axis_coord;\n";
    src += "    for (uint t = 0; t < num_tensors; ++t) {\n";
    src += "        uint offset = data[offsets_off + t];\n";
    src += "        uint size = data[sizes_off + t];\n";
    src += "        if (axis_coord >= offset && axis_coord < offset + size) {\n";
    src += "            tensor_idx = t;\n";
    src += "            adjusted_coord = axis_coord - offset;\n";
    src += "            break;\n";
    src += "        }\n";
    src += "    }\n";
    src += "    \n";
    src += "    // Calculate source index within the selected input tensor using its strides\n";
    src += "    uint in_strides_off = in_strides_base + tensor_idx * rank;\n";
    src += "    uint src_idx = 0;\n";
    src += "    for (uint i = 0; i < rank; ++i) {\n";
    src += "        uint coord = (i == axis) ? adjusted_coord : coords[i];\n";
    src += "        uint stride = data[in_strides_off + i];\n";
    src += "        src_idx += coord * stride;\n";
    src += "    }\n";
    src += "    \n";
    src += "    // Read from appropriate input buffer (unrolled switch for up to 8 tensors)\n";
    src += "    " + std::string(glsl_type) + " value;\n";
    src += "    if (tensor_idx == 0) value = data0[src_idx];\n";
    src += "    else if (tensor_idx == 1) value = data1[src_idx];\n";
    src += "    else if (tensor_idx == 2) value = data2[src_idx];\n";
    src += "    else if (tensor_idx == 3) value = data3[src_idx];\n";
    src += "    else if (tensor_idx == 4) value = data4[src_idx];\n";
    src += "    else if (tensor_idx == 5) value = data5[src_idx];\n";
    src += "    else if (tensor_idx == 6) value = data6[src_idx];\n";
    src += "    else value = data7[src_idx];\n";
    src += "    \n";
    src += "    data_out[out_idx] = value;\n";
    src += "}\n";
    
    return src;
}

std::string TensorOperations::generate_layer_norm_kernel_source(DataType dtype) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    return "#version 450\n"
           "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
           "layout(push_constant) uniform PushConstants {\n"
           "    uint batch_size;\n"
           "    uint feature_dim;\n"
           "    float epsilon;\n"
           "} pc;\n"
           "layout(set = 0, binding = 0, std430) restrict readonly buffer Input {\n"
           "    " + std::string(glsl_type) + " data_in[];\n"
           "};\n"
           "layout(set = 0, binding = 1, std430) restrict readonly buffer Gamma {\n"
           "    " + std::string(glsl_type) + " gamma[];\n"
           "};\n"
           "layout(set = 0, binding = 2, std430) restrict readonly buffer Beta {\n"
           "    " + std::string(glsl_type) + " beta[];\n"
           "};\n"
           "layout(set = 0, binding = 3, std430) restrict writeonly buffer Output {\n"
           "    " + std::string(glsl_type) + " data_out[];\n"
           "};\n"
           "void main() {\n"
           "    uint batch_idx = gl_GlobalInvocationID.x;\n"
           "    if (batch_idx >= pc.batch_size) return;\n"
           "    \n"
           "    uint offset = batch_idx * pc.feature_dim;\n"
           "    \n"
           "    " + std::string(glsl_type) + " mean = " + std::string(glsl_type) + "(0);\n"
           "    for (uint i = 0; i < pc.feature_dim; ++i) {\n"
           "        mean += data_in[offset + i];\n"
           "    }\n"
           "    mean /= " + std::string(glsl_type) + "(pc.feature_dim);\n"
           "    \n"
           "    " + std::string(glsl_type) + " variance = " + std::string(glsl_type) + "(0);\n"
           "    for (uint i = 0; i < pc.feature_dim; ++i) {\n"
           "        " + std::string(glsl_type) + " diff = data_in[offset + i] - mean;\n"
           "        variance += diff * diff;\n"
           "    }\n"
           "    variance /= " + std::string(glsl_type) + "(pc.feature_dim);\n"
           "    " + std::string(glsl_type) + " std_dev = sqrt(variance + " + std::string(glsl_type) + "(pc.epsilon));\n"
           "    \n"
           "    for (uint i = 0; i < pc.feature_dim; ++i) {\n"
           "        " + std::string(glsl_type) + " normalized = (data_in[offset + i] - mean) / std_dev;\n"
           "        data_out[offset + i] = gamma[i] * normalized + beta[i];\n"
           "    }\n"
           "}\n";
}

} // namespace QuasarML
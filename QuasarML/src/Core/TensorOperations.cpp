#include "TensorOperations.h"
#include "Accelerator.h"
#include "Tensor.h"
#include "Kernel.h"
#include <cstring>
#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

namespace QuasarML {

// forward declarations for helpers used by scalar fast-paths
static float half_to_float(uint16_t h);
static std::pair<std::array<u8,4>, u32> make_push_constant(const u8* buf, DataType dtype);

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
            // download both tensors
            u64 out_count = calculate_element_count(out_shape);
            std::vector<float> out_buf(out_count);
            // simple generic broadcasting: expand indices
            // download raw data as float for now (only F32 supported in CPU fallback)
            if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32 in this simplified path");
            u64 a_count = a->get_element_count();
            u64 b_count = b->get_element_count();
            std::vector<float> a_buf(a_count), b_buf(b_count);
            a->download_data(a_buf.data());
            b->download_data(b_buf.data());
            // compute broadcasted add
            auto get_val = [](const std::vector<float>& buf, const std::vector<u32>& shape, const std::vector<u32>& idx) {
                // compute flat index (C-order)
                u64 flat = 0; u64 stride = 1;
                for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
                    flat += static_cast<u64>(idx[i]) * stride;
                    stride *= shape[i];
                }
                return buf[flat];
            };
            std::vector<u32> idx(out_shape.size());
            std::vector<u32> a_shape = a->get_shape(); std::vector<u32> b_shape = b->get_shape();
            // iterate over out indices (naive multi-loop)
            for (u64 i = 0; i < out_count; ++i) {
                // unravel
                u64 rem = i; u64 denom = out_count;
                for (size_t d = 0; d < out_shape.size(); ++d) {
                    denom /= out_shape[d];
                    idx[d] = static_cast<u32>(rem / denom);
                    rem = rem % denom;
                }
                // map to a/b coords (right align)
                std::vector<u32> a_idx(a_shape.size()), b_idx(b_shape.size());
                for (int d = 0; d < static_cast<int>(a_shape.size()); ++d) {
                    int od = static_cast<int>(out_shape.size()) - static_cast<int>(a_shape.size()) + d;
                    if (od < 0) a_idx[d] = 0; else a_idx[d] = (a_shape[d] == 1) ? 0 : idx[od];
                }
                for (int d = 0; d < static_cast<int>(b_shape.size()); ++d) {
                    int od = static_cast<int>(out_shape.size()) - static_cast<int>(b_shape.size()) + d;
                    if (od < 0) b_idx[d] = 0; else b_idx[d] = (b_shape[d] == 1) ? 0 : idx[od];
                }
                float va = get_val(a_buf, a_shape, a_idx);
                float vb = get_val(b_buf, b_shape, b_idx);
                out_buf[i] = va + vb;
            }
            auto result = _accelerator.create_tensor(out_buf.data(), out_shape, dtype, /*device_only=*/false);
            return result;
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
        // CPU elementwise path for equal-shaped tensors (F32 only simplified)
        if (a->get_dtype() != DataType::F32 || b->get_dtype() != DataType::F32) throw std::runtime_error("CPU fallback only supports F32 for elementwise ops");
        u64 count = a->get_element_count();
        std::vector<float> a_buf(count), b_buf(count), out_buf(count);
        a->download_data(a_buf.data());
        b->download_data(b_buf.data());
        for (u64 i = 0; i < count; ++i) out_buf[i] = a_buf[i] + b_buf[i];
        auto result = _accelerator.create_tensor(out_buf.data(), a->get_shape(), a->get_dtype(), /*device_only=*/false);
        return result;
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
            if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32");
            u64 out_count = calculate_element_count(out_shape);
            std::vector<float> out_buf(out_count);
            u64 a_count = a->get_element_count(); std::vector<float> a_buf(a_count); a->download_data(a_buf.data());
            u64 b_count = b->get_element_count(); std::vector<float> b_buf(b_count); b->download_data(b_buf.data());
            // naive broadcast as in add
            std::vector<u32> idx(out_shape.size()); std::vector<u32> a_shape = a->get_shape(); std::vector<u32> b_shape = b->get_shape();
            for (u64 i = 0; i < out_count; ++i) {
                u64 rem = i; u64 denom = out_count;
                for (size_t d = 0; d < out_shape.size(); ++d) { denom /= out_shape[d]; idx[d] = static_cast<u32>(rem / denom); rem = rem % denom; }
                std::vector<u32> a_idx(a_shape.size()), b_idx(b_shape.size());
                for (int d = 0; d < static_cast<int>(a_shape.size()); ++d) { int od = static_cast<int>(out_shape.size()) - static_cast<int>(a_shape.size()) + d; if (od < 0) a_idx[d] = 0; else a_idx[d] = (a_shape[d] == 1) ? 0 : idx[od]; }
                for (int d = 0; d < static_cast<int>(b_shape.size()); ++d) { int od = static_cast<int>(out_shape.size()) - static_cast<int>(b_shape.size()) + d; if (od < 0) b_idx[d] = 0; else b_idx[d] = (b_shape[d] == 1) ? 0 : idx[od]; }
                auto get_val = [](const std::vector<float>& buf, const std::vector<u32>& shape, const std::vector<u32>& idx)->float { u64 flat=0, stride=1; for (int i = static_cast<int>(shape.size())-1;i>=0;--i){ flat += static_cast<u64>(idx[i])*stride; stride *= shape[i]; } return buf[flat]; };
                out_buf[i] = get_val(a_buf, a_shape, a_idx) - get_val(b_buf, b_shape, b_idx);
            }
            return _accelerator.create_tensor(out_buf.data(), out_shape, dtype, false);
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
        if (a->get_dtype() != DataType::F32 || b->get_dtype() != DataType::F32) throw std::runtime_error("CPU fallback only supports F32");
        u64 count = a->get_element_count(); std::vector<float> a_buf(count), b_buf(count), out_buf(count);
        a->download_data(a_buf.data()); b->download_data(b_buf.data());
        for (u64 i = 0; i < count; ++i) out_buf[i] = a_buf[i] - b_buf[i];
        return _accelerator.create_tensor(out_buf.data(), a->get_shape(), a->get_dtype(), false);
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
            if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32");
            u64 out_count = calculate_element_count(out_shape); std::vector<float> out_buf(out_count);
            u64 a_count = a->get_element_count(); std::vector<float> a_buf(a_count); a->download_data(a_buf.data());
            u64 b_count = b->get_element_count(); std::vector<float> b_buf(b_count); b->download_data(b_buf.data());
            std::vector<u32> idx(out_shape.size()); std::vector<u32> a_shape=a->get_shape(), b_shape=b->get_shape();
            for (u64 i=0;i<out_count;++i){ u64 rem=i, denom=out_count; for (size_t d=0; d<out_shape.size(); ++d){ denom/=out_shape[d]; idx[d]=static_cast<u32>(rem/denom); rem%=denom; } std::vector<u32> a_idx(a_shape.size()), b_idx(b_shape.size()); for (int d=0; d<static_cast<int>(a_shape.size()); ++d){ int od = static_cast<int>(out_shape.size())-static_cast<int>(a_shape.size())+d; a_idx[d]= (od<0)?0:((a_shape[d]==1)?0:idx[od]); } for (int d=0; d<static_cast<int>(b_shape.size()); ++d){ int od = static_cast<int>(out_shape.size())-static_cast<int>(b_shape.size())+d; b_idx[d]= (od<0)?0:((b_shape[d]==1)?0:idx[od]); } auto get_val=[&](const std::vector<float>& buf,const std::vector<u32>& shape,const std::vector<u32>& idx){ u64 flat=0,stride=1; for (int k=static_cast<int>(shape.size())-1;k>=0;--k){ flat+=static_cast<u64>(idx[k])*stride; stride*=shape[k]; } return buf[flat]; }; out_buf[i]=get_val(a_buf,a_shape,a_idx)*get_val(b_buf,b_shape,b_idx); }
            return _accelerator.create_tensor(out_buf.data(), out_shape, dtype, false);
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
            if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32");
            u64 out_count = calculate_element_count(out_shape); std::vector<float> out_buf(out_count);
            u64 a_count = a->get_element_count(); std::vector<float> a_buf(a_count); a->download_data(a_buf.data());
            u64 b_count = b->get_element_count(); std::vector<float> b_buf(b_count); b->download_data(b_buf.data());
            std::vector<u32> idx(out_shape.size()); std::vector<u32> a_shape=a->get_shape(), b_shape=b->get_shape();
            for (u64 i=0;i<out_count;++i){ u64 rem=i, denom=out_count; for (size_t d=0; d<out_shape.size(); ++d){ denom/=out_shape[d]; idx[d]=static_cast<u32>(rem/denom); rem%=denom; } std::vector<u32> a_idx(a_shape.size()), b_idx(b_shape.size()); for (int d=0; d<static_cast<int>(a_shape.size()); ++d){ int od = static_cast<int>(out_shape.size())-static_cast<int>(a_shape.size())+d; a_idx[d]= (od<0)?0:((a_shape[d]==1)?0:idx[od]); } for (int d=0; d<static_cast<int>(b_shape.size()); ++d){ int od = static_cast<int>(out_shape.size())-static_cast<int>(b_shape.size())+d; b_idx[d]= (od<0)?0:((b_shape[d]==1)?0:idx[od]); } auto get_val=[&](const std::vector<float>& buf,const std::vector<u32>& shape,const std::vector<u32>& idx){ u64 flat=0,stride=1; for (int k=static_cast<int>(shape.size())-1;k>=0;--k){ flat+=static_cast<u64>(idx[k])*stride; stride*=shape[k]; } return buf[flat]; }; out_buf[i]=get_val(a_buf,a_shape,a_idx)/get_val(b_buf,b_shape,b_idx); }
            return _accelerator.create_tensor(out_buf.data(), out_shape, dtype, false);
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
        if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32");
        u64 count = tensor->get_element_count(); std::vector<float> in(count), out(count);
        tensor->download_data(in.data()); for (u64 i=0;i<count;++i) out[i]=in[i]+scalar; return _accelerator.create_tensor(out.data(), tensor->get_shape(), dtype, false);
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
        if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32");
        u64 count = tensor->get_element_count(); std::vector<float> in(count), out(count);
        tensor->download_data(in.data()); for (u64 i=0;i<count;++i) out[i]=in[i]*scalar; return _accelerator.create_tensor(out.data(), tensor->get_shape(), dtype, false);
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
        if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32");
        u64 count = tensor->get_element_count(); std::vector<float> in(count), out(count);
        tensor->download_data(in.data()); for (u64 i=0;i<count;++i) out[i]= std::max(0.0f, in[i]); return _accelerator.create_tensor(out.data(), tensor->get_shape(), dtype, false);
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
        if (dtype != DataType::F32) throw std::runtime_error("CPU fallback only supports F32");
        u32 M = a_shape[0]; u32 K = a_shape[1]; u32 N = b_shape[1]; std::vector<float> A(M*K), B(K*N), C(M*N);
        a->download_data(A.data()); b->download_data(B.data()); for (u32 i=0;i<M;++i) for (u32 j=0;j<N;++j){ float s=0; for (u32 k=0;k<K;++k) s+=A[i*K+k]*B[k*N+j]; C[i*N+j]=s; }
        return _accelerator.create_tensor(C.data(), result_shape, dtype, false);
    }

    auto result = _accelerator.create_tensor(result_shape, dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("matmul", dtype);
    std::string glsl_source = generate_matmul_kernel_source(dtype);
    
    struct MatMulPushConstants {
        u32 M, N, K;
    } push_data = {a_shape[0], b_shape[1], a_shape[1]};

    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3, sizeof(MatMulPushConstants));
    
    u32 dispatch_x = (a_shape[0] + 15) / 16;
    u32 dispatch_y = (b_shape[1] + 15) / 16;
    
    _accelerator.execute(kernel, {a, b, result}, dispatch_x, dispatch_y, 1, &push_data);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::transpose(std::shared_ptr<Tensor> tensor) {
    validate_tensor_shape_2d(tensor);
    
    auto input_shape = tensor->get_shape();
    std::vector<u32> result_shape = {input_shape[1], input_shape[0]};
    
    DataType dtype = tensor->get_dtype();
    if (!_accelerator.use_gpu()) {
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
    
    u32 dispatch_x = (input_shape[0] + 15) / 16;
    u32 dispatch_y = (input_shape[1] + 15) / 16;
    
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
    auto partials = _accelerator.create_tensor(nullptr, {static_cast<u32>(partials_count)}, dtype, true);

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
    // division by scalar
    u32 len = tensor->get_shape()[axis];
    float denom = static_cast<float>(len);
    auto denom_t = _accelerator.create_tensor(&denom, {1}, tensor->get_dtype());
    return div(sumt, denom_t);
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
    auto partials = _accelerator.create_tensor(nullptr, {static_cast<u32>(partials_count)}, dtype, true);

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
    DataType dtype = tensor->get_dtype(); auto shape = tensor->get_shape(); std::vector<u32> out_shape; for (u32 i=0;i<shape.size();++i) if (i!=axis) out_shape.push_back(shape[i]); auto result = _accelerator.create_tensor(out_shape, dtype);
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
    auto partials2 = _accelerator.create_tensor(nullptr, {static_cast<u32>(partials_count2)}, dtype, true);

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
    std::cout << "[slice] non-contiguous GPU gather: dtype=" << dtype_to_string(dtype) << " rank=" << rank << " res_count=" << res_count << " start_flat=" << start_flat << "\n";
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

    std::cout << "[slice] meta(" << meta.size() << "): ";
    for (auto m : meta) std::cout << m << " ";
    std::cout << "\n";

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
    
    return _accelerator.create_kernel(name, glsl_source, num_tensors, push_constant_size);
}

std::string TensorOperations::generate_elementwise_kernel_source(const std::string& operation, 
                                                                DataType dtype, 
                                                                bool is_scalar,
                                                                bool supports_broadcast,
                                                                u32 max_rank) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    std::string source = "#version 450\n"
                        "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n";
    
    if (is_scalar) {
        // choose push-constant type matching dtype
        const char* pc_type = "float";
        switch (dtype) {
            case DataType::F32: pc_type = "float"; break;
            case DataType::F16: pc_type = "float"; break; // promote half to float for push constant
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
           "    uint row = gl_GlobalInvocationID.x;\n"
           "    uint col = gl_GlobalInvocationID.y;\n"
           "    if (row >= pc.M || col >= pc.N) return;\n"
           "    " + std::string(glsl_type) + " sum = " + std::string(glsl_type) + "(0);\n"
           "    for (uint k = 0; k < pc.K; ++k) {\n"
           "        sum += data_a[row * pc.K + k] * data_b[k * pc.N + col];\n"
           "    }\n"
           "    data_result[row * pc.N + col] = sum;\n"
           "}\n";
}

std::string TensorOperations::generate_transpose_kernel_source(DataType dtype) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    return "#version 450\n"
           "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n"
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
           "    uint row = gl_GlobalInvocationID.x;\n"
           "    uint col = gl_GlobalInvocationID.y;\n"
           "    if (row >= pc.rows || col >= pc.cols) return;\n"
           "    data_out[col * pc.rows + row] = data_in[row * pc.cols + col];\n"
           "}\n";
}

std::string TensorOperations::generate_sum_axis_kernel_source(DataType dtype) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    return "#version 450\n"
           "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
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
    src += "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n";
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

} // namespace QuasarML
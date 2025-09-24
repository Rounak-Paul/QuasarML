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

    // scalar broadcast fast-path
    if (a->get_element_count() == 1) {
        std::vector<u8> buf(a->get_element_size());
        a->download_data(buf.data());
        float scalar = 0.0f;
        if (a->get_dtype() == DataType::F32) std::memcpy(&scalar, buf.data(), sizeof(float));
        return add_scalar(b, scalar);
    }
    if (b->get_element_count() == 1) {
        std::vector<u8> buf(b->get_element_size());
        b->download_data(buf.data());
        float scalar = 0.0f;
        if (b->get_dtype() == DataType::F32) std::memcpy(&scalar, buf.data(), sizeof(float));
        return add_scalar(a, scalar);
    }

    if (!a->is_shape_compatible(*b)) {
        // try general broadcasting
        auto out_shape = compute_broadcast_shape(a->get_shape(), b->get_shape());
        if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");

        DataType dtype = a->get_dtype();
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

        // create host-visible tensor for meta
        auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, false);

        std::string kernel_name = get_kernel_name_for_dtype("add_broadcast", dtype);
        std::string glsl_source = generate_elementwise_kernel_source("data_a[a_index] + data_b[b_index]", dtype, false, true, max_rank);
        auto kernel = get_or_create_kernel(kernel_name, glsl_source, 4);
        u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(calculate_element_count(out_shape)));
        _accelerator.execute(kernel, {a, b, result, meta_tensor}, dispatch_size);
        return result;
    }

    if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");

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

    if (a->get_element_count() == 1) {
        std::vector<u8> buf(a->get_element_size());
        a->download_data(buf.data());
        float scalar = 0.0f;
        if (a->get_dtype() == DataType::F32) std::memcpy(&scalar, buf.data(), sizeof(float));
        // scalar - tensor : create kernel that computes pc.scalar - data_in[index]
        DataType dtype = b->get_dtype();
        auto result = _accelerator.create_tensor(b->get_shape(), dtype);
        std::string kernel_name = get_kernel_name_for_dtype("sub_scalar_left", dtype);
        std::string glsl_source = generate_elementwise_kernel_source("pc.scalar - data_in[index]", dtype, true);
        auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2, sizeof(float));
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(b->get_element_count()));
    auto pc = make_push_constant(buf.data(), b->get_dtype());
    _accelerator.execute(kernel, {b, result}, dispatch_size, 1, 1, pc.first.data());
        return result;
    }
    if (b->get_element_count() == 1) {
        std::vector<u8> buf(b->get_element_size());
        b->download_data(buf.data());
        float scalar = 0.0f;
        if (b->get_dtype() == DataType::F32) std::memcpy(&scalar, buf.data(), sizeof(float));
        auto result = _accelerator.create_tensor(a->get_shape(), a->get_dtype());
        std::string kernel_name = get_kernel_name_for_dtype("sub_scalar", a->get_dtype());
        std::string glsl_source = generate_elementwise_kernel_source("data_in[index] - pc.scalar", a->get_dtype(), true);
        auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2, sizeof(float));
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    auto pc = make_push_constant(buf.data(), a->get_dtype());
    _accelerator.execute(kernel, {a, result}, dispatch_size, 1, 1, pc.first.data());
        return result;
    }

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

        auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, false);

        std::string kernel_name = get_kernel_name_for_dtype("sub_broadcast", dtype);
        std::string glsl_source = generate_elementwise_kernel_source("data_a[a_index] - data_b[b_index]", dtype, false, true, max_rank);
        auto kernel = get_or_create_kernel(kernel_name, glsl_source, 4);
        u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(calculate_element_count(out_shape)));
        _accelerator.execute(kernel, {a, b, result, meta_tensor}, dispatch_size);
        return result;
    }

    if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");

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

    if (a->get_element_count() == 1) {
        std::vector<u8> buf(a->get_element_size());
        a->download_data(buf.data());
        float scalar = 0.0f;
        if (a->get_dtype() == DataType::F32) std::memcpy(&scalar, buf.data(), sizeof(float));
        return mul_scalar(b, scalar);
    }
    if (b->get_element_count() == 1) {
        std::vector<u8> buf(b->get_element_size());
        b->download_data(buf.data());
        float scalar = 0.0f;
        if (b->get_dtype() == DataType::F32) std::memcpy(&scalar, buf.data(), sizeof(float));
        return mul_scalar(a, scalar);
    }

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

        auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, false);

        std::string kernel_name = get_kernel_name_for_dtype("mul_broadcast", dtype);
        std::string glsl_source = generate_elementwise_kernel_source("data_a[a_index] * data_b[b_index]", dtype, false, true, max_rank);
        auto kernel = get_or_create_kernel(kernel_name, glsl_source, 4);
        u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(calculate_element_count(out_shape)));
        _accelerator.execute(kernel, {a, b, result, meta_tensor}, dispatch_size);
        return result;
    }

    if (a->get_dtype() != b->get_dtype()) throw std::invalid_argument("Tensors must have the same data type");

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

    if (a->get_element_count() == 1) {
        std::vector<u8> buf(a->get_element_size());
        a->download_data(buf.data());
        float scalar = 0.0f;
        if (a->get_dtype() == DataType::F32) std::memcpy(&scalar, buf.data(), sizeof(float));
        auto result = _accelerator.create_tensor(b->get_shape(), b->get_dtype());
        std::string kernel_name = get_kernel_name_for_dtype("div_scalar_left", b->get_dtype());
        std::string glsl_source = generate_elementwise_kernel_source("pc.scalar / data_in[index]", b->get_dtype(), true);
        auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2, sizeof(float));
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(b->get_element_count()));
    auto pc = make_push_constant(buf.data(), b->get_dtype());
    _accelerator.execute(kernel, {b, result}, dispatch_size, 1, 1, pc.first.data());
        return result;
    }
    if (b->get_element_count() == 1) {
        std::vector<u8> buf(b->get_element_size());
        b->download_data(buf.data());
        float scalar = 0.0f;
        if (b->get_dtype() == DataType::F32) std::memcpy(&scalar, buf.data(), sizeof(float));
        auto result = _accelerator.create_tensor(a->get_shape(), a->get_dtype());
        std::string kernel_name = get_kernel_name_for_dtype("div_scalar", a->get_dtype());
        std::string glsl_source = generate_elementwise_kernel_source("data_in[index] / pc.scalar", a->get_dtype(), true);
        auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2, sizeof(float));
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    auto pc = make_push_constant(buf.data(), a->get_dtype());
    _accelerator.execute(kernel, {a, result}, dispatch_size, 1, 1, pc.first.data());
        return result;
    }

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

        auto meta_tensor = _accelerator.create_tensor(meta.data(), {static_cast<u32>(meta.size())}, DataType::U32, false);

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
    validate_tensor_shape_2d(tensor);
    
    auto input_shape = tensor->get_shape();
    
    if (axis >= 2) {
        throw std::invalid_argument("Axis must be 0 or 1 for 2D tensors");
    }
    
    u32 expected_result_size = (axis == 0) ? input_shape[1] : input_shape[0];
    std::vector<u32> result_shape = {expected_result_size};
    
    DataType dtype = tensor->get_dtype();
    auto result = _accelerator.create_tensor(result_shape, dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("sum_axis", dtype);
    std::string glsl_source = generate_sum_axis_kernel_source(dtype);
    
    struct SumAxisPushConstants {
        u32 rows, cols, axis;
    } push_data = {input_shape[0], input_shape[1], axis};

    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2, sizeof(SumAxisPushConstants));
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(expected_result_size);
    
    _accelerator.execute(kernel, {tensor, result}, dispatch_size, 1, 1, &push_data);
    return result;
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

std::string TensorOperations::get_kernel_name_for_dtype(const std::string& base_name, DataType dtype) const {
    return base_name + "_" + std::string(dtype_to_string(dtype));
}

} // namespace QuasarML
#include "TensorOperations.h"
#include "Accelerator.h"
#include "Tensor.h"
#include "Kernel.h"

namespace QuasarML {

std::shared_ptr<Tensor> TensorOperations::add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    validate_tensor_op_compatibility(a, b);
    
    DataType dtype = a->get_dtype();
    auto result = _accelerator.create_tensor(a->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("add", dtype);
    std::string glsl_source = generate_elementwise_kernel_source("data_a[index] + data_b[index]", dtype);
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3);
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    
    _accelerator.execute(kernel, {a, b, result}, dispatch_size);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    validate_tensor_op_compatibility(a, b);
    
    DataType dtype = a->get_dtype();
    auto result = _accelerator.create_tensor(a->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("sub", dtype);
    std::string glsl_source = generate_elementwise_kernel_source("data_a[index] - data_b[index]", dtype);
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3);
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    
    _accelerator.execute(kernel, {a, b, result}, dispatch_size);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    validate_tensor_op_compatibility(a, b);
    
    DataType dtype = a->get_dtype();
    auto result = _accelerator.create_tensor(a->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("mul", dtype);
    std::string glsl_source = generate_elementwise_kernel_source("data_a[index] * data_b[index]", dtype);
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3);
    u32 dispatch_size = _accelerator.calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    
    _accelerator.execute(kernel, {a, b, result}, dispatch_size);
    return result;
}

std::shared_ptr<Tensor> TensorOperations::div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    validate_tensor_op_compatibility(a, b);
    
    DataType dtype = a->get_dtype();
    auto result = _accelerator.create_tensor(a->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("div", dtype);
    std::string glsl_source = generate_elementwise_kernel_source("data_a[index] / data_b[index]", dtype);
    
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
    
    _accelerator.execute(kernel, {tensor, result}, dispatch_size, 1, 1, &scalar);
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
    
    _accelerator.execute(kernel, {tensor, result}, dispatch_size, 1, 1, &scalar);
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
                                                                bool is_scalar) const {
    const char* glsl_type = dtype_to_glsl_type(dtype);
    
    std::string source = "#version 450\n"
                        "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n";
    
    if (is_scalar) {
        source += "layout(push_constant) uniform PushConstants { float scalar; } pc;\n"
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
    }
    
    return source;
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
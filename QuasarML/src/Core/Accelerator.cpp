#include "Accelerator.h"
#include "Kernel.h"
#include "Tensor.h"

namespace QuasarML {

Accelerator::Accelerator(const std::string& name, u32 gpu_idx)
    : _backend(std::make_unique<VulkanBackend>(name, gpu_idx))
{
    if (!_backend) {
        throw std::runtime_error("Failed to create Vulkan backend");
    }
}

Accelerator::~Accelerator() {
    if (_backend) {
        // Wait for all operations to complete before cleanup
        _backend->device_wait_idle();
        
        // Clear all kernels first (they depend on the backend)
        _kernels.clear();
        
        // Cleanup any remaining tensor references
        cleanup_dead_tensor_references();
        
        // Backend will be automatically destroyed by unique_ptr
    }
}

std::shared_ptr<Kernel> Accelerator::create_kernel(const std::string& name,
                                                  const std::string& glsl_source,
                                                  u32 num_tensors,
                                                  u32 push_constant_size) {
    if (!_backend) {
        throw std::runtime_error("Backend not initialized");
    }
    
    // Check if kernel with this name already exists
    if (_kernels.find(name) != _kernels.end()) {
        throw std::runtime_error("Kernel with name '" + name + "' already exists");
    }
    
    // Create the kernel
    auto kernel = std::make_shared<Kernel>(_backend.get(), name, glsl_source, 
                                         num_tensors, push_constant_size);
    
    // Store in our registry
    _kernels[name] = kernel;
    
    return kernel;
}

std::shared_ptr<Kernel> Accelerator::get_kernel(const std::string& name) const {
    auto it = _kernels.find(name);
    if (it != _kernels.end()) {
        return it->second;
    }
    return nullptr;
}

bool Accelerator::remove_kernel(const std::string& name) {
    auto it = _kernels.find(name);
    if (it != _kernels.end()) {
        _kernels.erase(it);
        return true;
    }
    return false;
}

std::vector<std::string> Accelerator::get_kernel_names() const {
    std::vector<std::string> names;
    names.reserve(_kernels.size());
    
    for (const auto& pair : _kernels) {
        names.push_back(pair.first);
    }
    
    return names;
}

std::shared_ptr<Tensor> Accelerator::create_tensor(const std::vector<u32>& shape,
                                                  DataType dtype,
                                                  bool device_only) {
    if (!_backend) {
        throw std::runtime_error("Backend not initialized");
    }
    
    auto tensor = std::make_shared<Tensor>(_backend.get(), shape, dtype, device_only);
    
    // Track the tensor
    _tensors.push_back(tensor);
    _allocated_memory += tensor->get_size_bytes();
    
    // Periodically cleanup dead references
    if (_tensors.size() % 100 == 0) {
        cleanup_dead_tensor_references();
    }
    
    return tensor;
}

std::shared_ptr<Tensor> Accelerator::create_tensor_from_data(const void* data,
                                                           const std::vector<u32>& shape,
                                                           DataType dtype,
                                                           bool device_only) {
    if (!_backend) {
        throw std::runtime_error("Backend not initialized");
    }
    
    if (!data) {
        throw std::invalid_argument("Data pointer cannot be null");
    }
    
    auto tensor = std::make_shared<Tensor>(_backend.get(), shape, dtype, device_only);
    tensor->upload_data(data);
    
    // Track the tensor
    _tensors.push_back(tensor);
    _allocated_memory += tensor->get_size_bytes();
    
    // Periodically cleanup dead references
    if (_tensors.size() % 100 == 0) {
        cleanup_dead_tensor_references();
    }
    
    return tensor;
}

void Accelerator::execute(std::shared_ptr<Kernel> kernel,
                         const std::vector<std::shared_ptr<Tensor>>& tensors,
                         u32 dispatch_x,
                         u32 dispatch_y,
                         u32 dispatch_z,
                         const void* push_data) {
    if (!kernel) {
        throw std::invalid_argument("Kernel cannot be null");
    }
    
    if (!_backend) {
        throw std::runtime_error("Backend not initialized");
    }
    
    validate_tensor_compatibility(tensors, kernel);
    
    // Bind tensors to kernel
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (!tensors[i]) {
            throw std::invalid_argument("Tensor at index " + std::to_string(i) + " is null");
        }
        kernel->bind_tensor(static_cast<u32>(i), tensors[i]);
    }
    
    // Execute the kernel
    kernel->execute(dispatch_x, dispatch_y, dispatch_z, push_data);
}

void Accelerator::begin_recording() {
    if (_recording) {
        throw std::runtime_error("Already recording - call end_recording() first");
    }
    
    if (!_backend) {
        throw std::runtime_error("Backend not initialized");
    }
    
    _backend->begin_compute_recording();
    _recording = true;
}

void Accelerator::record_execution(std::shared_ptr<Kernel> kernel,
                                  const std::vector<std::shared_ptr<Tensor>>& tensors,
                                  u32 dispatch_x,
                                  u32 dispatch_y,
                                  u32 dispatch_z,
                                  const void* push_data) {
    if (!_recording) {
        throw std::runtime_error("Not recording - call begin_recording() first");
    }
    
    if (!kernel) {
        throw std::invalid_argument("Kernel cannot be null");
    }
    
    validate_tensor_compatibility(tensors, kernel);
    
    // Bind tensors to kernel
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (!tensors[i]) {
            throw std::invalid_argument("Tensor at index " + std::to_string(i) + " is null");
        }
        kernel->bind_tensor(static_cast<u32>(i), tensors[i]);
    }
    
    // Record the execution
    kernel->record_execution(dispatch_x, dispatch_y, dispatch_z, push_data);
}

void Accelerator::end_recording() {
    if (!_recording) {
        throw std::runtime_error("Not recording - call begin_recording() first");
    }
    
    if (!_backend) {
        throw std::runtime_error("Backend not initialized");
    }
    
    _backend->execute_recorded_commands();
    _backend->wait_for_compute();
    _recording = false;
}

void Accelerator::synchronize() {
    if (!_backend) {
        throw std::runtime_error("Backend not initialized");
    }
    
    _backend->device_wait_idle();
}

void Accelerator::memory_barrier() {
    if (!_recording) {
        throw std::runtime_error("Memory barrier can only be called during recording");
    }
    
    if (!_backend) {
        throw std::runtime_error("Backend not initialized");
    }
    
    _backend->memory_barrier();
}

// ============================================================================
// IMPLEMENTATION (Add to Accelerator.cpp)
// ============================================================================

void Accelerator::tensor_add(std::shared_ptr<Tensor> a, 
                            std::shared_ptr<Tensor> b, 
                            std::shared_ptr<Tensor> result) {
    validate_tensor_op_compatibility(a, b, result);
    
    const std::string glsl_source = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer InputA {
    float data_a[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer InputB {
    float data_b[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer Output {
    float data_out[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= data_a.length()) return;
    
    data_out[index] = data_a[index] + data_b[index];
}
)";

    auto kernel = get_or_create_kernel("tensor_add", glsl_source, 3);
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    
    execute(kernel, {a, b, result}, dispatch_size);
}

void Accelerator::tensor_sub(std::shared_ptr<Tensor> a, 
                            std::shared_ptr<Tensor> b, 
                            std::shared_ptr<Tensor> result) {
    validate_tensor_op_compatibility(a, b, result);
    
    const std::string glsl_source = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer InputA {
    float data_a[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer InputB {
    float data_b[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer Output {
    float data_out[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= data_a.length()) return;
    
    data_out[index] = data_a[index] - data_b[index];
}
)";

    auto kernel = get_or_create_kernel("tensor_sub", glsl_source, 3);
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    
    execute(kernel, {a, b, result}, dispatch_size);
}

void Accelerator::tensor_mul(std::shared_ptr<Tensor> a, 
                            std::shared_ptr<Tensor> b, 
                            std::shared_ptr<Tensor> result) {
    validate_tensor_op_compatibility(a, b, result);
    
    const std::string glsl_source = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer InputA {
    float data_a[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer InputB {
    float data_b[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer Output {
    float data_out[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= data_a.length()) return;
    
    data_out[index] = data_a[index] * data_b[index];
}
)";

    auto kernel = get_or_create_kernel("tensor_mul", glsl_source, 3);
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    
    execute(kernel, {a, b, result}, dispatch_size);
}

void Accelerator::tensor_div(std::shared_ptr<Tensor> a, 
                            std::shared_ptr<Tensor> b, 
                            std::shared_ptr<Tensor> result) {
    validate_tensor_op_compatibility(a, b, result);
    
    const std::string glsl_source = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer InputA {
    float data_a[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer InputB {
    float data_b[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer Output {
    float data_out[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= data_a.length()) return;
    
    data_out[index] = data_a[index] / data_b[index];
}
)";

    auto kernel = get_or_create_kernel("tensor_div", glsl_source, 3);
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    
    execute(kernel, {a, b, result}, dispatch_size);
}

void Accelerator::tensor_add_scalar(std::shared_ptr<Tensor> tensor, 
                                   float scalar, 
                                   std::shared_ptr<Tensor> result) {
    if (!tensor->is_shape_compatible(*result)) {
        throw std::invalid_argument("Input and output tensors must have same shape");
    }
    
    const std::string glsl_source = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    float scalar;
} pc;

layout(set = 0, binding = 0, std430) restrict readonly buffer Input {
    float data_in[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {
    float data_out[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= data_in.length()) return;
    
    data_out[index] = data_in[index] + pc.scalar;
}
)";

    auto kernel = get_or_create_kernel("tensor_add_scalar", glsl_source, 2, sizeof(float));
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    
    execute(kernel, {tensor, result}, dispatch_size, 1, 1, &scalar);
}

void Accelerator::tensor_mul_scalar(std::shared_ptr<Tensor> tensor, 
                                   float scalar, 
                                   std::shared_ptr<Tensor> result) {
    if (!tensor->is_shape_compatible(*result)) {
        throw std::invalid_argument("Input and output tensors must have same shape");
    }
    
    const std::string glsl_source = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    float scalar;
} pc;

layout(set = 0, binding = 0, std430) restrict readonly buffer Input {
    float data_in[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {
    float data_out[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= data_in.length()) return;
    
    data_out[index] = data_in[index] * pc.scalar;
}
)";

    auto kernel = get_or_create_kernel("tensor_mul_scalar", glsl_source, 2, sizeof(float));
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    
    execute(kernel, {tensor, result}, dispatch_size, 1, 1, &scalar);
}

void Accelerator::tensor_relu(std::shared_ptr<Tensor> tensor, 
                             std::shared_ptr<Tensor> result) {
    if (!tensor->is_shape_compatible(*result)) {
        throw std::invalid_argument("Input and output tensors must have same shape");
    }
    
    const std::string glsl_source = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer Input {
    float data_in[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {
    float data_out[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= data_in.length()) return;
    
    data_out[index] = max(0.0, data_in[index]);
}
)";

    auto kernel = get_or_create_kernel("tensor_relu", glsl_source, 2);
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    
    execute(kernel, {tensor, result}, dispatch_size);
}

void Accelerator::tensor_matmul(std::shared_ptr<Tensor> a, 
                               std::shared_ptr<Tensor> b, 
                               std::shared_ptr<Tensor> result) {
    validate_tensor_shape_2d(a);
    validate_tensor_shape_2d(b);
    validate_tensor_shape_2d(result);
    
    auto a_shape = a->get_shape();
    auto b_shape = b->get_shape();
    auto result_shape = result->get_shape();
    
    if (a_shape[1] != b_shape[0]) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    if (result_shape[0] != a_shape[0] || result_shape[1] != b_shape[1]) {
        throw std::invalid_argument("Result matrix has incorrect dimensions");
    }
    
    const std::string glsl_source = R"(
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint M; // rows of A and result
    uint N; // cols of B and result  
    uint K; // cols of A and rows of B
} pc;

layout(set = 0, binding = 0, std430) restrict readonly buffer MatrixA {
    float data_a[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer MatrixB {
    float data_b[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer Result {
    float data_result[];
};

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    
    if (row >= pc.M || col >= pc.N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < pc.K; ++k) {
        sum += data_a[row * pc.K + k] * data_b[k * pc.N + col];
    }
    
    data_result[row * pc.N + col] = sum;
}
)";

    struct MatMulPushConstants {
        u32 M, N, K;
    } push_data = {
        a_shape[0],  // M
        b_shape[1],  // N
        a_shape[1]   // K
    };

    auto kernel = get_or_create_kernel("tensor_matmul", glsl_source, 3, sizeof(MatMulPushConstants));
    
    u32 dispatch_x = (a_shape[0] + 15) / 16;
    u32 dispatch_y = (b_shape[1] + 15) / 16;
    
    execute(kernel, {a, b, result}, dispatch_x, dispatch_y, 1, &push_data);
}

void Accelerator::tensor_transpose(std::shared_ptr<Tensor> tensor, 
                                  std::shared_ptr<Tensor> result) {
    validate_tensor_shape_2d(tensor);
    validate_tensor_shape_2d(result);
    
    auto input_shape = tensor->get_shape();
    auto result_shape = result->get_shape();
    
    if (input_shape[0] != result_shape[1] || input_shape[1] != result_shape[0]) {
        throw std::invalid_argument("Result tensor must have transposed dimensions");
    }
    
    const std::string glsl_source = R"(
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint rows;
    uint cols;
} pc;

layout(set = 0, binding = 0, std430) restrict readonly buffer Input {
    float data_in[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {
    float data_out[];
};

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    
    if (row >= pc.rows || col >= pc.cols) return;
    
    data_out[col * pc.rows + row] = data_in[row * pc.cols + col];
}
)";

    struct TransposePushConstants {
        u32 rows, cols;
    } push_data = {
        input_shape[0],
        input_shape[1]
    };

    auto kernel = get_or_create_kernel("tensor_transpose", glsl_source, 2, sizeof(TransposePushConstants));
    
    u32 dispatch_x = (input_shape[0] + 15) / 16;
    u32 dispatch_y = (input_shape[1] + 15) / 16;
    
    execute(kernel, {tensor, result}, dispatch_x, dispatch_y, 1, &push_data);
}

void Accelerator::tensor_sum_axis(std::shared_ptr<Tensor> tensor, 
                                 std::shared_ptr<Tensor> result, 
                                 u32 axis) {
    validate_tensor_shape_2d(tensor);
    
    auto input_shape = tensor->get_shape();
    auto result_shape = result->get_shape();
    
    if (axis >= 2) {
        throw std::invalid_argument("Axis must be 0 or 1 for 2D tensors");
    }
    
    u32 expected_result_size = (axis == 0) ? input_shape[1] : input_shape[0];
    if (result_shape.size() != 1 || result_shape[0] != expected_result_size) {
        throw std::invalid_argument("Result tensor has incorrect shape for reduction");
    }
    
    const std::string glsl_source = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint rows;
    uint cols;
    uint axis; // 0 = sum rows (result is cols), 1 = sum cols (result is rows)
} pc;

layout(set = 0, binding = 0, std430) restrict readonly buffer Input {
    float data_in[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {
    float data_out[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    
    if (pc.axis == 0) {
        // Sum along rows (axis=0), output has shape [cols]
        if (index >= pc.cols) return;
        
        float sum = 0.0;
        for (uint row = 0; row < pc.rows; ++row) {
            sum += data_in[row * pc.cols + index];
        }
        data_out[index] = sum;
    } else {
        // Sum along columns (axis=1), output has shape [rows]
        if (index >= pc.rows) return;
        
        float sum = 0.0;
        for (uint col = 0; col < pc.cols; ++col) {
            sum += data_in[index * pc.cols + col];
        }
        data_out[index] = sum;
    }
}
)";

    struct SumAxisPushConstants {
        u32 rows, cols, axis;
    } push_data = {
        input_shape[0],
        input_shape[1],
        axis
    };

    auto kernel = get_or_create_kernel("tensor_sum_axis", glsl_source, 2, sizeof(SumAxisPushConstants));
    u32 dispatch_size = calculate_optimal_dispatch_1d(expected_result_size);
    
    execute(kernel, {tensor, result}, dispatch_size, 1, 1, &push_data);
}

// Helper method implementations
void Accelerator::validate_tensor_op_compatibility(std::shared_ptr<Tensor> a, 
                                                   std::shared_ptr<Tensor> b, 
                                                   std::shared_ptr<Tensor> result) const {
    if (!a->is_valid() || !b->is_valid() || !result->is_valid()) {
        throw std::invalid_argument("All tensors must be valid");
    }
    
    if (!a->is_shape_compatible(*b) || !a->is_shape_compatible(*result)) {
        throw std::invalid_argument("All tensors must have compatible shapes");
    }
    
    if (a->get_dtype() != DataType::F32 || b->get_dtype() != DataType::F32 || 
        result->get_dtype() != DataType::F32) {
        throw std::invalid_argument("Currently only F32 tensors are supported");
    }
}

void Accelerator::validate_tensor_shape_2d(std::shared_ptr<Tensor> tensor) const {
    if (!tensor->is_valid()) {
        throw std::invalid_argument("Tensor must be valid");
    }
    
    if (tensor->get_rank() != 2) {
        throw std::invalid_argument("Tensor must be 2D");
    }
    
    if (tensor->get_dtype() != DataType::F32) {
        throw std::invalid_argument("Currently only F32 tensors are supported");
    }
}

std::shared_ptr<Kernel> Accelerator::get_or_create_kernel(const std::string& name, 
                                                            const std::string& glsl_source, 
                                                            u32 num_tensors, u32 push_constant_size) {
    auto existing_kernel = get_kernel(name);
    if (existing_kernel) {
        return existing_kernel;
    }
    
    return create_kernel(name, glsl_source, num_tensors, push_constant_size);
}

VulkanBackend::ComputeLimits Accelerator::get_device_limits() const {
    if (!_backend) {
        throw std::runtime_error("Backend not initialized");
    }
    
    return _backend->get_compute_limits();
}

u32 Accelerator::calculate_optimal_dispatch_1d(u32 total_elements, u32 local_size) const {
    if (!_backend) {
        throw std::runtime_error("Backend not initialized");
    }
    
    return _backend->calculate_dispatch_1d(total_elements, local_size);
}

std::pair<u64, u64> Accelerator::get_memory_usage() const {
    
    // For now, return allocated memory and a reasonable estimate of total device memory
    // In a real implementation, you would query the actual device memory
    auto limits = get_device_limits();
    u64 estimated_total = 4ULL * 1024 * 1024 * 1024; // 4GB estimate
    
    return {_allocated_memory, estimated_total};
}

bool Accelerator::is_valid() const {
    return _backend != nullptr;
}

void Accelerator::cleanup_dead_tensor_references() {
    auto old_size = _tensors.size();
    
    _tensors.erase(
        std::remove_if(_tensors.begin(), _tensors.end(),
                      [this](const std::weak_ptr<Tensor>& weak_tensor) {
                          if (weak_tensor.expired()) {
                              // Note: We can't accurately track deallocated memory without
                              // more complex bookkeeping, but this is a reasonable approximation
                              return true;
                          }
                          return false;
                      }),
        _tensors.end());
    
    // Rough estimate of freed memory (this is not perfectly accurate)
    if (old_size > _tensors.size()) {
        // Reset memory counter periodically to prevent drift
        _allocated_memory = 0;
        for (const auto& weak_tensor : _tensors) {
            if (auto tensor = weak_tensor.lock()) {
                _allocated_memory += tensor->get_size_bytes();
            }
        }
    }
}

void Accelerator::validate_tensor_compatibility(const std::vector<std::shared_ptr<Tensor>>& tensors,
                                               std::shared_ptr<Kernel> kernel) const {
    if (tensors.size() != kernel->get_expected_tensor_count()) {
        throw std::invalid_argument(
            "Tensor count mismatch: kernel expects " + 
            std::to_string(kernel->get_expected_tensor_count()) + 
            " tensors but got " + std::to_string(tensors.size())
        );
    }
    
    for (const auto& tensor : tensors) {
        if (!tensor) {
            throw std::invalid_argument("Tensor cannot be null");
        }
        if (!tensor->is_valid()) {
            throw std::invalid_argument("Tensor is not valid");
        }
    }
}

const char* dtype_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return "float32";
        case DataType::F16: return "float16";
        case DataType::I32:   return "int32";
        case DataType::I16:   return "int16";
        case DataType::I8:    return "int8";
        case DataType::U32:  return "uint32";
        case DataType::U16:  return "uint16";
        case DataType::U8:   return "uint8";
        default:                return "unknown";
    }
}

} // namespace QuasarML
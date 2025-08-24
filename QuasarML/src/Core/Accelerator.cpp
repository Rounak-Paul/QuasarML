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

std::shared_ptr<Tensor> Accelerator::create_tensor(const void* data,
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

std::shared_ptr<Tensor> Accelerator::add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    validate_tensor_op_compatibility(a, b);
    
    DataType dtype = a->get_dtype();
    auto result = create_tensor(a->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("add", dtype);
    std::string glsl_source = generate_elementwise_kernel_source("data_a[index] + data_b[index]", dtype);
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3);
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    
    execute(kernel, {a, b, result}, dispatch_size);
    return result;
}

std::shared_ptr<Tensor> Accelerator::sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    validate_tensor_op_compatibility(a, b);
    
    DataType dtype = a->get_dtype();
    auto result = create_tensor(a->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("sub", dtype);
    std::string glsl_source = generate_elementwise_kernel_source("data_a[index] - data_b[index]", dtype);
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3);
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    
    execute(kernel, {a, b, result}, dispatch_size);
    return result;
}

std::shared_ptr<Tensor> Accelerator::mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    validate_tensor_op_compatibility(a, b);
    
    DataType dtype = a->get_dtype();
    auto result = create_tensor(a->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("mul", dtype);
    std::string glsl_source = generate_elementwise_kernel_source("data_a[index] * data_b[index]", dtype);
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3);
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    
    execute(kernel, {a, b, result}, dispatch_size);
    return result;
}

std::shared_ptr<Tensor> Accelerator::div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    validate_tensor_op_compatibility(a, b);
    
    DataType dtype = a->get_dtype();
    auto result = create_tensor(a->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("div", dtype);
    std::string glsl_source = generate_elementwise_kernel_source("data_a[index] / data_b[index]", dtype);
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3);
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(a->get_element_count()));
    
    execute(kernel, {a, b, result}, dispatch_size);
    return result;
}

std::shared_ptr<Tensor> Accelerator::add_scalar(std::shared_ptr<Tensor> tensor, float scalar) {
    if (!tensor->is_valid()) {
        throw std::invalid_argument("Tensor must be valid");
    }
    
    DataType dtype = tensor->get_dtype();
    auto result = create_tensor(tensor->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("add_scalar", dtype);
    std::string glsl_source = generate_elementwise_kernel_source("data_in[index] + pc.scalar", dtype, true);
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2, sizeof(float));
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    
    execute(kernel, {tensor, result}, dispatch_size, 1, 1, &scalar);
    return result;
}

std::shared_ptr<Tensor> Accelerator::mul_scalar(std::shared_ptr<Tensor> tensor, float scalar) {
    if (!tensor->is_valid()) {
        throw std::invalid_argument("Tensor must be valid");
    }
    
    DataType dtype = tensor->get_dtype();
    auto result = create_tensor(tensor->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("mul_scalar", dtype);
    std::string glsl_source = generate_elementwise_kernel_source("data_in[index] * pc.scalar", dtype, true);
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2, sizeof(float));
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    
    execute(kernel, {tensor, result}, dispatch_size, 1, 1, &scalar);
    return result;
}

std::shared_ptr<Tensor> Accelerator::relu(std::shared_ptr<Tensor> tensor) {
    if (!tensor->is_valid()) {
        throw std::invalid_argument("Tensor must be valid");
    }
    
    DataType dtype = tensor->get_dtype();
    auto result = create_tensor(tensor->get_shape(), dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("relu", dtype);
    std::string glsl_source = generate_relu_kernel_source(dtype);
    
    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2);
    u32 dispatch_size = calculate_optimal_dispatch_1d(static_cast<u32>(tensor->get_element_count()));
    
    execute(kernel, {tensor, result}, dispatch_size);
    return result;
}

// Updated matmul, transpose, and sum_axis methods
std::shared_ptr<Tensor> Accelerator::matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
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
    auto result = create_tensor(result_shape, dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("matmul", dtype);
    std::string glsl_source = generate_matmul_kernel_source(dtype);
    
    struct MatMulPushConstants {
        u32 M, N, K;
    } push_data = {a_shape[0], b_shape[1], a_shape[1]};

    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 3, sizeof(MatMulPushConstants));
    
    u32 dispatch_x = (a_shape[0] + 15) / 16;
    u32 dispatch_y = (b_shape[1] + 15) / 16;
    
    execute(kernel, {a, b, result}, dispatch_x, dispatch_y, 1, &push_data);
    return result;
}

std::shared_ptr<Tensor> Accelerator::transpose(std::shared_ptr<Tensor> tensor) {
    validate_tensor_shape_2d(tensor);
    
    auto input_shape = tensor->get_shape();
    std::vector<u32> result_shape = {input_shape[1], input_shape[0]};
    
    DataType dtype = tensor->get_dtype();
    auto result = create_tensor(result_shape, dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("transpose", dtype);
    std::string glsl_source = generate_transpose_kernel_source(dtype);
    
    struct TransposePushConstants {
        u32 rows, cols;
    } push_data = {input_shape[0], input_shape[1]};

    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2, sizeof(TransposePushConstants));
    
    u32 dispatch_x = (input_shape[0] + 15) / 16;
    u32 dispatch_y = (input_shape[1] + 15) / 16;
    
    execute(kernel, {tensor, result}, dispatch_x, dispatch_y, 1, &push_data);
    return result;
}

std::shared_ptr<Tensor> Accelerator::sum_axis(std::shared_ptr<Tensor> tensor, u32 axis) {
    validate_tensor_shape_2d(tensor);
    
    auto input_shape = tensor->get_shape();
    
    if (axis >= 2) {
        throw std::invalid_argument("Axis must be 0 or 1 for 2D tensors");
    }
    
    u32 expected_result_size = (axis == 0) ? input_shape[1] : input_shape[0];
    std::vector<u32> result_shape = {expected_result_size};
    
    DataType dtype = tensor->get_dtype();
    auto result = create_tensor(result_shape, dtype);
    
    std::string kernel_name = get_kernel_name_for_dtype("sum_axis", dtype);
    std::string glsl_source = generate_sum_axis_kernel_source(dtype);
    
    struct SumAxisPushConstants {
        u32 rows, cols, axis;
    } push_data = {input_shape[0], input_shape[1], axis};

    auto kernel = get_or_create_kernel(kernel_name, glsl_source, 2, sizeof(SumAxisPushConstants));
    u32 dispatch_size = calculate_optimal_dispatch_1d(expected_result_size);
    
    execute(kernel, {tensor, result}, dispatch_size, 1, 1, &push_data);
    return result;
}

// Helper method implementations
void Accelerator::validate_tensor_op_compatibility(std::shared_ptr<Tensor> a, 
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

std::string Accelerator::generate_elementwise_kernel_source(const std::string& operation, 
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

std::string Accelerator::generate_relu_kernel_source(DataType dtype) const {
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

std::string Accelerator::generate_matmul_kernel_source(DataType dtype) const {
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

std::string Accelerator::generate_transpose_kernel_source(DataType dtype) const {
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

std::string Accelerator::generate_sum_axis_kernel_source(DataType dtype) const {
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

std::string Accelerator::get_kernel_name_for_dtype(const std::string& base_name, DataType dtype) const {
    return base_name + "_" + std::string(dtype_to_string(dtype));
}

// Update the validate_tensor_shape_2d method to remove F32-only restriction
void Accelerator::validate_tensor_shape_2d(std::shared_ptr<Tensor> tensor) const {
    if (!tensor->is_valid()) {
        throw std::invalid_argument("Tensor must be valid");
    }
    
    if (tensor->get_rank() != 2) {
        throw std::invalid_argument("Tensor must be 2D");
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
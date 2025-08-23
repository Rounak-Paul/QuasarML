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
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

bool Accelerator::use_gpu() const {
    // If user forced CPU, return false. If forced GPU, return true. Otherwise Auto - return true when backend exists and is_valid
    if (_device_mode == DeviceMode::CPU) return false;
    if (_device_mode == DeviceMode::GPU) return true;
    // Auto: check backend existence
    return _backend != nullptr && is_valid();
}

Accelerator::~Accelerator() {
    if (_backend) {
        try {
            if (_backend->is_valid()) {
                _backend->device_wait_idle();
            }
            _kernels.clear();
            cleanup_dead_tensor_references();
        } catch (...) {
        }
    }
}

std::shared_ptr<Kernel> Accelerator::create_kernel(const std::string& name,
                                                  const std::string& glsl_source,
                                                  u32 num_tensors,
                                                  u32 push_constant_size) {
    if (!_backend) {
        throw std::runtime_error("Backend not initialized");
    }
    
    std::lock_guard<std::mutex> lock(_kernel_mutex);
    
    auto existing = _kernels.find(name);
    if (existing != _kernels.end()) {
        return existing->second;
    }
    
    auto kernel = std::make_shared<Kernel>(_backend.get(), name, glsl_source, 
                                         num_tensors, push_constant_size);
    
    _kernels[name] = kernel;
    return kernel;
}

std::shared_ptr<Kernel> Accelerator::get_kernel(const std::string& name) const {
    std::lock_guard<std::mutex> lock(_kernel_mutex);
    auto it = _kernels.find(name);
    if (it != _kernels.end()) {
        return it->second;
    }
    return nullptr;
}

bool Accelerator::remove_kernel(const std::string& name) {
    std::lock_guard<std::mutex> lock(_kernel_mutex);
    auto it = _kernels.find(name);
    if (it != _kernels.end()) {
        _kernels.erase(it);
        return true;
    }
    return false;
}

std::vector<std::string> Accelerator::get_kernel_names() const {
    std::lock_guard<std::mutex> lock(_kernel_mutex);
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
    
    if (_tensor_pooling_enabled && device_only && use_gpu()) {
        auto pooled = get_pooled_tensor(shape, dtype);
        if (pooled) {
            return pooled;
        }
    }
    
    bool final_device_only = device_only;
    if (!use_gpu()) {
        final_device_only = false;
    }

    auto tensor = std::make_shared<Tensor>(this, _backend.get(), shape, dtype, final_device_only);
    
    {
        std::lock_guard<std::mutex> lock(_tensor_mutex);
        _tensors.push_back(tensor);
        _allocated_memory += tensor->get_size_bytes();
        
        if (_tensors.size() % 100 == 0) {
            cleanup_dead_tensor_references();
        }
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
    
    bool final_device_only = device_only;
    if (!use_gpu()) final_device_only = false;

    auto tensor = std::make_shared<Tensor>(this, _backend.get(), shape, dtype, final_device_only);
    tensor->upload_data(data);
    
    _tensors.push_back(tensor);
    _allocated_memory += tensor->get_size_bytes();
    
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
    
    if (_auto_batching_enabled && !_recording) {
        DeferredOperation op;
        op.kernel = kernel;
        op.tensors = tensors;
        op.dispatch_x = dispatch_x;
        op.dispatch_y = dispatch_y;
        op.dispatch_z = dispatch_z;
        
        if (push_data && kernel->get_push_constant_size() > 0) {
            op.push_data.resize(kernel->get_push_constant_size());
            std::memcpy(op.push_data.data(), push_data, kernel->get_push_constant_size());
        }
        
        _deferred_ops.push_back(std::move(op));
        
        if (should_flush_batch()) {
            submit_batched_operations();
        }
        return;
    }
    
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (!tensors[i]) {
            throw std::invalid_argument("Tensor at index " + std::to_string(i) + " is null");
        }
        kernel->bind_tensor(static_cast<u32>(i), tensors[i]);
    }
    
    kernel->execute(dispatch_x, dispatch_y, dispatch_z, push_data);
}void Accelerator::begin_recording() {
    std::lock_guard<std::mutex> lock(_command_mutex);
    
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
    
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (!tensors[i]) {
            throw std::invalid_argument("Tensor at index " + std::to_string(i) + " is null");
        }
        kernel->bind_tensor(static_cast<u32>(i), tensors[i]);
    }
    
    kernel->record_execution(dispatch_x, dispatch_y, dispatch_z, push_data);
}

void Accelerator::end_recording() {
    std::lock_guard<std::mutex> lock(_command_mutex);
    
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
    std::lock_guard<std::mutex> lock(_command_mutex);
    
    if (_auto_batching_enabled && !_deferred_ops.empty()) {
        submit_batched_operations();
    }
    
    if (!_backend) {
        throw std::runtime_error("Backend not initialized");
    }
    
    _backend->device_wait_idle();
}

void Accelerator::memory_barrier() {
    std::lock_guard<std::mutex> lock(_command_mutex);
    
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
    auto limits = get_device_limits();
    u64 estimated_total = 4ULL * 1024 * 1024 * 1024; // 4GB estimate
    
    return {_allocated_memory, estimated_total};
}

bool Accelerator::is_valid() const {
    return _backend != nullptr && _backend->is_valid();
}

void Accelerator::notify_cpu_fallback() {
    _cpu_fallback_count.fetch_add(1u, std::memory_order_relaxed);
}

u32 Accelerator::get_cpu_fallback_count() const {
    return _cpu_fallback_count.load(std::memory_order_relaxed);
}

void Accelerator::reset_cpu_fallback_count() {
    _cpu_fallback_count.store(0u, std::memory_order_relaxed);
}
void Accelerator::cleanup_dead_tensor_references() {
    auto old_size = _tensors.size();
    
    _tensors.erase(
        std::remove_if(_tensors.begin(), _tensors.end(),
                      [this](const std::weak_ptr<Tensor>& weak_tensor) {
                          if (weak_tensor.expired()) {
                              return true;
                          }
                          return false;
                      }),
        _tensors.end());
    
    if (old_size > _tensors.size()) {
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

void Accelerator::enable_auto_batching(bool enable) {
    std::lock_guard<std::mutex> lock(_command_mutex);
    
    if (!enable && !_deferred_ops.empty()) {
        submit_batched_operations();
    }
    
    _auto_batching_enabled = enable;
}

void Accelerator::flush_pipeline() {
    std::lock_guard<std::mutex> lock(_command_mutex);
    
    if (!_deferred_ops.empty()) {
        submit_batched_operations();
    }
}

void Accelerator::submit_batched_operations() {
    if (_deferred_ops.empty() || !_backend) {
        return;
    }
    
    _backend->begin_compute_recording();
    
    for (auto& op : _deferred_ops) {
        for (size_t i = 0; i < op.tensors.size(); ++i) {
            op.kernel->bind_tensor(static_cast<u32>(i), op.tensors[i]);
        }
        
        const void* push_ptr = op.push_data.empty() ? nullptr : op.push_data.data();
        op.kernel->record_execution(op.dispatch_x, op.dispatch_y, op.dispatch_z, push_ptr);
    }
    
    _backend->execute_recorded_commands();
    _backend->wait_for_compute();
    _deferred_ops.clear();
}

bool Accelerator::should_flush_batch() const {
    return _deferred_ops.size() >= MAX_BATCH_SIZE;
}

void Accelerator::enable_tensor_pooling(bool enable) {
    std::lock_guard<std::mutex> lock(_pool_mutex);
    _tensor_pooling_enabled = enable;
    if (!enable) {
        _tensor_pool.clear();
    }
}

void Accelerator::clear_tensor_pool() {
    std::lock_guard<std::mutex> lock(_pool_mutex);
    _tensor_pool.clear();
}

u64 Accelerator::calculate_tensor_size_key(const std::vector<u32>& shape, DataType dtype) const {
    u64 total_elements = 1;
    for (u32 dim : shape) {
        total_elements *= dim;
    }
    return (total_elements << 8) | static_cast<u64>(dtype);
}

std::shared_ptr<Tensor> Accelerator::get_pooled_tensor(const std::vector<u32>& shape, DataType dtype) {
    std::lock_guard<std::mutex> lock(_pool_mutex);
    
    u64 size_key = calculate_tensor_size_key(shape, dtype);
    auto it = _tensor_pool.find(size_key);
    
    if (it != _tensor_pool.end() && !it->second.empty()) {
        auto entry = it->second.back();
        it->second.pop_back();
        
        if (it->second.empty()) {
            _tensor_pool.erase(it);
        }
        
        return entry.tensor;
    }
    
    _current_frame++;
    if (_current_frame % POOL_CLEANUP_INTERVAL == 0) {
        cleanup_tensor_pool();
    }
    
    return nullptr;
}

void Accelerator::return_tensor_to_pool(std::shared_ptr<Tensor> tensor) {
    if (!_tensor_pooling_enabled || !tensor) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(_pool_mutex);
    
    u64 size_key = calculate_tensor_size_key(tensor->get_shape(), tensor->get_dtype());
    TensorPoolEntry entry{tensor, _current_frame};
    _tensor_pool[size_key].push_back(entry);
}

void Accelerator::cleanup_tensor_pool() {
    const u64 max_age = 50;
    
    for (auto it = _tensor_pool.begin(); it != _tensor_pool.end();) {
        auto& entries = it->second;
        entries.erase(
            std::remove_if(entries.begin(), entries.end(),
                [this, max_age](const TensorPoolEntry& entry) {
                    return (_current_frame - entry.last_used_frame) > max_age;
                }),
            entries.end()
        );
        
        if (entries.empty()) {
            it = _tensor_pool.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace QuasarML
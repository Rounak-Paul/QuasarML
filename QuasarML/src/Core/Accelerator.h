#pragma once

#include <qspch.h>
#include "DataTypes.h"
#include "TensorOperations.h"
#include <VulkanBackend/VulkanBackend.h>
#include <memory>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <string>

namespace QuasarML {

class Kernel;
class Tensor;

class Accelerator {
public:
    explicit Accelerator(const std::string& name = "QuasarML", u32 gpu_idx = 0);
    ~Accelerator();

    Accelerator(const Accelerator&) = delete;
    Accelerator& operator=(const Accelerator&) = delete;
    Accelerator(Accelerator&&) = delete;
    Accelerator& operator=(Accelerator&&) = delete;

    std::shared_ptr<Kernel> create_kernel(const std::string& name,
                                          const std::string& glsl_source,
                                          u32 num_tensors,
                                          u32 push_constant_size = 0);
    
    std::shared_ptr<Kernel> get_kernel(const std::string& name) const;
    bool remove_kernel(const std::string& name);
    std::vector<std::string> get_kernel_names() const;

    std::shared_ptr<Tensor> create_tensor(const std::vector<u32>& shape,
                                          DataType dtype = DataType::F32,
                                          bool host_visible = false);
    
    std::shared_ptr<Tensor> create_tensor(const void* data,
                                          const std::vector<u32>& shape,
                                          DataType dtype = DataType::F32,
                                          bool host_visible = false);

    void execute(std::shared_ptr<Kernel> kernel,
                 const std::vector<std::shared_ptr<Tensor>>& tensors,
                 u32 dispatch_x,
                 u32 dispatch_y = 1,
                 u32 dispatch_z = 1,
                 const void* push_data = nullptr);
    
    void begin_recording();
    
    void record_execution(std::shared_ptr<Kernel> kernel,
                          const std::vector<std::shared_ptr<Tensor>>& tensors,
                          u32 dispatch_x,
                          u32 dispatch_y = 1,
                          u32 dispatch_z = 1,
                          const void* push_data = nullptr);
    
    void end_recording();
    void synchronize();
    void memory_barrier();

    TensorOperations& ops() { return _tensor_ops; }
    const TensorOperations& ops() const { return _tensor_ops; }

    void enable_auto_batching(bool enable = true);
    bool is_auto_batching_enabled() const { return _auto_batching_enabled; }
    void flush_pipeline();
    
    void enable_tensor_pooling(bool enable = true);
    bool is_tensor_pooling_enabled() const { return _tensor_pooling_enabled; }
    void clear_tensor_pool();

    VulkanBackend::ComputeLimits get_device_limits() const;
    u32 calculate_optimal_dispatch_1d(u32 total_elements, u32 local_size = 256) const;
    std::pair<u64, u64> get_memory_usage() const;
    bool is_valid() const;
    u32 get_gpu_index() const { return _gpu_idx; }
    VulkanBackend* get_backend() { return _backend.get(); }
    const VulkanBackend* get_backend() const { return _backend.get(); }

private:
    struct DeferredOperation {
        std::shared_ptr<Kernel> kernel;
        std::vector<std::shared_ptr<Tensor>> tensors;
        u32 dispatch_x;
        u32 dispatch_y;
        u32 dispatch_z;
        std::vector<u8> push_data;
    };
    
    struct TensorPoolEntry {
        std::shared_ptr<Tensor> tensor;
        u64 last_used_frame;
    };
    
    std::unique_ptr<VulkanBackend> _backend;
    std::unordered_map<std::string, std::shared_ptr<Kernel>> _kernels;
    std::vector<std::weak_ptr<Tensor>> _tensors;
    bool _recording = false;
    mutable u64 _allocated_memory = 0;
    u32 _gpu_idx = 0;
    
    TensorOperations _tensor_ops{*this};
    
    bool _auto_batching_enabled = true;
    std::vector<DeferredOperation> _deferred_ops;
    static constexpr size_t MAX_BATCH_SIZE = 32;
    
    std::unordered_map<u64, std::vector<TensorPoolEntry>> _tensor_pool;
    u64 _current_frame = 0;
    bool _tensor_pooling_enabled = true;
    static constexpr u64 POOL_CLEANUP_INTERVAL = 100;
    
    mutable std::mutex _kernel_mutex;
    mutable std::mutex _tensor_mutex;
    mutable std::mutex _command_mutex;
    mutable std::mutex _pool_mutex;
    
    void cleanup_dead_tensor_references();
    void validate_tensor_compatibility(const std::vector<std::shared_ptr<Tensor>>& tensors, 
                                       std::shared_ptr<Kernel> kernel) const;
    void submit_batched_operations();
    bool should_flush_batch() const;
    
    std::shared_ptr<Tensor> get_pooled_tensor(const std::vector<u32>& shape, DataType dtype);
    void return_tensor_to_pool(std::shared_ptr<Tensor> tensor);
    void cleanup_tensor_pool();
    u64 calculate_tensor_size_key(const std::vector<u32>& shape, DataType dtype) const;
};

}
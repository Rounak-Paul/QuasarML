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

// Forward declarations
class Kernel;
class Tensor;

class Accelerator {
public:
    explicit Accelerator(const std::string& name = "QuasarML", u32 gpu_idx = 0);
    ~Accelerator();

    enum class DeviceMode {
        Auto, // use GPU if available, otherwise CPU
        GPU,  // force GPU
        CPU   // force CPU (host-visible buffers and CPU implementations)
    };

    // Non-copyable and non-movable due to TensorOperations reference and atomic members
    Accelerator(const Accelerator&) = delete;
    Accelerator& operator=(const Accelerator&) = delete;
    Accelerator(Accelerator&&) = delete;
    Accelerator& operator=(Accelerator&&) = delete;

    // ============================================================================
    // KERNEL MANAGEMENT
    // ============================================================================
    
    std::shared_ptr<Kernel> create_kernel(const std::string& name,
                                        const std::string& glsl_source,
                                        u32 num_tensors,
                                        u32 push_constant_size = 0);
    
    std::shared_ptr<Kernel> get_kernel(const std::string& name) const;
    bool remove_kernel(const std::string& name);
    std::vector<std::string> get_kernel_names() const;

    // ============================================================================
    // TENSOR MANAGEMENT
    // ============================================================================
    
    std::shared_ptr<Tensor> create_tensor(const std::vector<u32>& shape,
                                        DataType dtype = DataType::F32,
                                        bool device_only = true);
    
    std::shared_ptr<Tensor> create_tensor(const void* data,
                                            const std::vector<u32>& shape,
                                            DataType dtype = DataType::F32,
                                            bool device_only = true);

    // ============================================================================
    // EXECUTION MANAGEMENT
    // ============================================================================
    
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

    // ============================================================================
    // TENSOR OPERATIONS ACCESS
    // ============================================================================

    TensorOperations& ops() { return _tensor_ops; }
    const TensorOperations& ops() const { return _tensor_ops; }

    // Device mode control
    void set_device_mode(DeviceMode mode) { _device_mode = mode; }
    DeviceMode get_device_mode() const { return _device_mode; }
    bool use_gpu() const;
    
    // Pipelining control
    void enable_auto_batching(bool enable = true);
    bool is_auto_batching_enabled() const { return _auto_batching_enabled; }
    void flush_pipeline();
    
    // Tensor pooling control
    void enable_tensor_pooling(bool enable = true);
    bool is_tensor_pooling_enabled() const { return _tensor_pooling_enabled; }
    void clear_tensor_pool(); // true when operations should run on GPU

    // CPU fallback instrumentation (tests can query how often CPU implementations ran)
    void notify_cpu_fallback();
    u32 get_cpu_fallback_count() const;
    void reset_cpu_fallback_count();

    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    
    VulkanBackend::ComputeLimits get_device_limits() const;
    u32 calculate_optimal_dispatch_1d(u32 total_elements, u32 local_size = 256) const;
    std::pair<u64, u64> get_memory_usage() const;
    bool is_valid() const;

private:
    struct DeferredOperation {
        std::shared_ptr<Kernel> kernel;
        std::vector<std::shared_ptr<Tensor>> tensors;
        u32 dispatch_x, dispatch_y, dispatch_z;
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
    
    TensorOperations _tensor_ops{*this};
    DeviceMode _device_mode = DeviceMode::Auto;
    mutable std::atomic<u32> _cpu_fallback_count{0};
    
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
    
    // Internal helper methods
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

} // namespace QuasarML
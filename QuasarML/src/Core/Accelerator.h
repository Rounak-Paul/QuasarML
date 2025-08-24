#pragma once

#include <qspch.h>
#include "DataTypes.h"
#include "TensorOperations.h"
#include <VulkanBackend/VulkanBackend.h>
#include <memory>
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

    // Non-copyable but movable
    Accelerator(const Accelerator&) = delete;
    Accelerator& operator=(const Accelerator&) = delete;
    Accelerator(Accelerator&&) = default;
    Accelerator& operator=(Accelerator&&) = default;

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

    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    
    VulkanBackend::ComputeLimits get_device_limits() const;
    u32 calculate_optimal_dispatch_1d(u32 total_elements, u32 local_size = 256) const;
    std::pair<u64, u64> get_memory_usage() const;
    bool is_valid() const;

private:
    std::unique_ptr<VulkanBackend> _backend;
    std::unordered_map<std::string, std::shared_ptr<Kernel>> _kernels;
    std::vector<std::weak_ptr<Tensor>> _tensors;
    bool _recording = false;
    mutable u64 _allocated_memory = 0;
    
    TensorOperations _tensor_ops{*this};
    
    // Internal helper methods
    void cleanup_dead_tensor_references();
    void validate_tensor_compatibility(const std::vector<std::shared_ptr<Tensor>>& tensors, 
                                        std::shared_ptr<Kernel> kernel) const;
};

} // namespace QuasarML
#pragma once

#include <qspch.h>
#include "DataTypes.h"
#include <VulkanBackend/VulkanBackend.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

namespace QuasarML {

// Forward declarations
class Kernel;
class Tensor;

/**
 * @brief High-level compute accelerator interface for ML operations
 * 
 * The Accelerator class provides a user-friendly interface to the Vulkan compute backend,
 * managing resources, kernels, and tensor operations with automatic memory management
 * and RAII principles.
 */
class Accelerator {
public:
    /**
     * @brief Construct a new Accelerator
     * @param name Application name for the accelerator
     * @param gpu_idx GPU device index to use (0 for primary GPU)
     */
    explicit Accelerator(const std::string& name = "QuasarML", u32 gpu_idx = 0);
    
    /**
     * @brief Destroy the Accelerator and cleanup all resources
     */
    ~Accelerator();

    // Non-copyable but movable
    Accelerator(const Accelerator&) = delete;
    Accelerator& operator=(const Accelerator&) = delete;
    Accelerator(Accelerator&&) = default;
    Accelerator& operator=(Accelerator&&) = default;

    // ============================================================================
    // KERNEL MANAGEMENT
    // ============================================================================
    
    /**
     * @brief Create a compute kernel from GLSL compute shader source
     * @param name Unique name identifier for the kernel
     * @param glsl_source GLSL compute shader source code
     * @param num_tensors Number of tensor bindings this kernel expects
     * @param push_constant_size Size in bytes of push constant data (default: 0)
     * @return Shared pointer to the created kernel
     */
    std::shared_ptr<Kernel> create_kernel(const std::string& name,
                                        const std::string& glsl_source,
                                        u32 num_tensors,
                                        u32 push_constant_size = 0);
    
    /**
     * @brief Get an existing kernel by name
     * @param name Name of the kernel to retrieve
     * @return Shared pointer to the kernel, or nullptr if not found
     */
    std::shared_ptr<Kernel> get_kernel(const std::string& name) const;
    
    /**
     * @brief Remove a kernel from the accelerator
     * @param name Name of the kernel to remove
     * @return true if kernel was found and removed, false otherwise
     */
    bool remove_kernel(const std::string& name);
    
    /**
     * @brief Get list of all kernel names
     * @return Vector of kernel names
     */
    std::vector<std::string> get_kernel_names() const;

    // ============================================================================
    // TENSOR MANAGEMENT
    // ============================================================================
    
    /**
     * @brief Create a tensor with specified shape and data type
     * @param shape Vector specifying tensor dimensions
     * @param dtype Data type of tensor elements
     * @param device_only If true, tensor resides only on GPU (default: true)
     * @return Shared pointer to the created tensor
     */
    std::shared_ptr<Tensor> create_tensor(const std::vector<u32>& shape,
                                        DataType dtype = DataType::F32,
                                        bool device_only = true);
    
    /**
     * @brief Create a tensor from existing data
     * @param data Pointer to source data
     * @param shape Vector specifying tensor dimensions  
     * @param dtype Data type of tensor elements
     * @param device_only If true, tensor resides only on GPU (default: true)
     * @return Shared pointer to the created tensor
     */
    std::shared_ptr<Tensor> create_tensor(const void* data,
                                            const std::vector<u32>& shape,
                                            DataType dtype = DataType::F32,
                                            bool device_only = true);

    // ============================================================================
    // EXECUTION MANAGEMENT
    // ============================================================================
    
    /**
     * @brief Execute a kernel with specified tensors and dispatch configuration
     * @param kernel Kernel to execute
     * @param tensors Vector of tensors to bind to kernel
     * @param dispatch_x Number of work groups in X dimension
     * @param dispatch_y Number of work groups in Y dimension (default: 1)
     * @param dispatch_z Number of work groups in Z dimension (default: 1)
     * @param push_data Pointer to push constant data (default: nullptr)
     */
    void execute(std::shared_ptr<Kernel> kernel,
                const std::vector<std::shared_ptr<Tensor>>& tensors,
                u32 dispatch_x,
                u32 dispatch_y = 1,
                u32 dispatch_z = 1,
                const void* push_data = nullptr);
    
    /**
     * @brief Begin recording multiple kernel executions for batched submission
     */
    void begin_recording();
    
    /**
     * @brief Record a kernel execution (must be called between begin_recording and end_recording)
     * @param kernel Kernel to record
     * @param tensors Vector of tensors to bind to kernel
     * @param dispatch_x Number of work groups in X dimension
     * @param dispatch_y Number of work groups in Y dimension (default: 1)
     * @param dispatch_z Number of work groups in Z dimension (default: 1)
     * @param push_data Pointer to push constant data (default: nullptr)
     */
    void record_execution(std::shared_ptr<Kernel> kernel,
                            const std::vector<std::shared_ptr<Tensor>>& tensors,
                            u32 dispatch_x,
                            u32 dispatch_y = 1,
                            u32 dispatch_z = 1,
                            const void* push_data = nullptr);
    
    /**
     * @brief End recording and execute all recorded commands
     */
    void end_recording();
    
    /**
     * @brief Wait for all GPU operations to complete
     */
    void synchronize();
    
    /**
     * @brief Insert memory barrier between operations during recording
     */
    void memory_barrier();

    // ============================================================================
    // TENSOR OPERATIONS
    // ============================================================================

    std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    std::shared_ptr<Tensor> add_scalar(std::shared_ptr<Tensor> tensor, float scalar);
    std::shared_ptr<Tensor> mul_scalar(std::shared_ptr<Tensor> tensor, float scalar);
    std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> tensor);
    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    std::shared_ptr<Tensor> transpose(std::shared_ptr<Tensor> tensor);
    std::shared_ptr<Tensor> sum_axis(std::shared_ptr<Tensor> tensor, u32 axis);

    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    
    /**
     * @brief Get compute capability limits of the device
     * @return Structure containing device limits
     */
    VulkanBackend::ComputeLimits get_device_limits() const;
    
    /**
     * @brief Calculate optimal 1D dispatch size for given workload
     * @param total_elements Total number of elements to process
     * @param local_size Work group size (default: 256)
     * @return Optimal number of work groups
     */
    u32 calculate_optimal_dispatch_1d(u32 total_elements, u32 local_size = 256) const;
    
    /**
     * @brief Get memory usage statistics
     * @return Pair of (used_bytes, total_bytes)
     */
    std::pair<u64, u64> get_memory_usage() const;
    
    /**
     * @brief Check if accelerator is in valid state
     * @return true if accelerator is ready for operations
     */
    bool is_valid() const;

private:
    std::unique_ptr<VulkanBackend> _backend;
    std::unordered_map<std::string, std::shared_ptr<Kernel>> _kernels;
    std::vector<std::weak_ptr<Tensor>> _tensors; // Track created tensors for cleanup
    bool _recording = false;
    mutable u64 _allocated_memory = 0;
    
    // Internal helper methods
    void cleanup_dead_tensor_references();
    void validate_tensor_compatibility(const std::vector<std::shared_ptr<Tensor>>& tensors, 
                                        std::shared_ptr<Kernel> kernel) const;
    
    // Internal helper methods for tensor operations
    void validate_tensor_shape_2d(std::shared_ptr<Tensor> tensor) const;
    std::shared_ptr<Kernel> get_or_create_kernel(const std::string& name, 
                                                const std::string& glsl_source, 
                                                u32 num_tensors, u32 push_constant_size = 0);
    void validate_tensor_op_compatibility(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) const;
    
    std::string generate_elementwise_kernel_source(const std::string& operation, 
                                                    DataType dtype, 
                                                    bool is_scalar = false) const;
    std::string generate_matmul_kernel_source(DataType dtype) const;
    std::string generate_transpose_kernel_source(DataType dtype) const;
    std::string generate_sum_axis_kernel_source(DataType dtype) const;
    std::string generate_relu_kernel_source(DataType dtype) const;
    std::string get_kernel_name_for_dtype(const std::string& base_name, DataType dtype) const;
};

} // namespace QuasarML
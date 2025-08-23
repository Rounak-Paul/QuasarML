#pragma once

#include <qspch.h>
#include <VulkanBackend/VulkanBackend.h>
#include <memory>
#include <string>

namespace QuasarML {

// Forward declarations
class Tensor;

/**
 * @brief Compute kernel wrapper for GLSL compute shaders
 * 
 * The Kernel class encapsulates a Vulkan compute pipeline, providing an interface
 * for executing compute shaders with tensor bindings and push constants.
 * Uses RAII principles for automatic resource management.
 */
class Kernel {
public:
    /**
     * @brief Construct a new Kernel
     * @param backend Pointer to the Vulkan backend (non-owning)
     * @param name Unique name identifier for this kernel
     * @param glsl_source GLSL compute shader source code
     * @param expected_tensor_count Number of tensor bindings this kernel expects
     * @param push_constant_size Size in bytes of push constant data (default: 0)
     */
    Kernel(VulkanBackend* backend,
           const std::string& name,
           const std::string& glsl_source,
           u32 expected_tensor_count,
           u32 push_constant_size = 0);
    
    /**
     * @brief Destroy the Kernel and cleanup Vulkan resources
     */
    ~Kernel();

    // Non-copyable but movable
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel(Kernel&& other) noexcept;
    Kernel& operator=(Kernel&& other) noexcept;

    // ============================================================================
    // TENSOR BINDING
    // ============================================================================
    
    /**
     * @brief Bind a tensor to a specific binding point
     * @param binding Binding point index (0-based)
     * @param tensor Shared pointer to the tensor to bind
     */
    void bind_tensor(u32 binding, std::shared_ptr<Tensor> tensor);
    
    /**
     * @brief Get the tensor bound at a specific binding point
     * @param binding Binding point index
     * @return Weak pointer to the bound tensor, or expired weak_ptr if none bound
     */
    std::weak_ptr<Tensor> get_bound_tensor(u32 binding) const;
    
    /**
     * @brief Check if all required tensor bindings are satisfied
     * @return true if all bindings have valid tensors
     */
    bool are_tensors_bound() const;
    
    /**
     * @brief Clear all tensor bindings
     */
    void clear_bindings();

    // ============================================================================
    // EXECUTION
    // ============================================================================
    
    /**
     * @brief Execute the kernel synchronously
     * @param dispatch_x Number of work groups in X dimension
     * @param dispatch_y Number of work groups in Y dimension (default: 1)
     * @param dispatch_z Number of work groups in Z dimension (default: 1)
     * @param push_data Pointer to push constant data (default: nullptr)
     */
    void execute(u32 dispatch_x, 
                u32 dispatch_y = 1, 
                u32 dispatch_z = 1,
                const void* push_data = nullptr);
    
    /**
     * @brief Record kernel execution for batched submission (must be called during recording)
     * @param dispatch_x Number of work groups in X dimension
     * @param dispatch_y Number of work groups in Y dimension (default: 1)
     * @param dispatch_z Number of work groups in Z dimension (default: 1)
     * @param push_data Pointer to push constant data (default: nullptr)
     */
    void record_execution(u32 dispatch_x,
                         u32 dispatch_y = 1,
                         u32 dispatch_z = 1,
                         const void* push_data = nullptr);

    // ============================================================================
    // PROPERTIES AND VALIDATION
    // ============================================================================
    
    /**
     * @brief Get the name of this kernel
     * @return Kernel name
     */
    const std::string& get_name() const { return _name; }
    
    /**
     * @brief Get the expected number of tensor bindings
     * @return Number of expected tensor bindings
     */
    u32 get_expected_tensor_count() const { return _expected_tensor_count; }
    
    /**
     * @brief Get the push constant size in bytes
     * @return Push constant size
     */
    u32 get_push_constant_size() const { return _push_constant_size; }
    
    /**
     * @brief Check if the kernel is in a valid state
     * @return true if kernel is ready for execution
     */
    bool is_valid() const;
    
    /**
     * @brief Get the original GLSL source code
     * @return GLSL source code string
     */
    const std::string& get_glsl_source() const { return _glsl_source; }

    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    
    /**
     * @brief Calculate optimal 1D dispatch size for this kernel
     * @param total_elements Total number of elements to process
     * @param local_size Work group size (default: 256)
     * @return Optimal number of work groups in X dimension
     */
    u32 calculate_dispatch_1d(u32 total_elements, u32 local_size = 256) const;
    
    /**
     * @brief Validate push constant data size
     * @param data_size Size of push constant data in bytes
     * @return true if size matches expected push constant size
     */
    bool validate_push_constant_size(u32 data_size) const;

private:
    VulkanBackend* _backend;                              // Non-owning pointer to backend
    std::string _name;                                    // Kernel identifier
    std::string _glsl_source;                             // Original GLSL source
    u32 _expected_tensor_count;                           // Expected number of tensor bindings
    u32 _push_constant_size;                              // Size of push constants in bytes
    
    VulkanBackend::ComputePipeline _pipeline;             // Vulkan compute pipeline
    std::vector<std::weak_ptr<Tensor>> _bound_tensors;    // Bound tensors (weak references)
    
    bool _is_valid;                                       // Internal validity flag

    // Internal helper methods
    void initialize_pipeline();
    void cleanup_pipeline();
    void validate_execution_parameters(u32 dispatch_x, u32 dispatch_y, u32 dispatch_z) const;
    void update_descriptor_bindings();
};

} // namespace QuasarML
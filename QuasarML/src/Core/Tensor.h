#pragma once

#include <qspch.h>
#include <VulkanBackend/VulkanBackend.h>
#include "Accelerator.h"
#include <memory>
#include <vector>
#include <numeric>

namespace QuasarML {

/**
 * @brief Multi-dimensional array container for GPU compute operations
 * 
 * The Tensor class provides a high-level interface for multi-dimensional arrays
 * that can be efficiently processed on the GPU using Vulkan compute shaders.
 * Uses RAII principles for automatic memory management.
 */
class Tensor {
public:
    /**
     * @brief Construct a new Tensor with specified shape and data type
     * @param backend Pointer to the Vulkan backend (non-owning)
     * @param shape Vector specifying tensor dimensions
     * @param dtype Data type of tensor elements
     * @param device_only If true, tensor resides only on GPU (default: false)
     */
    Tensor(VulkanBackend* backend,
           const std::vector<u32>& shape,
           DataType dtype = DataType::F32,
           bool device_only = false);
    
    /**
     * @brief Destroy the Tensor and cleanup GPU resources
     */
    ~Tensor();

    // Non-copyable but movable for efficient transfers
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // ============================================================================
    // DATA MANAGEMENT
    // ============================================================================
    
    /**
     * @brief Upload data from host memory to tensor
     * @param data Pointer to source data (must match tensor size)
     */
    void upload_data(const void* data);
    
    /**
     * @brief Upload data from host memory with offset
     * @param data Pointer to source data
     * @param size_bytes Number of bytes to upload
     * @param offset_bytes Offset in tensor buffer (default: 0)
     */
    void upload_data(const void* data, u64 size_bytes, u64 offset_bytes = 0);
    
    /**
     * @brief Download data from tensor to host memory
     * @param data Pointer to destination buffer (must be large enough)
     */
    void download_data(void* data) const;
    
    /**
     * @brief Download data from tensor to host memory with offset
     * @param data Pointer to destination buffer
     * @param size_bytes Number of bytes to download
     * @param offset_bytes Offset in tensor buffer (default: 0)
     */
    void download_data(void* data, u64 size_bytes, u64 offset_bytes = 0) const;
    
    /**
     * @brief Fill tensor with a single scalar value
     * @param value Pointer to scalar value to fill with
     */
    void fill(const void* value);
    
    /**
     * @brief Fill tensor with zeros
     */
    void zero();
    
    /**
     * @brief Copy data from another tensor
     * @param src Source tensor to copy from
     */
    void copy_from(const Tensor& src);

    // ============================================================================
    // SHAPE AND PROPERTIES
    // ============================================================================
    
    /**
     * @brief Get tensor shape (dimensions)
     * @return Vector containing size of each dimension
     */
    const std::vector<u32>& get_shape() const { return _shape; }
    
    /**
     * @brief Get number of dimensions
     * @return Number of tensor dimensions
     */
    u32 get_rank() const { return static_cast<u32>(_shape.size()); }
    
    /**
     * @brief Get size of specific dimension
     * @param dimension Dimension index (0-based)
     * @return Size of the specified dimension
     */
    u32 get_dimension_size(u32 dimension) const;
    
    /**
     * @brief Get total number of elements
     * @return Total element count across all dimensions
     */
    u64 get_element_count() const { return _element_count; }
    
    /**
     * @brief Get tensor data type
     * @return Data type enum
     */
    DataType get_dtype() const { return _dtype; }
    
    /**
     * @brief Get size of individual elements in bytes
     * @return Element size in bytes
     */
    u32 get_element_size() const { return get_dtype_size(_dtype); }
    
    /**
     * @brief Get total tensor size in bytes
     * @return Total size in bytes
     */
    u64 get_size_bytes() const { return _element_count * get_element_size(); }
    
    /**
     * @brief Check if tensor is stored only on device (GPU)
     * @return true if device-only tensor
     */
    bool is_device_only() const { return _device_only; }

    // ============================================================================
    // SHAPE MANIPULATION
    // ============================================================================
    
    /**
     * @brief Reshape tensor to new dimensions (total elements must match)
     * @param new_shape New shape for the tensor
     * @return Reference to this tensor for chaining
     */
    Tensor& reshape(const std::vector<u32>& new_shape);
    
    /**
     * @brief Create a new tensor with reshaped dimensions
     * @param new_shape New shape for the tensor
     * @return New tensor with reshaped dimensions (shares same data)
     */
    std::shared_ptr<Tensor> create_reshaped_view(const std::vector<u32>& new_shape) const;
    
    /**
     * @brief Flatten tensor to 1D
     * @return Reference to this tensor for chaining
     */
    Tensor& flatten();
    
    /**
     * @brief Create flattened view of tensor
     * @return New 1D tensor view (shares same data)
     */
    std::shared_ptr<Tensor> create_flattened_view() const;

    // ============================================================================
    // VALIDATION AND STATE
    // ============================================================================
    
    /**
     * @brief Check if tensor is in valid state
     * @return true if tensor can be used in operations
     */
    bool is_valid() const;
    
    /**
     * @brief Check if tensor shapes are compatible for element-wise operations
     * @param other Tensor to compare with
     * @return true if shapes match exactly
     */
    bool is_shape_compatible(const Tensor& other) const;
    
    /**
     * @brief Check if tensor can be broadcast to target shape
     * @param target_shape Shape to broadcast to
     * @return true if broadcasting is possible
     */
    bool is_broadcastable_to(const std::vector<u32>& target_shape) const;
    
    /**
     * @brief Validate tensor for compute operations
     * @return true if tensor is ready for GPU operations
     */
    bool validate_for_compute() const;

    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    
    /**
     * @brief Get string representation of tensor shape
     * @return Human-readable shape string like "[2, 3, 4]"
     */
    std::string get_shape_string() const;
    
    /**
     * @brief Get detailed tensor information string
     * @return Formatted string with shape, dtype, and size info
     */
    std::string get_info_string() const;
    
    /**
     * @brief Calculate strides for this tensor shape
     * @return Vector of strides for each dimension
     */
    std::vector<u64> calculate_strides() const;
    
    /**
     * @brief Convert flat index to multi-dimensional coordinates
     * @param flat_index Linear index
     * @return Vector of coordinates in each dimension
     */
    std::vector<u32> unravel_index(u64 flat_index) const;
    
    /**
     * @brief Convert multi-dimensional coordinates to flat index
     * @param coords Coordinates in each dimension
     * @return Linear index
     */
    u64 ravel_index(const std::vector<u32>& coords) const;

    // ============================================================================
    // BACKEND ACCESS
    // ============================================================================
    
    /**
     * @brief Get reference to underlying Vulkan buffer (for advanced usage)
     * @return Reference to VulkanBackend::Buffer
     */
    VulkanBackend::Buffer& get_buffer() { return _buffer; }
    
    /**
     * @brief Get const reference to underlying Vulkan buffer
     * @return Const reference to VulkanBackend::Buffer
     */
    const VulkanBackend::Buffer& get_buffer() const { return _buffer; }

    /** for creating a view onto an existing buffer */
    Tensor(VulkanBackend* backend,
        VulkanBackend::Buffer buffer,
        const std::vector<u32>& shape,
        DataType dtype,
        bool device_only);
    
private:
    VulkanBackend* _backend;                    // Non-owning pointer to backend
    std::vector<u32> _shape;                    // Tensor dimensions
    DataType _dtype;                            // Element data type
    u64 _element_count;                         // Total number of elements
    bool _device_only;                          // Whether tensor is GPU-only
    VulkanBackend::Buffer _buffer;              // Underlying Vulkan buffer
    bool _is_valid;                             // Internal validity flag

    // Internal helper methods
    void calculate_element_count();
    void allocate_buffer();
    void validate_shape(const std::vector<u32>& shape) const;
    void validate_data_transfer(u64 size_bytes, u64 offset_bytes = 0) const;
    std::shared_ptr<Tensor> create_view_with_shape(const std::vector<u32>& new_shape) const;
    void cleanup_buffer();
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Calculate total number of elements for given shape
 * @param shape Vector of dimension sizes
 * @return Total element count
 */
inline u64 calculate_element_count(const std::vector<u32>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<u64>());
}

/**
 * @brief Check if two shapes are exactly equal
 * @param shape1 First shape
 * @param shape2 Second shape
 * @return true if shapes match
 */
inline bool shapes_equal(const std::vector<u32>& shape1, const std::vector<u32>& shape2) {
    return shape1 == shape2;
}

/**
 * @brief Validate tensor shape (all dimensions > 0)
 * @param shape Shape to validate
 * @return true if shape is valid
 */
inline bool is_valid_shape(const std::vector<u32>& shape) {
    if (shape.empty()) return false;
    for (u32 dim : shape) {
        if (dim == 0) return false;
    }
    return true;
}

/**
 * @brief Create string representation of shape
 * @param shape Shape vector
 * @return String like "[2, 3, 4]"
 */
std::string shape_to_string(const std::vector<u32>& shape);

} // namespace QuasarML
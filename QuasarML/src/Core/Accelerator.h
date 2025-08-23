#pragma once
#include <memory>
#include <unordered_map>
#include <string>
#include <VulkanBackend/VulkanBackend.h>
#include "Tensor.h"
#include "Kernel.h"
#include "DataType.h"

namespace QuasarML {

/**
 * Accelerator: Manages backend device, tensor creation, automatic op kernel management.
 * Handles kernel caching, error checking, extensibility for future ops.
 */
class Accelerator {
public:
    explicit Accelerator(const std::string& name = "QuasarAccelerator", uint32_t gpu_idx = 0);
    ~Accelerator();

    // Tensor creation
    Tensor create_tensor(const std::vector<uint32_t>& shape, DataType dtype, bool host_visible = false);

    // Kernel creation
    Kernel create_kernel(const std::string& glsl_source, uint32_t num_storage_buffers,
                        uint32_t push_constant_size = 0);

    // Synchronize compute operations
    void sync();

    VulkanBackend& backend();

    // High-level ops. Kernel code is auto-generated for dtype/shape. Result tensor is returned by value (move).
    Tensor add(const Tensor& A, const Tensor& B);
    Tensor multiply(const Tensor& A, const Tensor& B);
    // Extend for matmul, relu, etc.
private:
    std::unique_ptr<VulkanBackend> backend_;
    // Kernel cache for auto-generated kernels (for performance)
    std::unordered_map<std::string, std::unique_ptr<Kernel>> kernel_cache_;

    // Helper for kernel caching/codegen
    std::string make_add_shader(const DataType dtype, const std::vector<uint32_t>& shape) const;
    std::string make_mul_shader(const DataType dtype, const std::vector<uint32_t>& shape) const;
    // More codegen helpers as you expand ops
};

} // namespace QuasarML

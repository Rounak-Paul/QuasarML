#pragma once
#include <string>
#include <vector>
#include <VulkanBackend/VulkanBackend.h>
#include "Tensor.h"

namespace QuasarML {

/**
 * Kernel: Encapsulates compute pipeline, manages GLSL source and buffer bindings.
 * If needed, can be extended for "OpKernel" with argument parsing, autotuning.
 */
class Kernel {
public:
    Kernel(VulkanBackend& backend,
        std::string glsl_code,
        uint32_t num_storage_buffers,
        uint32_t push_constant_size = 0);
    ~Kernel();

    // Disallow copy
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    // Allow move
    Kernel(Kernel&& other) noexcept;
    Kernel& operator=(Kernel&& other) noexcept;

    bool is_valid() const noexcept;
    void bind(uint32_t binding, Tensor& tensor);
    void run(uint32_t group_x, uint32_t group_y = 1, uint32_t group_z = 1,
                const void* push_constants = nullptr, uint32_t push_constant_size = 0);

private:
    VulkanBackend* backend_ {nullptr};
    VulkanBackend::ComputePipeline pipeline_ {};
};

} // namespace QuasarML

#pragma once
#include <vector>
#include <string>
#include <memory>
#include <VulkanBackend/VulkanBackend.h>
#include "DataType.h"

namespace QuasarML {

/**
 * Tensor: Opaque GPU storage, owned by backend. 
 * RAII, supports upload/download and query interfaces.
 */
class Tensor {
public:
    Tensor(VulkanBackend& backend,
        std::vector<uint32_t> shape,
        DataType dtype,
        bool host_visible = false);

    ~Tensor();

    // Disallow copy
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    // Allow move
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    bool is_valid() const noexcept;
    size_t num_elements() const noexcept;
    size_t size_bytes() const noexcept;
    const std::vector<uint32_t>& shape() const noexcept;
    DataType dtype() const noexcept;

    void upload(const void* src, size_t byte_size, size_t offset = 0);
    void download(void* dst, size_t byte_size, size_t offset = 0);

    VulkanBackend::Buffer& buffer();

private:
    VulkanBackend* backend_ {nullptr};
    VulkanBackend::Buffer buffer_ {};
    std::vector<uint32_t> shape_ {};
    DataType dtype_ {DataType::FLOAT32};
    size_t size_in_bytes_ {0};
};

} // namespace QuasarML

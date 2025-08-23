#include "Kernel.h"
using namespace QuasarML;

Kernel::Kernel(VulkanBackend& backend,
               std::string glsl_code,
               uint32_t num_storage_buffers,
               uint32_t push_constant_size)
    : backend_(&backend)
{
    pipeline_ = backend_->create_compute_pipeline(glsl_code, num_storage_buffers, push_constant_size);
}

Kernel::~Kernel() {
    if (is_valid()) backend_->destroy_compute_pipeline(pipeline_);
}

Kernel::Kernel(Kernel&& other) noexcept :
    backend_(other.backend_),
    pipeline_(other.pipeline_)
{
    other.backend_ = nullptr;
    other.pipeline_ = VulkanBackend::ComputePipeline{};
}

Kernel& Kernel::operator=(Kernel&& other) noexcept {
    if (this != &other) {
        if (is_valid()) backend_->destroy_compute_pipeline(pipeline_);
        backend_ = other.backend_;
        pipeline_ = other.pipeline_;
        other.backend_ = nullptr;
        other.pipeline_ = VulkanBackend::ComputePipeline{};
    }
    return *this;
}

bool Kernel::is_valid() const noexcept { return pipeline_.is_valid(); }
void Kernel::bind(uint32_t binding, Tensor& tensor) {
    backend_->bind_buffer_to_pipeline(pipeline_, binding, tensor.buffer());
}
void Kernel::run(uint32_t group_x, uint32_t group_y, uint32_t group_z,
                 const void* push_constants, uint32_t push_constant_size) {
    backend_->execute_compute(pipeline_, group_x, group_y, group_z, push_constants, push_constant_size);
}

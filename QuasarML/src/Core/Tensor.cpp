#include "Tensor.h"
#include <stdexcept>
#include <numeric>
using namespace QuasarML;

Tensor::Tensor(VulkanBackend& backend,
               std::vector<uint32_t> shape,
               DataType dtype,
               bool host_visible)
    : backend_(&backend),
      shape_(std::move(shape)),
      dtype_(dtype)
{
    if (shape_.empty()) throw std::runtime_error("Tensor shape can't be empty");
    size_t n = std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    size_in_bytes_ = DataTypeSize(dtype_) * n;
    buffer_ = backend_->create_storage_buffer(size_in_bytes_, host_visible);
}

Tensor::~Tensor() {
    if (is_valid()) backend_->destroy_buffer(buffer_);
}

Tensor::Tensor(Tensor&& other) noexcept :
    backend_(other.backend_),
    buffer_(other.buffer_),
    shape_(std::move(other.shape_)),
    dtype_(other.dtype_),
    size_in_bytes_(other.size_in_bytes_)
{
    other.backend_ = nullptr;
    other.buffer_ = VulkanBackend::Buffer{};
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (is_valid()) backend_->destroy_buffer(buffer_);
        backend_ = other.backend_;
        buffer_ = other.buffer_;
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        size_in_bytes_ = other.size_in_bytes_;
        other.backend_ = nullptr;
        other.buffer_ = VulkanBackend::Buffer{};
    }
    return *this;
}

bool Tensor::is_valid() const noexcept { return buffer_.is_valid(); }
size_t Tensor::num_elements() const noexcept {
    return std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
}
size_t Tensor::size_bytes() const noexcept { return size_in_bytes_; }
const std::vector<uint32_t>& Tensor::shape() const noexcept { return shape_; }
DataType Tensor::dtype() const noexcept { return dtype_; }

void Tensor::upload(const void* src, size_t byte_size, size_t offset) {
    backend_->upload_to_buffer(buffer_, src, byte_size, offset);
}
void Tensor::download(void* dst, size_t byte_size, size_t offset) {
    backend_->download_from_buffer(buffer_, dst, byte_size, offset);
}
VulkanBackend::Buffer& Tensor::buffer() { return buffer_; }

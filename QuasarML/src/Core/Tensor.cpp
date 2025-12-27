#include "Tensor.h"
#include "Device.h"
#include <Common/Assert.h>
#include <cstring>
#include <sstream>

namespace QuasarML {

Tensor::Tensor(Device* device, const std::vector<u32>& shape, DataType dtype, bool host_visible)
    : _device(device), _shape(shape), _dtype(dtype), _host_visible(host_visible), _owns_buffer(true) {
    validate_shape(shape);
    compute_numel();
    alloc_buffer();
    _valid = _buffer.valid();
}

Tensor::Tensor(Device* device, BufferHandle buffer, const std::vector<u32>& shape,
               DataType dtype, bool host_visible, u64 element_offset, bool owns_buffer)
    : _device(device), _shape(shape), _dtype(dtype), _host_visible(host_visible),
      _buffer(buffer), _element_offset(element_offset), _owns_buffer(owns_buffer) {
    compute_numel();
    _valid = _buffer.valid();
}

Tensor::~Tensor() {
    if (_owns_buffer && _buffer.valid() && _device) {
        _device->backend()->destroy_buffer(_buffer);
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : _device(other._device), _shape(std::move(other._shape)), _dtype(other._dtype),
      _numel(other._numel), _host_visible(other._host_visible), _buffer(other._buffer),
      _element_offset(other._element_offset), _owns_buffer(other._owns_buffer), _valid(other._valid) {
    other._buffer = {};
    other._owns_buffer = false;
    other._valid = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (_owns_buffer && _buffer.valid() && _device) {
            _device->backend()->destroy_buffer(_buffer);
        }
        _device = other._device;
        _shape = std::move(other._shape);
        _dtype = other._dtype;
        _numel = other._numel;
        _host_visible = other._host_visible;
        _buffer = other._buffer;
        _element_offset = other._element_offset;
        _owns_buffer = other._owns_buffer;
        _valid = other._valid;
        other._buffer = {};
        other._owns_buffer = false;
        other._valid = false;
    }
    return *this;
}

void Tensor::compute_numel() {
    _numel = 1;
    for (u32 d : _shape) _numel *= d;
}

void Tensor::alloc_buffer() {
    u64 bytes = _numel * dtype_size(_dtype);
    _buffer = _device->backend()->create_storage_buffer(bytes, _host_visible);
}

void Tensor::validate_shape(const std::vector<u32>& shape) const {
    QS_ASSERT(!shape.empty(), "Shape cannot be empty");
    for (u32 d : shape) QS_ASSERT(d > 0, "Dimension must be positive");
}

u32 Tensor::dim(u32 i) const {
    QS_ASSERT(i < _shape.size(), "Dimension index out of range");
    return _shape[i];
}

void Tensor::upload(const void* data) {
    upload(data, size_bytes(), 0);
}

void Tensor::upload(const void* data, u64 size, u64 offset) {
    QS_ASSERT(_valid, "Tensor not valid");
    _device->backend()->upload_buffer(_buffer, data, size, offset + _element_offset * element_size());
}

void Tensor::download(void* data) const {
    download(data, size_bytes(), 0);
}

void Tensor::download(void* data, u64 size, u64 offset) const {
    QS_ASSERT(_valid, "Tensor not valid");
    BufferHandle& buf = const_cast<BufferHandle&>(_buffer);
    _device->backend()->download_buffer(buf, data, size, offset + _element_offset * element_size());
}

void Tensor::fill(const void* value) {
    u64 elem_size = element_size();
    std::vector<u8> buf(_numel * elem_size);
    for (u64 i = 0; i < _numel; ++i) {
        std::memcpy(buf.data() + i * elem_size, value, elem_size);
    }
    upload(buf.data());
}

void Tensor::zero() {
    std::vector<u8> buf(size_bytes(), 0);
    upload(buf.data());
}

void Tensor::copy_from(const Tensor& src) {
    QS_ASSERT(size_bytes() == src.size_bytes(), "Size mismatch");
    _device->backend()->copy_buffer(
        const_cast<BufferHandle&>(src._buffer), _buffer, size_bytes(),
        src._element_offset * src.element_size(), _element_offset * element_size());
}

Tensor& Tensor::reshape(const std::vector<u32>& new_shape) {
    u64 new_numel = QuasarML::compute_numel(new_shape);
    QS_ASSERT(new_numel == _numel, "Element count mismatch in reshape");
    _shape = new_shape;
    return *this;
}

std::shared_ptr<Tensor> Tensor::view(const std::vector<u32>& new_shape) const {
    u64 new_numel = QuasarML::compute_numel(new_shape);
    QS_ASSERT(new_numel == _numel, "Element count mismatch in view");
    return std::make_shared<Tensor>(_device, _buffer, new_shape, _dtype, _host_visible, _element_offset, false);
}

std::shared_ptr<Tensor> Tensor::flatten() const {
    return view({static_cast<u32>(_numel)});
}

bool Tensor::is_shape_compatible(const Tensor& other) const {
    return _shape == other._shape;
}

bool Tensor::is_broadcastable(const std::vector<u32>& target) const {
    if (target.size() < _shape.size()) return false;
    size_t diff = target.size() - _shape.size();
    for (size_t i = 0; i < _shape.size(); ++i) {
        u32 a = _shape[i];
        u32 b = target[diff + i];
        if (a != b && a != 1 && b != 1) return false;
    }
    return true;
}

std::vector<u64> Tensor::strides() const {
    std::vector<u64> s(_shape.size());
    u64 stride = 1;
    for (int i = static_cast<int>(_shape.size()) - 1; i >= 0; --i) {
        s[i] = stride;
        stride *= _shape[i];
    }
    return s;
}

std::string Tensor::shape_str() const {
    return shape_to_string(_shape);
}

std::string Tensor::info() const {
    std::ostringstream ss;
    ss << "Tensor(" << shape_str() << ", " << dtype_to_string(_dtype) << ")";
    return ss.str();
}

std::string shape_to_string(const std::vector<u32>& shape) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

}

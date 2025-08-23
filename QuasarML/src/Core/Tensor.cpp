#include "Tensor.h"
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace QuasarML {

Tensor::Tensor(VulkanBackend* backend,
               const std::vector<u32>& shape,
               DataType dtype,
               bool device_only)
    : _backend(backend)
    , _shape(shape)
    , _dtype(dtype)
    , _device_only(device_only)
    , _is_valid(false)
{
    if (!_backend) {
        throw std::invalid_argument("Backend cannot be null");
    }
    
    validate_shape(_shape);
    calculate_element_count();
    allocate_buffer();
}

Tensor::~Tensor() {
    cleanup_buffer();
}

Tensor::Tensor(Tensor&& other) noexcept
    : _backend(other._backend)
    , _shape(std::move(other._shape))
    , _dtype(other._dtype)
    , _element_count(other._element_count)
    , _device_only(other._device_only)
    , _buffer(std::move(other._buffer))
    , _is_valid(other._is_valid)
{
    // Reset other object
    other._backend = nullptr;
    other._element_count = 0;
    other._is_valid = false;
    other._buffer = {};
}

Tensor::Tensor(VulkanBackend* backend, VulkanBackend::Buffer buffer, const std::vector<u32>& shape, DataType dtype, bool device_only)
    : _backend(backend)
    , _buffer(buffer)
    , _shape(shape)
    , _dtype(dtype)
    , _device_only(device_only)
    , _is_valid(true)
{
    validate_shape(_shape);
    calculate_element_count();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Cleanup current resources
        cleanup_buffer();
        
        // Move data
        _backend = other._backend;
        _shape = std::move(other._shape);
        _dtype = other._dtype;
        _element_count = other._element_count;
        _device_only = other._device_only;
        _buffer = std::move(other._buffer);
        _is_valid = other._is_valid;
        
        // Reset other object
        other._backend = nullptr;
        other._element_count = 0;
        other._is_valid = false;
        other._buffer = {};
    }
    return *this;
}

void Tensor::upload_data(const void* data) {
    if (!data) {
        throw std::invalid_argument("Data pointer cannot be null");
    }
    
    if (!is_valid()) {
        throw std::runtime_error("Tensor is not in valid state");
    }
    
    u64 total_size = get_size_bytes();
    _backend->upload_to_buffer(_buffer, data, total_size, 0);
}

void Tensor::upload_data(const void* data, u64 size_bytes, u64 offset_bytes) {
    if (!data) {
        throw std::invalid_argument("Data pointer cannot be null");
    }
    
    if (!is_valid()) {
        throw std::runtime_error("Tensor is not in valid state");
    }
    
    validate_data_transfer(size_bytes, offset_bytes);
    _backend->upload_to_buffer(_buffer, data, size_bytes, offset_bytes);
}

void Tensor::download_data(void* data) const {
    if (!data) {
        throw std::invalid_argument("Data pointer cannot be null");
    }
    
    if (!is_valid()) {
        throw std::runtime_error("Tensor is not in valid state");
    }
    
    u64 total_size = get_size_bytes();
    _backend->download_from_buffer(const_cast<VulkanBackend::Buffer&>(_buffer),
+                                  data, total_size, 0);
}

void Tensor::download_data(void* data, u64 size_bytes, u64 offset_bytes) const {
    if (!data) {
        throw std::invalid_argument("Data pointer cannot be null");
    }
    
    if (!is_valid()) {
        throw std::runtime_error("Tensor is not in valid state");
    }
    
    validate_data_transfer(size_bytes, offset_bytes);
    _backend->download_from_buffer(const_cast<VulkanBackend::Buffer&>(_buffer),
+                                  data, size_bytes, offset_bytes);
}

void Tensor::fill(const void* value) {
    if (!value) {
        throw std::invalid_argument("Value pointer cannot be null");
    }
    
    if (!is_valid()) {
        throw std::runtime_error("Tensor is not in valid state");
    }
    
    u32 element_size = get_element_size();
    u64 total_size = get_size_bytes();
    
    // Create temporary buffer to repeat the value
    std::vector<u8> fill_data(total_size);
    const u8* value_bytes = static_cast<const u8*>(value);
    
    // Fill buffer with repeated value
    for (u64 i = 0; i < _element_count; ++i) {
        std::memcpy(fill_data.data() + i * element_size, value_bytes, element_size);
    }
    
    upload_data(fill_data.data());
}

void Tensor::zero() {
    if (!is_valid()) {
        throw std::runtime_error("Tensor is not in valid state");
    }
    
    u64 total_size = get_size_bytes();
    std::vector<u8> zero_data(total_size, 0);
    upload_data(zero_data.data());
}

void Tensor::copy_from(const Tensor& src) {
    if (!src.is_valid() || !is_valid()) {
        throw std::runtime_error("Source or destination tensor is not valid");
    }
    
    if (!is_shape_compatible(src)) {
        throw std::invalid_argument("Tensor shapes are not compatible for copying");
    }
    
    if (_dtype != src._dtype) {
        throw std::invalid_argument("Data types must match for copying");
    }
    
    u64 copy_size = get_size_bytes();
    _backend->copy_buffer(const_cast<VulkanBackend::Buffer&>(src._buffer), _buffer, copy_size);
}

u32 Tensor::get_dimension_size(u32 dimension) const {
    if (dimension >= _shape.size()) {
        throw std::out_of_range("Dimension index exceeds tensor rank");
    }
    return _shape[dimension];
}

Tensor& Tensor::reshape(const std::vector<u32>& new_shape) {
    validate_shape(new_shape);
    
    u64 new_element_count = ::QuasarML::calculate_element_count(new_shape);
    if (new_element_count != _element_count) {
        throw std::invalid_argument("New shape must have same total number of elements");
    }
    
    _shape = new_shape;
    return *this;
}

std::shared_ptr<Tensor> Tensor::create_reshaped_view(const std::vector<u32>& new_shape) const {
    return create_view_with_shape(new_shape);
}

Tensor& Tensor::flatten() {
    _shape = {static_cast<u32>(_element_count)};
    return *this;
}

std::shared_ptr<Tensor> Tensor::create_flattened_view() const {
    return create_view_with_shape({static_cast<u32>(_element_count)});
}

bool Tensor::is_valid() const {
    return _is_valid && _backend && _buffer.is_valid();
}

bool Tensor::is_shape_compatible(const Tensor& other) const {
    return shapes_equal(_shape, other._shape);
}

bool Tensor::is_broadcastable_to(const std::vector<u32>& target_shape) const {
    // Simple broadcasting rules - can be extended for more complex cases
    if (target_shape.size() < _shape.size()) {
        return false;
    }
    
    // Check from the last dimension backwards
    int target_idx = static_cast<int>(target_shape.size()) - 1;
    int current_idx = static_cast<int>(_shape.size()) - 1;
    
    while (current_idx >= 0 && target_idx >= 0) {
        u32 current_dim = _shape[current_idx];
        u32 target_dim = target_shape[target_idx];
        
        if (current_dim != target_dim && current_dim != 1) {
            return false;
        }
        
        --current_idx;
        --target_idx;
    }
    
    return true;
}

bool Tensor::validate_for_compute() const {
    return is_valid() && _element_count > 0;
}

std::string Tensor::get_shape_string() const {
    return shape_to_string(_shape);
}

std::string Tensor::get_info_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=" << get_shape_string()
        << ", dtype=" << dtype_to_string(_dtype)
        << ", elements=" << _element_count
        << ", bytes=" << get_size_bytes()
        << ", device_only=" << (_device_only ? "true" : "false")
        << ")";
    return oss.str();
}

std::vector<u64> Tensor::calculate_strides() const {
    std::vector<u64> strides(_shape.size());
    if (_shape.empty()) return strides;
    
    strides.back() = 1;
    for (int i = static_cast<int>(_shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * _shape[i + 1];
    }
    
    return strides;
}

std::vector<u32> Tensor::unravel_index(u64 flat_index) const {
    if (flat_index >= _element_count) {
        throw std::out_of_range("Flat index exceeds tensor size");
    }
    
    std::vector<u32> coords(_shape.size());
    auto strides = calculate_strides();
    
    for (size_t i = 0; i < _shape.size(); ++i) {
        coords[i] = static_cast<u32>(flat_index / strides[i]);
        flat_index %= strides[i];
    }
    
    return coords;
}

u64 Tensor::ravel_index(const std::vector<u32>& coords) const {
    if (coords.size() != _shape.size()) {
        throw std::invalid_argument("Coordinate dimension must match tensor rank");
    }
    
    auto strides = calculate_strides();
    u64 flat_index = 0;
    
    for (size_t i = 0; i < coords.size(); ++i) {
        if (coords[i] >= _shape[i]) {
            throw std::out_of_range("Coordinate exceeds dimension size");
        }
        flat_index += coords[i] * strides[i];
    }
    
    return flat_index;
}

void Tensor::calculate_element_count() {
    _element_count = ::QuasarML::calculate_element_count(_shape);
}

void Tensor::allocate_buffer() {
    if (!_backend) {
        throw std::runtime_error("Backend not available");
    }
    
    u64 buffer_size = get_size_bytes();
    if (buffer_size == 0) {
        throw std::runtime_error("Cannot allocate zero-sized buffer");
    }
    
    try {
        _buffer = _backend->create_storage_buffer(buffer_size, !_device_only);
        _is_valid = _buffer.is_valid();
        
        if (!_is_valid) {
            throw std::runtime_error("Failed to create storage buffer");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Buffer allocation failed: " + std::string(e.what()));
    }
}

void Tensor::validate_shape(const std::vector<u32>& shape) const {
    if (!is_valid_shape(shape)) {
        throw std::invalid_argument("Invalid tensor shape - must be non-empty with all dimensions > 0");
    }
}

void Tensor::validate_data_transfer(u64 size_bytes, u64 offset_bytes) const {
    u64 total_size = get_size_bytes();
    
    if (offset_bytes >= total_size) {
        throw std::out_of_range("Offset exceeds tensor size");
    }
    
    if (offset_bytes + size_bytes > total_size) {
        throw std::out_of_range("Data transfer exceeds tensor bounds");
    }
}

std::shared_ptr<Tensor> Tensor::create_view_with_shape(const std::vector<u32>& new_shape) const {
    return std::make_shared<Tensor>(_backend, _buffer, new_shape, _dtype, _device_only);
}

void Tensor::cleanup_buffer() {
    if (_backend && _buffer.is_valid()) {
        _backend->destroy_buffer(_buffer);
    }
    _buffer = {};
    _is_valid = false;
}

// Utility function implementations
std::string shape_to_string(const std::vector<u32>& shape) {
    if (shape.empty()) {
        return "[]";
    }
    
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

} // namespace QuasarML
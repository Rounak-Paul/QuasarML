

#pragma once

#include <qspch.h>
#include "DataTypes.h"
#include <VulkanBackend/VulkanBackend.h>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <functional>

namespace QuasarML {

class Accelerator;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Tensor(Accelerator* accelerator,
           VulkanBackend* backend,
           const std::vector<u32>& shape,
           DataType dtype = DataType::F32,
           bool host_visible = false);

    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    void upload_data(const void* data);
    void upload_data(const void* data, u64 size_bytes, u64 offset_bytes = 0);
    void download_data(void* data) const;
    void download_data(void* data, u64 size_bytes, u64 offset_bytes = 0) const;
    void fill(const void* value);
    void zero();
    void copy_from(const Tensor& src);

    const std::vector<u32>& get_shape() const { return _shape; }
    u32 get_rank() const { return static_cast<u32>(_shape.size()); }
    u32 get_dimension_size(u32 dimension) const;
    u64 get_element_count() const { return _element_count; }
    DataType get_dtype() const { return _dtype; }
    u32 get_element_size() const { return get_dtype_size(_dtype); }
    u64 get_size_bytes() const { return _element_count * get_element_size(); }
    bool is_host_visible() const { return _host_visible; }

    Tensor& reshape(const std::vector<u32>& new_shape);
    std::shared_ptr<Tensor> create_reshaped_view(const std::vector<u32>& new_shape) const;
    Tensor& flatten();
    std::shared_ptr<Tensor> create_flattened_view() const;

    bool is_valid() const;
    bool is_shape_compatible(const Tensor& other) const;
    bool is_broadcastable_to(const std::vector<u32>& target_shape) const;
    bool validate_for_compute() const;

    std::string get_shape_string() const;
    std::string get_info_string() const;
    std::vector<u64> calculate_strides() const;
    std::vector<u32> unravel_index(u64 flat_index) const;
    u64 ravel_index(const std::vector<u32>& coords) const;

    VulkanBackend::Buffer& get_buffer() { return _buffer; }
    const VulkanBackend::Buffer& get_buffer() const { return _buffer; }
    u64 get_element_offset() const { return _element_offset; }

    Tensor(Accelerator* accelerator,
           VulkanBackend* backend,
           VulkanBackend::Buffer buffer,
           const std::vector<u32>& shape,
           DataType dtype,
           bool host_visible,
           bool owns_buffer = true);
           
    Tensor(Accelerator* accelerator,
           VulkanBackend* backend,
           VulkanBackend::Buffer buffer,
           const std::vector<u32>& shape,
           DataType dtype,
           bool host_visible,
           u64 element_offset,
           bool owns_buffer = true);

    std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& other) const;
    std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& other) const;
    std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& other) const;
    std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& other) const;

    std::shared_ptr<Tensor> operator+(float scalar) const;
    std::shared_ptr<Tensor> operator-(float scalar) const;
    std::shared_ptr<Tensor> operator*(float scalar) const;
    std::shared_ptr<Tensor> operator/(float scalar) const;

    friend std::shared_ptr<Tensor> operator+(float scalar, const Tensor& tensor);
    friend std::shared_ptr<Tensor> operator-(float scalar, const Tensor& tensor);
    friend std::shared_ptr<Tensor> operator*(float scalar, const Tensor& tensor);
    friend std::shared_ptr<Tensor> operator/(float scalar, const Tensor& tensor);
    
    Accelerator* get_accelerator() const { return _accelerator; }

    std::shared_ptr<Tensor> create_view_with_shape(const std::vector<u32>& new_shape) const;
    std::shared_ptr<Tensor> create_view_with_shape_and_offset(const std::vector<u32>& new_shape, u64 element_offset) const;

private:
    Accelerator* _accelerator;
    VulkanBackend* _backend;
    std::vector<u32> _shape;
    DataType _dtype;
    u64 _element_count;
    bool _host_visible;
    VulkanBackend::Buffer _buffer;
    bool _is_valid;
    u64 _element_offset = 0;
    bool _owns_buffer = true;

    void calculate_element_count();
    void check_accelerator_match(const std::shared_ptr<Tensor>& other) const;
    void allocate_buffer();
    void validate_shape(const std::vector<u32>& shape) const;
    void validate_data_transfer(u64 size_bytes, u64 offset_bytes = 0) const;
    void cleanup_buffer();
};

inline u64 calculate_element_count(const std::vector<u32>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<u64>());
}

inline bool shapes_equal(const std::vector<u32>& a, const std::vector<u32>& b) { 
    return a == b; 
}

inline bool is_valid_shape(const std::vector<u32>& shape) {
    if (shape.empty()) return false;
    for (auto d : shape) if (d == 0) return false;
    return true;
}

std::string shape_to_string(const std::vector<u32>& shape);

}
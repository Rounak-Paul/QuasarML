#pragma once

#include <Common/Types.h>
#include <Backend/BackendInterface.h>
#include <memory>
#include <vector>
#include <string>
#include <numeric>
#include <functional>

namespace QuasarML {

class Device;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Tensor(Device* device, const std::vector<u32>& shape, DataType dtype = DataType::F32, bool host_visible = false);
    ~Tensor();
    
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    void upload(const void* data);
    void upload(const void* data, u64 size, u64 offset = 0);
    void download(void* data) const;
    void download(void* data, u64 size, u64 offset = 0) const;
    void fill(const void* value);
    void zero();
    void copy_from(const Tensor& src);
    
    const std::vector<u32>& shape() const { return _shape; }
    u32 rank() const { return static_cast<u32>(_shape.size()); }
    u32 dim(u32 i) const;
    u64 numel() const { return _numel; }
    DataType dtype() const { return _dtype; }
    u32 element_size() const { return dtype_size(_dtype); }
    u64 size_bytes() const { return _numel * element_size(); }
    bool is_host_visible() const { return _host_visible; }
    
    Tensor& reshape(const std::vector<u32>& new_shape);
    std::shared_ptr<Tensor> view(const std::vector<u32>& new_shape) const;
    std::shared_ptr<Tensor> flatten() const;
    
    bool is_valid() const { return _valid; }
    bool is_shape_compatible(const Tensor& other) const;
    bool is_broadcastable(const std::vector<u32>& target) const;
    
    std::string shape_str() const;
    std::string info() const;
    std::vector<u64> strides() const;
    
    BufferHandle& buffer() { return _buffer; }
    const BufferHandle& buffer() const { return _buffer; }
    u64 element_offset() const { return _element_offset; }
    Device* device() const { return _device; }
    
    Tensor(Device* device, BufferHandle buffer, const std::vector<u32>& shape, 
           DataType dtype, bool host_visible, u64 element_offset = 0, bool owns_buffer = true);

private:
    Device* _device;
    std::vector<u32> _shape;
    DataType _dtype;
    u64 _numel;
    bool _host_visible;
    BufferHandle _buffer;
    u64 _element_offset = 0;
    bool _owns_buffer = true;
    bool _valid = false;
    
    void compute_numel();
    void alloc_buffer();
    void validate_shape(const std::vector<u32>& shape) const;
};

inline u64 compute_numel(const std::vector<u32>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<u64>());
}

inline bool shapes_equal(const std::vector<u32>& a, const std::vector<u32>& b) { return a == b; }

std::string shape_to_string(const std::vector<u32>& shape);

}

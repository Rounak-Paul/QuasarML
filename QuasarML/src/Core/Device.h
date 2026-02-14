#pragma once

#include <Common/Types.h>
#include <Backend/BackendInterface.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace QuasarML {

class Tensor;
class Kernel;

class QS_API Device {
public:
    Device(const std::string& name, u32 device_index = 0);
    ~Device();
    
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;
    
    bool is_valid() const;
    u32 device_index() const { return _device_index; }
    const std::string& name() const { return _name; }
    
    Backend* backend() const { return _backend.get(); }
    ComputeLimits limits() const;
    const DeviceCapabilities& capabilities() const;
    
    std::shared_ptr<Kernel> create_kernel(const std::string& name, const std::string& glsl_source, u32 num_bindings, u32 push_size = 0);
    std::shared_ptr<Kernel> get_kernel(const std::string& name);
    
    std::shared_ptr<Tensor> create_tensor(const std::vector<u32>& shape, DataType dtype = DataType::F32, bool host_visible = false);
    
    void begin_batch();
    void end_batch();
    void flush_pending();
    void synchronize();
    void wait_idle();
    
private:
    std::unique_ptr<Backend> _backend;
    std::unordered_map<std::string, std::shared_ptr<Kernel>> _kernels;
    std::string _name;
    u32 _device_index = 0;
    mutable std::mutex _kernel_mutex;
    bool _in_batch = false;
};

}

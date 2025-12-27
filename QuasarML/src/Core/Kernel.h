#pragma once

#include <Common/Types.h>
#include <Backend/BackendInterface.h>
#include <memory>
#include <vector>
#include <string>

namespace QuasarML {

class Device;
class Tensor;

class Kernel {
public:
    Kernel(Device* device, const std::string& name, const std::string& glsl_source, u32 num_bindings, u32 push_constant_size = 0);
    ~Kernel();
    
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel(Kernel&& other) noexcept;
    Kernel& operator=(Kernel&& other) noexcept;
    
    void bind(u32 binding, std::shared_ptr<Tensor> tensor);
    void clear_bindings();
    
    void execute(u32 dispatch_x, u32 dispatch_y = 1, u32 dispatch_z = 1, const void* push_data = nullptr);
    void record(u32 dispatch_x, u32 dispatch_y = 1, u32 dispatch_z = 1, const void* push_data = nullptr);
    
    const std::string& name() const { return _name; }
    u32 num_bindings() const { return _num_bindings; }
    u32 push_constant_size() const { return _push_size; }
    bool is_valid() const { return _valid; }
    const std::string& glsl() const { return _glsl; }
    
    u32 optimal_dispatch_1d(u32 total, u32 local_size = 256) const;

    PipelineHandle& pipeline() { return _pipeline; }

private:
    Device* _device;
    std::string _name;
    std::string _glsl;
    u32 _num_bindings;
    u32 _push_size;
    PipelineHandle _pipeline;
    std::vector<std::weak_ptr<Tensor>> _bindings;
    bool _valid = false;
};

}

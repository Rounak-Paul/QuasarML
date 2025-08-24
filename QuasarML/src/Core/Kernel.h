#pragma once

#include <qspch.h>
#include <VulkanBackend/VulkanBackend.h>
#include "Tensor.h"

namespace QuasarML {

class Kernel {
public:
    Kernel(VulkanBackend* backend,
           const std::string& name,
           const std::string& glsl_source,
           u32 expected_tensor_count,
           u32 push_constant_size = 0);

    ~Kernel();

    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel(Kernel&& other) noexcept;
    Kernel& operator=(Kernel&& other) noexcept;

    void bind_tensor(u32 binding, std::shared_ptr<Tensor> tensor);
    std::weak_ptr<Tensor> get_bound_tensor(u32 binding) const;
    bool are_tensors_bound() const;
    void clear_bindings();

    void execute(u32 dispatch_x,
                 u32 dispatch_y = 1,
                 u32 dispatch_z = 1,
                 const void* push_data = nullptr);

    void record_execution(u32 dispatch_x,
                          u32 dispatch_y = 1,
                          u32 dispatch_z = 1,
                          const void* push_data = nullptr);

    const std::string& get_name() const { return _name; }
    u32 get_expected_tensor_count() const { return _expected_tensor_count; }
    u32 get_push_constant_size() const { return _push_constant_size; }
    bool is_valid() const;
    const std::string& get_glsl_source() const { return _glsl_source; }

    u32 calculate_dispatch_1d(u32 total_elements, u32 local_size = 256) const;
    bool validate_push_constant_size(u32 data_size) const;

private:
    VulkanBackend* _backend;
    std::string _name;
    std::string _glsl_source;
    u32 _expected_tensor_count;
    u32 _push_constant_size;

    VulkanBackend::ComputePipeline _pipeline;
    std::vector<std::weak_ptr<Tensor>> _bound_tensors;
    bool _is_valid;

    void initialize_pipeline();
    void cleanup_pipeline();
    void validate_execution_parameters(u32 dispatch_x, u32 dispatch_y, u32 dispatch_z) const;
    void update_descriptor_bindings();
};

} // namespace QuasarML
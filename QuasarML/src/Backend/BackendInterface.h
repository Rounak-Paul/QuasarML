#pragma once

#include <Common/Types.h>
#include <Backend/DeviceCapabilities.h>
#include <vector>
#include <string>
#include <memory>

namespace QuasarML {

struct BufferHandle {
    void* native_handle = nullptr;
    void* allocation = nullptr;
    u64 size = 0;
    void* mapped = nullptr;
    
    bool valid() const { return native_handle != nullptr; }
};

struct PipelineHandle {
    void* pipeline = nullptr;
    void* layout = nullptr;
    void* descriptor_layout = nullptr;
    void* descriptor_pool = nullptr;
    u32 binding_count = 0;
    
    bool valid() const { return pipeline != nullptr; }
};

struct ComputeLimits {
    u32 max_workgroup_size[3];
    u32 max_workgroup_count[3];
    u32 max_workgroup_invocations;
    u32 max_shared_memory;
};

struct BufferBinding {
    BufferHandle* buffer = nullptr;
    u64 offset = 0;
    u64 range = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    
    virtual bool init(const std::string& name, u32 device_index) = 0;
    virtual void shutdown() = 0;
    virtual bool is_valid() const = 0;
    virtual u32 get_device_index() const = 0;
    
    virtual BufferHandle create_storage_buffer(u64 size, bool host_visible) = 0;
    virtual BufferHandle create_staging_buffer(u64 size) = 0;
    virtual void destroy_buffer(BufferHandle& buffer) = 0;
    
    virtual void upload_buffer(BufferHandle& buffer, const void* data, u64 size, u64 offset = 0) = 0;
    virtual void download_buffer(BufferHandle& buffer, void* data, u64 size, u64 offset = 0) = 0;
    virtual void copy_buffer(BufferHandle& src, BufferHandle& dst, u64 size, u64 src_offset = 0, u64 dst_offset = 0) = 0;
    
    virtual PipelineHandle create_compute_pipeline(const std::string& glsl_source, u32 num_bindings, u32 push_constant_size = 0) = 0;
    virtual void destroy_pipeline(PipelineHandle& pipeline) = 0;
    
    virtual void execute_compute(PipelineHandle& pipeline, 
                                 const std::vector<BufferBinding>& buffers,
                                 u32 group_x, u32 group_y = 1, u32 group_z = 1,
                                 const void* push_data = nullptr, u32 push_size = 0) = 0;
    
    virtual void begin_recording() = 0;
    virtual void record_compute(PipelineHandle& pipeline,
                                const std::vector<BufferBinding>& buffers,
                                u32 group_x, u32 group_y = 1, u32 group_z = 1,
                                const void* push_data = nullptr, u32 push_size = 0) = 0;
    virtual void end_recording() = 0;
    
    virtual void synchronize() = 0;
    virtual void memory_barrier() = 0;
    virtual void device_wait_idle() = 0;
    
    virtual ComputeLimits get_compute_limits() const = 0;
    virtual u32 optimal_dispatch_1d(u32 total, u32 local_size = 256) const = 0;
    virtual const DeviceCapabilities& get_capabilities() const = 0;
};

std::unique_ptr<Backend> create_vulkan_backend();

}

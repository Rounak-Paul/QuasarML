#pragma once

#include <Backend/BackendInterface.h>
#include "VulkanTypes.h"
#include <mutex>
#include <unordered_map>
#include <thread>
#include <deque>
#include <functional>

namespace QuasarML {

struct DeletionQueue {
    std::deque<std::function<void()>> deletors;
    
    void push(std::function<void()>&& fn) { deletors.push_back(std::move(fn)); }
    void flush() {
        for (auto it = deletors.rbegin(); it != deletors.rend(); ++it) (*it)();
        deletors.clear();
    }
};

class VulkanBackend : public Backend {
public:
    VulkanBackend() = default;
    ~VulkanBackend() override;
    
    bool init(const std::string& name, u32 device_index) override;
    void shutdown() override;
    bool is_valid() const override;
    u32 get_device_index() const override { return _device_index; }
    
    BufferHandle create_storage_buffer(u64 size, bool host_visible) override;
    BufferHandle create_staging_buffer(u64 size) override;
    void destroy_buffer(BufferHandle& buffer) override;
    
    void upload_buffer(BufferHandle& buffer, const void* data, u64 size, u64 offset = 0) override;
    void download_buffer(BufferHandle& buffer, void* data, u64 size, u64 offset = 0) override;
    void copy_buffer(BufferHandle& src, BufferHandle& dst, u64 size, u64 src_offset = 0, u64 dst_offset = 0) override;
    
    PipelineHandle create_compute_pipeline(const std::string& glsl_source, u32 num_bindings, u32 push_constant_size = 0) override;
    void destroy_pipeline(PipelineHandle& pipeline) override;
    
    void execute_compute(PipelineHandle& pipeline,
                         const std::vector<BufferBinding>& buffers,
                         u32 group_x, u32 group_y = 1, u32 group_z = 1,
                         const void* push_data = nullptr, u32 push_size = 0) override;
    
    void begin_recording() override;
    void record_compute(PipelineHandle& pipeline,
                        const std::vector<BufferBinding>& buffers,
                        u32 group_x, u32 group_y = 1, u32 group_z = 1,
                        const void* push_data = nullptr, u32 push_size = 0) override;
    void end_recording() override;
    
    void synchronize() override;
    void memory_barrier() override;
    void device_wait_idle() override;
    
    ComputeLimits get_compute_limits() const override;
    u32 optimal_dispatch_1d(u32 total, u32 local_size = 256) const override;
    const DeviceCapabilities& get_capabilities() const override;

private:
    struct ThreadResources {
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool recording = false;
    };
    
    ThreadResources& get_thread_resources();
    VkDescriptorSet allocate_descriptor_set(VkDescriptorPool pool, VkDescriptorSetLayout layout);
    
    VulkanContext _ctx;
    DeletionQueue _deletion_queue;
    u32 _device_index = 0;
    
    VkCommandPool _imm_command_pool = VK_NULL_HANDLE;
    VkCommandBuffer _imm_command_buffer = VK_NULL_HANDLE;
    VkFence _imm_fence = VK_NULL_HANDLE;
    
    std::unordered_map<std::thread::id, ThreadResources> _thread_resources;
    mutable std::mutex _thread_mutex;
    mutable std::mutex _buffer_mutex;
    mutable std::mutex _descriptor_mutex;
    mutable std::mutex _queue_mutex;
};

}

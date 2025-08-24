#pragma once

#include <qspch.h>
#include <Container/DeletionQueue.h>
#include "VulkanTypes.h"

namespace QuasarML {

class VulkanBackend {
    public:
    VulkanBackend(const std::string& name="QuasarAccelerator", u32 gpu_idx=0);
    ~VulkanBackend();

    void device_wait_idle();

    // ============================================================================
    // BUFFER MANAGEMENT
    // ============================================================================
    
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocation_info = {};
        VkDeviceSize size = 0;
        void* mapped_data = nullptr;
        
        // Helper to check if buffer is valid
        bool is_valid() const { return buffer != VK_NULL_HANDLE; }
    };
    
    // Create GPU buffer (storage buffer for compute operations)
    Buffer create_storage_buffer(VkDeviceSize size, bool host_visible = false);
    
    // Create staging buffer (for CPU->GPU transfers)
    Buffer create_staging_buffer(VkDeviceSize size);
    
    // Create uniform buffer (for small constant data)
    Buffer create_uniform_buffer(VkDeviceSize size);
    
    // Destroy buffer
    void destroy_buffer(Buffer& buffer);
    
    // Upload data to buffer
    void upload_to_buffer(Buffer& buffer, const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    
    // Download data from buffer
    void download_from_buffer(Buffer& buffer, void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    
    // Copy buffer to buffer on GPU
    void copy_buffer(Buffer& src, Buffer& dst, VkDeviceSize size, VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0);
    
    // ============================================================================
    // COMPUTE PIPELINE MANAGEMENT
    // ============================================================================
    
    struct ComputePipeline {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        u32 binding_count = 0;
        
        bool is_valid() const { return pipeline != VK_NULL_HANDLE; }
    };

    struct DescriptorSetFrame {
        std::vector<VkDescriptorSet> available_sets;
        std::vector<VkDescriptorSet> used_sets;
        size_t current_set_index = 0;
    };
    
    // Create compute pipeline from GLSL source
    ComputePipeline create_compute_pipeline(const std::string& glsl_source, 
                                            u32 num_storage_buffers,
                                            u32 push_constant_size = 0);
    
    // Destroy compute pipeline
    void destroy_compute_pipeline(ComputePipeline& pipeline);
    
    // ============================================================================
    // COMMAND EXECUTION
    // ============================================================================
    
    // Execute compute shader synchronously (one-shot execution)
    void execute_compute(ComputePipeline& pipeline, u32 group_x, u32 group_y = 1, u32 group_z = 1, 
                    const void* push_constants = nullptr, u32 push_constant_size = 0,
                    const std::vector<Buffer*>& buffers = {});
    
    // Begin recording compute commands (for batched operations)
    void begin_compute_recording();
    
    // Record compute dispatch
    void record_compute_dispatch(ComputePipeline& pipeline, u32 group_x, u32 group_y = 1, u32 group_z = 1,
                            const void* push_constants = nullptr, u32 push_constant_size = 0,
                            const std::vector<Buffer*>& buffers = {});
    
    // Submit and execute recorded commands
    void execute_recorded_commands();
    
    // Wait for all compute operations to complete
    void wait_for_compute();
    
    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    
    // Get device compute limits for optimal dispatch sizes
    struct ComputeLimits {
        u32 max_work_group_size[3];
        u32 max_work_group_count[3];
        u32 max_work_group_invocations;
        u32 max_shared_memory_size;
    };
    ComputeLimits get_compute_limits();
    
    // Calculate optimal dispatch size for 1D workload
    u32 calculate_dispatch_1d(u32 total_work_items, u32 local_size = 256);
    
    // Memory barrier for compute operations
    void memory_barrier();

    private:
    bool _recording = false;  // Track if we're recording commands
    
    private:
    DeletionQueue _deletion_queue;
    VulkanContext _ctx;

    VkFence _compute_fence;
    VkCommandPool _compute_command_pool;
    VkCommandBuffer _compute_command_buffer;

    // immediate submit structures
    VkFence _imm_fence;
    VkCommandBuffer _imm_command_buffer;
    VkCommandPool _imm_command_pool;
    VkSampler _imm_sampler;
    VkDescriptorPool _immui_pool;

    static constexpr u32 MAX_FRAMES_IN_FLIGHT = 2;
    std::array<DescriptorSetFrame, MAX_FRAMES_IN_FLIGHT> _descriptor_frames;
    u32 _current_frame = 0;

    VkDescriptorSet allocate_descriptor_set(ComputePipeline& pipeline);
    void reset_descriptor_sets();

    b8 create_command_buffers();
    b8 create_sync_objects();
};
}

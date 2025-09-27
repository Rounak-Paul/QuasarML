#include "VulkanBackend.h"
#include "VulkanInitInfo.h"

#define VMA_IMPLEMENTATION
// #define VMA_DEBUG_LOG(format, ...) printf(format "\n", __VA_ARGS__)
#include "vk_mem_alloc.h"

#include <shaderc/shaderc.h>
#include <fstream>
#include <sstream>

namespace QuasarML {

std::vector<const char*> validation_layers = {
    "VK_LAYER_KHRONOS_validation"
    // ,"VK_LAYER_LUNARG_api_dump" // For all vulkan calls
};

static b8 check_validation_layer_support();
static void fetch_api_version(VulkanContext& ctx);
static b8 create_instance(const std::string& name, VulkanContext& ctx);
static void setup_debug_messenger(VulkanContext& ctx);
static b8 load_compute_shader_module(const std::string& glsl_code, VkDevice device, VkShaderModule* out_shader_module);

VulkanBackend::VulkanBackend(const std::string& name, u32 gpu_idx)
{
    if (_ctx.validation_enabled) {
        if (_ctx.validation_enabled && !check_validation_layer_support()) {
            LOG_ERROR("validation layers requested but not available");
        }
    }

    fetch_api_version(_ctx);
    
    {    
        if (!create_instance(name, _ctx)) LOG_ERROR("Vulkan instance creation failed!");
        _deletion_queue.push_function([&]() {
            vkDestroyInstance(_ctx.instance, nullptr);
        });
    }

    {    
        setup_debug_messenger(_ctx);
        _deletion_queue.push_function([&]() {
            auto destroy_func = (PFN_vkDestroyDebugUtilsMessengerEXT) 
                vkGetInstanceProcAddr(_ctx.instance, "vkDestroyDebugUtilsMessengerEXT");
            if (destroy_func != nullptr) {
                destroy_func(_ctx.instance, _ctx.debug_messenger, nullptr);
                _ctx.debug_messenger = VK_NULL_HANDLE;
            }
        });
    }

    {
        if (!vulkan_device_create(_ctx.instance, gpu_idx, _ctx.device)) {
            LOG_ERROR("Failed to create device!");
        }
        _deletion_queue.push_function([&]() {
            vulkan_device_destroy(_ctx.instance, _ctx.device);
            _ctx.device.logical_device = VK_NULL_HANDLE;
        });
    }

    {
        VmaAllocatorCreateInfo allocatorInfo = {};
        allocatorInfo.physicalDevice = _ctx.device.physical_device;
        allocatorInfo.device = _ctx.device.logical_device;
        allocatorInfo.instance = _ctx.instance;
        allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT; // literally enables c-style pointers for device memory
        if (vmaCreateAllocator(&allocatorInfo, &_ctx.allocator) != VK_SUCCESS) {
            LOG_ERROR("Failed to create memory allocator!");
        }
        _deletion_queue.push_function([&]() {
            vmaDestroyAllocator(_ctx.allocator);
        });
    }

    {    
        if (!create_command_buffers()) {
            LOG_ERROR("Failed to create command buffers!");
        }
    }

    {    if (!create_sync_objects()) {
            LOG_ERROR("Failed to create sync objects!");
        }
    }
}

VulkanBackend::~VulkanBackend()
{
    device_wait_idle();

    _deletion_queue.flush();
    
    // Reset API version info
    _ctx.api_major = 0;
    _ctx.api_minor = 0;
    _ctx.api_patch = 0;
}

bool VulkanBackend::is_valid() const {
    // Consider backend valid if logical device handle was created
    return _ctx.device.logical_device != VK_NULL_HANDLE;
}

void VulkanBackend::device_wait_idle()
{
    vkDeviceWaitIdle(_ctx.device.logical_device);
}

b8 VulkanBackend::create_command_buffers() {
    VkCommandPoolCreateInfo pool_info = command_pool_create_info(
        _ctx.device.compute_queue_index,  // Use compute queue instead of graphics
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    );

    // Single command pool for compute operations
    VK_CHECK(vkCreateCommandPool(_ctx.device.logical_device, &pool_info, nullptr, &_compute_command_pool));
    
    // Allocate one or more command buffers for compute work
    VkCommandBufferAllocateInfo cmd_info = command_buffer_allocate_info(_compute_command_pool, 1);
    VK_CHECK(vkAllocateCommandBuffers(_ctx.device.logical_device, &cmd_info, &_compute_command_buffer));

    // Optional: Keep the immediate command buffer for one-off operations
    VK_CHECK(vkCreateCommandPool(_ctx.device.logical_device, &pool_info, nullptr, &_imm_command_pool));
    VkCommandBufferAllocateInfo cmdAllocInfo = command_buffer_allocate_info(_imm_command_pool, 1);
    VK_CHECK(vkAllocateCommandBuffers(_ctx.device.logical_device, &cmdAllocInfo, &_imm_command_buffer));

    _deletion_queue.push_function([&]() {
        vkDestroyCommandPool(_ctx.device.logical_device, _compute_command_pool, nullptr);
        vkDestroyCommandPool(_ctx.device.logical_device, _imm_command_pool, nullptr);
    });

    return true;
}

b8 VulkanBackend::create_sync_objects() {
    VkFenceCreateInfo fence_info = fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    
    // Main compute fence - for waiting on compute operations to complete
    VK_CHECK(vkCreateFence(_ctx.device.logical_device, &fence_info, nullptr, &_compute_fence));
    
    // Immediate fence - for one-off operations (data uploads/downloads)
    VK_CHECK(vkCreateFence(_ctx.device.logical_device, &fence_info, nullptr, &_imm_fence));

    _deletion_queue.push_function([&]() {
        vkDestroyFence(_ctx.device.logical_device, _compute_fence, nullptr);
        vkDestroyFence(_ctx.device.logical_device, _imm_fence, nullptr);
    });

    return true;
}

VulkanBackend::Buffer VulkanBackend::create_storage_buffer(VkDeviceSize size, bool host_visible) {
    Buffer buffer = {};
    buffer.size = size;
    
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VmaAllocationCreateInfo alloc_info = {};
    if (host_visible) {
        alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else {
        alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    }
    
    VK_CHECK(vmaCreateBuffer(_ctx.allocator, &buffer_info, &alloc_info, 
                            &buffer.buffer, &buffer.allocation, &buffer.allocation_info));
    
    if (host_visible) {
        buffer.mapped_data = buffer.allocation_info.pMappedData;
    }
    
    return buffer;
}

VulkanBackend::Buffer VulkanBackend::create_staging_buffer(VkDeviceSize size) {
    Buffer buffer = {};
    buffer.size = size;
    
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    VK_CHECK(vmaCreateBuffer(_ctx.allocator, &buffer_info, &alloc_info, 
                            &buffer.buffer, &buffer.allocation, &buffer.allocation_info));
    
    buffer.mapped_data = buffer.allocation_info.pMappedData;
    return buffer;
}

VulkanBackend::Buffer VulkanBackend::create_uniform_buffer(VkDeviceSize size) {
    Buffer buffer = {};
    buffer.size = size;
    
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    VK_CHECK(vmaCreateBuffer(_ctx.allocator, &buffer_info, &alloc_info, 
                            &buffer.buffer, &buffer.allocation, &buffer.allocation_info));
    
    buffer.mapped_data = buffer.allocation_info.pMappedData;
    return buffer;
}

void VulkanBackend::destroy_buffer(Buffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(_ctx.allocator, buffer.buffer, buffer.allocation);
        buffer = {};
    }
}

void VulkanBackend::upload_to_buffer(Buffer& buffer, const void* data, VkDeviceSize size, VkDeviceSize offset) {
    if (buffer.mapped_data) {
        // Direct copy to mapped memory
        memcpy(static_cast<char*>(buffer.mapped_data) + offset, data, size);
        vmaFlushAllocation(_ctx.allocator, buffer.allocation, offset, size);
    } else {
        // Use staging buffer for GPU-only buffers
        Buffer staging = create_staging_buffer(size);
        memcpy(staging.mapped_data, data, size);
        copy_buffer(staging, buffer, size, 0, offset);
        destroy_buffer(staging);
    }
}

void VulkanBackend::download_from_buffer(Buffer& buffer, void* data, VkDeviceSize size, VkDeviceSize offset) {
    if (buffer.mapped_data) {
        // Direct copy from mapped memory
        vmaInvalidateAllocation(_ctx.allocator, buffer.allocation, offset, size);
        memcpy(data, static_cast<char*>(buffer.mapped_data) + offset, size);
    } else {
        // Use staging buffer for GPU-only buffers
        Buffer staging = create_staging_buffer(size);
        copy_buffer(buffer, staging, size, offset, 0);
        memcpy(data, staging.mapped_data, size);
        destroy_buffer(staging);
    }
}

void VulkanBackend::copy_buffer(Buffer& src, Buffer& dst, VkDeviceSize size, VkDeviceSize src_offset, VkDeviceSize dst_offset) {
    // Use immediate command buffer for buffer copies
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkResetFences(_ctx.device.logical_device, 1, &_imm_fence);
    VK_CHECK(vkBeginCommandBuffer(_imm_command_buffer, &begin_info));
    
    VkBufferCopy copy_region = {};
    copy_region.srcOffset = src_offset;
    copy_region.dstOffset = dst_offset;
    copy_region.size = size;
    
    vkCmdCopyBuffer(_imm_command_buffer, src.buffer, dst.buffer, 1, &copy_region);
    
    VK_CHECK(vkEndCommandBuffer(_imm_command_buffer));
    
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &_imm_command_buffer;
    
    VK_CHECK(vkQueueSubmit(_ctx.device.compute_queue, 1, &submit_info, _imm_fence));
    VK_CHECK(vkWaitForFences(_ctx.device.logical_device, 1, &_imm_fence, VK_TRUE, UINT64_MAX));
}

VulkanBackend::ComputePipeline VulkanBackend::create_compute_pipeline(const std::string& glsl_source, 
                                                                     u32 num_storage_buffers,
                                                                     u32 push_constant_size) {
    ComputePipeline pipeline = {};
    pipeline.binding_count = num_storage_buffers;
    
    // Create descriptor set layout
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_storage_buffers);
    for (u32 i = 0; i < num_storage_buffers; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }
    
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<u32>(bindings.size());
    layout_info.pBindings = bindings.data();
    
    VK_CHECK(vkCreateDescriptorSetLayout(_ctx.device.logical_device, &layout_info, 
                                        nullptr, &pipeline.descriptor_layout));
    
    // Create a larger descriptor pool to handle multiple allocations
    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = num_storage_buffers * 100; // Allow for 100 concurrent descriptor sets
    
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | 
                      VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pool_info.maxSets = 100; // Allow 100 concurrent descriptor sets
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    
    VK_CHECK(vkCreateDescriptorPool(_ctx.device.logical_device, &pool_info, 
                                   nullptr, &pipeline.descriptor_pool));
    
    // Don't allocate descriptor set here anymore - do it per-use
    
    // Create pipeline layout with push constants if needed
    VkPushConstantRange push_constant_range = {};
    if (push_constant_size > 0) {
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = push_constant_size;
    }
    
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &pipeline.descriptor_layout;
    pipeline_layout_info.pushConstantRangeCount = push_constant_size > 0 ? 1 : 0;
    pipeline_layout_info.pPushConstantRanges = push_constant_size > 0 ? &push_constant_range : nullptr;
    
    VK_CHECK(vkCreatePipelineLayout(_ctx.device.logical_device, &pipeline_layout_info, 
                                   nullptr, &pipeline.layout));
    
    // Shader module and pipeline creation remains the same...
    VkShaderModule shader_module;
    if (!load_compute_shader_module(glsl_source, _ctx.device.logical_device, &shader_module)) {
        LOG_ERROR("Failed to create compute shader module");
        destroy_compute_pipeline(pipeline);
        return {};
    }
    
    VkPipelineShaderStageCreateInfo stage_info = {};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module;
    stage_info.pName = "main";
    
    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = pipeline.layout;
    
    VK_CHECK(vkCreateComputePipelines(_ctx.device.logical_device, VK_NULL_HANDLE, 1, 
                                        &pipeline_info, nullptr, &pipeline.pipeline));
    
    vkDestroyShaderModule(_ctx.device.logical_device, shader_module, nullptr);
    
    return pipeline;
}

VkDescriptorSet VulkanBackend::allocate_descriptor_set(ComputePipeline& pipeline) {
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pipeline.descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &pipeline.descriptor_layout;
    
    VkDescriptorSet descriptor_set;
    VkResult result = vkAllocateDescriptorSets(_ctx.device.logical_device, &alloc_info, &descriptor_set);
    
    if (result != VK_SUCCESS) {
        // If allocation fails, reset the pool and try again
        vkResetDescriptorPool(_ctx.device.logical_device, pipeline.descriptor_pool, 0);
        result = vkAllocateDescriptorSets(_ctx.device.logical_device, &alloc_info, &descriptor_set);
        VK_CHECK(result);
    }
    
    return descriptor_set;
}

void VulkanBackend::execute_compute(ComputePipeline& pipeline, u32 group_x, u32 group_y, u32 group_z, 
                                   const void* push_constants, u32 push_constant_size,
                                   const std::vector<BufferBinding>& buffers) {
    begin_compute_recording();
    record_compute_dispatch(pipeline, group_x, group_y, group_z, push_constants, push_constant_size, buffers);
    execute_recorded_commands();
    wait_for_compute();
}

void VulkanBackend::record_compute_dispatch(ComputePipeline& pipeline, u32 group_x, u32 group_y, u32 group_z,
                                           const void* push_constants, u32 push_constant_size,
                                           const std::vector<BufferBinding>& buffers) {
    if (!_recording) {
        LOG_ERROR("Must call begin_compute_recording() first");
        return;
    }
    
    // Allocate a fresh descriptor set for this dispatch
    VkDescriptorSet descriptor_set = allocate_descriptor_set(pipeline);
    
    // Update the descriptor set with the provided buffers
    if (!buffers.empty()) {
        std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
        std::vector<VkWriteDescriptorSet> writes(buffers.size());
        
        for (size_t i = 0; i < buffers.size(); ++i) {
            auto b = buffers[i].buffer;
            if (!b) throw std::runtime_error("Null buffer binding provided");
            buffer_infos[i].buffer = b->buffer;
            buffer_infos[i].offset = buffers[i].offset;
            // If caller passed range==0 treat it as "to the end of the underlying buffer from offset".
            // Use (b->size - offset) as the descriptor range so that offset + range does not exceed buffer size.
            VkDeviceSize effective_range = 0;
            if (buffers[i].range == 0) {
                if (buffers[i].offset >= b->size) {
                    throw std::runtime_error("Buffer binding offset exceeds underlying buffer size");
                }
                effective_range = b->size - buffers[i].offset;
            } else {
                effective_range = buffers[i].range;
            }
            buffer_infos[i].range = effective_range;
            
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = static_cast<u32>(i);
            writes[i].dstArrayElement = 0;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = &buffer_infos[i];
        }

        vkUpdateDescriptorSets(_ctx.device.logical_device, static_cast<u32>(writes.size()), 
                              writes.data(), 0, nullptr);
    }
    
    vkCmdBindPipeline(_compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(_compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline.layout, 0, 1, &descriptor_set, 0, nullptr);
    
    if (push_constants && push_constant_size > 0) {
        vkCmdPushConstants(_compute_command_buffer, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 
                          0, push_constant_size, push_constants);
    }
    
    vkCmdDispatch(_compute_command_buffer, group_x, group_y, group_z);
}

void VulkanBackend::begin_compute_recording() {
    // Reset descriptor pools at the beginning of each frame
    _current_frame = (_current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    
    VK_CHECK(vkResetCommandBuffer(_compute_command_buffer, 0));
    
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    VK_CHECK(vkBeginCommandBuffer(_compute_command_buffer, &begin_info));
    _recording = true;
}

void VulkanBackend::destroy_compute_pipeline(ComputePipeline& pipeline) {
    if (pipeline.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(_ctx.device.logical_device, pipeline.pipeline, nullptr);
    }
    if (pipeline.layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(_ctx.device.logical_device, pipeline.layout, nullptr);
    }
    if (pipeline.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(_ctx.device.logical_device, pipeline.descriptor_pool, nullptr);
    }
    if (pipeline.descriptor_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(_ctx.device.logical_device, pipeline.descriptor_layout, nullptr);
    }
    pipeline = {};
}

void VulkanBackend::execute_recorded_commands() {
    if (!_recording) {
        LOG_ERROR("No commands recorded");
        return;
    }
    
    VK_CHECK(vkEndCommandBuffer(_compute_command_buffer));
    _recording = false;
    
    VK_CHECK(vkResetFences(_ctx.device.logical_device, 1, &_compute_fence));
    
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &_compute_command_buffer;
    
    VK_CHECK(vkQueueSubmit(_ctx.device.compute_queue, 1, &submit_info, _compute_fence));
}

void VulkanBackend::wait_for_compute() {
    VK_CHECK(vkWaitForFences(_ctx.device.logical_device, 1, &_compute_fence, VK_TRUE, UINT64_MAX));
}

void VulkanBackend::memory_barrier() {
    if (!_recording) {
        LOG_ERROR("Must be recording commands to insert memory barrier");
        return;
    }
    
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    
    vkCmdPipelineBarrier(_compute_command_buffer, 
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

VulkanBackend::ComputeLimits VulkanBackend::get_compute_limits() {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(_ctx.device.physical_device, &properties);
    
    ComputeLimits limits = {};
    for (int i = 0; i < 3; i++) {
        limits.max_work_group_size[i] = properties.limits.maxComputeWorkGroupSize[i];
        limits.max_work_group_count[i] = properties.limits.maxComputeWorkGroupCount[i];
    }
    limits.max_work_group_invocations = properties.limits.maxComputeWorkGroupInvocations;
    limits.max_shared_memory_size = properties.limits.maxComputeSharedMemorySize;
    
    return limits;
}

u32 VulkanBackend::calculate_dispatch_1d(u32 total_work_items, u32 local_size) {
    return (total_work_items + local_size - 1) / local_size;
}

static b8 load_compute_shader_module(const std::string& glsl_code, VkDevice device, VkShaderModule* out_shader_module) {
    shaderc_compile_options_t options = shaderc_compile_options_initialize();
    
    // Target Vulkan 1.2
    shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    shaderc_compile_options_set_source_language(options, shaderc_source_language_glsl);
    shaderc_compile_options_set_optimization_level(options, shaderc_optimization_level_performance);
    
    LOG_DEBUG("Compiling compute shader...");
    
    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    
    shaderc_compilation_result_t result = shaderc_compile_into_spv(
        compiler,
        glsl_code.c_str(),
        glsl_code.length(),
        shaderc_glsl_default_compute_shader,
        "compute_shader",
        "main",
        options
    );
    
    shaderc_compiler_release(compiler);
    shaderc_compile_options_release(options);
    
    if (!result) {
        LOG_ERROR("Unknown error occurred while compiling compute shader");
        return false;
    }
    
    shaderc_compilation_status status = shaderc_result_get_compilation_status(result);
    if (status != shaderc_compilation_status_success) {
        const char* error_message = shaderc_result_get_error_message(result);
        u64 error_count = shaderc_result_get_num_errors(result);
        LOG_ERROR("Error compiling compute shader with {} errors:", error_count);
        LOG_ERROR("Error(s):\n{}", error_message);
        // Print a snippet of the GLSL source to help debug malformed generated shaders
        size_t snippet_len = std::min<size_t>(glsl_code.size(), 2048);
        if (snippet_len > 0) {
            std::string snippet = glsl_code.substr(0, snippet_len);
                LOG_ERROR("GLSL snippet (first {} chars):\n{}", snippet_len, snippet);
                // Also write the full GLSL (or truncated snippet) to a temp file for post-mortem inspection
                try {
                    auto now = std::chrono::system_clock::now();
                    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
                    std::ostringstream fname;
                    fname << "/tmp/quasar_shader_error_" << ms << ".glsl";
                    std::ofstream ofs(fname.str(), std::ios::out | std::ios::trunc);
                    if (ofs.is_open()) {
                        ofs << glsl_code;
                        ofs.close();
                        LOG_ERROR("Wrote GLSL snippet to {}", fname.str());
                    }
                } catch (...) {
                    // best-effort only
                }
        }
        shaderc_result_release(result);
        return false;
    }
    
    LOG_DEBUG("Compute shader compiled successfully");
    
    // Output warnings if any
    u64 warning_count = shaderc_result_get_num_warnings(result);
    if (warning_count > 0) {
        LOG_WARN("{} warnings during compute shader compilation:\n{}", 
                warning_count, shaderc_result_get_error_message(result));
    }
    
    // Create shader module
    const char* bytes = shaderc_result_get_bytes(result);
    size_t result_length = shaderc_result_get_length(result);
    
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = result_length;
    create_info.pCode = reinterpret_cast<const uint32_t*>(bytes);
    
    VkResult vk_result = vkCreateShaderModule(device, &create_info, nullptr, out_shader_module);
    
    shaderc_result_release(result);
    
    if (vk_result != VK_SUCCESS) {
        LOG_ERROR("Failed to create compute shader module");
        return false;
    }
    
    return true;
}

static b8 check_validation_layer_support() 
{
    u32 layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    
    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    for (const char* layer_name : validation_layers) {
        b8 layer_found = false;

        for (const auto& layer_properties : available_layers) {
            if (strcmp(layer_name, layer_properties.layerName) == 0) {
                layer_found = true;
                break;
            }
        }
        
        if (!layer_found) {
            return false;
        }
    }

    return true;

}

static void fetch_api_version(VulkanContext& ctx) {
    u32 api_version = 0;
    vkEnumerateInstanceVersion(&api_version);
    ctx.api_major = VK_VERSION_MAJOR(api_version);
    ctx.api_minor = VK_VERSION_MINOR(api_version);
    ctx.api_patch = VK_VERSION_PATCH(api_version);
}

static std::vector<const char*> get_required_extensions(b8 validation_enabled)
{
    std::vector<const char*> extensions;
    
    #ifdef QS_PLATFORM_APPLE
    extensions.emplace_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    #endif
    
    if (validation_enabled)
        extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    
    return extensions;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL vk_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) 
{
    switch (messageSeverity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            LOG_ERROR(pCallbackData->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            LOG_WARN(pCallbackData->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            LOG_DEBUG(pCallbackData->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            LOG_DEBUG(pCallbackData->pMessage);
            break;
        default:
            LOG_TRACE(pCallbackData->pMessage);
            break;
    }
    return VK_FALSE;
}

static void populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& create_info)
{
    create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity = 
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ;
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
    create_info.messageType = 
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = vk_debug_callback;
    create_info.pUserData = nullptr;
}

static b8 create_instance(const std::string& name, VulkanContext& ctx) {
    VkApplicationInfo app_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = name.c_str();
    app_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);
    app_info.pEngineName = "Quasar Engine";
    app_info.engineVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);
    app_info.apiVersion = VK_MAKE_API_VERSION(0, ctx.api_major, ctx.api_minor, ctx.api_patch);

    auto extensions = get_required_extensions(ctx.validation_enabled);

    VkInstanceCreateInfo create_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create_info.pApplicationInfo = &app_info;
#ifdef QS_PLATFORM_APPLE
    create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    create_info.enabledExtensionCount = static_cast<u32>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
    if (ctx.validation_enabled) {
        create_info.enabledLayerCount = static_cast<u32>(validation_layers.size());
        create_info.ppEnabledLayerNames = validation_layers.data();
        populate_debug_messenger_create_info(debug_create_info);
        create_info.pNext = &debug_create_info;
    } else {
        create_info.enabledLayerCount = 0;
        create_info.pNext = nullptr;
    }

    if (vkCreateInstance(&create_info, nullptr, &ctx.instance) != VK_SUCCESS) {
        LOG_FATAL("Failed to create vulkan instance!");
        return false;
    }

    return true;
}

static VkResult create_debug_utils_messenger_ext(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger) 
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) 
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static void setup_debug_messenger(VulkanContext& ctx) {
    if (ctx.validation_enabled) {
        VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
        populate_debug_messenger_create_info(debug_create_info);
        if (create_debug_utils_messenger_ext(ctx.instance, &debug_create_info, nullptr, &ctx.debug_messenger) != VK_SUCCESS) {
            LOG_WARN("Failed to create vulkan debug messenger! Validation errors may be omitted.");
        }
    }
}
}
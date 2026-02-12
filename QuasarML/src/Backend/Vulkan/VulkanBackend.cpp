#include "VulkanBackend.h"
#include <shaderc/shaderc.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace QuasarML {

static std::vector<const char*> s_validation_layers = {"VK_LAYER_KHRONOS_validation"};

static bool check_validation_layers() {
    u32 count;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> available(count);
    vkEnumerateInstanceLayerProperties(&count, available.data());
    
    for (const char* name : s_validation_layers) {
        bool found = false;
        for (const auto& layer : available) {
            if (strcmp(name, layer.layerName) == 0) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* user) {
    
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) LOG_ERROR("{}", data->pMessage);
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) LOG_WARN("{}", data->pMessage);
    else LOG_DEBUG("{}", data->pMessage);
    return VK_FALSE;
}

static bool compile_glsl_to_spirv(const std::string& glsl, std::vector<u32>& spirv) {
    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    shaderc_compile_options_t options = shaderc_compile_options_initialize();
    
    shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    shaderc_compile_options_set_optimization_level(options, shaderc_optimization_level_performance);
    
    shaderc_compilation_result_t result = shaderc_compile_into_spv(
        compiler, glsl.c_str(), glsl.length(),
        shaderc_glsl_compute_shader, "compute", "main", options);
    
    shaderc_compile_options_release(options);
    
    if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) {
        LOG_ERROR("Shader compile error: {}", shaderc_result_get_error_message(result));
        
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::ostringstream fname;
        std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
        fname << (temp_dir / "quasar_shader_error_").string() << ms << ".glsl";
        std::ofstream ofs(fname.str());
        if (ofs) { ofs << glsl; LOG_ERROR("Wrote shader to {}", fname.str()); }
        
        shaderc_result_release(result);
        shaderc_compiler_release(compiler);
        return false;
    }
    
    size_t size = shaderc_result_get_length(result);
    const u32* code = reinterpret_cast<const u32*>(shaderc_result_get_bytes(result));
    spirv.assign(code, code + size / sizeof(u32));
    
    shaderc_result_release(result);
    shaderc_compiler_release(compiler);
    return true;
}

bool VulkanBackend::init(const std::string& name, u32 device_index) {
    _device_index = device_index;
    
    if (_ctx.validation_enabled && !check_validation_layers()) {
        LOG_WARN("Validation layers not available");
        _ctx.validation_enabled = false;
    }
    
    vkEnumerateInstanceVersion(&_ctx.api_major);
    _ctx.api_patch = VK_VERSION_PATCH(_ctx.api_major);
    _ctx.api_minor = VK_VERSION_MINOR(_ctx.api_major);
    _ctx.api_major = VK_VERSION_MAJOR(_ctx.api_major);
    
    VkApplicationInfo app_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = name.c_str();
    app_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    app_info.pEngineName = "QuasarML";
    app_info.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    app_info.apiVersion = VK_MAKE_API_VERSION(0, _ctx.api_major, _ctx.api_minor, _ctx.api_patch);
    
    std::vector<const char*> extensions;
#ifdef QS_PLATFORM_APPLE
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif
    if (_ctx.validation_enabled) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    
    VkDebugUtilsMessengerCreateInfoEXT debug_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    debug_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    debug_info.pfnUserCallback = debug_callback;
    
    VkInstanceCreateInfo instance_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app_info;
    instance_info.enabledExtensionCount = static_cast<u32>(extensions.size());
    instance_info.ppEnabledExtensionNames = extensions.data();
#ifdef QS_PLATFORM_APPLE
    instance_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    if (_ctx.validation_enabled) {
        instance_info.enabledLayerCount = static_cast<u32>(s_validation_layers.size());
        instance_info.ppEnabledLayerNames = s_validation_layers.data();
        instance_info.pNext = &debug_info;
    }
    
    if (vkCreateInstance(&instance_info, nullptr, &_ctx.instance) != VK_SUCCESS) {
        LOG_ERROR("Failed to create Vulkan instance");
        return false;
    }
    _deletion_queue.push([this]() { vkDestroyInstance(_ctx.instance, nullptr); });
    
    if (_ctx.validation_enabled) {
        auto create_fn = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(_ctx.instance, "vkCreateDebugUtilsMessengerEXT");
        if (create_fn) create_fn(_ctx.instance, &debug_info, nullptr, &_ctx.debug_messenger);
        _deletion_queue.push([this]() {
            auto destroy_fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(_ctx.instance, "vkDestroyDebugUtilsMessengerEXT");
            if (destroy_fn && _ctx.debug_messenger) destroy_fn(_ctx.instance, _ctx.debug_messenger, nullptr);
        });
    }
    
    if (!vulkan_create_device(_ctx.instance, device_index, _ctx.device)) {
        LOG_ERROR("Failed to create Vulkan device");
        return false;
    }
    _deletion_queue.push([this]() { vulkan_destroy_device(_ctx.device); });
    
    vulkan_query_capabilities(_ctx.device.physical, _ctx.device.properties, _ctx.capabilities);
    
    VmaAllocatorCreateInfo alloc_info = {};
    alloc_info.physicalDevice = _ctx.device.physical;
    alloc_info.device = _ctx.device.logical;
    alloc_info.instance = _ctx.instance;
    alloc_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    
    if (vmaCreateAllocator(&alloc_info, &_ctx.allocator) != VK_SUCCESS) {
        LOG_ERROR("Failed to create VMA allocator");
        return false;
    }
    _deletion_queue.push([this]() { vmaDestroyAllocator(_ctx.allocator); });
    
    VkCommandPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = _ctx.device.compute_queue_index;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(_ctx.device.logical, &pool_info, nullptr, &_imm_command_pool));
    _deletion_queue.push([this]() { vkDestroyCommandPool(_ctx.device.logical, _imm_command_pool, nullptr); });
    
    VkCommandBufferAllocateInfo cmd_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_info.commandPool = _imm_command_pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(_ctx.device.logical, &cmd_info, &_imm_command_buffer));
    
    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(_ctx.device.logical, &fence_info, nullptr, &_imm_fence));
    _deletion_queue.push([this]() { vkDestroyFence(_ctx.device.logical, _imm_fence, nullptr); });
    
    LOG_INFO("VulkanBackend initialized: {} (API {}.{}.{})", 
             _ctx.device.properties.deviceName, _ctx.api_major, _ctx.api_minor, _ctx.api_patch);
    return true;
}

VulkanBackend::~VulkanBackend() { shutdown(); }

void VulkanBackend::shutdown() {
    if (!_ctx.device.logical) return;
    
    vkDeviceWaitIdle(_ctx.device.logical);
    
    for (auto& [buf, alloc] : _deferred_buffers) {
        vmaDestroyBuffer(_ctx.allocator, buf, alloc);
    }
    _deferred_buffers.clear();
    
    {
        std::lock_guard<std::mutex> lock(_thread_mutex);
        for (auto& [tid, res] : _thread_resources) {
            for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
                if (res.frames[i].descriptor_pool) vkDestroyDescriptorPool(_ctx.device.logical, res.frames[i].descriptor_pool, nullptr);
                if (res.frames[i].fence) vkDestroyFence(_ctx.device.logical, res.frames[i].fence, nullptr);
            }
            if (res.command_pool) vkDestroyCommandPool(_ctx.device.logical, res.command_pool, nullptr);
        }
        _thread_resources.clear();
    }
    
    _deletion_queue.flush();
}

bool VulkanBackend::is_valid() const { return _ctx.device.logical != VK_NULL_HANDLE; }

VulkanBackend::ThreadResources& VulkanBackend::get_thread_resources() {
    std::thread::id tid = std::this_thread::get_id();
    
    {
        std::lock_guard<std::mutex> lock(_thread_mutex);
        auto it = _thread_resources.find(tid);
        if (it != _thread_resources.end()) return it->second;
    }
    
    ThreadResources res;
    
    VkCommandPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = _ctx.device.compute_queue_index;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(_ctx.device.logical, &pool_info, nullptr, &res.command_pool));
    
    VkCommandBufferAllocateInfo cmd_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_info.commandPool = res.command_pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;
    
    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    
    VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_BINDINGS * MAX_DISPATCHES_PER_BATCH * 2};
    VkDescriptorPoolCreateInfo dpool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpool_info.maxSets = MAX_DISPATCHES_PER_BATCH * 2;
    dpool_info.poolSizeCount = 1;
    dpool_info.pPoolSizes = &pool_size;
    
    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VK_CHECK(vkAllocateCommandBuffers(_ctx.device.logical, &cmd_info, &res.frames[i].command_buffer));
        VK_CHECK(vkCreateFence(_ctx.device.logical, &fence_info, nullptr, &res.frames[i].fence));
        VK_CHECK(vkCreateDescriptorPool(_ctx.device.logical, &dpool_info, nullptr, &res.frames[i].descriptor_pool));
    }
    
    std::lock_guard<std::mutex> lock(_thread_mutex);
    _thread_resources[tid] = res;
    return _thread_resources[tid];
}

BufferHandle VulkanBackend::create_storage_buffer(u64 size, bool host_visible) {
    std::lock_guard<std::mutex> lock(_buffer_mutex);
    
    VkBufferCreateInfo buf_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buf_info.size = size;
    buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VmaAllocationCreateInfo alloc = {};
    if (host_visible) {
        alloc.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else {
        alloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    }
    
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
    VK_CHECK(vmaCreateBuffer(_ctx.allocator, &buf_info, &alloc, &buffer, &allocation, &info));
    
    BufferHandle handle;
    handle.native_handle = buffer;
    handle.allocation = allocation;
    handle.size = size;
    handle.mapped = host_visible ? info.pMappedData : nullptr;
    return handle;
}

BufferHandle VulkanBackend::create_staging_buffer(u64 size) {
    VkBufferCreateInfo buf_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buf_info.size = size;
    buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VmaAllocationCreateInfo alloc = {};
    alloc.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
    VK_CHECK(vmaCreateBuffer(_ctx.allocator, &buf_info, &alloc, &buffer, &allocation, &info));
    
    BufferHandle handle;
    handle.native_handle = buffer;
    handle.allocation = allocation;
    handle.size = size;
    handle.mapped = info.pMappedData;
    return handle;
}

void VulkanBackend::destroy_buffer(BufferHandle& buffer) {
    if (!buffer.native_handle) return;
    std::lock_guard<std::mutex> lock(_buffer_mutex);
    
    auto& res = get_thread_resources();
    bool has_pending = res.recording;
    for (u32 i = 0; i < FRAMES_IN_FLIGHT && !has_pending; ++i) {
        has_pending = res.frames[i].submitted;
    }
    
    if (has_pending) {
        _deferred_buffers.push_back({
            static_cast<VkBuffer>(buffer.native_handle),
            static_cast<VmaAllocation>(buffer.allocation)
        });
    } else {
        vmaDestroyBuffer(_ctx.allocator, static_cast<VkBuffer>(buffer.native_handle), static_cast<VmaAllocation>(buffer.allocation));
    }
    buffer = {};
}

void VulkanBackend::upload_buffer(BufferHandle& buffer, const void* data, u64 size, u64 offset) {
    if (buffer.mapped) {
        memcpy(static_cast<char*>(buffer.mapped) + offset, data, size);
        vmaFlushAllocation(_ctx.allocator, static_cast<VmaAllocation>(buffer.allocation), offset, size);
    } else {
        flush_pending();
        BufferHandle staging = create_staging_buffer(size);
        memcpy(staging.mapped, data, size);
        copy_buffer(staging, buffer, size, 0, offset);
        destroy_buffer(staging);
    }
}

void VulkanBackend::download_buffer(BufferHandle& buffer, void* data, u64 size, u64 offset) {
    if (buffer.mapped) {
        flush_pending();
        vmaInvalidateAllocation(_ctx.allocator, static_cast<VmaAllocation>(buffer.allocation), offset, size);
        memcpy(data, static_cast<char*>(buffer.mapped) + offset, size);
    } else {
        flush_pending();
        BufferHandle staging = create_staging_buffer(size);
        copy_buffer(buffer, staging, size, offset, 0);
        memcpy(data, staging.mapped, size);
        destroy_buffer(staging);
    }
}

void VulkanBackend::copy_buffer(BufferHandle& src, BufferHandle& dst, u64 size, u64 src_offset, u64 dst_offset) {
    vkResetFences(_ctx.device.logical, 1, &_imm_fence);
    
    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(_imm_command_buffer, &begin));
    
    VkBufferCopy region = {src_offset, dst_offset, size};
    vkCmdCopyBuffer(_imm_command_buffer, static_cast<VkBuffer>(src.native_handle), static_cast<VkBuffer>(dst.native_handle), 1, &region);
    
    VK_CHECK(vkEndCommandBuffer(_imm_command_buffer));
    
    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &_imm_command_buffer;
    VK_CHECK(vkQueueSubmit(_ctx.device.compute_queue, 1, &submit, _imm_fence));
    VK_CHECK(vkWaitForFences(_ctx.device.logical, 1, &_imm_fence, VK_TRUE, UINT64_MAX));
}

PipelineHandle VulkanBackend::create_compute_pipeline(const std::string& glsl_source, u32 num_bindings, u32 push_constant_size) {
    std::vector<u32> spirv;
    if (!compile_glsl_to_spirv(glsl_source, spirv)) return {};
    
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
    for (u32 i = 0; i < num_bindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    
    VkDescriptorSetLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = num_bindings;
    layout_info.pBindings = bindings.data();
    
    VkDescriptorSetLayout desc_layout;
    VK_CHECK(vkCreateDescriptorSetLayout(_ctx.device.logical, &layout_info, nullptr, &desc_layout));
    
    VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, num_bindings * 10000};
    VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 10000;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    
    VkDescriptorPool desc_pool;
    VK_CHECK(vkCreateDescriptorPool(_ctx.device.logical, &pool_info, nullptr, &desc_pool));
    
    VkPushConstantRange push_range = {};
    if (push_constant_size > 0) {
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.size = push_constant_size;
    }
    
    VkPipelineLayoutCreateInfo pipe_layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipe_layout_info.setLayoutCount = 1;
    pipe_layout_info.pSetLayouts = &desc_layout;
    pipe_layout_info.pushConstantRangeCount = push_constant_size > 0 ? 1 : 0;
    pipe_layout_info.pPushConstantRanges = push_constant_size > 0 ? &push_range : nullptr;
    
    VkPipelineLayout pipe_layout;
    VK_CHECK(vkCreatePipelineLayout(_ctx.device.logical, &pipe_layout_info, nullptr, &pipe_layout));
    
    VkShaderModuleCreateInfo shader_info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = spirv.size() * sizeof(u32);
    shader_info.pCode = spirv.data();
    
    VkShaderModule shader;
    VK_CHECK(vkCreateShaderModule(_ctx.device.logical, &shader_info, nullptr, &shader));
    
    VkPipelineShaderStageCreateInfo stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    
    VkComputePipelineCreateInfo pipe_info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipe_info.stage = stage;
    pipe_info.layout = pipe_layout;
    
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(_ctx.device.logical, VK_NULL_HANDLE, 1, &pipe_info, nullptr, &pipeline));
    
    vkDestroyShaderModule(_ctx.device.logical, shader, nullptr);
    
    PipelineHandle handle;
    handle.pipeline = pipeline;
    handle.layout = pipe_layout;
    handle.descriptor_layout = desc_layout;
    handle.descriptor_pool = desc_pool;
    handle.binding_count = num_bindings;
    return handle;
}

void VulkanBackend::destroy_pipeline(PipelineHandle& pipeline) {
    if (pipeline.pipeline) vkDestroyPipeline(_ctx.device.logical, static_cast<VkPipeline>(pipeline.pipeline), nullptr);
    if (pipeline.layout) vkDestroyPipelineLayout(_ctx.device.logical, static_cast<VkPipelineLayout>(pipeline.layout), nullptr);
    if (pipeline.descriptor_pool) vkDestroyDescriptorPool(_ctx.device.logical, static_cast<VkDescriptorPool>(pipeline.descriptor_pool), nullptr);
    if (pipeline.descriptor_layout) vkDestroyDescriptorSetLayout(_ctx.device.logical, static_cast<VkDescriptorSetLayout>(pipeline.descriptor_layout), nullptr);
    pipeline = {};
}

VkDescriptorSet VulkanBackend::allocate_descriptor_set(VkDescriptorPool pool, VkDescriptorSetLayout layout) {
    std::lock_guard<std::mutex> lock(_descriptor_mutex);
    
    VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &layout;
    
    VkDescriptorSet set;
    VK_CHECK(vkAllocateDescriptorSets(_ctx.device.logical, &alloc, &set));
    
    return set;
}

void VulkanBackend::retire_frame(FrameData& frame) {
    if (frame.descriptor_pool) {
        vkResetDescriptorPool(_ctx.device.logical, frame.descriptor_pool, 0);
    }
}

void VulkanBackend::ensure_recording(ThreadResources& res) {
    if (res.recording) return;
    
    FrameData& frame = res.frames[res.current_frame];
    
    if (frame.submitted) {
        VK_CHECK(vkWaitForFences(_ctx.device.logical, 1, &frame.fence, VK_TRUE, UINT64_MAX));
        frame.submitted = false;
        retire_frame(frame);
    }
    
    VK_CHECK(vkResetCommandBuffer(frame.command_buffer, 0));
    
    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(frame.command_buffer, &begin));
    
    res.recording = true;
    res.dispatch_count = 0;
}

void VulkanBackend::flush_frame(ThreadResources& res) {
    if (!res.recording) return;
    
    FrameData& frame = res.frames[res.current_frame];
    
    VK_CHECK(vkEndCommandBuffer(frame.command_buffer));
    res.recording = false;
    
    VK_CHECK(vkResetFences(_ctx.device.logical, 1, &frame.fence));
    
    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &frame.command_buffer;
    
    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        VK_CHECK(vkQueueSubmit(_ctx.device.compute_queue, 1, &submit, frame.fence));
    }
    
    frame.submitted = true;
    res.dispatch_count = 0;
    res.current_frame = (res.current_frame + 1) % FRAMES_IN_FLIGHT;
}

void VulkanBackend::record_dispatch(ThreadResources& res, PipelineHandle& pipeline,
                                     const BufferBinding* buffers, u32 buffer_count,
                                     u32 group_x, u32 group_y, u32 group_z,
                                     const void* push_data, u32 push_size) {
    FrameData& frame = res.frames[res.current_frame];
    
    if (res.dispatch_count > 0) {
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(frame.command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    
    VkDescriptorSet desc_set = allocate_descriptor_set(
        frame.descriptor_pool,
        static_cast<VkDescriptorSetLayout>(pipeline.descriptor_layout));
    
    if (buffer_count > 0) {
        VkDescriptorBufferInfo buf_infos[MAX_BINDINGS];
        VkWriteDescriptorSet writes[MAX_BINDINGS];
        
        for (u32 i = 0; i < buffer_count; ++i) {
            buf_infos[i].buffer = static_cast<VkBuffer>(buffers[i].buffer->native_handle);
            buf_infos[i].offset = buffers[i].offset;
            buf_infos[i].range = buffers[i].range ? buffers[i].range : (buffers[i].buffer->size - buffers[i].offset);
            
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(_ctx.device.logical, buffer_count, writes, 0, nullptr);
    }
    
    vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, static_cast<VkPipeline>(pipeline.pipeline));
    vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            static_cast<VkPipelineLayout>(pipeline.layout), 0, 1, &desc_set, 0, nullptr);
    
    if (push_data && push_size > 0) {
        vkCmdPushConstants(frame.command_buffer, static_cast<VkPipelineLayout>(pipeline.layout),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push_data);
    }
    
    vkCmdDispatch(frame.command_buffer, group_x, group_y, group_z);
    res.dispatch_count++;
}

void VulkanBackend::begin_recording() {
    auto& res = get_thread_resources();
    if (res.recording && !res.explicit_batch) {
        flush_frame(res);
    }
    ensure_recording(res);
    res.explicit_batch = true;
}

void VulkanBackend::record_compute(PipelineHandle& pipeline, const std::vector<BufferBinding>& buffers,
                                   u32 group_x, u32 group_y, u32 group_z,
                                   const void* push_data, u32 push_size) {
    auto& res = get_thread_resources();
    if (!res.recording) { LOG_ERROR("Must call begin_recording first"); return; }
    
    record_dispatch(res, pipeline, buffers.data(), static_cast<u32>(buffers.size()),
                    group_x, group_y, group_z, push_data, push_size);
}

void VulkanBackend::end_recording() {
    auto& res = get_thread_resources();
    if (!res.recording) return;
    
    res.explicit_batch = false;
    flush_frame(res);
}

void VulkanBackend::execute_compute(PipelineHandle& pipeline, const std::vector<BufferBinding>& buffers,
                                    u32 group_x, u32 group_y, u32 group_z,
                                    const void* push_data, u32 push_size) {
    auto& res = get_thread_resources();
    ensure_recording(res);
    record_dispatch(res, pipeline, buffers.data(), static_cast<u32>(buffers.size()),
                    group_x, group_y, group_z, push_data, push_size);
    
    if (res.dispatch_count >= MAX_DISPATCHES_PER_BATCH) {
        flush_frame(res);
    }
}

void VulkanBackend::flush_pending() {
    auto& res = get_thread_resources();
    if (!res.recording || res.explicit_batch) return;
    
    flush_frame(res);
    
    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        if (res.frames[i].submitted) {
            VK_CHECK(vkWaitForFences(_ctx.device.logical, 1, &res.frames[i].fence, VK_TRUE, UINT64_MAX));
            res.frames[i].submitted = false;
            retire_frame(res.frames[i]);
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(_buffer_mutex);
        for (auto& [buf, alloc] : _deferred_buffers) {
            vmaDestroyBuffer(_ctx.allocator, buf, alloc);
        }
        _deferred_buffers.clear();
    }
}

void VulkanBackend::synchronize() {
    auto& res = get_thread_resources();
    
    if (res.recording && !res.explicit_batch) {
        flush_frame(res);
    }
    
    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        if (res.frames[i].submitted) {
            VK_CHECK(vkWaitForFences(_ctx.device.logical, 1, &res.frames[i].fence, VK_TRUE, UINT64_MAX));
            res.frames[i].submitted = false;
            retire_frame(res.frames[i]);
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(_buffer_mutex);
        for (auto& [buf, alloc] : _deferred_buffers) {
            vmaDestroyBuffer(_ctx.allocator, buf, alloc);
        }
        _deferred_buffers.clear();
    }
}

void VulkanBackend::memory_barrier() {
    auto& res = get_thread_resources();
    if (!res.recording) return;
    
    FrameData& frame = res.frames[res.current_frame];
    
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    
    vkCmdPipelineBarrier(frame.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void VulkanBackend::device_wait_idle() { vkDeviceWaitIdle(_ctx.device.logical); }

ComputeLimits VulkanBackend::get_compute_limits() const {
    ComputeLimits limits = {};
    for (int i = 0; i < 3; ++i) {
        limits.max_workgroup_size[i] = _ctx.device.properties.limits.maxComputeWorkGroupSize[i];
        limits.max_workgroup_count[i] = _ctx.device.properties.limits.maxComputeWorkGroupCount[i];
    }
    limits.max_workgroup_invocations = _ctx.device.properties.limits.maxComputeWorkGroupInvocations;
    limits.max_shared_memory = _ctx.device.properties.limits.maxComputeSharedMemorySize;
    return limits;
}

u32 VulkanBackend::optimal_dispatch_1d(u32 total, u32 local_size) const {
    return (total + local_size - 1) / local_size;
}

const DeviceCapabilities& VulkanBackend::get_capabilities() const {
    return _ctx.capabilities;
}

std::unique_ptr<Backend> create_vulkan_backend() {
    return std::make_unique<VulkanBackend>();
}

}

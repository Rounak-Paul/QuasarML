#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_mem_alloc.h>
#include <Common/Types.h>
#include <Common/Logger.h>
#include <Backend/DeviceCapabilities.h>

namespace QuasarML {

#define VK_CHECK(x) \
    do { \
        VkResult err__ = (x); \
        if (err__ != VK_SUCCESS) { \
            LOG_ERROR("Vulkan error: {}", string_VkResult(err__)); \
            abort(); \
        } \
    } while (0)

struct VulkanDevice {
    u32 api_major = 0;
    u32 api_minor = 0;
    u32 api_patch = 0;
    
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice logical = VK_NULL_HANDLE;
    
    i32 compute_queue_index = -1;
    i32 transfer_queue_index = -1;
    
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkQueue transfer_queue = VK_NULL_HANDLE;
    
    VkPhysicalDeviceProperties properties = {};
    VkPhysicalDeviceFeatures features = {};
    VkPhysicalDeviceMemoryProperties memory = {};
    VkPhysicalDeviceSubgroupProperties subgroup_properties = {};
};

struct VulkanContext {
    u32 api_major = 0;
    u32 api_minor = 0;
    u32 api_patch = 0;
    
    bool validation_enabled = true;
    
    VkInstance instance = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    
    VulkanDevice device;
    DeviceCapabilities capabilities = {};
};

struct VulkanPhysicalDeviceInfo {
    VkPhysicalDevice device;
    VkPhysicalDeviceProperties properties;
    i32 compute_queue_index;
    bool meets_requirements;
};

std::vector<VulkanPhysicalDeviceInfo> vulkan_enumerate_devices(VkInstance instance);
bool vulkan_create_device(VkInstance instance, u32 device_index, VulkanDevice& device);
void vulkan_destroy_device(VulkanDevice& device);
void vulkan_query_capabilities(VkPhysicalDevice physical, const VkPhysicalDeviceProperties& props, DeviceCapabilities& caps);

}

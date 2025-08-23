#pragma once

#include <qspch.h>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_mem_alloc.h>

#include "VulkanDevice.h"

namespace QuasarML {

#define VK_CHECK(x)                                                           \
    do {                                                                     \
        VkResult err__ = (x);                                                \
        if (err__ != VK_SUCCESS) {                                           \
            LOG_ERROR("Detected Vulkan error: {}", string_VkResult(err__));  \
            abort();                                                         \
        }                                                                    \
    } while (0)

typedef struct VulkanContext {
    u32 api_major; // The instance-level api major version.
    u32 api_minor; // The instance-level api minor version.
    u32 api_patch; // The instance-level api patch version.

    b8 validation_enabled = true;
    
    VkInstance instance;
    VmaAllocator allocator;
    VkDebugUtilsMessengerEXT debug_messenger;

    VkSurfaceKHR surface;
    VulkanDevice device;
} VulkanContext;

}
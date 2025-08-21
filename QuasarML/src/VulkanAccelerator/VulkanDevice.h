#pragma once
#include <qspch.h>
#include <vulkan/vulkan.h>

namespace QuasarML {

/** @brief Bitwise flags for device support. */
typedef u32 VulkanDeviceSupportFlags;

typedef enum VulkanDeviceSupportFlagsBits {
    VULKAN_DEVICE_SUPPORT_FLAG_NONE_BIT = 0x00,
    /** @brief Indicates if the device supports native sync2 (i.e. using Vulkan API >= 1.3). */
    VULKAN_DEVICE_SUPPORT_FLAG_NATIVE_13_FEATURES_BIT = 0x01,
    VULKAN_DEVICE_SUPPORT_FLAG_NATIVE_12_FEATURES_BIT = 0x02,
} VulkanDeviceSupportFlagsBits;

typedef struct VulkanDevice {
    /** @brief The supported device-level api major version. */
    u32 api_major;
    /** @brief The supported device-level api minor version. */
    u32 api_minor;
    /** @brief The supported device-level api patch version. */
    u32 api_patch;

    VkPhysicalDevice physical_device;
    VkDevice logical_device;

    i32 compute_queue_index;
    i32 transfer_queue_index;
    
    b8 supports_device_local_host_visible;

    VkQueue compute_queue;
    VkQueue transfer_queue;

    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceFeatures features;
    VkPhysicalDeviceMemoryProperties memory;

    /** @brief Indicates support for various features. */
    VulkanDeviceSupportFlags support_flags;

    // Sync2 extension function pointers (if not native)
    PFN_vkCmdPipelineBarrier2KHR vkCmdPipelineBarrier2KHR;
    PFN_vkCmdWriteTimestamp2KHR vkCmdWriteTimestamp2KHR;
    PFN_vkQueueSubmit2KHR vkQueueSubmit2KHR;
    PFN_vkCmdWaitEvents2KHR vkCmdWaitEvents2KHR;
    PFN_vkCmdSetEvent2KHR vkCmdSetEvent2KHR;
    PFN_vkCmdResetEvent2KHR vkCmdResetEvent2KHR;
    
    // Copy commands for buffer operations
    PFN_vkCmdCopyBuffer2KHR vkCmdCopyBuffer2KHR;
    PFN_vkCmdCopyImage2KHR vkCmdCopyImage2KHR;
    PFN_vkCmdCopyBufferToImage2KHR vkCmdCopyBufferToImage2KHR;
    PFN_vkCmdCopyImageToBuffer2KHR vkCmdCopyImageToBuffer2KHR;
} VulkanDevice;

b8 vulkan_device_create(VkInstance instance, VulkanDevice& device);
void vulkan_device_destroy(VkInstance instance, VulkanDevice& device);

}
#include "VulkanTypes.h"
#include <vector>
#include <cstring>

namespace QuasarML {

std::vector<VulkanPhysicalDeviceInfo> vulkan_enumerate_devices(VkInstance instance) {
    std::vector<VulkanPhysicalDeviceInfo> infos;
    
    u32 count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) return infos;
    
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    
    for (u32 i = 0; i < count; ++i) {
        VulkanPhysicalDeviceInfo info = {};
        info.device = devices[i];
        vkGetPhysicalDeviceProperties(devices[i], &info.properties);
        
        u32 queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queue_count, queues.data());
        
        info.compute_queue_index = -1;
        for (u32 j = 0; j < queue_count; ++j) {
            if (queues[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                info.compute_queue_index = j;
                break;
            }
        }
        
        info.meets_requirements = (info.compute_queue_index >= 0);
        infos.push_back(info);
    }
    
    return infos;
}

bool vulkan_create_device(VkInstance instance, u32 device_index, VulkanDevice& device) {
    auto infos = vulkan_enumerate_devices(instance);
    if (device_index >= infos.size()) {
        LOG_ERROR("Device index {} out of range", device_index);
        return false;
    }
    
    const auto& info = infos[device_index];
    if (!info.meets_requirements) {
        LOG_ERROR("Device {} does not meet requirements", device_index);
        return false;
    }
    
    device.physical = info.device;
    device.properties = info.properties;
    device.compute_queue_index = info.compute_queue_index;
    
    device.api_major = VK_VERSION_MAJOR(info.properties.apiVersion);
    device.api_minor = VK_VERSION_MINOR(info.properties.apiVersion);
    device.api_patch = VK_VERSION_PATCH(info.properties.apiVersion);
    
    vkGetPhysicalDeviceFeatures(device.physical, &device.features);
    vkGetPhysicalDeviceMemoryProperties(device.physical, &device.memory);
    
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = device.compute_queue_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    
    std::vector<const char*> extensions = {
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME
    };
    
    if (device.api_minor < 3) extensions.push_back("VK_KHR_synchronization2");
    if (device.api_minor < 2) extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    
#ifdef QS_PLATFORM_APPLE
    extensions.push_back("VK_KHR_portability_subset");
#endif
    
    VkPhysicalDeviceBufferDeviceAddressFeatures bda = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bda.bufferDeviceAddress = VK_TRUE;
    
    VkPhysicalDeviceSynchronization2Features sync2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    sync2.synchronization2 = VK_TRUE;
    sync2.pNext = &bda;
    
    VkPhysicalDeviceDescriptorIndexingFeatures indexing = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT};
    indexing.descriptorBindingPartiallyBound = VK_TRUE;
    indexing.pNext = &sync2;
    
    VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &indexing;
    
    VkDeviceCreateInfo create_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_info;
    create_info.enabledExtensionCount = static_cast<u32>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    create_info.pNext = &features2;
    
    VK_CHECK(vkCreateDevice(device.physical, &create_info, nullptr, &device.logical));
    
    vkGetDeviceQueue(device.logical, device.compute_queue_index, 0, &device.compute_queue);
    device.transfer_queue = device.compute_queue;
    device.transfer_queue_index = device.compute_queue_index;
    
    LOG_INFO("Created Vulkan device: {}", device.properties.deviceName);
    return true;
}

void vulkan_destroy_device(VulkanDevice& device) {
    if (device.logical) {
        vkDestroyDevice(device.logical, nullptr);
        device.logical = VK_NULL_HANDLE;
    }
    device.physical = VK_NULL_HANDLE;
    device.compute_queue = VK_NULL_HANDLE;
    device.transfer_queue = VK_NULL_HANDLE;
}

}

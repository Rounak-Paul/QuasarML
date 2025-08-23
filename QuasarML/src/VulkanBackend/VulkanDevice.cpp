#include "VulkanDevice.h"
#include "VulkanTypes.h"

namespace QuasarML
{

typedef struct VulkanPhysicalDeviceRequirements {
    b8 compute;
    std::vector<const char*> device_extension_names;
} VulkanPhysicalDeviceRequirements;

static b8 PhysicalDeviceMeetsRequirements(
    VkPhysicalDevice device,
    const VkPhysicalDeviceProperties* properties,
    const VkPhysicalDeviceFeatures* features,
    const VulkanPhysicalDeviceRequirements* requirements,
    u32* out_compute_queue_index);

std::vector<VulkanPhysicalDeviceInfo> vulkan_query_physical_devices(VkInstance instance) {
    std::vector<VulkanPhysicalDeviceInfo> device_infos;
    
    uint32_t physical_device_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr));
    if (physical_device_count == 0) {
        LOG_WARN("No devices which support Vulkan were found.");
        return device_infos;
    }

    std::vector<VkPhysicalDevice> physical_devices;
    physical_devices.resize(physical_device_count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data()));
    
    device_infos.reserve(physical_device_count);
    
    VulkanPhysicalDeviceRequirements requirements = {};
    requirements.compute = true;
    requirements.device_extension_names = {};
    
    // Essential extensions for compute workloads
    requirements.device_extension_names.emplace_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    requirements.device_extension_names.emplace_back(VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME);

#ifdef QS_PLATFORM_APPLE
    requirements.device_extension_names.emplace_back("VK_KHR_portability_subset");
#endif
    
    for (u32 i = 0; i < physical_device_count; ++i) {
        VulkanPhysicalDeviceInfo info = {};
        info.device = physical_devices[i];
        
        vkGetPhysicalDeviceProperties(physical_devices[i], &info.properties);
        vkGetPhysicalDeviceFeatures(physical_devices[i], &info.features);
        vkGetPhysicalDeviceMemoryProperties(physical_devices[i], &info.memory);

        // Check if device supports local/host visible combo
        info.supports_device_local_host_visible = false;
        for (u32 j = 0; j < info.memory.memoryTypeCount; ++j) {
            if (((info.memory.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) &&
                ((info.memory.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)) {
                info.supports_device_local_host_visible = true;
                break;
            }
        }

        // Add sync2 if not native for this device
        auto temp_requirements = requirements;
        if (VK_VERSION_MAJOR(info.properties.apiVersion) < 1 || 
            (VK_VERSION_MAJOR(info.properties.apiVersion) == 1 && VK_VERSION_MINOR(info.properties.apiVersion) < 3)) {
            temp_requirements.device_extension_names.emplace_back("VK_KHR_synchronization2");
        }

        // Add buffer device address extension if needed
        if (VK_VERSION_MAJOR(info.properties.apiVersion) < 1 || 
            (VK_VERSION_MAJOR(info.properties.apiVersion) == 1 && VK_VERSION_MINOR(info.properties.apiVersion) < 2)) {
            temp_requirements.device_extension_names.emplace_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        }

        info.meets_requirements = PhysicalDeviceMeetsRequirements(
            physical_devices[i],
            &info.properties,
            &info.features,
            &temp_requirements,
            &info.compute_queue_index);

        device_infos.push_back(info);
    }

    // Log available devices
    LOG_DEBUG("Found {} Vulkan device(s):", physical_device_count);
    for (u32 i = 0; i < device_infos.size(); ++i) {
        const auto& info = device_infos[i];
        const char* device_type_str = "Unknown";
        switch (info.properties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: device_type_str = "Integrated GPU"; break;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: device_type_str = "Discrete GPU"; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: device_type_str = "Virtual GPU"; break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU: device_type_str = "CPU"; break;
            default: break;
        }
        
        f32 memory_size_gib = 0.0f;
        for (u32 j = 0; j < info.memory.memoryHeapCount; ++j) {
            if (info.memory.memoryHeaps[j].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                memory_size_gib = (f32)info.memory.memoryHeaps[j].size / 1024.0f / 1024.0f / 1024.0f;
                break;
            }
        }
        
        LOG_DEBUG("  [{}] {} ({}) - {:.1f} GiB - Requirements: {}", 
                  i, info.properties.deviceName, device_type_str, memory_size_gib,
                  info.meets_requirements ? "Met" : "Not Met");
    }
    
    return device_infos;
}

static b8 PhysicalDeviceMeetsRequirements(
    VkPhysicalDevice device,
    const VkPhysicalDeviceProperties* properties,
    const VkPhysicalDeviceFeatures* features,
    const VulkanPhysicalDeviceRequirements* requirements,
    u32* out_compute_queue_index) {
    
    *out_compute_queue_index = UINT32_MAX;

    u32 queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

    // Find compute queue
    for (u32 i = 0; i < queue_family_count; ++i) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            *out_compute_queue_index = i;
            break;
        }
    }

    if (requirements->compute && *out_compute_queue_index == UINT32_MAX) {
        LOG_DEBUG("Device '{}' doesn't support compute queues.", properties->deviceName);
        return false;
    }

    // Check buffer device address support
    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceFeatures2 device_features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    device_features2.pNext = &buffer_device_address_features;
    
    vkGetPhysicalDeviceFeatures2(device, &device_features2);
    
    if (!buffer_device_address_features.bufferDeviceAddress) {
        LOG_DEBUG("Device '{}' doesn't support buffer device address feature.", properties->deviceName);
        return false;
    }

    // Check device extensions
    if (!requirements->device_extension_names.empty()) {
        u32 available_extension_count = 0;
        VK_CHECK(vkEnumerateDeviceExtensionProperties(
            device, nullptr, &available_extension_count, nullptr));
        
        std::vector<VkExtensionProperties> available_extensions(available_extension_count);
        if (available_extension_count != 0) {
            VK_CHECK(vkEnumerateDeviceExtensionProperties(
                device, nullptr, &available_extension_count, available_extensions.data()));

            for (const char* required_ext : requirements->device_extension_names) {
                b8 found = false;
                for (const auto& available_ext : available_extensions) {
                    if (strcmp(required_ext, available_ext.extensionName) == 0) {
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    LOG_DEBUG("Device '{}' missing required extension: '{}'", 
                             properties->deviceName, required_ext);
                    return false;
                }
            }
        }
    }

    return true;
}

b8 vulkan_device_create(VkInstance instance, u32 device_index, VulkanDevice& device) {
    device = {};
    
    // Query available devices
    auto device_infos = vulkan_query_physical_devices(instance);
    
    if (device_infos.empty()) {
        LOG_FATAL("No Vulkan devices found.");
        return false;
    }
    
    if (device_index >= device_infos.size()) {
        LOG_FATAL("Device index {} is out of range. Found {} device(s).", 
                  device_index, device_infos.size());
        return false;
    }
    
    const auto& selected_device_info = device_infos[device_index];
    
    if (!selected_device_info.meets_requirements) {
        LOG_FATAL("Device at index {} does not meet requirements for compute workloads.", device_index);
        return false;
    }

    LOG_DEBUG("Creating logical device using device [{}]: '{}'", 
              device_index, selected_device_info.properties.deviceName);

    // Fill device structure with selected device info
    device.physical_device = selected_device_info.device;
    device.compute_queue_index = selected_device_info.compute_queue_index;
    device.properties = selected_device_info.properties;
    device.features = selected_device_info.features;
    device.memory = selected_device_info.memory;
    device.supports_device_local_host_visible = selected_device_info.supports_device_local_host_visible;

    // Save off the device-supported API version
    device.api_major = VK_VERSION_MAJOR(device.properties.apiVersion);
    device.api_minor = VK_VERSION_MINOR(device.properties.apiVersion);
    device.api_patch = VK_VERSION_PATCH(device.properties.apiVersion);

    LOG_DEBUG("Selected GPU type: {}", [&]() -> const char* {
        switch (device.properties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "Integrated";
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "Discrete";
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "Virtual";
            case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
            default: return "Unknown";
        }
    }());

    LOG_DEBUG("Vulkan API version: {}.{}.{}", device.api_major, device.api_minor, device.api_patch);

    // Create logical device with compute queue only
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos(1);
    f32 queue_priority = 1.0f;
    
    queue_create_infos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_infos[0].queueFamilyIndex = device.compute_queue_index;
    queue_create_infos[0].queueCount = 1;
    queue_create_infos[0].flags = 0;
    queue_create_infos[0].pNext = nullptr;
    queue_create_infos[0].pQueuePriorities = &queue_priority;

    std::vector<const char*> extension_names = {
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME
    };

    // Check API version support and add extensions if needed
    if (device.api_major >= 1 && device.api_minor >= 3) {
        device.support_flags |= VULKAN_DEVICE_SUPPORT_FLAG_NATIVE_13_FEATURES_BIT;
        LOG_DEBUG("Device supports Vulkan 1.3+ - using native features");
    } else if (device.api_major >= 1 && device.api_minor >= 2) {
        device.support_flags |= VULKAN_DEVICE_SUPPORT_FLAG_NATIVE_12_FEATURES_BIT;
        extension_names.push_back("VK_KHR_synchronization2");
        // Buffer device address is core in 1.2, but add extension for safety
        LOG_DEBUG("Device supports Vulkan 1.2 - buffer device address is core");
    } else {
        extension_names.push_back("VK_KHR_synchronization2");
        extension_names.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        LOG_DEBUG("Device supports Vulkan 1.1 - adding buffer device address extension");
    }

#ifdef QS_PLATFORM_APPLE
    extension_names.push_back("VK_KHR_portability_subset");
#endif

    // Required features for compute workloads
    VkPhysicalDeviceFeatures2 device_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    
    VkPhysicalDeviceDescriptorIndexingFeatures descriptor_indexing_features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT};
    descriptor_indexing_features.descriptorBindingPartiallyBound = VK_TRUE;

    VkPhysicalDeviceSynchronization2Features sync2_features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    sync2_features.synchronization2 = VK_TRUE;

    // Add buffer device address features
    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    buffer_device_address_features.bufferDeviceAddress = VK_TRUE;

    // Chain the features: buffer_device_address -> sync2 -> descriptor_indexing
    buffer_device_address_features.pNext = &sync2_features;
    sync2_features.pNext = &descriptor_indexing_features;
    device_features.pNext = &buffer_device_address_features;

    VkDeviceCreateInfo device_create_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = queue_create_infos.data();
    device_create_info.pEnabledFeatures = nullptr;
    device_create_info.enabledExtensionCount = static_cast<u32>(extension_names.size());
    device_create_info.ppEnabledExtensionNames = extension_names.data();
    device_create_info.enabledLayerCount = 0;
    device_create_info.ppEnabledLayerNames = nullptr;
    device_create_info.pNext = &device_features;

    VK_CHECK(vkCreateDevice(device.physical_device, &device_create_info, nullptr, &device.logical_device));
    LOG_DEBUG("Logical device created with buffer device address support.");

    // Get compute queue (transfer queue will be the same as compute for simplicity)
    vkGetDeviceQueue(device.logical_device, device.compute_queue_index, 0, &device.compute_queue);
    device.transfer_queue = device.compute_queue;  // Same queue for transfers
    device.transfer_queue_index = device.compute_queue_index;
    LOG_DEBUG("Compute queue obtained.");

    // Load function pointers for extensions if needed
    if (device.support_flags & VULKAN_DEVICE_SUPPORT_FLAG_NATIVE_13_FEATURES_BIT) {
        LOG_DEBUG("Vulkan device supports native sync2 and copy commands.");
    } else {
        LOG_DEBUG("Vulkan device doesn't support native sync2, loading extensions.");
        
        // Sync2 extension functions
        device.vkCmdPipelineBarrier2KHR = (PFN_vkCmdPipelineBarrier2KHR)
            vkGetDeviceProcAddr(device.logical_device, "vkCmdPipelineBarrier2KHR");
        device.vkCmdWriteTimestamp2KHR = (PFN_vkCmdWriteTimestamp2KHR)
            vkGetDeviceProcAddr(device.logical_device, "vkCmdWriteTimestamp2KHR");
        device.vkQueueSubmit2KHR = (PFN_vkQueueSubmit2KHR)
            vkGetDeviceProcAddr(device.logical_device, "vkQueueSubmit2KHR");
        device.vkCmdWaitEvents2KHR = (PFN_vkCmdWaitEvents2KHR)
            vkGetDeviceProcAddr(device.logical_device, "vkCmdWaitEvents2KHR");
        device.vkCmdSetEvent2KHR = (PFN_vkCmdSetEvent2KHR)
            vkGetDeviceProcAddr(device.logical_device, "vkCmdSetEvent2KHR");
        device.vkCmdResetEvent2KHR = (PFN_vkCmdResetEvent2KHR)
            vkGetDeviceProcAddr(device.logical_device, "vkCmdResetEvent2KHR");
        
        // Copy commands
        device.vkCmdCopyBuffer2KHR = (PFN_vkCmdCopyBuffer2KHR)
            vkGetDeviceProcAddr(device.logical_device, "vkCmdCopyBuffer2KHR");
        device.vkCmdCopyImage2KHR = (PFN_vkCmdCopyImage2KHR)
            vkGetDeviceProcAddr(device.logical_device, "vkCmdCopyImage2KHR");
        device.vkCmdCopyBufferToImage2KHR = (PFN_vkCmdCopyBufferToImage2KHR)
            vkGetDeviceProcAddr(device.logical_device, "vkCmdCopyBufferToImage2KHR");
        device.vkCmdCopyImageToBuffer2KHR = (PFN_vkCmdCopyImageToBuffer2KHR)
            vkGetDeviceProcAddr(device.logical_device, "vkCmdCopyImageToBuffer2KHR");
    }

    return true;
}

void vulkan_device_destroy(VkInstance instance, VulkanDevice& device) {
    LOG_DEBUG("Destroying vulkan_device...");
    
    if (device.logical_device == VK_NULL_HANDLE) {
        LOG_WARN("Device logical_device is null, skipping device destruction.");
        return;
    }

    // Clear queue handles
    device.compute_queue = VK_NULL_HANDLE;
    device.transfer_queue = VK_NULL_HANDLE;

    // Destroy logical device
    LOG_DEBUG("Destroying logical device...");
    try {
        vkDestroyDevice(device.logical_device, nullptr);
        device.logical_device = VK_NULL_HANDLE;
        LOG_DEBUG("Logical device destroyed successfully.");
    } catch (...) {
        LOG_ERROR("Exception occurred while destroying logical device!");
        device.logical_device = VK_NULL_HANDLE;
    }

    // Physical devices are not destroyed
    device.physical_device = VK_NULL_HANDLE;

    // Reset queue indices
    device.compute_queue_index = UINT32_MAX;
    device.transfer_queue_index = UINT32_MAX;
}

} // namespace QuasarML
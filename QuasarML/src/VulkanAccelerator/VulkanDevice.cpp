#include "VulkanDevice.h"
#include "VulkanTypes.h"

namespace QuasarML
{

typedef struct VulkanPhysicalDeviceRequirements {
    b8 compute;
    b8 transfer;
    std::vector<const char*> device_extension_names;
    b8 discrete_gpu;
} VulkanPhysicalDeviceRequirements;

typedef struct VulkanPhysicalDeviceQueueFamilyInfo {
    u32 compute_family_index;
    u32 transfer_family_index;
} VulkanPhysicalDeviceQueueFamilyInfo;

static b8 select_physical_device(VkInstance instance, b8 discrete_gpu, VulkanDevice& device);
b8 PhysicalDeviceMeetsRequirements(
    VkPhysicalDevice device,
    const VkPhysicalDeviceProperties* properties,
    const VkPhysicalDeviceFeatures* features,
    const VulkanPhysicalDeviceRequirements* requirements,
    VulkanPhysicalDeviceQueueFamilyInfo* out_queue_family_info);

static b8 select_physical_device(VkInstance instance, b8 discrete_gpu, VulkanDevice& device) {
    uint32_t physical_device_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr));
    if (physical_device_count == 0) {
        LOG_FATAL("No devices which support Vulkan were found.");
        return false;
    }

    std::vector<VkPhysicalDevice> physical_devices;
    physical_devices.resize(physical_device_count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data()));
    
    for (u32 i = 0; i < physical_device_count; ++i) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physical_devices[i], &properties);

        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(physical_devices[i], &features);

        VkPhysicalDeviceMemoryProperties memory;
        vkGetPhysicalDeviceMemoryProperties(physical_devices[i], &memory);

        // Check if device supports local/host visible combo
        b8 supports_device_local_host_visible = false;
        for (u32 j = 0; j < memory.memoryTypeCount; ++j) {
            if (((memory.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) &&
                ((memory.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)) {
                supports_device_local_host_visible = true;
                break;
            }
        }

        VulkanPhysicalDeviceRequirements requirements = {};
        requirements.transfer = true;
        requirements.compute = true;
        requirements.discrete_gpu = discrete_gpu;
        requirements.device_extension_names = {};
        
        // Essential extensions for compute workloads
        requirements.device_extension_names.emplace_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
        requirements.device_extension_names.emplace_back(VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME);
        
        // Add sync2 if not native
        if (VK_VERSION_MAJOR(properties.apiVersion) < 1 || 
            (VK_VERSION_MAJOR(properties.apiVersion) == 1 && VK_VERSION_MINOR(properties.apiVersion) < 3)) {
            requirements.device_extension_names.emplace_back("VK_KHR_synchronization2");
        }

#ifdef QS_PLATFORM_APPLE
        requirements.discrete_gpu = false;
        requirements.device_extension_names.emplace_back("VK_KHR_portability_subset");
#endif

        VulkanPhysicalDeviceQueueFamilyInfo queue_info = {};
        b8 result = PhysicalDeviceMeetsRequirements(
            physical_devices[i],
            &properties,
            &features,
            &requirements,
            &queue_info);

        if (result) {
            LOG_DEBUG("Selected device: '{}'.", properties.deviceName);
            
            // GPU type
            switch (properties.deviceType) {
                default:
                case VK_PHYSICAL_DEVICE_TYPE_OTHER:
                    LOG_DEBUG("GPU type is Unknown.");
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                    LOG_DEBUG("GPU type is Integrated.");
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                    LOG_DEBUG("GPU type is Discrete.");
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                    LOG_DEBUG("GPU type is Virtual.");
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_CPU:
                    LOG_DEBUG("GPU type is CPU.");
                    break;
            }

            LOG_DEBUG(
                "GPU Driver version: {}.{}.{}",
                VK_VERSION_MAJOR(properties.driverVersion),
                VK_VERSION_MINOR(properties.driverVersion),
                VK_VERSION_PATCH(properties.driverVersion));

            // Save off the device-supported API version
            device.api_major = VK_VERSION_MAJOR(properties.apiVersion);
            device.api_minor = VK_VERSION_MINOR(properties.apiVersion);
            device.api_patch = VK_VERSION_PATCH(properties.apiVersion);

            LOG_DEBUG(
                "Vulkan API version: {}.{}.{}",
                device.api_major,
                device.api_minor,
                device.api_patch);

            // Memory information
            for (u32 j = 0; j < memory.memoryHeapCount; ++j) {
                f32 memory_size_gib = (((f32)memory.memoryHeaps[j].size) / 1024.0f / 1024.0f / 1024.0f);
                if (memory.memoryHeaps[j].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    LOG_DEBUG("Local GPU memory: {} GiB", memory_size_gib);
                } else {
                    LOG_DEBUG("Shared System memory: {} GiB", memory_size_gib);
                }
            }

            device.physical_device = physical_devices[i];
            device.transfer_queue_index = queue_info.transfer_family_index;
            device.compute_queue_index = queue_info.compute_family_index;

            // Keep a copy of properties, features and memory info for later use
            device.properties = properties;
            device.features = features;
            device.memory = memory;
            device.supports_device_local_host_visible = supports_device_local_host_visible;

            // Check API version support
            if (device.api_major >= 1 && device.api_minor >= 3) {
                device.support_flags |= VULKAN_DEVICE_SUPPORT_FLAG_NATIVE_13_FEATURES_BIT;
            } else if (device.api_major >= 1 && device.api_minor >= 2) {
                device.support_flags |= VULKAN_DEVICE_SUPPORT_FLAG_NATIVE_12_FEATURES_BIT;
            }
            
            break;
        }
    }

    // Ensure a device was selected
    if (!device.physical_device) {
        LOG_ERROR("No physical devices were found which meet the requirements.");
        return false;
    }

    physical_devices.clear();
    LOG_DEBUG("Physical device selected.");
    return true;
}

b8 PhysicalDeviceMeetsRequirements(
    VkPhysicalDevice device,
    const VkPhysicalDeviceProperties* properties,
    const VkPhysicalDeviceFeatures* features,
    const VulkanPhysicalDeviceRequirements* requirements,
    VulkanPhysicalDeviceQueueFamilyInfo* out_queue_info) {
    
    out_queue_info->compute_family_index = UINT32_MAX;
    out_queue_info->transfer_family_index = UINT32_MAX;

    // Discrete GPU check
    if (requirements->discrete_gpu) {
        if (properties->deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            LOG_DEBUG("Device is not a discrete GPU, and one is required. Skipping.");
            return false;
        }
    }

    u32 queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

    LOG_DEBUG("Compute | Transfer | Name");
    u8 min_transfer_score = 255;
    for (u32 i = 0; i < queue_family_count; ++i) {
        u8 current_transfer_score = 0;

        // Compute queue?
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            out_queue_info->compute_family_index = i;
            ++current_transfer_score;
        }

        // Transfer queue?
        if (queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
            // Take the index if it is the current lowest. This increases the
            // likelihood that it is a dedicated transfer queue.
            if (current_transfer_score <= min_transfer_score) {
                min_transfer_score = current_transfer_score;
                out_queue_info->transfer_family_index = i;
            }
        }
    }

    LOG_DEBUG("      {} |        {} | {}",
            out_queue_info->compute_family_index != UINT32_MAX,
            out_queue_info->transfer_family_index != UINT32_MAX,
            properties->deviceName);

    if ((!requirements->compute || (requirements->compute && out_queue_info->compute_family_index != UINT32_MAX)) &&
        (!requirements->transfer || (requirements->transfer && out_queue_info->transfer_family_index != UINT32_MAX))) {
        
        LOG_DEBUG("Device meets queue requirements.");
        LOG_TRACE("Transfer Family Index: {}", out_queue_info->transfer_family_index);
        LOG_TRACE("Compute Family Index:  {}", out_queue_info->compute_family_index);

        // Device extensions
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
                        LOG_DEBUG("Required extension not found: '{}', skipping device.", required_ext);
                        return false;
                    }
                }
            }
        }

        return true;
    }

    return false;
}

b8 vulkan_device_create(VkInstance instance, VulkanDevice& device) {
    device = {};
    
    if (!select_physical_device(instance, true, device)) {
        LOG_WARN("No Discrete GPU with Vulkan support found. Defaulting to Integrated GPU.");
        if (!select_physical_device(instance, false, device)) {
            LOG_FATAL("No Device with Vulkan support found");
            return false;
        }
    }

    LOG_DEBUG("Creating logical device...");
    
    b8 transfer_shares_compute_queue = device.compute_queue_index == device.transfer_queue_index;
    u32 index_count = 1;
    if (!transfer_shares_compute_queue) index_count++;

    std::vector<i32> indices(index_count);
    u8 index = 0;
    indices[index++] = device.compute_queue_index;
    if (!transfer_shares_compute_queue) indices[index++] = device.transfer_queue_index;

    std::vector<VkDeviceQueueCreateInfo> queue_create_infos(index_count);
    f32 queue_priority = 1.0f;

    for (u32 i = 0; i < index_count; ++i) {
        queue_create_infos[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_infos[i].queueFamilyIndex = indices[i];
        queue_create_infos[i].queueCount = 1;
        queue_create_infos[i].flags = 0;
        queue_create_infos[i].pNext = nullptr;
        queue_create_infos[i].pQueuePriorities = &queue_priority;
    }

    std::vector<const char*> extension_names = {
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME
    };

    // Add sync2 extension if not native
    if (!(device.support_flags & VULKAN_DEVICE_SUPPORT_FLAG_NATIVE_13_FEATURES_BIT)) {
        extension_names.push_back("VK_KHR_synchronization2");
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

    // Chain the features
    sync2_features.pNext = &descriptor_indexing_features;
    device_features.pNext = &sync2_features;

    VkDeviceCreateInfo device_create_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_create_info.queueCreateInfoCount = index_count;
    device_create_info.pQueueCreateInfos = queue_create_infos.data();
    device_create_info.pEnabledFeatures = nullptr;
    device_create_info.enabledExtensionCount = static_cast<u32>(extension_names.size());
    device_create_info.ppEnabledExtensionNames = extension_names.data();
    device_create_info.enabledLayerCount = 0;
    device_create_info.ppEnabledLayerNames = nullptr;
    device_create_info.pNext = &device_features;

    VK_CHECK(vkCreateDevice(device.physical_device, &device_create_info, nullptr, &device.logical_device));
    LOG_DEBUG("Logical device created.");

    // Get queues
    vkGetDeviceQueue(device.logical_device, device.transfer_queue_index, 0, &device.transfer_queue);
    vkGetDeviceQueue(device.logical_device, device.compute_queue_index, 0, &device.compute_queue);
    LOG_DEBUG("Queues obtained.");

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
    
    if (device.logical_device != VK_NULL_HANDLE) {
        
    } else {
        LOG_WARN("Device logical_device is null, skipping device destruction.");
        return;
    }

    // Clear queue handles
    device.compute_queue = VK_NULL_HANDLE;
    device.transfer_queue = VK_NULL_HANDLE;

    // Destroy logical device
    if (device.logical_device != VK_NULL_HANDLE) {
        LOG_DEBUG("Destroying logical device...");
        try {
            vkDestroyDevice(device.logical_device, nullptr);
            device.logical_device = VK_NULL_HANDLE;
            LOG_DEBUG("Logical device destroyed successfully.");
        } catch (...) {
            LOG_ERROR("Exception occurred while destroying logical device!");
            device.logical_device = VK_NULL_HANDLE;
        }
    }

    // Physical devices are not destroyed
    device.physical_device = VK_NULL_HANDLE;

    // Reset queue indices
    device.compute_queue_index = UINT32_MAX;
    device.transfer_queue_index = UINT32_MAX;
}

} // namespace QuasarML
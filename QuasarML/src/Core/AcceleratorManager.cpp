#include "AcceleratorManager.h"
#include "Logger.h"
#include <VulkanBackend/VulkanDevice.h>

namespace QuasarML {

AcceleratorManager& AcceleratorManager::instance() {
    static AcceleratorManager mgr;
    return mgr;
}

AcceleratorManager::AcceleratorManager()
    : _device_count(0)
    , _default_device(0)
{
    initialize_devices();
}

AcceleratorManager::~AcceleratorManager() {
    if (!_shutdown_called) {
        shutdown();
    }
}

void AcceleratorManager::shutdown() {
    if (_shutdown_called) return;
    
    try {
        std::lock_guard<std::mutex> lock(_manager_mutex);
        _accelerators.clear();
        _shutdown_called = true;
    } catch (...) {
    }
}

void AcceleratorManager::initialize_devices() {
    VkInstance instance = VK_NULL_HANDLE;
    
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "QuasarML";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "QuasarML";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> extensions;
#ifdef QS_PLATFORM_APPLE
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.push_back("VK_KHR_get_physical_device_properties2");
#endif

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = static_cast<u32>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
#ifdef QS_PLATFORM_APPLE
    create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    
    if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
        LOG_ERROR("Failed to create temporary Vulkan instance for device enumeration");
        _device_count = 0;
        return;
    }

    u32 device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    
    _device_count = device_count;
    
    if (_device_count > 0) {
        LOG_INFO("Found {} Vulkan device(s)", _device_count);
    } else {
        LOG_WARN("No Vulkan devices found");
    }
    
    vkDestroyInstance(instance, nullptr);
}

u32 AcceleratorManager::get_device_count() const {
    return _device_count;
}

Accelerator* AcceleratorManager::get_accelerator(u32 device_id) {
    std::lock_guard<std::mutex> lock(_manager_mutex);
    
    if (device_id >= _device_count) {
        LOG_ERROR("Device ID {} out of range (max {})", device_id, _device_count - 1);
        return nullptr;
    }
    
    auto it = _accelerators.find(device_id);
    if (it != _accelerators.end()) {
        return it->second.get();
    }
    
    try {
        auto acc = std::make_unique<Accelerator>("QuasarML", device_id);
        auto* ptr = acc.get();
        _accelerators[device_id] = std::move(acc);
        LOG_INFO("Initialized accelerator for device {}", device_id);
        return ptr;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create accelerator for device {}: {}", device_id, e.what());
        return nullptr;
    }
}

std::vector<std::string> AcceleratorManager::get_device_names() const {
    std::vector<std::string> names;
    
    VkInstance instance = VK_NULL_HANDLE;
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "QuasarML";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "QuasarML";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> extensions;
#ifdef __APPLE__
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = static_cast<u32>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
#ifdef __APPLE__
    create_info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    
    if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
        return names;
    }

    u32 device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    
    if (device_count > 0) {
        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
        
        for (auto device : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(device, &props);
            names.push_back(props.deviceName);
        }
    }
    
    vkDestroyInstance(instance, nullptr);
    return names;
}

void AcceleratorManager::set_default_device(u32 device_id) {
    if (device_id >= _device_count) {
        LOG_WARN("Cannot set default device to {} (max {})", device_id, _device_count - 1);
        return;
    }
    _default_device = device_id;
    LOG_INFO("Default device set to {}", device_id);
}

} // namespace QuasarML

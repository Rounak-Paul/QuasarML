#include "DeviceManager.h"
#include <Backend/Vulkan/VulkanTypes.h>
#include <vulkan/vulkan.h>

namespace QuasarML {

DeviceManager& DeviceManager::instance() {
    static DeviceManager mgr;
    return mgr;
}

DeviceManager::DeviceManager() {
    enumerate();
}

DeviceManager::~DeviceManager() {
    shutdown();
}

void DeviceManager::enumerate() {
    VkApplicationInfo app_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = "QuasarML";
    app_info.apiVersion = VK_API_VERSION_1_2;
    
    std::vector<const char*> extensions;
#ifdef QS_PLATFORM_APPLE
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif
    
    VkInstanceCreateInfo create_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = static_cast<u32>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
#ifdef QS_PLATFORM_APPLE
    create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    
    VkInstance instance;
    if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
        _device_count = 0;
        return;
    }
    
    u32 count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    _device_count = count;
    
    vkDestroyInstance(instance, nullptr);
}

Device* DeviceManager::get(u32 device_id) {
    std::lock_guard<std::mutex> lock(_mutex);
    
    if (_shutdown || device_id >= _device_count) return nullptr;
    
    auto it = _devices.find(device_id);
    if (it != _devices.end()) return it->second.get();
    
    auto device = std::make_unique<Device>("QuasarML", device_id);
    if (!device->is_valid()) return nullptr;
    
    Device* ptr = device.get();
    _devices[device_id] = std::move(device);
    return ptr;
}

std::vector<std::string> DeviceManager::device_names() const {
    std::vector<std::string> names;
    
    VkApplicationInfo app_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = "QuasarML";
    app_info.apiVersion = VK_API_VERSION_1_2;
    
    std::vector<const char*> extensions;
#ifdef QS_PLATFORM_APPLE
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif
    
    VkInstanceCreateInfo create_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = static_cast<u32>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
#ifdef QS_PLATFORM_APPLE
    create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    
    VkInstance instance;
    if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) return names;
    
    u32 count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    
    for (auto dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        names.push_back(props.deviceName);
    }
    
    vkDestroyInstance(instance, nullptr);
    return names;
}

void DeviceManager::set_default(u32 device_id) {
    if (device_id < _device_count) _default_device = device_id;
}

void DeviceManager::shutdown() {
    if (_shutdown) return;
    _shutdown = true;
    _devices.clear();
}

}

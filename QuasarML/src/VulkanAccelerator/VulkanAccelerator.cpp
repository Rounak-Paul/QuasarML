#include "VulkanAccelerator.h"

namespace QuasarML {

std::vector<const char*> validation_layers = {
    "VK_LAYER_KHRONOS_validation"
    // ,"VK_LAYER_LUNARG_api_dump" // For all vulkan calls
};

static b8 check_validation_layer_support();
static void fetch_api_version(VulkanContext& ctx);
static b8 create_instance(const std::string& name, VulkanContext& ctx);
static void setup_debug_messenger(VulkanContext& ctx);

VulkanAccelerator::VulkanAccelerator(const std::string& name)
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
        if (!vulkan_device_create(_ctx.instance, _ctx.device)) {
            LOG_ERROR("Failed to create device!");
        }
        _deletion_queue.push_function([&]() {
            vulkan_device_destroy(_ctx.instance, _ctx.device);
            _ctx.device.logical_device = VK_NULL_HANDLE;
        });
    }
}

VulkanAccelerator::~VulkanAccelerator()
{
    device_wait_idle();

    _deletion_queue.flush();
    
    // Reset API version info
    _ctx.api_major = 0;
    _ctx.api_minor = 0;
    _ctx.api_patch = 0;
}

void VulkanAccelerator::device_wait_idle()
{
    vkDeviceWaitIdle(_ctx.device.logical_device);
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

static std::vector<const char*> get_required_extensions()
{
    std::vector<const char*> extensions;
    
    #ifdef QS_PLATFORM_APPLE
    extensions.emplace_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    #endif
    
    #ifdef QS_DEBUG 
    extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    #endif
    
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

    auto extensions = get_required_extensions();

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
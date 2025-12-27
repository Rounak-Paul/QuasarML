#pragma once

#include "Device.h"
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace QuasarML {

class DeviceManager {
public:
    static DeviceManager& instance();
    
    u32 device_count() const { return _device_count; }
    Device* get(u32 device_id = 0);
    std::vector<std::string> device_names() const;
    
    void set_default(u32 device_id);
    u32 get_default() const { return _default_device; }
    
    void shutdown();
    ~DeviceManager();

private:
    DeviceManager();
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    
    mutable std::mutex _mutex;
    std::unordered_map<u32, std::unique_ptr<Device>> _devices;
    u32 _device_count = 0;
    u32 _default_device = 0;
    bool _shutdown = false;
    
    void enumerate();
};

}

#pragma once

#include <qspch.h>
#include "Accelerator.h"
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace QuasarML {

class AcceleratorManager {
public:
    static AcceleratorManager& instance();
    
    u32 get_device_count() const;
    
    Accelerator* get_accelerator(u32 device_id = 0);
    
    std::vector<std::string> get_device_names() const;
    
    void set_default_device(u32 device_id);
    u32 get_default_device() const { return _default_device; }
    
    void shutdown();
    
    ~AcceleratorManager();

private:
    AcceleratorManager();
    
    AcceleratorManager(const AcceleratorManager&) = delete;
    AcceleratorManager& operator=(const AcceleratorManager&) = delete;
    
    mutable std::mutex _manager_mutex;
    std::unordered_map<u32, std::unique_ptr<Accelerator>> _accelerators;
    u32 _device_count;
    u32 _default_device;
    bool _shutdown_called = false;
    
    void initialize_devices();
};

} // namespace QuasarML

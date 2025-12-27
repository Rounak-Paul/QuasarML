#pragma once

#include <Core/Device.h>
#include <unordered_map>
#include <mutex>

namespace QuasarML {
namespace Ops {

struct DeviceParams {
    u32 elementwise_workgroup;
    u32 reduction_workgroup;
    u32 softmax_workgroup;
    u32 matmul_tile_size;
    u32 matmul_block_size;
    u32 matmul_effective_tile;
    bool matmul_use_register_blocking;
    u32 elements_per_thread;
};

inline DeviceParams compute_device_params(const ComputeLimits& limits) {
    DeviceParams params;
    
    u32 max_invocations = limits.max_workgroup_invocations;
    u32 shared_mem = limits.max_shared_memory;
    
    if (max_invocations >= 1024) {
        params.elementwise_workgroup = 1024;
        params.reduction_workgroup = 512;
        params.softmax_workgroup = 512;
        params.elements_per_thread = 4;
    } else if (max_invocations >= 512) {
        params.elementwise_workgroup = 512;
        params.reduction_workgroup = 256;
        params.softmax_workgroup = 256;
        params.elements_per_thread = 2;
    } else if (max_invocations >= 256) {
        params.elementwise_workgroup = 256;
        params.reduction_workgroup = 256;
        params.softmax_workgroup = 256;
        params.elements_per_thread = 1;
    } else {
        params.elementwise_workgroup = 128;
        params.reduction_workgroup = 128;
        params.softmax_workgroup = 128;
        params.elements_per_thread = 1;
    }
    
    if (max_invocations >= 256 && shared_mem >= 32768) {
        params.matmul_tile_size = 16;
        params.matmul_block_size = 4;
        params.matmul_use_register_blocking = true;
    } else if (max_invocations >= 256 && shared_mem >= 16384) {
        params.matmul_tile_size = 16;
        params.matmul_block_size = 2;
        params.matmul_use_register_blocking = true;
    } else {
        params.matmul_tile_size = 16;
        params.matmul_block_size = 1;
        params.matmul_use_register_blocking = false;
    }
    
    params.matmul_effective_tile = params.matmul_tile_size * params.matmul_block_size;
    
    return params;
}

class DeviceTuning {
public:
    static DeviceTuning& instance() {
        static DeviceTuning inst;
        return inst;
    }
    
    const DeviceParams& get(Device& device) {
        u32 idx = device.device_index();
        
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _params.find(idx);
        if (it != _params.end()) return it->second;
        
        auto limits = device.limits();
        _params[idx] = compute_device_params(limits);
        return _params[idx];
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        _params.clear();
    }

private:
    DeviceTuning() = default;
    std::unordered_map<u32, DeviceParams> _params;
    std::mutex _mutex;
};

inline const DeviceParams& get_device_params(Device& device) {
    return DeviceTuning::instance().get(device);
}

}
}

#include "Device.h"
#include "Tensor.h"
#include "Kernel.h"
#include <Common/Assert.h>
#include <Common/Logger.h>

namespace QuasarML {

Device::Device(const std::string& name, u32 device_index) 
    : _name(name), _device_index(device_index) {
    _backend = create_vulkan_backend();
    if (!_backend->init(name, device_index)) {
        _backend.reset();
        return;
    }
    Logger::init();
}

Device::~Device() {
    if (_in_batch) end_batch();
    _kernels.clear();
    if (_backend) _backend->shutdown();
}

Device::Device(Device&& other) noexcept
    : _backend(std::move(other._backend))
    , _kernels(std::move(other._kernels))
    , _name(std::move(other._name))
    , _device_index(other._device_index)
    , _in_batch(other._in_batch) {
}

Device& Device::operator=(Device&& other) noexcept {
    if (this != &other) {
        if (_in_batch) end_batch();
        _kernels.clear();
        if (_backend) _backend->shutdown();
        
        _backend = std::move(other._backend);
        _kernels = std::move(other._kernels);
        _name = std::move(other._name);
        _device_index = other._device_index;
        _in_batch = other._in_batch;
    }
    return *this;
}

bool Device::is_valid() const {
    return _backend && _backend->is_valid();
}

ComputeLimits Device::limits() const {
    if (!_backend) return ComputeLimits{};
    return _backend->get_compute_limits();
}

const DeviceCapabilities& Device::capabilities() const {
    return _backend->get_capabilities();
}

std::shared_ptr<Kernel> Device::create_kernel(const std::string& name, const std::string& glsl_source, u32 num_bindings, u32 push_size) {
    std::lock_guard<std::mutex> lock(_kernel_mutex);
    
    auto it = _kernels.find(name);
    if (it != _kernels.end()) {
        return it->second;
    }
    
    auto kernel = std::make_shared<Kernel>(this, name, glsl_source, num_bindings, push_size);
    _kernels[name] = kernel;
    return kernel;
}

std::shared_ptr<Kernel> Device::get_kernel(const std::string& name) {
    std::lock_guard<std::mutex> lock(_kernel_mutex);
    auto it = _kernels.find(name);
    return it != _kernels.end() ? it->second : nullptr;
}

std::shared_ptr<Tensor> Device::create_tensor(const std::vector<u32>& shape, DataType dtype, bool host_visible) {
    return std::make_shared<Tensor>(this, shape, dtype, host_visible);
}

void Device::begin_batch() {
    QS_ASSERT(!_in_batch, "Already in batch");
    _backend->begin_recording();
    _in_batch = true;
}

void Device::end_batch() {
    QS_ASSERT(_in_batch, "Not in batch");
    _backend->end_recording();
    _in_batch = false;
}

void Device::synchronize() {
    if (_backend) _backend->synchronize();
}

}

#include "Kernel.h"
#include "Device.h"
#include "Tensor.h"
#include <Common/Assert.h>

namespace QuasarML {

Kernel::Kernel(Device* device, const std::string& name, const std::string& glsl_source, u32 num_bindings, u32 push_constant_size)
    : _device(device), _name(name), _glsl(glsl_source), _num_bindings(num_bindings), _push_size(push_constant_size) {
    _bindings.resize(num_bindings);
    _pipeline = _device->backend()->create_compute_pipeline(glsl_source, num_bindings, push_constant_size);
    _valid = _pipeline.valid();
}

Kernel::~Kernel() {
    if (_valid && _device) {
        _device->backend()->destroy_pipeline(_pipeline);
    }
}

Kernel::Kernel(Kernel&& other) noexcept
    : _device(other._device), _name(std::move(other._name)), _glsl(std::move(other._glsl)),
      _num_bindings(other._num_bindings), _push_size(other._push_size),
      _pipeline(other._pipeline), _bindings(std::move(other._bindings)), _valid(other._valid) {
    other._pipeline = {};
    other._valid = false;
}

Kernel& Kernel::operator=(Kernel&& other) noexcept {
    if (this != &other) {
        if (_valid && _device) {
            _device->backend()->destroy_pipeline(_pipeline);
        }
        _device = other._device;
        _name = std::move(other._name);
        _glsl = std::move(other._glsl);
        _num_bindings = other._num_bindings;
        _push_size = other._push_size;
        _pipeline = other._pipeline;
        _bindings = std::move(other._bindings);
        _valid = other._valid;
        other._pipeline = {};
        other._valid = false;
    }
    return *this;
}

void Kernel::bind(u32 binding, std::shared_ptr<Tensor> tensor) {
    QS_ASSERT(binding < _num_bindings, "Binding out of range");
    _bindings[binding] = tensor;
}

void Kernel::clear_bindings() {
    for (auto& b : _bindings) b.reset();
}

void Kernel::execute(u32 dispatch_x, u32 dispatch_y, u32 dispatch_z, const void* push_data) {
    BufferBinding buffers[8];
    for (size_t i = 0; i < _bindings.size(); ++i) {
        auto t = _bindings[i].lock();
        QS_ASSERT(t, "Tensor binding expired");
        buffers[i].buffer = &t->buffer();
        buffers[i].offset = t->element_offset() * t->element_size();
        buffers[i].range = t->size_bytes();
    }
    std::vector<BufferBinding> buf_vec(buffers, buffers + _bindings.size());
    _device->backend()->execute_compute(_pipeline, buf_vec, dispatch_x, dispatch_y, dispatch_z, push_data, _push_size);
}

void Kernel::record(u32 dispatch_x, u32 dispatch_y, u32 dispatch_z, const void* push_data) {
    BufferBinding buffers[8];
    for (size_t i = 0; i < _bindings.size(); ++i) {
        auto t = _bindings[i].lock();
        QS_ASSERT(t, "Tensor binding expired");
        buffers[i].buffer = &t->buffer();
        buffers[i].offset = t->element_offset() * t->element_size();
        buffers[i].range = t->size_bytes();
    }
    std::vector<BufferBinding> buf_vec(buffers, buffers + _bindings.size());
    _device->backend()->record_compute(_pipeline, buf_vec, dispatch_x, dispatch_y, dispatch_z, push_data, _push_size);
}

u32 Kernel::optimal_dispatch_1d(u32 total, u32 local_size) const {
    return _device->backend()->optimal_dispatch_1d(total, local_size);
}

}

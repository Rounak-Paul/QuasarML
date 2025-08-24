#include "Kernel.h"
#include <stdexcept>

namespace QuasarML {

Kernel::Kernel(VulkanBackend* backend,
               const std::string& name,
               const std::string& glsl_source,
               u32 expected_tensor_count,
               u32 push_constant_size)
    : _backend(backend)
    , _name(name)
    , _glsl_source(glsl_source)
    , _expected_tensor_count(expected_tensor_count)
    , _push_constant_size(push_constant_size)
    , _is_valid(false)
{
    if (!_backend) throw std::invalid_argument("Backend cannot be null");
    if (_name.empty()) throw std::invalid_argument("Kernel name cannot be empty");
    if (_glsl_source.empty()) throw std::invalid_argument("GLSL source cannot be empty");
    if (_expected_tensor_count == 0) throw std::invalid_argument("Expected tensor count must be > 0");

    _bound_tensors.resize(_expected_tensor_count);
    initialize_pipeline();
}

Kernel::~Kernel() {
    cleanup_pipeline();
}

Kernel::Kernel(Kernel&& other) noexcept
    : _backend(other._backend)
    , _name(std::move(other._name))
    , _glsl_source(std::move(other._glsl_source))
    , _expected_tensor_count(other._expected_tensor_count)
    , _push_constant_size(other._push_constant_size)
    , _pipeline(std::move(other._pipeline))
    , _bound_tensors(std::move(other._bound_tensors))
    , _is_valid(other._is_valid)
{
    other._backend = nullptr;
    other._expected_tensor_count = 0;
    other._push_constant_size = 0;
    other._is_valid = false;
    other._pipeline = {};
}

Kernel& Kernel::operator=(Kernel&& other) noexcept {
    if (this != &other) {
        cleanup_pipeline();
        _backend = other._backend;
        _name = std::move(other._name);
        _glsl_source = std::move(other._glsl_source);
        _expected_tensor_count = other._expected_tensor_count;
        _push_constant_size = other._push_constant_size;
        _pipeline = std::move(other._pipeline);
        _bound_tensors = std::move(other._bound_tensors);
        _is_valid = other._is_valid;

        other._backend = nullptr;
        other._expected_tensor_count = 0;
        other._push_constant_size = 0;
        other._is_valid = false;
        other._pipeline = {};
    }
    return *this;
}

void Kernel::bind_tensor(u32 binding, std::shared_ptr<Tensor> tensor) {
    if (binding >= _expected_tensor_count) {
        throw std::out_of_range("Binding index " + std::to_string(binding) + " exceeds expected tensor count");
    }
    if (!tensor) throw std::invalid_argument("Tensor cannot be null");
    if (!tensor->is_valid()) throw std::invalid_argument("Tensor is not valid");

    // store weak ptr; actual descriptor binding will be provided at dispatch time via backend API
    _bound_tensors[binding] = tensor;
}

std::weak_ptr<Tensor> Kernel::get_bound_tensor(u32 binding) const {
    if (binding >= _expected_tensor_count) throw std::out_of_range("Binding index exceeds expected tensor count");
    return _bound_tensors[binding];
}

bool Kernel::are_tensors_bound() const {
    for (const auto& wt : _bound_tensors) {
        if (wt.expired()) return false;
    }
    return true;
}

void Kernel::clear_bindings() {
    for (auto& wt : _bound_tensors) wt.reset();
}

void Kernel::execute(u32 dispatch_x, u32 dispatch_y, u32 dispatch_z, const void* push_data) {
    if (!_is_valid) throw std::runtime_error("Kernel is not valid");
    if (!are_tensors_bound()) throw std::runtime_error("Not all required tensors are bound");
    validate_execution_parameters(dispatch_x, dispatch_y, dispatch_z);

    if (_push_constant_size > 0) {
        if (!push_data) throw std::invalid_argument("Kernel expects push constants but none provided");
    }

    // Collect raw buffer pointers in binding order for backend
    std::vector<VulkanBackend::Buffer*> buffers;
    buffers.reserve(_expected_tensor_count);
    for (u32 i = 0; i < _expected_tensor_count; ++i) {
        auto sp = _bound_tensors[i].lock();
        if (!sp) throw std::runtime_error("Bound tensor expired unexpectedly");
        buffers.push_back(&sp->get_buffer());
    }

    _backend->execute_compute(_pipeline, dispatch_x, dispatch_y, dispatch_z,
                              push_data, _push_constant_size, buffers);
}

void Kernel::record_execution(u32 dispatch_x, u32 dispatch_y, u32 dispatch_z, const void* push_data) {
    if (!_is_valid) throw std::runtime_error("Kernel is not valid");
    if (!are_tensors_bound()) throw std::runtime_error("Not all required tensors are bound");
    validate_execution_parameters(dispatch_x, dispatch_y, dispatch_z);

    if (_push_constant_size > 0) {
        if (!push_data) throw std::invalid_argument("Kernel expects push constants but none provided");
    }

    std::vector<VulkanBackend::Buffer*> buffers;
    buffers.reserve(_expected_tensor_count);
    for (u32 i = 0; i < _expected_tensor_count; ++i) {
        auto sp = _bound_tensors[i].lock();
        if (!sp) throw std::runtime_error("Bound tensor expired unexpectedly");
        buffers.push_back(&sp->get_buffer());
    }

    _backend->record_compute_dispatch(_pipeline, dispatch_x, dispatch_y, dispatch_z,
                                      push_data, _push_constant_size, buffers);
}

bool Kernel::is_valid() const {
    return _is_valid && _backend && _pipeline.is_valid();
}

u32 Kernel::calculate_dispatch_1d(u32 total_elements, u32 local_size) const {
    if (!_backend) throw std::runtime_error("Backend not available");
    return _backend->calculate_dispatch_1d(total_elements, local_size);
}

bool Kernel::validate_push_constant_size(u32 data_size) const {
    return data_size == _push_constant_size;
}

void Kernel::initialize_pipeline() {
    try {
        _pipeline = _backend->create_compute_pipeline(_glsl_source, _expected_tensor_count, _push_constant_size);
        _is_valid = _pipeline.is_valid();
        if (!_is_valid) throw std::runtime_error("Failed to create compute pipeline for kernel '" + _name + "'");
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to initialize kernel '") + _name + "': " + e.what());
    }
}

void Kernel::cleanup_pipeline() {
    if (_backend && _pipeline.is_valid()) {
        _backend->destroy_compute_pipeline(_pipeline);
    }
    _pipeline = {};
    _is_valid = false;
}

void Kernel::validate_execution_parameters(u32 dispatch_x, u32 dispatch_y, u32 dispatch_z) const {
    if (dispatch_x == 0 || dispatch_y == 0 || dispatch_z == 0) throw std::invalid_argument("Dispatch dimensions must be > 0");
    auto limits = _backend->get_compute_limits();
    if (dispatch_x > limits.max_work_group_count[0] ||
        dispatch_y > limits.max_work_group_count[1] ||
        dispatch_z > limits.max_work_group_count[2]) {
        throw std::invalid_argument("Dispatch dimensions exceed device limits");
    }
}

void Kernel::update_descriptor_bindings() {
    // Backend now expects buffer list at dispatch time; nothing to do here.
}

} // namespace QuasarML
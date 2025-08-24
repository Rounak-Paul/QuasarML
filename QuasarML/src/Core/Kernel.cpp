    #include "Kernel.h"
    #include "Tensor.h"

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
        if (!_backend) {
            throw std::invalid_argument("Backend cannot be null");
        }
        
        if (_name.empty()) {
            throw std::invalid_argument("Kernel name cannot be empty");
        }
        
        if (_glsl_source.empty()) {
            throw std::invalid_argument("GLSL source cannot be empty");
        }
        
        if (_expected_tensor_count == 0) {
            throw std::invalid_argument("Expected tensor count must be greater than 0");
        }
        
        // Initialize bound tensors vector
        _bound_tensors.resize(_expected_tensor_count);
        
        // Initialize the compute pipeline
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
        // Reset other object
        other._backend = nullptr;
        other._expected_tensor_count = 0;
        other._push_constant_size = 0;
        other._is_valid = false;
        other._pipeline = {};
    }

    Kernel& Kernel::operator=(Kernel&& other) noexcept {
        if (this != &other) {
            // Cleanup current resources
            cleanup_pipeline();
            
            // Move data
            _backend = other._backend;
            _name = std::move(other._name);
            _glsl_source = std::move(other._glsl_source);
            _expected_tensor_count = other._expected_tensor_count;
            _push_constant_size = other._push_constant_size;
            _pipeline = std::move(other._pipeline);
            _bound_tensors = std::move(other._bound_tensors);
            _is_valid = other._is_valid;
            
            // Reset other object
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
            throw std::out_of_range("Binding index " + std::to_string(binding) + 
                                " exceeds expected tensor count " + std::to_string(_expected_tensor_count));
        }
        
        if (!tensor) {
            throw std::invalid_argument("Tensor cannot be null");
        }
        
        if (!tensor->is_valid()) {
            throw std::invalid_argument("Tensor is not valid");
        }
        
        // Store weak reference to tensor
        _bound_tensors[binding] = tensor;
        
        // Update Vulkan descriptor binding
        auto& buffer = tensor->get_buffer();
        _backend->bind_buffer_to_pipeline(_pipeline, binding, buffer);
    }

    std::weak_ptr<Tensor> Kernel::get_bound_tensor(u32 binding) const {
        if (binding >= _expected_tensor_count) {
            throw std::out_of_range("Binding index exceeds expected tensor count");
        }
        
        return _bound_tensors[binding];
    }

    bool Kernel::are_tensors_bound() const {
        for (const auto& weak_tensor : _bound_tensors) {
            if (weak_tensor.expired()) {
                return false;
            }
        }
        return true;
    }

    void Kernel::clear_bindings() {
        for (auto& weak_tensor : _bound_tensors) {
            weak_tensor.reset();
        }
    }

    void Kernel::execute(u32 dispatch_x, u32 dispatch_y, u32 dispatch_z, const void* push_data) {
        if (!_is_valid) {
            throw std::runtime_error("Kernel is not in valid state");
        }
        
        if (!are_tensors_bound()) {
            throw std::runtime_error("Not all required tensors are bound");
        }
        
        validate_execution_parameters(dispatch_x, dispatch_y, dispatch_z);
        
        if (push_data && _push_constant_size > 0) {
            // Validate push constant data if provided
            if (!validate_push_constant_size(_push_constant_size)) {
                throw std::invalid_argument("Push constant size mismatch");
            }
        } else if (_push_constant_size > 0 && !push_data) {
            throw std::invalid_argument("Kernel expects push constants but none provided");
        }
        
        _backend->execute_compute(_pipeline, dispatch_x, dispatch_y, dispatch_z, 
                                push_data, _push_constant_size);
    }

    void Kernel::record_execution(u32 dispatch_x, u32 dispatch_y, u32 dispatch_z, const void* push_data) {
        if (!_is_valid) {
            throw std::runtime_error("Kernel is not in valid state");
        }
        
        if (!are_tensors_bound()) {
            throw std::runtime_error("Not all required tensors are bound");
        }
        
        validate_execution_parameters(dispatch_x, dispatch_y, dispatch_z);
        
        if (push_data && _push_constant_size > 0) {
            if (!validate_push_constant_size(_push_constant_size)) {
                throw std::invalid_argument("Push constant size mismatch");
            }
        } else if (_push_constant_size > 0 && !push_data) {
            throw std::invalid_argument("Kernel expects push constants but none provided");
        }
        
        _backend->record_compute_dispatch(_pipeline, dispatch_x, dispatch_y, dispatch_z,
                                        push_data, _push_constant_size);
    }

    bool Kernel::is_valid() const {
        return _is_valid && _backend && _pipeline.is_valid();
    }

    u32 Kernel::calculate_dispatch_1d(u32 total_elements, u32 local_size) const {
        if (!_backend) {
            throw std::runtime_error("Backend not available");
        }
        
        return _backend->calculate_dispatch_1d(total_elements, local_size);
    }

    bool Kernel::validate_push_constant_size(u32 data_size) const {
        return data_size == _push_constant_size;
    }

    void Kernel::initialize_pipeline() {
        try {
            _pipeline = _backend->create_compute_pipeline(_glsl_source, _expected_tensor_count, _push_constant_size);
            _is_valid = _pipeline.is_valid();
            
            if (!_is_valid) {
                throw std::runtime_error("Failed to create compute pipeline for kernel '" + _name + "'");
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to initialize kernel '" + _name + "': " + e.what());
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
        if (dispatch_x == 0 || dispatch_y == 0 || dispatch_z == 0) {
            throw std::invalid_argument("Dispatch dimensions must be greater than 0");
        }
        
        // Check against device limits
        auto limits = _backend->get_compute_limits();
        if (dispatch_x > limits.max_work_group_count[0] ||
            dispatch_y > limits.max_work_group_count[1] ||
            dispatch_z > limits.max_work_group_count[2]) {
            throw std::invalid_argument("Dispatch dimensions exceed device limits");
        }
    }

    void Kernel::update_descriptor_bindings() {
        // This method can be used to refresh descriptor bindings if needed
        // Currently, bindings are updated immediately in bind_tensor()
        for (u32 i = 0; i < _expected_tensor_count; ++i) {
            if (auto tensor = _bound_tensors[i].lock()) {
                auto& buffer = tensor->get_buffer();
                _backend->bind_buffer_to_pipeline(_pipeline, i, buffer);
            }
        }
    }

    } // namespace QuasarML
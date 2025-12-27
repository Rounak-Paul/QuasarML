#pragma once

#include <Core/Accelerator.h>
#include <Core/AcceleratorManager.h>
#include <Core/Kernel.h>
#include <Core/Tensor.h>
#include <Core/DataTypes.h>
#include <memory>
#include <vector>

namespace qsml {

using QuasarML::Accelerator;
using QuasarML::AcceleratorManager;
using QuasarML::Kernel;

using Tensor = std::shared_ptr<QuasarML::Tensor>;

namespace detail {
    inline Accelerator*& default_accelerator() {
        static Accelerator* acc = nullptr;
        return acc;
    }
    
    inline std::unique_ptr<Accelerator>& owned_accelerator() {
        static std::unique_ptr<Accelerator> acc;
        return acc;
    }
}

inline u32 device_count() {
    return AcceleratorManager::instance().get_device_count();
}

inline std::vector<std::string> device_names() {
    return AcceleratorManager::instance().get_device_names();
}

inline void set_device(u32 device_id) {
    AcceleratorManager::instance().set_default_device(device_id);
    auto* acc = AcceleratorManager::instance().get_accelerator(device_id);
    if (acc) {
        detail::default_accelerator() = acc;
    }
}

inline u32 current_device() {
    return AcceleratorManager::instance().get_default_device();
}

inline Accelerator& get_device(u32 device_id) {
    auto* acc = AcceleratorManager::instance().get_accelerator(device_id);
    if (!acc) {
        throw std::runtime_error("Failed to get accelerator for device " + std::to_string(device_id));
    }
    return *acc;
}

inline void set_accelerator(Accelerator& acc) {
    detail::default_accelerator() = &acc;
}

inline Accelerator& accelerator() {
    if (!detail::default_accelerator()) {
        detail::owned_accelerator() = std::make_unique<Accelerator>("QuasarML");
        detail::default_accelerator() = detail::owned_accelerator().get();
    }
    return *detail::default_accelerator();
}

inline bool is_valid() {
    return accelerator().is_valid();
}

inline void enable_auto_batching(bool enable = true) {
    accelerator().enable_auto_batching(enable);
}

inline bool is_auto_batching_enabled() {
    return accelerator().is_auto_batching_enabled();
}

inline void flush_pipeline() {
    accelerator().flush_pipeline();
}

inline void synchronize() {
    accelerator().synchronize();
}

inline Tensor zeros(const std::vector<u32>& shape, DataType dtype = DataType::F32) {
    auto t = accelerator().create_tensor(shape, dtype);
    t->zero();
    return t;
}

inline Tensor ones(const std::vector<u32>& shape, DataType dtype = DataType::F32) {
    auto t = accelerator().create_tensor(shape, dtype);
    if (dtype == DataType::F32) {
        float one = 1.0f;
        t->fill(&one);
    } else if (dtype == DataType::I32) {
        int32_t one = 1;
        t->fill(&one);
    }
    return t;
}

inline Tensor empty(const std::vector<u32>& shape, DataType dtype = DataType::F32) {
    return accelerator().create_tensor(shape, dtype);
}

inline Tensor randn(const std::vector<u32>& shape, DataType dtype = DataType::F32, float mean = 0.0f, float stddev = 1.0f) {
    return accelerator().ops().random_normal(shape, dtype, mean, stddev);
}

inline Tensor rand(const std::vector<u32>& shape, DataType dtype = DataType::F32, float low = 0.0f, float high = 1.0f) {
    return accelerator().ops().random_uniform(shape, dtype, low, high);
}

inline Tensor tensor(const std::vector<float>& data, const std::vector<u32>& shape, DataType dtype = DataType::F32) {
    return accelerator().create_tensor(data.data(), shape, dtype);
}

inline Tensor from_data(const void* data, const std::vector<u32>& shape, DataType dtype = DataType::F32) {
    return accelerator().create_tensor(data, shape, dtype);
}

inline Tensor add(Tensor a, Tensor b) {
    return a->get_accelerator()->ops().add(a, b);
}

inline Tensor sub(Tensor a, Tensor b) {
    return a->get_accelerator()->ops().sub(a, b);
}

inline Tensor mul(Tensor a, Tensor b) {
    return a->get_accelerator()->ops().mul(a, b);
}

inline Tensor div(Tensor a, Tensor b) {
    return a->get_accelerator()->ops().div(a, b);
}

inline Tensor pow(Tensor a, Tensor b) {
    return a->get_accelerator()->ops().pow(a, b);
}

inline Tensor abs(Tensor x) {
    return x->get_accelerator()->ops().abs(x);
}

inline Tensor neg(Tensor x) {
    return x->get_accelerator()->ops().neg(x);
}

inline Tensor clamp(Tensor x, float min_val, float max_val) {
    return x->get_accelerator()->ops().clamp(x, min_val, max_val);
}

inline Tensor add_scalar(Tensor x, float scalar) {
    return x->get_accelerator()->ops().add_scalar(x, scalar);
}

inline Tensor mul_scalar(Tensor x, float scalar) {
    return x->get_accelerator()->ops().mul_scalar(x, scalar);
}

inline Tensor relu(Tensor x) {
    return x->get_accelerator()->ops().relu(x);
}

inline Tensor sigmoid(Tensor x) {
    return x->get_accelerator()->ops().sigmoid(x);
}

inline Tensor tanh(Tensor x) {
    return x->get_accelerator()->ops().tanh(x);
}

inline Tensor softmax(Tensor x, int axis = -1) {
    return x->get_accelerator()->ops().softmax(x, axis);
}

inline Tensor gelu(Tensor x) {
    auto& acc = *x->get_accelerator();
    auto half = acc.ops().mul_scalar(x, 0.5f);
    auto cubed_term = acc.ops().mul(x, acc.ops().mul(x, x));
    auto inner = acc.ops().add(x, acc.ops().mul_scalar(cubed_term, 0.044715f));
    auto tanhval = acc.ops().tanh(acc.ops().mul_scalar(inner, 0.7978845608f));
    auto one_plus = acc.ops().add_scalar(tanhval, 1.0f);
    return acc.ops().mul(half, one_plus);
}

inline Tensor exp(Tensor x) {
    return x->get_accelerator()->ops().exp(x);
}

inline Tensor log(Tensor x) {
    return x->get_accelerator()->ops().log(x);
}

inline Tensor sin(Tensor x) {
    return x->get_accelerator()->ops().sin(x);
}

inline Tensor cos(Tensor x) {
    return x->get_accelerator()->ops().cos(x);
}

inline Tensor sqrt(Tensor x) {
    return x->get_accelerator()->ops().sqrt(x);
}

inline Tensor matmul(Tensor a, Tensor b) {
    return a->get_accelerator()->ops().matmul(a, b);
}

inline Tensor dot(Tensor a, Tensor b) {
    return a->get_accelerator()->ops().dot(a, b);
}

inline Tensor transpose(Tensor x) {
    return x->get_accelerator()->ops().transpose(x);
}

inline Tensor permute(Tensor x, const std::vector<u32>& dims) {
    return x->get_accelerator()->ops().permute(x, dims);
}

inline Tensor sum(Tensor x, u32 axis) {
    return x->get_accelerator()->ops().sum_axis(x, axis);
}

inline Tensor sum_axis(Tensor x, u32 axis) {
    return sum(x, axis);
}

inline Tensor mean(Tensor x, u32 axis) {
    return x->get_accelerator()->ops().mean_axis(x, axis);
}

inline Tensor mean_axis(Tensor x, u32 axis) {
    return mean(x, axis);
}

inline Tensor min(Tensor x, u32 axis) {
    return x->get_accelerator()->ops().min_axis(x, axis);
}

inline Tensor min_axis(Tensor x, u32 axis) {
    return min(x, axis);
}

inline Tensor max(Tensor x, u32 axis) {
    return x->get_accelerator()->ops().max_axis(x, axis);
}

inline Tensor max_axis(Tensor x, u32 axis) {
    return max(x, axis);
}

inline Tensor layer_norm(Tensor x, Tensor gamma, Tensor beta, float epsilon = 1e-5f) {
    return x->get_accelerator()->ops().layer_norm(x, gamma, beta, epsilon);
}

inline Tensor cat(const std::vector<Tensor>& tensors, u32 axis = 0) {
    if (tensors.empty()) throw std::invalid_argument("Cannot concatenate empty tensor list");
    return tensors[0]->get_accelerator()->ops().concatenate(tensors, axis);
}

inline Tensor concatenate(const std::vector<Tensor>& tensors, u32 axis = 0) {
    return cat(tensors, axis);
}

inline std::vector<Tensor> split(Tensor x, u32 num_splits, u32 axis = 0) {
    return x->get_accelerator()->ops().split(x, num_splits, axis);
}

inline Tensor squeeze(Tensor x, int axis = -1) {
    return x->get_accelerator()->ops().squeeze(x, axis);
}

inline Tensor unsqueeze(Tensor x, u32 axis) {
    return x->get_accelerator()->ops().unsqueeze(x, axis);
}

inline Tensor reshape(Tensor x, const std::vector<u32>& shape) {
    return x->create_reshaped_view(shape);
}

inline Tensor flatten(Tensor x) {
    return x->create_flattened_view();
}

inline Tensor slice(Tensor x, const std::vector<u32>& start, const std::vector<u32>& lengths) {
    return x->get_accelerator()->ops().slice(x, start, lengths);
}

inline std::vector<u32> shape(Tensor x) {
    return x->get_shape();
}

inline std::string shape_str(Tensor x) {
    return x->get_shape_string();
}

inline u32 ndim(Tensor x) {
    return x->get_rank();
}

inline u64 numel(Tensor x) {
    return x->get_element_count();
}

inline DataType dtype(Tensor x) {
    return x->get_dtype();
}

inline void save(Tensor x, const std::string& path) {
    x->get_accelerator()->ops().save_tensor(x, path);
}

inline Tensor load(const std::string& path) {
    return accelerator().ops().load_tensor(path);
}

}

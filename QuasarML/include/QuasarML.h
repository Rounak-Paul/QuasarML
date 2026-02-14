#pragma once

#include <Common/Types.h>
#include <Common/Assert.h>
#include <Common/Logger.h>
#include <Core/Device.h>
#include <Core/DeviceManager.h>
#include <Core/Tensor.h>
#include <Core/Kernel.h>
#include <Ops/Elementwise.h>
#include <Ops/Reduction.h>
#include <Ops/LinAlg.h>
#include <Ops/Manipulation.h>

namespace qsml {

using QuasarML::Device;
using QuasarML::DeviceManager;
using QuasarML::Tensor;
using QuasarML::Kernel;
using QuasarML::DataType;
using QuasarML::u8;
using QuasarML::u16;
using QuasarML::u32;
using QuasarML::u64;
using QuasarML::i8;
using QuasarML::i16;
using QuasarML::i32;
using QuasarML::i64;
using QuasarML::f32;
using QuasarML::f64;

using TensorPtr = std::shared_ptr<Tensor>;
using KernelPtr = std::shared_ptr<Kernel>;

namespace detail {
    inline Device*& current_device_ptr() {
        static Device* dev = nullptr;
        return dev;
    }
}

inline void init() {
    QuasarML::Logger::init();
}

inline void shutdown() {
    QuasarML::DeviceManager::instance().shutdown();
    QuasarML::Logger::shutdown();
}

inline u32 device_count() {
    return DeviceManager::instance().device_count();
}

inline std::vector<std::string> device_names() {
    return DeviceManager::instance().device_names();
}

inline void set_device(u32 device_id) {
    DeviceManager::instance().set_default(device_id);
    detail::current_device_ptr() = DeviceManager::instance().get(device_id);
}

inline u32 current_device_id() {
    return DeviceManager::instance().get_default();
}

inline Device& device() {
    if (!detail::current_device_ptr()) {
        detail::current_device_ptr() = DeviceManager::instance().get(0);
    }
    QS_ASSERT(detail::current_device_ptr(), "No GPU device available");
    return *detail::current_device_ptr();
}

inline Device& device(u32 device_id) {
    Device* dev = DeviceManager::instance().get(device_id);
    QS_ASSERT(dev, "Failed to get device");
    return *dev;
}

inline bool is_valid() {
    return detail::current_device_ptr() && detail::current_device_ptr()->is_valid();
}

inline void synchronize() {
    device().synchronize();
}

inline void flush_pending() {
    device().flush_pending();
}

inline TensorPtr zeros(const std::vector<u32>& shape, DataType dtype = DataType::F32) {
    auto t = device().create_tensor(shape, dtype);
    t->zero();
    return t;
}

inline TensorPtr ones(const std::vector<u32>& shape, DataType dtype = DataType::F32) {
    auto t = device().create_tensor(shape, dtype);
    if (dtype == DataType::F32) {
        f32 one = 1.0f;
        t->fill(&one);
    } else if (dtype == DataType::I32) {
        i32 one = 1;
        t->fill(&one);
    }
    return t;
}

inline TensorPtr empty(const std::vector<u32>& shape, DataType dtype = DataType::F32) {
    return device().create_tensor(shape, dtype);
}

inline TensorPtr tensor(const std::vector<f32>& data, const std::vector<u32>& shape, DataType dtype = DataType::F32) {
    auto t = device().create_tensor(shape, dtype);
    t->upload(data.data(), data.size() * sizeof(f32));
    return t;
}

inline TensorPtr from_data(const void* data, const std::vector<u32>& shape, DataType dtype = DataType::F32) {
    auto t = device().create_tensor(shape, dtype);
    t->upload(data, t->numel() * QuasarML::dtype_size(dtype));
    return t;
}

inline TensorPtr add(TensorPtr a, TensorPtr b) {
    return QuasarML::Ops::add(device(), a, b);
}

inline TensorPtr sub(TensorPtr a, TensorPtr b) {
    return QuasarML::Ops::subtract(device(), a, b);
}

inline TensorPtr mul(TensorPtr a, TensorPtr b) {
    return QuasarML::Ops::multiply(device(), a, b);
}

inline TensorPtr div(TensorPtr a, TensorPtr b) {
    return QuasarML::Ops::divide(device(), a, b);
}

inline TensorPtr neg(TensorPtr x) {
    return QuasarML::Ops::neg(device(), x);
}

inline TensorPtr abs(TensorPtr x) {
    return QuasarML::Ops::abs(device(), x);
}

inline TensorPtr add_scalar(TensorPtr x, f32 scalar) {
    return QuasarML::Ops::add_scalar(device(), x, scalar);
}

inline TensorPtr mul_scalar(TensorPtr x, f32 scalar) {
    return QuasarML::Ops::multiply_scalar(device(), x, scalar);
}

inline TensorPtr sqrt(TensorPtr x) {
    return QuasarML::Ops::sqrt(device(), x);
}

inline TensorPtr exp(TensorPtr x) {
    return QuasarML::Ops::exp(device(), x);
}

inline TensorPtr log(TensorPtr x) {
    return QuasarML::Ops::log(device(), x);
}

inline TensorPtr sin(TensorPtr x) {
    return QuasarML::Ops::sin(device(), x);
}

inline TensorPtr cos(TensorPtr x) {
    return QuasarML::Ops::cos(device(), x);
}

inline TensorPtr tanh(TensorPtr x) {
    return QuasarML::Ops::tanh(device(), x);
}

inline TensorPtr relu(TensorPtr x) {
    return QuasarML::Ops::relu(device(), x);
}

inline TensorPtr sigmoid(TensorPtr x) {
    return QuasarML::Ops::sigmoid(device(), x);
}

inline TensorPtr gelu(TensorPtr x) {
    return QuasarML::Ops::gelu(device(), x);
}

inline TensorPtr softmax(TensorPtr x, i32 axis = -1) {
    return QuasarML::Ops::softmax(device(), x, axis);
}

inline TensorPtr sum(TensorPtr x) {
    return QuasarML::Ops::sum(device(), x);
}

inline TensorPtr sum(TensorPtr x, i32 axis, bool keepdims = false) {
    return QuasarML::Ops::sum(device(), x, axis, keepdims);
}

inline TensorPtr mean(TensorPtr x) {
    return QuasarML::Ops::mean(device(), x);
}

inline TensorPtr mean(TensorPtr x, i32 axis, bool keepdims = false) {
    return QuasarML::Ops::mean(device(), x, axis, keepdims);
}

inline TensorPtr max(TensorPtr x) {
    return QuasarML::Ops::max(device(), x);
}

inline TensorPtr min(TensorPtr x) {
    return QuasarML::Ops::min(device(), x);
}

inline TensorPtr matmul(TensorPtr a, TensorPtr b) {
    return QuasarML::Ops::matmul(device(), a, b);
}

inline TensorPtr transpose(TensorPtr x) {
    return QuasarML::Ops::transpose(device(), x);
}

inline TensorPtr reshape(TensorPtr x, const std::vector<u32>& new_shape) {
    return QuasarML::Ops::reshape(device(), x, new_shape);
}

inline TensorPtr flatten(TensorPtr x) {
    return QuasarML::Ops::flatten(device(), x);
}

inline TensorPtr squeeze(TensorPtr x) {
    return QuasarML::Ops::squeeze(device(), x);
}

inline TensorPtr unsqueeze(TensorPtr x, i32 dim) {
    return QuasarML::Ops::unsqueeze(device(), x, dim);
}

inline TensorPtr copy(TensorPtr x) {
    return QuasarML::Ops::copy(device(), x);
}

}

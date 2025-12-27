#include "Elementwise.h"
#include "ShaderGen.h"
#include <Core/Kernel.h>
#include <Common/Assert.h>
#include <sstream>

namespace QuasarML {
namespace Ops {

namespace {

std::shared_ptr<Kernel> get_or_create_binary_kernel(Device& device, const char* name, const char* op, DataType dtype) {
    std::string key = std::string(name) + "_" + dtype_to_string(dtype);
    auto k = device.get_kernel(key);
    if (k) return k;
    
    std::string code = ShaderGen::elementwise_binary(op, dtype);
    return device.create_kernel(key, code, 3, sizeof(u32));
}

std::shared_ptr<Kernel> get_or_create_unary_kernel(Device& device, const char* name, const char* op, DataType dtype) {
    std::string key = std::string(name) + "_" + dtype_to_string(dtype);
    auto k = device.get_kernel(key);
    if (k) return k;
    
    std::string code = ShaderGen::elementwise_unary(op, dtype);
    return device.create_kernel(key, code, 2, sizeof(u32));
}

std::shared_ptr<Kernel> get_or_create_custom_kernel(Device& device, const std::string& key, const std::string& code, u32 bindings, u32 push_size) {
    auto k = device.get_kernel(key);
    if (k) return k;
    return device.create_kernel(key, code, bindings, push_size);
}

std::string scalar_binary_shader(const char* op, DataType dtype) {
    std::ostringstream ss;
    ss << ShaderGen::glsl_header();
    ss << ShaderGen::buffer_binding(0, "in_data", dtype, true);
    ss << ShaderGen::buffer_binding(1, "result", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; " << dtype_to_glsl(dtype) << " scalar; };\n\n";
    ss << ShaderGen::main_begin();
    ss << ShaderGen::global_index();
    ss << ShaderGen::bounds_check("n");
    ss << "    result[idx] = in_data[idx] " << op << " scalar;\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

std::string relu_shader(DataType dtype) {
    std::ostringstream ss;
    const char* type = dtype_to_glsl(dtype);
    ss << ShaderGen::glsl_header();
    ss << ShaderGen::buffer_binding(0, "in_data", dtype, true);
    ss << ShaderGen::buffer_binding(1, "result", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << ShaderGen::main_begin();
    ss << ShaderGen::global_index();
    ss << ShaderGen::bounds_check("n");
    ss << "    result[idx] = max(in_data[idx], " << type << "(0));\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

std::string sigmoid_shader(DataType dtype) {
    std::ostringstream ss;
    const char* type = dtype_to_glsl(dtype);
    ss << ShaderGen::glsl_header();
    ss << ShaderGen::buffer_binding(0, "in_data", dtype, true);
    ss << ShaderGen::buffer_binding(1, "result", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << ShaderGen::main_begin();
    ss << ShaderGen::global_index();
    ss << ShaderGen::bounds_check("n");
    ss << "    result[idx] = " << type << "(1) / (" << type << "(1) + exp(-in_data[idx]));\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

std::string gelu_shader(DataType dtype) {
    std::ostringstream ss;
    const char* type = dtype_to_glsl(dtype);
    ss << ShaderGen::glsl_header();
    ss << ShaderGen::buffer_binding(0, "in_data", dtype, true);
    ss << ShaderGen::buffer_binding(1, "result", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << ShaderGen::main_begin();
    ss << ShaderGen::global_index();
    ss << ShaderGen::bounds_check("n");
    ss << "    " << type << " x = in_data[idx];\n";
    ss << "    result[idx] = " << type << "(0.5) * x * (" << type << "(1) + tanh(" << type << "(0.7978845608) * (x + " << type << "(0.044715) * x * x * x)));\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

std::string softmax_shader(DataType dtype) {
    std::ostringstream ss;
    const char* type = dtype_to_glsl(dtype);
    ss << "#version 450\n";
    ss << "layout(local_size_x = 256) in;\n\n";
    ss << ShaderGen::buffer_binding(0, "in_data", dtype, true);
    ss << ShaderGen::buffer_binding(1, "out_data", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint batch_size; uint axis_size; };\n\n";
    ss << "shared " << type << " shared_data[256];\n\n";
    ss << ShaderGen::main_begin();
    ss << "    uint batch_idx = gl_WorkGroupID.x;\n";
    ss << "    uint tid = gl_LocalInvocationID.x;\n";
    ss << "    uint offset = batch_idx * axis_size;\n";
    ss << "    " << type << " local_max = " << type << "(-3.402823466e+38);\n";
    ss << "    for (uint i = tid; i < axis_size; i += 256) local_max = max(local_max, in_data[offset + i]);\n";
    ss << "    shared_data[tid] = local_max;\n";
    ss << "    barrier();\n";
    ss << "    for (uint s = 128; s > 0; s >>= 1) { if (tid < s) shared_data[tid] = max(shared_data[tid], shared_data[tid + s]); barrier(); }\n";
    ss << "    " << type << " max_val = shared_data[0];\n";
    ss << "    barrier();\n";
    ss << "    " << type << " local_sum = " << type << "(0);\n";
    ss << "    for (uint i = tid; i < axis_size; i += 256) local_sum += exp(in_data[offset + i] - max_val);\n";
    ss << "    shared_data[tid] = local_sum;\n";
    ss << "    barrier();\n";
    ss << "    for (uint s = 128; s > 0; s >>= 1) { if (tid < s) shared_data[tid] += shared_data[tid + s]; barrier(); }\n";
    ss << "    " << type << " sum_val = shared_data[0];\n";
    ss << "    barrier();\n";
    ss << "    for (uint i = tid; i < axis_size; i += 256) out_data[offset + i] = exp(in_data[offset + i] - max_val) / sum_val;\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

}

std::shared_ptr<Tensor> add(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    QS_ASSERT(a->shape() == b->shape(), "Shape mismatch");
    QS_ASSERT(a->dtype() == b->dtype(), "Type mismatch");
    
    auto result = device.create_tensor(a->shape(), a->dtype());
    auto kernel = get_or_create_binary_kernel(device, "add", "+", a->dtype());
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, b);
    kernel->bind(2, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> subtract(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    QS_ASSERT(a->shape() == b->shape(), "Shape mismatch");
    QS_ASSERT(a->dtype() == b->dtype(), "Type mismatch");
    
    auto result = device.create_tensor(a->shape(), a->dtype());
    auto kernel = get_or_create_binary_kernel(device, "sub", "-", a->dtype());
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, b);
    kernel->bind(2, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> multiply(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    QS_ASSERT(a->shape() == b->shape(), "Shape mismatch");
    QS_ASSERT(a->dtype() == b->dtype(), "Type mismatch");
    
    auto result = device.create_tensor(a->shape(), a->dtype());
    auto kernel = get_or_create_binary_kernel(device, "mul", "*", a->dtype());
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, b);
    kernel->bind(2, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> divide(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    QS_ASSERT(a->shape() == b->shape(), "Shape mismatch");
    QS_ASSERT(a->dtype() == b->dtype(), "Type mismatch");
    
    auto result = device.create_tensor(a->shape(), a->dtype());
    auto kernel = get_or_create_binary_kernel(device, "div", "/", a->dtype());
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, b);
    kernel->bind(2, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> add_scalar(Device& device, std::shared_ptr<Tensor> a, f32 scalar) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    
    std::string key = "add_scalar_" + std::string(dtype_to_string(a->dtype()));
    struct { u32 n; f32 scalar; } pc;
    auto kernel = get_or_create_custom_kernel(device, key, scalar_binary_shader("+", a->dtype()), 2, sizeof(pc));
    
    pc.n = static_cast<u32>(a->numel());
    pc.scalar = scalar;
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(pc.n);
    kernel->execute(groups, 1, 1, &pc);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> multiply_scalar(Device& device, std::shared_ptr<Tensor> a, f32 scalar) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    
    std::string key = "mul_scalar_" + std::string(dtype_to_string(a->dtype()));
    struct { u32 n; f32 scalar; } pc;
    auto kernel = get_or_create_custom_kernel(device, key, scalar_binary_shader("*", a->dtype()), 2, sizeof(pc));
    
    pc.n = static_cast<u32>(a->numel());
    pc.scalar = scalar;
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(pc.n);
    kernel->execute(groups, 1, 1, &pc);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> neg(Device& device, std::shared_ptr<Tensor> a) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    auto kernel = get_or_create_unary_kernel(device, "neg", "-", a->dtype());
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> abs(Device& device, std::shared_ptr<Tensor> a) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    auto kernel = get_or_create_unary_kernel(device, "abs", "abs", a->dtype());
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> sqrt(Device& device, std::shared_ptr<Tensor> a) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    auto kernel = get_or_create_unary_kernel(device, "sqrt", "sqrt", a->dtype());
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> exp(Device& device, std::shared_ptr<Tensor> a) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    auto kernel = get_or_create_unary_kernel(device, "exp", "exp", a->dtype());
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> log(Device& device, std::shared_ptr<Tensor> a) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    auto kernel = get_or_create_unary_kernel(device, "log", "log", a->dtype());
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> sin(Device& device, std::shared_ptr<Tensor> a) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    auto kernel = get_or_create_unary_kernel(device, "sin", "sin", a->dtype());
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> cos(Device& device, std::shared_ptr<Tensor> a) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    auto kernel = get_or_create_unary_kernel(device, "cos", "cos", a->dtype());
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> tanh(Device& device, std::shared_ptr<Tensor> a) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    auto kernel = get_or_create_unary_kernel(device, "tanh", "tanh", a->dtype());
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> relu(Device& device, std::shared_ptr<Tensor> a) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    
    std::string key = "relu_" + std::string(dtype_to_string(a->dtype()));
    auto kernel = get_or_create_custom_kernel(device, key, relu_shader(a->dtype()), 2, sizeof(u32));
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> sigmoid(Device& device, std::shared_ptr<Tensor> a) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    
    std::string key = "sigmoid_" + std::string(dtype_to_string(a->dtype()));
    auto kernel = get_or_create_custom_kernel(device, key, sigmoid_shader(a->dtype()), 2, sizeof(u32));
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> gelu(Device& device, std::shared_ptr<Tensor> a) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    
    std::string key = "gelu_" + std::string(dtype_to_string(a->dtype()));
    auto kernel = get_or_create_custom_kernel(device, key, gelu_shader(a->dtype()), 2, sizeof(u32));
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> softmax(Device& device, std::shared_ptr<Tensor> a, i32 axis) {
    if (axis < 0) axis = static_cast<i32>(a->rank()) + axis;
    QS_ASSERT(axis >= 0 && axis < static_cast<i32>(a->rank()), "Invalid axis");
    
    auto result = device.create_tensor(a->shape(), a->dtype());
    
    std::string key = "softmax_" + std::string(dtype_to_string(a->dtype()));
    struct { u32 batch_size; u32 axis_size; } pc;
    auto kernel = get_or_create_custom_kernel(device, key, softmax_shader(a->dtype()), 2, sizeof(pc));
    
    u64 axis_size = a->dim(axis);
    u64 batch_size = a->numel() / axis_size;
    
    pc.batch_size = static_cast<u32>(batch_size);
    pc.axis_size = static_cast<u32>(axis_size);
    
    kernel->bind(0, a);
    kernel->bind(1, result);
    kernel->execute(static_cast<u32>(batch_size), 1, 1, &pc);
    device.synchronize();
    
    return result;
}

}
}

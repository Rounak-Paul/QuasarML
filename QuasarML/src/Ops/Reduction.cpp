#include "Reduction.h"
#include "Elementwise.h"
#include "ShaderGen.h"
#include "DeviceTuning.h"
#include <Core/Kernel.h>
#include <Common/Assert.h>
#include <sstream>

namespace QuasarML {
namespace Ops {

namespace {

std::shared_ptr<Kernel> get_or_create_kernel(Device& device, const std::string& key, const std::string& code, u32 bindings, u32 push_size) {
    auto k = device.get_kernel(key);
    if (k) return k;
    return device.create_kernel(key, code, bindings, push_size);
}

std::string sum_shader(DataType dtype, u32 workgroup_size) {
    std::ostringstream ss;
    const char* type = dtype_to_glsl(dtype);
    ss << "#version 450\n";
    ss << "layout(local_size_x = " << workgroup_size << ") in;\n\n";
    ss << ShaderGen::buffer_binding(0, "in_data", dtype, true);
    ss << ShaderGen::buffer_binding(1, "out_data", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << "shared " << type << " shared_data[" << workgroup_size << "];\n\n";
    ss << ShaderGen::main_begin();
    ss << "    uint tid = gl_LocalInvocationID.x;\n";
    ss << "    uint gid = gl_GlobalInvocationID.x;\n";
    ss << "    shared_data[tid] = (gid < n) ? in_data[gid] : " << type << "(0);\n";
    ss << "    barrier();\n";
    for (u32 s = workgroup_size / 2; s > 0; s >>= 1) {
        ss << "    if (tid < " << s << ") shared_data[tid] += shared_data[tid + " << s << "];\n";
        ss << "    barrier();\n";
    }
    ss << "    if (tid == 0) out_data[gl_WorkGroupID.x] = shared_data[0];\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

std::string max_shader(DataType dtype, u32 workgroup_size) {
    std::ostringstream ss;
    const char* type = dtype_to_glsl(dtype);
    const char* init = "-3.402823466e+38";
    ss << "#version 450\n";
    ss << "layout(local_size_x = " << workgroup_size << ") in;\n\n";
    ss << ShaderGen::buffer_binding(0, "in_data", dtype, true);
    ss << ShaderGen::buffer_binding(1, "out_data", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << "shared " << type << " shared_data[" << workgroup_size << "];\n\n";
    ss << ShaderGen::main_begin();
    ss << "    uint tid = gl_LocalInvocationID.x;\n";
    ss << "    uint gid = gl_GlobalInvocationID.x;\n";
    ss << "    shared_data[tid] = (gid < n) ? in_data[gid] : " << type << "(" << init << ");\n";
    ss << "    barrier();\n";
    for (u32 s = workgroup_size / 2; s > 0; s >>= 1) {
        ss << "    if (tid < " << s << ") shared_data[tid] = max(shared_data[tid], shared_data[tid + " << s << "]);\n";
        ss << "    barrier();\n";
    }
    ss << "    if (tid == 0) out_data[gl_WorkGroupID.x] = shared_data[0];\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

std::string min_shader(DataType dtype, u32 workgroup_size) {
    std::ostringstream ss;
    const char* type = dtype_to_glsl(dtype);
    const char* init = "3.402823466e+38";
    ss << "#version 450\n";
    ss << "layout(local_size_x = " << workgroup_size << ") in;\n\n";
    ss << ShaderGen::buffer_binding(0, "in_data", dtype, true);
    ss << ShaderGen::buffer_binding(1, "out_data", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << "shared " << type << " shared_data[" << workgroup_size << "];\n\n";
    ss << ShaderGen::main_begin();
    ss << "    uint tid = gl_LocalInvocationID.x;\n";
    ss << "    uint gid = gl_GlobalInvocationID.x;\n";
    ss << "    shared_data[tid] = (gid < n) ? in_data[gid] : " << type << "(" << init << ");\n";
    ss << "    barrier();\n";
    for (u32 s = workgroup_size / 2; s > 0; s >>= 1) {
        ss << "    if (tid < " << s << ") shared_data[tid] = min(shared_data[tid], shared_data[tid + " << s << "]);\n";
        ss << "    barrier();\n";
    }
    ss << "    if (tid == 0) out_data[gl_WorkGroupID.x] = shared_data[0];\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

std::string prod_shader(DataType dtype, u32 workgroup_size) {
    std::ostringstream ss;
    const char* type = dtype_to_glsl(dtype);
    ss << "#version 450\n";
    ss << "layout(local_size_x = " << workgroup_size << ") in;\n\n";
    ss << ShaderGen::buffer_binding(0, "in_data", dtype, true);
    ss << ShaderGen::buffer_binding(1, "out_data", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << "shared " << type << " shared_data[" << workgroup_size << "];\n\n";
    ss << ShaderGen::main_begin();
    ss << "    uint tid = gl_LocalInvocationID.x;\n";
    ss << "    uint gid = gl_GlobalInvocationID.x;\n";
    ss << "    shared_data[tid] = (gid < n) ? in_data[gid] : " << type << "(1);\n";
    ss << "    barrier();\n";
    for (u32 s = workgroup_size / 2; s > 0; s >>= 1) {
        ss << "    if (tid < " << s << ") shared_data[tid] *= shared_data[tid + " << s << "];\n";
        ss << "    barrier();\n";
    }
    ss << "    if (tid == 0) out_data[gl_WorkGroupID.x] = shared_data[0];\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

std::shared_ptr<Tensor> reduce_full(Device& device, std::shared_ptr<Tensor> a, const std::string& op_name,
                                     std::string(*shader_fn)(DataType, u32)) {
    u32 wg_size = get_device_params(device).reduction_workgroup;
    u64 n = a->numel();
    u32 num_workgroups = static_cast<u32>((n + wg_size - 1) / wg_size);
    
    std::string key = op_name + "_" + dtype_to_string(a->dtype()) + "_wg" + std::to_string(wg_size);
    auto kernel = get_or_create_kernel(device, key, shader_fn(a->dtype(), wg_size), 2, sizeof(u32));
    
    auto temp = device.create_tensor({num_workgroups}, a->dtype());
    temp->zero();
    
    u32 count = static_cast<u32>(n);
    kernel->bind(0, a);
    kernel->bind(1, temp);
    u32 groups = kernel->optimal_dispatch_1d(count);
    kernel->execute(groups, 1, 1, &count);
    device.synchronize();
    
    while (num_workgroups > 1) {
        u32 next_workgroups = (num_workgroups + wg_size - 1) / wg_size;
        auto next = device.create_tensor({next_workgroups}, a->dtype());
        next->zero();
        
        kernel->bind(0, temp);
        kernel->bind(1, next);
        u32 g = kernel->optimal_dispatch_1d(num_workgroups);
        kernel->execute(g, 1, 1, &num_workgroups);
        device.synchronize();
        
        temp = next;
        num_workgroups = next_workgroups;
    }
    
    return temp;
}

}

std::shared_ptr<Tensor> sum(Device& device, std::shared_ptr<Tensor> a) {
    return reduce_full(device, a, "reduce_sum", sum_shader);
}

std::shared_ptr<Tensor> sum(Device& device, std::shared_ptr<Tensor> a, i32 axis, bool keepdims) {
    if (axis < 0) axis = static_cast<i32>(a->rank()) + axis;
    QS_ASSERT(axis >= 0 && axis < static_cast<i32>(a->rank()), "Invalid axis");
    
    std::vector<u32> out_shape;
    for (u32 i = 0; i < a->rank(); ++i) {
        if (static_cast<i32>(i) == axis) {
            if (keepdims) out_shape.push_back(1);
        } else {
            out_shape.push_back(a->dim(i));
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);
    
    auto result = device.create_tensor(out_shape, a->dtype());
    result->zero();
    
    u64 outer = 1, inner = 1, axis_size = a->dim(axis);
    for (i32 i = 0; i < axis; ++i) outer *= a->dim(i);
    for (u32 i = axis + 1; i < a->rank(); ++i) inner *= a->dim(i);
    
    u32 wg_size = get_device_params(device).elementwise_workgroup;
    std::ostringstream ss;
    const char* type = dtype_to_glsl(a->dtype());
    ss << "#version 450\n";
    ss << "layout(local_size_x = " << wg_size << ") in;\n";
    ss << ShaderGen::buffer_binding(0, "in_data", a->dtype(), true);
    ss << ShaderGen::buffer_binding(1, "out_data", a->dtype(), false);
    ss << "layout(push_constant) uniform PushConstants { uint outer; uint axis_size; uint inner; };\n\n";
    ss << ShaderGen::main_begin();
    ss << "    uint idx = gl_GlobalInvocationID.x;\n";
    ss << "    if (idx >= outer * inner) return;\n";
    ss << "    uint o = idx / inner;\n";
    ss << "    uint i = idx % inner;\n";
    ss << "    " << type << " acc = " << type << "(0);\n";
    ss << "    for (uint k = 0; k < axis_size; k++) acc += in_data[(o * axis_size + k) * inner + i];\n";
    ss << "    out_data[idx] = acc;\n";
    ss << ShaderGen::main_end();
    
    std::string key = "sum_axis_" + std::string(dtype_to_string(a->dtype())) + "_wg" + std::to_string(wg_size);
    struct { u32 outer; u32 axis_size; u32 inner; } pc = {
        static_cast<u32>(outer), static_cast<u32>(axis_size), static_cast<u32>(inner)
    };
    auto kernel = get_or_create_kernel(device, key, ss.str(), 2, sizeof(pc));
    
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 n = static_cast<u32>(outer * inner);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &pc);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> mean(Device& device, std::shared_ptr<Tensor> a) {
    auto s = sum(device, a);
    return multiply_scalar(device, s, 1.0f / static_cast<f32>(a->numel()));
}

std::shared_ptr<Tensor> mean(Device& device, std::shared_ptr<Tensor> a, i32 axis, bool keepdims) {
    auto s = sum(device, a, axis, keepdims);
    if (axis < 0) axis = static_cast<i32>(a->rank()) + axis;
    return multiply_scalar(device, s, 1.0f / static_cast<f32>(a->dim(axis)));
}

std::shared_ptr<Tensor> max(Device& device, std::shared_ptr<Tensor> a) {
    return reduce_full(device, a, "reduce_max", max_shader);
}

std::shared_ptr<Tensor> max(Device& device, std::shared_ptr<Tensor> a, i32 axis, bool keepdims) {
    (void)axis; (void)keepdims;
    return max(device, a);
}

std::shared_ptr<Tensor> min(Device& device, std::shared_ptr<Tensor> a) {
    return reduce_full(device, a, "reduce_min", min_shader);
}

std::shared_ptr<Tensor> min(Device& device, std::shared_ptr<Tensor> a, i32 axis, bool keepdims) {
    (void)axis; (void)keepdims;
    return min(device, a);
}

std::shared_ptr<Tensor> prod(Device& device, std::shared_ptr<Tensor> a) {
    return reduce_full(device, a, "reduce_prod", prod_shader);
}

std::shared_ptr<Tensor> argmax(Device& device, std::shared_ptr<Tensor> a, i32 axis) {
    (void)axis;
    return max(device, a);
}

std::shared_ptr<Tensor> argmin(Device& device, std::shared_ptr<Tensor> a, i32 axis) {
    (void)axis;
    return min(device, a);
}

}
}

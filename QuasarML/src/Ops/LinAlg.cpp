#include "LinAlg.h"
#include "ShaderGen.h"
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

std::string matmul_shader(DataType dtype) {
    std::ostringstream ss;
    const char* type = dtype_to_glsl(dtype);
    ss << "#version 450\n";
    ss << "layout(local_size_x = 16, local_size_y = 16) in;\n\n";
    ss << ShaderGen::buffer_binding(0, "A", dtype, true);
    ss << ShaderGen::buffer_binding(1, "B", dtype, true);
    ss << ShaderGen::buffer_binding(2, "C", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint M; uint K; uint N; };\n\n";
    ss << ShaderGen::main_begin();
    ss << "    uint row = gl_GlobalInvocationID.y;\n";
    ss << "    uint col = gl_GlobalInvocationID.x;\n";
    ss << "    if (row >= M || col >= N) return;\n";
    ss << "    " << type << " acc = " << type << "(0);\n";
    ss << "    for (uint k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];\n";
    ss << "    C[row * N + col] = acc;\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

std::string transpose_shader(DataType dtype) {
    std::ostringstream ss;
    ss << "#version 450\n";
    ss << "layout(local_size_x = 256) in;\n\n";
    ss << ShaderGen::buffer_binding(0, "in_data", dtype, true);
    ss << ShaderGen::buffer_binding(1, "out_data", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint rows; uint cols; };\n\n";
    ss << ShaderGen::main_begin();
    ss << "    uint idx = gl_GlobalInvocationID.x;\n";
    ss << "    if (idx >= rows * cols) return;\n";
    ss << "    uint r = idx / cols;\n";
    ss << "    uint c = idx % cols;\n";
    ss << "    out_data[c * rows + r] = in_data[idx];\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

std::string dot_shader(DataType dtype, u32 workgroup_size) {
    std::ostringstream ss;
    const char* type = dtype_to_glsl(dtype);
    ss << "#version 450\n";
    ss << "layout(local_size_x = " << workgroup_size << ") in;\n\n";
    ss << ShaderGen::buffer_binding(0, "A", dtype, true);
    ss << ShaderGen::buffer_binding(1, "B", dtype, true);
    ss << ShaderGen::buffer_binding(2, "C", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << "shared " << type << " shared_data[" << workgroup_size << "];\n\n";
    ss << ShaderGen::main_begin();
    ss << "    uint tid = gl_LocalInvocationID.x;\n";
    ss << "    uint gid = gl_GlobalInvocationID.x;\n";
    ss << "    shared_data[tid] = (gid < n) ? A[gid] * B[gid] : " << type << "(0);\n";
    ss << "    barrier();\n";
    for (u32 s = workgroup_size / 2; s > 0; s >>= 1) {
        ss << "    if (tid < " << s << ") shared_data[tid] += shared_data[tid + " << s << "];\n";
        ss << "    barrier();\n";
    }
    ss << "    if (tid == 0) C[gl_WorkGroupID.x] = shared_data[0];\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

std::string outer_shader(DataType dtype) {
    std::ostringstream ss;
    ss << "#version 450\n";
    ss << "layout(local_size_x = 256) in;\n\n";
    ss << ShaderGen::buffer_binding(0, "A", dtype, true);
    ss << ShaderGen::buffer_binding(1, "B", dtype, true);
    ss << ShaderGen::buffer_binding(2, "C", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint M; uint N; };\n\n";
    ss << ShaderGen::main_begin();
    ss << "    uint idx = gl_GlobalInvocationID.x;\n";
    ss << "    if (idx >= M * N) return;\n";
    ss << "    uint r = idx / N;\n";
    ss << "    uint c = idx % N;\n";
    ss << "    C[idx] = A[r] * B[c];\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

}

std::shared_ptr<Tensor> matmul(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    QS_ASSERT(a->rank() == 2 && b->rank() == 2, "matmul requires 2D tensors");
    QS_ASSERT(a->dim(1) == b->dim(0), "Inner dimensions must match");
    QS_ASSERT(a->dtype() == b->dtype(), "Type mismatch");
    
    u32 M = a->dim(0), K = a->dim(1), N = b->dim(1);
    auto result = device.create_tensor({M, N}, a->dtype());
    
    std::string key = "matmul_" + std::string(dtype_to_string(a->dtype()));
    struct { u32 M; u32 K; u32 N; } pc = { M, K, N };
    auto kernel = get_or_create_kernel(device, key, matmul_shader(a->dtype()), 3, sizeof(pc));
    
    kernel->bind(0, a);
    kernel->bind(1, b);
    kernel->bind(2, result);
    
    u32 gx = (N + 15) / 16;
    u32 gy = (M + 15) / 16;
    kernel->execute(gx, gy, 1, &pc);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> batch_matmul(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    QS_ASSERT(a->rank() == 3 && b->rank() == 3, "batch_matmul requires 3D tensors");
    QS_ASSERT(a->dim(0) == b->dim(0), "Batch size must match");
    QS_ASSERT(a->dim(2) == b->dim(1), "Inner dimensions must match");
    QS_ASSERT(a->dtype() == b->dtype(), "Type mismatch");
    
    u32 batch = a->dim(0), M = a->dim(1), K = a->dim(2), N = b->dim(2);
    auto result = device.create_tensor({batch, M, N}, a->dtype());
    
    for (u32 i = 0; i < batch; ++i) {
        auto slice_a = a->view({M, K});
        auto slice_b = b->view({K, N});
        auto slice_c = matmul(device, slice_a, slice_b);
        (void)slice_c;
    }
    
    return result;
}

std::shared_ptr<Tensor> dot(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    QS_ASSERT(a->numel() == b->numel(), "Tensors must have same number of elements");
    QS_ASSERT(a->dtype() == b->dtype(), "Type mismatch");
    
    constexpr u32 WG = 256;
    u64 n = a->numel();
    u32 num_workgroups = static_cast<u32>((n + WG - 1) / WG);
    
    std::string key = "dot_" + std::string(dtype_to_string(a->dtype()));
    auto kernel = get_or_create_kernel(device, key, dot_shader(a->dtype(), WG), 3, sizeof(u32));
    
    auto temp = device.create_tensor({num_workgroups}, a->dtype());
    
    u32 count = static_cast<u32>(n);
    kernel->bind(0, a);
    kernel->bind(1, b);
    kernel->bind(2, temp);
    u32 groups = kernel->optimal_dispatch_1d(count);
    kernel->execute(groups, 1, 1, &count);
    device.synchronize();
    
    while (num_workgroups > 1) {
        u32 next_workgroups = (num_workgroups + WG - 1) / WG;
        auto next = device.create_tensor({next_workgroups}, a->dtype());
        
        kernel->bind(0, temp);
        kernel->bind(1, temp);
        kernel->bind(2, next);
        u32 g = kernel->optimal_dispatch_1d(num_workgroups);
        kernel->execute(g, 1, 1, &num_workgroups);
        device.synchronize();
        
        temp = next;
        num_workgroups = next_workgroups;
    }
    
    return temp;
}

std::shared_ptr<Tensor> transpose(Device& device, std::shared_ptr<Tensor> a) {
    QS_ASSERT(a->rank() == 2, "transpose requires 2D tensor");
    
    u32 rows = a->dim(0), cols = a->dim(1);
    auto result = device.create_tensor({cols, rows}, a->dtype());
    
    std::string key = "transpose_" + std::string(dtype_to_string(a->dtype()));
    struct { u32 rows; u32 cols; } pc = { rows, cols };
    auto kernel = get_or_create_kernel(device, key, transpose_shader(a->dtype()), 2, sizeof(pc));
    
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 n = rows * cols;
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &pc);
    device.synchronize();
    
    return result;
}

std::shared_ptr<Tensor> transpose(Device& device, std::shared_ptr<Tensor> a, i32 dim0, i32 dim1) {
    (void)dim0; (void)dim1;
    return transpose(device, a);
}

std::shared_ptr<Tensor> outer(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    QS_ASSERT(a->rank() == 1 && b->rank() == 1, "outer requires 1D tensors");
    QS_ASSERT(a->dtype() == b->dtype(), "Type mismatch");
    
    u32 M = static_cast<u32>(a->numel());
    u32 N = static_cast<u32>(b->numel());
    auto result = device.create_tensor({M, N}, a->dtype());
    
    std::string key = "outer_" + std::string(dtype_to_string(a->dtype()));
    struct { u32 M; u32 N; } pc = { M, N };
    auto kernel = get_or_create_kernel(device, key, outer_shader(a->dtype()), 3, sizeof(pc));
    
    kernel->bind(0, a);
    kernel->bind(1, b);
    kernel->bind(2, result);
    u32 n = M * N;
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &pc);
    device.synchronize();
    
    return result;
}

}
}

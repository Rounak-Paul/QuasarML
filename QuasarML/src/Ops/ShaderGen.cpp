#include "ShaderGen.h"
#include <sstream>

namespace QuasarML {
namespace ShaderGen {

std::string glsl_header(u32 local_size_x) {
    std::ostringstream ss;
    ss << "#version 450\n";
    ss << "layout(local_size_x = " << local_size_x << ") in;\n\n";
    return ss.str();
}

std::string buffer_binding(u32 binding, const char* name, DataType dtype, bool readonly) {
    std::ostringstream ss;
    ss << "layout(std430, binding = " << binding << ") ";
    if (readonly) ss << "readonly ";
    ss << "buffer " << name << "_buf { " << dtype_to_glsl(dtype) << " " << name << "[]; };\n";
    return ss.str();
}

std::string main_begin() {
    return "void main() {\n";
}

std::string main_end() {
    return "}\n";
}

std::string global_index() {
    return "    uint idx = gl_GlobalInvocationID.x;\n";
}

std::string bounds_check(const char* size_var) {
    std::ostringstream ss;
    ss << "    if (idx >= " << size_var << ") return;\n";
    return ss.str();
}

std::string elementwise_binary(const char* op, DataType dtype) {
    std::ostringstream ss;
    ss << glsl_header();
    ss << buffer_binding(0, "a", dtype, true);
    ss << buffer_binding(1, "b", dtype, true);
    ss << buffer_binding(2, "result", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << main_begin();
    ss << global_index();
    ss << bounds_check("n");
    ss << "    result[idx] = a[idx] " << op << " b[idx];\n";
    ss << main_end();
    return ss.str();
}

std::string elementwise_unary(const char* op, DataType dtype) {
    std::ostringstream ss;
    ss << glsl_header();
    ss << buffer_binding(0, "in_data", dtype, true);
    ss << buffer_binding(1, "result", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << main_begin();
    ss << global_index();
    ss << bounds_check("n");
    ss << "    result[idx] = " << op << "(in_data[idx]);\n";
    ss << main_end();
    return ss.str();
}

std::string fill_shader(DataType dtype) {
    std::ostringstream ss;
    ss << glsl_header();
    ss << buffer_binding(0, "data", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; " << dtype_to_glsl(dtype) << " value; };\n\n";
    ss << main_begin();
    ss << global_index();
    ss << bounds_check("n");
    ss << "    data[idx] = value;\n";
    ss << main_end();
    return ss.str();
}

std::string copy_shader(DataType dtype) {
    std::ostringstream ss;
    ss << glsl_header();
    ss << buffer_binding(0, "src", dtype, true);
    ss << buffer_binding(1, "dst", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << main_begin();
    ss << global_index();
    ss << bounds_check("n");
    ss << "    dst[idx] = src[idx];\n";
    ss << main_end();
    return ss.str();
}

std::string reduce_sum_shader(DataType dtype, u32 workgroup_size) {
    std::ostringstream ss;
    ss << "#version 450\n";
    ss << "layout(local_size_x = " << workgroup_size << ") in;\n\n";
    ss << buffer_binding(0, "in_data", dtype, true);
    ss << buffer_binding(1, "out_data", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << "shared " << dtype_to_glsl(dtype) << " shared_data[" << workgroup_size << "];\n\n";
    ss << main_begin();
    ss << "    uint tid = gl_LocalInvocationID.x;\n";
    ss << "    uint gid = gl_GlobalInvocationID.x;\n";
    ss << "    shared_data[tid] = (gid < n) ? in_data[gid] : " << dtype_to_glsl(dtype) << "(0);\n";
    ss << "    barrier();\n";
    for (u32 s = workgroup_size / 2; s > 0; s >>= 1) {
        ss << "    if (tid < " << s << ") shared_data[tid] += shared_data[tid + " << s << "];\n";
        ss << "    barrier();\n";
    }
    ss << "    if (tid == 0) atomicAdd(out_data[0], shared_data[0]);\n";
    ss << main_end();
    return ss.str();
}

std::string reduce_max_shader(DataType dtype, u32 workgroup_size) {
    std::ostringstream ss;
    const char* type = dtype_to_glsl(dtype);
    const char* min_val = (dtype == DataType::I32) ? "-2147483647" : "-3.402823466e+38";
    
    ss << "#version 450\n";
    ss << "layout(local_size_x = " << workgroup_size << ") in;\n\n";
    ss << buffer_binding(0, "in_data", dtype, true);
    ss << buffer_binding(1, "out_data", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << "shared " << type << " shared_data[" << workgroup_size << "];\n\n";
    ss << main_begin();
    ss << "    uint tid = gl_LocalInvocationID.x;\n";
    ss << "    uint gid = gl_GlobalInvocationID.x;\n";
    ss << "    shared_data[tid] = (gid < n) ? in_data[gid] : " << type << "(" << min_val << ");\n";
    ss << "    barrier();\n";
    for (u32 s = workgroup_size / 2; s > 0; s >>= 1) {
        ss << "    if (tid < " << s << ") shared_data[tid] = max(shared_data[tid], shared_data[tid + " << s << "]);\n";
        ss << "    barrier();\n";
    }
    ss << "    if (tid == 0) out_data[gl_WorkGroupID.x] = shared_data[0];\n";
    ss << main_end();
    return ss.str();
}

std::string reduce_min_shader(DataType dtype, u32 workgroup_size) {
    std::ostringstream ss;
    const char* type = dtype_to_glsl(dtype);
    const char* max_val = (dtype == DataType::I32) ? "2147483647" : "3.402823466e+38";
    
    ss << "#version 450\n";
    ss << "layout(local_size_x = " << workgroup_size << ") in;\n\n";
    ss << buffer_binding(0, "in_data", dtype, true);
    ss << buffer_binding(1, "out_data", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << "shared " << type << " shared_data[" << workgroup_size << "];\n\n";
    ss << main_begin();
    ss << "    uint tid = gl_LocalInvocationID.x;\n";
    ss << "    uint gid = gl_GlobalInvocationID.x;\n";
    ss << "    shared_data[tid] = (gid < n) ? in_data[gid] : " << type << "(" << max_val << ");\n";
    ss << "    barrier();\n";
    for (u32 s = workgroup_size / 2; s > 0; s >>= 1) {
        ss << "    if (tid < " << s << ") shared_data[tid] = min(shared_data[tid], shared_data[tid + " << s << "]);\n";
        ss << "    barrier();\n";
    }
    ss << "    if (tid == 0) out_data[gl_WorkGroupID.x] = shared_data[0];\n";
    ss << main_end();
    return ss.str();
}

std::string matmul_shader(DataType dtype, u32 tile_size) {
    std::ostringstream ss;
    const char* type = dtype_to_glsl(dtype);
    
    ss << "#version 450\n";
    ss << "layout(local_size_x = " << tile_size << ", local_size_y = " << tile_size << ") in;\n\n";
    ss << buffer_binding(0, "A", dtype, true);
    ss << buffer_binding(1, "B", dtype, true);
    ss << buffer_binding(2, "C", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint M; uint N; uint K; };\n\n";
    ss << "shared " << type << " As[" << tile_size << "][" << tile_size << "];\n";
    ss << "shared " << type << " Bs[" << tile_size << "][" << tile_size << "];\n\n";
    ss << main_begin();
    ss << "    uint row = gl_GlobalInvocationID.y;\n";
    ss << "    uint col = gl_GlobalInvocationID.x;\n";
    ss << "    uint local_row = gl_LocalInvocationID.y;\n";
    ss << "    uint local_col = gl_LocalInvocationID.x;\n";
    ss << "    " << type << " sum = " << type << "(0);\n";
    ss << "    for (uint t = 0; t < (K + " << tile_size - 1 << ") / " << tile_size << "; t++) {\n";
    ss << "        uint a_col = t * " << tile_size << " + local_col;\n";
    ss << "        uint b_row = t * " << tile_size << " + local_row;\n";
    ss << "        As[local_row][local_col] = (row < M && a_col < K) ? A[row * K + a_col] : " << type << "(0);\n";
    ss << "        Bs[local_row][local_col] = (b_row < K && col < N) ? B[b_row * N + col] : " << type << "(0);\n";
    ss << "        barrier();\n";
    ss << "        for (uint k = 0; k < " << tile_size << "; k++) sum += As[local_row][k] * Bs[k][local_col];\n";
    ss << "        barrier();\n";
    ss << "    }\n";
    ss << "    if (row < M && col < N) C[row * N + col] = sum;\n";
    ss << main_end();
    return ss.str();
}

std::string transpose_shader(DataType dtype) {
    std::ostringstream ss;
    ss << glsl_header();
    ss << buffer_binding(0, "in_data", dtype, true);
    ss << buffer_binding(1, "out_data", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint rows; uint cols; };\n\n";
    ss << main_begin();
    ss << global_index();
    ss << "    if (idx >= rows * cols) return;\n";
    ss << "    uint row = idx / cols;\n";
    ss << "    uint col = idx % cols;\n";
    ss << "    out_data[col * rows + row] = in_data[idx];\n";
    ss << main_end();
    return ss.str();
}

}
}

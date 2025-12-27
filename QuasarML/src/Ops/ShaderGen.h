#pragma once

#include <Common/Types.h>
#include <string>

namespace QuasarML {
namespace ShaderGen {

std::string glsl_header(u32 local_size_x = 256);
std::string buffer_binding(u32 binding, const char* name, DataType dtype, bool readonly = false);
std::string main_begin();
std::string main_end();
std::string global_index();
std::string bounds_check(const char* size_var);

std::string elementwise_binary(const char* op, DataType dtype);
std::string elementwise_unary(const char* op, DataType dtype);
std::string fill_shader(DataType dtype);
std::string copy_shader(DataType dtype);

std::string reduce_sum_shader(DataType dtype, u32 workgroup_size = 256);
std::string reduce_max_shader(DataType dtype, u32 workgroup_size = 256);
std::string reduce_min_shader(DataType dtype, u32 workgroup_size = 256);

std::string matmul_shader(DataType dtype, u32 tile_size = 16);
std::string transpose_shader(DataType dtype);

}
}

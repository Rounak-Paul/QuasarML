#include "Manipulation.h"
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

std::string copy_shader(DataType dtype) {
    std::ostringstream ss;
    ss << ShaderGen::glsl_header();
    ss << ShaderGen::buffer_binding(0, "in_data", dtype, true);
    ss << ShaderGen::buffer_binding(1, "out_data", dtype, false);
    ss << "layout(push_constant) uniform PushConstants { uint n; };\n\n";
    ss << ShaderGen::main_begin();
    ss << ShaderGen::global_index();
    ss << ShaderGen::bounds_check("n");
    ss << "    out_data[idx] = in_data[idx];\n";
    ss << ShaderGen::main_end();
    return ss.str();
}

}

std::shared_ptr<Tensor> reshape(Device& device, std::shared_ptr<Tensor> a, const std::vector<u32>& new_shape) {
    u64 new_numel = 1;
    for (u32 d : new_shape) new_numel *= d;
    QS_ASSERT(new_numel == a->numel(), "Element count mismatch");
    return a->view(new_shape);
}

std::shared_ptr<Tensor> flatten(Device& device, std::shared_ptr<Tensor> a) {
    return a->flatten();
}

std::shared_ptr<Tensor> squeeze(Device& device, std::shared_ptr<Tensor> a) {
    std::vector<u32> new_shape;
    for (u32 i = 0; i < a->rank(); ++i) {
        if (a->dim(i) != 1) new_shape.push_back(a->dim(i));
    }
    if (new_shape.empty()) new_shape.push_back(1);
    return a->view(new_shape);
}

std::shared_ptr<Tensor> squeeze(Device& device, std::shared_ptr<Tensor> a, i32 dim) {
    if (dim < 0) dim = static_cast<i32>(a->rank()) + dim;
    QS_ASSERT(dim >= 0 && dim < static_cast<i32>(a->rank()), "Invalid dim");
    QS_ASSERT(a->dim(dim) == 1, "Dimension is not 1");
    
    std::vector<u32> new_shape;
    for (u32 i = 0; i < a->rank(); ++i) {
        if (static_cast<i32>(i) != dim) new_shape.push_back(a->dim(i));
    }
    if (new_shape.empty()) new_shape.push_back(1);
    return a->view(new_shape);
}

std::shared_ptr<Tensor> unsqueeze(Device& device, std::shared_ptr<Tensor> a, i32 dim) {
    if (dim < 0) dim = static_cast<i32>(a->rank()) + dim + 1;
    QS_ASSERT(dim >= 0 && dim <= static_cast<i32>(a->rank()), "Invalid dim");
    
    std::vector<u32> new_shape;
    for (u32 i = 0; i < a->rank(); ++i) {
        if (static_cast<i32>(i) == dim) new_shape.push_back(1);
        new_shape.push_back(a->dim(i));
    }
    if (static_cast<i32>(new_shape.size()) == static_cast<i32>(a->rank())) {
        new_shape.push_back(1);
    }
    return a->view(new_shape);
}

std::shared_ptr<Tensor> concat(Device& device, const std::vector<std::shared_ptr<Tensor>>& tensors, i32 axis) {
    QS_ASSERT(!tensors.empty(), "Empty tensor list");
    
    auto first = tensors[0];
    if (axis < 0) axis = static_cast<i32>(first->rank()) + axis;
    QS_ASSERT(axis >= 0 && axis < static_cast<i32>(first->rank()), "Invalid axis");
    
    u32 total_axis = 0;
    for (const auto& t : tensors) {
        QS_ASSERT(t->rank() == first->rank(), "Rank mismatch");
        QS_ASSERT(t->dtype() == first->dtype(), "Type mismatch");
        total_axis += t->dim(axis);
    }
    
    std::vector<u32> out_shape = first->shape();
    out_shape[axis] = total_axis;
    
    auto result = device.create_tensor(out_shape, first->dtype());
    
    u64 offset = 0;
    u64 elem_size = dtype_size(first->dtype());
    for (const auto& t : tensors) {
        std::vector<u8> buf(t->size_bytes());
        t->download(buf.data());
        result->upload(buf.data(), t->size_bytes(), offset);
        offset += t->size_bytes();
    }
    
    return result;
}

std::shared_ptr<Tensor> stack(Device& device, const std::vector<std::shared_ptr<Tensor>>& tensors, i32 axis) {
    QS_ASSERT(!tensors.empty(), "Empty tensor list");
    
    auto first = tensors[0];
    for (const auto& t : tensors) {
        QS_ASSERT(t->shape() == first->shape(), "Shape mismatch");
        QS_ASSERT(t->dtype() == first->dtype(), "Type mismatch");
    }
    
    std::vector<u32> new_shape;
    if (axis < 0) axis = static_cast<i32>(first->rank()) + axis + 1;
    
    for (u32 i = 0; i < first->rank(); ++i) {
        if (static_cast<i32>(i) == axis) new_shape.push_back(static_cast<u32>(tensors.size()));
        new_shape.push_back(first->dim(i));
    }
    if (axis == static_cast<i32>(first->rank())) {
        new_shape.push_back(static_cast<u32>(tensors.size()));
    }
    
    auto result = device.create_tensor(new_shape, first->dtype());
    
    u64 offset = 0;
    for (const auto& t : tensors) {
        std::vector<u8> buf(t->size_bytes());
        t->download(buf.data());
        result->upload(buf.data(), t->size_bytes(), offset);
        offset += t->size_bytes();
    }
    
    return result;
}

std::shared_ptr<Tensor> slice(Device& device, std::shared_ptr<Tensor> a, i32 axis, u64 start, u64 end) {
    if (axis < 0) axis = static_cast<i32>(a->rank()) + axis;
    QS_ASSERT(axis >= 0 && axis < static_cast<i32>(a->rank()), "Invalid axis");
    QS_ASSERT(start < end && end <= a->dim(axis), "Invalid slice range");
    
    std::vector<u32> out_shape = a->shape();
    out_shape[axis] = static_cast<u32>(end - start);
    
    auto result = device.create_tensor(out_shape, a->dtype());
    
    std::vector<u8> buf(result->size_bytes());
    a->download(buf.data(), result->size_bytes(), start * dtype_size(a->dtype()));
    result->upload(buf.data());
    
    return result;
}

std::shared_ptr<Tensor> broadcast_to(Device& device, std::shared_ptr<Tensor> a, const std::vector<u32>& shape) {
    QS_ASSERT(a->is_broadcastable(shape), "Cannot broadcast to shape");
    
    auto result = device.create_tensor(shape, a->dtype());
    
    return result;
}

std::shared_ptr<Tensor> copy(Device& device, std::shared_ptr<Tensor> a) {
    auto result = device.create_tensor(a->shape(), a->dtype());
    
    std::string key = "copy_" + std::string(dtype_to_string(a->dtype()));
    auto kernel = get_or_create_kernel(device, key, copy_shader(a->dtype()), 2, sizeof(u32));
    
    u32 n = static_cast<u32>(a->numel());
    kernel->bind(0, a);
    kernel->bind(1, result);
    u32 groups = kernel->optimal_dispatch_1d(n);
    kernel->execute(groups, 1, 1, &n);
    
    return result;
}

}
}

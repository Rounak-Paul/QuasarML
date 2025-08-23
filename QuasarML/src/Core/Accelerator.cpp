#include "Accelerator.h"
#include <cassert>
using namespace QuasarML;

Accelerator::Accelerator(const std::string& name, uint32_t gpu_idx)
    : backend_(std::make_unique<VulkanBackend>(name, gpu_idx)) {}

Accelerator::~Accelerator() = default;

Tensor Accelerator::create_tensor(const std::vector<uint32_t>& shape, DataType dtype, bool host_visible) {
    return Tensor(*backend_, shape, dtype, host_visible);
}

Kernel Accelerator::create_kernel(const std::string& glsl_source, uint32_t num_storage_buffers, uint32_t push_constant_size) {
    return Kernel(*backend_, glsl_source, num_storage_buffers, push_constant_size);
}

void Accelerator::sync() {
    backend_->wait_for_compute();
}
VulkanBackend& Accelerator::backend() { return *backend_; }

// Auto-generate correct GLSL code for add dtype/shape (cache kernels for perf)
std::string Accelerator::make_add_shader(const DataType dtype, const std::vector<uint32_t>& shape) const {
    auto glsl_type = DataTypeGLSL(dtype);
    return "#version 450\n"
        "layout(local_size_x=256) in;\n"
        "layout(set=0, binding=0) buffer A { " + std::string(glsl_type) + " a[]; };\n"
        "layout(set=0, binding=1) buffer B { " + std::string(glsl_type) + " b[]; };\n"
        "layout(set=0, binding=2) buffer C { " + std::string(glsl_type) + " c[]; };\n"
        "void main() {\n"
        "    uint idx = gl_GlobalInvocationID.x;\n"
        "    c[idx] = a[idx] + b[idx];\n"
        "}\n";
}
std::string Accelerator::make_mul_shader(const DataType dtype, const std::vector<uint32_t>& shape) const {
    auto glsl_type = DataTypeGLSL(dtype);
    return "#version 450\n"
        "layout(local_size_x=256) in;\n"
        "layout(set=0, binding=0) buffer A { " + std::string(glsl_type) + " a[]; };\n"
        "layout(set=0, binding=1) buffer B { " + std::string(glsl_type) + " b[]; };\n"
        "layout(set=0, binding=2) buffer C { " + std::string(glsl_type) + " c[]; };\n"
        "void main() {\n"
        "    uint idx = gl_GlobalInvocationID.x;\n"
        "    c[idx] = a[idx] * b[idx];\n"
        "}\n";
}

Tensor Accelerator::add(const Tensor& A, const Tensor& B) {
    assert(A.shape() == B.shape() && "Shape mismatch in add");
    assert(A.dtype() == B.dtype() && "Dtype mismatch in add");
    Tensor C = create_tensor(A.shape(), A.dtype());

    std::string shader = make_add_shader(A.dtype(), A.shape());
    Kernel kernel = create_kernel(shader, 3);
    kernel.bind(0, const_cast<Tensor&>(A));
    kernel.bind(1, const_cast<Tensor&>(B));
    kernel.bind(2, C);

    uint32_t groups = backend_->calculate_dispatch_1d((uint32_t)A.num_elements(), 256);
    kernel.run(groups);
    return C;
}

Tensor Accelerator::multiply(const Tensor& A, const Tensor& B) {
    assert(A.shape() == B.shape() && "Shape mismatch in multiply");
    assert(A.dtype() == B.dtype() && "Dtype mismatch in multiply");
    Tensor C = create_tensor(A.shape(), A.dtype());

    std::string shader = make_mul_shader(A.dtype(), A.shape());
    Kernel kernel = create_kernel(shader, 3);
    kernel.bind(0, const_cast<Tensor&>(A));
    kernel.bind(1, const_cast<Tensor&>(B));
    kernel.bind(2, C);

    uint32_t groups = backend_->calculate_dispatch_1d((uint32_t)A.num_elements(), 256);
    kernel.run(groups);
    return C;
}

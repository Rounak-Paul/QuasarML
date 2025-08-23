// test_compute_execute.cpp
#include <QuasarML.h>

static const char* double_shader_glsl = R"(
#version 450
layout(local_size_x = 64) in;
layout(binding = 0) readonly buffer In { float data[]; } inBuf;
layout(binding = 1) writeonly buffer Out { float data[]; } outBuf;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    outBuf.data[idx] = inBuf.data[idx] * 2.0;
}
)";

using namespace QuasarML;

int main() {
    {
        const size_t N = 1024;
        std::vector<float> in(N), out(N);
        for (size_t i = 0; i < N; ++i) in[i] = float(i)*1.20000304900203923f;

        Accelerator accel("TestApp");
        auto inT  = accel.create_tensor({(u32)N}, DataType::F32);
        auto outT = accel.create_tensor({(u32)N}, DataType::F32);
        inT->upload_data(in.data());

        auto kernel = accel.create_kernel("double", double_shader_glsl, 2);

        // One-shot execute: binds, dispatches, waits
        accel.execute(kernel, {inT, outT}, 
                        accel.calculate_optimal_dispatch_1d((u32)N, 64));

        outT->download_data(out.data());

        for (size_t i = 0; i < N; ++i) {
            std::cout << in[i] << " " << out[i] << std::endl;
            if (fabs(out[i] - in[i]*2.0f) > 1e-6f) {
                std::cerr << "Mismatch at " << i << "\n";
                return 1;
            }
        }
        std::cout << "PASSED\n";
    }
    {
        // Create accelerator and tensors
        Accelerator accel;
        auto a = accel.create_tensor({1024, 512}, DataType::F32);
        auto b = accel.create_tensor({1024, 512}, DataType::F32);
        auto result = accel.create_tensor({1024, 512}, DataType::F32);

        // Element-wise operations
        accel.tensor_add(a, b, result);
        accel.tensor_mul_scalar(result, 2.0f, result);
        accel.tensor_relu(result, result);

        // Matrix operations
        auto mat_a = accel.create_tensor({256, 128}, DataType::F32);
        auto mat_b = accel.create_tensor({128, 64}, DataType::F32);
        auto mat_result = accel.create_tensor({256, 64}, DataType::F32);
        accel.tensor_matmul(mat_a, mat_b, mat_result);
    }
    return 0;
}

#include "Engine.h"
#include <iostream>
#include <iomanip>

namespace QuasarML {

Engine::Engine(const std::string& application_name)
{
    _backend = new VulkanBackend{application_name, 0};
}

Engine::~Engine()
{
    LOG_DEBUG("Destroying QuasarML Engine...");
    if (_backend) {
        delete _backend;
        _backend = nullptr;
        LOG_DEBUG("VulkanBackend destroyed");
    }
}

void Engine::run_benchmark(size_t iterations) {
    if (!_backend) {
        std::cerr << "Engine not initialized!" << std::endl;
        return;
    }

    const size_t data_size   = 1024 * 1024 * 64; 
    const size_t buffer_size = data_size * sizeof(float);

    auto input  = _backend->create_storage_buffer(buffer_size);
    auto output = _backend->create_storage_buffer(buffer_size);

    std::vector<float> test_data(data_size, 3.14159f);
    _backend->upload_to_buffer(input, test_data.data(), buffer_size);

    // ================================
    // Stage 1: Memory copy throughput
    // ================================
    std::string shader_memcopy = R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer InputBuffer  { float input_data[]; };
        layout(binding = 1) buffer OutputBuffer { float output_data[]; };
        void main() {
            uint index = gl_GlobalInvocationID.x;
            if (index >= input_data.length()) return;
            output_data[index] = input_data[index];
        }
    )";

    // ================================
    // Stage 2: Arithmetic (ALU-stress, FMA-heavy, ILP + vec4)
    // ================================
    std::string shader_arithmetic = R"(
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer InputBuffer  { float input_data[]; };
        layout(binding = 1) buffer OutputBuffer { float output_data[]; };

        // We process 4 scalars per thread using vec4 math.
        void main() {
            uint tid  = gl_GlobalInvocationID.x;
            uint base = tid * 4u;

            // Ensure we have 4 contiguous elements
            if (base + 3u >= input_data.length()) return;

            // Load 4 elements as a vec4 (manual pack)
            vec4 v0 = vec4(
                input_data[base + 0u],
                input_data[base + 1u],
                input_data[base + 2u],
                input_data[base + 3u]
            );

            // Second accumulator to increase ILP
            vec4 v1 = v0 + vec4(1e-6);

            // Constant vectors (avoid loop-invariant recompute)
            const vec4 A0 = vec4(1.00010, 1.00020, 1.00030, 1.00040);
            const vec4 B0 = vec4(0.00010, 0.00020, 0.00030, 0.00040);
            const vec4 A1 = vec4(0.99995, 1.00005, 1.00015, 1.00025);
            const vec4 B1 = vec4(0.00015, 0.00025, 0.00035, 0.00045);

            // Do lots of FMAs per iteration to raise arithmetic intensity.
            // Per iteration below: 8 FMAs total (4 on v0, 4 on v1)
            // 1 FMA = 2 FLOPs; vec4 has 4 lanes → 8 FLOPs/FMA across lanes.
            // So 8 FMAs * 8 FLOPs = 64 FLOPs per iteration per vec4 (i.e., per 4 elements).
            // Per element = 64 / 4 = 16 FLOPs per iteration per element.
            // With 128 iterations → 128 * 16 = **2048 FLOPs per element**.
            for (int i = 0; i < 128; ++i) {
                // Unrolled pattern to improve ILP and reduce dependency chains
                v0 = fma(v0, A0, B0);
                v1 = fma(v1, A1, B1);
                v0 = fma(v0, A1, B1);
                v1 = fma(v1, A0, B0);

                v0 = fma(v0, A0, B1);
                v1 = fma(v1, A1, B0);
                v0 = fma(v0, A1, B0);
                v1 = fma(v1, A0, B1);
            }

            // Combine and store back
            vec4 outv = 0.5 * (v0 + v1);
            output_data[base + 0u] = outv.x;
            output_data[base + 1u] = outv.y;
            output_data[base + 2u] = outv.z;
            output_data[base + 3u] = outv.w;
        }
    )";

    // ================================
    // Stage 3: Heavy math (sin/cos/sqrt)
    // ================================
    std::string shader_special = R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer InputBuffer  { float input_data[]; };
        layout(binding = 1) buffer OutputBuffer { float output_data[]; };
        void main() {
            uint index = gl_GlobalInvocationID.x;
            if (index >= input_data.length()) return;
            float v = input_data[index];
            for (int i = 0; i < 10; ++i) {
                v = sin(v) * cos(v) + sqrt(abs(v));
            }
            output_data[index] = v;
        }
    )";

    auto run_stage = [&](const std::string& shader, const char* label,
                         double flops_per_element) {
        auto pipeline = _backend->create_compute_pipeline(shader, 2);
        _backend->bind_buffer_to_pipeline(pipeline, 0, input);
        _backend->bind_buffer_to_pipeline(pipeline, 1, output);

        auto dispatch_info = _backend->calculate_dispatch_1d(data_size, 256);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            _backend->execute_compute(pipeline, dispatch_info);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double total_sec = duration.count() / 1e6;

        std::cout << "\n=== " << label << " ===" << std::endl;
        std::cout << "Total time: " << total_sec << " s\n";

        if (flops_per_element == 0.0) {
            // Memory throughput test
            double bytes_moved = double(buffer_size) * iterations * 2.0; 
            // input read + output write
            double gb_per_sec = (bytes_moved / total_sec) / 1e9;
            std::cout << "Effective bandwidth: " << gb_per_sec << " GB/s\n";
        } else {
            // FLOP throughput test
            double total_flops = data_size * iterations * flops_per_element;
            double gflops = (total_flops / total_sec) / 1e9;
            std::cout << "Throughput: " << gflops << " GFLOPs\n";
        }

        _backend->destroy_compute_pipeline(pipeline);
    };

    // Stage 1: Memcopy (0 FLOPs, just bytes moved)
    run_stage(shader_memcopy, "Stage 1: Memory copy", 0.0);

    // Stage 2: Arithmetic 
    run_stage(shader_arithmetic, "Stage 2: Arithmetic (FMA-like)", 2048.0);

    // Stage 3: Special functions 
    // (sin, cos, sqrt)
    run_stage(shader_special, "Stage 3: Special functions", 50.0);

    _backend->destroy_buffer(input);
    _backend->destroy_buffer(output);
}

} // namespace QuasarML
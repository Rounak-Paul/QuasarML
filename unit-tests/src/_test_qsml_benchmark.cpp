// simple_benchmark.cpp
// Clean, simple benchmark for QuasarML that actually compiles and works
//
// Usage: compile and run with your QuasarML API

#include <QuasarML.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace QuasarML;
using Clock = std::chrono::high_resolution_clock;

class SimpleBenchmark {
private:
    Accelerator& accel;
    std::mt19937 rng;

    void fillRandom(std::vector<float>& data) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : data) {
            x = dist(rng);
        }
    }

    double median(std::vector<double>& times) {
        std::sort(times.begin(), times.end());
        size_t n = times.size();
        if (n % 2 == 0) {
            return (times[n/2 - 1] + times[n/2]) / 2.0;
        } else {
            return times[n/2];
        }
    }

    void printHeader(const std::string& title) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << title << "\n";
        std::cout << std::string(60, '=') << "\n";
    }

public:
    SimpleBenchmark(Accelerator& accelerator) : accel(accelerator), rng(12345) {}

    void benchmarkMemory() {
        printHeader("Memory Transfer Benchmark");
        
        const std::vector<uint32_t> sizes = {
            1024 * 1024,      // 1M elements (4 MB)
            4 * 1024 * 1024,  // 4M elements (16 MB) 
            16 * 1024 * 1024  // 16M elements (64 MB)
        };

        for (uint32_t size : sizes) {
            std::cout << "\nTesting " << size << " elements (" 
                      << (size * sizeof(float)) / (1024*1024) << " MB):\n";

            std::vector<float> hostData(size);
            std::vector<float> downloadData(size);
            fillRandom(hostData);

            auto tensor = accel.create_tensor({size}, DataType::F32);
            if (!tensor) {
                std::cout << "  ERROR: Failed to create tensor\n";
                continue;
            }

            // Warmup
            for (int i = 0; i < 10; ++i) {
                tensor->upload_data(hostData.data());
                tensor->download_data(downloadData.data());
            }

            // Test upload speed
            std::vector<double> uploadTimes;
            for (int i = 0; i < 50; ++i) {
                auto start = Clock::now();
                tensor->upload_data(hostData.data());
                auto end = Clock::now();
                double seconds = std::chrono::duration<double>(end - start).count();
                uploadTimes.push_back(seconds);
            }

            // Test download speed
            std::vector<double> downloadTimes;
            for (int i = 0; i < 50; ++i) {
                auto start = Clock::now();
                tensor->download_data(downloadData.data());
                auto end = Clock::now();
                double seconds = std::chrono::duration<double>(end - start).count();
                downloadTimes.push_back(seconds);
            }

            double uploadMedian = median(uploadTimes);
            double downloadMedian = median(downloadTimes);
            
            double uploadGB = (size * sizeof(float)) / (1024.0*1024.0*1024.0);
            double downloadGB = (size * sizeof(float)) / (1024.0*1024.0*1024.0);
            
            std::cout << "  Upload:   " << std::fixed << std::setprecision(2) 
                      << (uploadGB / uploadMedian) << " GB/s\n";
            std::cout << "  Download: " << std::fixed << std::setprecision(2) 
                      << (downloadGB / downloadMedian) << " GB/s\n";
        }
    }

    void benchmarkElementwise() {
        printHeader("Element-wise Operations Benchmark");

        const std::vector<uint32_t> sizes = {
            1024 * 1024,     // 1M elements
            4 * 1024 * 1024, // 4M elements
            16 * 1024 * 1024 // 16M elements
        };

        for (uint32_t size : sizes) {
            std::cout << "\nTesting " << size << " elements:\n";

            std::vector<float> dataA(size), dataB(size);
            fillRandom(dataA);
            fillRandom(dataB);

            auto tensorA = accel.create_tensor(dataA.data(), {size}, DataType::F32);
            auto tensorB = accel.create_tensor(dataB.data(), {size}, DataType::F32);
            
            if (!tensorA || !tensorB) {
                std::cout << "  ERROR: Failed to create tensors\n";
                continue;
            }

            // Test ADD operation
            {
                // Warmup
                for (int i = 0; i < 10; ++i) {
                    auto result = accel.ops().add(tensorA, tensorB);
                    float sample;
                    result->download_data(&sample, sizeof(float), 0);
                }

                std::vector<double> times;
                for (int i = 0; i < 100; ++i) {
                    auto start = Clock::now();
                    auto result = accel.ops().add(tensorA, tensorB);
                    float sample;
                    result->download_data(&sample, sizeof(float), 0);
                    auto end = Clock::now();
                    times.push_back(std::chrono::duration<double>(end - start).count());
                }

                double medianTime = median(times);
                double gops = double(size) / (medianTime * 1e9);
                std::cout << "  Add: " << std::fixed << std::setprecision(2) << gops << " GOPS\n";
            }

            // Test MUL operation
            {
                // Warmup
                for (int i = 0; i < 10; ++i) {
                    auto result = accel.ops().mul(tensorA, tensorB);
                    float sample;
                    result->download_data(&sample, sizeof(float), 0);
                }

                std::vector<double> times;
                for (int i = 0; i < 100; ++i) {
                    auto start = Clock::now();
                    auto result = accel.ops().mul(tensorA, tensorB);
                    float sample;
                    result->download_data(&sample, sizeof(float), 0);
                    auto end = Clock::now();
                    times.push_back(std::chrono::duration<double>(end - start).count());
                }

                double medianTime = median(times);
                double gops = double(size) / (medianTime * 1e9);
                std::cout << "  Mul: " << std::fixed << std::setprecision(2) << gops << " GOPS\n";
            }

            // Test scalar operations
            {
                // Warmup
                for (int i = 0; i < 10; ++i) {
                    auto result = accel.ops().add_scalar(tensorA, 2.5f);
                    float sample;
                    result->download_data(&sample, sizeof(float), 0);
                }

                std::vector<double> times;
                for (int i = 0; i < 100; ++i) {
                    auto start = Clock::now();
                    auto result = accel.ops().add_scalar(tensorA, 2.5f);
                    float sample;
                    result->download_data(&sample, sizeof(float), 0);
                    auto end = Clock::now();
                    times.push_back(std::chrono::duration<double>(end - start).count());
                }

                double medianTime = median(times);
                double gops = double(size) / (medianTime * 1e9);
                std::cout << "  Add Scalar: " << std::fixed << std::setprecision(2) << gops << " GOPS\n";
            }
        }
    }

    void benchmarkMatMul() {
        printHeader("Matrix Multiplication Benchmark");

        const std::vector<std::array<uint32_t, 3>> sizes = {
            {{128, 128, 128}},
            {{256, 256, 256}},
            {{512, 512, 512}},
            {{1024, 1024, 1024}}
        };

        for (const auto& [M, K, N] : sizes) {
            std::cout << "\nTesting " << M << "x" << K << " * " << K << "x" << N << ":\n";

            std::vector<float> matA(M * K, 1.0f);
            std::vector<float> matB(K * N, 1.0f);

            auto tensorA = accel.create_tensor(matA.data(), {M, K}, DataType::F32);
            auto tensorB = accel.create_tensor(matB.data(), {K, N}, DataType::F32);

            if (!tensorA || !tensorB) {
                std::cout << "  ERROR: Failed to create matrices\n";
                continue;
            }

            // Warmup
            for (int i = 0; i < 5; ++i) {
                auto result = accel.ops().matmul(tensorA, tensorB);
                float sample;
                result->download_data(&sample, sizeof(float), 0);
            }

            std::vector<double> times;
            for (int i = 0; i < 50; ++i) {
                auto start = Clock::now();
                auto result = accel.ops().matmul(tensorA, tensorB);
                float sample;
                result->download_data(&sample, sizeof(float), 0);
                auto end = Clock::now();
                times.push_back(std::chrono::duration<double>(end - start).count());
            }

            double medianTime = median(times);
            double flops = 2.0 * double(M) * double(K) * double(N);
            double gflops = flops / (medianTime * 1e9);

            std::cout << "  MatMul: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n";
            std::cout << "  Time: " << std::fixed << std::setprecision(3) << medianTime * 1000.0 << " ms\n";
        }
    }

    void benchmarkCustomKernel() {
        printHeader("Custom Kernel Benchmark");

        const char* simpleShader = R"(
#version 450
layout(local_size_x = 256) in;
layout(binding = 0) readonly buffer Input { float data[]; } input_buf;
layout(binding = 1) writeonly buffer Output { float data[]; } output_buf;
layout(push_constant) uniform PushConstants { float multiplier; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= input_buf.data.length()) return;
    output_buf.data[idx] = input_buf.data[idx] * multiplier;
}
)";

        const uint32_t size = 4 * 1024 * 1024; // 4M elements
        std::vector<float> inputData(size);
        fillRandom(inputData);

        auto inputTensor = accel.create_tensor(inputData.data(), {size}, DataType::F32);
        auto outputTensor = accel.create_tensor({size}, DataType::F32);

        if (!inputTensor || !outputTensor) {
            std::cout << "ERROR: Failed to create tensors for custom kernel\n";
            return;
        }

        auto kernel = accel.create_kernel("simple_multiply", simpleShader, 2, sizeof(float));
        if (!kernel) {
            std::cout << "ERROR: Failed to create custom kernel\n";
            return;
        }

        float multiplier = 2.5f;
        auto dispatchSize = accel.calculate_optimal_dispatch_1d(size, 256);

        // Warmup
        for (int i = 0; i < 10; ++i) {
            accel.execute(kernel, {inputTensor, outputTensor}, dispatchSize, 1, 1, &multiplier);
            float sample;
            outputTensor->download_data(&sample, sizeof(float), 0);
        }

        std::vector<double> times;
        for (int i = 0; i < 100; ++i) {
            auto start = Clock::now();
            accel.execute(kernel, {inputTensor, outputTensor}, dispatchSize, 1, 1, &multiplier);
            float sample;
            outputTensor->download_data(&sample, sizeof(float), 0);
            auto end = Clock::now();
            times.push_back(std::chrono::duration<double>(end - start).count());
        }

        double medianTime = median(times);
        double gops = double(size) / (medianTime * 1e9);

        std::cout << "\nCustom kernel (" << size << " elements):\n";
        std::cout << "  Performance: " << std::fixed << std::setprecision(2) << gops << " GOPS\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << medianTime * 1000.0 << " ms\n";

        // Verify correctness
        std::vector<float> result(size);
        outputTensor->download_data(result.data());
        
        bool correct = true;
        for (int i = 0; i < std::min(100, (int)size); ++i) {
            float expected = inputData[i] * multiplier;
            if (std::abs(result[i] - expected) > 1e-5f) {
                correct = false;
                break;
            }
        }
        std::cout << "  Correctness: " << (correct ? "PASS" : "FAIL") << "\n";
    }

    void runAll() {
        auto startTime = Clock::now();
        
        std::cout << "QuasarML Simple Benchmark Suite\n";
        std::cout << "===============================\n";

        auto [usedMem, totalMem] = accel.get_memory_usage();
        std::cout << "Device Memory: " << usedMem << " / " << totalMem << " bytes\n";

        benchmarkMemory();
        benchmarkElementwise();
        benchmarkMatMul();
        benchmarkCustomKernel();

        auto endTime = Clock::now();
        double totalSeconds = std::chrono::duration<double>(endTime - startTime).count();

        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "Benchmark completed in " << std::fixed << std::setprecision(1) 
                  << totalSeconds << " seconds\n";
        std::cout << std::string(60, '=') << "\n";
    }
};

int main() {
    try {
        std::cout << "Initializing QuasarML...\n";
        Accelerator accel("QuasarML_SimpleBenchmark");

        if (!accel.is_valid()) {
            std::cerr << "ERROR: Failed to initialize accelerator\n";
            return 1;
        }

        std::cout << "Accelerator initialized successfully\n";

        SimpleBenchmark benchmark(accel);
        benchmark.runAll();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
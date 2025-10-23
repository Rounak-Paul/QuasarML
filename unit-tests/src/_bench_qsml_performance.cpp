/**
 * QuasarML Performance Benchmark Suite
 * Measures throughput in TFLOPS for various tensor operations
 */

#include "QuasarML.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std::chrono;

struct BenchmarkResult {
    std::string operation;
    size_t problem_size;
    int64_t total_ops;
    double time_ms;
    double gflops;
    double tflops;
};

class PerformanceBenchmark {
private:
    std::vector<BenchmarkResult> results;
    const int warmup_iters = 3;
    const int bench_iters = 10;

    template<typename Func>
    double benchmark_op(Func&& op) {
        // Warmup
        for (int i = 0; i < warmup_iters; i++) {
            op();
        }
        qsml::accelerator().synchronize();

        // Actual benchmark
        auto start = high_resolution_clock::now();
        for (int i = 0; i < bench_iters; i++) {
            op();
        }
        qsml::accelerator().synchronize();
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(end - start).count();
        return duration / 1000.0; // Convert to milliseconds
    }

    void add_result(const std::string& op_name, size_t problem_size, 
                   int64_t total_ops, double time_ms) {
        double gflops = (total_ops * bench_iters) / (time_ms * 1e6);
        double tflops = gflops / 1000.0;
        results.push_back({op_name, problem_size, total_ops, time_ms, gflops, tflops});
    }

public:
    void bench_elementwise_ops() {
        std::cout << "\n=== Elementwise Operations Benchmark ===\n";
        
        std::vector<size_t> sizes = {
            1024 * 1024,        // 1M elements
            4 * 1024 * 1024,    // 4M elements
            16 * 1024 * 1024,   // 16M elements
            64 * 1024 * 1024    // 64M elements
        };

        for (auto size : sizes) {
            auto a = qsml::randn({static_cast<u32>(size)}, DataType::F32);
            auto b = qsml::randn({static_cast<u32>(size)}, DataType::F32);

            // Addition (2 ops per element: load, add)
            auto time_add = benchmark_op([&]() { auto c = qsml::add(a, b); });
            add_result("Add", size, size, time_add);

            // Multiplication (2 ops per element)
            auto time_mul = benchmark_op([&]() { auto c = qsml::mul(a, b); });
            add_result("Mul", size, size, time_mul);

            // ReLU (comparison + conditional)
            auto time_relu = benchmark_op([&]() { auto c = qsml::relu(a); });
            add_result("ReLU", size, size, time_relu);

            // Sigmoid (exp + division, ~10 ops per element)
            auto time_sigmoid = benchmark_op([&]() { auto c = qsml::sigmoid(a); });
            add_result("Sigmoid", size, size * 10, time_sigmoid);

            // Tanh (~12 ops per element)
            auto time_tanh = benchmark_op([&]() { auto c = qsml::tanh(a); });
            add_result("Tanh", size, size * 12, time_tanh);
        }
    }

    void bench_matmul() {
        std::cout << "\n=== Matrix Multiplication Benchmark ===\n";
        
        std::vector<size_t> sizes = {128, 256, 512, 1024, 2048, 4096};

        for (auto N : sizes) {
            auto a = qsml::randn({static_cast<u32>(N), static_cast<u32>(N)}, DataType::F32);
            auto b = qsml::randn({static_cast<u32>(N), static_cast<u32>(N)}, DataType::F32);

            // Matrix multiplication: 2*N^3 FLOPs (N^3 muls + N^3 adds)
            int64_t flops = 2LL * N * N * N;
            
            auto time_matmul = benchmark_op([&]() { auto c = qsml::matmul(a, b); });
            add_result("MatMul " + std::to_string(N) + "x" + std::to_string(N), 
                      N * N, flops, time_matmul);
        }
    }

    void bench_batched_matmul() {
        std::cout << "\n=== Batched Matrix Multiplication Benchmark ===\n";
        
        std::vector<std::tuple<size_t, size_t>> configs = {
            {32, 512},   // batch=32, size=512x512
            {64, 256},   // batch=64, size=256x256
            {128, 128},  // batch=128, size=128x128
            {256, 64},   // batch=256, size=64x64
        };

        for (auto [batch, N] : configs) {
            // Create batch of matrices as separate tensors
            std::vector<qsml::Tensor> a_batch, b_batch;
            for (size_t i = 0; i < batch; i++) {
                a_batch.push_back(qsml::randn({static_cast<u32>(N), static_cast<u32>(N)}, DataType::F32));
                b_batch.push_back(qsml::randn({static_cast<u32>(N), static_cast<u32>(N)}, DataType::F32));
            }

            // Batched matrix multiplication: batch * 2*N^3 FLOPs
            int64_t flops = batch * 2LL * N * N * N;
            
            auto time_matmul = benchmark_op([&]() { 
                for (size_t i = 0; i < batch; i++) {
                    auto c = qsml::matmul(a_batch[i], b_batch[i]);
                }
            });
            add_result("Batched MatMul [" + std::to_string(batch) + "," + 
                      std::to_string(N) + "x" + std::to_string(N) + "]", 
                      batch * N * N, flops, time_matmul);
        }
    }

    void bench_reductions() {
        std::cout << "\n=== Reduction Operations Benchmark ===\n";
        
        std::vector<std::tuple<size_t, size_t, size_t>> shapes = {
            {1024, 1024, 1},      // 2D: 1024x1024
            {512, 512, 4},        // 3D: 512x512x4
            {256, 256, 16},       // 3D: 256x256x16
            {128, 128, 64},       // 3D: 128x128x64
        };

        for (auto [H, W, C] : shapes) {
            qsml::Tensor x;
            size_t total_size;
            std::string shape_str;
            
            if (C == 1) {
                std::cout << "  Testing 2D [" << H << "x" << W << "]..." << std::flush;
                x = qsml::randn({static_cast<u32>(H), static_cast<u32>(W)}, DataType::F32);
                total_size = H * W;
                shape_str = std::to_string(H) + "x" + std::to_string(W);
            } else {
                std::cout << "  Testing 3D [" << C << "x" << H << "x" << W << "]..." << std::flush;
                x = qsml::randn({static_cast<u32>(C), static_cast<u32>(H), static_cast<u32>(W)}, DataType::F32);
                total_size = C * H * W;
                shape_str = std::to_string(C) + "x" + std::to_string(H) + "x" + std::to_string(W);
            }

            // Sum reduction (1 add per element)
            auto time_sum = benchmark_op([&]() { auto s = qsml::sum_axis(x, 0); });
            add_result("Sum " + shape_str, total_size, total_size, time_sum);

            // Mean reduction (1 add + 1 div per element)
            auto time_mean = benchmark_op([&]() { auto m = qsml::mean_axis(x, 0); });
            add_result("Mean " + shape_str, total_size, total_size * 2, time_mean);
            std::cout << "OK\n";
        }
    }

    void bench_softmax() {
        std::cout << "\n=== Softmax Benchmark ===\n";
        
        // Note: Current softmax implementation works best with 1D tensors
        // For 2D batch processing, we iterate over batches
        std::vector<size_t> sizes = {
            1000,    // Classification
            10000,   // Large vocabulary
            50000,   // Very large vocabulary  
            100000   // Extreme size
        };

        for (auto size : sizes) {
            std::cout << "  Testing [" << size << "]..." << std::flush;
            auto x = qsml::randn({static_cast<u32>(size)}, DataType::F32);
            
            // Softmax: max, sub, exp, sum, div = ~5 ops per element
            int64_t flops = size * 5;
            
            auto time_softmax = benchmark_op([&]() { auto s = qsml::softmax(x, 0); });
            add_result("Softmax [" + std::to_string(size) + "]", 
                      size, flops, time_softmax);
            std::cout << "OK\n";
        }
    }

    void bench_mixed_precision() {
        std::cout << "\n=== Large Matrix Multiplication ===\n";
        
        std::vector<size_t> sizes = {1024, 2048, 4096};
        
        for (auto N : sizes) {
            std::cout << "  Testing [" << N << "x" << N << "]..." << std::flush;
            auto a = qsml::randn({static_cast<u32>(N), static_cast<u32>(N)}, DataType::F32);
            auto b = qsml::randn({static_cast<u32>(N), static_cast<u32>(N)}, DataType::F32);
            int64_t flops = 2LL * N * N * N;
            
            auto time = benchmark_op([&]() { auto c = qsml::matmul(a, b); });
            add_result("MatMul " + std::to_string(N) + "x" + std::to_string(N) + " FP32", 
                      N * N, flops, time);
            std::cout << "OK\n";
        }
    }

    void bench_complex_operations() {
        std::cout << "\n=== Complex Operations Benchmark ===\n";
        
        // Concatenate
        {
            size_t N = 1024;
            auto a = qsml::randn({static_cast<u32>(N), static_cast<u32>(N)}, DataType::F32);
            auto b = qsml::randn({static_cast<u32>(N), static_cast<u32>(N)}, DataType::F32);
            
            auto time = benchmark_op([&]() { 
                auto c = qsml::concatenate({a, b}, 0); 
            });
            // Memory copy operations
            add_result("Concatenate 2x[1024,1024]", 2 * N * N, 2 * N * N, time);
        }

        // Permute (transpose-like)
        {
            size_t B = 32, H = 64, W = 64, C = 128;
            auto x = qsml::randn({static_cast<u32>(B), static_cast<u32>(H), static_cast<u32>(W), static_cast<u32>(C)}, DataType::F32);
            
            auto time = benchmark_op([&]() { 
                auto y = qsml::permute(x, {0, 3, 1, 2}); // BHWC -> BCHW
            });
            add_result("Permute [32,64,64,128]", B * H * W * C, B * H * W * C, time);
        }

        // Clamp
        {
            size_t size = 16 * 1024 * 1024;
            auto x = qsml::randn({static_cast<u32>(size)}, DataType::F32);
            
            auto time = benchmark_op([&]() { 
                auto y = qsml::clamp(x, -1.0f, 1.0f); 
            });
            // 2 comparisons per element
            add_result("Clamp 16M elements", size, size * 2, time);
        }
    }

    void bench_realistic_workloads() {
        std::cout << "\n=== Realistic ML Workloads ===\n";
        
        // Simulated fully connected layer: Y = ReLU(X @ W + b)
        {
            size_t batch = 256, in_dim = 1024, out_dim = 1024;
            auto x = qsml::randn({static_cast<u32>(batch), static_cast<u32>(in_dim)}, DataType::F32);
            auto w = qsml::randn({static_cast<u32>(in_dim), static_cast<u32>(out_dim)}, DataType::F32);
            auto b = qsml::randn({static_cast<u32>(out_dim)}, DataType::F32);
            
            int64_t flops = 2LL * batch * in_dim * out_dim + // matmul
                           batch * out_dim +                  // add bias
                           batch * out_dim;                   // relu
            
            auto time = benchmark_op([&]() { 
                auto y = qsml::matmul(x, w);
                y = qsml::add(y, b);
                y = qsml::relu(y);
            });
            add_result("FC Layer [256,1024->1024]", batch * out_dim, flops, time);
        }

        // Simulated attention mechanism (simplified - single batch)
        {
            size_t seq_len = 128, embed_dim = 512;
            auto q = qsml::randn({static_cast<u32>(seq_len), static_cast<u32>(embed_dim)}, DataType::F32);
            auto k = qsml::randn({static_cast<u32>(seq_len), static_cast<u32>(embed_dim)}, DataType::F32);
            auto v = qsml::randn({static_cast<u32>(seq_len), static_cast<u32>(embed_dim)}, DataType::F32);
            
            // Q @ K^T + softmax + @ V
            int64_t flops = 
                2LL * seq_len * seq_len * embed_dim +  // Q @ K^T
                seq_len * seq_len * 5 +                 // softmax
                2LL * seq_len * seq_len * embed_dim;    // scores @ V
            
            auto time = benchmark_op([&]() {
                // Simplified attention: Q @ K^T
                auto scores = qsml::matmul(q, qsml::permute(k, {1, 0}));
                scores = qsml::softmax(scores, -1);
                auto out = qsml::matmul(scores, v);
            });
            add_result("Attention [128,512]", 
                      seq_len * embed_dim, flops, time);
        }
    }

    void print_results() {
        std::cout << "\n\n";
        std::cout << "============================================================================================================\n";
        std::cout << "                                    PERFORMANCE BENCHMARK RESULTS                                           \n";
        std::cout << "============================================================================================================\n";
        std::cout << std::left << std::setw(40) << "Operation" 
                  << std::right << std::setw(15) << "Problem Size"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(15) << "GFLOPS"
                  << std::setw(15) << "TFLOPS" << "\n";
        std::cout << "------------------------------------------------------------------------------------------------------------\n";

        for (const auto& r : results) {
            std::cout << std::left << std::setw(40) << r.operation
                     << std::right << std::setw(15) << r.problem_size
                     << std::setw(15) << std::fixed << std::setprecision(3) << r.time_ms
                     << std::setw(15) << std::fixed << std::setprecision(2) << r.gflops
                     << std::setw(15) << std::fixed << std::setprecision(4) << r.tflops << "\n";
        }
        std::cout << "============================================================================================================\n";

        // Find peak performance
        auto max_tflops = std::max_element(results.begin(), results.end(),
            [](const auto& a, const auto& b) { return a.tflops < b.tflops; });
        
        if (max_tflops != results.end()) {
            std::cout << "\nPeak Performance: " << std::fixed << std::setprecision(4) 
                     << max_tflops->tflops << " TFLOPS"
                     << " (" << max_tflops->operation << ")\n";
        }

        // Calculate average TFLOPS
        double avg_tflops = 0.0;
        for (const auto& r : results) {
            avg_tflops += r.tflops;
        }
        avg_tflops /= results.size();
        std::cout << "Average Performance: " << std::fixed << std::setprecision(4) 
                 << avg_tflops << " TFLOPS\n";
        std::cout << "Total Benchmarks: " << results.size() << "\n";
        std::cout << "============================================================================================================\n\n";
    }

    void run_all() {
        std::cout << "\n";
        std::cout << "████████████████████████████████████████████████████████████████\n";
        std::cout << "█                                                              █\n";
        std::cout << "█            QuasarML Performance Benchmark Suite             █\n";
        std::cout << "█                  TFLOPS Throughput Analysis                 █\n";
        std::cout << "█                                                              █\n";
        std::cout << "████████████████████████████████████████████████████████████████\n";

        auto& device = qsml::accelerator();
        std::cout << "\nDevice: GPU Accelerator\n";
        std::cout << "Warmup iterations: " << warmup_iters << "\n";
        std::cout << "Benchmark iterations: " << bench_iters << "\n";

        bench_elementwise_ops();
        bench_matmul();
        bench_batched_matmul();
        bench_reductions();
        bench_softmax();
        bench_mixed_precision();
        bench_complex_operations();
        bench_realistic_workloads();

        print_results();
    }
};

int main() {
    try {
        // Device is automatically initialized as GPU by default
        // No need to explicitly set accelerator
        
        PerformanceBenchmark bench;
        bench.run_all();

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

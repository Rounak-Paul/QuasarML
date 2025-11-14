#include "QuasarML.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std::chrono;

class Benchmark {
private:
    const int warmup = 2;
    const int iters = 5;

    template<typename Func>
    double time_op(Func&& op) {
        for (int i = 0; i < warmup; i++) op();
        qsml::accelerator().synchronize();

        auto start = high_resolution_clock::now();
        for (int i = 0; i < iters; i++) op();
        qsml::accelerator().synchronize();
        auto end = high_resolution_clock::now();

        return duration_cast<microseconds>(end - start).count() / 1000.0;
    }

    void print_result(const std::string& name, size_t ops, double time_ms) {
        double gflops = (ops * iters) / (time_ms * 1e6);
        double tflops = gflops / 1000.0;
        std::cout << "  " << std::setw(35) << std::left << name;
        std::cout << std::setw(10) << std::right << std::fixed << std::setprecision(2) << time_ms << " ms";
        std::cout << std::setw(12) << std::fixed << std::setprecision(4) << tflops << " TFLOPS\n";
    }

public:
    void run() {
        std::cout << "\n=== QuasarML Performance Benchmark ===\n";
        std::cout << "Using VMA's built-in memory pooling\n\n";

        matmul();
        elementwise();
        allocation_speed();
    }

    void matmul() {
        std::cout << "Matrix Multiplication:\n";
        
        for (auto N : {256, 512, 1024, 2048}) {
            auto a = qsml::randn({(u32)N, (u32)N}, DataType::F32);
            auto b = qsml::randn({(u32)N, (u32)N}, DataType::F32);
            
            auto t = time_op([&]() { auto c = qsml::matmul(a, b); });
            print_result("MatMul " + std::to_string(N) + "x" + std::to_string(N), 
                        2LL * N * N * N, t);
        }
        std::cout << "\n";
    }

    void elementwise() {
        std::cout << "Elementwise Operations (16M elements):\n";
        
        const size_t N = 16 * 1024 * 1024;
        auto a = qsml::randn({(u32)N}, DataType::F32);
        auto b = qsml::randn({(u32)N}, DataType::F32);

        print_result("Add", N, time_op([&]() { auto c = qsml::add(a, b); }));
        print_result("Mul", N, time_op([&]() { auto c = qsml::mul(a, b); }));
        print_result("ReLU", N, time_op([&]() { auto c = qsml::relu(a); }));
        print_result("Sigmoid", N * 10, time_op([&]() { auto c = qsml::sigmoid(a); }));

        std::cout << "\n";
    }

    void allocation_speed() {
        std::cout << "VMA Allocation Performance:\n";
        
        auto start = high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            auto t = qsml::zeros({512, 512}, DataType::F32);
        }
        auto end = high_resolution_clock::now();
        auto ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        
        std::cout << "  100 allocations (512x512):    " << std::fixed << std::setprecision(2) << ms << " ms\n";
        std::cout << "  Average per allocation:        " << (ms / 100.0) << " ms\n";
        std::cout << "\n";
    }
};

int main() {
    Benchmark bench;
    bench.run();
    return 0;
}

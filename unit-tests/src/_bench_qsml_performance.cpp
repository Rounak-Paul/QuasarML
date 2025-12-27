#include <QuasarML.h>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace std;

void benchmark_elementwise(qsml::u32 size, int iterations) {
    cout << "\n=== Elementwise Operations (" << size << " elements) ===\n";
    
    auto a = qsml::ones({size});
    auto b = qsml::ones({size});
    
    qsml::synchronize();
    
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto c = qsml::add(a, b);
    }
    qsml::synchronize();
    auto end = chrono::high_resolution_clock::now();
    
    double ms = chrono::duration<double, milli>(end - start).count();
    double gflops = (double)size * iterations / (ms * 1e6);
    cout << "  add: " << fixed << setprecision(2) << ms / iterations << " ms/op, " << gflops << " GFLOPS\n";
    
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto c = qsml::mul(a, b);
    }
    qsml::synchronize();
    end = chrono::high_resolution_clock::now();
    
    ms = chrono::duration<double, milli>(end - start).count();
    gflops = (double)size * iterations / (ms * 1e6);
    cout << "  mul: " << fixed << setprecision(2) << ms / iterations << " ms/op, " << gflops << " GFLOPS\n";
}

void benchmark_matmul(qsml::u32 n, int iterations) {
    cout << "\n=== Matrix Multiplication (" << n << "x" << n << ") ===\n";
    
    auto a = qsml::ones({n, n});
    auto b = qsml::ones({n, n});
    
    qsml::synchronize();
    
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto c = qsml::matmul(a, b);
    }
    qsml::synchronize();
    auto end = chrono::high_resolution_clock::now();
    
    double ms = chrono::duration<double, milli>(end - start).count();
    double flops = 2.0 * n * n * n * iterations;
    double gflops = flops / (ms * 1e6);
    cout << "  matmul: " << fixed << setprecision(2) << ms / iterations << " ms/op, " << gflops << " GFLOPS\n";
}

void benchmark_activations(qsml::u32 size, int iterations) {
    cout << "\n=== Activation Functions (" << size << " elements) ===\n";
    
    auto a = qsml::ones({size});
    qsml::synchronize();
    
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto c = qsml::relu(a);
    }
    qsml::synchronize();
    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(end - start).count();
    cout << "  relu: " << fixed << setprecision(2) << ms / iterations << " ms/op\n";
    
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto c = qsml::sigmoid(a);
    }
    qsml::synchronize();
    end = chrono::high_resolution_clock::now();
    ms = chrono::duration<double, milli>(end - start).count();
    cout << "  sigmoid: " << fixed << setprecision(2) << ms / iterations << " ms/op\n";
    
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto c = qsml::gelu(a);
    }
    qsml::synchronize();
    end = chrono::high_resolution_clock::now();
    ms = chrono::duration<double, milli>(end - start).count();
    cout << "  gelu: " << fixed << setprecision(2) << ms / iterations << " ms/op\n";
    
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto c = qsml::softmax(a);
    }
    qsml::synchronize();
    end = chrono::high_resolution_clock::now();
    ms = chrono::duration<double, milli>(end - start).count();
    cout << "  softmax: " << fixed << setprecision(2) << ms / iterations << " ms/op\n";
}

void benchmark_reductions(qsml::u32 size, int iterations) {
    cout << "\n=== Reduction Operations (" << size << " elements) ===\n";
    
    auto a = qsml::ones({size});
    qsml::synchronize();
    
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto c = qsml::sum(a);
    }
    qsml::synchronize();
    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(end - start).count();
    cout << "  sum: " << fixed << setprecision(2) << ms / iterations << " ms/op\n";
    
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto c = qsml::mean(a);
    }
    qsml::synchronize();
    end = chrono::high_resolution_clock::now();
    ms = chrono::duration<double, milli>(end - start).count();
    cout << "  mean: " << fixed << setprecision(2) << ms / iterations << " ms/op\n";
    
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto c = qsml::max(a);
    }
    qsml::synchronize();
    end = chrono::high_resolution_clock::now();
    ms = chrono::duration<double, milli>(end - start).count();
    cout << "  max: " << fixed << setprecision(2) << ms / iterations << " ms/op\n";
}

static void run_all_benchmarks() {
    auto& dev = qsml::device();
    cout << "\nDevice: " << dev.name() << "\n";
    
    benchmark_elementwise(1000000, 100);
    benchmark_elementwise(10000000, 50);
    
    benchmark_matmul(256, 50);
    benchmark_matmul(512, 20);
    benchmark_matmul(1024, 10);
    benchmark_matmul(2048, 5);
    benchmark_matmul(4096, 2);
    benchmark_matmul(8192, 2);
    benchmark_matmul(16384, 1);
    
    benchmark_activations(1000000, 100);
    
    benchmark_reductions(1000000, 100);
}

int main() {
    cout << "QuasarML Performance Benchmarks\n";
    cout << "================================\n";
    
    qsml::init();
    run_all_benchmarks();
    qsml::shutdown();
    
    cout << "\n================================\n";
    cout << "Benchmark complete\n";
    
    return 0;
}

#include <QuasarML.h>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>

using namespace std::chrono;

void test_thread_safety() {
    std::cout << "\n=== Thread Safety Test ===\n";
    
    auto& acc = qsml::accelerator();
    
    const int num_threads = 4;
    const int iterations = 10;
    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;
    
    auto worker = [&](int thread_id) {
        try {
            for (int i = 0; i < iterations; i++) {
                auto a = qsml::randn({128, 128}, DataType::F32);
                auto b = qsml::randn({128, 128}, DataType::F32);
                
                auto c = qsml::add(a, b);
                auto d = qsml::mul(c, b);
                auto e = qsml::relu(d);
                
                acc.synchronize();
            }
            success_count++;
        } catch (const std::exception& e) {
            std::cerr << "Thread " << thread_id << " failed: " << e.what() << "\n";
        }
    };
    
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = high_resolution_clock::now();
    auto ms = duration_cast<milliseconds>(end - start).count();
    
    std::cout << "  Threads: " << num_threads << "\n";
    std::cout << "  Iterations per thread: " << iterations << "\n";
    std::cout << "  Successful threads: " << success_count << "/" << num_threads << "\n";
    std::cout << "  Total time: " << ms << " ms\n";
    std::cout << "  Throughput: " << (num_threads * iterations * 1000.0 / ms) << " ops/sec\n";
    
    if (success_count == num_threads) {
        std::cout << "  ✓ PASSED\n";
    } else {
        std::cout << "  ✗ FAILED\n";
    }
}

void test_multi_gpu() {
    std::cout << "\n=== Multi-GPU Support Test ===\n";
    
    u32 device_count = qsml::device_count();
    std::cout << "  Available devices: " << device_count << "\n";
    
    auto names = qsml::device_names();
    for (size_t i = 0; i < names.size(); i++) {
        std::cout << "    [" << i << "] " << names[i] << "\n";
    }
    
    if (device_count > 0) {
        std::cout << "  Testing device 0...\n";
        qsml::set_device(0);
        
        auto& acc = qsml::accelerator();
        
        auto a = qsml::randn({256, 256}, DataType::F32);
        auto b = qsml::randn({256, 256}, DataType::F32);
        auto c = qsml::add(a, b);
        acc.synchronize();
        
        std::cout << "  Current device: " << qsml::current_device() << "\n";
        std::cout << "  ✓ Multi-GPU API working\n";
    }
}

void test_concurrent_kernel_creation() {
    std::cout << "\n=== Concurrent Kernel Creation Test ===\n";
    
    auto& acc = qsml::accelerator();
    
    const int num_threads = 4;
    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;
    
    auto worker = [&](int thread_id) {
        try {
            std::string glsl = R"(
                #version 450
                layout(local_size_x = 256) in;
                layout(binding = 0) buffer InputBuffer { float data[]; } input_buf;
                layout(binding = 1) buffer OutputBuffer { float data[]; } output_buf;
                
                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    output_buf.data[idx] = input_buf.data[idx] * 2.0;
                }
            )";
            
            auto kernel = acc.create_kernel(
                "test_kernel_" + std::to_string(thread_id),
                glsl,
                2
            );
            
            success_count++;
        } catch (const std::exception& e) {
            std::cerr << "Thread " << thread_id << " failed: " << e.what() << "\n";
        }
    };
    
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "  Kernels created: " << success_count << "/" << num_threads << "\n";
    
    if (success_count == num_threads) {
        std::cout << "  ✓ PASSED\n";
    } else {
        std::cout << "  ✗ FAILED\n";
    }
}

int main() {
    std::cout << "=== QuasarML Thread-Safety & Multi-GPU Test Suite ===\n";
    
    try {
        test_multi_gpu();
        test_thread_safety();
        test_concurrent_kernel_creation();
        
        std::cout << "\n=== All tests completed ===\n";
        
        qsml::AcceleratorManager::instance().shutdown();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

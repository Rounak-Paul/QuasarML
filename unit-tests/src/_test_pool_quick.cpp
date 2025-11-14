#include <QuasarML.h>
#include <iostream>
#include <chrono>

using namespace std::chrono;
using namespace QuasarML;

int main() {
    std::cout << "=== Quick Memory Pool Test ===\n";
    
    auto& acc = qsml::accelerator();
    
    std::cout << "\n1. Small allocations (256x256)...\n";
    acc.clear_memory_pool();
    acc.reset_pool_statistics();
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < 50; i++) {
        auto t = qsml::zeros({256, 256}, DataType::F32);
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    
    auto stats = acc.get_pool_statistics();
    std::cout << "   Time: " << (duration / 1000.0) << " ms\n";
    std::cout << "   Avg: " << (duration / 50.0) << " µs/alloc\n";
    std::cout << "   Hit rate: " << (stats.hit_rate * 100.0f) << "%\n";
    std::cout << "   Cached: " << (stats.total_cached_bytes / 1024.0 / 1024.0) << " MB\n";
    
    std::cout << "\n2. Mixed size pattern...\n";
    acc.clear_memory_pool();
    acc.reset_pool_statistics();
    
    start = high_resolution_clock::now();
    for (int batch = 0; batch < 10; batch++) {
        auto input = qsml::zeros({32, 512}, DataType::F32);
        auto hidden1 = qsml::zeros({32, 256}, DataType::F32);
        auto hidden2 = qsml::zeros({32, 128}, DataType::F32);
        auto output = qsml::zeros({32, 10}, DataType::F32);
    }
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    
    stats = acc.get_pool_statistics();
    std::cout << "   Time: " << (duration / 1000.0) << " ms\n";
    std::cout << "   Hit rate: " << (stats.hit_rate * 100.0f) << "%\n";
    std::cout << "   Cached: " << (stats.total_cached_bytes / 1024.0 / 1024.0) << " MB\n";
    
    std::cout << "\n3. Operations with pool...\n";
    acc.clear_memory_pool();
    acc.reset_pool_statistics();
    
    start = high_resolution_clock::now();
    for (int i = 0; i < 5; i++) {
        auto a = qsml::randn({512, 512}, DataType::F32);
        auto b = qsml::randn({512, 512}, DataType::F32);
        auto c = qsml::add(a, b);
    }
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    
    stats = acc.get_pool_statistics();
    std::cout << "   Time: " << (duration / 1000.0) << " ms\n";
    std::cout << "   Avg: " << (duration / 5.0) << " ms/iter\n";
    std::cout << "   Hit rate: " << (stats.hit_rate * 100.0f) << "%\n";
    std::cout << "   Cached: " << (stats.total_cached_bytes / 1024.0 / 1024.0) << " MB\n";
    
    std::cout << "\n=== All tests completed successfully ===\n";
    return 0;
}

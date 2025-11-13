#include "QuasarML.h"
#include <Core/MemoryPool.h>
#include <iostream>
#include <vector>

using namespace QuasarML;

static bool test_memory_pool_basic() {
    std::cout << "[test_memory_pool_basic]\n";
    
    auto& acc = qsml::accelerator();
    
    auto stats_before = acc.get_pool_statistics();
    std::cout << "  Initial stats:\n";
    std::cout << "    Allocated: " << stats_before.total_allocated_bytes << " bytes\n";
    std::cout << "    Cached: " << stats_before.total_cached_bytes << " bytes\n";
    std::cout << "    Active allocations: " << stats_before.active_allocations << "\n";
    
    {
        auto t1 = qsml::zeros({1024, 1024}, DataType::F32);
        auto stats_active = acc.get_pool_statistics();
        std::cout << "  After allocation:\n";
        std::cout << "    Active allocations: " << stats_active.active_allocations << "\n";
        std::cout << "    Cache misses: " << stats_active.cache_misses << "\n";
        
        if (stats_active.active_allocations == 0) {
            std::cerr << "  FAIL: No active allocations recorded\n";
            return false;
        }
    }
    
    auto stats_after = acc.get_pool_statistics();
    std::cout << "  After deallocation:\n";
    std::cout << "    Active allocations: " << stats_after.active_allocations << "\n";
    std::cout << "    Cached allocations: " << stats_after.cached_allocations << "\n";
    std::cout << "    Cached bytes: " << stats_after.total_cached_bytes << "\n";
    
    if (stats_after.cached_allocations == 0) {
        std::cerr << "  FAIL: No cached allocations after deallocation\n";
        return false;
    }
    
    std::cout << "  OK\n";
    return true;
}

static bool test_memory_pool_cache_hit() {
    std::cout << "[test_memory_pool_cache_hit]\n";
    
    auto& acc = qsml::accelerator();
    acc.reset_pool_statistics();
    
    {
        auto t1 = qsml::zeros({512, 512}, DataType::F32);
    }
    
    auto stats_after_first = acc.get_pool_statistics();
    u64 first_misses = stats_after_first.cache_misses;
    
    {
        auto t2 = qsml::zeros({512, 512}, DataType::F32);
    }
    
    auto stats_after_second = acc.get_pool_statistics();
    u64 second_hits = stats_after_second.cache_hits;
    
    std::cout << "  First allocation cache misses: " << first_misses << "\n";
    std::cout << "  Second allocation cache hits: " << second_hits << "\n";
    std::cout << "  Cache hit rate: " << (stats_after_second.hit_rate * 100.0f) << "%\n";
    
    if (second_hits == 0) {
        std::cerr << "  FAIL: Second allocation didn't hit cache\n";
        return false;
    }
    
    std::cout << "  OK\n";
    return true;
}

static bool test_memory_pool_different_sizes() {
    std::cout << "[test_memory_pool_different_sizes]\n";
    
    auto& acc = qsml::accelerator();
    acc.reset_pool_statistics();
    
    std::vector<qsml::Tensor> tensors;
    for (int i = 0; i < 10; i++) {
        tensors.push_back(qsml::zeros({256, 256}, DataType::F32));
        tensors.push_back(qsml::zeros({512, 512}, DataType::F32));
        tensors.push_back(qsml::zeros({128, 128}, DataType::F32));
    }
    
    auto stats_allocated = acc.get_pool_statistics();
    std::cout << "  After allocating 30 tensors:\n";
    std::cout << "    Active allocations: " << stats_allocated.active_allocations << "\n";
    std::cout << "    Total allocated: " << stats_allocated.total_allocated_bytes << " bytes\n";
    
    tensors.clear();
    
    auto stats_cached = acc.get_pool_statistics();
    std::cout << "  After clearing:\n";
    std::cout << "    Active allocations: " << stats_cached.active_allocations << "\n";
    std::cout << "    Cached allocations: " << stats_cached.cached_allocations << "\n";
    std::cout << "    Cached bytes: " << stats_cached.total_cached_bytes << " bytes\n";
    
    for (int i = 0; i < 5; i++) {
        tensors.push_back(qsml::zeros({256, 256}, DataType::F32));
        tensors.push_back(qsml::zeros({512, 512}, DataType::F32));
    }
    
    auto stats_reused = acc.get_pool_statistics();
    std::cout << "  After reallocating 10 tensors:\n";
    std::cout << "    Cache hits: " << stats_reused.cache_hits << "\n";
    std::cout << "    Cache misses: " << stats_reused.cache_misses << "\n";
    std::cout << "    Hit rate: " << (stats_reused.hit_rate * 100.0f) << "%\n";
    
    if (stats_reused.cache_hits == 0) {
        std::cerr << "  FAIL: No cache hits on reallocation\n";
        return false;
    }
    
    std::cout << "  OK\n";
    return true;
}

static bool test_memory_pool_clear() {
    std::cout << "[test_memory_pool_clear]\n";
    
    auto& acc = qsml::accelerator();
    
    {
        auto t1 = qsml::zeros({1024, 1024}, DataType::F32);
        auto t2 = qsml::zeros({512, 512}, DataType::F32);
    }
    
    auto stats_before = acc.get_pool_statistics();
    std::cout << "  Before clear:\n";
    std::cout << "    Cached allocations: " << stats_before.cached_allocations << "\n";
    std::cout << "    Cached bytes: " << stats_before.total_cached_bytes << "\n";
    
    acc.clear_memory_pool();
    
    auto stats_after = acc.get_pool_statistics();
    std::cout << "  After clear:\n";
    std::cout << "    Cached allocations: " << stats_after.cached_allocations << "\n";
    std::cout << "    Cached bytes: " << stats_after.total_cached_bytes << "\n";
    
    if (stats_after.cached_allocations != 0 || stats_after.total_cached_bytes != 0) {
        std::cerr << "  FAIL: Cache not fully cleared\n";
        return false;
    }
    
    std::cout << "  OK\n";
    return true;
}

static bool test_memory_pool_operations() {
    std::cout << "[test_memory_pool_operations]\n";
    
    auto& acc = qsml::accelerator();
    acc.reset_pool_statistics();
    
    auto a = qsml::randn({256, 256}, DataType::F32);
    auto b = qsml::randn({256, 256}, DataType::F32);
    
    auto c = qsml::add(a, b);
    auto d = qsml::mul(c, b);
    auto e = qsml::relu(d);
    
    auto stats = acc.get_pool_statistics();
    std::cout << "  After operations:\n";
    std::cout << "    Total allocations: " << (stats.active_allocations + stats.cached_allocations) << "\n";
    std::cout << "    Cache hits: " << stats.cache_hits << "\n";
    std::cout << "    Cache misses: " << stats.cache_misses << "\n";
    std::cout << "    Hit rate: " << (stats.hit_rate * 100.0f) << "%\n";
    
    std::cout << "  OK\n";
    return true;
}

int main() {
    std::cout << "=== Memory Pool Tests ===\n\n";
    
    bool all_passed = true;
    
    all_passed &= test_memory_pool_basic();
    all_passed &= test_memory_pool_cache_hit();
    all_passed &= test_memory_pool_different_sizes();
    all_passed &= test_memory_pool_clear();
    all_passed &= test_memory_pool_operations();
    
    std::cout << "\n=== Summary ===\n";
    if (all_passed) {
        std::cout << "All memory pool tests PASSED\n";
        return 0;
    } else {
        std::cout << "Some memory pool tests FAILED\n";
        return 1;
    }
}

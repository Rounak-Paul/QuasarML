#pragma once

#include <qspch.h>
#include <VulkanBackend/VulkanBackend.h>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace QuasarML {

class MemoryPool {
public:
    struct PoolStats {
        u64 total_allocated_bytes;
        u64 total_cached_bytes;
        u64 active_allocations;
        u64 cached_allocations;
        u64 cache_hits;
        u64 cache_misses;
    };

    explicit MemoryPool(VulkanBackend* backend, VkDeviceSize max_cache_size = 512 * 1024 * 1024);
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    VulkanBackend::Buffer allocate(VkDeviceSize size, bool host_visible);
    void deallocate(VulkanBackend::Buffer buffer, bool host_visible);
    
    void clear_cache();
    void clear_cache_for_size(VkDeviceSize size, bool host_visible);
    
    PoolStats get_stats() const;
    void reset_stats();

private:
    struct CacheKey {
        VkDeviceSize size;
        bool host_visible;
        
        bool operator==(const CacheKey& other) const {
            return size == other.size && host_visible == other.host_visible;
        }
    };
    
    struct CacheKeyHash {
        std::size_t operator()(const CacheKey& key) const {
            return std::hash<VkDeviceSize>()(key.size) ^ (std::hash<bool>()(key.host_visible) << 1);
        }
    };

    VulkanBackend* _backend;
    std::unordered_map<CacheKey, std::vector<VulkanBackend::Buffer>, CacheKeyHash> _free_buffers;
    
    mutable std::mutex _mutex;
    
    u64 _total_allocated_bytes;
    u64 _total_cached_bytes;
    u64 _active_allocations;
    VkDeviceSize _max_cache_size;
    u64 _cached_allocations;
    u64 _cache_hits;
    u64 _cache_misses;

    VkDeviceSize round_up_size(VkDeviceSize size) const;
    VulkanBackend::Buffer allocate_new_buffer(VkDeviceSize size, bool host_visible);
    void evict_if_needed(VkDeviceSize incoming_size);
};

} // namespace QuasarML

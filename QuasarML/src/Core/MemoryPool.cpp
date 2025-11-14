#include "MemoryPool.h"
#include <algorithm>

namespace QuasarML {

MemoryPool::MemoryPool(VulkanBackend* backend, VkDeviceSize max_cache_size)
    : _backend(backend)
    , _total_allocated_bytes(0)
    , _total_cached_bytes(0)
    , _active_allocations(0)
    , _cached_allocations(0)
    , _cache_hits(0)
    , _cache_misses(0)
    , _max_cache_size(max_cache_size)
{
    if (!_backend) {
        throw std::invalid_argument("Backend cannot be null");
    }
}

MemoryPool::~MemoryPool() {
    clear_cache();
}

VkDeviceSize MemoryPool::round_up_size(VkDeviceSize size) const {
    if (size == 0) return 256;
    
    if (size <= 1024) {
        return ((size + 255) / 256) * 256;
    }
    else if (size <= 1024 * 1024) {
        return ((size + 1023) / 1024) * 1024;
    }
    else if (size <= 16 * 1024 * 1024) {
        VkDeviceSize mb = 1024 * 1024;
        return ((size + mb - 1) / mb) * mb;
    }
    else {
        VkDeviceSize chunk = 16 * 1024 * 1024;
        return ((size + chunk - 1) / chunk) * chunk;
    }
}

VulkanBackend::Buffer MemoryPool::allocate_new_buffer(VkDeviceSize size, bool host_visible) {
    VulkanBackend::Buffer buffer;
    
    if (host_visible) {
        buffer = _backend->create_storage_buffer(size, true);
    } else {
        buffer = _backend->create_storage_buffer(size, false);
    }
    
    return buffer;
}

VulkanBackend::Buffer MemoryPool::allocate(VkDeviceSize size, bool host_visible) {
    VkDeviceSize rounded_size = round_up_size(size);
    CacheKey key{rounded_size, host_visible};
    
    std::lock_guard<std::mutex> lock(_mutex);
    
    auto it = _free_buffers.find(key);
    if (it != _free_buffers.end() && !it->second.empty()) {
        VulkanBackend::Buffer buffer = it->second.back();
        it->second.pop_back();
        
        _total_cached_bytes -= rounded_size;
        _cached_allocations--;
        _active_allocations++;
        _cache_hits++;
        
        return buffer;
    }
    
    _cache_misses++;
    _active_allocations++;
    _total_allocated_bytes += rounded_size;
    
    VulkanBackend::Buffer buffer = allocate_new_buffer(rounded_size, host_visible);
    
    return buffer;
}

void MemoryPool::deallocate(VulkanBackend::Buffer buffer, bool host_visible) {
    if (!buffer.is_valid()) {
        return;
    }
    
    VkDeviceSize rounded_size = round_up_size(buffer.size);
    CacheKey key{rounded_size, host_visible};
    
    std::lock_guard<std::mutex> lock(_mutex);
    
    evict_if_needed(rounded_size);
    
    _free_buffers[key].push_back(buffer);
    _total_cached_bytes += rounded_size;
    _cached_allocations++;
    _active_allocations--;
}

void MemoryPool::evict_if_needed(VkDeviceSize incoming_size) {
    if (_total_cached_bytes + incoming_size <= _max_cache_size) {
        return;
    }
    
    VkDeviceSize target_free = (_total_cached_bytes + incoming_size) - (_max_cache_size * 3 / 4);
    VkDeviceSize freed = 0;
    
    for (auto it = _free_buffers.begin(); it != _free_buffers.end() && freed < target_free;) {
        if (it->second.empty()) {
            it = _free_buffers.erase(it);
            continue;
        }
        
        auto& buffer = it->second.back();
        VkDeviceSize buffer_size = buffer.size;
        _backend->destroy_buffer(buffer);
        it->second.pop_back();
        
        _total_cached_bytes -= buffer_size;
        _cached_allocations--;
        freed += buffer_size;
        
        if (it->second.empty()) {
            it = _free_buffers.erase(it);
        } else {
            ++it;
        }
    }
}

void MemoryPool::clear_cache() {
    std::lock_guard<std::mutex> lock(_mutex);
    
    for (auto& pair : _free_buffers) {
        for (auto& buffer : pair.second) {
            _backend->destroy_buffer(buffer);
        }
    }
    
    _free_buffers.clear();
    _total_cached_bytes = 0;
    _cached_allocations = 0;
}

void MemoryPool::clear_cache_for_size(VkDeviceSize size, bool host_visible) {
    VkDeviceSize rounded_size = round_up_size(size);
    CacheKey key{rounded_size, host_visible};
    
    std::lock_guard<std::mutex> lock(_mutex);
    
    auto it = _free_buffers.find(key);
    if (it != _free_buffers.end()) {
        for (auto& buffer : it->second) {
            _backend->destroy_buffer(buffer);
            _total_cached_bytes -= rounded_size;
            _cached_allocations--;
        }
        _free_buffers.erase(it);
    }
}

MemoryPool::PoolStats MemoryPool::get_stats() const {
    std::lock_guard<std::mutex> lock(_mutex);
    
    PoolStats stats;
    stats.total_allocated_bytes = _total_allocated_bytes;
    stats.total_cached_bytes = _total_cached_bytes;
    stats.active_allocations = _active_allocations;
    stats.cached_allocations = _cached_allocations;
    stats.cache_hits = _cache_hits;
    stats.cache_misses = _cache_misses;
    
    return stats;
}

void MemoryPool::reset_stats() {
    std::lock_guard<std::mutex> lock(_mutex);
    
    _cache_hits = 0;
    _cache_misses = 0;
}

} // namespace QuasarML

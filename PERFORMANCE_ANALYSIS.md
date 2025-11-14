# QuasarML Performance Analysis

## Current Performance (Apple M3 Pro)
```
MatMul 2048x2048:     250ms  (0.34 TFLOPS)  - Should be ~5-10ms
Add (16M elements):   27ms   (0.003 TFLOPS) - Should be ~2ms
VMA allocations:      6.5ms each            - Should be <0.5ms
```

## Root Causes

### 1. **Shader Compilation Overhead** ❌ CRITICAL
- **Problem**: Logs show "Compiling compute shader..." before EVERY operation
- **Impact**: ~40ms compilation time per unique shader
- **Root Cause**: Kernels ARE cached, but benchmark creates new Accelerator each time
- **Evidence**: Different kernel names should be cached, but you see compilation every run

### 2. **Synchronous Execution** ❌ CRITICAL  
- **Problem**: Each operation calls `synchronize()` → waits for GPU
- **Impact**: Kills parallelism, adds ~5-10ms latency per operation
- **Should**: Submit work → continue CPU → sync only when needed

### 3. **Poor Memory Allocation Performance** ⚠️ HIGH
- **Problem**: VMA allocations take 6.5ms each (should be <0.5ms)
- **Possible causes**:
  - Not using VMA pools effectively
  - Creating new allocations instead of reusing
  - vmaCreateBuffer() doing device memory allocation every time
  - Memory fragmentation

### 4. **Inefficient MatMul Implementation** ⚠️ HIGH
- **Current**: 250ms for 2048x2048 = 0.34 TFLOPS
- **Expected**: M3 Pro GPU ~3.5 TFLOPS → should be ~5-10ms
- **Issues**:
  - Naive O(N³) algorithm without tiling
  - No shared memory usage
  - Poor cache locality
  - Not using tensor cores (if available)

### 5. **Memory Bandwidth Underutilization** ⚠️ MEDIUM
- **Bandwidth**: M3 Pro has ~200 GB/s unified memory
- **Measured**: Add operation → 0.003 TFLOPS = ~2.4 GB/s effective
- **Should achieve**: ~50-100 GB/s for memory-bound ops
- **Issues**:
  - Small dispatch sizes
  - Not coalesced memory access
  - Synchronization overhead dominates

### 6. **Command Buffer Management** ℹ️ LOW
- **Current**: Creates command buffer per operation
- **Should**: Batch multiple operations into single command buffer
- **Impact**: Command submission overhead adds ~1-2ms per operation

## Recommended Fixes (Priority Order)

### 1. Fix Kernel Caching (IMMEDIATE - 10x speedup)
```cpp
// Problem: TensorOperations creates Accelerator reference but Accelerator is recreated
// Solution: Use single global Accelerator or ensure kernel cache persists

// In benchmark:
auto& acc = qsml::accelerator(); // Use singleton, don't create new ones
```

### 2. Implement Asynchronous Execution (HIGH - 3-5x speedup)
```cpp
// Current (bad):
auto c = add(a, b);  // Internally calls synchronize()
auto d = mul(c, e);  // Waits for previous op

// Fixed:
acc.begin_recording();
auto c = add(a, b);  // Queue work
auto d = mul(c, e);  // Queue work
acc.end_recording(); // Submit all at once
acc.synchronize();   // Wait once at end
```

### 3. Optimize MatMul with Tiling (HIGH - 20-50x speedup)
```glsl
// Current: Naive per-element
// out[row][col] = sum(A[row][k] * B[k][col])

// Needed: Tiled MatMul with shared memory
layout(local_size_x = 16, local_size_y = 16) in;
shared float tileA[16][16];
shared float tileB[16][16];
// Load tiles → compute → accumulate
```

### 4. Pre-warm Shader Cache (MEDIUM - eliminates 40ms overhead)
```cpp
// Create all common kernels at initialization
void Accelerator::prewarm_shaders() {
    // Create add, mul, matmul, etc. kernels once
    ops().add(zeros({1,1}), zeros({1,1}));  // Force compilation
    ops().matmul(zeros({2,2}), zeros({2,2}));
}
```

### 5. Implement Memory Pool (MEDIUM - 10x allocation speedup)
```cpp
// Keep pool of pre-allocated buffers
class BufferPool {
    std::map<size_t, std::vector<Buffer>> free_buffers_;
    Buffer allocate(size_t size) {
        // Round up to power of 2
        // Return from pool or allocate new
    }
};
```

### 6. Optimize Memory Access Patterns (LOW-MEDIUM - 2x bandwidth)
```glsl
// Coalesced reads: threads access consecutive memory
uint idx = gl_GlobalInvocationID.x;
// Good: data[idx], data[idx+1], data[idx+2], ...
// Bad:  data[idx*stride], data[random], ...
```

## Expected Performance After Fixes
```
MatMul 2048x2048:     ~8ms   (10+ TFLOPS)
Add (16M elements):   ~2ms   (0.5 TFLOPS)
VMA allocations:      <0.5ms
Overall:              20-50x faster
```

## Quick Wins (Can implement now)

1. **Use single global accelerator** in benchmarks
2. **Remove synchronize() calls** from individual operations
3. **Batch operations** with begin_recording/end_recording
4. **Pre-compile common shaders** at startup

## Long-term Optimizations

1. **Implement tiled MatMul** with shared memory
2. **Add memory pool** with size-based buckets
3. **Pipeline CPU/GPU work** with double buffering
4. **Optimize kernel dispatch sizes** (use 256 threads per workgroup)
5. **Add profiling** with Vulkan timestamps

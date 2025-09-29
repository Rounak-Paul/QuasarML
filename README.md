# 🚀 Roadmap: QuasarML

This roadmap outlines the development plan for building a **feature-rich scientific computing & ML library** in C++ with Vulkan acceleration.  
The goal is to combine the usability of **NumPy** with the deep learning power of **PyTorch**.

---

## ✅ Phase 1 — Core Foundations (MVP)
**Goal:** A minimal but usable Vulkan-accelerated tensor library.  

 - [x] Tensor class (shape, strides, dtype, device, ownership flags)  
 - [x] Memory management (CPU + GPU buffers, host-device transfers)  
 - [x] Basic elementwise ops (add, mul, sub, div)  
 - [x] pow  
 - [x] Reductions: sum (axis)  
 - [x] mean  
 - [x] min  
 - [x] max  
 - [x] Broadcasting rules (NumPy-style)  
 - [x] Indexing/slicing

### Recent progress (summary)

- Implemented GPU-first non-contiguous slicing using a generic strided-gather compute shader (no CPU fallback).
- Added N‑D two-pass reductions (sum/mean/min/max) and pow; all Phase 1 functionality now implemented and covered by unit tests.
 - Implemented GPU-first non-contiguous slicing using a generic strided-gather compute shader (no CPU fallback).
 - Added N‑D two-pass reductions (sum/mean/min/max) and pow; all Phase 1 functionality implemented and covered by unit tests.
 - Operator overloads for shared_ptr<Tensor> (ergonomic arithmetic) are implemented and tested.
 - CPU/GPU mode control and instrumentation (CPU-fallback counter) are available and exercised by unit tests.
 - Vulkan backend writes failing GLSL snippets to /tmp for shader-debug dumps when compilation fails.
 - Vendor allocators (mimalloc) and Vulkan memory allocator (VMA) are integrated as dependencies.
 - Unit tests: full suite passes locally (19/19) on the repo I inspected.

---

## ✅ Phase 2 — NumPy Parity
**Goal:** Become a NumPy alternative in C++.  

 - [x] Basic elementwise ops (add, mul, sub, div)
 - [x] pow
 - [x] ReLU (and other small elementwise helpers)
 - [ ] More elementwise ops (exp, log, sin, cos, sqrt, etc.)
 - [ ] Advanced indexing (boolean masks, fancy indexing)
 - [x] Matrix operations: matmul, transpose, reshape, flatten
	 - note: matmul and transpose have kernel implementations; reshape/flatten exist as zero-copy views.
 - [ ] dot (alias / convenience)
 - [ ] Random number generation (uniform, normal, Bernoulli)
 - [ ] I/O basics (save/load tensors from disk, file extention ".qsbin" for "Quasar Binary")

---

## ✅ Phase 3 — Autograd Engine
**Goal:** Add PyTorch-style automatic differentiation.  

- [ ] Computation graph (track ops + parents)  
- [ ] Backpropagation (chain rule)  
- [ ] `.grad` fields on tensors  
- [ ] Support for in-place ops  
- [ ] Custom gradient support  

---

## ✅ Phase 4 — Neural Network API
**Goal:** Build the `torch.nn` equivalent.  

- [ ] Modules / Layers (Linear, Conv2D, Dropout, BatchNorm)  
- [ ] Activations (ReLU, Sigmoid, Tanh, GELU)  
- [ ] Loss functions (MSE, CrossEntropy)  
- [ ] Optimizers (SGD, Adam)  
- [ ] Parameter management (save/load weights)  

---

## ✅ Phase 5 — Performance & Vulkan Magic
**Goal:** Make it fast and scalable.  

 - [x] Vulkan kernel library (optimized shaders for ops)  
- [ ] Kernel fusion (combine multiple ops into one dispatch)  
 - [ ] Kernel fusion (combine multiple ops into one dispatch)  
 - [ ] Mixed precision support (FP16, BF16)  
	 - note: FP16 (F16) has explicit handling in several ops; BF16 support not present.
 - [ ] Memory pool / caching (avoid malloc/free overhead)  
	 - note: allocator/backends include mimalloc and VMA (vendor libs are integrated), but there is no QuasarML-specific pooled allocator yet.
 - [ ] Multi-GPU support (basic)  

---

## ✅ Phase 6 — ML Ecosystem Features
**Goal:** Add developer convenience features.  

- [ ] Data loaders (batching, shuffling)  
- [ ] Metrics (accuracy, precision, F1)  
- [ ] Callbacks (checkpoints, LR schedulers)  
- [ ] Logging/profiling utilities  

---

## ✅ Phase 7 — Advanced & Research-Level Features
**Goal:** Compete with cutting-edge ML frameworks.  

- [ ] Transformers (Attention, MHA, LayerNorm, etc.)  
- [ ] Distributed training (multi-node, NCCL-like over Vulkan)  
- [ ] Sparse tensor support  
- [ ] Quantization / pruning for inference  
- [ ] ONNX model import/export  

---

## 🔑 Key Advice
- **Phases 1–2 (Core + NumPy parity)** are the foundation.  
- **Phase 3 (autograd)** makes it PyTorch-like.  
- **Phases 4–5 (nn + performance)** make it competitive.  
- **Phases 6–7** are advanced and can come later.  
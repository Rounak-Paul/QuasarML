# 🚀 Roadmap: QuasarML

This roadmap outlines the development plan for building a **feature-rich scientific computing & ML library** in C++ with Vulkan acceleration.  
The goal is to combine the usability of **NumPy** with the deep learning power of **PyTorch**.

---

## ✅ Phase 1 — Core Foundations (MVP)
**Goal:** A minimal but usable Vulkan-accelerated tensor library.  

 - [x] Tensor class (shape, strides, dtype, device, ownership flags)  
 - [x] Memory management (CPU + GPU buffers, host-device transfers)  
 - [x] Basic elementwise ops (add, mul, sub, div)  
 - [ ] pow  
 - [ ] Reductions: sum (axis)  
 - [ ] mean  
 - [ ] min  
 - [ ] max  
 - [x] Broadcasting rules (NumPy-style)  
 - [ ] Indexing/slicing

---

## ✅ Phase 2 — NumPy Parity
**Goal:** Become a NumPy alternative in C++.  

- [ ] More elementwise ops (exp, log, sin, cos, sqrt, etc.)  
- [ ] Advanced indexing (boolean masks, fancy indexing)  
 - [x] Matrix operations: matmul, transpose, reshape, flatten  
 - [ ] dot (alias / convenience)  
- [ ] Random number generation (uniform, normal, Bernoulli)  
- [ ] I/O basics (save/load tensors from disk, NumPy `.npy` support)  

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
- [ ] Mixed precision support (FP16, BF16)  
- [ ] Memory pool / caching (avoid malloc/free overhead)  
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
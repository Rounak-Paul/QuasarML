# NanoGPT - Tiny Language Model

A complete GPT-style transformer language model implemented using QuasarML.

## Features

- **Full Transformer Architecture**: Multi-head self-attention, layer normalization, MLP blocks
- **GPU Accelerated**: All operations run on Metal via QuasarML's Vulkan backend
- **Minimal Dependencies**: Only QuasarML required
- **Character-level Tokenizer**: Simple 34-token vocabulary (a-z, punctuation)
- **Text Generation**: Autoregressive sampling with temperature control

## Model Architecture

```
Vocabulary: 34 tokens
Context Length: 32 tokens
Layers: 2 transformer blocks
Attention Heads: 4
Embedding Dimension: 64
Total Parameters: ~85K
```

## Building

```bash
cd examples/nano-gpt
mkdir build && cd build
cmake ..
make
```

## Running

```bash
../../bin/nano_gpt
```

## Implementation Details

The model implements:
- Token + Positional embeddings
- Multi-head self-attention with scaled dot-product
- Layer normalization (pre-norm architecture)
- MLP with GELU activation (4x expansion ratio)
- Temperature-based sampling for generation

## Notes

This is a **demonstration model** showing how to build a complete neural architecture with QuasarML. The model is randomly initialized and not trained. To train it, you would need:

1. Training dataset (text corpus)
2. Optimizer (Adam/SGD) 
3. Cross-entropy loss computation
4. Gradient computation and backpropagation
5. Training loop with batching

The architecture follows the GPT design closely but at a tiny scale for demonstration purposes.

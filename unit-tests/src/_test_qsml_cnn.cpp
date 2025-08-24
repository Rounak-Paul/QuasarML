// cnn_demo.cpp - CNN Implementation using QuasarML Accelerator
#include <QuasarML.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <memory>

using namespace QuasarML;

class CNNDemo {
private:
    Accelerator accel;
    std::mt19937 rng;
    std::uniform_real_distribution<float> weight_dist;
    std::normal_distribution<float> normal_dist;
    
    // Network parameters
    struct ConvLayer {
        std::shared_ptr<Tensor> weights;    // [out_channels, in_channels, kernel_h, kernel_w]
        std::shared_ptr<Tensor> bias;       // [out_channels]
        u32 in_channels, out_channels;
        u32 kernel_size;
        u32 stride, padding;
        std::string name;
    };
    
    struct FCLayer {
        std::shared_ptr<Tensor> weights;    // [out_features, in_features]
        std::shared_ptr<Tensor> bias;       // [out_features]
        u32 in_features, out_features;
        std::string name;
    };
    
    std::vector<ConvLayer> conv_layers;
    std::vector<FCLayer> fc_layers;
    
    // Custom kernels
    std::shared_ptr<Kernel> conv2d_kernel;
    std::shared_ptr<Kernel> maxpool2d_kernel;
    std::shared_ptr<Kernel> flatten_kernel;
    
public:
    CNNDemo() : accel("CNN_Demo"), rng(42), weight_dist(-0.1f, 0.1f), normal_dist(0.0f, 0.1f) {
        log_section("QuasarML CNN Demo - Image Classification");
        initialize_kernels();
        build_network();
    }
    
    void log_section(const std::string& section) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "  " << section << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }
    
    void log_info(const std::string& info) {
        std::cout << "[INFO] " << info << std::endl;
    }
    
    void initialize_kernels() {
        log_info("Initializing custom compute kernels...");
        
        // 2D Convolution kernel (simplified version for 1D dispatch)
        const char* conv2d_glsl = R"(
#version 450
layout(local_size_x = 64) in;

layout(binding = 0) readonly buffer Input { float data[]; } input_buf;
layout(binding = 1) readonly buffer Weights { float data[]; } weight_buf;
layout(binding = 2) readonly buffer Bias { float data[]; } bias_buf;
layout(binding = 3) writeonly buffer Output { float data[]; } output_buf;

layout(push_constant) uniform PushData {
    uint input_h, input_w, input_c;
    uint output_h, output_w, output_c;
    uint kernel_size, stride, padding;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total_outputs = output_h * output_w * output_c;
    if (idx >= total_outputs) return;
    
    // Decode output position
    uint out_c = idx % output_c;
    uint temp = idx / output_c;
    uint out_x = temp % output_w;
    uint out_y = temp / output_w;
    
    float sum = 0.0;
    
    // Convolution operation
    for (uint in_c = 0; in_c < input_c; ++in_c) {
        for (uint ky = 0; ky < kernel_size; ++ky) {
            for (uint kx = 0; kx < kernel_size; ++kx) {
                int in_y = int(out_y * stride + ky) - int(padding);
                int in_x = int(out_x * stride + kx) - int(padding);
                
                if (in_y >= 0 && in_y < int(input_h) && in_x >= 0 && in_x < int(input_w)) {
                    uint in_idx = uint(in_y) * input_w * input_c + uint(in_x) * input_c + in_c;
                    uint w_idx = out_c * input_c * kernel_size * kernel_size + 
                               in_c * kernel_size * kernel_size + ky * kernel_size + kx;
                    sum += input_buf.data[in_idx] * weight_buf.data[w_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias_buf.data[out_c];
    
    output_buf.data[idx] = sum;
}
)";

        // MaxPool2D kernel (1D dispatch version)
        const char* maxpool2d_glsl = R"(
#version 450
layout(local_size_x = 64) in;

layout(binding = 0) readonly buffer Input { float data[]; } input_buf;
layout(binding = 1) writeonly buffer Output { float data[]; } output_buf;

layout(push_constant) uniform PushData {
    uint input_h, input_w, channels;
    uint output_h, output_w;
    uint pool_size, stride;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total_outputs = output_h * output_w * channels;
    if (idx >= total_outputs) return;
    
    // Decode output position
    uint c = idx % channels;
    uint temp = idx / channels;
    uint out_x = temp % output_w;
    uint out_y = temp / output_w;
    
    float max_val = -3.402823466e+38; // -FLT_MAX
    
    for (uint py = 0; py < pool_size; ++py) {
        for (uint px = 0; px < pool_size; ++px) {
            uint in_y = out_y * stride + py;
            uint in_x = out_x * stride + px;
            
            if (in_y < input_h && in_x < input_w) {
                uint in_idx = in_y * input_w * channels + in_x * channels + c;
                max_val = max(max_val, input_buf.data[in_idx]);
            }
        }
    }
    
    output_buf.data[idx] = max_val;
}
)";

        // Flatten kernel
        const char* flatten_glsl = R"(
#version 450
layout(local_size_x = 64) in;

layout(binding = 0) readonly buffer Input { float data[]; } input_buf;
layout(binding = 1) writeonly buffer Output { float data[]; } output_buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= input_buf.data.length()) return;
    output_buf.data[idx] = input_buf.data[idx];
}
)";
        
        // Create kernels
        conv2d_kernel = accel.create_kernel("conv2d", conv2d_glsl, 4, sizeof(u32) * 9);
        maxpool2d_kernel = accel.create_kernel("maxpool2d", maxpool2d_glsl, 2, sizeof(u32) * 7);
        flatten_kernel = accel.create_kernel("flatten", flatten_glsl, 2);
        
        if (!conv2d_kernel || !maxpool2d_kernel || !flatten_kernel) {
            throw std::runtime_error("Failed to create custom kernels");
        }
        
        log_info("Custom kernels initialized successfully");
    }
    
    void build_network() {
        log_info("Building CNN architecture...");
        
        // Network architecture: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Flatten -> FC -> ReLU -> FC
        // Input: 28x28x1 (MNIST-like)
        
        // Conv Layer 1: 1 -> 32 channels, 5x5 kernel
        conv_layers.push_back(create_conv_layer("conv1", 1, 32, 5, 1, 2, 28, 28));
        
        // Conv Layer 2: 32 -> 64 channels, 5x5 kernel  
        conv_layers.push_back(create_conv_layer("conv2", 32, 64, 5, 1, 2, 14, 14));
        
        // FC Layer 1: 7*7*64 -> 128
        fc_layers.push_back(create_fc_layer("fc1", 7 * 7 * 64, 128));
        
        // FC Layer 2: 128 -> 10 (output classes)
        fc_layers.push_back(create_fc_layer("fc2", 128, 10));
        
        log_info("CNN architecture built:");
        log_info("  Input: 28x28x1");
        log_info("  Conv1: 1->32 channels, 5x5 kernel -> 28x28x32");
        log_info("  MaxPool1: 2x2 -> 14x14x32");
        log_info("  Conv2: 32->64 channels, 5x5 kernel -> 14x14x64");
        log_info("  MaxPool2: 2x2 -> 7x7x64");
        log_info("  Flatten: -> 3136");
        log_info("  FC1: 3136->128");
        log_info("  FC2: 128->10");
        
        // Display parameter count
        u32 total_params = 0;
        for (const auto& layer : conv_layers) {
            u32 weight_params = layer.out_channels * layer.in_channels * layer.kernel_size * layer.kernel_size;
            u32 bias_params = layer.out_channels;
            total_params += weight_params + bias_params;
            log_info("  " + layer.name + " parameters: " + std::to_string(weight_params + bias_params));
        }
        for (const auto& layer : fc_layers) {
            u32 weight_params = layer.out_features * layer.in_features;
            u32 bias_params = layer.out_features;
            total_params += weight_params + bias_params;
            log_info("  " + layer.name + " parameters: " + std::to_string(weight_params + bias_params));
        }
        log_info("Total parameters: " + std::to_string(total_params));
    }
    
    ConvLayer create_conv_layer(const std::string& name, u32 in_channels, u32 out_channels, 
                               u32 kernel_size, u32 stride, u32 padding, u32 input_h, u32 input_w) {
        ConvLayer layer;
        layer.name = name;
        layer.in_channels = in_channels;
        layer.out_channels = out_channels;
        layer.kernel_size = kernel_size;
        layer.stride = stride;
        layer.padding = padding;
        
        // Initialize weights with Xavier initialization
        u32 weight_count = out_channels * in_channels * kernel_size * kernel_size;
        std::vector<float> weights(weight_count);
        float xavier_std = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
        
        for (float& w : weights) {
            w = normal_dist(rng) * xavier_std;
        }
        
        // Initialize bias to small positive values
        std::vector<float> bias(out_channels);
        for (float& b : bias) {
            b = 0.01f;
        }
        
        layer.weights = accel.create_tensor(weights.data(), {weight_count}, DataType::F32);
        layer.bias = accel.create_tensor(bias.data(), {out_channels}, DataType::F32);
        
        return layer;
    }
    
    FCLayer create_fc_layer(const std::string& name, u32 in_features, u32 out_features) {
        FCLayer layer;
        layer.name = name;
        layer.in_features = in_features;
        layer.out_features = out_features;
        
        // Initialize weights with Xavier initialization
        u32 weight_count = out_features * in_features;
        std::vector<float> weights(weight_count);
        float xavier_std = std::sqrt(2.0f / in_features);
        
        for (float& w : weights) {
            w = normal_dist(rng) * xavier_std;
        }
        
        // Initialize bias to small positive values
        std::vector<float> bias(out_features);
        for (float& b : bias) {
            b = 0.01f;
        }
        
        layer.weights = accel.create_tensor(weights.data(), {out_features, in_features}, DataType::F32);
        layer.bias = accel.create_tensor(bias.data(), {out_features}, DataType::F32);
        
        return layer;
    }
    
    std::shared_ptr<Tensor> conv2d_forward(std::shared_ptr<Tensor> input, const ConvLayer& layer,
                                          u32 input_h, u32 input_w, u32& output_h, u32& output_w) {
        output_h = (input_h + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
        output_w = (input_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
        
        auto output = accel.create_tensor({output_h * output_w * layer.out_channels}, DataType::F32);
        
        // Push constants for the kernel
        struct ConvPushData {
            u32 input_h, input_w, input_c;
            u32 output_h, output_w, output_c;
            u32 kernel_size, stride, padding;
        } push_data = {
            input_h, input_w, layer.in_channels,
            output_h, output_w, layer.out_channels,
            layer.kernel_size, layer.stride, layer.padding
        };
        
        // Calculate dispatch size for 1D workgroup covering all output elements
        u32 total_outputs = output_h * output_w * layer.out_channels;
        auto dispatch = accel.calculate_optimal_dispatch_1d(total_outputs, 64);
        
        accel.execute(conv2d_kernel, {input, layer.weights, layer.bias, output}, 
                     dispatch, 1, 1, &push_data);
        
        return output;
    }
    
    std::shared_ptr<Tensor> maxpool2d_forward(std::shared_ptr<Tensor> input, u32 input_h, u32 input_w, 
                                             u32 channels, u32 pool_size, u32 stride,
                                             u32& output_h, u32& output_w) {
        output_h = (input_h - pool_size) / stride + 1;
        output_w = (input_w - pool_size) / stride + 1;
        
        auto output = accel.create_tensor({output_h * output_w * channels}, DataType::F32);
        
        struct PoolPushData {
            u32 input_h, input_w, channels;
            u32 output_h, output_w;
            u32 pool_size, stride;
        } push_data = {
            input_h, input_w, channels,
            output_h, output_w,
            pool_size, stride
        };
        
        u32 total_outputs = output_h * output_w * channels;
        auto dispatch = accel.calculate_optimal_dispatch_1d(total_outputs, 64);
        
        accel.execute(maxpool2d_kernel, {input, output}, dispatch, 1, 1, &push_data);
        
        return output;
    }
    
    std::shared_ptr<Tensor> flatten_forward(std::shared_ptr<Tensor> input, u32 total_elements) {
        auto output = accel.create_tensor({total_elements}, DataType::F32);
        
        auto dispatch = accel.calculate_optimal_dispatch_1d(total_elements, 64);
        accel.execute(flatten_kernel, {input, output}, dispatch);
        
        return output;
    }
    
    std::shared_ptr<Tensor> fc_forward(std::shared_ptr<Tensor> input, const FCLayer& layer) {
        // For matrix multiplication: input [1, in_features] @ weights [in_features, out_features]
        // But our weights are stored as [out_features, in_features], so we need to transpose
        auto weights_t = accel.ops().transpose(layer.weights);  // Now [in_features, out_features]
        auto output = accel.ops().matmul(input, weights_t);
        
        // Add bias (broadcast addition)
        output = accel.ops().add(output, layer.bias);
        
        return output;
    }
    
    std::shared_ptr<Tensor> generate_sample_input() {
        // Generate a synthetic 28x28 image (like MNIST digit)
        const u32 size = 28 * 28;
        std::vector<float> image_data(size);
        
        // Create a simple pattern - a crude "8" shape
        for (u32 y = 0; y < 28; ++y) {
            for (u32 x = 0; x < 28; ++x) {
                u32 idx = y * 28 + x;
                float val = 0.0f;
                
                // Top circle of "8"
                float dx1 = x - 14.0f, dy1 = y - 8.0f;
                if (dx1*dx1 + dy1*dy1 < 25 && dx1*dx1 + dy1*dy1 > 16) val = 1.0f;
                
                // Bottom circle of "8"
                float dx2 = x - 14.0f, dy2 = y - 20.0f;
                if (dx2*dx2 + dy2*dy2 < 25 && dx2*dx2 + dy2*dy2 > 16) val = 1.0f;
                
                // Connection in the middle
                if (std::abs(x - 14.0f) < 2 && y >= 12 && y <= 16) val = 1.0f;
                
                // Add some noise
                val += (weight_dist(rng) * 0.1f);
                image_data[idx] = std::max(0.0f, std::min(1.0f, val));
            }
        }
        
        return accel.create_tensor(image_data.data(), {size}, DataType::F32);
    }
    
    void print_image(std::shared_ptr<Tensor> image_tensor, u32 width = 28, u32 height = 28) {
        std::vector<float> image_data(width * height);
        image_tensor->download_data(image_data.data());
        
        std::cout << "Input image visualization:\n";
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                float val = image_data[y * width + x];
                if (val > 0.7f) std::cout << "██";
                else if (val > 0.4f) std::cout << "▓▓";
                else if (val > 0.2f) std::cout << "░░";
                else std::cout << "  ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
    
    void run_inference() {
        log_section("Running CNN Inference");
        
        // Generate sample input
        auto input = generate_sample_input();
        log_info("Generated synthetic 28x28 input image");
        print_image(input);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Forward pass through the network
        log_info("Forward pass...");
        
        // Conv1 + ReLU + MaxPool
        log_info("  Layer 1: Convolution 1->32 channels");
        u32 h = 28, w = 28;
        auto conv1_out = conv2d_forward(input, conv_layers[0], h, w, h, w);
        conv1_out = accel.ops().relu(conv1_out);
        
        log_info("  Layer 2: MaxPool 2x2");
        auto pool1_out = maxpool2d_forward(conv1_out, h, w, 32, 2, 2, h, w);
        
        // Conv2 + ReLU + MaxPool
        log_info("  Layer 3: Convolution 32->64 channels");
        auto conv2_out = conv2d_forward(pool1_out, conv_layers[1], h, w, h, w);
        conv2_out = accel.ops().relu(conv2_out);
        
        log_info("  Layer 4: MaxPool 2x2");
        auto pool2_out = maxpool2d_forward(conv2_out, h, w, 64, 2, 2, h, w);
        
        // Flatten
        log_info("  Layer 5: Flatten");
        u32 flattened_size = h * w * 64;
        auto flattened = flatten_forward(pool2_out, flattened_size);
        
        // Create proper 2D tensor for matrix multiplication [1, flattened_size]
        std::vector<float> temp_data(flattened_size);
        flattened->download_data(temp_data.data());
        auto reshaped_input = accel.create_tensor(temp_data.data(), {1, flattened_size}, DataType::F32);
        
        // FC1 + ReLU
        log_info("  Layer 6: Fully Connected 3136->128");
        auto fc1_out = fc_forward(reshaped_input, fc_layers[0]);
        fc1_out = accel.ops().relu(fc1_out);
        
        // FC2 (output layer)
        log_info("  Layer 7: Fully Connected 128->10");
        auto fc2_out = fc_forward(fc1_out, fc_layers[1]);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Get final predictions
        std::vector<float> predictions(10);
        fc2_out->download_data(predictions.data());
        
        // Apply softmax for visualization
        float max_val = *std::max_element(predictions.begin(), predictions.end());
        float sum = 0.0f;
        for (float& p : predictions) {
            p = std::exp(p - max_val);
            sum += p;
        }
        for (float& p : predictions) {
            p /= sum;
        }
        
        // Display results
        log_info("Inference completed in " + std::to_string(duration.count()) + " ms");
        std::cout << "\nClass probabilities:\n";
        for (int i = 0; i < 10; ++i) {
            std::cout << "  Class " << i << ": " << std::fixed << std::setprecision(4) 
                     << predictions[i] << " ";
            
            // Simple bar visualization
            int bar_length = static_cast<int>(predictions[i] * 50);
            for (int j = 0; j < bar_length; ++j) std::cout << "█";
            std::cout << std::endl;
        }
        
        int predicted_class = std::max_element(predictions.begin(), predictions.end()) - predictions.begin();
        std::cout << "\nPredicted class: " << predicted_class 
                 << " (confidence: " << std::fixed << std::setprecision(2) 
                 << predictions[predicted_class] * 100 << "%)" << std::endl;
    }
    
    void benchmark_performance() {
        log_section("Performance Benchmarking");
        
        // Benchmark individual operations
        const int num_runs = 10;
        
        // Convolution benchmark
        auto input = generate_sample_input();
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; ++i) {
            u32 h = 28, w = 28;
            auto conv_out = conv2d_forward(input, conv_layers[0], h, w, h, w);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double conv_time = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
        log_info("Average Conv2D time: " + std::to_string(conv_time) + " ms");
        
        // Matrix multiplication benchmark
        auto flat_input = accel.create_tensor({1, 3136}, DataType::F32);
        
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; ++i) {
            auto fc_out = fc_forward(flat_input, fc_layers[0]);
        }
        end = std::chrono::high_resolution_clock::now();
        
        double fc_time = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
        log_info("Average FC layer time: " + std::to_string(fc_time) + " ms");
        
        // Memory usage
        auto [used, total] = accel.get_memory_usage();
        log_info("GPU Memory usage: " + std::to_string(used / 1024 / 1024) + 
                " MB / " + std::to_string(total / 1024 / 1024) + " MB");
    }
    
    void run_demo() {
        try {
            run_inference();
            benchmark_performance();
            
            log_section("Demo Summary");
            log_info("✅ Successfully demonstrated CNN capabilities:");
            log_info("  - Custom GLSL compute shaders for Conv2D and MaxPool2D");
            log_info("  - Built-in tensor operations (MatMul, ReLU, Add, etc.)");
            log_info("  - Multi-layer neural network inference");
            log_info("  - Memory management and GPU acceleration");
            log_info("  - Real-time performance benchmarking");
            
        } catch (const std::exception& e) {
            std::cout << "[ERROR] Demo failed: " << e.what() << std::endl;
        }
    }
};

int main() {
    std::cout << std::fixed << std::setprecision(4);
    
    try {
        CNNDemo demo;
        demo.run_demo();
        return 0;
    } catch (const std::exception& e) {
        std::cout << "[FATAL] Failed to initialize CNN demo: " << e.what() << std::endl;
        return 1;
    }
}
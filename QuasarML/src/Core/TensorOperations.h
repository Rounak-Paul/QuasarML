#pragma once

#include <qspch.h>
#include "DataTypes.h"
#include <memory>
#include <vector>

namespace QuasarML {

// Forward declarations
class Accelerator;
class Tensor;
class Kernel;

class TensorOperations {
private:
    Accelerator& _accelerator;
    friend class Accelerator;
    
    explicit TensorOperations(Accelerator& acc) : _accelerator(acc) {}
    

public:
    // Element-wise operations
    std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

    // Note: operator overloads are intentionally omitted; use explicit methods for clarity
    
    // Scalar operations
    std::shared_ptr<Tensor> add_scalar(std::shared_ptr<Tensor> tensor, float scalar);
    std::shared_ptr<Tensor> mul_scalar(std::shared_ptr<Tensor> tensor, float scalar);

    // scalar on left convenience
    std::shared_ptr<Tensor> add_scalar_left(float scalar, std::shared_ptr<Tensor> tensor) { return add_scalar(tensor, scalar); }
    std::shared_ptr<Tensor> mul_scalar_left(float scalar, std::shared_ptr<Tensor> tensor) { return mul_scalar(tensor, scalar); }
    
    // Activation functions
    std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> tensor);
    
    // Linear algebra operations
    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    std::shared_ptr<Tensor> transpose(std::shared_ptr<Tensor> tensor);
    
    // Reduction operations
    std::shared_ptr<Tensor> sum_axis(std::shared_ptr<Tensor> tensor, u32 axis);

private:
    void validate_tensor_op_compatibility(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) const;
    void validate_tensor_shape_2d(std::shared_ptr<Tensor> tensor) const;
    std::vector<u32> compute_broadcast_shape(const std::vector<u32>& a, const std::vector<u32>& b) const;
    std::shared_ptr<Kernel> get_or_create_kernel(const std::string& name, 
                                                const std::string& glsl_source, 
                                                u32 num_tensors, u32 push_constant_size = 0);
    
    std::string generate_elementwise_kernel_source(const std::string& operation, 
                                                    DataType dtype, 
                                                    bool is_scalar = false,
                                                    bool supports_broadcast = false,
                                                    u32 max_rank = 0) const;
    std::vector<u32> compute_strides_padded(const std::vector<u32>& shape, u32 rank) const;
    std::string generate_matmul_kernel_source(DataType dtype) const;
    std::string generate_transpose_kernel_source(DataType dtype) const;
    std::string generate_sum_axis_kernel_source(DataType dtype) const;
    std::string generate_relu_kernel_source(DataType dtype) const;
    std::string get_kernel_name_for_dtype(const std::string& base_name, DataType dtype) const;
};

} // namespace QuasarML
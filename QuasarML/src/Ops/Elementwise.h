#pragma once

#include <Core/Tensor.h>
#include <Core/Device.h>
#include <memory>

namespace QuasarML {
namespace Ops {

std::shared_ptr<Tensor> add(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> subtract(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> multiply(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> divide(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

std::shared_ptr<Tensor> add_scalar(Device& device, std::shared_ptr<Tensor> a, f32 scalar);
std::shared_ptr<Tensor> multiply_scalar(Device& device, std::shared_ptr<Tensor> a, f32 scalar);

std::shared_ptr<Tensor> neg(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> abs(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sqrt(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> exp(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> log(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sin(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> cos(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> tanh(Device& device, std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> relu(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sigmoid(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> gelu(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> softmax(Device& device, std::shared_ptr<Tensor> a, i32 axis = -1);

}
}

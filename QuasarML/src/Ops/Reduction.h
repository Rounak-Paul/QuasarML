#pragma once

#include <Core/Tensor.h>
#include <Core/Device.h>
#include <memory>

namespace QuasarML {
namespace Ops {

std::shared_ptr<Tensor> sum(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sum(Device& device, std::shared_ptr<Tensor> a, i32 axis, bool keepdims = false);

std::shared_ptr<Tensor> mean(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> mean(Device& device, std::shared_ptr<Tensor> a, i32 axis, bool keepdims = false);

std::shared_ptr<Tensor> max(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> max(Device& device, std::shared_ptr<Tensor> a, i32 axis, bool keepdims = false);

std::shared_ptr<Tensor> min(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> min(Device& device, std::shared_ptr<Tensor> a, i32 axis, bool keepdims = false);

std::shared_ptr<Tensor> prod(Device& device, std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> argmax(Device& device, std::shared_ptr<Tensor> a, i32 axis = -1);
std::shared_ptr<Tensor> argmin(Device& device, std::shared_ptr<Tensor> a, i32 axis = -1);

}
}

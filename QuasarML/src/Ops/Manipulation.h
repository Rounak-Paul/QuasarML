#pragma once

#include <Core/Tensor.h>
#include <Core/Device.h>
#include <memory>

namespace QuasarML {
namespace Ops {

std::shared_ptr<Tensor> reshape(Device& device, std::shared_ptr<Tensor> a, const std::vector<u32>& new_shape);
std::shared_ptr<Tensor> flatten(Device& device, std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> squeeze(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> squeeze(Device& device, std::shared_ptr<Tensor> a, i32 dim);
std::shared_ptr<Tensor> unsqueeze(Device& device, std::shared_ptr<Tensor> a, i32 dim);

std::shared_ptr<Tensor> concat(Device& device, const std::vector<std::shared_ptr<Tensor>>& tensors, i32 axis = 0);
std::shared_ptr<Tensor> stack(Device& device, const std::vector<std::shared_ptr<Tensor>>& tensors, i32 axis = 0);

std::shared_ptr<Tensor> slice(Device& device, std::shared_ptr<Tensor> a, i32 axis, u64 start, u64 end);

std::shared_ptr<Tensor> broadcast_to(Device& device, std::shared_ptr<Tensor> a, const std::vector<u32>& shape);

std::shared_ptr<Tensor> copy(Device& device, std::shared_ptr<Tensor> a);

}
}

#pragma once

#include <Core/Tensor.h>
#include <Core/Device.h>
#include <memory>

namespace QuasarML {
namespace Ops {

std::shared_ptr<Tensor> matmul(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> batch_matmul(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

std::shared_ptr<Tensor> dot(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

std::shared_ptr<Tensor> transpose(Device& device, std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> transpose(Device& device, std::shared_ptr<Tensor> a, i32 dim0, i32 dim1);

std::shared_ptr<Tensor> outer(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

}
}

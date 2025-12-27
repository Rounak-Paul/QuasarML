#pragma once

#include <Core/Tensor.h>
#include <Core/Device.h>
#include <memory>

namespace QuasarML {
namespace Ops {

QS_API std::shared_ptr<Tensor> matmul(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
QS_API std::shared_ptr<Tensor> batch_matmul(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

QS_API std::shared_ptr<Tensor> dot(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

QS_API std::shared_ptr<Tensor> transpose(Device& device, std::shared_ptr<Tensor> a);
QS_API std::shared_ptr<Tensor> transpose(Device& device, std::shared_ptr<Tensor> a, i32 dim0, i32 dim1);

QS_API std::shared_ptr<Tensor> outer(Device& device, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

}
}

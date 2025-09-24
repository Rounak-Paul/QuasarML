#pragma once
#include "Tensor.h"
#include <memory>

namespace QuasarML {
inline std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return (*a) + b; }
inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return (*a) - b; }
inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return (*a) * b; }
inline std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return (*a) / b; }

inline std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, float scalar) { return (*a) + scalar; }
inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, float scalar) { return (*a) - scalar; }
inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, float scalar) { return (*a) * scalar; }
inline std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, float scalar) { return (*a) / scalar; }

inline std::shared_ptr<Tensor> operator+(float scalar, const std::shared_ptr<Tensor>& a) { return scalar + (*a); }
inline std::shared_ptr<Tensor> operator-(float scalar, const std::shared_ptr<Tensor>& a) { return scalar - (*a); }
inline std::shared_ptr<Tensor> operator*(float scalar, const std::shared_ptr<Tensor>& a) { return scalar * (*a); }
inline std::shared_ptr<Tensor> operator/(float scalar, const std::shared_ptr<Tensor>& a) { return scalar / (*a); }
}

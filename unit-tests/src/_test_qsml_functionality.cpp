#include <QuasarML.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

static int passed = 0;
static int failed = 0;

static void check(const char* name, bool condition) {
    if (condition) {
        std::cout << "  PASS " << name << "\n";
        passed++;
    } else {
        std::cout << "  FAIL " << name << "\n";
        failed++;
    }
}

static bool test_device_init() {
    std::cout << "[test_device_init]\n";
    
    qsml::u32 count = qsml::device_count();
    check("device_count > 0", count > 0);
    
    auto names = qsml::device_names();
    check("device_names not empty", !names.empty());
    
    auto& dev = qsml::device();
    check("device is_valid", dev.is_valid());
    
    return failed == 0;
}

static bool test_tensor_creation() {
    std::cout << "[test_tensor_creation]\n";
    
    auto a = qsml::zeros({4, 4});
    check("zeros shape", a->dim(0) == 4 && a->dim(1) == 4);
    check("zeros numel", a->numel() == 16);
    check("zeros dtype", a->dtype() == qsml::DataType::F32);
    
    auto b = qsml::ones({2, 3});
    std::vector<float> b_data(6);
    b->download(b_data.data(), b_data.size() * sizeof(float));
    check("ones values", std::abs(b_data[0] - 1.0f) < 1e-5f);
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto c = qsml::from_data(data.data(), {4}, qsml::DataType::F32);
    std::vector<float> c_out(4);
    c->download(c_out.data(), c_out.size() * sizeof(float));
    check("from_data values", std::abs(c_out[2] - 3.0f) < 1e-5f);
    
    return failed == 0;
}

static bool test_elementwise_ops() {
    std::cout << "[test_elementwise_ops]\n";
    
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_data = {10.0f, 20.0f, 30.0f, 40.0f};
    
    auto a = qsml::from_data(a_data.data(), {4}, qsml::DataType::F32);
    auto b = qsml::from_data(b_data.data(), {4}, qsml::DataType::F32);
    
    auto sum = qsml::add(a, b);
    std::vector<float> sum_out(4);
    sum->download(sum_out.data(), sum_out.size() * sizeof(float));
    check("add", std::abs(sum_out[0] - 11.0f) < 1e-5f && std::abs(sum_out[3] - 44.0f) < 1e-5f);
    
    auto diff = qsml::sub(b, a);
    std::vector<float> diff_out(4);
    diff->download(diff_out.data(), diff_out.size() * sizeof(float));
    check("sub", std::abs(diff_out[0] - 9.0f) < 1e-5f);
    
    auto prod = qsml::mul(a, b);
    std::vector<float> prod_out(4);
    prod->download(prod_out.data(), prod_out.size() * sizeof(float));
    check("mul", std::abs(prod_out[0] - 10.0f) < 1e-5f);
    
    auto quot = qsml::div(b, a);
    std::vector<float> quot_out(4);
    quot->download(quot_out.data(), quot_out.size() * sizeof(float));
    check("div", std::abs(quot_out[0] - 10.0f) < 1e-5f);
    
    auto neg_a = qsml::neg(a);
    std::vector<float> neg_out(4);
    neg_a->download(neg_out.data(), neg_out.size() * sizeof(float));
    check("neg", std::abs(neg_out[0] + 1.0f) < 1e-5f);
    
    auto scaled = qsml::mul_scalar(a, 2.0f);
    std::vector<float> scaled_out(4);
    scaled->download(scaled_out.data(), scaled_out.size() * sizeof(float));
    check("mul_scalar", std::abs(scaled_out[0] - 2.0f) < 1e-5f);
    
    return failed == 0;
}

static bool test_unary_ops() {
    std::cout << "[test_unary_ops]\n";
    
    std::vector<float> data = {0.25f, 1.0f, 4.0f, 9.0f};
    auto a = qsml::from_data(data.data(), {4}, qsml::DataType::F32);
    
    auto sqrt_a = qsml::sqrt(a);
    std::vector<float> sqrt_out(4);
    sqrt_a->download(sqrt_out.data(), sqrt_out.size() * sizeof(float));
    check("sqrt", std::abs(sqrt_out[0] - 0.5f) < 1e-4f && std::abs(sqrt_out[2] - 2.0f) < 1e-4f);
    
    return failed == 0;
}

static bool test_activation_functions() {
    std::cout << "[test_activation_functions]\n";
    
    std::vector<float> data = {-1.0f, 0.0f, 1.0f, 2.0f};
    auto a = qsml::from_data(data.data(), {4}, qsml::DataType::F32);
    
    auto relu_a = qsml::relu(a);
    std::vector<float> relu_out(4);
    relu_a->download(relu_out.data(), relu_out.size() * sizeof(float));
    check("relu", relu_out[0] == 0.0f && relu_out[2] == 1.0f);
    
    auto sig_a = qsml::sigmoid(a);
    std::vector<float> sig_out(4);
    sig_a->download(sig_out.data(), sig_out.size() * sizeof(float));
    check("sigmoid at 0", std::abs(sig_out[1] - 0.5f) < 1e-4f);
    
    return failed == 0;
}

static bool test_matmul() {
    std::cout << "[test_matmul]\n";
    
    auto a = qsml::ones({2, 3});
    auto b = qsml::ones({3, 4});
    
    auto c = qsml::matmul(a, b);
    check("matmul shape", c->dim(0) == 2 && c->dim(1) == 4);
    
    std::vector<float> c_out(8);
    c->download(c_out.data(), c_out.size() * sizeof(float));
    check("matmul values", std::abs(c_out[0] - 3.0f) < 1e-4f);
    
    return failed == 0;
}

static void run_all_tests() {
    test_device_init();
    test_tensor_creation();
    test_elementwise_ops();
    test_unary_ops();
    test_activation_functions();
    test_matmul();
}

int main() {
    std::cout << "QuasarML Functionality Tests\n";
    std::cout << "============================\n\n";
    
    qsml::init();
    run_all_tests();
    qsml::shutdown();
    
    std::cout << "\n============================\n";
    std::cout << "Passed: " << passed << ", Failed: " << failed << "\n";
    
    return failed == 0 ? 0 : 1;
}


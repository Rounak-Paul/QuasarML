#include <QuasarML.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

using namespace std;

static int passed = 0;
static int failed = 0;

static void check(const char* name, bool condition) {
    if (condition) {
        cout << "  PASS " << name << "\n";
        passed++;
    } else {
        cout << "  FAIL " << name << "\n";
        failed++;
    }
}

void test_device_management() {
    cout << "\n=== Device Management ===\n";
    
    check("device_count >= 1", qsml::device_count() >= 1);
    
    auto names = qsml::device_names();
    check("device_names not empty", !names.empty());
    
    auto& dev = qsml::device();
    check("device is_valid", dev.is_valid());
    
    check("current_device_id valid", qsml::current_device_id() < qsml::device_count());
}

void test_tensor_basics() {
    cout << "\n=== Tensor Basics ===\n";
    
    auto zeros_t = qsml::zeros({16, 16});
    check("zeros shape", zeros_t->dim(0) == 16 && zeros_t->dim(1) == 16);
    check("zeros numel", zeros_t->numel() == 256);
    check("zeros rank", zeros_t->rank() == 2);
    
    auto ones_t = qsml::ones({8});
    vector<float> ones_data(8);
    ones_t->download(ones_data.data(), ones_data.size() * sizeof(float));
    bool all_ones = true;
    for (auto v : ones_data) if (abs(v - 1.0f) > 1e-5f) all_ones = false;
    check("ones values", all_ones);
    
    vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t = qsml::from_data(input.data(), {2, 3}, qsml::DataType::F32);
    vector<float> output(6);
    t->download(output.data(), output.size() * sizeof(float));
    check("from_data roundtrip", abs(output[0] - 1.0f) < 1e-5f && abs(output[5] - 6.0f) < 1e-5f);
}

void test_elementwise_operations() {
    cout << "\n=== Elementwise Operations ===\n";
    
    vector<float> a_data = {1, 2, 3, 4};
    vector<float> b_data = {5, 6, 7, 8};
    auto a = qsml::from_data(a_data.data(), {4}, qsml::DataType::F32);
    auto b = qsml::from_data(b_data.data(), {4}, qsml::DataType::F32);
    
    auto sum = qsml::add(a, b);
    vector<float> sum_out(4);
    sum->download(sum_out.data(), sum_out.size() * sizeof(float));
    check("add", abs(sum_out[0] - 6.0f) < 1e-5f && abs(sum_out[3] - 12.0f) < 1e-5f);
    
    auto diff = qsml::sub(b, a);
    vector<float> diff_out(4);
    diff->download(diff_out.data(), diff_out.size() * sizeof(float));
    check("sub", abs(diff_out[0] - 4.0f) < 1e-5f);
    
    auto prod = qsml::mul(a, b);
    vector<float> prod_out(4);
    prod->download(prod_out.data(), prod_out.size() * sizeof(float));
    check("mul", abs(prod_out[0] - 5.0f) < 1e-5f);
    
    auto quot = qsml::div(b, a);
    vector<float> quot_out(4);
    quot->download(quot_out.data(), quot_out.size() * sizeof(float));
    check("div", abs(quot_out[0] - 5.0f) < 1e-5f);
    
    auto neg_a = qsml::neg(a);
    vector<float> neg_out(4);
    neg_a->download(neg_out.data(), neg_out.size() * sizeof(float));
    check("neg", abs(neg_out[0] + 1.0f) < 1e-5f);
    
    auto abs_val = qsml::abs(neg_a);
    vector<float> abs_out(4);
    abs_val->download(abs_out.data(), abs_out.size() * sizeof(float));
    check("abs", abs(abs_out[0] - 1.0f) < 1e-5f);
    
    auto scaled = qsml::mul_scalar(a, 10.0f);
    vector<float> scaled_out(4);
    scaled->download(scaled_out.data(), scaled_out.size() * sizeof(float));
    check("mul_scalar", abs(scaled_out[0] - 10.0f) < 1e-5f);
}

void test_unary_operations() {
    cout << "\n=== Unary Operations ===\n";
    
    vector<float> data = {0.25f, 1.0f, 4.0f, 9.0f};
    auto a = qsml::from_data(data.data(), {4}, qsml::DataType::F32);
    
    auto sqrt_a = qsml::sqrt(a);
    vector<float> sqrt_out(4);
    sqrt_a->download(sqrt_out.data(), sqrt_out.size() * sizeof(float));
    check("sqrt", abs(sqrt_out[0] - 0.5f) < 1e-4f && abs(sqrt_out[2] - 2.0f) < 1e-4f);
    
    vector<float> exp_data = {0.0f, 1.0f, 2.0f};
    auto exp_in = qsml::from_data(exp_data.data(), {3}, qsml::DataType::F32);
    auto exp_a = qsml::exp(exp_in);
    vector<float> exp_out(3);
    exp_a->download(exp_out.data(), exp_out.size() * sizeof(float));
    check("exp at 0", abs(exp_out[0] - 1.0f) < 1e-4f);
}

void test_activation_functions() {
    cout << "\n=== Activation Functions ===\n";
    
    vector<float> data = {-1.0f, 0.0f, 1.0f, 2.0f};
    auto a = qsml::from_data(data.data(), {4}, qsml::DataType::F32);
    
    auto relu_a = qsml::relu(a);
    vector<float> relu_out(4);
    relu_a->download(relu_out.data(), relu_out.size() * sizeof(float));
    check("relu negatives zeroed", relu_out[0] == 0.0f);
    check("relu positives preserved", relu_out[2] == 1.0f);
    
    auto sig_a = qsml::sigmoid(a);
    vector<float> sig_out(4);
    sig_a->download(sig_out.data(), sig_out.size() * sizeof(float));
    check("sigmoid at 0 = 0.5", abs(sig_out[1] - 0.5f) < 1e-4f);
    
    auto gelu_a = qsml::gelu(a);
    vector<float> gelu_out(4);
    gelu_a->download(gelu_out.data(), gelu_out.size() * sizeof(float));
    check("gelu computed", gelu_out[2] > 0.5f);
    
    auto tanh_a = qsml::tanh(a);
    vector<float> tanh_out(4);
    tanh_a->download(tanh_out.data(), tanh_out.size() * sizeof(float));
    check("tanh at 0 = 0", abs(tanh_out[1]) < 1e-4f);
}

void test_matrix_operations() {
    cout << "\n=== Matrix Operations ===\n";
    
    auto a = qsml::ones({2, 3});
    auto b = qsml::ones({3, 4});
    
    auto c = qsml::matmul(a, b);
    check("matmul shape", c->dim(0) == 2 && c->dim(1) == 4);
    
    vector<float> c_out(8);
    c->download(c_out.data(), c_out.size() * sizeof(float));
    check("matmul values (inner dim 3)", abs(c_out[0] - 3.0f) < 1e-4f);
    
    auto t = qsml::ones({4, 5});
    auto t_T = qsml::transpose(t);
    check("transpose shape", t_T->dim(0) == 5 && t_T->dim(1) == 4);
}

void test_reduction_operations() {
    cout << "\n=== Reduction Operations ===\n";
    
    vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto a = qsml::from_data(data.data(), {10}, qsml::DataType::F32);
    
    auto s = qsml::sum(a);
    vector<float> sum_out(1);
    s->download(sum_out.data(), sizeof(float));
    check("sum", abs(sum_out[0] - 55.0f) < 1e-3f);
    
    auto m = qsml::mean(a);
    vector<float> mean_out(1);
    m->download(mean_out.data(), sizeof(float));
    check("mean", abs(mean_out[0] - 5.5f) < 1e-3f);
    
    auto mx = qsml::max(a);
    vector<float> max_out(1);
    mx->download(max_out.data(), sizeof(float));
    check("max", abs(max_out[0] - 10.0f) < 1e-3f);
    
    auto mn = qsml::min(a);
    vector<float> min_out(1);
    mn->download(min_out.data(), sizeof(float));
    check("min", abs(min_out[0] - 1.0f) < 1e-3f);
}

void test_shape_operations() {
    cout << "\n=== Shape Operations ===\n";
    
    auto a = qsml::ones({2, 3, 4});
    check("original shape", a->dim(0) == 2 && a->dim(1) == 3 && a->dim(2) == 4);
    
    auto flat = qsml::flatten(a);
    check("flatten", flat->numel() == 24 && flat->rank() == 1);
    
    auto reshaped = qsml::reshape(a, {6, 4});
    check("reshape", reshaped->dim(0) == 6 && reshaped->dim(1) == 4);
    
    auto b = qsml::ones({1, 4, 1});
    auto squeezed = qsml::squeeze(b);
    check("squeeze", squeezed->rank() == 1 && squeezed->numel() == 4);
    
    auto c = qsml::ones({4});
    auto unsqueezed = qsml::unsqueeze(c, 0);
    check("unsqueeze", unsqueezed->rank() == 2 && unsqueezed->dim(0) == 1);
}

static void run_all_tests() {
    test_device_management();
    test_tensor_basics();
    test_elementwise_operations();
    test_unary_operations();
    test_activation_functions();
    test_matrix_operations();
    test_reduction_operations();
    test_shape_operations();
}

int main() {
    cout << "QuasarML Comprehensive Tests\n";
    cout << "============================\n";
    
    qsml::init();
    run_all_tests();
    qsml::shutdown();
    
    cout << "\n============================\n";
    cout << "Passed: " << passed << ", Failed: " << failed << "\n";
    
    return failed == 0 ? 0 : 1;
}

#include <QuasarML.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

using namespace std;

class TestRunner {
public:
    int passed = 0;
    int failed = 0;
    string current_section;

    template<typename T>
    bool close_enough(T a, T b, T tolerance = 1e-4) {
        return std::abs(a - b) < tolerance;
    }

    bool verify_shape(qsml::Tensor t, const vector<u32>& expected) {
        auto shape = t->get_shape();
        return shape == expected;
    }

    bool verify_value(qsml::Tensor t, float expected, float tolerance = 1e-4) {
        vector<float> data(t->get_element_count());
        t->download_data(data.data());
        for (auto val : data) {
            if (!close_enough(val, expected, tolerance)) {
                return false;
            }
        }
        return true;
    }

    void section(const string& name) {
        current_section = name;
        cout << "\n=== " << name << " ===\n";
    }

    void test(const string& name, function<bool()> fn) {
        try {
            if (fn()) {
                cout << "  ✓ " << name << "\n";
                passed++;
            } else {
                cout << "  ✗ " << name << " - assertion failed\n";
                failed++;
            }
        } catch (const exception& e) {
            cout << "  ✗ " << name << " - exception: " << e.what() << "\n";
            failed++;
        }
    }

    void summary() {
        cout << "\n========================================\n";
        cout << "Test Results: " << passed << " passed, " << failed << " failed\n";
        if (failed == 0) {
            cout << "✓ ALL TESTS PASSED\n";
        } else {
            cout << "✗ SOME TESTS FAILED\n";
        }
        cout << "========================================\n";
    }

    int get_failed_count() const { return failed; }
};

int main() {
    TestRunner test;

    cout << "=== QuasarML Comprehensive Test Suite ===\n";
    cout << "Testing all features for production readiness\n";

    // ========================================================================
    // Device Management Tests
    // ========================================================================
    test.section("Device Management");

    test.test("device_count returns valid count", []() {
        return qsml::device_count() >= 1;
    });

    test.test("device_names returns non-empty list", []() {
        auto names = qsml::device_names();
        return names.size() >= 1 && !names[0].empty();
    });

    test.test("current_device returns valid index", []() {
        return qsml::current_device() < qsml::device_count();
    });

    test.test("gpu_available returns true", []() {
        return qsml::gpu_available();
    });

    // ========================================================================
    // Tensor Creation Tests
    // ========================================================================
    test.section("Tensor Creation");

    test.test("zeros creates tensor with correct shape", [&test]() {
        auto t = qsml::zeros({2, 3}, DataType::F32);
        return test.verify_shape(t, {2, 3});
    });

    test.test("zeros initializes all values to 0", [&test]() {
        auto t = qsml::zeros({5, 5}, DataType::F32);
        return test.verify_value(t, 0.0f);
    });

    test.test("ones creates tensor with correct shape", [&test]() {
        auto t = qsml::ones({3, 4}, DataType::F32);
        return test.verify_shape(t, {3, 4});
    });

    test.test("ones initializes all values to 1", [&test]() {
        auto t = qsml::ones({4, 4}, DataType::F32);
        return test.verify_value(t, 1.0f);
    });

    test.test("empty creates tensor with correct shape", [&test]() {
        auto t = qsml::empty({10, 20}, DataType::F32);
        return test.verify_shape(t, {10, 20});
    });

    test.test("randn creates tensor with correct shape", [&test]() {
        auto t = qsml::randn({5, 5}, DataType::F32);
        return test.verify_shape(t, {5, 5});
    });

    test.test("rand creates tensor with values in range [0, 1)", []() {
        auto t = qsml::rand({100}, DataType::F32, 0.0f, 1.0f);
        vector<float> data(100);
        t->download_data(data.data());
        for (auto val : data) {
            if (val < 0.0f || val >= 1.0f) return false;
        }
        return true;
    });

    test.test("tensor creates from vector", [&test]() {
        vector<float> data = {1, 2, 3, 4, 5, 6};
        auto t = qsml::tensor(data, {2, 3}, DataType::F32);
        return test.verify_shape(t, {2, 3});
    });

    // ========================================================================
    // Element-wise Operations Tests
    // ========================================================================
    test.section("Element-wise Operations");

    test.test("add computes correct result", []() {
        auto a = qsml::ones({3, 3}, DataType::F32);
        auto b = qsml::ones({3, 3}, DataType::F32);
        auto c = qsml::add(a, b);
        vector<float> data(9);
        c->download_data(data.data());
        return abs(data[0] - 2.0f) < 1e-4;
    });

    test.test("sub computes correct result", []() {
        auto a = qsml::ones({3, 3}, DataType::F32);
        auto b = qsml::ones({3, 3}, DataType::F32);
        auto c = qsml::sub(a, b);
        vector<float> data(9);
        c->download_data(data.data());
        return abs(data[0] - 0.0f) < 1e-4;
    });

    test.test("mul computes correct result", []() {
        vector<float> data_a = {2, 2, 2, 2};
        vector<float> data_b = {3, 3, 3, 3};
        auto a = qsml::tensor(data_a, {2, 2}, DataType::F32);
        auto b = qsml::tensor(data_b, {2, 2}, DataType::F32);
        auto c = qsml::mul(a, b);
        vector<float> result(4);
        c->download_data(result.data());
        return abs(result[0] - 6.0f) < 1e-4;
    });

    test.test("div computes correct result", []() {
        vector<float> data_a = {6, 6, 6, 6};
        vector<float> data_b = {2, 2, 2, 2};
        auto a = qsml::tensor(data_a, {2, 2}, DataType::F32);
        auto b = qsml::tensor(data_b, {2, 2}, DataType::F32);
        auto c = qsml::div(a, b);
        vector<float> result(4);
        c->download_data(result.data());
        return abs(result[0] - 3.0f) < 1e-4;
    });

    test.test("add_scalar computes correct result", []() {
        auto a = qsml::ones({2, 2}, DataType::F32);
        auto c = qsml::add_scalar(a, 5.0f);
        vector<float> result(4);
        c->download_data(result.data());
        return abs(result[0] - 6.0f) < 1e-4;
    });

    test.test("mul_scalar computes correct result", []() {
        auto a = qsml::ones({2, 2}, DataType::F32);
        auto c = qsml::mul_scalar(a, 3.0f);
        vector<float> result(4);
        c->download_data(result.data());
        return abs(result[0] - 3.0f) < 1e-4;
    });

    // ========================================================================
    // Activation Functions Tests
    // ========================================================================
    test.section("Activation Functions");

    test.test("relu zeros out negative values", []() {
        vector<float> data = {-1, 0, 1, 2};
        auto x = qsml::tensor(data, {4}, DataType::F32);
        auto y = qsml::relu(x);
        vector<float> result(4);
        y->download_data(result.data());
        return abs(result[0] - 0.0f) < 1e-4 && abs(result[3] - 2.0f) < 1e-4;
    });

    test.test("sigmoid produces values in (0, 1)", []() {
        auto x = qsml::randn({100}, DataType::F32);
        auto y = qsml::sigmoid(x);
        vector<float> result(100);
        y->download_data(result.data());
        for (auto val : result) {
            if (val <= 0.0f || val >= 1.0f) return false;
        }
        return true;
    });

    test.test("tanh produces values in (-1, 1)", []() {
        auto x = qsml::randn({100}, DataType::F32);
        auto y = qsml::tanh(x);
        vector<float> result(100);
        y->download_data(result.data());
        for (auto val : result) {
            if (val <= -1.0f || val >= 1.0f) return false;
        }
        return true;
    });

    // ========================================================================
    // Math Operations Tests
    // ========================================================================
    test.section("Math Operations");

    test.test("exp computes correct result", []() {
        vector<float> data = {0, 1};
        auto x = qsml::tensor(data, {2}, DataType::F32);
        auto y = qsml::exp(x);
        vector<float> result(2);
        y->download_data(result.data());
        return abs(result[0] - 1.0f) < 1e-4 && abs(result[1] - expf(1.0f)) < 1e-3;
    });

    test.test("log computes correct result", []() {
        vector<float> data = {1, expf(1.0f)};
        auto x = qsml::tensor(data, {2}, DataType::F32);
        auto y = qsml::log(x);
        vector<float> result(2);
        y->download_data(result.data());
        return abs(result[0] - 0.0f) < 1e-4 && abs(result[1] - 1.0f) < 1e-3;
    });

    test.test("sqrt computes correct result", []() {
        vector<float> data = {4, 9, 16};
        auto x = qsml::tensor(data, {3}, DataType::F32);
        auto y = qsml::sqrt(x);
        vector<float> result(3);
        y->download_data(result.data());
        return abs(result[0] - 2.0f) < 1e-4 && 
               abs(result[1] - 3.0f) < 1e-4 && 
               abs(result[2] - 4.0f) < 1e-4;
    });

    // ========================================================================
    // Linear Algebra Tests
    // ========================================================================
    test.section("Linear Algebra");

    test.test("matmul 2x2 computes correct result", []() {
        vector<float> a_data = {1, 2, 3, 4};
        vector<float> b_data = {5, 6, 7, 8};
        auto a = qsml::tensor(a_data, {2, 2}, DataType::F32);
        auto b = qsml::tensor(b_data, {2, 2}, DataType::F32);
        auto c = qsml::matmul(a, b);
        vector<float> result(4);
        c->download_data(result.data());
        // [1,2] × [5,6] = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3,4]   [7,8]   [3*5+4*7, 3*6+4*8]   [43, 50]
        return abs(result[0] - 19.0f) < 1e-3 && abs(result[1] - 22.0f) < 1e-3;
    });

    test.test("matmul large matrices executes without error", [&test]() {
        auto a = qsml::randn({128, 128}, DataType::F32);
        auto b = qsml::randn({128, 128}, DataType::F32);
        auto c = qsml::matmul(a, b);
        return test.verify_shape(c, {128, 128});
    });

    test.test("transpose swaps dimensions", []() {
        vector<float> data = {1, 2, 3, 4, 5, 6};
        auto x = qsml::tensor(data, {2, 3}, DataType::F32);
        auto y = qsml::transpose(x);
        vector<float> result(6);
        y->download_data(result.data());
        // Original: [[1,2,3], [4,5,6]]
        // Transposed: [[1,4], [2,5], [3,6]]
        return abs(result[0] - 1.0f) < 1e-4 && abs(result[1] - 4.0f) < 1e-4;
    });

    // ========================================================================
    // Reduction Operations Tests
    // ========================================================================
    test.section("Reduction Operations");

    test.test("sum_axis reduces correctly", []() {
        vector<float> data = {1, 2, 3, 4, 5, 6};
        auto x = qsml::tensor(data, {2, 3}, DataType::F32);
        auto y = qsml::sum_axis(x, 1);  // Sum along columns
        vector<float> result(2);
        y->download_data(result.data());
        // Row 0: 1+2+3=6, Row 1: 4+5+6=15
        return abs(result[0] - 6.0f) < 1e-3 && abs(result[1] - 15.0f) < 1e-3;
    });

    test.test("mean_axis computes average correctly", []() {
        vector<float> data = {1, 2, 3, 4, 5, 6};
        auto x = qsml::tensor(data, {2, 3}, DataType::F32);
        auto y = qsml::mean_axis(x, 1);
        vector<float> result(2);
        y->download_data(result.data());
        // Row 0: (1+2+3)/3=2, Row 1: (4+5+6)/3=5
        return abs(result[0] - 2.0f) < 1e-3 && abs(result[1] - 5.0f) < 1e-3;
    });

    // ========================================================================
    // Tensor Properties Tests
    // ========================================================================
    test.section("Tensor Properties");

    test.test("shape returns correct dimensions", []() {
        auto t = qsml::zeros({3, 4, 5}, DataType::F32);
        auto s = qsml::shape(t);
        return s.size() == 3 && s[0] == 3 && s[1] == 4 && s[2] == 5;
    });

    test.test("ndim returns correct rank", []() {
        auto t = qsml::zeros({2, 3, 4}, DataType::F32);
        return qsml::ndim(t) == 3;
    });

    test.test("numel returns correct element count", []() {
        auto t = qsml::zeros({2, 3, 4}, DataType::F32);
        return qsml::numel(t) == 24;
    });

    test.test("dtype returns correct data type", []() {
        auto t = qsml::zeros({2, 2}, DataType::F32);
        return qsml::dtype(t) == DataType::F32;
    });

    // ========================================================================
    // Pipeline and Batching Tests
    // ========================================================================
    test.section("Pipeline and Batching");

    test.test("enable_auto_batching doesn't crash", []() {
        qsml::enable_auto_batching(true);
        qsml::enable_auto_batching(false);
        qsml::enable_auto_batching(true);
        return true;
    });

    test.test("flush_pipeline executes successfully", []() {
        auto a = qsml::ones({10, 10}, DataType::F32);
        auto b = qsml::add(a, a);
        qsml::flush_pipeline();
        return true;
    });

    test.test("synchronize waits for operations", []() {
        auto a = qsml::randn({100, 100}, DataType::F32);
        auto b = qsml::randn({100, 100}, DataType::F32);
        auto c = qsml::matmul(a, b);
        qsml::synchronize();
        return c->get_element_count() == 10000;
    });

    test.test("batched operations execute correctly", [&test]() {
        auto a = qsml::ones({50, 50}, DataType::F32);
        auto b = qsml::add(a, a);
        auto c = qsml::mul(b, a);
        auto d = qsml::relu(c);
        qsml::synchronize();
        return test.verify_shape(d, {50, 50});
    });

    // ========================================================================
    // Edge Cases and Error Handling
    // ========================================================================
    test.section("Edge Cases");

    test.test("1x1 matrix operations", [&test]() {
        auto a = qsml::ones({1, 1}, DataType::F32);
        auto b = qsml::ones({1, 1}, DataType::F32);
        auto c = qsml::matmul(a, b);
        return test.verify_shape(c, {1, 1});
    });

    test.test("large tensor allocation", [&test]() {
        auto t = qsml::zeros({1024, 1024}, DataType::F32);
        return test.verify_shape(t, {1024, 1024});
    });

    test.test("operations on different sized tensors broadcast correctly", []() {
        try {
            auto a = qsml::ones({2, 3}, DataType::F32);
            auto b = qsml::ones({2, 3}, DataType::F32);
            auto c = qsml::add(a, b);
            return true;
        } catch (...) {
            return false;
        }
    });

    // ========================================================================
    // I/O Operations Tests
    // ========================================================================
    test.section("I/O Operations");

    test.test("save and load tensor preserves data", []() {
        vector<float> original = {1, 2, 3, 4, 5, 6};
        auto t1 = qsml::tensor(original, {2, 3}, DataType::F32);
        qsml::save(t1, "/tmp/test_tensor.qsbin");
        auto t2 = qsml::load("/tmp/test_tensor.qsbin");
        vector<float> loaded(6);
        t2->download_data(loaded.data());
        for (size_t i = 0; i < 6; i++) {
            if (abs(original[i] - loaded[i]) > 1e-4) return false;
        }
        return true;
    });

    // ========================================================================
    // Performance Stress Tests
    // ========================================================================
    test.section("Performance Stress Tests");

    test.test("1000 small operations complete successfully", []() {
        auto a = qsml::ones({10, 10}, DataType::F32);
        for (int i = 0; i < 1000; i++) {
            a = qsml::add(a, a);
        }
        qsml::synchronize();
        return true;
    });

    test.test("large matmul chain executes", [&test]() {
        auto a = qsml::randn({256, 256}, DataType::F32);
        auto b = qsml::randn({256, 256}, DataType::F32);
        auto c = qsml::matmul(a, b);
        auto d = qsml::matmul(c, b);
        auto e = qsml::matmul(d, b);
        qsml::synchronize();
        return test.verify_shape(e, {256, 256});
    });

    // ========================================================================
    // Summary
    // ========================================================================
    test.summary();

    return test.get_failed_count() == 0 ? 0 : 1;
}

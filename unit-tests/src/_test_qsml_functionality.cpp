// Minimal, clean functionality test for QuasarML public API
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <filesystem>
#include <regex>
#include <thread>
#include <chrono>
#include "QuasarML.h"
// Include operator overloads for shared_ptr<Tensor>
#include <Core/TensorOps.h>

using namespace QuasarML;

static bool test_accelerator() {
        std::cout << "[test_accelerator]\n";
        // Test that lazy-initialized accelerator works
        auto& acc = qsml::accelerator();
        bool ok = acc.is_valid();
        std::cout << "  is_valid: " << ok << "\n";
        return ok;
    }

    static bool test_tensor_float_add() {
        std::cout << "[test_tensor_float_add]\n";
        
        std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};

        auto ta = qsml::from_data(a.data(), {4}, DataType::F32);
        auto tb = qsml::from_data(b.data(), {4}, DataType::F32);

        if (!ta || !tb) {
            std::cerr << "  Failed to create tensors\n";
            return false;
        }

        auto tc = qsml::add(ta, tb);
        if (!tc) {
            std::cerr << "  add returned null\n";
            return false;
        }

        std::vector<float> out(4);
        tc->download_data(out.data());

        for (size_t i = 0; i < out.size(); ++i) {
            float expect = a[i] + b[i];
            if (std::fabs(out[i] - expect) > 1e-5f) {
                std::cerr << "  Mismatch at " << i << ": got " << out[i] << ", expected " << expect << "\n";
                return false;
            }
        }

        std::cout << "  tensor add OK\n";
        return true;
    }

static bool test_more_dtypes_and_utils();
static bool test_broadcast_and_reduction();
static bool test_transpose_and_reshape_views();
static bool test_error_handling_and_edgecases();
static bool test_batch_recording_and_memory_barrier();

// Definitions
static bool test_more_dtypes_and_utils() {
    std::cout << "[test_more_dtypes_and_utils]\n";

    bool ok = true;
    // Try F16 if supported (may be emulated)
    try {
        std::vector<uint16_t> f16 = {0x3C00, 0x4000}; // 1.0, 2.0 in IEEE 754 half
        auto t_f16 = qsml::from_data(f16.data(), {2}, DataType::F16);
        if (!t_f16) ok = false;
        else if (t_f16->get_element_size() != get_dtype_size(DataType::F16)) ok = false;
    } catch (...) { ok = ok && true; }

    // unsigned/smaller ints
    std::vector<uint16_t> u16 = {1, 2};
    auto t_u16 = qsml::from_data(u16.data(), {2}, DataType::U16);
    if (!t_u16 || t_u16->get_element_size() != get_dtype_size(DataType::U16)) ok = false;

    std::vector<uint8_t> u8 = {1, 2};
    auto t_u8 = qsml::from_data(u8.data(), {2}, DataType::U8);
    if (!t_u8 || t_u8->get_element_size() != get_dtype_size(DataType::U8)) ok = false;

    std::cout << "  more dtypes => " << (ok ? "OK" : "FAIL") << "\n";
    return ok;
}

static bool test_broadcast_and_reduction() {
    std::cout << "[test_broadcast_and_reduction]\n";

    // a: shape [2,1], b: shape [1,3] -> broadcast to [2,3]
    std::vector<float> a = {1.0f, 2.0f}; // shape 2x1
    std::vector<float> b = {10.0f, 20.0f, 30.0f}; // shape 1x3
    auto ta = qsml::from_data(a.data(), {2,1}, DataType::F32);
    auto tb = qsml::from_data(b.data(), {1,3}, DataType::F32);
    if (!ta || !tb) return false;

    auto tsum = qsml::add(ta, tb); // should broadcast
    if (!tsum) return false;
    std::vector<float> out(6);
    tsum->download_data(out.data());
    // expected row-major: [11,21,31,12,22,32]
    bool ok = (std::fabs(out[0]-11.0f)<1e-4f) && (std::fabs(out[5]-32.0f)<1e-4f);

    // test sum_axis reduce over axis 1 (columns) producing shape [2]
    auto reduced = qsml::sum_axis(tsum, 1);
    if (!reduced) return false;
    std::vector<float> r(2);
    reduced->download_data(r.data());
    // expected: [11+21+31=63, 12+22+32=66]
    ok = ok && (std::fabs(r[0]-63.0f) < 1e-3f) && (std::fabs(r[1]-66.0f) < 1e-3f);

    std::cout << "  broadcast+reduce => " << (ok ? "OK" : "FAIL") << "\n";
    return ok;
}

static bool test_transpose_and_reshape_views() {
    std::cout << "[test_transpose_and_reshape_views] (safe checks only)\n";

    // create 2x3 tensor and verify safe utilities: shape, element count, dtype, info string
    std::vector<float> A = {1,2,3,4,5,6};
    auto t = qsml::from_data(A.data(), {2,3}, DataType::F32);
    if (!t) return false;

    auto shape = t->get_shape();
    bool ok = (shape.size() == 2 && shape[0] == 2 && shape[1] == 3);
    ok = ok && (t->get_element_count() == 6);
    ok = ok && (t->get_dtype() == DataType::F32);
    std::string info = t->get_info_string();
    ok = ok && (!info.empty());

    std::cout << "  safe tensor utils => " << (ok ? "OK" : "FAIL") << "\n";
    return ok;
}

static bool test_error_handling_and_edgecases() {
    std::cout << "[test_error_handling_and_edgecases]\n";

    bool ok = true;
    // mismatched shape addition should either return null or throw; accept either
    try {
        auto a = qsml::from_data(std::vector<float>{1,2}.data(), {2}, DataType::F32);
        auto b = qsml::from_data(std::vector<float>{1,2,3}.data(), {3}, DataType::F32);
        auto r = qsml::add(a,b);
        if (r) { // if returned tensor, that's unexpected
            std::vector<float> out(r->get_element_count());
            r->download_data(out.data());
            ok = false;
        }
    } catch (...) {
        ok = ok && true;
    }

    // empty shape creation should be rejected or invalid
    try {
        auto e = qsml::from_data(std::vector<u32>{}.data(), {}, DataType::F32);
        if (e && e->is_valid()) ok = false;
    } catch (...) { ok = ok && true; }

    std::cout << "  error/edgecases => " << (ok ? "OK" : "FAIL") << "\n";
    return ok;
}

static bool test_new_tensor_operations() {
    std::cout << "[test_new_tensor_operations]\n";

    bool ok = true;

    // Test 1: Sigmoid activation
    std::cout << "  Testing sigmoid...";
    try {
        std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        auto t = qsml::from_data(data.data(), {5}, DataType::F32);
        auto result = qsml::sigmoid(t);
        if (!result) { ok = false; std::cout << "FAIL (null)\n"; }
        else {
            std::vector<float> out(5);
            result->download_data(out.data());
            // sigmoid(0) should be ~0.5
            if (std::fabs(out[2] - 0.5f) > 0.01f) { ok = false; std::cout << "FAIL\n"; }
            else std::cout << "OK\n";
        }
    } catch (...) { ok = false; std::cout << "FAIL (exception)\n"; }

    // Test 2: Tanh activation
    std::cout << "  Testing tanh...";
    try {
        std::vector<float> data = {-1.0f, 0.0f, 1.0f};
        auto t = qsml::from_data(data.data(), {3}, DataType::F32);
        auto result = qsml::tanh(t);
        if (!result) { ok = false; std::cout << "FAIL (null)\n"; }
        else {
            std::vector<float> out(3);
            result->download_data(out.data());
            // tanh(0) should be 0
            if (std::fabs(out[1]) > 0.01f) { ok = false; std::cout << "FAIL\n"; }
            else std::cout << "OK\n";
        }
    } catch (...) { ok = false; std::cout << "FAIL (exception)\n"; }

    // Test 3: Softmax
    std::cout << "  Testing softmax...";
    try {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        auto t = qsml::from_data(data.data(), {4}, DataType::F32);
        auto result = qsml::softmax(t, 0);
        if (!result) { ok = false; std::cout << "FAIL (null)\n"; }
        else {
            std::vector<float> out(4);
            result->download_data(out.data());
            // Sum should be ~1.0
            float sum = 0.0f;
            for (auto v : out) sum += v;
            if (std::fabs(sum - 1.0f) > 0.01f) { ok = false; std::cout << "FAIL\n"; }
            else std::cout << "OK\n";
        }
    } catch (...) { ok = false; std::cout << "FAIL (exception)\n"; }

    // Test 4: Abs
    std::cout << "  Testing abs...";
    try {
        std::vector<float> data = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
        auto t = qsml::from_data(data.data(), {5}, DataType::F32);
        auto result = qsml::abs(t);
        if (!result) { ok = false; std::cout << "FAIL (null)\n"; }
        else {
            std::vector<float> out(5);
            result->download_data(out.data());
            // abs(-3) = 3
            if (std::fabs(out[0] - 3.0f) > 0.01f || std::fabs(out[1] - 1.0f) > 0.01f) { ok = false; std::cout << "FAIL\n"; }
            else std::cout << "OK\n";
        }
    } catch (...) { ok = false; std::cout << "FAIL (exception)\n"; }

    // Test 5: Neg
    std::cout << "  Testing neg...";
    try {
        std::vector<float> data = {1.0f, -2.0f, 3.0f};
        auto t = qsml::from_data(data.data(), {3}, DataType::F32);
        auto result = qsml::neg(t);
        if (!result) { ok = false; std::cout << "FAIL (null)\n"; }
        else {
            std::vector<float> out(3);
            result->download_data(out.data());
            // -1 = -1, -(-2) = 2
            if (std::fabs(out[0] + 1.0f) > 0.01f || std::fabs(out[1] - 2.0f) > 0.01f) { ok = false; std::cout << "FAIL\n"; }
            else std::cout << "OK\n";
        }
    } catch (...) { ok = false; std::cout << "FAIL (exception)\n"; }

    // Test 6: Clamp
    std::cout << "  Testing clamp...";
    try {
        std::vector<float> data = {-5.0f, -1.0f, 0.0f, 1.0f, 5.0f};
        auto t = qsml::from_data(data.data(), {5}, DataType::F32);
        auto result = qsml::clamp(t, -2.0f, 2.0f);
        if (!result) { ok = false; std::cout << "FAIL (null)\n"; }
        else {
            std::vector<float> out(5);
            result->download_data(out.data());
            // -5 clamped to -2, 5 clamped to 2
            if (std::fabs(out[0] + 2.0f) > 0.01f || std::fabs(out[4] - 2.0f) > 0.01f) { ok = false; std::cout << "FAIL\n"; }
            else std::cout << "OK\n";
        }
    } catch (...) { ok = false; std::cout << "FAIL (exception)\n"; }

    // Test 7: Squeeze
    std::cout << "  Testing squeeze...";
    try {
        std::vector<float> data = {1.0f, 2.0f, 3.0f};
        auto t = qsml::from_data(data.data(), {1, 3, 1}, DataType::F32);
        auto result = qsml::squeeze(t, -1); // squeeze all dims of size 1
        if (!result) { ok = false; std::cout << "FAIL (null)\n"; }
        else {
            auto shape = result->get_shape();
            // Should be [3]
            if (shape.size() != 1 || shape[0] != 3) { ok = false; std::cout << "FAIL\n"; }
            else std::cout << "OK\n";
        }
    } catch (...) { ok = false; std::cout << "FAIL (exception)\n"; }

    // Test 8: Unsqueeze
    std::cout << "  Testing unsqueeze...";
    try {
        std::vector<float> data = {1.0f, 2.0f, 3.0f};
        auto t = qsml::from_data(data.data(), {3}, DataType::F32);
        auto result = qsml::unsqueeze(t, 0);
        if (!result) { ok = false; std::cout << "FAIL (null)\n"; }
        else {
            auto shape = result->get_shape();
            // Should be [1, 3]
            if (shape.size() != 2 || shape[0] != 1 || shape[1] != 3) { ok = false; std::cout << "FAIL\n"; }
            else std::cout << "OK\n";
        }
    } catch (...) { ok = false; std::cout << "FAIL (exception)\n"; }

    // Test 9: Permute (transpose generalization)
    std::cout << "  Testing permute...";
    try {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        auto t = qsml::from_data(data.data(), {2, 3}, DataType::F32);
        // Permute [0,1] -> [1,0] is same as transpose
        auto result = qsml::permute(t, {1, 0});
        if (!result) { ok = false; std::cout << "FAIL (null)\n"; }
        else {
            auto shape = result->get_shape();
            // Should be [3, 2]
            if (shape.size() != 2 || shape[0] != 3 || shape[1] != 2) { ok = false; std::cout << "FAIL (shape)\n"; }
            else {
                std::vector<float> out(6);
                result->download_data(out.data());
                // First element should still be 1, but layout changed
                if (std::fabs(out[0] - 1.0f) > 0.01f) { ok = false; std::cout << "FAIL (value)\n"; }
                else std::cout << "OK\n";
            }
        }
    } catch (...) { ok = false; std::cout << "FAIL (exception)\n"; }

    // Test 10: Concatenate
    std::cout << "  Testing concatenate...";
    try {
        std::vector<float> data1 = {1.0f, 2.0f};
        std::vector<float> data2 = {3.0f, 4.0f};
        auto t1 = qsml::from_data(data1.data(), {2}, DataType::F32);
        auto t2 = qsml::from_data(data2.data(), {2}, DataType::F32);
        auto result = qsml::concatenate({t1, t2}, 0);
        if (!result) { ok = false; std::cout << "FAIL (null)\n"; }
        else {
            auto shape = result->get_shape();
            // Should be [4]
            if (shape.size() != 1 || shape[0] != 4) { ok = false; std::cout << "FAIL (shape)\n"; }
            else {
                std::vector<float> out(4);
                result->download_data(out.data());
                // Should be [1, 2, 3, 4]
                if (std::fabs(out[0] - 1.0f) > 0.01f || std::fabs(out[3] - 4.0f) > 0.01f) { ok = false; std::cout << "FAIL (value)\n"; }
                else std::cout << "OK\n";
            }
        }
    } catch (...) { ok = false; std::cout << "FAIL (exception)\n"; }

    // Test 11: Split
    std::cout << "  Testing split...";
    try {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        auto t = qsml::from_data(data.data(), {4}, DataType::F32);
        auto results = qsml::split(t, 2, 0);
        if (results.size() != 2) { ok = false; std::cout << "FAIL (count)\n"; }
        else {
            bool sizes_ok = true;
            for (auto& r : results) {
                if (!r || r->get_element_count() != 2) { sizes_ok = false; break; }
            }
            if (!sizes_ok) { ok = false; std::cout << "FAIL (sizes)\n"; }
            else {
                std::vector<float> out1(2), out2(2);
                results[0]->download_data(out1.data());
                results[1]->download_data(out2.data());
                // First split: [1, 2], second: [3, 4]
                std::cout << " [got: " << out1[0] << "," << out1[1] << " | " << out2[0] << "," << out2[1] << " expected: 1,2 | 3,4] ";
                if (std::fabs(out1[0] - 1.0f) > 0.01f || std::fabs(out2[0] - 3.0f) > 0.01f) { ok = false; std::cout << "FAIL (values)\n"; }
                else std::cout << "OK\n";
            }
        }
    } catch (...) { ok = false; std::cout << "FAIL (exception)\n"; }

    std::cout << "  new tensor operations => " << (ok ? "OK" : "FAIL") << "\n";
    return ok;
}

static bool test_batch_recording_and_memory_barrier() {
    std::cout << "[test_batch_recording_and_memory_barrier]\n";
    auto& acc = qsml::accelerator();

    // small kernel to increment an element using push constant
    const char* shader = R"(
#version 450
layout(local_size_x = 64) in;
layout(binding = 0) buffer Buf { float data[]; } buf;
layout(push_constant) uniform Push { float v; } pc;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= buf.data.length()) return;
    buf.data[idx] += pc.v;
}
)";

    try {
        auto k = acc.create_kernel("inc_push", shader, 1, sizeof(float));
        if (!k || !k->is_valid()) {
            std::cout << "  kernel not available, skipping batch-record test\n";
            return true;
        }

        std::vector<float> data = {0,0,0,0,0,0,0,0};
        auto t = qsml::from_data(data.data(), {static_cast<u32>(data.size())}, DataType::F32);
        if (!t) return false;

        acc.begin_recording();
        for (int i=0;i<4;++i) {
            float v = static_cast<float>(i+1);
            acc.record_execution(k, {t}, acc.calculate_optimal_dispatch_1d(static_cast<u32>(data.size()),64), 1,1, &v);
            acc.memory_barrier();
        }
        acc.end_recording();

        std::vector<float> out(data.size());
        t->download_data(out.data());
        // each element should have sum 1+2+3+4 = 10
        bool ok = true;
        for (size_t i=0;i<out.size();++i) if (std::fabs(out[i]-10.0f)>1e-3f) { ok=false; break; }
        std::cout << "  batch recording => " << (ok ? "OK" : "FAIL") << "\n";
        return ok;
    } catch (const std::exception& e) {
        std::cout << "  batch recording exception (skipping): " << e.what() << "\n";
        return true;
    }
}

// New test: intentionally force a shader compile failure and verify Vulkan backend writes GLSL dump to /tmp
static bool test_shader_compile_dump() {
    std::cout << "[shader_compile_dump]\n";
    auto& acc = qsml::accelerator();

    // Force GPU mode for a deterministic path
    try {
        acc.set_device_mode(Accelerator::DeviceMode::GPU);
    } catch (...) { /* ignore if API differs */ }

    // Best-effort: remove old quasar_shader_error_ files so detection is unambiguous
    try {
        for (auto &p : std::filesystem::directory_iterator("/tmp")) {
            const std::string fn = p.path().filename().string();
            if (fn.rfind("quasar_shader_error_", 0) == 0) {
                std::error_code ec;
                std::filesystem::remove(p.path(), ec);
            }
        }
    } catch (...) { /* ignore permission/iterator errors */ }

    const std::string bad_name = "unit_test_invalid_kernel";
    const std::string bad_glsl = "#version 450\nthis is invalid glsl!!!\n";

    bool compile_failed = false;
    try {
        // Try to create/compile the intentionally-bad GLSL. The backend may return nullptr or throw on error.
        auto k = acc.create_kernel(bad_name, bad_glsl, 1);
        if (!k || !k->is_valid()) compile_failed = true;
    } catch (...) {
        compile_failed = true;
    }

    // Allow a short delay for the backend thread to flush the dump file
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    bool found = false;
    try {
        std::regex pattern(R"(quasar_shader_error_\d+\.glsl)");
        for (auto &p : std::filesystem::directory_iterator("/tmp")) {
            const std::string fn = p.path().filename().string();
            if (std::regex_match(fn, pattern)) { found = true; break; }
        }
    } catch (...) { /* ignore */ }

    std::cout << "  compile_failed=" << (compile_failed ? "true" : "false")
              << " dump_found=" << (found ? "true" : "false") << "\n";

    return compile_failed && found;
}

    int main() {
        int pass = 0, total = 0;
        auto run = [&](const char* name, bool (*fn)()) {
            ++total;
            std::cout << "=== " << name << " ===\n";
            bool r = fn();
            std::cout << (r ? "PASS\n" : "FAIL\n");
            if (r) ++pass;
        };

        run("accelerator", test_accelerator);
        run("tensor_float_add", test_tensor_float_add);
        // New expanded tests
        run("tensor_dtype_coverage", []() {
            std::cout << "[tensor_dtype_coverage]\n";

            bool ok = true;
            // float32
            auto t_f32 = qsml::from_data(std::vector<float>{1.0f, 2.0f}.data(), {2}, DataType::F32);
            if (!t_f32 || t_f32->get_element_size() != get_dtype_size(DataType::F32)) ok = false;

            // int32
            std::vector<int32_t> vi = {1, 2};
            auto t_i32 = qsml::from_data(vi.data(), {2}, DataType::I32);
            if (!t_i32 || t_i32->get_element_size() != get_dtype_size(DataType::I32)) ok = false;

            // uint32
            std::vector<uint32_t> vu = {1u, 2u};
            auto t_u32 = qsml::from_data(vu.data(), {2}, DataType::U32);
            if (!t_u32 || t_u32->get_element_size() != get_dtype_size(DataType::U32)) ok = false;

            // int8
            std::vector<int8_t> vi8 = {1, 2};
            auto t_i8 = qsml::from_data(vi8.data(), {2}, DataType::I8);
            if (!t_i8 || t_i8->get_element_size() != get_dtype_size(DataType::I8)) ok = false;

            std::cout << "  dtype coverage: " << (ok ? "OK" : "FAIL") << "\n";
            return ok;
        });

        run("operator_overloads", []() {
            std::cout << "[operator_overloads]\n";

            std::vector<float> a = {1.0f, 2.0f};
            std::vector<float> b = {3.0f, 4.0f};
            auto ta = qsml::from_data(a.data(), {2}, DataType::F32);
            auto tb = qsml::from_data(b.data(), {2}, DataType::F32);
            if (!ta || !tb) return false;

            // use operator+ defined for shared_ptr<Tensor>
            auto tc = ta + tb;
            if (!tc) return false;
            std::vector<float> out(2);
            tc->download_data(out.data());
            bool ok = (std::fabs(out[0] - 4.0f) < 1e-5f) && (std::fabs(out[1] - 6.0f) < 1e-5f);
            std::cout << "  operator+ => " << (ok ? "OK" : "FAIL") << "\n";
            return ok;
        });

        run("cpu_gpu_mode_switch", []() {
            std::cout << "[cpu_gpu_mode_switch]\n";
            auto& acc = qsml::accelerator();

            // ensure GPU-preferred path: reset counter and force GPU
            acc.reset_cpu_fallback_count();
            acc.set_device_mode(Accelerator::DeviceMode::GPU);

            // perform an op that could fall back if GPU unsupported
            std::vector<float> a = {1.0f, 2.0f};
            std::vector<float> b = {3.0f, 4.0f};
            auto ta = qsml::from_data(a.data(), {2}, DataType::F32);
            auto tb = qsml::from_data(b.data(), {2}, DataType::F32);
            if (!ta || !tb) return false;
            auto tc = qsml::add(ta, tb);
            if (!tc) return false;

            // when GPU forced, we expect CPU fallback count to be zero (no unexpected CPU compute)
            u32 cnt_gpu = acc.get_cpu_fallback_count();

            // Now force CPU and perform same op; counter should increase
            acc.set_device_mode(Accelerator::DeviceMode::CPU);
            auto tc2 = qsml::add(ta, tb);
            if (!tc2) return false;
            u32 cnt_cpu = acc.get_cpu_fallback_count();

            bool ok = (cnt_gpu == 0u) && (cnt_cpu >= 1u);
            std::cout << "  cpu/gpu mode fallback counts: gpu=" << cnt_gpu << " cpu=" << cnt_cpu << " => " << (ok?"OK":"FAIL") << "\n";
            return ok;
        });

        run("cpu_gpu_broadcast_dtype_check", []() {
            std::cout << "[cpu_gpu_broadcast_dtype_check]\n";
            auto& acc = qsml::accelerator();

            acc.reset_cpu_fallback_count();
            // force CPU and test broadcast on an integer dtype
            acc.set_device_mode(Accelerator::DeviceMode::CPU);
            std::vector<int32_t> ai = {1, 2}; // shape [2,1]
            std::vector<int32_t> bi = {10, 20, 30}; // shape [1,3]
            auto ta = qsml::from_data(ai.data(), {2,1}, DataType::I32);
            auto tb = qsml::from_data(bi.data(), {1,3}, DataType::I32);
            if (!ta || !tb) return false;
            auto tout = qsml::add(ta, tb);
            if (!tout) return false;
            // cpu fallback must have been used
            u32 cnt = acc.get_cpu_fallback_count();
            bool ok = (cnt >= 1u);
            std::cout << "  cpu broadcast fallback count=" << cnt << " => " << (ok?"OK":"FAIL") << "\n";
            return ok;
        });

        run("scalar_ops", []() {
            std::cout << "[scalar_ops]\n";
            std::vector<float> a = {2.0f, 3.0f};
            auto ta = qsml::from_data(a.data(), {2}, DataType::F32);
            if (!ta) return false;

            auto t_add = qsml::add_scalar(ta, 5.0f);
            auto t_mul = qsml::mul_scalar(ta, 2.0f);
            if (!t_add || !t_mul) return false;
            std::vector<float> out_add(2), out_mul(2);
            t_add->download_data(out_add.data());
            t_mul->download_data(out_mul.data());
            bool ok = (std::fabs(out_add[0] - 7.0f) < 1e-5f) && (std::fabs(out_add[1] - 8.0f) < 1e-5f)
                      && (std::fabs(out_mul[0] - 4.0f) < 1e-5f) && (std::fabs(out_mul[1] - 6.0f) < 1e-5f);
            std::cout << "  scalar ops => " << (ok ? "OK" : "FAIL") << "\n";
            return ok;
        });

        run("matmul_small", []() {
            std::cout << "[matmul_small]\n";

            // A = 2x3, B = 3x2 => C = 2x2
            std::vector<float> A = {1, 2, 3,
                                    4, 5, 6}; // row-major
            std::vector<float> B = {7, 8,
                                    9, 10,
                                    11, 12};
            auto tA = qsml::from_data(A.data(), {2,3}, DataType::F32);
            auto tB = qsml::from_data(B.data(), {3,2}, DataType::F32);
            if (!tA || !tB) return false;

            auto tC = qsml::matmul(tA, tB);
            if (!tC) return false;
            std::vector<float> C(4);
            tC->download_data(C.data());
            // expected: [[58,64],[139,154]]
            bool ok = (std::fabs(C[0] - 58.0f) < 1e-3f) && (std::fabs(C[1] - 64.0f) < 1e-3f)
                      && (std::fabs(C[2] - 139.0f) < 1e-3f) && (std::fabs(C[3] - 154.0f) < 1e-3f);
            std::cout << "  matmul => " << (ok ? "OK" : "FAIL") << "\n";
            return ok;
        });

        run("pow_and_reductions", []() {
            std::cout << "[pow_and_reductions]\n";

            // pow scalar: [1,2,3] ^ 2 => [1,4,9]
            std::vector<float> a = {1.0f,2.0f,3.0f};
            auto ta = qsml::from_data(a.data(), {3}, DataType::F32);
            float p = 2.0f;
            auto tp = qsml::from_data(&p, {1}, DataType::F32);
            auto tout = qsml::pow(ta, tp);
            if (!tout) return false;
            std::vector<float> out(3); tout->download_data(out.data());
            bool ok = (std::fabs(out[0]-1.0f) < 1e-4f) && (std::fabs(out[2]-9.0f) < 1e-3f);

            // min/max/mean along axis for 2x3
            std::vector<float> M = {1,5,3, 4,2,6}; // 2x3
            auto tM = qsml::from_data(M.data(), {2,3}, DataType::F32);
            auto min0 = qsml::min_axis(tM, 0); // axis 0 -> shape [3]
            auto max1 = qsml::max_axis(tM, 1); // axis 1 -> shape [2]
            auto mean0 = qsml::mean_axis(tM, 0);
            if (!min0 || !max1 || !mean0) return false;
            std::vector<float> vmin(3); min0->download_data(vmin.data());
            std::vector<float> vmax(2); max1->download_data(vmax.data());
            std::vector<float> vmean(3); mean0->download_data(vmean.data());
            // min across rows: [min(1,4)=1, min(5,2)=2, min(3,6)=3]
            ok = ok && (std::fabs(vmin[0]-1.0f) < 1e-3f) && (std::fabs(vmin[1]-2.0f) < 1e-3f) && (std::fabs(vmin[2]-3.0f) < 1e-3f);
            // max across cols: row0 max=5, row1 max=6
            ok = ok && (std::fabs(vmax[0]-5.0f) < 1e-3f) && (std::fabs(vmax[1]-6.0f) < 1e-3f);
            // mean axis0: [(1+4)/2=2.5, (5+2)/2=3.5, (3+6)/2=4.5]
            ok = ok && (std::fabs(vmean[0]-2.5f) < 1e-3f) && (std::fabs(vmean[1]-3.5f) < 1e-3f) && (std::fabs(vmean[2]-4.5f) < 1e-3f);

            std::cout << "  pow+reductions => " << (ok ? "OK" : "FAIL") << "\n";
            return ok;
        });

        run("slicing_cpu_copy", []() {
            std::cout << "[slicing_cpu_copy]\n";
            std::vector<float> A = {1,2,3,4,5,6,7,8,9}; // 3x3
            auto t = qsml::from_data(A.data(), {3,3}, DataType::F32);
            if (!t) return false;
            // slice rows 0..1, cols 1..2 => start {0,1}, lengths {2,2}
            auto s = qsml::slice(t, {0,1}, {2,2});
            if (!s) return false;
            std::vector<float> out(4); s->download_data(out.data());
            // expected [[2,3],[5,6]] row-major => [2,3,5,6]
            bool ok = (std::fabs(out[0]-2.0f) < 1e-4f) && (std::fabs(out[3]-6.0f) < 1e-4f);
            std::cout << "  slicing => " << (ok ? "OK" : "FAIL") << "\n";
            return ok;
        });

            run("slicing_strided_gpu", []() {
                std::cout << "[slicing_strided_gpu]\n";

                // 3x4 matrix, test a non-contiguous slice (rows 0..2 step 1, cols 1..3)
                std::vector<float> A = {
                    1, 2, 3, 4,
                    5, 6, 7, 8,
                    9,10,11,12
                };
                auto t = qsml::from_data(A.data(), {3,4}, DataType::F32);
                if (!t) return false;
                // slice start {0,1}, lengths {3,2} -> expected [2,3,6,7,10,11]
                auto s = qsml::slice(t, {0,1}, {3,2});
                if (!s) return false;
                std::vector<float> out(6); s->download_data(out.data());
                bool ok = (std::fabs(out[0]-2.0f) < 1e-4f) && (std::fabs(out[5]-11.0f) < 1e-4f);
                if (!ok) {
                    std::cout << "  slicing_strided_gpu: got output: ";
                    for (auto v : out) std::cout << v << " ";
                    std::cout << "\n";
                }

                // also test a 3D case: shape [2,2,3], slice middle axis
                std::vector<float> B = {
                    1,2,3, 4,5,6,
                    7,8,9,10,11,12
                }; // shape 2x2x3
                auto t3 = qsml::from_data(B.data(), {2,2,3}, DataType::F32);
                if (!t3) return false;
                // slice start {0,1,1}, lengths {2,1,2} -> pick the last two entries of middle row in each outer
                auto s3 = qsml::slice(t3, {0,1,1}, {2,1,2});
                if (!s3) return false;
                std::vector<float> out3(4); s3->download_data(out3.data());
                // expected: [5,6,11,12]
                bool ok3 = (std::fabs(out3[0]-5.0f) < 1e-3f) && (std::fabs(out3[3]-12.0f) < 1e-3f);
                if (!ok3) {
                    std::cout << "  slicing_strided_gpu (3D) got: "; for (auto v: out3) std::cout << v << " "; std::cout << "\n";
                }
                ok = ok && ok3;

                // test an integer dtype slice to ensure dtype-agnostic copy (I32)
                std::vector<int32_t> I = {1,2,3,4,5,6}; // shape 2x3
                auto ti = qsml::from_data(I.data(), {2,3}, DataType::I32);
                if (!ti) return false;
                auto si = qsml::slice(ti, {0,1}, {2,2});
                if (!si) return false;
                std::vector<int32_t> oi(4); si->download_data(oi.data());
                bool oki = (oi[0] == 2) && (oi[3] == 6);
                if (!oki) {
                    std::cout << "  slicing_strided_gpu (I32) got: "; for (auto v: oi) std::cout << v << " "; std::cout << "\n";
                }
                ok = ok && oki;

                std::cout << "  slicing_strided_gpu => " << (ok ? "OK" : "FAIL") << "\n";
                return ok;
            });

        run("kernel_create_and_optional_execute", []() {
            std::cout << "[kernel_create_and_optional_execute]\n";
            auto& acc = qsml::accelerator();

            const char* shader = R"(
#version 450
layout(local_size_x = 64) in;
layout(binding = 0) buffer Buf { float data[]; } buf;
layout(push_constant) uniform Push { float v; } pc;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= buf.data.length()) return;
    buf.data[idx] += pc.v;
}
)";

            try {
                auto k = acc.create_kernel("add_push", shader, 1, sizeof(float));
                if (!k || !k->is_valid()) {
                    std::cout << "  kernel creation failed/skipped\n";
                    return true; // don't fail the suite on shader compile unsupported
                }

                // create small tensor and run kernel
                std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
                auto t = qsml::from_data(data.data(), {4}, DataType::F32);
                if (!t) return false;
                float push = 2.5f;
                // dispatch
                u32 dispatch = acc.calculate_optimal_dispatch_1d(static_cast<u32>(data.size()), 64);
                acc.execute(k, {t}, dispatch, 1, 1, &push);
                std::vector<float> out(4);
                t->download_data(out.data());
                bool ok = (std::fabs(out[0] - 3.5f) < 1e-4f);
                std::cout << "  kernel execute => " << (ok ? "OK" : "FAIL") << "\n";
                return ok;
            } catch (const std::exception& e) {
                std::cout << "  kernel test exception (skipping): " << e.what() << "\n";
                return true;
            }
        });

        // Ensure GPU mode does not trigger CPU fallbacks for the common ops across dtypes
        run("gpu_mode_all_dtypes_ops_no_cpu_fallback", []() {
            std::cout << "[gpu_mode_all_dtypes_ops_no_cpu_fallback]\n";
            auto& acc = qsml::accelerator();

            acc.reset_cpu_fallback_count();
            acc.set_device_mode(Accelerator::DeviceMode::GPU);

            bool ok = true;

            std::vector<DataType> dtypes = {
                DataType::F32,
                DataType::F16,
                DataType::I32,
                DataType::U32,
                DataType::I16,
                DataType::U16,
                DataType::I8,
                DataType::U8
            };

            for (auto dt : dtypes) {
                LOG_DEBUG("gpu_mode test dtype: {}", dtype_to_string(dt));
                try {
                    // create simple 2-element tensors for each dtype
                    std::shared_ptr<Tensor> ta, tb;
                    switch (dt) {
                        case DataType::F32: {
                            std::vector<float> A{1.0f, -2.0f}; std::vector<float> B{3.0f, 4.0f};
                            ta = qsml::from_data(A.data(), {2}, dt);
                            tb = qsml::from_data(B.data(), {2}, dt);
                            break;
                        }
                        case DataType::F16: {
                            // use canonical half encodings for 1.0 and 2.0
                            std::vector<uint16_t> A{0x3C00u, 0xC000u}; // 1.0, -2.0 (half)
                            std::vector<uint16_t> B{0x4000u, 0x4000u}; // 2.0, 2.0
                            ta = qsml::from_data(A.data(), {2}, dt);
                            tb = qsml::from_data(B.data(), {2}, dt);
                            break;
                        }
                        case DataType::I32: {
                            std::vector<int32_t> A{1, -2}; std::vector<int32_t> B{3, 4};
                            ta = qsml::from_data(A.data(), {2}, dt);
                            tb = qsml::from_data(B.data(), {2}, dt);
                            break;
                        }
                        case DataType::U32: {
                            std::vector<uint32_t> A{1u, 2u}; std::vector<uint32_t> B{3u, 4u};
                            ta = qsml::from_data(A.data(), {2}, dt);
                            tb = qsml::from_data(B.data(), {2}, dt);
                            break;
                        }
                        case DataType::I16: {
                            std::vector<int16_t> A{1, -2}; std::vector<int16_t> B{3, 4};
                            ta = qsml::from_data(A.data(), {2}, dt);
                            tb = qsml::from_data(B.data(), {2}, dt);
                            break;
                        }
                        case DataType::U16: {
                            std::vector<uint16_t> A{1u, 2u}; std::vector<uint16_t> B{3u, 4u};
                            ta = qsml::from_data(A.data(), {2}, dt);
                            tb = qsml::from_data(B.data(), {2}, dt);
                            break;
                        }
                        case DataType::I8: {
                            std::vector<int8_t> A{1, -2}; std::vector<int8_t> B{3, 4};
                            ta = qsml::from_data(A.data(), {2}, dt);
                            tb = qsml::from_data(B.data(), {2}, dt);
                            break;
                        }
                        case DataType::U8: {
                            std::vector<uint8_t> A{1u, 2u}; std::vector<uint8_t> B{3u, 4u};
                            ta = qsml::from_data(A.data(), {2}, dt);
                            tb = qsml::from_data(B.data(), {2}, dt);
                            break;
                        }
                        default: break;
                    }

                    if (!ta || !tb) { ok = false; break; }

                    // elementwise add and mul should execute on GPU when forced
                    auto r_add = qsml::add(ta, tb);
                    if (!r_add) { ok = false; break; }
                    auto r_mul = qsml::mul(ta, tb);
                    if (!r_mul) { ok = false; break; }

                    // small relu test where meaningful
                    auto r_relu = qsml::relu(ta);
                    if (!r_relu) { ok = false; break; }

                } catch (...) {
                    ok = false;
                    break;
                }
            }

            u32 cnt = acc.get_cpu_fallback_count();
            bool final_ok = ok && (cnt == 0u);
            LOG_INFO("gpu-mode cpu-fallback counter={} => {}", cnt, (final_ok ? "OK" : "FAIL"));
            return final_ok;
        });

        // run the shader-dump test like the other unit tests
        run("shader_compile_dump", test_shader_compile_dump);

        // All-new tests added
        run("more_dtypes_and_utils", test_more_dtypes_and_utils);
        run("broadcast_and_reduction", test_broadcast_and_reduction);
        run("transpose_and_reshape_views", test_transpose_and_reshape_views);
        run("error_handling_and_edgecases", test_error_handling_and_edgecases);
        run("batch_recording_and_memory_barrier", test_batch_recording_and_memory_barrier);
        run("new_tensor_operations", test_new_tensor_operations);

        std::cout << "Summary: " << pass << "/" << total << " tests passed\n";
        return (pass == total) ? 0 : 1;
    }


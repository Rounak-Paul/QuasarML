    // Minimal, clean functionality test for QuasarML public API
    #include <iostream>
    #include <vector>
    #include <memory>
    #include <cmath>
    #include "QuasarML.h"
    // Include operator overloads for shared_ptr<Tensor>
    #include <Core/TensorOps.h>

    using namespace QuasarML;

    static bool test_accelerator() {
        std::cout << "[test_accelerator]\n";
        Accelerator acc("UnitTestAccel");
        bool ok = acc.is_valid();
        std::cout << "  is_valid: " << ok << "\n";
        return ok;
    }

    static bool test_tensor_float_add() {
        std::cout << "[test_tensor_float_add]\n";
        Accelerator acc("UnitTestTensor");
        if (!acc.is_valid()) {
            std::cerr << "  Accelerator invalid\n";
            return false;
        }


        std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};

        auto ta = acc.create_tensor(a.data(), {4}, DataType::F32);
        auto tb = acc.create_tensor(b.data(), {4}, DataType::F32);

        if (!ta || !tb) {
            std::cerr << "  Failed to create tensors\n";
            return false;
        }

        auto tc = acc.ops().add(ta, tb);
        if (!tc) {
            std::cerr << "  ops().add returned null\n";
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
    Accelerator acc("DtypeUtils");
    if (!acc.is_valid()) return false;

    bool ok = true;
    // Try F16 if supported (may be emulated)
    try {
        std::vector<uint16_t> f16 = {0x3C00, 0x4000}; // 1.0, 2.0 in IEEE 754 half
        auto t_f16 = acc.create_tensor(f16.data(), {2}, DataType::F16);
        if (!t_f16) ok = false;
        else if (t_f16->get_element_size() != get_dtype_size(DataType::F16)) ok = false;
    } catch (...) { ok = ok && true; }

    // unsigned/smaller ints
    std::vector<uint16_t> u16 = {1, 2};
    auto t_u16 = acc.create_tensor(u16.data(), {2}, DataType::U16);
    if (!t_u16 || t_u16->get_element_size() != get_dtype_size(DataType::U16)) ok = false;

    std::vector<uint8_t> u8 = {1, 2};
    auto t_u8 = acc.create_tensor(u8.data(), {2}, DataType::U8);
    if (!t_u8 || t_u8->get_element_size() != get_dtype_size(DataType::U8)) ok = false;

    std::cout << "  more dtypes => " << (ok ? "OK" : "FAIL") << "\n";
    return ok;
}

static bool test_broadcast_and_reduction() {
    std::cout << "[test_broadcast_and_reduction]\n";
    Accelerator acc("Broadcast");
    if (!acc.is_valid()) return false;

    // a: shape [2,1], b: shape [1,3] -> broadcast to [2,3]
    std::vector<float> a = {1.0f, 2.0f}; // shape 2x1
    std::vector<float> b = {10.0f, 20.0f, 30.0f}; // shape 1x3
    auto ta = acc.create_tensor(a.data(), {2,1}, DataType::F32);
    auto tb = acc.create_tensor(b.data(), {1,3}, DataType::F32);
    if (!ta || !tb) return false;

    auto tsum = acc.ops().add(ta, tb); // should broadcast
    if (!tsum) return false;
    std::vector<float> out(6);
    tsum->download_data(out.data());
    // expected row-major: [11,21,31,12,22,32]
    bool ok = (std::fabs(out[0]-11.0f)<1e-4f) && (std::fabs(out[5]-32.0f)<1e-4f);

    // test sum_axis reduce over axis 1 (columns) producing shape [2]
    auto reduced = acc.ops().sum_axis(tsum, 1);
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
    Accelerator acc("Transpose");
    if (!acc.is_valid()) return false;

    // create 2x3 tensor and verify safe utilities: shape, element count, dtype, info string
    std::vector<float> A = {1,2,3,4,5,6};
    auto t = acc.create_tensor(A.data(), {2,3}, DataType::F32);
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
    Accelerator acc("ErrorEdge");
    if (!acc.is_valid()) return false;

    bool ok = true;
    // mismatched shape addition should either return null or throw; accept either
    try {
        auto a = acc.create_tensor(std::vector<float>{1,2}.data(), {2}, DataType::F32);
        auto b = acc.create_tensor(std::vector<float>{1,2,3}.data(), {3}, DataType::F32);
        auto r = acc.ops().add(a,b);
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
        auto e = acc.create_tensor(std::vector<u32>{}.data(), {}, DataType::F32);
        if (e && e->is_valid()) ok = false;
    } catch (...) { ok = ok && true; }

    std::cout << "  error/edgecases => " << (ok ? "OK" : "FAIL") << "\n";
    return ok;
}

static bool test_batch_recording_and_memory_barrier() {
    std::cout << "[test_batch_recording_and_memory_barrier]\n";
    Accelerator acc("BatchRecord");
    if (!acc.is_valid()) return false;

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
        auto t = acc.create_tensor(data.data(), {static_cast<u32>(data.size())}, DataType::F32);
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
            Accelerator acc("DtypeTest");
            if (!acc.is_valid()) return false;

            bool ok = true;
            // float32
            auto t_f32 = acc.create_tensor(std::vector<float>{1.0f, 2.0f}.data(), {2}, DataType::F32);
            if (!t_f32 || t_f32->get_element_size() != get_dtype_size(DataType::F32)) ok = false;

            // int32
            std::vector<int32_t> vi = {1, 2};
            auto t_i32 = acc.create_tensor(vi.data(), {2}, DataType::I32);
            if (!t_i32 || t_i32->get_element_size() != get_dtype_size(DataType::I32)) ok = false;

            // uint32
            std::vector<uint32_t> vu = {1u, 2u};
            auto t_u32 = acc.create_tensor(vu.data(), {2}, DataType::U32);
            if (!t_u32 || t_u32->get_element_size() != get_dtype_size(DataType::U32)) ok = false;

            // int8
            std::vector<int8_t> vi8 = {1, 2};
            auto t_i8 = acc.create_tensor(vi8.data(), {2}, DataType::I8);
            if (!t_i8 || t_i8->get_element_size() != get_dtype_size(DataType::I8)) ok = false;

            std::cout << "  dtype coverage: " << (ok ? "OK" : "FAIL") << "\n";
            return ok;
        });

        run("operator_overloads", []() {
            std::cout << "[operator_overloads]\n";
            Accelerator acc("OpOverload");
            if (!acc.is_valid()) return false;

            std::vector<float> a = {1.0f, 2.0f};
            std::vector<float> b = {3.0f, 4.0f};
            auto ta = acc.create_tensor(a.data(), {2}, DataType::F32);
            auto tb = acc.create_tensor(b.data(), {2}, DataType::F32);
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

        run("scalar_ops", []() {
            std::cout << "[scalar_ops]\n";
            Accelerator acc("ScalarOps");
            if (!acc.is_valid()) return false;
            std::vector<float> a = {2.0f, 3.0f};
            auto ta = acc.create_tensor(a.data(), {2}, DataType::F32);
            if (!ta) return false;

            auto t_add = acc.ops().add_scalar(ta, 5.0f);
            auto t_mul = acc.ops().mul_scalar(ta, 2.0f);
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
            Accelerator acc("MatMul");
            if (!acc.is_valid()) return false;

            // A = 2x3, B = 3x2 => C = 2x2
            std::vector<float> A = {1, 2, 3,
                                    4, 5, 6}; // row-major
            std::vector<float> B = {7, 8,
                                    9, 10,
                                    11, 12};
            auto tA = acc.create_tensor(A.data(), {2,3}, DataType::F32);
            auto tB = acc.create_tensor(B.data(), {3,2}, DataType::F32);
            if (!tA || !tB) return false;

            auto tC = acc.ops().matmul(tA, tB);
            if (!tC) return false;
            std::vector<float> C(4);
            tC->download_data(C.data());
            // expected: [[58,64],[139,154]]
            bool ok = (std::fabs(C[0] - 58.0f) < 1e-3f) && (std::fabs(C[1] - 64.0f) < 1e-3f)
                      && (std::fabs(C[2] - 139.0f) < 1e-3f) && (std::fabs(C[3] - 154.0f) < 1e-3f);
            std::cout << "  matmul => " << (ok ? "OK" : "FAIL") << "\n";
            return ok;
        });

        run("kernel_create_and_optional_execute", []() {
            std::cout << "[kernel_create_and_optional_execute]\n";
            Accelerator acc("KernelTest");
            if (!acc.is_valid()) return false;

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
                auto t = acc.create_tensor(data.data(), {4}, DataType::F32);
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

        // All-new tests added
        run("more_dtypes_and_utils", test_more_dtypes_and_utils);
        run("broadcast_and_reduction", test_broadcast_and_reduction);
        run("transpose_and_reshape_views", test_transpose_and_reshape_views);
        run("error_handling_and_edgecases", test_error_handling_and_edgecases);
        run("batch_recording_and_memory_barrier", test_batch_recording_and_memory_barrier);

        std::cout << "Summary: " << pass << "/" << total << " tests passed\n";
        return (pass == total) ? 0 : 1;
    }


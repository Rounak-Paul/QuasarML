// stable_benchmark_batched.cpp
// Improved stable benchmark that reduces kernel creation overhead by warming up
// and batching many dispatches per measured iteration.
//
// Usage: compile and run with your existing QuasarML API.

#include <QuasarML.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <numeric>
#include <string>
#include <functional>

using namespace QuasarML;
using Clock = std::chrono::high_resolution_clock;
using Sec = double;

static void fill_random(std::vector<float>& v) {
    static std::mt19937 rng(1234567);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto &x : v) x = d(rng);
}

static void stats_from_seconds(const std::vector<Sec>& secs, Sec &mean, Sec &stddev) {
    if (secs.empty()) { mean = stddev = 0.0; return; }
    mean = std::accumulate(secs.begin(), secs.end(), 0.0) / secs.size();
    if (secs.size() < 2) { stddev = 0.0; return; }
    double var = 0.0;
    for (auto s : secs) var += (s - mean) * (s - mean);
    var /= (secs.size() - 1);
    stddev = std::sqrt(var);
}

static inline double bytes_to_gb(uint64_t bytes) {
    return double(bytes) / (1024.0 * 1024.0 * 1024.0);
}

static inline void print_hdr(const std::string &s) {
    std::cout << "\n" << std::string(80, '=') << "\n  " << s << "\n" << std::string(80, '=') << "\n";
}

int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(3);

    try {
        print_hdr("QuasarML - Stable Benchmark (Batched / Warmup)");

        Accelerator accel("QuasarML_Benchmark");
        if (!accel.is_valid()) {
            std::cerr << "[FATAL] Accelerator not valid\n";
            return 2;
        }

        auto limits = accel.get_device_limits();
        auto [used_before, total_mem] = accel.get_memory_usage();
        std::cout << "[INFO] Device memory used: " << used_before << " / " << total_mem << " bytes\n";

        // ---------- PARAMETERS ----------
        const u32 MEM_ELEMENTS      = 1u << 22;    // ~4,194,304 floats (~16 MiB)
        const int MEM_ITERS         = 200;         // fewer iterations but batched/averaged
        const int MEM_WARMUP        = 8;           // warmup uploads/downloads

        const u32 ELEM_ELEMENTS     = 1u << 20;    // ~1,048,576 floats (~4 MiB)
        const int ELEM_ITERS        = 400;         // measured iterations (each measures a batch)
        const int ELEM_WARMUP       = 8;
        const int ELEM_BATCH        = 8;           // perform this many ops before a single sync

        const u32 MAT_M             = 512;
        const u32 MAT_K             = 512;
        const u32 MAT_N             = 512;
        const int MAT_ITERS         = 200;
        const int MAT_WARMUP        = 4;
        const int MAT_BATCH         = 4;

        const u32 SAMPLE_COUNT      = 16;          // bytes to download as sync per timed iteration

        // ---------- MEMORY BANDWIDTH ----------
        print_hdr("Memory Bandwidth (Host <-> Device) - averaged & warmed");

        {
            u64 bytes = uint64_t(MEM_ELEMENTS) * sizeof(float);
            std::vector<float> host_src(MEM_ELEMENTS);
            std::vector<float> host_dst(MEM_ELEMENTS);
            fill_random(host_src);

            // create device-only tensor once (reuse for all uploads/downloads)
            auto dev = accel.create_tensor({MEM_ELEMENTS}, DataType::F32);
            if (!dev) {
                std::cerr << "[ERROR] create_tensor failed for memory test\n";
                return 3;
            }

            // Warmup uploads/downloads so any kernel / staging allocations happen early
            for (int i = 0; i < MEM_WARMUP; ++i) {
                dev->upload_data(host_src.data());
                dev->download_data(host_dst.data());
            }

            // Timed uploads
            std::vector<Sec> up_secs; up_secs.reserve(MEM_ITERS);
            for (int i = 0; i < MEM_ITERS; ++i) {
                auto t0 = Clock::now();
                dev->upload_data(host_src.data());
                auto t1 = Clock::now();
                up_secs.push_back(std::chrono::duration<Sec>(t1 - t0).count());
            }
            Sec up_mean, up_std; stats_from_seconds(up_secs, up_mean, up_std);
            double up_gbs = bytes_to_gb(bytes) / up_mean;
            std::cout << "[PERF] Upload (Host->Device)    : " << up_gbs << " GB/s"
                      << "  (mean time = " << up_mean*1000.0 << " ms, std = " << up_std*1000.0 << " ms)\n";

            // Timed downloads
            std::vector<Sec> down_secs; down_secs.reserve(MEM_ITERS);
            for (int i = 0; i < MEM_ITERS; ++i) {
                auto t0 = Clock::now();
                dev->download_data(host_dst.data());
                auto t1 = Clock::now();
                down_secs.push_back(std::chrono::duration<Sec>(t1 - t0).count());
            }
            Sec down_mean, down_std; stats_from_seconds(down_secs, down_mean, down_std);
            double down_gbs = bytes_to_gb(bytes) / down_mean;
            std::cout << "[PERF] Download (Device->Host)  : " << down_gbs << " GB/s"
                      << "  (mean time = " << down_mean*1000.0 << " ms, std = " << down_std*1000.0 << " ms)\n";

            // quick spot-check sync (upload -> download small sample)
            dev->upload_data(host_src.data());
            std::vector<float> sample(SAMPLE_COUNT);
            dev->download_data(sample.data(), sizeof(float) * SAMPLE_COUNT, 0);
            bool ok = true;
            for (u32 i = 0; i < std::min<u32>(SAMPLE_COUNT, MEM_ELEMENTS); ++i) {
                if (std::abs(sample[i] - host_src[i]) > 1e-6f) { ok = false; break; }
            }
            std::cout << "[INFO] Memory spot-check: " << (ok ? "OK" : "MISMATCH") << "\n";
        }

        // ---------- ELEMENTWISE OPS (batched) ----------
        print_hdr("Elementwise Ops (GigaOps/s) - warmed & batched");

        {
            std::vector<float> hA(ELEM_ELEMENTS), hB(ELEM_ELEMENTS);
            fill_random(hA);
            fill_random(hB);

            auto A = accel.create_tensor(hA.data(), {ELEM_ELEMENTS}, DataType::F32);
            auto B = accel.create_tensor(hB.data(), {ELEM_ELEMENTS}, DataType::F32);
            if (!A || !B) {
                std::cerr << "[ERROR] create_tensor failed for elementwise test\n";
                return 4;
            }

            // single small sample used to force completion/sync
            std::vector<float> sample(std::min<u32>(SAMPLE_COUNT, ELEM_ELEMENTS));

            // Warmup: run each op a few times so kernels are created once
            for (int i = 0; i < ELEM_WARMUP; ++i) {
                auto R = accel.ops().add(A, B); R->download_data(sample.data(), sizeof(float) * sample.size(), 0);
                auto R2 = accel.ops().mul(A, B); R2->download_data(sample.data(), sizeof(float) * sample.size(), 0);
                auto R3 = accel.ops().add_scalar(A, 1.2345f); R3->download_data(sample.data(), sizeof(float) * sample.size(), 0);
                auto R4 = accel.ops().mul_scalar(A, 2.0f); R4->download_data(sample.data(), sizeof(float) * sample.size(), 0);
            }

            auto run_batched = [&](const std::string &name,
                                   std::function<std::shared_ptr<Tensor>()> op,
                                   int iters,
                                   int batch_size,
                                   double ops_per_element = 1.0) {
                std::vector<Sec> iter_secs; iter_secs.reserve(iters);
                for (int i = 0; i < iters; ++i) {
                    auto t0 = Clock::now();
                    // dispatch batch_size operations back-to-back; force single sync at end
                    for (int b = 0; b < batch_size; ++b) {
                        auto R = op(); // dispatch
                        (void)R; // don't download here
                    }
                    // single small download to ensure all batched dispatches complete
                    // NOTE: some backends may implicitly sync earlier; this still reduces syncs.
                    // download from a fresh op result (guarantees work finished)
                    auto Rsync = op();
                    Rsync->download_data(sample.data(), sizeof(float) * sample.size(), 0);
                    auto t1 = Clock::now();
                    iter_secs.push_back(std::chrono::duration<Sec>(t1 - t0).count());
                }
                Sec mean_s, std_s; stats_from_seconds(iter_secs, mean_s, std_s);
                // each measured iteration executed batch_size dispatches; compute per-dispatch GOp/s
                double gops_mean = double(ELEM_ELEMENTS) * ops_per_element * double(batch_size) / (mean_s * 1e9);
                double gops_std  = (std_s > 0.0) ? gops_mean * (std_s / mean_s) : 0.0;
                std::cout << "[PERF] " << name << " : " << gops_mean << " GOPS"
                          << " (mean time per iter = " << mean_s*1000.0 << " ms, std = " << std_s*1000.0 << " ms)"
                          << "  [batch=" << batch_size << "]\n";
            };

            run_batched("Elementwise Add",   [&]() { return accel.ops().add(A, B); }, ELEM_ITERS, ELEM_BATCH, 1.0);
            run_batched("Elementwise Mul",   [&]() { return accel.ops().mul(A, B); }, ELEM_ITERS, ELEM_BATCH, 1.0);
            run_batched("Add Scalar",        [&]() { return accel.ops().add_scalar(A, 1.2345f); }, ELEM_ITERS, ELEM_BATCH, 1.0);
            run_batched("Mul Scalar",        [&]() { return accel.ops().mul_scalar(A, 2.0f); }, ELEM_ITERS, ELEM_BATCH, 1.0);
        }

        // ---------- MATMUL (batched) ----------
        print_hdr("Matrix Multiply (GFLOPS) - warmed & batched");

        {
            std::vector<float> hA(MAT_M * MAT_K, 1.0f);
            std::vector<float> hB(MAT_K * MAT_N, 1.0f);

            auto A = accel.create_tensor(hA.data(), {MAT_M, MAT_K}, DataType::F32);
            auto B = accel.create_tensor(hB.data(), {MAT_K, MAT_N}, DataType::F32);
            if (!A || !B) {
                std::cerr << "[ERROR] create_tensor failed for matmul test\n";
                return 5;
            }

            const u32 sample_count = std::min<u32>(16, MAT_M * MAT_N);
            std::vector<float> sample(sample_count);

            // Warmup matmul a few times
            for (int i = 0; i < MAT_WARMUP; ++i) {
                auto C = accel.ops().matmul(A, B);
                C->download_data(sample.data(), sizeof(float) * sample_count, 0);
            }

            std::vector<Sec> iter_secs; iter_secs.reserve(MAT_ITERS);
            for (int i = 0; i < MAT_ITERS; ++i) {
                auto t0 = Clock::now();
                // batch MAT_BATCH matmuls and single sync
                for (int b = 0; b < MAT_BATCH; ++b) {
                    (void)accel.ops().matmul(A, B);
                }
                auto Csync = accel.ops().matmul(A, B);
                Csync->download_data(sample.data(), sizeof(float) * sample_count, 0);
                auto t1 = Clock::now();
                iter_secs.push_back(std::chrono::duration<Sec>(t1 - t0).count());
            }
            Sec mean_s, std_s; stats_from_seconds(iter_secs, mean_s, std_s);
            double flops = 2.0 * double(MAT_M) * double(MAT_K) * double(MAT_N);
            double gflops = flops * MAT_BATCH / (mean_s * 1e9); // account batch
            std::cout << "[PERF] MatMul (" << MAT_M << "x" << MAT_K << " * " << MAT_K << "x" << MAT_N << ") : "
                        << gflops << " GFLOPS"
                        << " (mean time per iter = " << mean_s*1000.0 << " ms, std = " << std_s*1000.0 << " ms)"
                        << "  [batch=" << MAT_BATCH << "]\n";
        }

        print_hdr("Stable benchmark complete");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[FATAL] Exception: " << e.what() << std::endl;
        return 1;
    }
}
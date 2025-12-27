#pragma once

#include <Common/Types.h>
#include <string>

namespace QuasarML {

enum class GpuVendor : u32 {
    Unknown = 0,
    Nvidia,
    Amd,
    Intel,
    Apple,
    Arm,
    Qualcomm,
    ImgTec
};

enum class GpuArchitecture : u32 {
    Unknown = 0,
    
    NvidiaKepler,
    NvidiaMaxwell,
    NvidiaPascal,
    NvidiaTuring,
    NvidiaAmpere,
    NvidiaAda,
    
    AmdGcn1,
    AmdGcn2,
    AmdGcn3,
    AmdGcn4,
    AmdGcn5,
    AmdRdna1,
    AmdRdna2,
    AmdRdna3,
    AmdRdna4,
    AmdCdna1,
    AmdCdna2,
    AmdCdna3,
    
    IntelGen9,
    IntelGen11,
    IntelGen12,
    IntelXeHpg,
    IntelXeHpc,
    
    AppleM1,
    AppleM2,
    AppleM3,
    AppleM4,
    
    ArmMali,
    ArmMaliValhall,
    ArmImmortalise,
    
    QualcommAdreno
};

struct SubgroupCapabilities {
    u32 size;
    bool basic;
    bool vote;
    bool arithmetic;
    bool ballot;
    bool shuffle;
    bool shuffle_relative;
    bool clustered;
    bool quad;
};

struct DeviceCapabilities {
    GpuVendor vendor;
    GpuArchitecture architecture;
    std::string device_name;
    u32 vendor_id;
    u32 device_id;
    u32 driver_version;
    
    u32 max_workgroup_size[3];
    u32 max_workgroup_invocations;
    u32 max_compute_workgroups[3];
    u32 max_shared_memory;
    u32 max_push_constant_size;
    
    SubgroupCapabilities subgroup;
    
    bool fp16_storage;
    bool fp16_compute;
    bool fp64;
    bool int8;
    bool int16;
    bool int64;
    
    bool buffer_device_address;
    bool descriptor_indexing;
    bool timeline_semaphores;
    bool synchronization2;
    
    u32 preferred_vector_width;
    u32 optimal_tile_size;
    u32 optimal_workgroup_size_1d;
    u32 optimal_workgroup_size_2d;
    bool prefer_subgroup_reduce;
    bool prefer_shared_memory_tiling;
    u32 shared_memory_bank_count;
    u32 shared_memory_bank_width;
};

QS_API GpuVendor detect_vendor(u32 vendor_id);
QS_API GpuArchitecture detect_architecture(GpuVendor vendor, u32 device_id, const char* device_name);
QS_API const char* vendor_to_string(GpuVendor vendor);
QS_API const char* architecture_to_string(GpuArchitecture arch);

QS_API void compute_optimal_parameters(DeviceCapabilities& caps);

}

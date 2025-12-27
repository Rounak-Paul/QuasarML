#include "DeviceCapabilities.h"
#include <cstring>
#include <algorithm>

namespace QuasarML {

GpuVendor detect_vendor(u32 vendor_id) {
    switch (vendor_id) {
        case 0x10DE: return GpuVendor::Nvidia;
        case 0x1002: return GpuVendor::Amd;
        case 0x8086: return GpuVendor::Intel;
        case 0x106B: return GpuVendor::Apple;
        case 0x13B5: return GpuVendor::Arm;
        case 0x5143: return GpuVendor::Qualcomm;
        case 0x1010: return GpuVendor::ImgTec;
        default: return GpuVendor::Unknown;
    }
}

GpuArchitecture detect_architecture(GpuVendor vendor, u32 device_id, const char* device_name) {
    switch (vendor) {
        case GpuVendor::Nvidia: {
            if (strstr(device_name, "RTX 40") || strstr(device_name, "RTX 4")) 
                return GpuArchitecture::NvidiaAda;
            if (strstr(device_name, "RTX 30") || strstr(device_name, "A100") || strstr(device_name, "A10"))
                return GpuArchitecture::NvidiaAmpere;
            if (strstr(device_name, "RTX 20") || strstr(device_name, "GTX 16"))
                return GpuArchitecture::NvidiaTuring;
            if (strstr(device_name, "GTX 10") || strstr(device_name, "TITAN X") || strstr(device_name, "P100"))
                return GpuArchitecture::NvidiaPascal;
            if (strstr(device_name, "GTX 9") || strstr(device_name, "TITAN") || strstr(device_name, "M40"))
                return GpuArchitecture::NvidiaMaxwell;
            if (strstr(device_name, "GTX 7") || strstr(device_name, "GTX 6") || strstr(device_name, "K80"))
                return GpuArchitecture::NvidiaKepler;
            return GpuArchitecture::NvidiaAda;
        }
        
        case GpuVendor::Amd: {
            if (strstr(device_name, "RX 9") || strstr(device_name, "Radeon 9"))
                return GpuArchitecture::AmdRdna4;
            if (strstr(device_name, "RX 7") || strstr(device_name, "Radeon 7"))
                return GpuArchitecture::AmdRdna3;
            if (strstr(device_name, "RX 6") || strstr(device_name, "Radeon 6"))
                return GpuArchitecture::AmdRdna2;
            if (strstr(device_name, "RX 5") || strstr(device_name, "Radeon 5"))
                return GpuArchitecture::AmdRdna1;
            if (strstr(device_name, "MI300"))
                return GpuArchitecture::AmdCdna3;
            if (strstr(device_name, "MI200") || strstr(device_name, "MI250"))
                return GpuArchitecture::AmdCdna2;
            if (strstr(device_name, "MI100"))
                return GpuArchitecture::AmdCdna1;
            if (strstr(device_name, "Vega") || strstr(device_name, "VII"))
                return GpuArchitecture::AmdGcn5;
            if (strstr(device_name, "RX 4") || strstr(device_name, "RX 5"))
                return GpuArchitecture::AmdGcn4;
            return GpuArchitecture::AmdRdna3;
        }
        
        case GpuVendor::Intel: {
            if (strstr(device_name, "Arc") || strstr(device_name, "A770") || strstr(device_name, "A750") ||
                strstr(device_name, "A580") || strstr(device_name, "A380"))
                return GpuArchitecture::IntelXeHpg;
            if (strstr(device_name, "Ponte Vecchio") || strstr(device_name, "Max"))
                return GpuArchitecture::IntelXeHpc;
            if (strstr(device_name, "Iris Xe") || strstr(device_name, "UHD 7"))
                return GpuArchitecture::IntelGen12;
            if (strstr(device_name, "Iris Plus") || strstr(device_name, "UHD 6"))
                return GpuArchitecture::IntelGen11;
            return GpuArchitecture::IntelGen9;
        }
        
        case GpuVendor::Apple: {
            if (strstr(device_name, "M4"))
                return GpuArchitecture::AppleM4;
            if (strstr(device_name, "M3"))
                return GpuArchitecture::AppleM3;
            if (strstr(device_name, "M2"))
                return GpuArchitecture::AppleM2;
            if (strstr(device_name, "M1"))
                return GpuArchitecture::AppleM1;
            return GpuArchitecture::AppleM1;
        }
        
        case GpuVendor::Arm: {
            if (strstr(device_name, "Immortalis"))
                return GpuArchitecture::ArmImmortalise;
            if (strstr(device_name, "G7") || strstr(device_name, "G6") || strstr(device_name, "G5"))
                return GpuArchitecture::ArmMaliValhall;
            return GpuArchitecture::ArmMali;
        }
        
        case GpuVendor::Qualcomm:
            return GpuArchitecture::QualcommAdreno;
        
        default:
            return GpuArchitecture::Unknown;
    }
    (void)device_id;
}

const char* vendor_to_string(GpuVendor vendor) {
    switch (vendor) {
        case GpuVendor::Nvidia: return "NVIDIA";
        case GpuVendor::Amd: return "AMD";
        case GpuVendor::Intel: return "Intel";
        case GpuVendor::Apple: return "Apple";
        case GpuVendor::Arm: return "ARM";
        case GpuVendor::Qualcomm: return "Qualcomm";
        case GpuVendor::ImgTec: return "Imagination Technologies";
        default: return "Unknown";
    }
}

const char* architecture_to_string(GpuArchitecture arch) {
    switch (arch) {
        case GpuArchitecture::NvidiaKepler: return "Kepler";
        case GpuArchitecture::NvidiaMaxwell: return "Maxwell";
        case GpuArchitecture::NvidiaPascal: return "Pascal";
        case GpuArchitecture::NvidiaTuring: return "Turing";
        case GpuArchitecture::NvidiaAmpere: return "Ampere";
        case GpuArchitecture::NvidiaAda: return "Ada Lovelace";
        case GpuArchitecture::AmdGcn1: return "GCN 1.0";
        case GpuArchitecture::AmdGcn2: return "GCN 2.0";
        case GpuArchitecture::AmdGcn3: return "GCN 3.0";
        case GpuArchitecture::AmdGcn4: return "GCN 4.0";
        case GpuArchitecture::AmdGcn5: return "GCN 5.0";
        case GpuArchitecture::AmdRdna1: return "RDNA 1";
        case GpuArchitecture::AmdRdna2: return "RDNA 2";
        case GpuArchitecture::AmdRdna3: return "RDNA 3";
        case GpuArchitecture::AmdRdna4: return "RDNA 4";
        case GpuArchitecture::AmdCdna1: return "CDNA 1";
        case GpuArchitecture::AmdCdna2: return "CDNA 2";
        case GpuArchitecture::AmdCdna3: return "CDNA 3";
        case GpuArchitecture::IntelGen9: return "Gen9";
        case GpuArchitecture::IntelGen11: return "Gen11";
        case GpuArchitecture::IntelGen12: return "Gen12";
        case GpuArchitecture::IntelXeHpg: return "Xe-HPG";
        case GpuArchitecture::IntelXeHpc: return "Xe-HPC";
        case GpuArchitecture::AppleM1: return "M1";
        case GpuArchitecture::AppleM2: return "M2";
        case GpuArchitecture::AppleM3: return "M3";
        case GpuArchitecture::AppleM4: return "M4";
        case GpuArchitecture::ArmMali: return "Mali";
        case GpuArchitecture::ArmMaliValhall: return "Mali Valhall";
        case GpuArchitecture::ArmImmortalise: return "Immortalis";
        case GpuArchitecture::QualcommAdreno: return "Adreno";
        default: return "Unknown";
    }
}

void compute_optimal_parameters(DeviceCapabilities& caps) {
    u32 subgroup_size = caps.subgroup.size > 0 ? caps.subgroup.size : 32;
    
    switch (caps.vendor) {
        case GpuVendor::Nvidia: {
            caps.preferred_vector_width = 4;
            caps.shared_memory_bank_count = 32;
            caps.shared_memory_bank_width = 4;
            caps.prefer_subgroup_reduce = caps.subgroup.arithmetic;
            caps.prefer_shared_memory_tiling = true;
            
            switch (caps.architecture) {
                case GpuArchitecture::NvidiaAda:
                case GpuArchitecture::NvidiaAmpere:
                    caps.optimal_workgroup_size_1d = 256;
                    caps.optimal_workgroup_size_2d = 16;
                    caps.optimal_tile_size = 128;
                    break;
                case GpuArchitecture::NvidiaTuring:
                case GpuArchitecture::NvidiaPascal:
                    caps.optimal_workgroup_size_1d = 256;
                    caps.optimal_workgroup_size_2d = 16;
                    caps.optimal_tile_size = 64;
                    break;
                default:
                    caps.optimal_workgroup_size_1d = 128;
                    caps.optimal_workgroup_size_2d = 8;
                    caps.optimal_tile_size = 32;
                    break;
            }
            break;
        }
        
        case GpuVendor::Amd: {
            caps.preferred_vector_width = 4;
            caps.shared_memory_bank_count = 32;
            caps.shared_memory_bank_width = 4;
            caps.prefer_subgroup_reduce = caps.subgroup.arithmetic;
            caps.prefer_shared_memory_tiling = true;
            
            switch (caps.architecture) {
                case GpuArchitecture::AmdRdna4:
                case GpuArchitecture::AmdRdna3:
                case GpuArchitecture::AmdRdna2:
                    caps.optimal_workgroup_size_1d = subgroup_size * 4;
                    caps.optimal_workgroup_size_2d = 16;
                    caps.optimal_tile_size = 64;
                    break;
                case GpuArchitecture::AmdRdna1:
                    caps.optimal_workgroup_size_1d = subgroup_size * 4;
                    caps.optimal_workgroup_size_2d = 16;
                    caps.optimal_tile_size = 32;
                    break;
                case GpuArchitecture::AmdCdna3:
                case GpuArchitecture::AmdCdna2:
                case GpuArchitecture::AmdCdna1:
                    caps.optimal_workgroup_size_1d = 256;
                    caps.optimal_workgroup_size_2d = 16;
                    caps.optimal_tile_size = 128;
                    break;
                default:
                    caps.optimal_workgroup_size_1d = 256;
                    caps.optimal_workgroup_size_2d = 16;
                    caps.optimal_tile_size = 32;
                    break;
            }
            break;
        }
        
        case GpuVendor::Intel: {
            caps.preferred_vector_width = 8;
            caps.shared_memory_bank_count = 16;
            caps.shared_memory_bank_width = 4;
            caps.prefer_subgroup_reduce = caps.subgroup.arithmetic;
            caps.prefer_shared_memory_tiling = true;
            
            switch (caps.architecture) {
                case GpuArchitecture::IntelXeHpg:
                case GpuArchitecture::IntelXeHpc:
                    caps.optimal_workgroup_size_1d = 256;
                    caps.optimal_workgroup_size_2d = 16;
                    caps.optimal_tile_size = 64;
                    break;
                default:
                    caps.optimal_workgroup_size_1d = 128;
                    caps.optimal_workgroup_size_2d = 8;
                    caps.optimal_tile_size = 32;
                    break;
            }
            break;
        }
        
        case GpuVendor::Apple: {
            caps.preferred_vector_width = 4;
            caps.shared_memory_bank_count = 32;
            caps.shared_memory_bank_width = 4;
            caps.prefer_subgroup_reduce = caps.subgroup.arithmetic;
            caps.prefer_shared_memory_tiling = true;
            caps.optimal_workgroup_size_1d = 256;
            caps.optimal_workgroup_size_2d = 16;
            caps.optimal_tile_size = 32;
            break;
        }
        
        case GpuVendor::Arm:
        case GpuVendor::Qualcomm: {
            caps.preferred_vector_width = 4;
            caps.shared_memory_bank_count = 16;
            caps.shared_memory_bank_width = 4;
            caps.prefer_subgroup_reduce = caps.subgroup.arithmetic;
            caps.prefer_shared_memory_tiling = false;
            caps.optimal_workgroup_size_1d = 64;
            caps.optimal_workgroup_size_2d = 8;
            caps.optimal_tile_size = 16;
            break;
        }
        
        default: {
            caps.preferred_vector_width = 4;
            caps.shared_memory_bank_count = 32;
            caps.shared_memory_bank_width = 4;
            caps.prefer_subgroup_reduce = false;
            caps.prefer_shared_memory_tiling = true;
            caps.optimal_workgroup_size_1d = 128;
            caps.optimal_workgroup_size_2d = 8;
            caps.optimal_tile_size = 16;
            break;
        }
    }
    
    caps.optimal_workgroup_size_1d = qs_min(caps.optimal_workgroup_size_1d, caps.max_workgroup_invocations);
    u32 max_2d = 1;
    while (max_2d * max_2d <= caps.max_workgroup_invocations) max_2d++;
    max_2d--;
    caps.optimal_workgroup_size_2d = qs_min(caps.optimal_workgroup_size_2d, max_2d);
}

}

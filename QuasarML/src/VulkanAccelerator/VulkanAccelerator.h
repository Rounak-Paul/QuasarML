#pragma once

#include <qspch.h>
#include <Container/DeletionQueue.h>
#include "VulkanTypes.h"

namespace QuasarML {

class VulkanAccelerator {
    public:
    VulkanAccelerator(const std::string& name="QuasarAccelerator", u32 gpu_idx=0);
    ~VulkanAccelerator();

    void device_wait_idle();
    
    private:
    DeletionQueue _deletion_queue;
    VulkanContext _ctx;
};
}

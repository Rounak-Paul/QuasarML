#pragma once

#include <qspch.h>

#include <VulkanAccelerator/VulkanAccelerator.h>

namespace QuasarML {

class QS_API Engine {
    public:
    Engine(const std::string& application_name);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    private:
    VulkanAccelerator* _accelerator;
};

}
#include "Engine.h"

namespace QuasarML {
Engine::Engine(const std::string& application_name)
{
    _accelerator = new VulkanAccelerator{application_name, 0};
}
Engine::~Engine()
{
    if (_accelerator)
        delete _accelerator;
}
}
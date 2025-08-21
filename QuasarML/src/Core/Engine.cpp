#include "Engine.h"

namespace QuasarML {
Engine::Engine()
{
    _accelerator = new VulkanAccelerator{};
}
Engine::~Engine()
{
    if (_accelerator)
        delete _accelerator;
}
}
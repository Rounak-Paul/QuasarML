#pragma once

#include <stdexcept>
#include <string>

namespace QuasarML {

#define QS_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error(std::string("Assertion failed: ") + (message)); \
        } \
    } while (0)

#define QS_CHECK(condition) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error("Check failed: " #condition); \
        } \
    } while (0)

#define QS_UNREACHABLE() \
    throw std::runtime_error("Unreachable code reached")

}

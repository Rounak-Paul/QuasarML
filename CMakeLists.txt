cmake_minimum_required(VERSION 3.20)
set (CMAKE_CXX_STANDARD 20)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebug")
else()
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

project(QuasarML)

# Performance build defaults and options
option(QS_ENABLE_FAST_MATH "Enable -ffast-math for math optimizations (may change numerical results)" OFF)
option(QS_ENABLE_PGO "Enable Profile-Guided Optimization helper flags (requires manual run steps)" OFF)

# Ensure a sensible default when not specified by caller
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Configure sensible Release flags per compiler and enable IPO/LTO when available
include(CheckIPOSupported)
check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_output)
if(_ipo_supported)
    message(STATUS "IPO / LTO supported: enabling interprocedural optimization for Release builds")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
else()
    message(STATUS "IPO / LTO not supported: ${_ipo_output}")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    set(_release_common_flags "-O3 -DNDEBUG -march=native")
    if(QS_ENABLE_FAST_MATH)
        set(_release_common_flags "${_release_common_flags} -ffast-math")
    endif()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${_release_common_flags}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${_release_common_flags}")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG /GL")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /DNDEBUG /GL")
endif()

# Debug Mode -----------------------------------------------
add_compile_definitions($<$<CONFIG:Debug>:QS_DEBUG>)
# ----------------------------------------------------------

# Renderer Backend -----------------------------------------
# NOTE: 0nly one of the following should be defined
add_compile_definitions(QS_RENDERER_BACKEND_VULKAN)
# add_compile_definitions(QS_RENDERER_BACKEND_D3DX12) # not implemented
# add_compile_definitions(QS_RENDERER_BACKEND_METAL) # not implemented
# ----------------------------------------------------------

set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "")

# Find Vulkan SDK dynamically
find_package(Vulkan REQUIRED)

if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan SDK not found. Please install the Vulkan SDK and ensure it's in your PATH.")
endif()

message(STATUS "Found Vulkan: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan Libraries: ${Vulkan_LIBRARIES}")

if(APPLE)
    message("Building on Apple macOS or iOS")
    add_definitions(-DQS_PLATFORM_APPLE)
    # Set up MoltenVK environment if Vulkan SDK provides it
    if(EXISTS "${Vulkan_INCLUDE_DIRS}/../share/vulkan/icd.d/MoltenVK_icd.json")
        set(ENV{VK_ICD_FILENAMES} "${Vulkan_INCLUDE_DIRS}/../share/vulkan/icd.d/MoltenVK_icd.json")
    endif()
    if(EXISTS "${Vulkan_INCLUDE_DIRS}/../share/vulkan/explicit_layer.d")
        set(ENV{VK_LAYER_PATH} "${Vulkan_INCLUDE_DIRS}/../share/vulkan/explicit_layer.d")
    endif()

elseif(WIN32)
    message("Building on Windows")
    add_definitions(-DQS_PLATFORM_WINDOWS)

elseif(UNIX)
    message("Building on Unix-like OS (including Linux)")
    add_definitions(-DQS_PLATFORM_LINUX)

endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(QuasarML)
add_subdirectory(unit-tests)
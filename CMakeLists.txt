cmake_minimum_required(VERSION 3.20)
set (CMAKE_CXX_STANDARD 20)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebug")
else()
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

project(QuasarML)

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

if(APPLE)
    message("Building on Apple macOS or iOS")
    set(VULKAN_PATH "/Users/duke/VulkanSDK/1.4.313.1/macOS")
    add_definitions(-DQS_PLATFORM_APPLE)
    set(ENV{VK_ICD_FILENAMES} "${VULKAN_PATH}/share/vulkan/icd.d/MoltenVK_icd.json")
    set(ENV{VK_LAYER_PATH} "${VULKAN_PATH}/share/vulkan/explicit_layer.d")

elseif(WIN32)
    message("Building on Windows")
    set(VULKAN_PATH $ENV{VULKAN_SDK})
    add_definitions(-DQS_PLATFORM_WINDOWS)

elseif(UNIX)
    message("Building on Unix-like OS (including Linux)")
    add_definitions(-DQS_PLATFORM_LINUX)

else()
set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_C_FLAGS_RELEASE "")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(QuasarML)
add_subdirectory(unit-tests)
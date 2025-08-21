file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    src/**/*.cpp
    src/*.cpp
    src/**/*.h
    src/*.h
    src/**/*.inl
)
add_library(QuasarML SHARED ${SOURCES})
set_target_properties(QuasarML PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(QuasarML PRIVATE QS_BUILD_DLL)

set(TINYGLTF_HEADER_ONLY ON CACHE INTERNAL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE INTERNAL "" FORCE)

add_subdirectory(vendor/VMA)
add_subdirectory(vendor/MIMALLOC/mimalloc)
add_subdirectory(vendor/JSON)

target_precompile_headers(QuasarML PUBLIC src/qspch.h)
target_precompile_headers(QuasarML PUBLIC src/defines.h)

set(MI_OVERRIDE OFF CACHE BOOL "" FORCE)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -g")

# Include directories
target_include_directories(QuasarML PUBLIC 
    src
    ${VULKAN_PATH}/include
    vendor/VMA/include
    vendor/MIMALLOC
    Vendor/JSON/single_include
)

# Link directories
target_link_directories(QuasarML PUBLIC ${VULKAN_PATH}/lib)

# Link core libraries
target_link_libraries(QuasarML PUBLIC 
    VulkanMemoryAllocator
    mimalloc-static
    nlohmann_json::nlohmann_json
)

if(APPLE)
    target_link_libraries(QuasarML PUBLIC 
        ${VULKAN_PATH}/lib/libvulkan.1.dylib
        ${VULKAN_PATH}/lib/libvulkan.1.4.313.dylib
        ${VULKAN_PATH}/lib/libshaderc_shared.dylib
        ${VULKAN_PATH}/lib/libspirv-cross-c-shared.dylib
    )
elseif(WIN32)
    target_link_libraries(QuasarML PUBLIC 
        ${VULKAN_PATH}/lib/vulkan-1.lib
        ${VULKAN_PATH}/lib/shaderc_shared.lib
    )
endif()


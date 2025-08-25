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

# Target-level performance flags
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    target_compile_options(QuasarML PRIVATE "$<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native>")
    if(QS_ENABLE_FAST_MATH)
        target_compile_options(QuasarML PRIVATE "$<$<CONFIG:Release>:-ffast-math>")
    endif()
elseif(MSVC)
    target_compile_options(QuasarML PRIVATE "$<$<CONFIG:Release>:/O2;/DNDEBUG;/GL>")
endif()

# Enable link time optimization if requested by top-level CMake
if(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE)
    set_target_properties(QuasarML PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

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
    SYSTEM ${CMAKE_SOURCE_DIR}/QuasarML/vendor/VMA/include
    SYSTEM ${CMAKE_SOURCE_DIR}/QuasarML/vendor/MIMALLOC
    SYSTEM ${CMAKE_SOURCE_DIR}/QuasarML/vendor/JSON/single_include
)

# Link directories
target_link_directories(QuasarML PUBLIC ${VULKAN_PATH}/lib)

# Link core libraries
target_link_libraries(QuasarML PUBLIC 
    VulkanMemoryAllocator
    mimalloc-static
    nlohmann_json::nlohmann_json
)

# Note: Do not globally hide symbols here because the project uses its own
# export macros (QS_BUILD_DLL) to control symbol visibility. Global hiding
# can prevent public API symbols from being exported on macOS.

# PGO helper: add instrumentation/usage flags when requested (user must run training/profile steps)
if(QS_ENABLE_PGO)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
        target_compile_options(QuasarML PRIVATE "$<$<CONFIG:Release>:-fprofile-instr-generate;-fcoverage-mapping>")
        target_link_options(QuasarML PRIVATE "$<$<CONFIG:Release>:-fprofile-instr-generate>")
    elseif(MSVC)
        # MSVC PGO flow is more involved; leave as a manual note
    endif()
endif()

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


cmake_minimum_required(VERSION 3.20)
project(NanoGPT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_SOURCE_DIR}/../../QuasarML/include
    ${CMAKE_SOURCE_DIR}/../../QuasarML/src
    ${CMAKE_SOURCE_DIR}/../../QuasarML/vendor/VMA/include
    ${CMAKE_SOURCE_DIR}/../../QuasarML/vendor/JSON/include
)
link_directories(${CMAKE_SOURCE_DIR}/../../bin)

add_executable(nano_gpt nano_gpt.cpp)
target_link_libraries(nano_gpt QuasarML)

set_target_properties(nano_gpt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../bin
)

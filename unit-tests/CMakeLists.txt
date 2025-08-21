file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    src/**/*.cpp
    src/*.cpp
    src/**/*.h
    src/*.h
    src/**/*.inl
)
add_executable(unit-tests ${TEST_SOURCES})
set_target_properties(unit-tests PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(unit-tests PRIVATE PROJECT_BUILD_DLL)

target_include_directories(unit-tests PUBLIC 
    src
    ${PROJECT_SOURCE_DIR}/QuasarML/include
)

target_link_libraries(unit-tests PUBLIC QuasarML)
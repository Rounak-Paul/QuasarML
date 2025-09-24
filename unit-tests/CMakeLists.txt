# Functionality test
add_executable(_test_qsml_functionality src/_test_qsml_functionality.cpp)
target_include_directories(_test_qsml_functionality PUBLIC src SYSTEM ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_test_qsml_functionality PUBLIC QuasarML)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
	target_compile_options(_test_qsml_functionality PRIVATE "$<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native>")
elseif(MSVC)
	target_compile_options(_test_qsml_functionality PRIVATE "$<$<CONFIG:Release>:/O2;/DNDEBUG;/GL>")
endif()
# Minimal test
add_executable(_test_minimal src/_test_minimal.cpp)
target_include_directories(_test_minimal PUBLIC src SYSTEM ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_test_minimal PUBLIC QuasarML)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
	target_compile_options(_test_minimal PRIVATE "$<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native>")
elseif(MSVC)
	target_compile_options(_test_minimal PRIVATE "$<$<CONFIG:Release>:/O2;/DNDEBUG;/GL>")
endif()

# Functionality test
add_executable(_test_qsml_functionality src/_test_qsml_functionality.cpp)
target_include_directories(_test_qsml_functionality PUBLIC src SYSTEM ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_test_qsml_functionality PUBLIC QuasarML)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
	target_compile_options(_test_qsml_functionality PRIVATE "$<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native>")
elseif(MSVC)
	target_compile_options(_test_qsml_functionality PRIVATE "$<$<CONFIG:Release>:/O2;/DNDEBUG;/GL>")
endif()

# Comprehensive test
add_executable(_test_comprehensive src/_test_comprehensive.cpp)
target_include_directories(_test_comprehensive PUBLIC src SYSTEM ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_test_comprehensive PUBLIC QuasarML)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
	target_compile_options(_test_comprehensive PRIVATE "$<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native>")
elseif(MSVC)
	target_compile_options(_test_comprehensive PRIVATE "$<$<CONFIG:Release>:/O2;/DNDEBUG;/GL>")
endif()

# Performance benchmark
add_executable(_bench_qsml_performance src/_bench_qsml_performance.cpp)
target_include_directories(_bench_qsml_performance PUBLIC src SYSTEM ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_bench_qsml_performance PUBLIC QuasarML)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
	target_compile_options(_bench_qsml_performance PRIVATE "$<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native>")
elseif(MSVC)
	target_compile_options(_bench_qsml_performance PRIVATE "$<$<CONFIG:Release>:/O2;/DNDEBUG;/GL>")
endif()

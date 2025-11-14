# Functionality test
add_executable(_test_qsml_functionality src/_test_qsml_functionality.cpp)
target_include_directories(_test_qsml_functionality PUBLIC src SYSTEM ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_test_qsml_functionality PUBLIC QuasarML)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
	target_compile_options(_test_qsml_functionality PRIVATE "$<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native>")
elseif(MSVC)
	target_compile_options(_test_qsml_functionality PRIVATE "$<$<CONFIG:Release>:/O2;/DNDEBUG;/GL>")
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

# Thread safety and multi-GPU test
add_executable(_test_threading src/_test_threading.cpp)
target_include_directories(_test_threading PUBLIC src SYSTEM ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_test_threading PUBLIC QuasarML)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
	target_compile_options(_test_threading PRIVATE "$<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native>")
elseif(MSVC)
	target_compile_options(_test_threading PRIVATE "$<$<CONFIG:Release>:/O2;/DNDEBUG;/GL>")
endif()

# Compute-bound benchmark (high TFLOPS operations)
add_executable(_bench_compute_bound src/_bench_compute_bound.cpp)
target_include_directories(_bench_compute_bound PUBLIC src SYSTEM ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_bench_compute_bound PUBLIC QuasarML)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
	target_compile_options(_bench_compute_bound PRIVATE "$<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native>")
elseif(MSVC)
	target_compile_options(_bench_compute_bound PRIVATE "$<$<CONFIG:Release>:/O2;/DNDEBUG;/GL>")
endif()

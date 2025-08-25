# Functionality test
add_executable(_test_qsml_functionality src/_test_qsml_functionality.cpp)
target_include_directories(_test_qsml_functionality PUBLIC src SYSTEM ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_test_qsml_functionality PUBLIC QuasarML)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
	target_compile_options(_test_qsml_functionality PRIVATE "$<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native>")
elseif(MSVC)
	target_compile_options(_test_qsml_functionality PRIVATE "$<$<CONFIG:Release>:/O2;/DNDEBUG;/GL>")
endif()

# Functionality test
add_executable(_test_qsml_benchmark src/_test_qsml_benchmark.cpp)
target_include_directories(_test_qsml_benchmark PUBLIC src SYSTEM ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_test_qsml_benchmark PUBLIC QuasarML)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
	target_compile_options(_test_qsml_benchmark PRIVATE "$<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native>")
elseif(MSVC)
	target_compile_options(_test_qsml_benchmark PRIVATE "$<$<CONFIG:Release>:/O2;/DNDEBUG;/GL>")
endif()

# CNN test
add_executable(_test_qsml_cnn src/_test_qsml_cnn.cpp)
target_include_directories(_test_qsml_cnn PUBLIC src SYSTEM ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_test_qsml_cnn PUBLIC QuasarML)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
	target_compile_options(_test_qsml_cnn PRIVATE "$<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native>")
elseif(MSVC)
	target_compile_options(_test_qsml_cnn PRIVATE "$<$<CONFIG:Release>:/O2;/DNDEBUG;/GL>")
endif()
if(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE)
	set_target_properties(_test_qsml_functionality PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
	set_target_properties(_test_qsml_benchmark PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
	set_target_properties(_test_qsml_cnn PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()
# Functionality test
add_executable(_test_qsml_functionality src/_test_qsml_functionality.cpp)
target_include_directories(_test_qsml_functionality PUBLIC src ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_test_qsml_functionality PUBLIC QuasarML)

# Functionality test
add_executable(_test_qsml_benchmark src/_test_qsml_benchmark.cpp)
target_include_directories(_test_qsml_benchmark PUBLIC src ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_test_qsml_benchmark PUBLIC QuasarML)

# CNN test
add_executable(_test_qsml_cnn src/_test_qsml_cnn.cpp)
target_include_directories(_test_qsml_cnn PUBLIC src ${PROJECT_SOURCE_DIR}/QuasarML/include)
target_link_libraries(_test_qsml_cnn PUBLIC QuasarML)
#include <QuasarML.h>
#include <iostream>
#include <vector>
#include <cmath>

bool run_test() {
    std::cout << "Creating tensors..." << std::endl;
    auto a = qsml::ones({2, 2});
    auto b = qsml::ones({2, 2});
    
    std::cout << "Downloading tensor 'a'..." << std::endl;
    std::vector<float> a_data(4);
    a->download(a_data.data(), a_data.size() * sizeof(float));
    std::cout << "a[0] = " << a_data[0] << ", expected 1.0" << std::endl;
    
    std::cout << "Downloading tensor 'b'..." << std::endl;
    std::vector<float> b_data(4);
    b->download(b_data.data(), b_data.size() * sizeof(float));
    std::cout << "b[0] = " << b_data[0] << ", expected 1.0" << std::endl;
    
    std::cout << "Computing add..." << std::endl;
    auto c = qsml::add(a, b);
    
    std::cout << "Shape of c: [" << c->dim(0) << ", " << c->dim(1) << "]" << std::endl;
    std::cout << "Element count: " << c->numel() << std::endl;
    
    std::cout << "Downloading result 'c'..." << std::endl;
    std::vector<float> c_data(4);
    c->download(c_data.data(), c_data.size() * sizeof(float));
    std::cout << "c[0] = " << c_data[0] << ", expected 2.0" << std::endl;
    std::cout << "c[1] = " << c_data[1] << ", expected 2.0" << std::endl;
    std::cout << "c[2] = " << c_data[2] << ", expected 2.0" << std::endl;
    std::cout << "c[3] = " << c_data[3] << ", expected 2.0" << std::endl;
    
    bool passed = std::abs(c_data[0] - 2.0f) < 1e-4;
    if (passed) {
        std::cout << "\n Test PASSED" << std::endl;
    } else {
        std::cout << "\n Test FAILED" << std::endl;
    }
    return passed;
}

int main() {
    qsml::init();
    bool passed = run_test();
    qsml::shutdown();
    return passed ? 0 : 1;
}

#include <QuasarML.h>
#include <iostream>
#include <vector>

using namespace QuasarML;

int main() {
    std::cout << "Creating tensors..." << std::endl;
    auto a = qsml::ones({2, 2}, DataType::F32);
    auto b = qsml::ones({2, 2}, DataType::F32);
    
    std::cout << "Downloading tensor 'a'..." << std::endl;
    std::vector<float> a_data(4);
    a->download_data(a_data.data());
    std::cout << "a[0] = " << a_data[0] << ", expected 1.0" << std::endl;
    
    std::cout << "Downloading tensor 'b'..." << std::endl;
    std::vector<float> b_data(4);
    b->download_data(b_data.data());
    std::cout << "b[0] = " << b_data[0] << ", expected 1.0" << std::endl;
    
    std::cout << "Computing add..." << std::endl;
    auto c = qsml::add(a, b);
    
    std::cout << "Shape of c: [" << c->get_shape()[0] << ", " << c->get_shape()[1] << "]" << std::endl;
    std::cout << "Element count: " << c->get_element_count() << std::endl;
    
    std::cout << "Downloading result 'c'..." << std::endl;
    std::vector<float> c_data(4);
    c->download_data(c_data.data());
    std::cout << "c[0] = " << c_data[0] << ", expected 2.0" << std::endl;
    std::cout << "c[1] = " << c_data[1] << ", expected 2.0" << std::endl;
    std::cout << "c[2] = " << c_data[2] << ", expected 2.0" << std::endl;
    std::cout << "c[3] = " << c_data[3] << ", expected 2.0" << std::endl;
    
    if (std::abs(c_data[0] - 2.0f) < 1e-4) {
        std::cout << "\n✓ Test PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ Test FAILED" << std::endl;
        return 1;
    }
}

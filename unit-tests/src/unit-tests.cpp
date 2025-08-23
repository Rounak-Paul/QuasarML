#include <QuasarML.h>

using namespace QuasarML;

void print_tensor_sample(QuasarML::Tensor& tensor, size_t max_elements = 10) {
    size_t num_elems = tensor.num_elements();
    size_t to_print = (num_elems < max_elements) ? num_elems : max_elements;

    // Allocate host buffer
    size_t elem_size = 4; // Assuming float32 here, you can adjust to tensor.dtype()
    std::vector<float> host_data(num_elems);

    // Download data from GPU to CPU
    tensor.download(host_data.data(), elem_size * num_elems);

    std::cout << "Tensor sample: [";
    for (size_t i = 0; i < to_print; ++i) {
        std::cout << host_data[i];
        if (i != to_print - 1) std::cout << ", ";
    }
    if (num_elems > to_print) {
        std::cout << ", ...";
    }
    std::cout << "]\n";
}

int main(int argc, char** argv) {
	Accelerator accelerator("QuasarML-GPU");

    Tensor A = accelerator.create_tensor({4096}, DataType::FLOAT32, true);
    Tensor B = accelerator.create_tensor({4096}, DataType::FLOAT32, true);

    std::vector<float> vecA(4096, 1.0f);
    std::vector<float> vecB(4096, 5.0f);
    A.upload(vecA.data(), sizeof(float) * vecA.size());
    B.upload(vecB.data(), sizeof(float) * vecB.size());

    Tensor C = accelerator.add(A, B);
    Tensor D = accelerator.multiply(A, B);

    std::vector<float> vecC(4096), vecD(4096);
    C.download(vecC.data(), sizeof(float) * vecC.size());
    D.download(vecD.data(), sizeof(float) * vecD.size());

    accelerator.sync();
    
    print_tensor_sample(C);
    print_tensor_sample(D);

    return 0;
}
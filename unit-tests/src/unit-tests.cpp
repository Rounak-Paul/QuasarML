#include <QuasarML.h>

int main(int argc, char** argv) {
	auto engine = QuasarML::Engine{"unit-tests"};
    engine.run_benchmark(100);

    return 0;
}
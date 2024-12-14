#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <vector>

void run_benchmark(const std::string& library, const std::string& precision, const std::string& dimension, 
                   double real_lower, double real_upper, double imag_lower, double imag_upper, int count) {
    // Build the command string
    std::ostringstream command;
    command << "./benchmarking_suite -l " << library 
            << " -p " << precision 
            << " -d " << dimension 
            << " -b precision " 
            << "-o classifier_data/ "
            << "-1 " << std::fixed << std::setprecision(4) << real_lower
            << " -2 " << std::fixed << std::setprecision(4) << real_upper
            << " -3 " << std::fixed << std::setprecision(4) << imag_lower
            << " -4 " << std::fixed << std::setprecision(4) << imag_upper;

    // Print the command to ensure it's correct
    std::cout << "Run: " << count << std::endl;
    std::cout << command.str() << std::endl;

    // Run the command
    system(command.str().c_str());
}

int main() {
    // Define the options to cycle through
    std::vector<std::string> libraries = {"vkfft", "cufft", "cufftdx"};
    std::vector<std::string> precisions = {"half", "single", "double"};
    std::vector<std::string> dimensions = {"1d", "2d"};

    int count = 0;

    for (const auto& library : libraries) {
        for (const auto& precision : precisions) {
            for (const auto& dimension : dimensions) {
                if (library == "cufftdx" && precision == "half" && dimension == "2d") {
                    continue;
                }
                for (int i = 0; i < 500; ++i) {
                    std::srand(static_cast<unsigned>(std::time(0)) + i);
                    double real_lower, real_upper, imag_lower, imag_upper;
                    count++;

                    bool swap_ranges = (std::rand() % 2 == 0); // Randomize which part gets large or small bounds

                    if (i % 50 == 0) {
                        // Zero variance (lower bound equals upper bound)
                        real_lower = std::rand() % 201 - 100;
                        real_upper = real_lower;
                        imag_lower = std::rand() % 201 - 100;
                        imag_upper = imag_lower;
                    } else if (i % 8 == 0) {
                        // Very large bounds between 100 and 1000
                        real_lower = std::rand() % 2001 - 1000;
                        real_upper = real_lower + 100 + std::rand() % 900;
                        imag_lower = std::rand() % 2001 - 1000;
                        imag_upper = imag_lower + 100 + std::rand() % 900;
                    } else if (i % 6 == 0) {
                        // Very small bounds between 0.01 and 1
                        real_lower = static_cast<double>(std::rand()) / RAND_MAX * 10 - 5;
                        real_upper = real_lower + (static_cast<double>(std::rand()) / RAND_MAX * 0.99 + 0.01);
                        imag_lower = static_cast<double>(std::rand()) / RAND_MAX * 10 - 5;
                        imag_upper = imag_lower + (static_cast<double>(std::rand()) / RAND_MAX * 0.99 + 0.01);
                    } else if (i % 4 == 0) {
                        // Large bounds between 50 and 100
                        real_lower = std::rand() % 201 - 100;
                        real_upper = real_lower + 50 + std::rand() % 50;
                        imag_lower = std::rand() % 201 - 100;
                        imag_upper = imag_lower + 50 + std::rand() % 50;
                    } else if (i % 3 == 0) {
                        // Medium bounds between 10 and 50
                        real_lower = static_cast<double>(std::rand()) / RAND_MAX * 10 - 5;
                        real_upper = real_lower + (static_cast<double>(std::rand()) / RAND_MAX * 40 + 10);
                        imag_lower = static_cast<double>(std::rand()) / RAND_MAX * 10 - 5;
                        imag_upper = imag_lower + (static_cast<double>(std::rand()) / RAND_MAX * 40 + 10);
                    } else if (i % 2 == 0) {
                        // Small bounds between 1 and 10
                        real_lower = static_cast<double>(std::rand()) / RAND_MAX * 10 - 5;
                        real_upper = real_lower + (static_cast<double>(std::rand()) / RAND_MAX * 9 + 1);
                        imag_lower = static_cast<double>(std::rand()) / RAND_MAX * 10 - 5;
                        imag_upper = imag_lower + (static_cast<double>(std::rand()) / RAND_MAX * 9 + 1);
                    } else {
                        // Regular bounds between 0.1 and 100
                        real_lower = std::rand() % 1001 - 500;
                        real_upper = real_lower + 0.1 + std::rand() % 100;
                        imag_lower = std::rand() % 1001 - 500;
                        imag_upper = imag_lower + 0.1 + std::rand() % 100;
                    }

                    // Call the benchmark function with the generated parameters
                    run_benchmark(library, precision, dimension, real_lower, real_upper, imag_lower, imag_upper, count);
                }
            }
        }
    }

    return 0;
}

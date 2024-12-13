#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <cmath>
#include <iomanip>
#include <fstream>

// Struct to represent complex numbers
struct Complex {
    double real;
    double imag;
};

// Function to compute FFTW and check for magnitudes exceeding a threshold
int compute_fftw_and_check_threshold(int N, double real_lower, double real_upper, double imag_lower, double imag_upper, double threshold) {
    // Allocate input and output arrays
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    // Initialize input array with random values within the specified bounds
    for (int i = 0; i < N; ++i) {
        in[i][0] = static_cast<double>(rand()) / RAND_MAX * (real_upper - real_lower) + real_lower;
        in[i][1] = static_cast<double>(rand()) / RAND_MAX * (imag_upper - imag_lower) + imag_lower;
    }

    // Create a plan for the forward FFT
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute the FFT
    fftw_execute(p);

    // Count how many output values have a magnitude greater than the threshold
    int count = 0;
    for (int i = 0; i < N; ++i) {
        if (std::abs(out[i][0]) > threshold || std::abs(out[i][1]) > threshold) {
            count++;
        }
    }

    // Cleanup
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    return count;
}

int main() {
    std::srand(static_cast<unsigned>(std::time(0))); // Seed random number generator
    double threshold = 65504.0;  // Threshold for checking FFT output magnitudes

    // Number of sample points (you can adjust or generate dynamically)
    std::vector<int> N_values;
    int start_power = 6;
    int end_power = 23;
    for (int i = start_power; i <= end_power; ++i) {
        N_values.push_back(1 << i); // 1 << i is equivalent to 2^i
    }

    // Open CSV file to save results
    std::ofstream csv_file("fftw_threshold.csv");
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open the CSV file." << std::endl;
        return 1;
    }

    // Write header to the CSV file
    csv_file << "N,diff_real,diff_imag,Count,Iteration" << std::endl;

    int iter = 0;
    // Loop over the different input configurations
    for (int i = 0; i < 5; ++i) {
        double real_lower, real_upper, imag_lower, imag_upper;
        iter++;
        real_lower = std::rand() % 20001;
        real_upper = real_lower + 120 + std::rand() % 10;
        imag_lower = std::rand() % 2001 - 1000;
        imag_upper = imag_lower + 120 + std::rand() % 10;
        // if (i % 20 == 0) {
        //     // Zero variance (lower bound equals upper bound)
        //     real_lower = static_cast<double>(std::rand()) / RAND_MAX * 2000 - 1000;
        //     real_upper = real_lower;  // Zero variance
        //     imag_lower = static_cast<double>(std::rand()) / RAND_MAX * 2000 - 1000;
        //     imag_upper = imag_lower;  // Zero variance
        // } else if (i % 15 == 0) {
        //     // Large variance
        //     real_lower = std::rand() % 2001 - 1000;
        //     real_upper = real_lower + 1 + std::rand() % 3000;
        //     imag_lower = std::rand() % 2001 - 1000;
        //     imag_upper = imag_lower + 1 + std::rand() % 3000;
        // } else if(i % 10 == 0){
        //     // Moderate variance
        //     real_lower = std::rand() % 20001;
        //     real_upper = real_lower + 1 + std::rand() % 10;
        //     imag_lower = std::rand() % 2001 - 1000;
        //     imag_upper = imag_lower + 1 + std::rand() % 10;
        // } else if (i % 5 == 0) {
        //     // Small numbers, variance between 0.1 and 20
        //     real_lower = static_cast<double>(std::rand()) / RAND_MAX * 10 - 5;
        //     real_upper = real_lower + (static_cast<double>(std::rand()) / RAND_MAX * 19.9 + 0.1);
        //     imag_lower = static_cast<double>(std::rand()) / RAND_MAX * 10 - 5;
        //     imag_upper = imag_lower + (static_cast<double>(std::rand()) / RAND_MAX * 19.9 + 0.1);
        // } else if (i % 3 == 0) {
        //     // Very small variance (between 0.1 and 10)
        //     real_lower = static_cast<double>(std::rand()) / RAND_MAX * 10 - 5;
        //     real_upper = real_lower + (static_cast<double>(std::rand()) / RAND_MAX * 9.9 + 0.1);
        //     imag_lower = static_cast<double>(std::rand()) / RAND_MAX * 10 - 5;
        //     imag_upper = imag_lower + (static_cast<double>(std::rand()) / RAND_MAX * 9.9 + 0.1);
        // } else {
        //     // Regular variance within 1 to 100
        //     real_lower = std::rand() % 1001 - 500;
        //     real_upper = real_lower + 1 + std::rand() % 99;
        //     imag_lower = std::rand() % 1001 - 500;
        //     imag_upper = imag_lower + 1 + std::rand() % 99;
        // }

        // Loop through the different N values
        for (int N : N_values) {
            // Call the FFTW function with the generated parameters
            int count = compute_fftw_and_check_threshold(N, real_lower, real_upper, imag_lower, imag_upper, threshold);
            // Output the result to CSV file
            csv_file << N << "," << real_upper - real_lower << "," << imag_upper - imag_lower << "," << count << "," << iter << std::endl;
            // Output the result
            std::cout << "N = " << N << ", Count of magnitudes > " << threshold << ": " << count << ", iter: " << iter << std::endl;
        }
    }
    csv_file.close();

    return 0;
}

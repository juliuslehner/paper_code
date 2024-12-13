#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <sys/stat.h>
#include <getopt.h>
#include <unistd.h>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <stdexcept>
using namespace std;
namespace fs = filesystem;

// Function to create Directory recursively
void createDirectory(const std::string& path) {
    if (!fs::exists(path)) {
        try {
            if (!fs::create_directories(path)) {
                throw std::runtime_error("Failed to create directory: " + path);
            }
        } catch (const std::filesystem::filesystem_error& e) {
            throw std::runtime_error("Filesystem error: " + std::string(e.what()));
        }
    }
}


// Function to convert string to lowercase
string to_lower(const string& str) {
    string result = str;
    for (auto& c : result) {
        c = tolower(c);
    }
    return result;
}

int execute_command(const string& command) {
    // Convert std::string to C-style string and call system()
    int result = system(command.c_str());
    return result;
}

string double_to_string(double value, int precision = 1){
    stringstream ss;
    ss << fixed << setprecision(precision) << value;
    string result = ss.str();
    replace(result.begin(), result.end(), '.', '_');
    return result;
}

int main(int argc, char* argv[]) {
    string library, precision, benchmark_type, dimension, output_directory;
    int multi_thread = 0;
    double real_low = -10.0, imag_low = -10.0, real_high = 10.0, imag_high = 10.0;
    
    // Get Arguments
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "l:p:b:t:d:o:1:2:3:4:")))
    {
        switch (opt_c)
        {
        case 'l':
            library = to_lower(optarg);
            break;
        case 'p':
            precision = to_lower(optarg);
            break;
        case 'b':
            benchmark_type = to_lower(optarg);
            break;
        case 't':
            multi_thread = atoi(optarg);  // MULTI_THREAD=1
            break;
        case 'd':
            dimension = to_lower(optarg);
            break;
        case 'o':
            output_directory = strdup(optarg);
            break;
        case '1':
            real_low = atof(optarg);
            break;
        case '2':
            real_high = atof(optarg);
            break;  
        case '3':
            imag_low = atof(optarg);
            break;
        case '4':
            imag_high = atof(optarg);
            break; 
        case '?':
            printf("Unknown option %c\n", optopt);
            break;
        default:
            break;
        }
    }

    // Validate inputs and run corresponding benchmarks
    if (library.empty() || precision.empty() || benchmark_type.empty() || dimension.empty() || output_directory.empty()) {
        cerr << "Usage: ./benchmark_suite -l {cufft|fftw|cufftdx|vkfft} -d {1d|2d} "
                        "-p {half|single|double|longdouble} -b {performance|precision|batch_performance} "
                        "-o {output_directory} [-t MULTI_THREAD=1 (only for fftw)]" << endl;
        return 1;
    }

    // Check if MULTI_THREAD is used only with FFTW
    if (multi_thread && library != "fftw") {
        cerr << "MULTI_THREAD option is only available for FFTW." << endl;
        return 1;
    }

    // Validate dimension
    std:string dimension_flag;
    if(dimension == "1d"){
        dimension_flag = "_1d";
    } else if (dimension == "2d"){
        dimension_flag = "_2d";
    } else{
        cerr << "Invalid dimension. Use '1d' or '2d'." << endl;
        return 1;
    }

    // Validate output directory
    if (!output_directory.empty() && output_directory.back() != '/') {
        output_directory = output_directory + '/';
    }

    string command;
    if (library == "cufft") {
        if (precision != "half" && precision != "single" && precision != "double") {
            cerr << "Invalid precision for cufft. Use 'half', 'single', or 'double'." << endl;
            return 1;
        }
        // Compile and run benchmark
        command = "make -C cufft " + precision + "_" + benchmark_type + dimension_flag;
        execute_command(command);
        if (benchmark_type == "precision") {
            output_directory = output_directory + library + "/" + dimension + "/" + benchmark_type + "/";
            createDirectory(output_directory);
            command = string("cufft/cufft_precision_benchmark") + dimension_flag + " -1 " + to_string(real_low) + " -2 " + to_string(real_high) + " -3 " + to_string(imag_low) + " -4 " + to_string(imag_high) + " -o " + output_directory;
        } else if (benchmark_type == "performance") {
            output_directory = output_directory + library + "/" + dimension + "/" + benchmark_type + "/";
            createDirectory(output_directory);
            command = string("cufft/cufft_performance_benchmark") + dimension_flag + " -1 " + to_string(real_low) + " -2 " + to_string(real_high) + " -3 " + to_string(imag_low) + " -4 " + to_string(imag_high) + " -o " + output_directory;
        } else {
            cerr << "Invalid benchmark type for cufft. Use 'precision' or 'performance'." << endl;
            return 1;
        }
    } else if (library == "fftw") {
        if (precision != "single" && precision != "double" && precision != "longdouble") {
            cerr << "Invalid precision for fftw. Use 'single', 'double', or 'longdouble'." << endl;
            return 1;
        }
        // Compile and run benchmark
        string multi_thread_flag = multi_thread == 1 ? "mt=1" : "";
        if (benchmark_type == "precision") {
            command = "make -C fftw precision_" + precision + dimension_flag + " " + multi_thread_flag;
            execute_command(command);
            output_directory = output_directory + library + "/" + dimension + "/" + benchmark_type + "/";
            createDirectory(output_directory);
            command = "fftw/fftw_precision_" + precision + dimension_flag +  " -1 " + to_string(real_low) + " -2 " + to_string(real_high) + " -3 " + to_string(imag_low) + " -4 " + to_string(imag_high) + " -o " + output_directory;
        } else if (benchmark_type == "performance") {
            command = "make -C fftw performance_" + precision + dimension_flag + " " + multi_thread_flag;
            execute_command(command);
            output_directory = output_directory + library + "/" + dimension + "/" + benchmark_type + "/";
            createDirectory(output_directory);           
            // Files need to have the right permission, otherwise sudo environment cant open them
            stringstream filename;
            string threads = multi_thread != 0 ? "multithreads" : "singlethread";
            filename << "/u/home/lehnj/benchmarking_suite/NewResults/fftw/" + dimension + "/performance/"
                     << precision << "_prec_benchmark_imag" << double_to_string(imag_low,0) << "to" << double_to_string(imag_high,0) << "_real" << double_to_string(real_low,0) << "to" << double_to_string(real_high,0) << "_" << threads << ".csv";
            string filename_str = filename.str();
            // Create the file using ofstream
            ofstream file(filename_str);
            if (!file) {
                cerr << "Failed to create the file: " << filename_str << endl;
                return 1;
            }
            file.close();
            // Set file permissions to 666
            if (chmod(filename_str.c_str(), 0666) != 0) {
                cerr << "Failed to set file permissions to 666 for: " << filename_str << endl;
                return 1;
            }

            command = string("sudo fftw/fftw_performance") + " -1 " + to_string(real_low) + " -2 " + to_string(real_high) + " -3 " + to_string(imag_low) + " -4 " + to_string(imag_high) + " -o " + output_directory;
        } else {
            cerr << "Invalid benchmark type for fftw. Use 'precision' or 'performance'." << endl;
            return 1;
        }
    } else if (library == "cufftdx") {
        if (dimension == "1d"){
            if (precision != "half" && precision != "single" && precision != "double") {
                cerr << "Invalid precision for cufftdx. Use 'half', 'single', or 'double'." << endl;
                return 1;
            }
        } else if (dimension == "2d"){
            if (precision != "single" && precision != "double") {
                cerr << "Invalid precision for cufftdx in 2D. Use 'single', or 'double'." << endl;
                return 1;
            }
        }
        // Compile and run benchmark
        command = "make -C cufftdx " + precision + " " + benchmark_type + "_" + dimension;
        if (benchmark_type == "precision") {
            output_directory = output_directory + library + "/" + dimension + "/" + benchmark_type + "/";
            createDirectory(output_directory);
            execute_command(command);
            command = "cufftdx/block_fft_" + dimension + "_precision_many_" + precision + " -1 " + to_string(real_low) + " -2 " + to_string(real_high) + " -3 " + to_string(imag_low) + " -4 " + to_string(imag_high) + " -o " + output_directory;
        } else if (benchmark_type == "performance") {
            output_directory = output_directory + library + "/" + dimension + "/" + benchmark_type + "/";
            createDirectory(output_directory);
            execute_command(command);
            command = "cufftdx/block_fft_" + dimension + "_performance_many_" + precision + " -1 " + to_string(real_low) + " -2 " + to_string(real_high) + " -3 " + to_string(imag_low) + " -4 " + to_string(imag_high) + " -o " + output_directory;
        } else {
            cerr << "Invalid benchmark type for cufftdx. Use 'precision' or 'performance'." << endl;
            return 1;
        }
    } else if (library == "vkfft") {
        if (precision != "half" && precision != "single" && precision != "double") {
            cerr << "Invalid precision for vkfft. Use 'half', 'single', or 'double'." << endl;
            return 1;
        }
        
        // Run benchmark
        string vkfft_dimension = dimension == "1d" ? "1" : "2";
        if (benchmark_type == "precision") {
            // Compile
            command = "make -C vkfft " + precision;
            execute_command(command);
            output_directory = output_directory + library + "/" + dimension + "/" + benchmark_type + "/";
            createDirectory(output_directory);
            command = "vkfft/vkfft_precision_benchmark -1 " + to_string(real_low) + " -2 " + to_string(real_high) + " -3 " + to_string(imag_low) + " -4 " + to_string(imag_high) + " -d " + vkfft_dimension + " -o " + output_directory;
        } else if (benchmark_type == "performance") {
            // Compile
            command = "make -C vkfft " + precision;
            output_directory = output_directory + library + "/" + dimension + "/" + benchmark_type + "/";
            createDirectory(output_directory);
            command = "vkfft/vkfft_performance_benchmark -1 " + to_string(real_low) + " -2 " + to_string(real_high) + " -3 " + to_string(imag_low) + " -4 " + to_string(imag_high) + " -d " + vkfft_dimension + " -o " + output_directory;
        } else {
            cerr << "Invalid benchmark type for vkfft. Use 'precision' or 'performance'." << endl;
            return 1;
        }
    } else {
        cerr << "Invalid library. Use 'cufft', 'fftw', 'cufftdx', 'vkfft', 'cusfft', or 'tcfft'." << endl;
        return 1;
    }

    // Execute the command
    int result = execute_command(command);
    return result;
}
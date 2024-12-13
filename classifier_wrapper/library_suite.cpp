#include <iostream>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include <getopt.h>
#include <unistd.h>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iomanip>
using namespace std;

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
    string library, precision, dimension, input_file, output_file;
    int size_x = 0;
    int size_y = 1;
    double real_low = -10.0, imag_low = -10.0, real_high = 10.0, imag_high = 10.0;
    
    // Get Arguments
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "l:p:x:y:d:i:o:")))
    {
        switch (opt_c)
        {
        case 'l':
            library = to_lower(optarg);
            break;
        case 'p':
            precision = to_lower(optarg);
            break;
        case 'x':
            size_x = atoi(optarg); 
            break;
        case 'y':
            size_y = atoi(optarg); 
            break;
        case 'd':
            dimension = to_lower(optarg);
            break;
        case 'i':
            input_file = to_lower(optarg);
            break;
        case 'o':
            output_file = to_lower(optarg);
            break;
        case '?':
            printf("Unknown option %c\n", optopt);
            break;
        default:
            break;
        }
    }

    // Validate inputs and run corresponding benchmarks
    if (library.empty() || precision.empty() || size_x == 0 || input_file.empty() || output_file.empty() || dimension.empty()) {
        cerr << "Usage: ./library_suite -l {cufft|cufftdx|vkfft} -x {Nx} -y {Ny}"
                     "-p {half|single|double} -i {input_filepath} -o {output_filepath}"
                     " -d {1d|2d}" << endl;
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

    string command;
    if (library == "cufft") {
        if (precision != "half" && precision != "single" && precision != "double") {
            cerr << "Invalid precision for cufft. Use 'half', 'single', or 'double'." << endl;
            return 1;
        }
        // Compile and run benchmark
        command = "make -C cufft " + precision + "_" + "transform" + dimension_flag;
        execute_command(command);
        if (dimension  == "1d"){
            command = "cufft/cufft_transform_" + dimension + "_" + precision + " -i " + input_file + " -o " + output_file + " -s " + to_string(size_x);
        } else {
            command = "cufft/cufft_transform_" + dimension + "_" + precision + " -i " + input_file + " -o " + output_file + " -x " + to_string(size_x) + " -y " + to_string(size_y);
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
        command = "make -C cufftdx " + precision + " " + dimension;
        execute_command(command);
        if (dimension  == "1d"){
            command = "cufftdx/cufftdx_transform_" + dimension + "_" + precision + " -i " + input_file + " -o " + output_file + " -s " + to_string(size_x);
        } else {
            command = "cufftdx/cufftdx_transform_" + dimension + "_" + precision + " -i " + input_file + " -o " + output_file + " -x " + to_string(size_x) + " -y " + to_string(size_y);
        }

    } else if (library == "vkfft") {
        if (precision != "half" && precision != "single" && precision != "double") {
            cerr << "Invalid precision for vkfft. Use 'half', 'single', or 'double'." << endl;
            return 1;
        }
        // Compile
        command = "make -C vkfft " + precision + "_"+  dimension;
        execute_command(command);
        if (dimension  == "1d"){
            command = "vkfft/vkfft_transform_" + dimension + "_" + precision + " -i " + input_file + " -o " + output_file + " -s " + to_string(size_x);
        } else {
            command = "vkfft/vkfft_transform_" + dimension + "_" + precision + " -i " + input_file + " -o " + output_file + " -x " + to_string(size_x) + " -y " + to_string(size_y);
        }
    } else {
        cerr << "Invalid library. Use 'cufft', 'fftw', 'cufftdx', 'vkfft', 'cusfft', or 'tcfft'." << endl;
        return 1;
    }

    // Execute the command
    int result = execute_command(command);
    return result;
}
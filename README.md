# Code for the Paper

## Directory Structure

```
benchmark_suite/
├── benchmarking_suite.cpp      # Implementation of the benchmarking suite
├── cufft/                      # Performance and precison benchmarking code for cuFFT library
├── cufftdx/                    # Performance and precison code for cuFFTDx library
├── fftw/                       # Performance and precison code for FFTW library
├── vkfft/                      # Performance and precison code for VkFFT library
├── results/                    # Directory for storing benchmarking results

classifier_wrapper/
├── classifier_model.py         # Main classifier model implementation
├── cufft/                      # 1D and 2D transform code for cuFFT
├── cufftdx/                    # 1D and 2D transform code for cuFFTDx
├── vkfft/                      # 1D and 2D transform code for VkFFT
├── model_files/                # Trained machine learning models and scalers for each library and precision
└── numba_features.py           # Feature extraction using Numba to reduce overhead (compile first)

predictors/
├── classifier_validation.py        # Creates validation data
├── classifier_model_validation.py  # Validate classifier model accuracy
├── knn.py                          # KNN-based error predictor model
├── linear_regression_analysis.py   # Linear regression analysis of errors
├── neural_network.py               # Neural network-based error predictor model
├── randomforest.py                 # Random forest-based error predictor model
└── performance_dic.py              # Creates performance dictionary for cost function
```

## Key Directories

### `benchmark_suite`
This directory contains the benchmarking utilities for evaluating different FFT libraries in terms of performance, accuracy and energy consumption.

### `classifier_wrapper`
Contains the implementation for the classifier model that chooses best performing combination of library and precision type. The combination is then called and performs the transform.

### `predictors`
Source code for the error predictor models and classifier model validation.

## How to Use

### Benchmarking
1. Navigate to the `benchmark_suite` directory.
2. Compile the `benchmarking_suite.cpp` file.
3. Run the binary and provide needed parameters, as definded in the .cpp file.

### Classifier
1. Navigate to the `classifier_wrapper` directory.
2. Compile the `numba_features.py` file (used to reduce overhead of feature extraction)
3. Provide an input array as a binary file in the form of interleaved complex data in double precision type (FP64).
4. Provide the desired error threshold to `classifier_model.py` 
5. Run `classifier_model.py` which picks the highest performing combination of library and precision type and performs the transform. 
6. The results is saved as a binary file to the provided file path.


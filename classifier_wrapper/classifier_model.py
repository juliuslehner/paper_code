import numpy as np
import pandas as pd
import json
from joblib import load
import time
import os
import subprocess
import compiled_feature_extraction


# Define file paths
model_dir = "model_files"
binary_file = "complex_data_-10_10_-10_10_2097152.bin"
output_file = "complex_data_-10_10_-10_10_2097152_out.bin"
performance_dir = "performance_data_multidimensional.json"

# Define dimensionality
dimension = "1d"

# Create a dictionary to hold model paths
# Each library and precision has its own predictor model
model_paths = []

# Traverse the directories
for subdir, _, files in os.walk(model_dir):
    for file in files:
        if file.endswith(".joblib"):
            precision = "double" if "double" in file else "single" if "single" in file else "half"
            model_type = "knn" if precision == "double" else "randomforest"
            if model_type in subdir:
                model_paths.append(os.path.join(subdir, file))

# Load Models and Scalers
predictor_models = {}
model_scalers = {}

for path in model_paths:
    # Extract model type from the path
    model_type = 'knn' if 'knn' in path else 'randomforest'
    model_name = os.path.basename(path).replace('.joblib', '')
    # model_name = path.split('\\')[-1].replace('.joblib', '')
    # print(f"Processing model_name: {model_name}")
    library, dimension, precision = model_name.split('_')

    # Load the model
    model = load(path)
    predictor_models[(library, dimension, precision)] = model

    # Assign scaler only for KNN models
    if model_type == 'knn':
        scaler_path = f'model_files/scalers/{library}_{dimension}_{precision}.joblib'
        model_scalers[(library, dimension, precision)] = load(scaler_path)
    else:
        model_scalers[(library, dimension, precision)] = None  # No scaling for RF models

# Load Performance Timings
with open('performance_data_multidimensional.json', 'r') as f:
    performance_data = json.load(f)


def run_library_suite(library, precision, size_x, size_y, input_file, output_file, dimension):
    """
    Runs the external 'library_suite' executable with the given parameters.
    """
    # Build the command as a list
    cmd = [
        './library_suite',  # Path to the executable
        '-l', library,      # Library
        '-p', precision,    # Precision
        '-x', str(size_x),  # Size X
        '-y', str(size_y),  # Size Y
        '-i', input_file,   # Input file path
        '-o', output_file,  # Output file path
        '-d', dimension     # Dimension (1d or 2d)
    ]

    # Run the command and capture output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Function executed successfully.")
        print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred while running library_suite:")
        print("Return Code:", e.returncode)
        print("Error Output:\n", e.stderr)
        raise


# Function to read binary input file
# File should be interleaved complex numbers in double precision
def extract_features(input_file, sample_size=2000, seed=None):
    # Read in file
    size_bytes = os.path.getsize(input_file)
    num_elements = size_bytes // 8
    if num_elements % 2 != 0:
        raise ValueError("Data does not contain an even number of elements.")
    n = num_elements // 2
    data = np.memmap(input_file, dtype=np.float64, mode='r', shape=(num_elements,))
    real_data = data[0::2]
    imag_data = data[1::2]

    # Zero-pad if array size is not a power of two
    target_size = 1 if n == 0 else 2 ** ((n - 1).bit_length())
    if n != target_size:
        padded_size = target_size - n
        real_data = np.pad(real_data, (0, padded_size), mode='constant', constant_values=0)
        imag_data = np.pad(imag_data, (0, padded_size), mode='constant', constant_values=0)
    
    feature_start = time.perf_counter()
    # Sampling
    # if sample_size < n:
    #     sample_indices = np.linspace(0, n - 1, num=sample_size, dtype=np.int32)
    #     #sample_indices = np.random.choice(total_size, size=sample_size, replace=False)
    #     sampled_real = real_data[sample_indices]
    #     sampled_imag = imag_data[sample_indices]
    # else:
    #     # Use all data if sample_size >= total_size
    #     sampled_real = real_data
    #     sampled_imag = imag_data
    # if sample_size < n:
    #     # Efficient sampling using slicing for evenly spaced indices
    #     step = n // sample_size
    #     sampled_real = real_data[::step][:sample_size]
    #     sampled_imag = imag_data[::step][:sample_size]
    # else:
    #     # Use all data if sample_size >= total_size
    #     sampled_real = real_data
    #     sampled_imag = imag_data
    
    # Extract features
    
    var_real, var_imag, mean_magnitude = compiled_feature_extraction.compute_metrics(real_data, imag_data, sample_size)
    feature_end = time.perf_counter()
    feature_compute_time = feature_end - feature_start

    return feature_compute_time, target_size, var_real, var_imag, mean_magnitude


# Define Cost Function
def cost_function(library, precision, fft_size):
    timing_key = f"{library}_{precision}"
    if timing_key in performance_data and str(fft_size) in performance_data[timing_key]:
        # print(f"Library: {library}, Precision: {precision}, Performance: {performance_data[timing_key][str(fft_size)]}\n")
        return performance_data[timing_key][str(fft_size)]
    return float('inf')


# Define Classifier
def classify_fft_library(features, error_threshold, fft_size, predictor_models, model_scalers, cost_function):
    candidates = []
    if fft_size[1] > 1:
        dimension_input = "2d"
    else:
        dimension_input = "1d"
    for (library, dimension, precision), model in predictor_models.items():
        if dimension == dimension_input:
            if dimension == "2d":
                feature_names = ['variance_real', 'variance_imag', 'Magnitude', 'Nx', 'Ny']
            elif library == "vkfft":
                feature_names = ['variance_real', 'variance_imag', 'Magnitude', 'Nx']
            else:
                feature_names = ['variance_real', 'variance_imag', 'Magnitude', 'N']

            features_df = pd.DataFrame([features], columns=feature_names)
            # Scale features if the model is KNN
            if model_scalers[(library, dimension, precision)] is not None:
                features_df = pd.DataFrame(
                    model_scalers[(library, dimension, precision)].transform(features_df),
                    columns=feature_names
                )

            # Predict using the model
            prediction_start = time.perf_counter()
            predicted_error = model.predict(features_df)[0]
            prediction_end = time.perf_counter()
            prediction_time = prediction_end - prediction_start
            # print(f"Prediction time of {library} in {precision}: {prediction_time}\n")
            # print(f"Library: {library}, Precision: {precision}, Error: {predicted_error}\n")
            if predicted_error <= error_threshold:
                candidates.append((library, precision, predicted_error))

    if not candidates:
        return "No suitable library and precision found"

    # Select the best candidate based on cost function
    best_candidate = min(candidates, key=lambda x: cost_function(x[0], x[1], fft_size))
    return best_candidate[:2]  # Return (library, precision)


# Usage
# Extract features
if dimension.lower() == "1d":
    feature_time, size, var_real, var_imag, magnitude = extract_features(binary_file)
    fft_size = (size, 1)
elif dimension.lower() == "2d":
    nx = 512  # Example size for Nx
    ny = 512  # Example size for Ny
    feature_time, _, var_real, var_imag, magnitude = extract_features(binary_file)
    fft_size = (nx,ny)
else:
    raise ValueError("Unsupported Dimension.")

# Define feature names based on fft_size and library
if fft_size[1] > 1:
    feature_names = ['variance_real', 'variance_imag', 'Magnitude', 'Nx', 'Ny']
elif "vkfft" in predictor_models.keys():
    feature_names = ['variance_real', 'variance_imag', 'Magnitude', 'Nx']
else:
    feature_names = ['variance_real', 'variance_imag', 'Magnitude', 'N']

# Map 'fft_size' to the correct feature name
features = {
    'variance_real': var_real,
    'variance_imag': var_imag,
    'Magnitude': magnitude
}
if 'N' in feature_names:
    features['N'] = fft_size[0]
elif 'Nx' in feature_names:
    features['Nx'] = fft_size[0]
    if 'Ny' in feature_names:
        features['Ny'] = fft_size[1]

# Reorder features to match the expected order
try:
    ordered_features = [features[name] for name in feature_names]
except KeyError as e:
    raise KeyError(f"Missing key {e} in features dictionary. Available keys: {features.keys()}")

# Provide Error Threshold for absolute error
error_threshold = 1e-4

# Run the classifier
classifier_start = time.perf_counter()
predicted_library, predicted_precision = classify_fft_library(
    features=ordered_features,
    error_threshold=error_threshold,
    fft_size=fft_size,
    predictor_models=predictor_models,
    model_scalers=model_scalers,
    cost_function=cost_function
)
classifier_end = time.perf_counter()
classifier_time = classifier_end - classifier_start

# Print the times
print(f"Inference Time: {classifier_time:.4e}s")
print(f"Feature Extraction Time: {feature_time:.4e}s")

# Call correct library
print(f"Calling Library {predicted_library} in Precision {predicted_precision}")
run_library_suite(
    library=predicted_library,
    precision=predicted_precision,
    size_x=fft_size[0],
    size_y=fft_size[0],
    input_file=binary_file,
    output_file=output_file,
    dimension=dimension
)

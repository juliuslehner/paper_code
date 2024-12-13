import numpy as np
import pandas as pd
import json
from joblib import load
import time
import os
from sklearn.preprocessing import StandardScaler
# Define the base directory
base_dir = "model_files"

# Create a dictionary to hold the paths based on precision type
model_paths = []

# Traverse the directories
for subdir, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".joblib"):
            precision = "double" if "double" in file else "single" if "single" in file else "half"
            model_type = "knn" if precision == "double" else "randomforest"
            if model_type in subdir:
                model_paths.append(os.path.join(subdir, file))

print(model_paths)

# Step 2: Load Models and Scalers
predictor_models = {}
model_scalers = {}

for path in model_paths:
    # Extract model type from the path
    model_type = 'knn' if 'knn' in path else 'randomforest'
    model_name = path.split('\\')[-1].replace('.joblib', '')
    library, dimension, precision = model_name.split('_')

    # Load the model
    model = load(path)
    predictor_models[(library, dimension, precision)] = model

    # Assign a scaler only for KNN models
    if model_type == 'knn':
        scaler_path = f'model_files/scalers/{library}_{dimension}_{precision}.joblib'
        model_scalers[(library, dimension, precision)] = load(scaler_path)
    else:
        model_scalers[(library, dimension, precision)] = None  # No scaling for RF models

# Step 3: Load Performance Timings
with open('performance_data_multidimensional.json', 'r') as f:
    performance_data = json.load(f)

with open('final_results_test.json', 'r') as f:
    final_results = json.load(f)


# Step 4: Define Cost Function
def cost_function(library, precision, fft_size):
    timing_key = f"{library}_{precision}"
    if timing_key in performance_data and str(fft_size) in performance_data[timing_key]:
        return performance_data[timing_key][str(fft_size)]
    return float('inf')


# Step 5: Define Classifier
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
            predicted_error = model.predict(features_df)[0]
            if predicted_error <= error_threshold:
                candidates.append((library, precision, predicted_error))

    if not candidates:
        return "No suitable library and precision found"

    # Select the best candidate based on cost function
    best_candidate = min(candidates, key=lambda x: cost_function(x[0], x[1], fft_size))
    return best_candidate[:2]  # Return (library, precision)


# Step 6: Example Usage
# features = [0, 0, 0, 2048]
# error_threshold = 3e-5
# fft_size = 2048

# Baseline by averaging for each library and precision
global_mean_error_by_lib_prec = {
    lib_prec: np.mean([
        error for key, results in final_results.items()
        for lib_prec_key, result in results["results"].items()
        if lib_prec_key == lib_prec
        for error in result["errors"]
    ])
    for key, results in final_results.items()
    for lib_prec in results["results"].keys()
}

# Baseline by averaging for each size
baseline_errors_by_size = {}

for key, results in final_results.items():
    boundaries_fft_size = eval(key)  # Convert the string key back to tuple
    fft_size = boundaries_fft_size[1]

    for lib_prec, result in results["results"].items():
        if (lib_prec, fft_size) not in baseline_errors_by_size:
            baseline_errors_by_size[(lib_prec, fft_size)] = []

        baseline_errors_by_size[(lib_prec, fft_size)].extend(result["errors"])

# Compute the mean error for each (library, precision, FFT size)
baseline_avg_error_by_size = {
    key: np.mean(errors) for key, errors in baseline_errors_by_size.items()
}

# Debug: Print mean error for each library and precision
# print("Global Mean Error by Library and Precision:")
# for lib_prec, mean_error in global_mean_error_by_lib_prec.items():
#     print(f"{lib_prec}: {mean_error}")

# Step 7: Validation of the classifier
# thresholds = np.logspace(2, -14, num=34)  # From 1e1 to 1e-13
thresholds = np.concatenate([
    np.logspace(1, -6, num=16, endpoint=False),  # Thresholds from 10^1 to 10^-7 (exclusive of -7)
    np.logspace(-12, -14, num=6)  # Thresholds from 10^-11 to 10^-14
])
correct_predictions_classifier = 0
correct_predictions_baseline = 0
correct_predictions_baseline_size = 0
total_tests = 0
counter = 0

for key, value in final_results.items():
    print(f"Validating Boundary Nr. {counter}")
    boundaries_fft_size = eval(key)  # Convert the string key back to tuple
    fft_size = boundaries_fft_size[1]
    true_results = value["results"]
    features = value["features"]

    # Define the correct feature order
    if fft_size[1] > 1:
        feature_names = ['variance_real', 'variance_imag', 'Magnitude', 'Nx', 'Ny']
    elif "vkfft" in true_results.keys():
        feature_names = ['variance_real', 'variance_imag', 'Magnitude', 'Nx']
    else:
        feature_names = ['variance_real', 'variance_imag', 'Magnitude', 'N']

    # Map 'fft_size' to the correct feature name ('N' or 'Nx')
    if 'N' in feature_names:
        features['N'] = fft_size[0]
    elif 'Nx' in feature_names:
        features['Nx'] = fft_size[0]
        if 'Ny' in feature_names:
            features['Ny'] = fft_size[1]

    # Reorder features to match expected order
    try:
        ordered_features = [features[name] for name in feature_names]
    except KeyError as e:
        raise KeyError(f"Missing key {e} in features dictionary. Available keys: {features.keys()}")

    threshold = thresholds[counter % len(thresholds)]
    counter += 1

    # Find the true best (fastest) library/precision satisfying the threshold
    valid_candidates = [
        (lib_prec, cost_function(*lib_prec.split('_'), fft_size))
        for lib_prec, result in true_results.items()
        if max(result["errors"]) <= threshold
    ]

    if not valid_candidates:
        continue  # Skip if no valid candidates for this threshold

    true_best = min(valid_candidates, key=lambda x: x[1])[0]  # Fastest valid lib/prec

    # Run the classifier
    classifier_start = time.perf_counter()
    predicted_library, predicted_precision = classify_fft_library(
        features=ordered_features,
        error_threshold=threshold,
        fft_size=fft_size,
        predictor_models=predictor_models,
        model_scalers=model_scalers,
        cost_function=cost_function
    )
    classifier_end = time.perf_counter()
    classifier_time = classifier_end - classifier_start

    predicted_lib_prec = f"{predicted_library}_{predicted_precision}"

    # Baseline prediction using global mean errors
    valid_baseline_candidates = [
        (lib_prec, cost_function(*lib_prec.split('_'), fft_size))
        for lib_prec, avg_error in global_mean_error_by_lib_prec.items()
        if avg_error <= threshold
    ]

    if valid_baseline_candidates:
        baseline_best = min(valid_baseline_candidates, key=lambda x: x[1])[0]  # Fastest valid lib/prec
    else:
        baseline_best = None  # No valid baseline prediction

    # Baseline prediction using mean errors per size
    valid_baseline_candidates_size = [
        (lib_prec, cost_function(*lib_prec.split('_'), fft_size))
        for (lib_prec, size), avg_error in baseline_avg_error_by_size.items()
        if avg_error <= threshold and size == fft_size
    ]

    if valid_baseline_candidates_size:
        baseline_best_size = min(valid_baseline_candidates_size, key=lambda x: x[1])[0]  # Fastest valid lib/prec
    else:
        baseline_best_size = None  # No valid baseline prediction for size

    # Compare true best with classifier's prediction
    if predicted_lib_prec == true_best:
        correct_predictions_classifier += 1

    # Compare true best with global mean baseline's prediction
    if baseline_best == true_best:
        correct_predictions_baseline += 1

    # Compare true best with size mean baseline's prediction
    if baseline_best_size == true_best:
        correct_predictions_baseline_size += 1

    total_tests += 1

classifier_accuracy = correct_predictions_classifier / total_tests * 100
baseline_accuracy = correct_predictions_baseline / total_tests * 100
baseline_size_accuracy = correct_predictions_baseline_size / total_tests * 100

print(f"Classifier Accuracy: {classifier_accuracy:.2f}%")
print(f"Classifier Duration: {classifier_time:.4e} seconds")
print(f"Baseline Global Mean Accuracy: {baseline_accuracy:.2f}%")
print(f"Baseline Size Mean Accuracy: {baseline_accuracy:.2f}%")
# accuracy = correct_predictions / total_tests * 100
# print(f"Classifier Accuracy: {accuracy:.2f}%")

#result = classify_fft_library(features, error_threshold, fft_size, predictor_models, model_scalers, cost_function)
#print(f"Optimal FFT Library and Precision: {result}")

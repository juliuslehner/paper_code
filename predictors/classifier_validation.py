import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
import json


def convert_tuple_keys_to_string(d):
    """Recursively convert tuple keys in a nested dictionary to string keys for JSON compatibility."""
    if isinstance(d, dict):
        return {str(k): convert_tuple_keys_to_string(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_tuple_keys_to_string(i) for i in d]
    else:
        return d


# Step 1: Group files by input boundaries
def group_files_by_boundaries(base_directory):
    grouped_files = defaultdict(lambda: defaultdict(list))  # {boundaries: {library: [file1, file2,...]}}
    for dimension in os.listdir(base_directory):
        dimension_path = os.path.join(base_directory, dimension)
        for library in os.listdir(dimension_path):
            library_path = os.path.join(dimension_path, library)
            if os.path.isdir(library_path):
                for file_path in glob.glob(os.path.join(library_path, "*.csv")):
                    # Extract boundaries from filename (adjust based on actual format)
                    filename = os.path.basename(file_path)
                    boundaries = "_".join(filename.split("_")[3:9])  # Extract boundaries
                    grouped_files[boundaries][library].append(file_path)

    return grouped_files


# Step 2: Parse CSV files into a nested dictionary
def create_feature_error_dict(grouped_files):
    feature_error_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for boundaries, lib_files in grouped_files.items():
        for library, files in lib_files.items():
            for file_path in files:
                # print(f"Reading file: {file_path}")  # Debug
                precision = "double" if "double" in file_path else "single" if "single" in file_path else "half"
                df = pd.read_csv(file_path)
                if df.empty:
                    print(f"Empty file: {file_path}")
                    continue

                # Handle different size columns
                for _, row in df.iterrows():
                    nx = row.get('Nx') or row.get('N')
                    ny = row.get('Ny', 1)
                    fft_size = (int(nx), int(ny))
                    error_value = row['Absolute_Error']

                    # Validate data: skip rows with invalid or missing data
                    if pd.isna(fft_size) or pd.isna(error_value):
                        print(f"Invalid data in file {file_path}, row skipped: {row.to_dict()}")
                        continue
                    if error_value < 0:
                        print(f"Skipping row with negative error in file {file_path}: {row.to_dict()}")
                        continue

                    entry = {
                        'variance_real': row['variance_real'],
                        'variance_imag': row['variance_imag'],
                        'Magnitude': row['Magnitude'],
                        'error': error_value
                    }

                    # print(f"Adding entry for {boundaries}, {library}, {precision}, size {fft_size}: {entry}")
                    feature_error_data[boundaries][(library, precision)][fft_size].append(entry)
    return feature_error_data


# Step 3: Filter and Average Input Features
def filter_and_average_features(feature_error_data):
    filtered_data = defaultdict(dict)

    for boundaries, lib_prec_data in feature_error_data.items():
        # Separate all sizes into 1D and 2D categories
        all_sizes_1d = [set(k for k in features.keys() if k[1] == 1) for features in lib_prec_data.values()]
        all_sizes_2d = [set(k for k in features.keys() if k[1] > 1) for features in lib_prec_data.values()]

        # Compute common sizes for 1D and 2D
        common_sizes_1d = set.intersection(*all_sizes_1d) if all_sizes_1d else set()
        common_sizes_2d = set.intersection(*all_sizes_2d) if all_sizes_2d else set()

        # Combine common sizes
        common_sizes = common_sizes_1d.union(common_sizes_2d)
        # Find common FFT sizes across all (library, precision) combinations
        # all_sizes = [set(features.keys()) for features in lib_prec_data.values()]
        # common_sizes = set.intersection(*all_sizes)

        for fft_size in common_sizes:
            combined_features = []
            for (library, precision), features in lib_prec_data.items():
                for entry in features[fft_size]:
                    combined_features.append(entry)

            # Average features for this size
            avg_features = {
                'variance_real': np.mean([f['variance_real'] for f in combined_features]),
                'variance_imag': np.mean([f['variance_imag'] for f in combined_features]),
                'Magnitude': np.mean([f['Magnitude'] for f in combined_features])
            }

            # Keep the error separate for each library and precision
            errors = {
                f"{library}_{precision}": [entry['error'] for entry in lib_prec_data[(library, precision)][fft_size]]
                for (library, precision) in lib_prec_data
            }

            filtered_data[boundaries][fft_size] = {
                "features": avg_features,
                "errors": errors
            }

    return filtered_data


# Step 4: Combine with Speed and Error Data
def combine_with_speed_and_error_data(feature_error_data, performance_data):
    final_data = {}

    for boundaries, fft_size_data in feature_error_data.items():
        for fft_size, data in fft_size_data.items():
            avg_features = data["features"]
            error_entries = data["errors"]  # Correct this access properly

            # Combine speed and error for each library/precision
            combined_entries = {}
            for lib_prec, errors in error_entries.items():  # Iterate errors which are stored as dict items
                # Access speeds for current library/precision & FFT size
                # print(fft_size, type(fft_size))
                speed = performance_data.get(lib_prec, {}).get(str(fft_size), None)
                print(speed)
                combined_entries[lib_prec] = {
                    "errors": errors,  # Keep all the error data for library/precision
                    "speed": speed     # Speed matched correctly for FFT size
                }

            # Store the combined details
            final_data[(boundaries, fft_size)] = {
                "features": avg_features,  # averaged features.
                "results": combined_entries  # speed and error per library-precision
            }

    return final_data


# Usage:

# Step 1: Group files by boundaries
base_directory = r"C:\Users\juliu\Documents\Elektrotechnik-Studium\Bachelorarbeit\NewResults\classifier\classifier_validation"  # Update this to your actual path
grouped_files = group_files_by_boundaries(base_directory)

# Step 2: Create feature error dictionary
feature_error_dict = create_feature_error_dict(grouped_files)

#Step 3: Filter and average the features
filtered_feature_error_dict = filter_and_average_features(feature_error_dict)

# Step 3: Load performance data
with open("performance_data_multidimensional.json", "r") as f:
    performance_data = json.load(f)

# Step 4: Combine filtered feature/error data with speed and error per library/precision
final_combined_data = combine_with_speed_and_error_data(filtered_feature_error_dict, performance_data)

# Save the final results to JSON
with open("final_results_test.json", "w") as f:
    json.dump(convert_tuple_keys_to_string(final_combined_data), f, indent=4)

# Inspect some sample entries
for key, value in list(final_combined_data.items())[:3]:
    print(f"Boundaries & FFT Size: {key}")
    print(f"Features: {value['features']}")
    for lib_prec, result in value['results'].items():
        print(f"Library & Precision: {lib_prec}, Error: {result['errors']}, Speed: {result['speed']}")
    print("\n")

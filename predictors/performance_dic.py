import pandas as pd
import glob
import os
import json

# Define the path where all CSV files are stored
directory_path = r"C:\Users\juliu\Documents\Elektrotechnik-Studium\Bachelorarbeit\NewResults\classifier\performance"  # Replace with the actual path

# Create a nested dictionary to store performance data
performance_data = {}

# Iterate through each CSV file in the directory
for file_path in glob.glob(os.path.join(directory_path, "*.csv")):
    # Extract the library and precision from the filename
    file_name = os.path.basename(file_path)
    library, precision, dimension = file_name.replace(".csv", "").split("_")  # Example: cufft_1d_half.csv

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Combine library and precision into a single key
    lib_prec_key = f"{library}_{precision}"

    # Ensure the combined key exists in performance_data
    if lib_prec_key not in performance_data:
        performance_data[lib_prec_key] = {}

    # # Ensure the library key exists
    # if library not in performance_data:
    #     performance_data[library] = {}
    #
    # # Ensure the dimension key exists within the library
    # if dimension not in performance_data[library]:
    #     performance_data[library][dimension] = {}
    #
    # # Ensure the precision key exists within the dimension
    # if precision not in performance_data[library][dimension]:
    #     performance_data[library][dimension][precision] = {}

    # Populate the nested dictionary with FFT size (N) as the key and AverageTime[ms] as the value
    for _, row in df.iterrows():
        nx = row.get('N') or row.get('Nx')  # Safely handle either column
        ny = row.get('Ny', 1)
        try:
            nx = int(nx)
            ny = int(ny)
        except ValueError:
            print(f"Skipping row with invalid FFT size: Nx={nx}, Ny={ny}")
            continue
        fft_size = (nx, ny)
        fft_key = str(fft_size)
        average_time = row['AverageTime(s)']
        performance_data[lib_prec_key][fft_key] = average_time
        #performance_data[library][dimension][precision][fft_size] = average_time

# Print the nested dictionary to check the structure
print(performance_data)

with open('performance_data_multidimensional.json', 'w') as f:
    json.dump(performance_data, f, indent=4)

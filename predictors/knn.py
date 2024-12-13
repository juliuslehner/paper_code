import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import time

library = "cufft"
dimension = "1d"
precision = "half"
# Step 1: Set the path to the directory containing the CSV files
directory_path = rf"C:\Users\juliu\Documents\Elektrotechnik-Studium\Bachelorarbeit\NewResults\Classifier\{library}\{dimension}\normalised_{precision}"

# Step 2: Read all CSV files and calculate the simple average for each file
all_files = glob.glob(os.path.join(directory_path, "*.csv"))
data = []

for filename in all_files:
    # print(f"Processing file: {filename}")
    df = pd.read_csv(filename)
    if df.empty:
        continue  # Skip if the file is empty

    # Append each row to the data list
    data.append(df)

# Combine all the data into a single DataFrame
combined_df = pd.concat(data, ignore_index=True)

# Define predictors and target variable
if dimension == "2d":
    X = combined_df[['variance_real', 'variance_imag', 'Magnitude', 'Nx', 'Ny']]
elif library == "vkfft":
    X = combined_df[['variance_real', 'variance_imag', 'Magnitude', 'Nx']]
else:
    X = combined_df[['variance_real', 'variance_imag', 'Magnitude', 'N']]
# X = combined_df[['variance_real', 'variance_imag', 'N']]
y = combined_df['Absolute_Error']
# y_log = np.sqrt(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

non_zero_indices = y_test >= 1e-5
#print(y_test == 0)
# Filter X_test and y_test to keep only non-zero entries
X_test = X_test.loc[non_zero_indices]
y_test = y_test.loc[non_zero_indices]
# Data Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""
# Baseline Model: Predict Mean of Training Errors
baseline_prediction = np.mean(y_train)  # Mean of training errors
y_baseline_pred = np.full_like(y_test, baseline_prediction)

# Baseline Metrics
baseline_mse = mean_squared_error(y_test, y_baseline_pred)
baseline_r2 = r2_score(y_test, y_baseline_pred)

print(f"Baseline Model Mean Squared Error (MSE): {baseline_mse}")
print(f"Baseline Model R^2 Score: {baseline_r2}")

# Save Baseline Model (Mean Predictor) as a simple JSON
baseline_model_path = f'model_files/baseline/{library}_{dimension}_{precision}.json'
os.makedirs(os.path.dirname(baseline_model_path), exist_ok=True)

with open(baseline_model_path, 'w') as f:
    import json
    json.dump({'mean_error': baseline_prediction}, f)

"""
start_time = time.perf_counter()
# Define the KNN Regressor
knn = KNeighborsRegressor()

# Set up the hyperparameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'p': [1, 2],  # p=1 is Manhattan distance, p=2 is Euclidean distance
    'weights': ['uniform', 'distance'],  # uniform or distance-weighted
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [20, 30, 40, 50]  # Default is 30
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
end_time = time.perf_counter()

knn_time = end_time - start_time

# Print the best parameters found by GridSearchCV
print(f"Best Parameters: {grid_search.best_params_}")

# Use the best estimator to make predictions
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
#print(y_pred)
# Calculate the Mean Squared Error and R^2 Score
mse = mean_squared_error(y_test, y_pred)
mrse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save best model and scaler
#joblib.dump(best_knn, f'model_files/knn/{library}_{dimension}_{precision}.joblib')
#joblib.dump(scaler, f'model_files/scalers/{library}_{dimension}_{precision}.joblib')

print(f"KNN Regression Mean Root Squared Error: {mrse}")
print(f"KNN Regression MAPE: {mape}")
print(f"KNN Regression R^2 Score: {r2}")
print(f"Time for Training and Grid Search: {knn_time:.4} seconds")
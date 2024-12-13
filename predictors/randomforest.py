import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from joblib import dump
import matplotlib.pyplot as plt
import glob
import os
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

y = combined_df['Absolute_Error']

# Exclude NaN lines
nan_inf_indices = np.where(~np.isfinite(y))[0]
X = X.drop(nan_inf_indices).reset_index(drop=True)
y = y.drop(nan_inf_indices).reset_index(drop=True)

# Filter out rows where y < 0
valid_indices = (y >= 0)
X = X[valid_indices].reset_index(drop=True)
y = y[valid_indices].reset_index(drop=True)

# y = np.log1p(y)

# Split into Training, Validation and Holdout Sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

non_zero_indices = y_test.abs() >= 1e-8
#print(y_test == 0)
# Filter X_test and y_test to keep only non-zero entries
X_test = X_test.loc[non_zero_indices]
y_test = y_test.loc[non_zero_indices]

# Check for NaN in y
nan_indices = np.where(pd.isna(y))[0]
print(f"Indices with NaN in y: {nan_indices}")

# Step 3: Train the Random Forest Model
start_train = time.perf_counter()
# Initialize the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)
end_train = time.perf_counter()
training_time = end_train - start_train

# Step 4: Evaluate the Model
# Make predictions on the test set
y_val_pred = rf.predict(X_val)

# Calculate Mean Squared Error and R^2 Score
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Step 5: Feature Importance
# Get feature importances
importances = rf.feature_importances_
features = X.columns

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest')
plt.show()

# Step 6: (Optional) Hyperparameter Tuning using GridSearchCV
# Define the parameter grid
"""
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
"""
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

start_grid = time.perf_counter()
# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the model with the best parameters
grid_search.fit(X_train, y_train)
end_grid = time.perf_counter()
grid_time = end_grid - start_grid

# Get the best parameters
print("Best Parameters:", grid_search.best_params_)

# Re-train the model with the best parameters
best_rf = grid_search.best_estimator_

# Evaluate the best model
prediction_start = time.perf_counter()
y_test_pred = best_rf.predict(X_test)
prediction_end = time.perf_counter()
prediction_time = prediction_end - prediction_start
print(f"Prediction time: {prediction_time}")
# y_test_pred = np.expm1(y_test_pred)
# y_test = np.expm1(y_test)
mse_best = mean_squared_error(y_test, y_test_pred)
mrse_best = np.sqrt(mse_best)
mape_best = mean_absolute_percentage_error(y_test, y_test_pred)
r2_best = r2_score(y_test, y_test_pred)

# Evaluate Default model on same test
y_test_pred_def = rf.predict(X_test)
mse_best_def = mean_squared_error(y_test, y_test_pred_def)
mrse_best_def = np.sqrt(mse_best_def)
mape_best_def = mean_absolute_percentage_error(y_test, y_test_pred_def)
r2_best_def = r2_score(y_test, y_test_pred_def)

# Save best model
# dump(best_rf, f'model_files/randomforest/{library}_{dimension}_{precision}.joblib')

print(f"Best Model Mean Root Squared Error: {mrse_best}")
print(f"Best Model MAPE: {mape_best}")
print(f"Best Model R^2 Score: {r2_best}")
print(f"Training Time: {training_time:.4} seconds")
print(f"Grid Search Time: {grid_time:.4} seconds")
print(f"Total Time: {(grid_time + training_time):.4} seconds")

print(f"Default Model Mean Root Squared Error: {mrse_best_def}")
print(f"Default Model MAPE: {mape_best_def}")
print(f"Default Model R^2 Score: {r2_best_def}")

residuals = y_test - y_test_pred
plt.hist(residuals, bins=100)
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

plt.scatter(y_test_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()

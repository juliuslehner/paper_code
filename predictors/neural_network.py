import os
import pandas as pd
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Activation
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error


# Step 1: Set the path to the directory containing the CSV files
directory_path = r"C:\Users\juliu\Documents\Elektrotechnik-Studium\Bachelorarbeit\NewResults\classifier\cufft\1d\classifier_normalised"
#unbiased_path = r"C:\Users\juliu\Documents\Elektrotechnik-Studium\Bachelorarbeit\Results\cufft\parameter_find_single_small_50"

# Step 2: Read all CSV files and calculate the simple average for each file
all_files = glob.glob(os.path.join(directory_path, "*.csv"))
data = []

#unbiased_files = glob.glob(os.path.join(unbiased_path, "*.csv"))
#unbiased_data = []

for filename in all_files:
    df = pd.read_csv(filename)
    if df.empty:
        continue  # Skip if the file is empty

    # Append each row to the data list
    data.append(df)
'''
for filename in unbiased_files:
    df = pd.read_csv(filename)
    if df.empty:
        continue  # Skip if the file is empty

    # Append each row to the data list
    unbiased_data.append(df)
'''
# Combine all the data into a single DataFrame
combined_df = pd.concat(data, ignore_index=True)
#unbiased_df = pd.concat(unbiased_data, ignore_index=True)

# Define predictors and target variable
X = combined_df[['variance_real', 'variance_imag', 'Magnitude', 'N']]
y = combined_df['Absolute_Error']

#unbiased_X = unbiased_df[['variance_real', 'variance_imag', 'Magnitude', 'N']]
#unbiased_y = unbiased_df['Absolute_Error']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#X_test_scaled = scaler.fit_transform(unbiased_X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#X_train = X_scaled
#y_train = y
#X_test = X_test_scaled
#y_test = unbiased_y

non_zero_indices = y_test != 0
#print(len(y_test))
# Filter X_test and y_test to keep only non-zero entries
X_test = X_test[non_zero_indices]
y_test = y_test[non_zero_indices]
#print(len(y_test))
"""
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))  # To avoid overfitting
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))  # Output layer for regression
"""
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1]))
model.add(Activation('swish'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.25, verbose=1)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mrse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Neural Network MAPE: {mape}")
print(f"Neural Network Mean Root Squared Error: {mrse}")
print(f"Neural Network R^2 Score: {r2}")

# Optional: Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

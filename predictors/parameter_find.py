import os
import pandas as pd
import numpy as np
import glob
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Step 1: Set the path to the directory containing the CSV files
directory_path = r"C:\Users\juliu\Documents\Elektrotechnik-Studium\Bachelorarbeit\NewResults\Classifier\cufft\1d\double"  # Update this path

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

# Correlation Matrix
correlation_matrix = combined_df[
    ['variance_real', 'variance_imag', 'Magnitude', 'N', 'rate_of_change', 'Absolute_Error']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualizing Correlation Matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Linear Regression Analysis for each feature against Relative_Error
X = combined_df[['variance_real', 'variance_imag', 'Magnitude', 'N']]
y = combined_df['Absolute_Error']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant to the independent variables matrix (for the intercept)
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Fit the regression model
model = sm.OLS(y_train, X_train).fit()

# Print the regression results
print("Regression Analysis:")
print(model.summary())

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
mrse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Mean Root Squared Error on Test Data: {mrse}")
print(f"R-squared on Test Data: {r2}")
print(f"Mean Absolute Percentage Error on Test Data: {mape}")


# Optional: Visualize the relationships with scatter plots and regression lines
sns.pairplot(combined_df, x_vars=['variance_real', 'variance_imag', 'Magnitude'],
             y_vars='Absolute_Error', kind='reg')
plt.show()

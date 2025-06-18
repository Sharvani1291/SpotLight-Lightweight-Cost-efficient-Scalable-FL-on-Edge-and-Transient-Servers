import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load the CSV file
file_path = '500-entries.csv'
data = pd.read_csv(file_path)

# Convert the timestamp column to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Filter the data where layer is 'L2'
data = data[data['layer'] == 'L2']

# Sort the data by timestamp
data = data.sort_values(by='timestamp')

# Take only the first 500 rows
data = data.head(500)

# Prepare the data for training
X = data[['timestamp']]
y = data['aggregation_time']

# Convert timestamp to numerical values
X['timestamp'] = X['timestamp'].astype(int) / 10**9

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model with GPU support
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, tree_method='gpu_hist')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X_test['timestamp'], y_test, 'o', label='True values')
plt.plot(X_test['timestamp'], y_pred, 'x', label='Predicted values')
plt.xlabel('Timestamp')
plt.ylabel('Aggregation Time')
plt.title('True vs Predicted Aggregation Time')
plt.legend()
plt.savefig('xgboost_results.png')
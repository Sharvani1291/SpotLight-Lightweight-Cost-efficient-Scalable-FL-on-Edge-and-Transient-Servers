import numpy as np
from sklearn.linear_model import LinearRegression
#import matplotlib.pyplot as plt

# Sample data: replace with actual values if needed
clients = np.array([500, 1000, 1500,2000]).reshape(-1, 1)
times = np.array([1.94,2.99,4.79,9.79])

# Fit the model
model = LinearRegression()
model.fit(clients, times)

# Predict for 7000 to 10000 clients at 500 intervals
future_clients = np.arange(2500, 10001, 500).reshape(-1, 1)
predicted_times = model.predict(future_clients)

# Print results
print("Predicted Aggregation Times (FedAvg - Papaya - k=10):")
for c, t in zip(future_clients.ravel(), predicted_times):
    print(f"{c}\t{t:.2f}")

# # Optional: plot the regression
# plt.scatter(clients, times, label="Observed")
# plt.plot(future_clients, predicted_times, color="red", label="Projected")
# plt.xlabel("Number of Clients")
# plt.ylabel("Aggregation Time (s)")
# plt.title("FedAvg Aggregation Time Projection")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

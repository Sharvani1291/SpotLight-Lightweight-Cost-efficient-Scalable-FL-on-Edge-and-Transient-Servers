import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '500-entries.csv'
data = pd.read_csv(file_path)

# Convert the timestamp column to datetime
#data['timestamp'] = pd.to_datetime(data['timestamp'])
data=data.dropna()
data=data[data["layer"]=="L3"]
# Sort the data by timestamp
data = data.sort_values(by='timestamp')
data = data.head(500)

# Plot aggregation_time vs timestamp as a line chart
plt.figure(figsize=(10, 6))
plt.plot(data['timestamp'], data['aggregation_time'], marker='o', linestyle='-')
plt.xlabel('Timestamp')
plt.ylabel('Aggregation Time')
plt.title('Aggregation Time vs Timestamp L3')

# Format x-axis for better readability
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Reduce the number of x-axis ticks  # Format the date labels
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('aggregation_time_vs_timestamp-l3.png')

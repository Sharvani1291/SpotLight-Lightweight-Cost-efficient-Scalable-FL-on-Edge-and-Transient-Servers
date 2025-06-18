import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('500-entries.csv')

# Ensure aggregation_time is in float format
df["aggregation_time"] = df["aggregation_time"].astype(float)

# Create a grouping index for every 100 rows
df["group"] = np.floor(df.index / 100)

# Sum aggregation_time for every 100-row chunk
grouped_sums = df.groupby(["group", "layer"])["aggregation_time"].sum().reset_index()

# Compute the 90th and 25th percentiles for each layer
percentiles = grouped_sums.groupby("layer")["aggregation_time"].quantile([0.25, 0.90]).unstack()

print(percentiles)

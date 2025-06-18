import pandas as pd

# Read the CSV file
df = pd.read_csv('/home/cloudlabgpu1/grpc_spotlight/GRPC_emulated/data-extraction/all_entries.csv')

# Sum over the aggregation_time column
df["aggregation_time"] = pd.to_numeric(df["aggregation_time"])
ranges = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 10000),(10000, 15000), (15000, 21000), 
          (21000, 28000), (28000, 34000), (35000, 42000)]
for start, end in ranges:
    sum_aggregation_time = df.iloc[start:end]["aggregation_time"].sum()
    print(f"Rows {start + 1} to {end}: Sum = {sum_aggregation_time}")
import pandas as pd

# Load the CSV file
df = pd.read_csv("fedavg-entries.csv")

start = 0
batch_size = 100
batch_num = 1

while start < len(df):
    end = start + batch_size
    end = min(end, len(df))  # Ensure we don't go out of bounds

    # Compute sum for this batch
    group_sum = df.iloc[start:end]["aggregation_time"].sum()
    print(f"Batch {batch_num}: Sum of rows {start+1} to {end}: {group_sum}")

    # Update for next batch
    start = end
    batch_size += 100
    batch_num += 1

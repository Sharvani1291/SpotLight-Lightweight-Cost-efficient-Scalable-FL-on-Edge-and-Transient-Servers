import pandas as pd

df = pd.read_csv("fedadam-l2-replicas.csv")

def compute_stepped_chunk_sums(df, column="aggregation_time"):
    results = []
    start_idx = 0
    chunk_size = 50

    while start_idx < len(df):
        end_idx = min(start_idx + chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx]
        agg_sum = chunk[column].sum()

        results.append({
            "Start Row": start_idx,
            "End Row": end_idx - 1,
            "Chunk Size": chunk_size,
            "Sum Aggregation Time (s)": agg_sum
        })

        print(f"Chunk {start_idx} to {end_idx - 1} (size {chunk_size}) has sum: {agg_sum}")

        start_idx = end_idx
        chunk_size += 50

    return pd.DataFrame(results)

compute_stepped_chunk_sums(df, column="aggregation_time")

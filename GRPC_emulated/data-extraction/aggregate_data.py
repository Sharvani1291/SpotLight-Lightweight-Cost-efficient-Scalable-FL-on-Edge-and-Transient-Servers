#!/usr/bin/env python3
# aggregate_times_by_chunks.py

import pandas as pd
import itertools

# ---- 1. Load the raw CSV -------------------------------------------------
INPUT_CSV  = "fedprox-entries.csv"          # change if your filename differs
OUTPUT_CSV = "aggregated_by_chunks.csv"

df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])

# ---- 2. Define the chunk sizes ------------------------------------------
# 50, 100, 150, …, 1000
chunk_sizes = list(range(50, 1001, 50))

results = []

start_idx = 0
for size in chunk_sizes:
    end_idx = start_idx + size
    chunk   = df.iloc[start_idx:end_idx]

    # stop if we’ve run out of rows
    if chunk.empty:
        break

    agg_sum  = chunk["aggregation_time"].sum()
    row_span = f"{start_idx}–{end_idx-1}"   # purely cosmetic

    results.append(
        {"chunk_size": size,
         "row_range": row_span,
         "sum_aggregation_time": agg_sum}
    )

    start_idx = end_idx

# ---- 3. Save / show results ---------------------------------------------
agg_df = pd.DataFrame(results)
agg_df.to_csv(OUTPUT_CSV, index=False)

print("\nAggregated sums by chunk size")
print(agg_df)

print(f"\n✓ Wrote {OUTPUT_CSV}")

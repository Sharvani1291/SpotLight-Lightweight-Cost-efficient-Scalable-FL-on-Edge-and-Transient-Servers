import pandas as pd
import numpy as np
import sys

CSV_PATH = "client-data.csv"          # adjust if needed
COL_NAME = "Total Training Time (seconds)"

# ---------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Force numeric; non-parsable entries become NaN
df[COL_NAME] = pd.to_numeric(df[COL_NAME], errors="coerce")

# Drop NaNs and warn
before = len(df)
df = df.dropna(subset=[COL_NAME])
dropped = before - len(df)
if dropped:
    print(f"[WARN] Dropped {dropped} non-numeric rows", file=sys.stderr)

# Bail out if nothing left
if df.empty:
    sys.exit("No valid numeric data – cannot build CDF.")

times = df[COL_NAME].values

# ---------------------------------------------------------------
# 2. Build empirical CDF
# ---------------------------------------------------------------
times_sorted = np.sort(times)
n            = len(times_sorted)
cdf_values   = np.arange(1, n + 1) / n

# ---------------------------------------------------------------
# 3. Pick random p in [0.90, 0.95]  and invert
# ---------------------------------------------------------------
p    = np.random.uniform(0.90, 0.95)
idx  = np.searchsorted(cdf_values, p, side="left")
value = times_sorted[min(idx, n - 1)]

print(f"Random p : {p:.4f}  (≈ {p*100:.2f}-th percentile)")
print(f"T*       : {value:.2f} seconds")

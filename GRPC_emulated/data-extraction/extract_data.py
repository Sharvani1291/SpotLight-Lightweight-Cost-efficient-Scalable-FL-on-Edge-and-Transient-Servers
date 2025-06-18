from pymongo import MongoClient
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)

# Connect to MongoDB
client = MongoClient("mongodb://clUser:CloudLab@172.22.85.17:27017", serverSelectionTimeoutMS=60000)
db = client["emulated-spotlight"]
collection = db["emulated-models"]

logging.info("Fetching and sorting all entries from MongoDB…")
try:
    # --- 2. Query --------------------------------------------------------
    #   • filter: layer == "L2"
    #   • projection: timestamp and aggregation_time only
    #   • sort: by timestamp ascending
    cursor = (
        collection
        .find(
            {"layer": "L2"},                       # filter
            {"_id": 0, "timestamp": 1, "aggregation_time": 1,"LayerId": 1}  # projection
        )
        .sort("timestamp", 1)                      # ascending
    )

    # --- 3. Cursor → DataFrame ------------------------------------------
    df = pd.DataFrame(list(cursor))

    # --- 4. Write CSV ----------------------------------------------------
    csv_file_path = "fedadam-l2-replicas.csv"
    df.to_csv(csv_file_path, index=False)

    logging.info(f"✓ Entries saved to {csv_file_path}")

except Exception as e:
    logging.error(f"Error fetching or processing entries: {e}")
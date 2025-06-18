from celery import Celery
from pymongo import MongoClient
import logging
from datetime import datetime
from bson.objectid import ObjectId

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery application
app = Celery('celery_tasks', backend='redis://127.0.0.1:6381/0', broker='redis://127.0.0.1:6381/0')
logger.info(f"Registered tasks: {app.tasks.keys()}")
app.autodiscover_tasks(['celery_tasks'])

# Function to store aggregated weights in MongoDB
def store_aggregated_weights(model_weights, model_name):
    SAMPLE_SIZE = 5
    LOG_WEIGHTS = True  # Toggle to skip logging weights

    try:
        logger.info(f"Invoked store_aggregated_weights with model_name: {model_name}")
        logger.info(f"Type of model_weights: {type(model_weights)}")
        custom_id = str(ObjectId())
        logger.info(f"Generated custom ID: {custom_id}")

        # Connect to MongoDB
        mongo_client = MongoClient("mongodb://172.22.85.17:27017/")
        db = mongo_client["model_updates"]

        # Define the collection name
        collection_name = f"{model_name}_weights"
        collection = db[collection_name]
        logger.info(f"Using collection: {collection_name}")

        # Insert model weights into MongoDB
        logger.info("Attempting to insert model weights into MongoDB...")
        result = collection.insert_one({"_id": custom_id, "model_weights": model_weights})

        logger.info(f"Stored aggregated weights successfully. Inserted ID: {result.inserted_id}")
        return custom_id
    except Exception as e:
        logger.error(f"Error inserting model weights: {e}", exc_info=True)
        return f"Error: {e}"

# Celery task to store aggregated weights
@app.task
def store_aggregated_weights_task(model_weights, model_name):
    if model_weights is None or model_name is None:
        raise ValueError("Both model_weights and model_name must be provided.")
    try:
        logger.info("store_aggregated_weights_task started")
        logger.info(f"Invoked store_aggregated_weights_task with model_name: {model_name}")
        
        # Store the aggregated weights using helper function
        result = store_aggregated_weights(model_weights, model_name)
        return result
    except Exception as e:
        logger.error(f"Error in store_aggregated_weights_task: {e}", exc_info=True)
        raise

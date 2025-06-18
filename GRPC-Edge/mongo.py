from pymongo import MongoClient

try:
    mongo_client = MongoClient("mongodb://clUser:CloudLab@172.22.85.17:27017/")
    db = mongo_client["model_updates"]
    print("Databases:", mongo_client.list_database_names())
    print("Collections in 'model_updates':", db.list_collection_names())
except Exception as e:
    print("MongoDB Connection Error:", e)

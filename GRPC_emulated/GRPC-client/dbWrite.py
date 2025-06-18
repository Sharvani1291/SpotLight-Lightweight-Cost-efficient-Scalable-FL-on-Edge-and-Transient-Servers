from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
#logging.basicConfig(filename='db.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',mode='w')

#loads the env variables
class writeToDb():
    def __init__(self):
        load_dotenv()
        self.uri=os.getenv("MONGO_URI")
        self.db=os.getenv("MONGO_DB")
        self.collection=os.getenv("MONGO_COLLECTION")
        self.client = AsyncIOMotorClient(self.uri, maxPoolSize=20, minPoolSize=10)
        # Select the database and collection
        self.db = self.client[self.db]
        self.collection = self.db[self.collection]
        self.collection.create_index("timestamp")
    
    async def insert(self,payload):
        try:
        # Connect to the MongoDB cluster
            data={
                "timestamp": datetime.now(),
                "private-ip": payload
            }
            result=await self.collection.insert_one(data)
            logging.info(f"Model sent successfully. Document ID: {result.inserted_id}")
        except Exception as e:
            logging.error(f"Error fetching the latest weights: {e}")
#!/usr/bin/python3

import logging
from pymongo import MongoClient
from datetime import datetime

# Configure the logging
logging.basicConfig(level=logging.INFO)

# Create a file handler
file_handler = logging.FileHandler('connection.log')
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger().addHandler(file_handler)

class LifeLogger:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_uri = "mongodb://172.22.85.17:27017"
        self.username = "root"
        self.password = "root"
        self.db_name = "spot_check"
        self.collection_name = "checkLife"
        self.client = None
        self.db = None
        self.collection = None

    def connect_to_db(self):
        try:
            self.logger.info(f"Attempting to connect to MongoDB at {self.db_uri}")
            self.client = MongoClient(self.db_uri, username=self.username, password=self.password)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.logger.info("Connection to MongoDB established")
        except Exception as e:
            self.logger.error(f"Error connecting to MongoDB: {e}")

    def log_status(self, status,layer):
        try:
            if self.collection is None:
                self.connect_to_db()
            current_time = datetime.now().isoformat()
            #{time:current_time,status:status(0,1),layer:layer2/3}
            document = {"time": current_time, "status": status,layer:layer}
            self.logger.info(f"Inserting document into collection {self.collection_name}: {document}")
            self.collection.insert_one(document)
            self.logger.info("Document inserted successfully")
        except Exception as e:
            self.logger.error(f"Error inserting document: {e}")



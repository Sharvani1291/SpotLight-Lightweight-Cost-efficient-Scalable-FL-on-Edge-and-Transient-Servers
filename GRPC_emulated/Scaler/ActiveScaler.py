import os
import motor.motor_asyncio
from dotenv import load_dotenv
import asyncio
# Load environment variables
load_dotenv()

#I'm using the 25th percentiles here for lower threshold and 90th percentile for upper threshold in l2 and l3
class ActiveScaler:
    def __init__(self,ewma_alpha=0.3, scale_up_threshold=0.50, scale_down_threshold=0.262):
        self.uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGO_DB_EMULATED")
        self.collection_name = os.getenv("MONGO_COLLECTION_EMULATED")

        # Initialize Motor client with connection pooling
        self.client = motor.motor_asyncio.AsyncIOMotorClient(
            self.uri,
            maxPoolSize=10,  # Set maximum connections in the pool
            minPoolSize=5    # Set minimum connections in the pool
        )

        # Get database and collection
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.ewma_alpha = ewma_alpha
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.ewma_value = None  # Initial EWMA value


    async def fetch_latest_aggregation_times(self, limit: int = 100):
        layer = "L2"
        """Fetch the latest `limit` aggregation_time values for a given layer (L2 or L3)."""
        cursor = self.collection.find(
            {"layer": layer},
            {"aggregation_time": 1, "_id": 0}
        ).sort("_id", -1).limit(limit)
        
        return [doc["aggregation_time"] async for doc in cursor]
    
    def update_ewma(self, new_value: float):
        """Update EWMA with the new aggregation time."""
        if self.ewma_value is None:
            self.ewma_value = new_value  # Initialize EWMA with first value
        else:
            self.ewma_value = self.ewma_alpha * new_value + (1 - self.ewma_alpha) * self.ewma_value  # EWMA formula
        return self.ewma_value

    
    async def scale_decider(self, scale_factor: float,current_replicas: int):
        replica= 0
        #get the latest aggregation times
        aggregation_times = await self.fetch_latest_aggregation_times()
        
        #calculate the ewma value
        if not aggregation_times:
            print("No aggregation times found.")
            return None
        
        #summing over 100 rows
        summed_time=sum(aggregation_times)
        #latest_time=aggregation_times[0]
        ewma_value = self.update_ewma(summed_time)
        print(f"Latest aggregation time: {summed_time}, EWMA: {ewma_value}")
        #scale up or down based on the ewma value
        if ewma_value > self.scale_up_threshold:
            print("Scaling up...")
            #scale up
            replica = current_replicas * scale_factor
            print(f"New replicas: {replica}")
        elif ewma_value < self.scale_down_threshold:
            print("Scaling down...")
            #scale down
            replica = round(current_replicas/scale_factor)
            print(f"Old replicas: {replica}")
        else:
            print("No scaling needed.")
        return replica
    
    
async def main():
    scaler=ActiveScaler()
    scale_factor=2.0
    current_replicas=10
    while True:
        new_replica=await scaler.scale_decider(scale_factor,current_replicas)
        current_replicas=new_replica
        await asyncio.sleep(5)
    
if __name__ == "__main__":
    asyncio.run(main())

     #need to add the active scaler here and k8s method to scale up and down
     #the active scale user is ewma to scale up and down

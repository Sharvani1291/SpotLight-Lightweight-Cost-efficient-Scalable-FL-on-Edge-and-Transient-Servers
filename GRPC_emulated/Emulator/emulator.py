from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
import pandas as pd
import argparse
import time
from kubernetes import client, config
from datetime import datetime, timezone
import asyncio
import random
import logging
import time
import zmq

#need to integrate zermoq here to share the pods that are going to be deleted wit the RM so that the RM can update the route weights and stop sending it traffic
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='Emulator.log', filemode='a')
class Emulator:
    def __init__(self):
        #loads the env variables
        load_dotenv()
        self.uri=os.getenv("MONGO_URI")
        self.db=os.getenv("MONGO_DB")
        self.collection=os.getenv("MONGO_COLLECTION")
        self.client = AsyncIOMotorClient(self.uri, maxPoolSize=20, minPoolSize=10)
        # Select the database and collection
        self.db = self.client[self.db]
        self.collection = self.db[self.collection]
        self.collection.create_index("timestamp")
    
    def readTrace(self,dataset,vm,zone):
        #get the dataset path 
        df = pd.read_csv(f'{os.getcwd()}/{dataset}/{zone}_{vm}_cdf.csv')
        cdf_value=random.uniform(0.7,0.9)
        logging.info(f"cdf_value: {cdf_value}")
        filtered_df = df[df['CDF'] >= cdf_value]
    
        # Get the deflection lifetime from the first row of the filtered dataframe
        deflection_lifetime = filtered_df.iloc[0]['VM_Lifetime_seconds']
        
        # Get the maximum lifetime where CDF is 1
        max_lifetime = df[df['CDF'] == 1]['VM_Lifetime_seconds'].values[0]
        logging.info(f"deflection_lifetime,max_lifetime: {deflection_lifetime},{max_lifetime}")
        return deflection_lifetime, max_lifetime
    async def delete_pods_with_peak_lifetime(self,dataset,vm,zone):
        while True:
            peak_lifetime,max_lifetime = self.readTrace(dataset,vm,zone)
        # Load the Kubernetes configuration
            cutoff_lifetime=random.randint(peak_lifetime,max_lifetime)
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            kubeconfig_path = os.path.join(script_dir, "kubeconfig.yaml")

            # Load Kubernetes config explicitly
            config.load_kube_config(config_file=kubeconfig_path)

            # Create a Kubernetes API client
            api = client.CoreV1Api()
            
            #Before starting the deletion process, write to the MongoDB collection
            await self.write_to_mongo("False","l2-l3 active")
            logging.info("Set the reset the status to active")
            # Get all pods in the namespace
            pods = api.list_namespaced_pod(namespace='default')
            #logging.info(f"Pods in the namespace: ", [pod.metadata.name for pod in pods.items])
            #print("Peak lifetime: ", peak_lifetime)
            
            
            # Iterate over the pods
            for pod in pods.items:
                # Get the pod creation timestamp
                creation_timestamp = pod.metadata.creation_timestamp.replace(tzinfo=timezone.utc)

                # Calculate the pod lifetime in seconds
                lifetime = (datetime.now(timezone.utc) - creation_timestamp).total_seconds()
                #logging.info(f"Pod name: ", {pod.metadata.name})
                #print("Pod name: ", pod.metadata.name)
                print(f"Pod lifetime: ", lifetime)
                
                # Check if the pod lifetime reaches the peak lifetime
                if lifetime >= cutoff_lifetime or lifetime == max_lifetime:
                    

                    # Delete the pod
                    if pod.metadata.name.startswith("l2"):
                        await self.write_to_mongo("True","l2")
                    elif pod.metadata.name.startswith("l3"):
                        await self.write_to_mongo("True","l3")
                    print(f"Deleting pod with name: ", pod.metadata.name)
                    api.delete_namespaced_pod(name=pod.metadata.name, namespace='default')
                    # Wait for 5 seconds before deleting the next pod
                    time.sleep(10)
                else:
                    logging.info(f"No pods to be deleted")
                
            logging.info("Pods with lifetime >= peak lifetime deleted.")
    async def write_to_mongo(self,state,layer):
        # Create a dictionary with the data
        data = {
            "timestamp": datetime.now(),
            "Layer": layer,
            "Kill_signal": state,
            
        }
        # Insert the data into the collection
        result = await self.collection.insert_one(data)
        print(f"Data inserted with id: {result.inserted_id}")
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add arguments to the parser
    parser.add_argument('--dataset', type=str, help='New-Dataset or Old-Dataset')
    parser.add_argument('--vm', type=str, help='The vm name')
    parser.add_argument('--zone', type=str, help='The zone name')

    # Parse the arguments
    args = parser.parse_args()
    
    # Create an instance of the Emulator class
    emulator = Emulator()
    
    # Read the trace and get the deflection lifetime and max lifetime
    #deflection_lifetime, max_lifetime = emulator.readTrace(args.dataset, args.vm, args.zone)
    
    # Delete the pods with peak lifetime
    loop = asyncio.get_event_loop()
    loop.run_until_complete(emulator.delete_pods_with_peak_lifetime(args.dataset, args.vm, args.zone))
    
    # Adding this for testing
    
    # loop.run_until_complete(emulator.write_to_mongo("True"))
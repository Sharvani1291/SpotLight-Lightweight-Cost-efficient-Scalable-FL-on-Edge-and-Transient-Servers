import os
import asyncio
import aiohttp
import json
from aiohttp import web
import logging
from datetime import datetime
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient


#This class gets the weights from the other module and pass it to the GossipNode class

# class getWeights:
#     def __init__(self):
#         self.logger = logging.getLogger("weights")
#         self.logger.setLevel(logging.INFO)
#         file_handler = logging.FileHandler("gossip_node.log")
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#         file_handler.setFormatter(formatter)
#         self.logger.addHandler(file_handler)
#         self.weights = None
    
#     def shareWeight(self, weights):
        
#         self.logger.info(f"Running shareWeight method from the main class")
#         self.weights = weights
    
#     def storeWeight(self):
#         return self.weights
    

class GossipNode:
    def __init__(self, node_name,base_name="gossip-node", port=8000, max_nodes=3):
        self.node_name = node_name
        self.base_name = base_name
        self.port = port
        self.max_nodes = max_nodes
        self.peers = set()
        self.session = None  # Initialize session as None
        self.alpha = 0.8  # Example value for EWMA, adjust as needed
        self.agg_weights= np.array([])  # Initialize as an empty array
        self.fedBuff_weights=None
        #self.weight_source = getWeights()

        self.logger = logging.getLogger(self.node_name)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler("gossip_node.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        #For storage of weights

        # # Database setup
        self.client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
        self.db_name = os.getenv("MONGO_DB")
        self.collection_name = os.getenv("MONGO_COLLECTION")
        
        #Create the time index for easy retrieval of weights
        asyncio.create_task(self.create_timestamp_index())
        
        # self.db = self.client[db_name]
        # self.weights_collection = self.db["weights"]
    async def create_timestamp_index(self):
        """Create an index on the 'timestamp' field of the collection."""
        if self.db_name and self.collection_name:
            db = self.client[self.db_name]
            collection = db[self.collection_name]  # Retrieve the actual collection object
            await collection.create_index([('timestamp', 1)], unique=False)
            self.logger.info("Timestamp index created")
        else:
            self.logger.warning("Database name or collection name not set")


    async def start_session(self):
        """Initialize the session when the app starts"""
        try:
            self.session = aiohttp.ClientSession()
            logging.info("Session started")
        except Exception as e:
            logging.error(f"Error starting session: {e}")
            

    # async def close_session(self):
    #     """Close the session properly on shutdown"""
    #     if self.session:
    #         await self.session.close()
    #     self.client.close()
    
    #Write the weights to the database
    async def write_db(self,agg_weights):
        """Write the weights to the MongoDB database"""
        if self.db_name and self.collection_name:
            db = self.client[self.db_name]
            collection = db[self.collection_name]
            document = {"weights": agg_weights, "timestamp": datetime.now()}
            await collection.insert_one(document)
            self.logger.info("Weights written to database")
        else:
            self.logger.warning("Database name or collection name not set")
            
            
    async def periodic_task(self, interval_seconds):
        """Runs the update_weights method periodically every `interval_seconds`."""
        while True:
            # Run the update_weights method or any other periodic task
            agg_weights_list= self.agg_weights.tolist() if hasattr(self.agg_weights, 'tolist') else self.agg_weights
            await self.write_db(agg_weights_list)
            logging.info(f"Periodic task executed every {interval_seconds} seconds.")
            
            # Sleep for the specified interval
            await asyncio.sleep(interval_seconds)

    
    #Register the peer to the node
    async def register_peer(self, request):
        data = await request.json()
        new_peer = data.get("peer")
        if new_peer and new_peer not in self.peers:
            self.peers.add(new_peer)
            self.logger.info(f"Registered new peer: {new_peer}")
        return web.json_response({"status": "ok", "peers": list(self.peers)})
    
    
    async def push_weights(self,fedBuff_weights, peer):
        current_time = datetime.now().timestamp() # Get current time in ISO format
        """Push the current weights to the specified peer"""
        logging.info(f"Pushing weights to {peer}")
        try:
            async with self.session.post(f"http://{peer}/push_weights", json={"weights": fedBuff_weights,"timestamp":current_time}) as resp:
                if await resp.status == 200:
                    self.logger.info(f"Weights pushed successfully to {peer}")
                else:
                    self.logger.warning(f"Failed to push weights to {peer}: {resp.status}")
        except Exception as e:
            self.logger.warning(f"Error pushing weights to {peer}: {e}")
            
    async def recv_weights(self, request):
        """Receive weights from a peer and store them in global variable"""
        data = await request.json()
        weights = data.get("weights")
        if weights is not None:
            self.fedBuff_weights = weights
            self.logger.info("Received weights from push peer")
            
    async def aggregatorWeights(self, request):
        """Return the current weights to the requester"""
        data = await request.json()
        weights= data.get("weights")
        if weights is not None:
            self.fedBuff_weights = weights
            await self.update_weights(weights)
            self.logger.info("Received weights from peer from aggregator method")
            return web.json_response({"status": "ok"})
        
        
    

    # #Method mainly to synchronize the weights with the peers, it will be called when I trigger it from the server side
    async def update_weights(self,weights):
        logging.info(f"Starting the weight update process")
        # if fedbuff_weights is None:
        #     self.logger.warning("No weights available to update")
            #return
        self.fedBuff_weights = weights
        fedBuff_weights = list(self.fedBuff_weights)
        #self.logger.info(f"Current weights set to: {self.fedBuff_weights}")   
        try:
                # Get the weights from the other module
                
            
                """Update the weights using Exponential Weighted Moving Average (EWMA) and push them to peers"""
                #self.fedBuff_weights = weights  # Store current weights
                current_time = datetime.now().timestamp()# Get current time in ISO format

                for peer in list(self.peers):
                    try:
                        async with self.session.get(f"http://{peer}/pull_weights") as resp:
                            logging.info(f"Pulling weights from peer: {peer}")
                            if resp.status == 200:
                                data = await resp.json()

                                # If no valid weights received, push current weights instead
                                if not data or "weights" not in data:
                                    self.logger.warning(f"No valid data received from {peer}")
                                    self.logger.info("Pushing current weights to peer")
                                    await self.push_weights(self.fedBuff_weights, peer)
                                    continue

                                peer_weights = data["weights"]
                                peer_time = data["timestamp"] if "timestamp" in data else None

                                # Ensure peer_time is valid
                                if not peer_time:
                                    self.logger.warning(f"Invalid timestamp from {peer}, ignoring its weights")
                                    continue

                                # Push current weights to peer
                                #await self.push_weights(self.fedBuff_weights, peer)

                                # Update weights using EWMA if peer's weights are older
                                logging.info(f"Peer {peer}  sent data with timestamp {peer_time}")
                                logging.info(f"Current time is {current_time}")
                                if float(peer_time) < float(current_time):
                                    fedBuff_weights = np.array(fedBuff_weights)
                                    peer_weights = np.array(peer_weights)
                                    logging.info(f"Converted weights to NumPy arrays for arithmetic operations with alpha {self.alpha}")
                                    self.agg_weights = self.alpha * fedBuff_weights + (1 - self.alpha) * peer_weights
                                    logging.info(f"Updated weights using {peer}'s data with EWMA with alpha {self.alpha}")
                                    logging.info(f"Updated weights: {self.agg_weights}")
                                else:  # If timestamps are equal or peer's weights are newer, give equal weight
                                    alpha = 0.5
                                    logging.info(f"Using equal weights for {peer}'s data for EWMA with alpha {alpha}")
                                    # Ensure both are NumPy arrays before performing arithmetic operations
                                    fedBuff_weights = np.array(fedBuff_weights)
                                    peer_weights = np.array(peer_weights)
                                    logging.info(f"Converted weights to NumPy arrays for arithmetic operations with alpha {self.alpha}")
                                    self.agg_weights = alpha * fedBuff_weights + (1 - alpha) * peer_weights

                                self.logger.info(f"Updated weights using {peer}'s data")

                    except Exception as e:
                        self.logger.warning(f"Failed to pull weights from {peer}: {e}")
                await asyncio.sleep(5)  # Wait before the next update cycle
        except Exception as e:
                self.logger.error(f"Error in update_weights: {e}")
                await asyncio.sleep(5)  # Wait before retrying in case of error
    
    async def pull_weights(self,request):
        current_time = datetime.now()
        """Return the current weights and timestamp to peers"""
        if self.fedBuff_weights is not None:
            logging.info(f"Pulling weights for peer request at {current_time}")
            logging.info(f"Pulling weights for the update process")
            fedBuff_weights = list(self.fedBuff_weights)
            return web.json_response({"weights": fedBuff_weights, "timestamp": current_time.timestamp()})
        
        return web.json_response({"status": "no weights available"}, status=404)
    
    
    async def heartbeat(self, request):
        return web.json_response({"status": "alive"})

    async def send_heartbeat(self, peer):
        for attempt in range(3):  # Retry up to 3 times
            try:
                async with self.session.get(f"http://{peer}/heartbeat") as resp:
                    if resp.status == 200:
                        self.logger.info("Heartbeat successful to peer: %s", peer)
                        return True
            except Exception:
                self.logger.warning(f"Heartbeat check failed for {peer}, attempt {attempt + 1}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return False

    async def heartbeat_check(self):
        #call the heartBeat method to check if the peers are alive and remove the unreachable ones
        self.logger.info("Starting heartbeat check")
        
        while True:
            unreachable_peers = set()
            for peer in list(self.peers):
                if not await self.send_heartbeat(peer):
                    unreachable_peers.add(peer)
            self.peers.difference_update(unreachable_peers)
            if unreachable_peers:
                self.logger.info(f"Removed unreachable peers: {unreachable_peers}")
            await asyncio.sleep(10)

    async def discover_peers(self):
        for i in range(1, self.max_nodes + 1):
            peer_host = f"{self.base_name}-{i}"
            if peer_host == self.node_name:
                continue
            if peer_host in self.peers:
                self.peers.remove(peer_host)  # Remove peer if it was previously unreachable
            peer_url = f"http://{peer_host}:{self.port}"
            try:
                async with self.session.get(f"{peer_url}/list_peers") as resp:
                    if resp.status == 200:
                        discovered_peers = await resp.json()
                        self.peers.update(discovered_peers.get("peers", []))
                async with self.session.post(f"{peer_url}/register", json={"peer": f"{self.node_name}:{self.port}"}) as resp:
                    if resp.status == 200:
                        self.logger.info(f"Registered with peer {peer_host}")
            except Exception:
                pass
        self.logger.info(f"Updated peer list: {self.peers}")
    async def list_peers(self, request):
        """Return the list of registered peers"""
        return web.json_response({"peers": list(self.peers)})
    
    
    async def background_discovery(self):
        logging.debug("Starting background peer discovery")
        logging.debug(f"Current peers numbers: {len(self.peers)}")
        if len(self.peers) == self.max_nodes -1:
            self.logger.info("All peers discovered, starting heartbeat check")
            return 
            
        while True:
            await self.discover_peers()
            await asyncio.sleep(30)

    async def start_app(self):
        await self.start_session()# Start session before app runs
        app = web.Application()
        #Registers the peers
        app.router.add_post("/register", self.register_peer)
        
        #Endpoints lists the peers
        app.router.add_get("/list_peers", self.list_peers)
        
        #Endpoints for heartbeat and weight management
        app.router.add_get("/heartbeat", self.heartbeat)
        app.router.add_get("/pull_weights", self.pull_weights)
        app.router.add_post("/push_weights", self.recv_weights)
        
        #Endpoints that gets weights from the aggregator
        app.router.add_post("/agg_weights",self.aggregatorWeights)
        
        
        # Run background tasks directly in the app setup
        asyncio.create_task(self.heartbeat_check())
        asyncio.create_task(self.background_discovery())
        asyncio.create_task(self.periodic_task(30))  # Run periodic task with 30-second interval

        # Setup and run the web server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()

        logging.info(f"Gossip node {self.node_name} started on port {self.port}")

        # Keep the app running without blocking periodic tasks
        while True:
            await asyncio.sleep(3600)  
        



import asyncio
import random
import aiohttp
from aiohttp import web
import os 
import socket
import logging
import time
#userModule
from dbWrite import writeToDb
from Layers.l2Server import StartServer
from Layers.l3Server import L3Start

class RaftNode:
    def __init__(self):
        self.node_id =  int(os.getenv("NODE_ID", 0))  # Unique identifier for the node
        self.num_peers = int(os.getenv("NUM_PEERS", 5))# Unique identifier for the node
        self.peers = self.discover_peers()  # List of peer (IP, port)
        self.state = "follower"  # Initial state
        self.current_term = 0
        self.voted_for = None
        self.leader = None
        self.election_timeout = random.uniform(0.15,0.3)  # 800ms - 1200ms # 150-300ms
        self.vote_count = 0
        self.last_heartbeat_time = time.time()
        # Get peer count from env
        self.peers = self.discover_peers()

    def discover_peers(self):
        """ Automatically discover peers using predictable Docker DNS. """
        base_name = "raft-raft-node"
        peers = [f"{base_name}-{i}-1:5000" for i in range(1, self.num_peers + 1)]
        return [peer for peer in peers if peer != f"{base_name}-{self.node_id}-1:5000"]

    
    async def request_votes(self):
        """ Start election and request votes from peers. """
        self.state = "candidate"
        self.current_term += 1
        self.voted_for = self.node_id
        self.vote_count = 1  # Vote for self
        logging.info(f"{self.node_id} requesting votes for term {self.current_term}")
        #print(f"{self.node_id} requesting votes for term {self.current_term}")
        logging.info(f"Self vote count is {self.vote_count}")
        #print(f"Self vote count is {self.vote_count}")
        
        async with aiohttp.ClientSession() as session:
            for peer in self.peers:
                try:
                    async with session.post(f"http://{peer}/vote", json={"term": self.current_term, "candidate_id": self.node_id}) as resp:
                        data = await resp.json()
                        if data["vote_granted"]:
                            self.vote_count += 1
                            logging.info(f"{self.node_id} received vote from {peer}")
                except Exception:
                    pass
        
        if self.vote_count > (len(self.peers) + 1) // 2:
            self.state = "leader"
            self.leader = self.node_id
            
            # Announce the leader to all peers

            await self.broadcast_leader()

            # Log leader information before writing to the database
            self.leader_ip = socket.gethostbyname(socket.gethostname())  
            logging.info(f"Leader {self.node_id} resolved IP: {self.leader_ip}")

            # Double-check: Only allow the node that first wins election to write
            if self.leader == self.node_id:  # Ensure it's the confirmed leader
                insertdb = writeToDb()
                #Once I know my leaderIP, the leader will write to the database
                #We can call the L2,L3 methods here to start the service once we know it after the insert is done
                await insertdb.insert(payload=self.leader_ip)
                if self.leader_ip:
                #Start the L2 and L3 servers
                    await insertdb.addStatus()
                    
                    logging.info(f"Leader {self.node_id} started L2 and L3 servers")
                
                    l2=StartServer()
                    l3=L3Start()
                    
                    try:    
                        task1=asyncio.create_task(l2.server())
                        task2=asyncio.create_task(l3.serveL3())
                        await asyncio.gather(task1, task2)
                        logging.info(f"Leader {self.node_id} started the servers") 
                    except Exception as e:
                        logging.error(f"Error starting the servers: {e}")
                logging.info(f"Leader {self.node_id} inserted IP {self.leader_ip} into DB")
    
    #to Handle the leader announcement           
    async def broadcast_leader(self):
        """ Announce the leader to all peers. """
        logging.info(f"Leader {self.node_id} announcing itself to peers.")
        async with aiohttp.ClientSession() as session:
            for peer in self.peers:
                try:
                    async with session.post(f"http://{peer}/leader_announcement", json={"leader_id": self.node_id, "term": self.current_term}) as resp:
                        data = await resp.json()
                        logging.info(f"Sent leader announcement to {peer}")
                except Exception:
                    pass
    
    async def handle_vote_request(self, request):
        logging.info(f"{self.node_id} received a vote request")
        data = await request.json()

        if data["term"] > self.current_term:  
            self.current_term = data["term"]  # Update term
            self.state = "follower"
            self.voted_for = None  # Reset vote
        
        vote_granted = self.voted_for is None or self.voted_for == data["candidate_id"]
        
        if vote_granted:
            self.voted_for = data["candidate_id"]
            logging.info(f"{self.node_id} granted vote to {data['candidate_id']}")
        
        return web.json_response({"vote_granted": vote_granted})
    
    async def handle_leader_announcement(self, request):
        """ Handle leader announcements from other nodes. """
        data = await request.json()
        if data["term"] >= self.current_term:
            self.current_term = data["term"]
            self.state = "follower"
            self.leader = data["leader_id"]
            logging.info(f"{self.node_id} acknowledged leader {self.leader} for term {self.current_term}")

        return web.json_response({"status": "acknowledged"})

    
    async def start_heartbeat(self):
        """ Periodically send heartbeats to the followers. """
        while True:
            if self.state == "leader":
                async with aiohttp.ClientSession() as session:
                    for peer in self.peers:
                        try:
                            async with session.post(f"http://{peer}/heartbeat", json={"term": self.current_term, "leader_id": self.node_id}) as resp:
                                data = await resp.json()
                                logging.info(f"Sent heartbeat to {peer}")
                        except Exception:
                            pass
            await asyncio.sleep(1)
            
    async def start_election_timer(self):
        """ Start election timeout and trigger election if no leader. """
        await asyncio.sleep(random.randint(1,10))
        while True:
            await asyncio.sleep(self.election_timeout)
            if self.state == "follower" and self.leader is None:
                logging.info(f"{self.node_id} starting an election")
                await self.request_votes()

    async def start_server(self):
        """ Start HTTP server for Raft communication. """
        app = web.Application()
        app.add_routes([web.post("/vote", self.handle_vote_request),
                        web.post("/leader_announcement", self.handle_leader_announcement),
                        web.post("/heartbeat", self.handle_heartbeat)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 5000)
        await site.start()
        logging.info(f"{self.node_id} HTTP server running on port 5000")

    async def handle_heartbeat(self, request):
        """ Handle heartbeat messages from the leader. """
        data = await request.json()
        if data["term"] > self.current_term:
            self.current_term = data["term"]
            self.state = "follower"
            self.leader = data["leader_id"]
            self.last_heartbeat_time = time.time()  # Reset heartbeat timer
            logging.info(f"{self.node_id} received heartbeat from leader {self.leader}")
        return web.json_response({"status": "received"})

async def main():
    node = RaftNode()  
    await node.start_server()
    await node.start_election_timer()

if __name__ == "__main__":
    asyncio.run(main())

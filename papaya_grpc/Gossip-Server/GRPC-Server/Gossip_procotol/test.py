import os
import asyncio
import numpy as np
import logging
from Protocol import GossipNode
from aiohttp import web, ClientSession

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize shared weights
#shared_weights = getWeights()
global_model = np.random.rand(1000)
global_weights = global_model.flatten().tolist()
logging.info("Weights initialized.")
#shared_weights.shareWeight(global_weights)
#logging.info(f"The weights are:{global_weights}")
async def send_weights(session,global_weights):
    
    """Send weights periodically using aiohttp."""
    while True:
            try:
                async with session.post("http://localhost:8000/agg_weights", json={"weights": global_weights}) as resp:
                    if resp.status == 200:
                        logging.info("Weights sent successfully.")
                    else:
                        logging.warning(f"Failed to send weights: {resp.status}")
            except Exception as e:
                logging.error(f"Error sending weights: {e}")

            await asyncio.sleep(5)

async def main():
    node_name = os.getenv("NODE_NAME", "default_node")
    node = GossipNode(node_name)
    logging.info(f"Node {node_name} started.")

    async with ClientSession() as session:
        # Run `start_app()` as a separate task so that it doesn't block `send_weights`
        node_task = asyncio.create_task(node.start_app())
        weights_task = asyncio.create_task(send_weights(session, global_weights))

        # Wait for both tasks to run indefinitely
        await asyncio.gather(node_task, weights_task)
    #await send_weights()

if __name__ == "__main__":
    asyncio.run(main())

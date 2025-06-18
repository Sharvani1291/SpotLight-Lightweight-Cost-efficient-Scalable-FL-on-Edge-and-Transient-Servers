import yaml

NUM_NODES = 3  # Change this for more nodes
BASE_NAME = "gossip-node"
BASE_PORT = 8000

# Define the base structure
compose_override = {
    "version": "3.8",
    "services": {},
    "networks": {
        "gossip-net": {"driver": "bridge"}
    }
}

for i in range(1, NUM_NODES + 1):
    node_name = f"{BASE_NAME}-{i}"
    port = BASE_PORT + i  # Incrementing port
    compose_override["services"][node_name] = {
        "image": "vinabirajdar/gossip:test",
        "hostname": node_name,
        "ports": [f"{port}:8000"],  # Map each container to a unique external port
        "networks": ["gossip-net"],
        "environment": {
            "NODE_NAME": node_name,
            "SERVICE_NAME": node_name,
            "PEER_DISCOVERY": "true"
        }
    }

# Write the generated YAML
with open("docker-compose.override.yml", "w") as f:
    yaml.dump(compose_override, f, default_flow_style=False)

print(f"Generated docker-compose.override.yml with {NUM_NODES} unique nodes!")

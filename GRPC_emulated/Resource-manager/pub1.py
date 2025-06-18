import zmq
import time

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

while True:
    topic = "routes"
    message = "Hello, World!"
    print(f"Sending message on topic '{topic}': {message}")
    socket.send_string(f"{topic} {message}")
    time.sleep(1)
import numpy as np
import time
import requests
import argparse

class PoissonLoadGenerator:
    def __init__(self, rate_lambda, url, method="GET", payload=None, headers=None):
        """
        Initialize the load generator with a given rate (lambda) and HTTP request details.
        
        :param rate_lambda: Average number of events (requests) per second.
        :param url: The URL to send requests to.
        :param method: HTTP method to use ('GET', 'POST', etc.).
        :param payload: Data to send in the case of a POST request.
        :param headers: Headers to include in the request.
        """
        self.rate_lambda = rate_lambda
        self.url = url
        self.method = method
        self.payload = payload
        self.headers = headers
    
    def generate_interarrival_time(self):
        """
        Generate an inter-arrival time based on the exponential distribution.
        :return: Inter-arrival time.
        """
        return np.random.exponential(1.0 / self.rate_lambda)
    
    def start_generating_load(self, duration):
        """
        Start generating load for a given duration by sending HTTP requests.
        
        :param duration: Duration for which to generate the load (in seconds).
        """
        end_time = time.time() + duration
        while time.time() < end_time:
            interarrival_time = self.generate_interarrival_time()
            time.sleep(interarrival_time)
            self.send_http_request()
    
    def send_http_request(self):
        """
        Send an HTTP request based on the specified method, URL, and other parameters.
        """
        try:
            if self.method.upper() == "GET":
                response = requests.get(self.url, headers=self.headers)
            elif self.method.upper() == "POST":
                response = requests.post(self.url, data=self.payload, headers=self.headers)
            # Add more HTTP methods as needed
            
            # Log or print the response status
            print(f"Request sent to {self.url}, Status Code: {response.status_code}")
        
        except requests.exceptions.RequestException as e:
            print(f"Error sending request: {e}")

def main():
    parser = argparse.ArgumentParser(description="Poisson Load Generator")
    parser.add_argument("url", type=str, help="The URL to send requests to")
    parser.add_argument("rate_lambda", type=float, help="The average number of requests per second")
    parser.add_argument("--duration", type=int, default=10, help="Duration to run the load generator (in seconds)")
    parser.add_argument("--method", type=str, default="GET", help="HTTP method to use (GET, POST, etc.)")
    parser.add_argument("--payload", type=str, help="Payload for POST requests")
    parser.add_argument("--headers", type=str, help="Headers for the HTTP requests in JSON format")

    args = parser.parse_args()

    # Convert headers from JSON string to Python dictionary (if provided)
    headers = None
    if args.headers:
        import json
        headers = json.loads(args.headers)

    generator = PoissonLoadGenerator(args.rate_lambda, args.url, method=args.method, payload=args.payload, headers=headers)
    generator.start_generating_load(args.duration)

if __name__ == "__main__":
    main()

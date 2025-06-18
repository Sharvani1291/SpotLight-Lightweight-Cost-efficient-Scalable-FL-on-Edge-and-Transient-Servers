import torch
import logging

class FedBuff:
    def __init__(self, buffer_size, aggregation_goal):
        """
        Initializes the FedBuff object with the given buffer size and aggregation goal.

        Args:
            buffer_size (int): The maximum number of client weights to store in the buffer.
            aggregation_goal (int): The number of clients' weights needed to trigger aggregation.
        """
        self.buffer_size = buffer_size
        self.aggregation_goal = aggregation_goal
        self.buffer = []
        self.sample_counts = []

        # Logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def add_to_buffer(self, client_weights, num_samples):
        """
        Adds the client weights and sample counts to the buffer if space is available.

        Args:
            client_weights (torch.Tensor): The weights of the client's model.
            num_samples (int): The number of samples used by the client.
        """
        logging.info(f"Adding to buffer. Current buffer size: {len(self.buffer)}")
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(client_weights)
            self.sample_counts.append(num_samples)
        else:
            logging.warning("Buffer is full. Ignoring new weights.")

    def can_aggregate(self):
        """
        Checks if the buffer has enough clients' weights to meet the aggregation goal.

        Returns:
            bool: True if the aggregation goal is met, False otherwise.
        """
        logging.info(f"Checking aggregation. Buffer size: {len(self.buffer)}, Goal: {self.aggregation_goal}")
        return len(self.buffer) >= self.aggregation_goal

    def aggregate(self, global_model):
        """
        Aggregates the client weights in the buffer using the FedBuff algorithm and updates the global model.

        Args:
            global_model (torch.nn.Module): The global model to update with the aggregated weights.
        """
        if not self.can_aggregate():
            logging.warning("Aggregation called but buffer does not meet aggregation goal.")
            return

        logging.info("Starting aggregation process.")

        # Sum up the weighted client models based on their sample counts
        total_samples = sum(self.sample_counts)
        avg_weights = torch.zeros_like(self.buffer[0])

        # Aggregate weights using the FedBuff algorithm (weighted average)
        for weight, num_samples in zip(self.buffer, self.sample_counts):
            avg_weights += weight * (num_samples / total_samples)

        # Update the global model with the aggregated weights
        offset = 0
        for param in global_model.parameters():
            param_size = param.numel()
            param.data.copy_(avg_weights[offset:offset + param_size].view_as(param))
            offset += param_size

        logging.info("Aggregation complete.")

    def clear_buffer(self):
        """
        Clears the buffer after performing aggregation.
        """
        logging.info("Clearing buffer after aggregation.")
        self.buffer.clear()
        self.sample_counts.clear()


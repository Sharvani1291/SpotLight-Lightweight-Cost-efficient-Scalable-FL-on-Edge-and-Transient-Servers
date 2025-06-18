import numpy as np
import logging

# logging.info("Starting the GRPC client script.")
# logging.basicConfig(filename='client.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
class Model:
    def __init__(self, model):
        self.model = model
        
    def generate_random_weights(self):
        np.random.seed(42)
        if self.model == 'cnn':
            # Generate a large matrix for CNN
            weights = np.random.rand(32 * 3 * 3 * 3 + 64 * 32 * 3 * 3 + 128 * 64 * 6 * 6 + 10 * 128)
        elif self.model == 'lstm':
            # Generate a large matrix for LSTM
            weights = np.random.rand(4 * 128 * 100 + 4 * 128 * 128 + 4 * 128 + 4 * 128 + 10 * 128)
        else:
            logging.warning(f"Model type {self.model} is not supported for random weight generation.")
        
        return weights
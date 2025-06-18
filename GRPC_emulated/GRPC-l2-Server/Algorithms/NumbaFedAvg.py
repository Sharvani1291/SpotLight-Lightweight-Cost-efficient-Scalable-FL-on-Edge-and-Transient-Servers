
from numba import njit
import numpy as np

class NumbaFedAvg:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, local_models, num_samples):
        total_samples = num_samples
        weighted_sum = np.zeros_like(self.global_model)

        # Convert local_models to a single NumPy array for Numba compatibility
        local_models_array = np.array(local_models)

        # Perform federated averaging using Numba
        weighted_sum = self._aggregate_numba(local_models_array, total_samples)

        return weighted_sum

    @staticmethod
    @njit
    def _aggregate_numba(local_models, total_samples):
        weighted_sum = np.zeros_like(local_models[0])

        for i in range(local_models.shape[0]):
            weighted_sum += local_models[i] / total_samples

        return weighted_sum
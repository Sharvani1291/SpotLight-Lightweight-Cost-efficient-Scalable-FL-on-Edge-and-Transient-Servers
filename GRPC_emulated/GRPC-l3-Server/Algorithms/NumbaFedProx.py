from numba import njit
import numpy as np

class NumbaFedProx:
    def __init__(self, global_model: np.ndarray, mu: float = 0.1):
        self.global_model = global_model.astype(np.float32)
        self.mu = mu  # proximal term weight

    def aggregate(self, local_models, sample_counts):
        local_models_array = np.array(local_models, dtype=np.float32)
        sample_counts_array = np.array(sample_counts, dtype=np.float32)

        total_samples = np.sum(sample_counts_array)

        aggregated_weights = self._aggregate_numba(
            local_models_array, sample_counts_array, total_samples
        )

        return aggregated_weights

    @staticmethod
    @njit
    def _aggregate_numba(local_models, sample_counts, total_samples):
        weighted_sum = np.zeros_like(local_models[0], dtype=np.float32)
        for i in range(local_models.shape[0]):
            weighted_sum += local_models[i] * sample_counts[i]

        if total_samples > 0:
            return weighted_sum / total_samples
        else:
            return weighted_sum  # return zeroed model if no updates

    def prox_update(self, local_weights: np.ndarray, initial_weights: np.ndarray) -> np.ndarray:
   
        return self._apply_proximal_term(local_weights, initial_weights, self.mu)

    @staticmethod
    @njit
    def _apply_proximal_term(local_weights, global_weights, mu):
     
        return local_weights - mu * (local_weights - global_weights)

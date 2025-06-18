from numba import njit
import numpy as np

class NumbaFedBuff:
    def __init__(self, global_model: np.ndarray):
        if not isinstance(global_model, np.ndarray):
            raise ValueError("Global model must be a NumPy array.")
        self.global_model = global_model.astype(np.float32)

    def aggregate(self, local_models, sample_counts):
        if not local_models or not sample_counts:
            raise ValueError("Input models or sample counts are empty.")

        local_models_array = np.array(local_models, dtype=np.float32)
        sample_counts_array = np.array(sample_counts, dtype=np.float32)

        if local_models_array.ndim != 2:
            raise ValueError(f"Expected 2D array for models, got shape {local_models_array.shape}")
        if local_models_array.shape[0] != sample_counts_array.shape[0]:
            raise ValueError("Mismatch between number of models and number of sample counts.")

        total_samples = np.sum(sample_counts_array)
        return self._aggregate_numba(local_models_array, sample_counts_array, total_samples)


    @staticmethod
    @njit
    def _aggregate_numba(local_models, sample_counts, total_samples):
        weighted_sum = np.zeros_like(local_models[0], dtype=np.float32)
        for i in range(local_models.shape[0]):
            weighted_sum += local_models[i] * sample_counts[i]

        if total_samples > 0:
            return weighted_sum / total_samples
        else:
            return weighted_sum  # return zeros if total_samples == 0

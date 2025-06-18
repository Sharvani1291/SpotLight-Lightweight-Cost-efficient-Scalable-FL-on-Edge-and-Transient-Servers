from numba import njit
import numpy as np

class NumbaFedAdam:
    def __init__(self, global_model: np.ndarray, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.global_model = global_model.astype(np.float32)
        self.m_t = np.zeros_like(self.global_model, dtype=np.float32)
        self.v_t = np.zeros_like(self.global_model, dtype=np.float32)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def aggregate(self, local_models, sample_counts):
        # Preprocess inputs
        local_models_array = np.array(local_models, dtype=np.float32)
        sample_counts_array = np.array(sample_counts, dtype=np.float32)
        total_samples = np.sum(sample_counts_array)

        # Compute weighted average model (FedAvg step)
        avg_model = self._fedavg_numba(local_models_array, sample_counts_array, total_samples)

        # Compute gradient estimate (difference from current global model)
        gradient = self.global_model - avg_model

        # Update step
        self.t += 1
        updated_weights, self.m_t, self.v_t = self._adam_update_numba(
            self.global_model, gradient, self.m_t, self.v_t,
            self.t, self.lr, self.beta1, self.beta2, self.epsilon
        )

        self.global_model = updated_weights
        return self.global_model

    @staticmethod
    @njit
    def _fedavg_numba(local_models, sample_counts, total_samples):
        weighted_sum = np.zeros_like(local_models[0], dtype=np.float32)
        for i in range(local_models.shape[0]):
            weighted_sum += local_models[i] * sample_counts[i]

        if total_samples > 0:
            return weighted_sum / total_samples
        else:
            return weighted_sum  # return zeroed model

    @staticmethod
    @njit
    def _adam_update_numba(w, g, m, v, t, lr, beta1, beta2, epsilon):
        m_t = beta1 * m + (1 - beta1) * g
        v_t = beta2 * v + (1 - beta2) * (g * g)

        # Bias correction
        m_hat = m_t / (1 - beta1 ** t)
        v_hat = v_t / (1 - beta2 ** t)

        updated_w = w - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        return updated_w, m_t, v_t

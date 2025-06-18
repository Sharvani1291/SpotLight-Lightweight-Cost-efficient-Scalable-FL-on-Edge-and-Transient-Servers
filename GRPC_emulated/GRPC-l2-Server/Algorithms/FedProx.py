import numpy as np

class FedProx:
    """
    FedBuff aggregation with an optional proximal parameter μ.

    If μ = 1   → identical to vanilla FedBuff / FedAvg (full update)
    If 0 < μ < 1 → partial move toward the incoming weighted average
    If μ = 0   → no update (global model stays as-is)

    Parameters
    ----------
    global_model : np.ndarray
        Current global weights (1-D array).
    mu : float, optional (default 1.0)
        Proximal coefficient μ ∈ [0, 1].  Smaller μ → more conservative updates.
    """

    def __init__(self, global_model: np.ndarray, *, mu: float = 0.1):
        if not isinstance(global_model, np.ndarray):
            raise TypeError("global_model must be a NumPy array")

        if not (0.0 <= mu <= 1.0):
            raise ValueError("mu must be in the range [0, 1]")

        self.global_model = global_model.astype(np.float32, copy=True)
        self.mu = float(mu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def aggregate(self, local_models, sample_counts):
        """Aggregate a list of local models with FedBuff + proximal update."""
        if not local_models or not sample_counts:
            raise ValueError("local_models and sample_counts may not be empty")

        local   = np.asarray(local_models,  dtype=np.float32)  # shape (K, D)
        counts  = np.asarray(sample_counts, dtype=np.float32)  # shape (K,)

        if local.ndim != 2:
            raise ValueError(f"Expected 2-D weight matrix; got shape {local.shape}")
        if local.shape[0] != counts.shape[0]:
            raise ValueError("local_models and sample_counts length mismatch")

        total = counts.sum(dtype=np.float32)
        weighted_avg = self._aggregate_numpy(local, counts, total)   # (D,)

        # ---------- proximal update ----------
        # new_global = (1-μ)·old_global + μ·weighted_avg
        self.global_model = (
            (1.0 - self.mu) * self.global_model + self.mu * weighted_avg
        ).astype(np.float32, copy=False)

        return self.global_model  # return the updated global weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_numpy(local_models, sample_counts, total_samples):
        """Pure NumPy weighted average."""
        if total_samples == 0:
            return np.zeros_like(local_models[0], dtype=np.float32)

        # (D, K) @ (K,)  -> (D,)
        return (local_models.T @ sample_counts) / total_samples

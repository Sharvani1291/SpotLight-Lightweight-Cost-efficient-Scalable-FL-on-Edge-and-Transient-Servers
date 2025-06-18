from __future__ import annotations
from typing import List
import numpy as np

class FedAvg:
    """
    Vectorised federated averaging with no external deps.
    """

    @staticmethod
    def aggregate(local_models: List[np.ndarray],
                  sample_counts: List[int]) -> np.ndarray:
        if not local_models:
            raise ValueError("local_models is empty")
        if len(local_models) != len(sample_counts):
            raise ValueError("local_models and sample_counts length mismatch")

        weights_2d = np.vstack(local_models).astype(np.float32, copy=False)
        counts_f32 = np.asarray(sample_counts, dtype=np.float32)
        total = counts_f32.sum()
        if total == 0:
            raise ValueError("Sum of sample_counts is zero")

  
        alpha = counts_f32 / total                    
        averaged = (weights_2d.T * alpha).sum(axis=1)    

        return averaged.astype(local_models[0].dtype, copy=False)

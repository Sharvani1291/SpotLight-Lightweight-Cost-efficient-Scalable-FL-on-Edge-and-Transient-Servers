import numpy as np
import logging
from math import ceil

class Model:
    """
    Synthetic-weight generator for stress tests.

    • model="cnn" or "lstm"  → use preset layer sizes
    • target_mb=N            → override and build ~N MiB tensor
    """

    def __init__(
        self,
        model: str,
        dtype=np.float64,
        seed: int | None = 42,
        target_mb: int | None = None,      # <── new
    ):
        self.model      = model.lower()
        self.dtype      = dtype
        self.seed       = seed
        self.target_mb  = target_mb
        if seed is not None:
            np.random.seed(seed)

    # ---------- original presets ----------
    def _param_counts(self) -> list[int]:
        if self.model == "cnn":
            return [
                64  * 3 * 3 * 3,
                128 * 64 * 3 * 3,
                256 * 128 * 3 * 3,
                1_000 * 256,
            ]
        elif self.model == "lstm":
            hidden, inp = 512, 256
            return [
                4 * hidden * inp,
                4 * hidden * hidden,
                8 * hidden,
                1_000 * hidden,
            ]
        else:
            raise ValueError(f"Unsupported model type: {self.model}")

    # ---------- generator ----------
    def generate_random_weights(self) -> np.ndarray:
        if self.target_mb is not None:
            # override: flat vector ≈ target_mb MiB
            bytes_per = np.dtype(self.dtype).itemsize
            target_bytes = self.target_mb * 1024 * 1024
            n_params = ceil(target_bytes / bytes_per)
        else:
            n_params = int(np.sum(self._param_counts()))

        arr = np.random.rand(n_params).astype(self.dtype, copy=False)
        logging.info(
            "%s params: %s (≈ %.1f MiB, dtype=%s)",
            self.model.upper(),
            f"{n_params:,}",
            arr.nbytes / (1024 * 1024),
            self.dtype,
        )
        return arr

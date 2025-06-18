import numpy as np

class FedAdam:
    """
    Federated Adam (FedOpt family, Reddi et al. 2021)
    Pure-NumPy, single-threaded reference – instrumented with gradient clipping.
    """

    def __init__(
        self,
        global_model: np.ndarray,
        lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.99,
        epsilon: float = 1e-8,
        *,
        dtype = np.float64,
        use_amsgrad: bool = False,
        keep_ema: float | None = None,
        allocate_tmp: bool = False,
        clip_threshold: float | None = None,  # new clipping threshold
    ):
        if not isinstance(global_model, np.ndarray):
            raise ValueError("Global model must be a NumPy array.")

        # model & state
        self.w = global_model.astype(dtype, copy=True)
        self.m = np.zeros_like(self.w)
        self.v = np.zeros_like(self.w)

        self.use_amsgrad = use_amsgrad
        if use_amsgrad:
            self.vhat = np.zeros_like(self.w)

        if keep_ema is not None:
            self.ema_decay = keep_ema
            self.ema = self.w.copy()

        self.allocate_tmp = allocate_tmp
        self.clip_threshold = clip_threshold

        # hyper-params
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, epsilon
        self.t = 0

    def aggregate(self, local_models, sample_counts):
        if not local_models or not sample_counts:
            raise ValueError("Empty inputs")

        L = np.asarray(local_models, dtype=self.w.dtype)
        N = np.asarray(sample_counts, dtype=np.float64)
        if L.ndim != 2 or L.shape[0] != N.shape[0]:
            raise ValueError("Shape mismatch")

        total = N.sum()
        return self._step(L, N, total)

    def _step(self, L, N, total):
        # FedAvg-style mean
        w_avg = np.tensordot(N, L, axes=1) / total
        g = self.w - w_avg  # gradient-like signal

        # GRADIENT CLIPPING
        if self.clip_threshold is not None:
            norm = np.linalg.norm(g)
            if norm > self.clip_threshold:
                g = g * (self.clip_threshold / norm)

        # Adam moments
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)

        # optional AMSGrad
        if self.use_amsgrad:
            np.maximum(self.vhat, self.v, out=self.vhat)
            v_corr = self.vhat
        else:
            v_corr = self.v

        # bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = v_corr / (1 - self.beta2 ** self.t)

        # optional temp allocation
        if self.allocate_tmp:
            _ = m_hat.copy() + v_hat.copy()

        # parameter update
        self.w -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # optional EMA
        if hasattr(self, "ema"):
            d = self.ema_decay
            self.ema = d * self.ema + (1 - d) * self.w

        return self.w

# Usage in L2server:
# self.optimizer = FedAdam(init_model, use_amsgrad=True, keep_ema=0.999,
#                          allocate_tmp=True, clip_threshold=1.0)

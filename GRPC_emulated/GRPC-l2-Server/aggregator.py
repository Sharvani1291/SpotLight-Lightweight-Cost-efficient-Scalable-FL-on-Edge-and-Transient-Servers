"""
Drop-in multiprocess aggregator using Python queues only.

    agg = AggregatorProxy(init_weights,
                          goal=2,                 # FedBuff trigger
                          algorithm="FedAdam",    # or FedAvg / FedProx / …
                          algo_kwargs={...})      # per-algo hyper-params

    agg.submit(weights, n_samples)   # enqueue from asyncio
    res = agg.try_get()              # -> (global_w, latency) | None
"""
import multiprocessing as mp
import importlib
import numpy as np
import time
import logging


# ------------------------------------------------------------------ helpers
def _resolve_algorithm(name: str):
    """Return the class object for Algorithms.<name> or a dotted path."""
    if "." not in name:
        module = importlib.import_module(f"Algorithms.{name}")
        return getattr(module, name)
    mod_path, cls_name = name.rsplit(".", 1)
    return getattr(importlib.import_module(mod_path), cls_name)


def _loop(in_q, out_q, init_w, goal, algo_cls, algo_kwargs):
    logging.basicConfig(level=logging.INFO,
                        format="AGG [%(process)d] %(message)s")
    optimizer = algo_cls(init_w, **algo_kwargs)

    cache, samples = [], []
    while True:
        cmd, payload = in_q.get()
        if cmd == "STOP":
            break
        if cmd == "UPDATE":
            w, n = payload
            cache.append(np.asarray(w))
            samples.append(n)

            if len(cache) >= goal:
                t0 = time.perf_counter()
                global_w = optimizer.aggregate(cache, samples)
                latency  = time.perf_counter() - t0
                cache.clear(); samples.clear()
                out_q.put(("GLOBAL_READY", global_w, latency))


# ------------------------------------------------------------------ proxy
class AggregatorProxy:
    def __init__(self,
                 init_weights: np.ndarray,
                 goal: int = 2,
                 algorithm: str = "FedAdam",
                 algo_kwargs: dict | None = None,
                 mp_ctx: str = "spawn"):

        ctx = mp.get_context(mp_ctx)
        self._in_q, self._out_q = ctx.Queue(), ctx.Queue()
        algo_cls = _resolve_algorithm(algorithm)

        self._proc = ctx.Process(
            target=_loop,
            args=(self._in_q, self._out_q,
                  np.asarray(init_weights),
                  goal,
                  algo_cls,
                  algo_kwargs or {})
        )
        self._proc.daemon = True
        self._proc.start()

    # ------------ API -------------
    def submit(self, weights, n_samples):
        self._in_q.put(("UPDATE", (weights, n_samples)))

    def try_get(self):
        try:
            tag, w, lat = self._out_q.get_nowait()
            return w, lat
        except mp.queues.Empty:
            return None

    def close(self):
        self._in_q.put(("STOP", None))
        self._proc.join()

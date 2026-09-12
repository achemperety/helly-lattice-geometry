#!/usr/bin/env python3
"""MC estimate of p~_k = P(k random triangles are PAIRWISE intersecting) (Helly-defect limit)."""
import sys, numpy as np, time
from mc_continuum import batch_hits
def estimate(k, S, batch=50000, seed=777):
    rng = np.random.default_rng(seed); hits = 0; done = 0
    while done < S:
        b = min(batch, S - done); tri = rng.random((b, k, 3, 2)); ok = np.ones(b, bool)
        for i in range(k):
            for j in range(i + 1, k):
                ok &= batch_hits(tri[:, [i, j]])
        hits += int(ok.sum()); done += b
    p = hits / S; return p, np.sqrt(p * (1 - p) / S)
if __name__ == "__main__":
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    for k in (3, 4):
        t = time.time(); p, se = estimate(k, S)
        print(f"k={k}: pairwise probability p~_k = {p:.5f} +/- {se:.5f}  [{time.time()-t:.0f}s]", flush=True)

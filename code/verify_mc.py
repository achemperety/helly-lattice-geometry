#!/usr/bin/env python3
"""Independent check of the Monte Carlo predicate: Sutherland-Hodgman clipping of T_1 by
the half-planes of T_2..T_k (pure Python, per sample) versus mc_continuum.batch_hits."""
import numpy as np, sys
from mc_continuum import batch_hits

def clip(poly, a, b):            # keep the side of line a->b where orient(a,b,x) >= 0
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        sp = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
        sq = (b[0]-a[0])*(q[1]-a[1]) - (b[1]-a[1])*(q[0]-a[0])
        if sp >= 0: out.append(p)
        if (sp >= 0) != (sq >= 0):
            t = sp / (sp - sq); out.append((p[0] + t*(q[0]-p[0]), p[1] + t*(q[1]-p[1])))
    return out

def ccw(T):
    a, b, c = T
    return list(T) if (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]) > 0 else [a, c, b]

def meets(tri):                  # tri: (k,3,2)
    poly = ccw([tuple(v) for v in tri[0]])
    for T in tri[1:]:
        t = ccw([tuple(v) for v in T])
        for e in range(3):
            poly = clip(poly, t[e], t[(e+1) % 3])
            if not poly: return False
    return len(poly) > 0

rng = np.random.default_rng(99)
for k in (2, 3, 4, 5):
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    tri = rng.random((S, k, 3, 2))
    fast = batch_hits(tri)
    slow = np.array([meets(tri[s]) for s in range(S)])
    print(f"k={k}: {S} samples, mismatches = {int((fast != slow).sum())}, p_hat fast={fast.mean():.4f} slow={slow.mean():.4f}")

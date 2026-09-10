#!/usr/bin/env python3
"""
mc_continuum.py -- Monte Carlo estimation of the continuum Helly constants

    p_k = P( k i.i.d. random triangles, vertices uniform in [0,1]^2, share a point ),

the leading constants in  f(n,k) = p_k (n^6/6)^k (1 + O(1/n)).

Exact predicate (holds with probability one):
    T_1 ∩ ... ∩ T_k ≠ ∅   ⇔   some vertex of some T_i lies in every other T_j,
                          or  some proper crossing of an edge of T_i with an edge
                              of T_j lies in every other T_l  (l ≠ i, j).
Proof: a non-empty intersection of closed triangles is a compact convex polygon
whose vertices are vertices of the T_i or crossings of two edges.  A crossing
point is only tested against the triangles that do not own the two edges, so no
floating-point membership test is ever performed on a point that lies (exactly)
on a boundary; the same holds for vertices.  Degenerate events have measure 0.

Everything is vectorised with numpy over batches of samples; the (b, 3k, k, 3)
orientation tensor is built once per batch.  Standard errors are binomial.
A Wendel check (P(centre ∈ T) = 1/4) is reported for k=2 as a sanity test.

Usage:  python3 mc_continuum.py [k_list] [samples_per_k] [batch] [seed]
        python3 mc_continuum.py 2,3,4 4000000 50000 12345
"""
import sys
import time
import numpy as np


def orient(p, q, r):
    """Signed twice-area of (p,q,r); broadcasting on leading axes."""
    return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])


def membership_tensor(pts, tri):
    """pts: (b, m, 2); tri: (b, k, 3, 2).  Returns (b, m, k) boolean: pts[b,i] in tri[b,j]."""
    a = tri[:, None, :, :, :]                      # (b,1,k,3,2)
    bpts = pts[:, :, None, None, :]                # (b,m,1,1,2)
    an = np.roll(a, -1, axis=3)                    # next vertex
    d = orient(a, an, bpts)                        # (b,m,k,3)
    return (d >= 0).all(axis=3) | (d <= 0).all(axis=3)


def crossings(tri, i, j):
    """All 9 edge-edge crossing points of triangles i and j.  Returns pts (b,9,2), valid (b,9)."""
    Ti, Tj = tri[:, i], tri[:, j]                  # (b,3,2)
    P1 = Ti[:, :, None, :]; P2 = np.roll(Ti, -1, axis=1)[:, :, None, :]   # (b,3,1,2)
    Q1 = Tj[:, None, :, :]; Q2 = np.roll(Tj, -1, axis=1)[:, None, :, :]   # (b,1,3,2)
    r = P2 - P1; s = Q2 - Q1; qp = Q1 - P1
    den = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]                    # (b,3,3)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / den
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / den
    valid = (den != 0) & (t > 0) & (t < 1) & (u > 0) & (u < 1)
    X = P1 + t[..., None] * r                                                # (b,3,3,2)
    b = tri.shape[0]
    return X.reshape(b, 9, 2), valid.reshape(b, 9)


def batch_hits(tri):
    """tri: (b,k,3,2) -> boolean (b,): the k closed triangles share a point."""
    b, k = tri.shape[0], tri.shape[1]
    verts = tri.reshape(b, 3 * k, 2)
    inside = membership_tensor(verts, tri)          # (b,3k,k)
    owner = np.repeat(np.arange(k), 3)              # vertex v belongs to triangle owner[v]
    inside[:, np.arange(3 * k), owner] = True       # a vertex is in its own triangle by definition
    found = inside.all(axis=2).any(axis=1)          # some vertex in all triangles
    if k >= 2:
        for i in range(k):
            for j in range(i + 1, k):
                X, valid = crossings(tri, i, j)
                if k == 2:
                    ok = valid
                else:
                    others = [l for l in range(k) if l != i and l != j]
                    ins = membership_tensor(X, tri[:, others])          # (b,9,k-2)
                    ok = valid & ins.all(axis=2)
                found |= ok.any(axis=1)
    return found


def estimate(k, samples, batch=50_000, seed=12345, wendel=False):
    rng = np.random.default_rng(seed + 1000 * k)
    hits = 0; done = 0; centre_hits = 0
    while done < samples:
        b = min(batch, samples - done)
        tri = rng.random((b, k, 3, 2))
        hits += int(batch_hits(tri).sum())
        if wendel:
            centre = np.full((b, 1, 2), 0.5)
            centre_hits += int(membership_tensor(centre, tri[:, :1])[:, 0, 0].sum())
        done += b
    p = hits / samples
    se = np.sqrt(p * (1 - p) / samples)
    return p, se, (centre_hits / samples if wendel else None)


if __name__ == "__main__":
    ks = [int(t) for t in (sys.argv[1] if len(sys.argv) > 1 else "2,3,4").split(",")]
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 50_000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 12345
    print(f"# samples per k = {S}, batch = {batch}, seed = {seed}", flush=True)
    for k in ks:
        t0 = time.time()
        p, se, pc = estimate(k, S, batch, seed, wendel=(k == 2))
        extra = f"   Wendel check P(centre in T) = {pc:.4f} (exact 0.25)" if pc is not None else ""
        print(f"k={k}: p_k = {p:.5f}  +/- {se:.5f}  (95%: [{p-1.96*se:.4f}, {p+1.96*se:.4f}])"
              f"   lower bound 4^-k = {4.0**-k:.5f}   [{time.time()-t0:.0f}s]{extra}", flush=True)

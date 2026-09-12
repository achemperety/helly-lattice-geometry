#!/usr/bin/env python3
"""
verify_helly.py -- independent brute-force checks of helly_grid.py

 (a) f(n,2) by the classical exact predicate (vertex containment / proper edge
     crossing), all pairs, integer arithmetic.
 (b) f(n,3), f(n,4) by candidate-point bitsets: ∩T_i ≠ ∅  ⇔  some arrangement
     vertex lies in all T_i  (vertices of the intersection polygon are
     arrangement vertices).  Enumerated over all ordered tuples via matrix
     products of the vertex-membership matrix.
 (c) Exact rational Sutherland-Hodgman clipping on random tuples, compared with
     the bitset predicate and with the open-interior predicate (area > 0).
 (d) Helly defect: ordered triples (quadruples) that are pairwise intersecting
     but have no common point  = tr(A^3) - f(n,3)  etc.
"""
import random, sys, time
from fractions import Fraction
import numpy as np
from helly_grid import solve, triangles, arrangement_vertices, grid_lines


def orient(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def point_in_tri(p, T):
    d = [orient(T[0], T[1], p), orient(T[1], T[2], p), orient(T[2], T[0], p)]
    return not (any(x < 0 for x in d) and any(x > 0 for x in d))


def proper_cross(p1, p2, q1, q2):
    d1, d2 = orient(p1, p2, q1), orient(p1, p2, q2)
    d3, d4 = orient(q1, q2, p1), orient(q1, q2, p2)
    return d1 * d2 < 0 and d3 * d4 < 0


def tris_intersect(T, U):
    return (any(point_in_tri(p, U) for p in T) or any(point_in_tri(q, T) for q in U)
            or any(proper_cross(T[i], T[(i + 1) % 3], U[j], U[(j + 1) % 3])
                   for i in range(3) for j in range(3)))


def clip(poly, a, b, c):
    """Exact Sutherland-Hodgman: keep {a x + b y + c >= 0}."""
    out = []
    m = len(poly)
    for i in range(m):
        P, Q = poly[i], poly[(i + 1) % m]
        fp, fq = a * P[0] + b * P[1] + c, a * Q[0] + b * Q[1] + c
        if fp >= 0:
            out.append(P)
        if (fp < 0 < fq) or (fq < 0 < fp):
            t = fp / (fp - fq)
            out.append((P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])))
    # dedupe consecutive duplicates
    ded = []
    for p in out:
        if not ded or ded[-1] != p:
            ded.append(p)
    if len(ded) > 1 and ded[0] == ded[-1]:
        ded.pop()
    return ded


def exact_intersection(tris, edges):
    poly = [(Fraction(int(x)), Fraction(int(y))) for x, y in tris[0]]
    for t in range(1, len(tris)):
        for (a, b, c) in edges[t]:
            poly = clip(poly, int(a), int(b), int(c))
            if not poly:
                return [], False
    if not poly:
        return [], False
    # area (shoelace) > 0  <=>  interiors share a point
    area = sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
               for i in range(len(poly)))
    return poly, area != 0


def vertex_membership(verts, E):
    """(N,V) boolean: closed triangle t contains arrangement vertex v."""
    X, Y, W = (verts[:, i].astype(np.int64) for i in range(3))
    A, B, C = (E[:, :, i].astype(np.int64) for i in range(3))
    s = A[:, :, None] * X[None, None, :] + B[:, :, None] * Y[None, None, :] + C[:, :, None] * W[None, None, :]
    return (s >= 0).all(axis=1)


def main():
    rng = random.Random(1)
    for n in [1, 2, 3]:
        res, (P, E, verts, germs, signs, mc, mo) = solve(n, kmax=4, verbose=False)
        N = len(P)
        Pl = P.tolist()
        print(f"\n=== n={n}: N={N}, |V|={len(verts)} ===")
        # (a) classical pairwise predicate
        t0 = time.time()
        A = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            A[i, i] = 1
            for j in range(i + 1, N):
                if tris_intersect(Pl[i], Pl[j]):
                    A[i, j] = A[j, i] = 1
        f2 = int(A.sum())
        print(f"(a) classical pairwise: f({n},2)={f2}  solver={res['f_closed'][2]}  "
              f"{'OK' if f2 == res['f_closed'][2] else 'MISMATCH'}  [{time.time()-t0:.1f}s]")
        # (b) bitset brute force
        Bm = vertex_membership(verts, E).astype(np.float32)          # (N,V)
        f2b = int(((Bm @ Bm.T) > 0).sum())
        print(f"(b) bitset pairwise:    f({n},2)={f2b}  {'OK' if f2b == res['f_closed'][2] else 'MISMATCH'}")
        t0 = time.time()
        f3 = 0
        for i in range(N):
            Ci = Bm * Bm[i]
            f3 += int(((Ci @ Ci.T) > 0).sum())
        print(f"(b) bitset triples:     f({n},3)={f3}  solver={res['f_closed'][3]}  "
              f"{'OK' if f3 == res['f_closed'][3] else 'MISMATCH'}  [{time.time()-t0:.1f}s]")
        if n <= 2:
            f4 = 0
            for i in range(N):
                for j in range(N):
                    Cij = Bm * (Bm[i] * Bm[j])
                    f4 += int(((Cij @ Cij.T) > 0).sum())
            print(f"(b) bitset quadruples:  f({n},4)={f4}  solver={res['f_closed'][4]}  "
                  f"{'OK' if f4 == res['f_closed'][4] else 'MISMATCH'}")
        # (c) exact clipping on random tuples vs bitset predicate, closed and open
        for k in (2, 3, 4, 5):
            bad = 0; trials = 3000 if n <= 2 else 1500
            for _ in range(trials):
                idx = [rng.randrange(N) for _ in range(k)]
                poly, openok = exact_intersection([P[i] for i in idx], [E[i] for i in idx])
                closed_bit = bool(np.logical_and.reduce([Bm[i] > 0 for i in idx]).any())
                if closed_bit != bool(poly):
                    bad += 1
            print(f"(c) exact clipping vs bitset, k={k}: {trials} random tuples, mismatches={bad}")
        # open-interior variant: compare solver's f_open(n,2) with exact clipping over all pairs (n<=2)
        if n <= 2:
            fo2 = 0
            for i in range(N):
                for j in range(N):
                    _, openok = exact_intersection([P[i], P[j]], [E[i], E[j]])
                    fo2 += openok
            print(f"(c) exact open pairs:   f°({n},2)={fo2}  solver={res['f_open'][2]}  "
                  f"{'OK' if fo2 == res['f_open'][2] else 'MISMATCH'}")
        # (d) Helly defect
        A2 = A @ A
        trA3 = int(np.round((A2 * A).sum()))
        print(f"(d) pairwise-intersecting ordered triples tr(A^3)={trA3}, f({n},3)={res['f_closed'][3]}, "
              f"Helly defect={trA3 - res['f_closed'][3]}")
        if n <= 3:
            # ordered 4-tuples with all six pairs intersecting = hom(K4 -> graph with loops)
            tot = 0
            for i in range(N):
                Ai = A * A[i][None, :]                     # Ai[j,m] = A_jm A_im
                U = Ai * A[i][:, None]                     # U[j,l]  = A_jl A_il A_ij
                tot += int(np.round((U * (Ai @ A)).sum()))
                # = sum_{j,l,m} A_ij A_il A_jl * (A_jm A_im A_ml)
            print(f"(d) pairwise-intersecting ordered 4-tuples={tot}, f({n},4)={res['f_closed'][4]}, "
                  f"Helly defect={tot - res['f_closed'][4]}")


if __name__ == "__main__":
    main()

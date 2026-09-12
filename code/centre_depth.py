#!/usr/bin/env python3
"""
centre_depth.py -- closed form for m_n(c), the number of non-degenerate lattice
triangles containing the centre c=(n/2,n/2), via the discrete Wendel sign-flip
argument, and comparison with the exact arrangement depth from helly_grid.

n odd  : S = all M=(n+1)^2 points, symmetric about c, c ∉ S.
n even : c ∈ S; use S' = S \ {c}, M' = M-1, and add the triangles with vertex c.

For a line ℓ through c let t_ℓ = #(S∩ℓ) (even).  Triples on three distinct
lines through c: A = C(M,3) - Σ_ℓ [C(t_ℓ,2)(M-t_ℓ) + C(t_ℓ,3)];  exactly 1/4 of
them contain c (Wendel: 2 of the 8 sign-flip variants).  Triples with two
points on ℓ on opposite sides of c and the third off ℓ contain c on an edge:
B = Σ_ℓ (t_ℓ/2)^2 (M - t_ℓ).  Hence  m(c) = A/4 + B.
t_ℓ is arithmetic: for a primitive direction (p,q) with r=max(|p|,|q|),
   n odd : lines with p,q both odd,  t_ℓ = 2*floor((n+r)/(2r))
   n even: all primitive (p,q),      t'_ℓ = 2*floor(n/(2r))
"""
from math import comb, gcd
from helly_grid import solve


def t_values(n):
    ts = []
    for p in range(-n, n + 1):
        for q in range(0, n + 1):
            if q == 0 and p <= 0:
                continue            # directions up to sign: q>0, or q==0 and p>0
            if gcd(abs(p), q) != 1:
                continue
            r = max(abs(p), q)
            if n % 2 == 1:
                if p % 2 == 0 or q % 2 == 0:
                    continue
                t = 2 * ((n + r) // (2 * r))
            else:
                t = 2 * (n // (2 * r))
            if t > 0:
                ts.append(t)
    return ts


def m_centre_formula(n):
    M = (n + 1) ** 2
    ts = t_values(n)
    if n % 2 == 1:
        assert sum(ts) == M
        A = comb(M, 3) - sum(comb(t, 2) * (M - t) + comb(t, 3) for t in ts)
        B = sum((t // 2) ** 2 * (M - t) for t in ts)
        return A // 4 + B
    else:
        Mp = M - 1
        assert sum(ts) == Mp
        A = comb(Mp, 3) - sum(comb(t, 2) * (Mp - t) + comb(t, 3) for t in ts)
        B = sum((t // 2) ** 2 * (Mp - t) for t in ts)
        vertex = comb(Mp, 2) - sum(comb(t, 2) for t in ts)    # c as a vertex, other two not collinear with c
        return A // 4 + B + vertex


if __name__ == "__main__":
    for n in range(1, 8):
        res, _ = solve(n, kmax=1, verbose=False, use_cpp=(n >= 5))
        f = m_centre_formula(n)
        print(f"n={n}: formula m(c)={f}   arrangement depth(c)={res['depth_center']}   "
              f"{'OK' if f == res['depth_center'] else 'MISMATCH'}   N={res['N']}  m(c)/N={f/res['N']:.4f}  "
              f"max depth={res['max_depth']} at {res['argmax']}")

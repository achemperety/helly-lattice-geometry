#!/usr/bin/env python3
"""
helly_grid.py  --  exact evaluation of

    f(n,k) = #{ ordered k-tuples (T_1..T_k) of non-degenerate lattice triangles
               on the (n+1)x(n+1) grid  with  T_1 ∩ ... ∩ T_k ≠ ∅ }

(closed triangles; also the open-interior variant f°, and the multiset/set
selection variants), for ALL k at once.

METHOD  (Euler-characteristic transfer on a line arrangement)
---------------------------------------------------------------
Let A_n be the arrangement of all lines through two grid points.  Every lattice
triangle is a union of closed cells of A_n, hence so is Q = ∩ T_i, and
        [Q ≠ ∅] = χ(Q) = Σ_{open cells c ⊆ Q} (-1)^{dim c}
because a non-empty compact convex set has Euler characteristic 1.
Since  c ⊆ ∩T_i  ⇔  c ⊆ T_i for every i,  summing over all tuples gives the
exact identity
        f(n,k) = Σ_c (-1)^{dim c} m(c)^k ,     m(c) = #{T : c ⊆ T}.
Cells are enumerated by "germs": each bounded edge / face is attached to its
lexicographically-minimal vertex v and represented by the symbolic point
v + ε d (d = edge direction, or a direction strictly inside the face's sector).
Membership of v + ε d in a closed triangle is decided lexicographically on the
signs (L(v), ∇L·d) of each edge functional L, with exact integer arithmetic.

Complexity: O(#cells * N) with N = #triangles;  #cells = O(#lines^2).

Usage:  python3 helly_grid.py --n 5 --kmax 8 [--cpp] [--check] [--json out.json]
"""
import argparse, json, os, subprocess, sys, time
from fractions import Fraction
from itertools import combinations
from math import comb, gcd

import numpy as np


# ----------------------------------------------------------------------------
# 1. Grid, lines, triangles
# ----------------------------------------------------------------------------
def grid_points(n):
    return [(x, y) for x in range(n + 1) for y in range(n + 1)]


def normalize_line(a, b, c):
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def grid_lines(n):
    """All distinct lines a x + b y + c = 0 through >= 2 grid points."""
    S = set()
    for p, q in combinations(grid_points(n), 2):
        a = q[1] - p[1]
        b = p[0] - q[0]
        c = -(a * p[0] + b * p[1])
        S.add(normalize_line(a, b, c))
    return np.array(sorted(S), dtype=np.int64)


def collinear_triples(n):
    """L(n): number of collinear (degenerate) point triples, via gcd counting."""
    tot = 0
    for p, q in combinations(grid_points(n), 2):
        tot += gcd(abs(p[0] - q[0]), abs(p[1] - q[1])) - 1
    return tot


def triangles(n):
    """Non-degenerate triangles, CCW oriented.  Returns (P, E):
       P : (N,3,2) vertices,  E : (N,3,3) edge functionals (a,b,c), >=0 inside."""
    pts = np.array(grid_points(n), dtype=np.int64)
    idx = np.array(list(combinations(range(len(pts)), 3)), dtype=np.int64)
    P = pts[idx]
    d = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) \
        - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])
    keep = d != 0
    P, d = P[keep], d[keep]
    sw = d < 0
    P[sw] = P[sw][:, [0, 2, 1], :]           # make CCW
    P2 = np.roll(P, -1, axis=1)
    a = -(P2[..., 1] - P[..., 1])
    b = P2[..., 0] - P[..., 0]
    c = -(a * P[..., 0] + b * P[..., 1])
    return P, np.stack([a, b, c], axis=-1)


# ----------------------------------------------------------------------------
# 2. Arrangement vertices (exact rationals X/W, Y/W) and incident lines
# ----------------------------------------------------------------------------
def arrangement_vertices(n, lines):
    a, b, c = lines[:, 0], lines[:, 1], lines[:, 2]
    I, J = np.triu_indices(len(lines), 1)
    W = a[I] * b[J] - a[J] * b[I]
    keep = W != 0
    I, J, W = I[keep], J[keep], W[keep]
    X = b[I] * c[J] - b[J] * c[I]
    Y = c[I] * a[J] - c[J] * a[I]
    sgn = np.where(W < 0, -1, 1)
    X, Y, W = X * sgn, Y * sgn, W * sgn
    g = np.gcd(np.gcd(np.abs(X), np.abs(Y)), W)
    X, Y, W = X // g, Y // g, W // g
    inside = (X >= 0) & (X <= n * W) & (Y >= 0) & (Y <= n * W)
    I, J, X, Y, W = I[inside], J[inside], X[inside], Y[inside], W[inside]
    keys = np.stack([X, Y, W], axis=1)
    verts, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    pairs = np.unique(np.stack([np.concatenate([inv, inv]),
                                np.concatenate([I, J])], axis=1), axis=0)
    starts = np.searchsorted(pairs[:, 0], np.arange(len(verts) + 1))
    return verts, pairs[:, 1], starts


# ----------------------------------------------------------------------------
# 3. Germs  (vertex, upper edges, upper faces at each vertex)
# ----------------------------------------------------------------------------
def lex_positive_dir(a, b):
    dx, dy = b, -a
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return dx, dy


def build_germs(lines, verts, vlines, starts):
    """Returns int32 array (G,5): X,Y,W,dx,dy ; int8 array (G,): sign (+1 vertex/face, -1 edge)."""
    rows, signs = [], []
    lines_l = lines.tolist()
    for vi in range(len(verts)):
        X, Y, W = (int(t) for t in verts[vi])
        dirs = []
        for li in vlines[starts[vi]:starts[vi + 1]]:
            a, b, _ = lines_l[li]
            dirs.append(lex_positive_dir(a, b))
        # sort by angle in (-pi/2, pi/2]: slope ascending, vertical last
        dirs.sort(key=lambda d: (d[0] == 0, Fraction(d[1], d[0]) if d[0] else 0))
        rows.append((X, Y, W, 0, 0)); signs.append(1)                       # the vertex
        for d in dirs:                                                     # upper edges
            rows.append((X, Y, W, d[0], d[1])); signs.append(-1)
        for d1, d2 in zip(dirs, dirs[1:]):                                 # upper faces
            rows.append((X, Y, W, d1[0] + d2[0], d1[1] + d2[1])); signs.append(1)
    return np.array(rows, dtype=np.int32), np.array(signs, dtype=np.int8)


# ----------------------------------------------------------------------------
# 4. Depth kernel: m(c) for closed and open triangles
# ----------------------------------------------------------------------------
def depths_numpy(germs, E, target_elems=12_000_000):
    """Exact int32 evaluation, batched.  Returns (m_closed, m_open) int64 arrays."""
    A = E[:, :, 0].reshape(-1).astype(np.int32)
    B = E[:, :, 1].reshape(-1).astype(np.int32)
    C = E[:, :, 2].reshape(-1).astype(np.int32)
    N = E.shape[0]
    G = len(germs)
    bs = max(1, target_elems // (3 * N))
    mc = np.zeros(G, dtype=np.int64)
    mo = np.zeros(G, dtype=np.int64)
    for s in range(0, G, bs):
        g = germs[s:s + bs]
        X, Y, W, dx, dy = (g[:, i][:, None] for i in range(5))
        s0 = X * A[None, :] + Y * B[None, :] + W * C[None, :]
        s1 = dx * A[None, :] + dy * B[None, :]
        z = s0 == 0
        okc = ((s0 > 0) | (z & (s1 >= 0))).reshape(len(g), N, 3).all(axis=2)
        oko = ((s0 > 0) | (z & (s1 > 0))).reshape(len(g), N, 3).all(axis=2)
        mc[s:s + bs] = okc.sum(axis=1)
        mo[s:s + bs] = oko.sum(axis=1)
    return mc, mo


CPP_SRC = r"""
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
int main(int argc, char** argv){
  long G = atol(argv[2]), T = atol(argv[4]);
  std::vector<int32_t> g(G*5), e(T*9), out(G*2);
  FILE* f = fopen(argv[1],"rb"); if(fread(g.data(),4,G*5,f)!=(size_t)(G*5)) return 1; fclose(f);
  f = fopen(argv[3],"rb");       if(fread(e.data(),4,T*9,f)!=(size_t)(T*9)) return 1; fclose(f);
  for(long i=0;i<G;i++){
    const int X=g[i*5],Y=g[i*5+1],W=g[i*5+2],dx=g[i*5+3],dy=g[i*5+4];
    int cc=0, co=0;
    const int32_t* E = e.data();
    for(long t=0;t<T;t++, E+=9){
      bool inC=true, inO=true;
      for(int k=0;k<3;k++){
        const int a=E[3*k], b=E[3*k+1], c=E[3*k+2];
        const int s0 = a*X + b*Y + c*W;
        if(s0<0){ inC=false; inO=false; break; }
        if(s0==0){ const int s1 = a*dx + b*dy;
                   if(s1<0){ inC=false; inO=false; break; }
                   if(s1==0) inO=false; }
      }
      cc += inC; co += inO;
    }
    out[i*2]=cc; out[i*2+1]=co;
  }
  f = fopen(argv[5],"wb"); fwrite(out.data(),4,G*2,f); fclose(f);
  return 0;
}
"""


def depths_cpp(germs, E, workdir="/tmp/helly_kernel"):
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, "kernel.cpp")
    exe = os.path.join(workdir, "kernel")
    if not os.path.exists(exe):
        open(src, "w").write(CPP_SRC)
        subprocess.check_call(["g++", "-O3", "-march=native", "-o", exe, src])
    gf, ef, of = (os.path.join(workdir, x) for x in ("germs.bin", "edges.bin", "out.bin"))
    np.ascontiguousarray(germs, dtype=np.int32).tofile(gf)
    np.ascontiguousarray(E.reshape(len(E), 9), dtype=np.int32).tofile(ef)
    subprocess.check_call([exe, gf, str(len(germs)), ef, str(len(E)), of])
    out = np.fromfile(of, dtype=np.int32).reshape(-1, 2).astype(np.int64)
    return out[:, 0], out[:, 1]


# ----------------------------------------------------------------------------
# 5. Signed depth spectrum  ->  f(n,k) for all k
# ----------------------------------------------------------------------------
def spectrum(m, signs, N):
    w = np.zeros(N + 1, dtype=np.int64)
    np.add.at(w, m, signs.astype(np.int64))
    return w


def f_from_spectrum(w, k):          # ordered k-tuples (with replacement)
    return sum(int(wj) * (j ** k) for j, wj in enumerate(w) if wj)


def f_multiset(w, k):               # unordered selections with replacement
    return sum(int(wj) * comb(j + k - 1, k) for j, wj in enumerate(w) if wj)


def f_set(w, k):                    # k distinct triangles, unordered
    return sum(int(wj) * comb(j, k) for j, wj in enumerate(w) if wj)


# ----------------------------------------------------------------------------
# 6. Driver
# ----------------------------------------------------------------------------
def solve(n, kmax=8, use_cpp=False, verbose=True):
    t0 = time.time()
    lines = grid_lines(n)
    P, E = triangles(n)
    N = len(P)
    L = comb((n + 1) ** 2, 3) - N
    assert L == collinear_triples(n)
    verts, vlines, starts = arrangement_vertices(n, lines)
    germs, signs = build_germs(lines, verts, vlines, starts)
    t1 = time.time()
    if verbose:
        print(f"[n={n}] M={(n+1)**2} points, {len(lines)} lines, N={N} triangles "
              f"(L={L} collinear), |V|={len(verts)} vertices, {len(germs)} cells "
              f"[setup {t1-t0:.1f}s]", flush=True)
    if use_cpp == 2:
        mc, mo = depths_cpp2(germs, E, lines)
    else:
        mc, mo = (depths_cpp if use_cpp else depths_numpy)(germs, E)
    t2 = time.time()
    if verbose:
        print(f"[n={n}] depth kernel {t2-t1:.1f}s", flush=True)
    wC, wO = spectrum(mc, signs, N), spectrum(mo, signs, N)
    # built-in consistency checks
    # every cell inside the square lies in some triangle (depth >= 1); cells with
    # depth 0 are the germs pointing outside the square and contribute 0 for k>=1
    assert int(wC[1:].sum()) == 1, "Euler characteristic of the square must be 1"
    wC[0] = 0; wO[0] = 0
    assert f_from_spectrum(wC, 1) == N, "f(n,1) must equal N"
    assert f_from_spectrum(wO, 1) == N
    # depth statistics
    isv = (germs[:, 3] == 0) & (germs[:, 4] == 0)
    vidx = np.where(isv)[0]
    jmax = int(mc[vidx].max())
    arg = vidx[np.argmax(mc[vidx])]
    cx, cy = Fraction(int(germs[arg, 0]), int(germs[arg, 2])), Fraction(int(germs[arg, 1]), int(germs[arg, 2]))
    # depth of the centre of the square
    cen = None
    for i in vidx:
        X, Y, W = (int(t) for t in germs[i, :3])
        if 2 * X == n * W and 2 * Y == n * W:
            cen = int(mc[i]); break
    res = dict(n=n, M=(n + 1) ** 2, lines=int(len(lines)), N=int(N), L=int(L),
               vertices=int(len(verts)), cells=int(len(germs)),
               max_depth=jmax, argmax=[str(cx), str(cy)], w_at_max=int(wC[jmax]),
               depth_center=cen,
               f_closed={k: f_from_spectrum(wC, k) for k in range(0, kmax + 1)},
               f_open={k: f_from_spectrum(wO, k) for k in range(0, kmax + 1)},
               f_multiset={k: f_multiset(wC, k) for k in range(1, kmax + 1)},
               f_set={k: f_set(wC, k) for k in range(1, kmax + 1)},
               spectrum_closed={int(j): int(w) for j, w in enumerate(wC) if w},
               spectrum_open={int(j): int(w) for j, w in enumerate(wO) if w},
               seconds=t2 - t0)
    return res, (P, E, verts, germs, signs, mc, mo)


# ----------------------------------------------------------------------------
# Faster exact kernel: evaluate the lexicographic half-plane test once per LINE
# of the arrangement (2 bits closed / 2 bits open), then each triangle is three
# table look-ups.  Identical results to depths_cpp / depths_numpy.
# ----------------------------------------------------------------------------
CPP_SRC2 = r"""
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
int main(int argc, char** argv) {
    long G = atol(argv[2]), Lc = atol(argv[4]), T = atol(argv[6]);
    std::vector<int32_t> g(G * 5), ln(Lc * 3), te(T * 3), out(G * 2);
    FILE* f = fopen(argv[1], "rb"); if (fread(g.data(), 4, G * 5, f) != (size_t)(G * 5)) return 1; fclose(f);
    f = fopen(argv[3], "rb"); if (fread(ln.data(), 4, Lc * 3, f) != (size_t)(Lc * 3)) return 1; fclose(f);
    f = fopen(argv[5], "rb"); if (fread(te.data(), 4, T * 3, f) != (size_t)(T * 3)) return 1; fclose(f);
    std::vector<uint8_t> cl(2 * Lc), op(2 * Lc);
    for (long i = 0; i < G; i++) {
        const int X = g[i*5], Y = g[i*5+1], W = g[i*5+2], dx = g[i*5+3], dy = g[i*5+4];
        for (long l = 0; l < Lc; l++) {
            const int a = ln[3*l], b = ln[3*l+1], c = ln[3*l+2];
            const int s0 = a * X + b * Y + c * W, s1 = a * dx + b * dy;
            cl[2*l]   = (s0 > 0) || (s0 == 0 && s1 >= 0);   // in closed half-plane  a x + b y + c >= 0
            cl[2*l+1] = (s0 < 0) || (s0 == 0 && s1 <= 0);   // in closed half-plane  a x + b y + c <= 0
            op[2*l]   = (s0 > 0) || (s0 == 0 && s1 > 0);    // open half-planes
            op[2*l+1] = (s0 < 0) || (s0 == 0 && s1 < 0);
        }
        int cc = 0, co = 0;
        const int32_t* e = te.data();
        for (long t = 0; t < T; t++, e += 3) {
            cc += cl[e[0]] & cl[e[1]] & cl[e[2]];
            co += op[e[0]] & op[e[1]] & op[e[2]];
        }
        out[2*i] = cc; out[2*i+1] = co;
    }
    f = fopen(argv[7], "wb"); fwrite(out.data(), 4, G * 2, f); fclose(f);
    return 0;
}
"""


def triangle_edge_codes(lines, E):
    """E[t,e]=(a,b,c) with inside a x+b y+c>=0.  Return codes 2*line_index+orient,
    orient=0 if (a,b,c) is a positive multiple of the normalized line, 1 otherwise."""
    T = E.shape[0]
    A, B, C = E[:, :, 0].reshape(-1), E[:, :, 1].reshape(-1), E[:, :, 2].reshape(-1)
    g = np.gcd(np.gcd(np.abs(A), np.abs(B)), np.abs(C))
    A, B, C = A // g, B // g, C // g
    flip = (A < 0) | ((A == 0) & (B < 0))
    An, Bn, Cn = np.where(flip, -A, A), np.where(flip, -B, B), np.where(flip, -C, C)
    # locate (An,Bn,Cn) in the sorted lines array
    key = lambda a, b, c: (a.astype(np.int64) * (4 * 10**6) + (b.astype(np.int64) + 10**3) * (2 * 10**3) + (c.astype(np.int64) + 10**3))
    lk = key(lines[:, 0], lines[:, 1], lines[:, 2])
    order = np.argsort(lk)
    idx = order[np.searchsorted(lk[order], key(An, Bn, Cn))]
    assert np.array_equal(lines[idx], np.stack([An, Bn, Cn], 1)), "edge line not found"
    return (2 * idx + flip.astype(np.int64)).astype(np.int32).reshape(T, 3)


def depths_cpp2(germs, E, lines, workdir="/tmp/helly_kernel2"):
    os.makedirs(workdir, exist_ok=True)
    exe = os.path.join(workdir, "kernel2")
    if not os.path.exists(exe):
        with open(os.path.join(workdir, "kernel2.cpp"), "w") as fh:
            fh.write(CPP_SRC2)
        subprocess.check_call(["g++", "-O3", "-march=native", "-o", exe, os.path.join(workdir, "kernel2.cpp")])
    codes = triangle_edge_codes(lines, E)
    np.ascontiguousarray(germs, dtype=np.int32).tofile(os.path.join(workdir, "germs.bin"))
    np.ascontiguousarray(lines, dtype=np.int32).tofile(os.path.join(workdir, "lines.bin"))
    np.ascontiguousarray(codes, dtype=np.int32).tofile(os.path.join(workdir, "tri.bin"))
    subprocess.check_call([exe, os.path.join(workdir, "germs.bin"), str(len(germs)),
                           os.path.join(workdir, "lines.bin"), str(len(lines)),
                           os.path.join(workdir, "tri.bin"), str(len(E)),
                           os.path.join(workdir, "out.bin")])
    o = np.fromfile(os.path.join(workdir, "out.bin"), dtype=np.int32).reshape(-1, 2).astype(np.int64)
    return o[:, 0], o[:, 1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--cpp", type=int, default=0, help="0: numpy kernel, 1: C++ kernel, 2: C++ line-table kernel (fastest)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    out = {}
    for n in args.n:
        res, _ = solve(n, args.kmax, args.cpp)
        out[n] = res
        print(f"  N={res['N']}  max depth={res['max_depth']} at {res['argmax']} "
              f"(w={res['w_at_max']}), depth(centre)={res['depth_center']}")
        for k in range(1, args.kmax + 1):
            print(f"  f({n},{k}) = {res['f_closed'][k]}   [open-interior: {res['f_open'][k]}]")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1, default=str)

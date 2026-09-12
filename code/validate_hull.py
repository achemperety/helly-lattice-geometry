import numpy as np
from scipy.spatial import ConvexHull
rng=np.random.default_rng(2026)
def hull_stats(npts, S, batch=200000):
    tot_area=0.0; tot_v=0; cnt=0
    while cnt<S:
        b=min(batch,S-cnt)
        P=rng.random((b,npts,2))
        for i in range(b):
            h=ConvexHull(P[i]); tot_area+=h.volume; tot_v+=len(h.vertices)
        cnt+=b
    return tot_area/S, tot_v/S
import sympy as sp
for npts,tA,th in [(3,sp.Rational(11,144),3),(4,sp.Rational(11,72),sp.Rational(133,36)),
                   (5,sp.Rational(79,360),sp.Rational(305,72)),(6,None,sp.Rational(281,60))]:
    a,v=hull_stats(npts,200000)
    print(f"n={npts}: E[A]_MC={a:.6f} (exact {float(tA) if tA is not None else float('nan'):.6f})   "
          f"E[h]_MC={v:.5f} (exact {float(th):.5f})")

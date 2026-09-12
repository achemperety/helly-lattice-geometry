import sys, numpy as np, helly_grid as H
for n in [int(a) for a in sys.argv[1:]]:
    lines=H.grid_lines(n); P,E=H.triangles(n); verts,vl,st=H.arrangement_vertices(n,lines)
    germs,signs=H.build_germs(lines,verts,vl,st)
    mc,mo=H.depths_cpp2(germs,E,lines) if n>=5 else H.depths_numpy(germs,E)
    isv=(germs[:,3]==0)&(germs[:,4]==0)
    j=mc.max(); at=np.where(mc==j)[0]
    dims=[(0 if isv[i] else (1 if signs[i]==-1 else 2)) for i in at]
    pts=sorted(set((int(germs[i,0])/int(germs[i,2]),int(germs[i,1])/int(germs[i,2])) for i in at))
    print(f"n={n}: max depth {j}, cells at max: {len(at)}, dims {sorted(set(dims))}, w={int(sum(signs[at]))}, points={pts}", flush=True)

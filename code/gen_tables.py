import json
from math import log
D={int(k):v for k,v in json.load(open('DATA.json')).items()}
LINES={1:6,2:20,3:62,4:140,5:306,6:536,7:938,8:1492}
GERMS={1:28,2:224,3:2180,4:11304,5:57716,6:176152,7:560076,8:1439968}
VERT={1:5,2:37,3:405,4:2225,5:11641,6:35677,7:114409,8:295701}
for n in D: D[n]['lines']=LINES[n]; D[n]['germs']=GERMS[n]; D[n]['vertices']=VERT[n]
for n in D: D[n]['cells']=GERMS[n]-2*LINES[n]+1
def g(x):  # digit grouping
    s=str(int(x)); out=''
    while len(s)>3: out=r'\,'+s[-3:]+out; s=s[:-3]
    return s+out
NS=list(range(1,9))

def w(fn,txt): open('out/'+fn,'w').write(txt.rstrip()+'%\n')

# ---- Table: arrangement & depth data -------------------------------------
rows=[]
dstr={1:r'$\{z_1\}$',2:r'$\{z_2\}$',3:r'$\{z_3\}\cup Q_3$',4:r'$\{z_4\}$',5:r'$Q_5$',
      6:r'$\{z_6\}$',7:r'$Q_7$',8:r'$\{z_8\}$'}
for n in NS:
    d=D[n]
    rows.append(' & '.join([str(n),g(d['M']),g(d['lines']),g(d['L']),g(d['N']),g(d['vertices']),
        g(d['cells']),g(d['germs']),g(d['J']),g(d['jmax']),str(d['wjmax']),dstr[n],
        g(d['mcentre']),f"{d['mcentre']/d['N']:.4f}"])+r' \\')
w('tab_arrangement.tex','\n'.join(rows))

# ---- Table: pairwise excess ----------------------------------------------
rows=[]
for n in NS:
    d=D[n]
    if 'k3' not in d: continue
    k3=d['k3']; f3=d['f']['3']; dl=k3-f3
    cells=[str(n),g(d['N']),g(d['f']['2']),g(k3),g(f3),g(dl),f"{100*dl/k3:.1f}"]
    if 'k4' in d:
        k4=d['k4']; f4=d['f']['4']
        cells+= [g(k4),g(f4),f"{100*(k4-f4)/k4:.1f}"]
    else: cells+=['---','---','---']
    rows.append(' & '.join(cells)+r' \\')
w('tab_defect.tex','\n'.join(rows))

# ---- Tables: f, f_open ----------------------------------------------------
for key,fn in [('f','tab_closed.tex'),('fo','tab_open.tex')]:
    rows=[' & '.join([str(n)]+[g(D[n][key][str(k)]) for k in range(2,6)])+r' \\' for n in NS]
    w(fn,'\n'.join(rows))
# appendix k=6..8
rows=[]
for n in NS:
    for k in (6,7,8):
        rows.append(' & '.join([str(n) if k==6 else '',str(k),g(D[n]['f'][str(k)]),g(D[n]['fo'][str(k)])])+r' \\')
    if n<8: rows.append(r'\addlinespace')
w('tab_appendix.tex','\n'.join(rows))
# multiset/set
rows=[' & '.join([str(n)]+[g(D[n][key][str(k)]) for k in (2,3,4) for key in ('fms','fset')])+r' \\' for n in NS]
w('tab_msset.tex','\n'.join(rows))
# ratios
rows=[]
for n in NS:
    d=D[n]; N=d['N']
    rows.append(' & '.join([str(n)]+[f"{d['f'][str(k)]/N**k:.4f}" for k in range(2,6)])+r' \\')
rows.append(r'\midrule')
rows.append(r'$p_k$ & $191/300=0.6367$ & $0.2526$ & $0.0863$ & $0.0275$ \\')
w('tab_ratios.tex','\n'.join(rows))
# touching
rows=[]
for n in NS:
    d=D[n]; t=d['f']['2']-d['fo']['2']; M=d['M']
    s2 = f"{d['ve']/(d['L']*M**3):.4f}" if d['L'] else '---'
    rows.append(' & '.join([str(n),g(t),g(d['vv']),g(d['ve']),f"{t/d['f']['2']:.4f}",
                            f"{d['vv']/M**5:.4f}",s2])+r' \\')
w('tab_touching.tex','\n'.join(rows))
print(open('out/tab_arrangement.tex').read())
print('---'); print(open('out/tab_defect.tex').read())
print('---'); print(open('out/tab_touching.tex').read())

// helly_defect.cpp -- pairwise intersection graph I_n of all lattice triangles on {0..n}^2
// (loops included: A_ii = 1), and the counts
//     hom(K3, I_n) = tr(A^3)  = ordered triples that are PAIRWISE intersecting,
//     hom(K4, I_n)            = ordered 4-tuples that are pairwise intersecting (only if n <= nK4),
// to be compared with the true Helly counts f(n,3), f(n,4).
// Exact integer predicate: closed triangles meet iff a vertex of one lies in the other
// (closed test) or two edges cross properly (strict test).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
using namespace std;
typedef long long ll;
struct Tri { int x[3], y[3]; };
static inline ll orient(ll ax, ll ay, ll bx, ll by, ll cx, ll cy) { return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax); }
static bool ptIn(const Tri& t, int px, int py) {           // closed
    ll d1 = orient(t.x[0],t.y[0],t.x[1],t.y[1],px,py), d2 = orient(t.x[1],t.y[1],t.x[2],t.y[2],px,py), d3 = orient(t.x[2],t.y[2],t.x[0],t.y[0],px,py);
    return (d1>=0&&d2>=0&&d3>=0)||(d1<=0&&d2<=0&&d3<=0);
}
static bool properCross(int ax,int ay,int bx,int by,int cx,int cy,int dx,int dy){
    ll d1=orient(ax,ay,bx,by,cx,cy), d2=orient(ax,ay,bx,by,dx,dy), d3=orient(cx,cy,dx,dy,ax,ay), d4=orient(cx,cy,dx,dy,bx,by);
    return ((d1>0&&d2<0)||(d1<0&&d2>0)) && ((d3>0&&d4<0)||(d3<0&&d4>0));
}
static bool sepAxis(const Tri& a, const Tri& b){   // some edge line of a has all of b on its closed outer side
    for(int i=0;i<3;i++){ int j=(i+1)%3, k=(i+2)%3;
        ll sk=orient(a.x[i],a.y[i],a.x[j],a.y[j],a.x[k],a.y[k]);   // side of a's third vertex (inner side)
        bool allOut=true; for(int v=0;v<3;v++){ ll s=orient(a.x[i],a.y[i],a.x[j],a.y[j],b.x[v],b.y[v]); if((sk>0&&s>0)||(sk<0&&s<0)){allOut=false;break;} }
        if(allOut) return true; }
    return false;
}
static bool interiorsDisjoint(const Tri& a, const Tri& b){ return sepAxis(a,b)||sepAxis(b,a); }
static bool shareVertex(const Tri& a, const Tri& b){ for(int i=0;i<3;i++)for(int j=0;j<3;j++) if(a.x[i]==b.x[j]&&a.y[i]==b.y[j]) return true; return false; }
static bool meet(const Tri& a, const Tri& b){
    for(int i=0;i<3;i++){ if(ptIn(b,a.x[i],a.y[i])) return true; if(ptIn(a,b.x[i],b.y[i])) return true; }
    for(int i=0;i<3;i++) for(int j=0;j<3;j++)
        if(properCross(a.x[i],a.y[i],a.x[(i+1)%3],a.y[(i+1)%3],b.x[j],b.y[j],b.x[(j+1)%3],b.y[(j+1)%3])) return true;
    return false;
}
int main(int argc,char**argv){
    int n=atoi(argv[1]); int nK4 = argc>2?atoi(argv[2]):4;
    int M=(n+1)*(n+1); vector<Tri> T;
    for(int a=0;a<M;a++)for(int b=a+1;b<M;b++)for(int c=b+1;c<M;c++){
        Tri t; int id[3]={a,b,c}; for(int i=0;i<3;i++){t.x[i]=id[i]/(n+1); t.y[i]=id[i]%(n+1);}
        if(orient(t.x[0],t.y[0],t.x[1],t.y[1],t.x[2],t.y[2])!=0) T.push_back(t);
    }
    size_t N=T.size(), Wd=(N+63)/64;
    ll edges=0, tvv=0, tve=0;
    if(nK4<0){   // pairs only: f2 and the touching decomposition, no adjacency matrix
        for(size_t i=0;i<N;i++) for(size_t j=i+1;j<N;j++) if(meet(T[i],T[j])){ edges++; if(interiorsDisjoint(T[i],T[j])){ if(shareVertex(T[i],T[j])) tvv++; else tve++; } }
        printf("n=%d N=%zu f2=%lld  touching ordered pairs: shared-vertex=%lld vertex-on-edge=%lld total=%lld  => f_open(n,2)=%lld\n", n, N, (ll)(N+2*edges), 2*tvv, 2*tve, 2*(tvv+tve), (ll)(N+2*edges)-2*(tvv+tve));
        return 0;
    }
    vector<uint64_t> A(N*Wd,0);
    for(size_t i=0;i<N;i++){ A[i*Wd+i/64]|=1ULL<<(i%64);
        for(size_t j=i+1;j<N;j++) if(meet(T[i],T[j])){ A[i*Wd+j/64]|=1ULL<<(j%64); A[j*Wd+i/64]|=1ULL<<(i%64); edges++; if(interiorsDisjoint(T[i],T[j])){ if(shareVertex(T[i],T[j])) tvv++; else tve++; } } }
    // hom(K2) = ordered pairs (incl. i=j) = N + 2*edges ; tr(A^3)
    __int128 tr3=0; 
    for(size_t i=0;i<N;i++){ const uint64_t* Ai=&A[i*Wd]; ll deg=0; for(size_t w=0;w<Wd;w++) deg+=__builtin_popcountll(Ai[w]); tr3+=deg; // j=i term
        for(size_t j=i+1;j<N;j++) if(Ai[j/64]>>(j%64)&1){ const uint64_t* Aj=&A[j*Wd]; ll c=0; for(size_t w=0;w<Wd;w++) c+=__builtin_popcountll(Ai[w]&Aj[w]); tr3+=2*c; } }
    printf("n=%d N=%zu f2=%lld  tr(A^3)=%lld  touching ordered pairs: shared-vertex=%lld vertex-on-edge=%lld total=%lld  => f_open(n,2)=%lld\n", n, N, (ll)(N+2*edges), (ll)tr3, 2*tvv, 2*tve, 2*(tvv+tve), (ll)(N+2*edges)-2*(tvv+tve));
    if(n<=nK4){
        __int128 k4=0; vector<uint64_t> C(Wd);
        for(size_t i=0;i<N;i++){ const uint64_t* Ai=&A[i*Wd];
            for(size_t j=i;j<N;j++) if(Ai[j/64]>>(j%64)&1){ const uint64_t* Aj=&A[j*Wd];
                for(size_t w=0;w<Wd;w++) C[w]=Ai[w]&Aj[w];
                ll s=0; for(size_t w=0;w<Wd;w++){ uint64_t m=C[w]; while(m){ int b=__builtin_ctzll(m); m&=m-1; size_t l=w*64+b; const uint64_t* Al=&A[l*Wd]; for(size_t v=0;v<Wd;v++) s+=__builtin_popcountll(Al[v]&C[v]); } }
                k4 += (i==j)? s : 2*s; } }
        printf("n=%d hom(K4,I_n) = %lld\n", n, (ll)k4);
    }
    return 0;
}

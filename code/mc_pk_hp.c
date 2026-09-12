/* mc_pk_hp.c -- high-precision parallel Monte Carlo for
 *
 *   p_k        = P( k uniform random triangles in the unit square share a point )
 *   ptilde_k   = P( they are PAIRWISE intersecting )
 *
 * Predicate: Lemma 4.10 of the paper (exact up to a null set).  A common point
 * exists iff some vertex of some T_i lies in every other T_j, or some proper
 * crossing of an edge of T_i with an edge of T_j lies in every other T_l.
 * Crossing points are never tested against the two triangles that own the
 * edges, and vertices never against their own triangle, so no membership test
 * is ever evaluated at a point lying exactly on the tested boundary.
 *
 * Build (macOS, Apple silicon; libomp from `brew install libomp`):
 *   clang -O3 -march=native -ffast-math -Xpreprocessor -fopenmp -lomp -o mc_pk_hp mc_pk_hp.c -lm
 * or without OpenMP (single core):
 *   clang -O3 -march=native -o mc_pk_hp mc_pk_hp.c -lm
 *
 * Run:   ./mc_pk_hp <k> <samples> [seed]
 * e.g.   ./mc_pk_hp 2 10000000000 20260909      (sanity check: must give 0.636667)
 *        ./mc_pk_hp 3 10000000000 20260909
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define KMAX 8

/* ---------------- xoshiro256++ ---------------- */
typedef struct { uint64_t s[4]; } rng_t;
static inline uint64_t rotl(uint64_t x, int k){ return (x<<k)|(x>>(64-k)); }
static inline uint64_t rng_next(rng_t *r){
    uint64_t *s = r->s, res = rotl(s[0]+s[3],23)+s[0], tt = s[1]<<17;
    s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3]; s[2]^=tt; s[3]=rotl(s[3],45);
    return res;
}
static inline double rng_u01(rng_t *r){ return (rng_next(r)>>11)*0x1.0p-53; }
static void rng_seed(rng_t *r, uint64_t seed){
    for(int i=0;i<4;i++){ seed += 0x9E3779B97F4A7C15ULL; uint64_t z=seed;
        z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL;
        r->s[i]=z^(z>>31); }
    for(int i=0;i<16;i++) rng_next(r);
}

/* ---------------- geometry ---------------- */
static inline double cross3(double ax,double ay,double bx,double by,double cx,double cy){
    return (bx-ax)*(cy-ay)-(by-ay)*(cx-ax);
}
/* point (px,py) inside triangle T (vertices T[0..5] = x0,y0,x1,y1,x2,y2) */
static inline int pt_in(const double *T,double px,double py){
    double d1=cross3(T[0],T[1],T[2],T[3],px,py);
    double d2=cross3(T[2],T[3],T[4],T[5],px,py);
    double d3=cross3(T[4],T[5],T[0],T[1],px,py);
    return ((d1>=0&&d2>=0&&d3>=0)||(d1<=0&&d2<=0&&d3<=0));
}

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s k samples [seed]\n",argv[0]); return 1; }
    int k = atoi(argv[1]);
    unsigned long long S = strtoull(argv[2],NULL,10);
    uint64_t seed = (argc>3)? strtoull(argv[3],NULL,10) : 20260909ULL;
    if(k<2||k>KMAX){ fprintf(stderr,"k must be in [2,%d]\n",KMAX); return 1; }

    unsigned long long hit_all=0, hit_pair=0;

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif
    fprintf(stderr,"k=%d  samples=%llu  threads=%d  seed=%llu\n",
            k,S,nthreads,(unsigned long long)seed);

#pragma omp parallel reduction(+:hit_all,hit_pair)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        rng_t rng; rng_seed(&rng, seed + 0x1000000ULL*(uint64_t)tid);
        double T[KMAX][6];
        unsigned long long ha=0, hp=0;

#pragma omp for schedule(static)
        for(long long it=0; it<(long long)S; ++it){
            for(int i=0;i<k;i++) for(int c=0;c<6;c++) T[i][c]=rng_u01(&rng);

            /* ---- pairwise intersection matrix ---- */
            int allpairs = 1;
            for(int i=0;i<k && allpairs;i++) for(int j=i+1;j<k && allpairs;j++){
                int meet=0;
                for(int v=0;v<3 && !meet;v++){
                    if(pt_in(T[j],T[i][2*v],T[i][2*v+1])) meet=1;
                    if(!meet && pt_in(T[i],T[j][2*v],T[j][2*v+1])) meet=1;
                }
                for(int a=0;a<3 && !meet;a++){
                    int a2=(a+1)%3;
                    double p1x=T[i][2*a],p1y=T[i][2*a+1],p2x=T[i][2*a2],p2y=T[i][2*a2+1];
                    for(int b=0;b<3 && !meet;b++){
                        int b2=(b+1)%3;
                        double q1x=T[j][2*b],q1y=T[j][2*b+1],q2x=T[j][2*b2],q2y=T[j][2*b2+1];
                        double d1=cross3(p1x,p1y,p2x,p2y,q1x,q1y);
                        double d2=cross3(p1x,p1y,p2x,p2y,q2x,q2y);
                        double d3=cross3(q1x,q1y,q2x,q2y,p1x,p1y);
                        double d4=cross3(q1x,q1y,q2x,q2y,p2x,p2y);
                        if(((d1>0&&d2<0)||(d1<0&&d2>0))&&((d3>0&&d4<0)||(d3<0&&d4>0))) meet=1;
                    }
                }
                if(!meet) allpairs=0;
            }
            if(allpairs) hp++;

            /* ---- common point (Lemma 4.10) ---- */
            int common = 0;
            if(allpairs){                       /* a common point forces pairwise */
                for(int i=0;i<k && !common;i++)
                    for(int v=0;v<3 && !common;v++){
                        double px=T[i][2*v],py=T[i][2*v+1]; int ok=1;
                        for(int j=0;j<k && ok;j++) if(j!=i && !pt_in(T[j],px,py)) ok=0;
                        if(ok) common=1;
                    }
                for(int i=0;i<k && !common;i++) for(int j=i+1;j<k && !common;j++)
                    for(int a=0;a<3 && !common;a++){
                        int a2=(a+1)%3;
                        double p1x=T[i][2*a],p1y=T[i][2*a+1],p2x=T[i][2*a2],p2y=T[i][2*a2+1];
                        for(int b=0;b<3 && !common;b++){
                            int b2=(b+1)%3;
                            double q1x=T[j][2*b],q1y=T[j][2*b+1],q2x=T[j][2*b2],q2y=T[j][2*b2+1];
                            double d1=cross3(p1x,p1y,p2x,p2y,q1x,q1y);
                            double d2=cross3(p1x,p1y,p2x,p2y,q2x,q2y);
                            double d3=cross3(q1x,q1y,q2x,q2y,p1x,p1y);
                            double d4=cross3(q1x,q1y,q2x,q2y,p2x,p2y);
                            if(!(((d1>0&&d2<0)||(d1<0&&d2>0))&&((d3>0&&d4<0)||(d3<0&&d4>0)))) continue;
                            double tt = d3/(d3-d4);
                            double zx = p1x + tt*(p2x-p1x), zy = p1y + tt*(p2y-p1y);
                            int ok=1;
                            for(int l=0;l<k && ok;l++) if(l!=i && l!=j && !pt_in(T[l],zx,zy)) ok=0;
                            if(ok) common=1;
                        }
                    }
            }
            if(common) ha++;
        }
        hit_all += ha; hit_pair += hp;
    }

    double pk = (double)hit_all/(double)S, pt = (double)hit_pair/(double)S;
    double sk = sqrt(pk*(1-pk)/(double)S), st = sqrt(pt*(1-pt)/(double)S);
    printf("k=%d samples=%llu seed=%llu\n",k,S,(unsigned long long)seed);
    printf("  p_%d      = %.9f  +/- %.9f   (hits %llu)\n",k,pk,sk,hit_all);
    printf("  ptilde_%d = %.9f  +/- %.9f   (hits %llu)\n",k,pt,st,hit_pair);
    if(k==2) printf("  [check] exact p_2 = 191/300 = %.9f   deviation %.2f sigma\n",
                    191.0/300.0, (pk-191.0/300.0)/sk);
    return 0;
}

# Helly-Type Combinatorial Configurations of Lattice Triangles: Exact Structure and Asymptotics

Official repository for the paper of the same name: source, computational
verification scripts, and all exact data.

> **Current version: v2 (September 2026).** v2 adds two new results — the exact
> value $p_2 = 191/300$ and the exact order $\tau_2(n) = \Theta(n^{10}\log n)$ —
> and corrects an incorrect statement about the deepest point of the arrangement.

## Status

- **Preprint:** Zenodo, DOI [10.5281/zenodo.22680094](https://doi.org/10.5281/zenodo.22680094)
- **Journal:** submitted to *Discrete & Computational Geometry* (Springer)



## Abstract

Let $G_n={0,\dots,n}^2$ and let $\mathcal T(n)$ be the set of non-degenerate
closed triangles with vertices in $G_n$. We study $f(n,k)$, the number of ordered
$k$-tuples of such triangles with a common point.

The main structural result is an **Euler-characteristic transfer theorem**: if
$\mathcal A_n$ is the arrangement of all lines through two grid points and
$m_n(\sigma)$ is the number of triangles containing the cell $\sigma$, then

$$f(n,k)=\sum_{\sigma}(-1)^{\dim\sigma}m_n(\sigma)^k\qquad\text{for every }k\ge1 .$$

Hence $f(n,\cdot)$ is encoded by a single signed *depth spectrum* $w_n$, has a
rational generating function in $k$, and satisfies a linear recurrence whose
characteristic roots are depth values; the same identity holds for open
interiors, multisets and sets. This makes it possible to compute $f(n,k)$ for
**all $k$ at once** from one pass over the arrangement, in exact integer
arithmetic, instead of the $N(n)^k$ cost of direct enumeration.

Asymptotically $f(n,k)=p_k6^{-k}n^{6k}(1+O(1/n))$, where $p_k$ is the probability
that $k$ independent uniform random triangles in a square share a point. We
evaluate the first of these constants exactly,

$$p_2=\frac{191}{300},$$

via a separating-line identity valid for any absolutely continuous planar sample
distribution; by-products are $\mathbb E[h_5]=\tfrac{305}{72}$,
$\mathbb E[h_6]=\tfrac{281}{60}$, $\mathbb E[A_4]=\tfrac{11}{72}$,
$\mathbb E[A_5]=\tfrac{79}{360}$ for uniform points in a square. Since $p_2$ is
rational, the rationality route to non-polynomiality is closed at $k=2$. We also
determine the exact order $\tau_2(n)=\Theta(n^{10}\log n)$ of the discrepancy
between closed and open intersection counts, prove that $f(n,1)=N(n)$ admits no
polynomial, quasi-polynomial or rational-generating-function closed form, and
quantify the failure of the pairwise intersection graph through the *pairwise
excess* $\delta_k(n)$.

## Repository layout

```
code/
  helly_grid.py          exact solver: depth spectrum of A_n, hence f(n,k) for all k
  helly_defect.cpp       pairwise intersection graph: hom(K3,I_n), hom(K4,I_n), vv/ve split
  centre_depth.py        discrete Wendel formula for m_n(z_n), checked against the solver
  check_deepest.py       locates the deepest points D*_n
  verify_helly.py        brute-force cross-check of f(n,k) for small n
  p2_exact.py            symbolic proof-by-computation of p_2 = 191/300  [new in v2]
  validate_hull.py       independent MC check of E[A_N], E[h_N]          [new in v2]
  mc_pk_hp.c             parallel high-precision MC for p_k and pairwise  [new in v2]
  mc_continuum.py        vectorised MC for p_k (original implementation)
  mc_pairwise.py         MC for the pairwise probabilities
  verify_mc.py           MC predicate vs. exact polygon clipping
  gen_tables.py          regenerates paper/tables/ from data/DATA.json
data/
  DATA.json              consolidated: spectra summaries, f/f°/f^ms/f^set, defect, touching
  helly_spectra.json     full signed depth spectra w_n, w°_n for 1 <= n <= 8
  results_n*.json        per-n solver output (arrangement data, spectra, timings)
  helly_exact_values.csv f, f°, f^ms, f^set for 1 <= n <= 8, 1 <= k <= 8
  defect_n*.txt          hom(K3,I_n), hom(K4,I_n), touching decomposition
  deepest_n1_8.txt       deepest points of K_n
  p2_exact_output.txt    output of p2_exact.py with the MC confirmation
  mc_k*.txt              raw Monte Carlo logs
```



## Reproducing the results

Verify the exact value of $p_2$ (about two minutes; every `assert` must pass —
the script re-derives $\mathbb E[\alpha]=1/2$ and $\mathbb E[\alpha^2]=133/432$,
the latter equivalent to Sylvester's $25/36$, before computing $p_2$):

```bash
python3 code/p2_exact.py
python3 code/validate_hull.py        # independent MC check of the by-products
```

Confirm $p_2$ numerically (must land on $0.6366\overline{6}$ within ~1 sigma):

```bash
clang -O3 -march=native -o mc_pk_hp code/mc_pk_hp.c -lm
./mc_pk_hp 2 1000000000
```

Recompute the exact spectra (cost grows like $n^{14}$; $n=8$ takes about seven
minutes on one core):

```bash
python3 code/helly_grid.py --n 8 --kmax 8 --cpp 2 --json data/results_n8.json
python3 code/centre_depth.py         # discrete Wendel formula vs. solver
python3 code/check_deepest.py 1 2 3 4 5 6 7 8
```

Recompute the pairwise intersection graph:

```bash
clang++ -O3 -march=native -o helly_defect code/helly_defect.cpp
./helly_defect 6 0                   # hom(K3, I_6) = 2 501 508 150 344
./helly_defect 4 4                   # also hom(K4, I_4)
```

Regenerate every table in the paper from the data (no number is transcribed by
hand):

```bash
python3 code/gen_tables.py
```



## Selected exact values


| $n$ | $N(n)$ | $j^*_n$ | deepest points $D^*_n$                | $f(n,2)$      | $f(n,3)$            |
| --- | ------ | ------- | ------------------------------------- | ------------- | ------------------- |
| 2   | 76     | 56      | centre                                | 5 420         | 312 388             |
| 3   | 516    | 240     | centre **and** 4 lattice neighbours   | 226 972       | 69 489 756          |
| 5   | 6 768  | 2 523   | 4 lattice neighbours (centre: 2 396)  | 34 737 956    | 112 710 050 556     |
| 7   | 40 120 | 13 316  | 4 lattice neighbours (centre: 12 512) | 1 150 885 648 | 20 417 231 442 376  |
| 8   | 82 608 | 27 776  | centre                                | 4 788 100 488 | 170 981 082 265 560 |


Full tables for $1\le n\le 8$ and $1\le k\le 8$ are in `data/helly_exact_values.csv`
and in the paper.

## Citation

```bibtex
@misc{baiandin2026helly,
  author    = {Baiandin, Ruslan},
  title     = {Helly-Type Combinatorial Configurations of Lattice Triangles:
               Exact Structure and Asymptotics},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22680094},
  url       = {https://doi.org/10.5281/zenodo.22680094}
}
```



## License

MIT — see [LICENSE](LICENSE).
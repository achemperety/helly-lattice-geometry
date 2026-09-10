# Helly-Type Combinatorial Configurations of Lattice Triangles: Exact Structure and Asymptotics

This repository contains the official computational verification scripts and raw simulation data for the research paper: **"Helly-Type Combinatorial Configurations of Lattice Triangles: Exact Structure and Asymptotics"**.

## Status
* **Preprint:** Published on Zenodo with permanent DOI: [10.5281/zenodo.22680095](https://doi.org)
* **Journal Status:** Submitted to *Discrete & Computational Geometry* (Springer, Q1).

## Abstract
We establish an exact algebraic invariant for the counting function $f(n,k)$ of intersecting families of lattice triangles using an Euler-characteristic transfer theorem on the grid line arrangement. This framework allows computing the exact integer depths of up to $1.44 \times 10^6$ cells and $8.3 \times 10^4$ triangles simultaneously, avoiding combinatorial explosion. Additionally, we provide tight bounds for the continuum constants $p_k$ and investigate the non-polynomial boundary-lattice mechanisms.

## Repository Contents
* `mc_continuum.py` — Highly optimized vectorised Monte Carlo simulation script utilizing barycentric coordinate tests to approximate continuous Helly intersection probabilities ($p_2, p_3, p_4$) with strict standard errors.
* `results_n8.json` — Raw combinatorial spectrum data containing exact integer depths computed for grid sizes up to $n=8$ (evaluating over $1.4 \times 10^{11}$ geometric predicates).

## Citation
If you use this code or data in your research, please cite the preprint as follows:
```bibtex
@misc{baiandin2026helly,
  author       = {Baiandin, Ruslan},
  title        = {Helly-Type Combinatorial Configurations of Lattice Triangles: Exact Structure and Asymptotics},
  month        = sep,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22680095},
  url          = {https://doi.org}
}
```

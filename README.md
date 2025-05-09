# Spectral Networks and the Riemann Hypothesis

## Overview

This project investigates the Riemann Hypothesis (RH) through a novel spectral network framework that interprets the nontrivial zeros of the Riemann zeta function as **modal attractors** emerging from the **structure–information duality** in frequency space.

We show numerically and theoretically that:

- Minimizing a spectral Lagrangian \( \mathcal{L}[\phi(x,t)] \)
- Minimizing boundary information entropy \( S(T) \)
- And reducing residual mismatch \( \delta(x) = \frac{\pi(x)}{x} - \rho(x) \)

all converge to the same condition:  
**Modal frequencies align with the nontrivial Riemann zeros**  
→ **ℜ(s) = 1/2 is the unique spectral-entropy equilibrium.**

---

## Files

| File | Description |
|------|-------------|
| `rh3.py` | Main simulation using ζ zeros as modal frequencies (HPO2) |
| `rh4.py` | Final verified version with structure-information duality output |
| `rh4_offcritical.py` | Control test with off-critical (σ ≠ 1/2) frequency perturbations |
| `README.md` | This file |
| `entropy_and_residual.png` | Visualizes entropy collapse and residual compression |


Spectral Networks and the Structural Origin of the Riemann Hypothesis
](https://zenodo.org/records/15376607)


## How to Run

```bash
python3 rh4.py

# thesis-algebrics

Symbolic derivation and verification of the model and equilibrium analysis for
the doctoral thesis *Altruism in Networked Public Good Games*. Every closed form
reported in the **Model** and **Equilibrium Analysis** chapters is re-derived and
checked symbolically with [SymPy](https://www.sympy.org); each script prints
`PASS`/`FAIL` per claim and asserts at the end.

## Model

Material payoff and (impact-written) moral component
```
X_i = e - theta_i c_i + (beta/d_i) * sum_{j in G_i} c_j ,   theta_i = 1 - beta/d_i
Y_i = ( 1 + sum_{j in G_i} beta/d_j ) c_i        # impact multiplier, inert under CD
U_i = X_i^{A_i} Y_i^{B_i} ,   A_i = 1 - B_i      # B_i = prosocial weight
```
Interior first-order condition: `c_i = kappa_i X_i`, with `kappa_i = B_i/(A_i theta_i)`.

## Files

| File | What it verifies |
|------|------------------|
| `model_preferences.py`   | CD/CES preference setup; the impact multiplier cancels from the CD first-order condition; scale-invariance of `B_i/A_i`. |
| `model_nonetwork.py`     | Complete-graph (no-network) primitives: `d_i=n`, common `theta`. |
| `model_cycle.py`         | Cycle primitives: `d_i=3` for all `n`; `C_3=K_3`. |
| `model_star.py`          | Star primitives: center `d=n`, leaf `d=2`. |
| `model_ces.py`           | CES ratio rule `(Y_i/X_i)^{rho-1}=delta theta/((1-delta)Lambda)`; CD limit. |
| `model_warmglow.py`      | Additive-baseline warm-glow benchmark `Y=T+c`; the baseline survives the FOC. |
| `ces_equilibrium_derivations.py`     | CES best response, symmetric and asymmetric Nash, feasibility, comparative statics. |
| `cd_equilibrium_derivations.py`      | Cobb–Douglas optimality and impact-multiplier irrelevance; best response; symmetric/asymmetric equilibria; scale-invariance. |
| `cd_vs_warmglow_derivations.py`      | Equilibrium comparison of the impact-based model vs. the baseline-`T` warm-glow model (symmetric and asymmetric). |
| `network_derivations.py`             | Cycle and star equilibria; center–periphery ordering; position–preference threshold. |
| `nN3_derivations.py`                 | Explicit `N=3` forms (no-network = cycle, and star). |
| `network_asymmetric_derivations.py`  | Asymmetric (heterogeneous `B_i`) equilibria: cycle `C_3`/`C_4`, star (general leaves), spectral radius, ordering. |

## Requirements

```
python >= 3.10
sympy
```
```
pip install sympy
```

## Run

Run any script directly; each is self-contained and prints its check results:
```bash
python cd_equilibrium_derivations.py
python network_asymmetric_derivations.py
# ... etc.
```
Run all:
```bash
for f in *.py; do echo "== $f =="; python "$f"; done
```

## Output

Console only: a list of `PASS` lines and a final `ALL CHECKS PASSED.` (the scripts
`assert` on failure, so a non-zero exit code flags any broken identity). No files
are written.

## Conventions

`B_i` is the **prosocial weight** (the sole moral primitive of interest). The
bracket `1 + sum beta/d_j` is the **impact multiplier** `Lambda`; under
Cobb–Douglas it factors out of the first-order condition and therefore does not
affect contributions. Experimental parameters used in numerical sanity checks:
`e = 15`, `n = 3`, `beta = 1.5`.

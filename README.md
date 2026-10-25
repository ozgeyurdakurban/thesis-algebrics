# thesis-algebrics

Symbolic derivation and verification of the model and equilibrium analysis for
the doctoral thesis *Altruism in Networked Public Good Games*. Every closed form
reported in the **Model** and **Equilibrium Analysis** chapters is re-derived and
checked symbolically with [SymPy](https://www.sympy.org); each script prints
`PASS`/`FAIL` per claim and asserts at the end.

## Chapter 2 re-derivation (`ch2/`)

The folder [`ch2/`](ch2/) contains the step-by-step **re-derivation** of Chapter 2:
each closed form is derived with SymPy (not typed in and checked), compared with the
thesis text, and exported as LaTeX. See [`ch2/README.md`](ch2/README.md) for the
roadmap, the new results and the audit notes. Run `python ch2/run_all.py`.

## Model (thesis version of 25 October 2026)

The general model is CES; Cobb-Douglas is its sigma = 1 reference specification.
```
X_i = W_i - theta_i c_i ,  W_i = e + (beta/d_i) sum_{j in G_i} c_j ,  theta_i = 1 - beta/d_i   # price of giving
Y_i = c_i + lambda_i sum_{j in G_i} (beta/d_j) c_i = Lambda~_i c_i                            # moral component
Lambda~_i = 1 + lambda_i (Lambda_i - 1),  Lambda_i = 1 + sum_{j in G_i} beta/d_j               # (effective) impact multiplier
U_i = [(1-B_i) X_i^rho + B_i Y_i^rho]^(1/rho),  sigma = 1/(1-rho)                              # B_i = prosocial weight
c_i = min{e, (pi_i/theta_i) W_i},  pi_i = 1/(1 + ((1-B_i)/B_i)^sigma p_i^(sigma-1)),  p_i = theta_i/Lambda~_i
a_i = A_i (X_i/Y_i)^(1/sigma),  A_i = lambda_i B_i/(1-B_i)                                     # altruism (willingness to pay)
```
`lambda_i = 0` is the act-based and `lambda_i = 1` the impact-based specification. At `sigma = 1`,
`pi_i = B_i` and the impact weight drops out. The current derivations are in [`ch2/`](ch2/)
(`s01`-`s10`, 262 checks); the scripts listed below are the earlier verification of the
5th-monitoring-report formulas and are kept for reference.

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
| `ces_network_derivations.py`         | CES on the cycle and the star: network FOC `c_i = kappa_i X_i` with `kappa_i = ((1-delta)/(delta theta_i))^sigma Lambda_i^(sigma-1)`; CD limit; sign of `d log kappa / d log Lambda = sigma - 1`; act- vs impact-based coincidence at `sigma = 1`; positional ordering over an `(a, beta, sigma)` grid. Requires `numpy`. |

## Requirements

```
python >= 3.10
sympy
numpy   # ces_network_derivations.py only
```
```
pip install sympy numpy
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

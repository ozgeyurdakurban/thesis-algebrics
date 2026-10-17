# Chapter 2 re-derivation (`ch2/`)

Independent re-derivation of every analytical result in Chapter 2
(*Altruism in Networked Public Good Games: Theory and Simulations*).

The older scripts in the repository root **verify** formulas that were typed in.
The scripts here **derive** each object with SymPy (`solve`, `diff`, `limit`),
print the derived expression, compare it with the expression reported in the
thesis, and export the derived closed forms as LaTeX.

```
pip install sympy numpy
cd ch2
python run_all.py          # or any single sNN_*.py
```

Each script writes
* `results/<script>.txt` – the full console log (derived expressions + PASS/FAIL);
* `tex/<script>.tex` – the derived closed forms as LaTeX (auto-generated).

A script exits with a non-zero status if any check fails.

## Model (notation used throughout)

```
X_i = e - theta_i c_i + (beta/d_i) S_i ,  S_i = sum_{j in G_i} c_j ,  theta_i = 1 - beta/d_i
Y_i = Lambda_i c_i ,  Lambda_i = 1 + sum_{j in G_i} beta/d_j        (impact multiplier)
Cobb-Douglas:  U_i = X_i^{A_i} Y_i^{B_i} ,   b_i = B_i/(A_i+B_i)  (= B_i under A_i+B_i = 1)
CES:           U_i = [delta_i X_i^rho + (1-delta_i) Y_i^rho]^{1/rho} ,  sigma = 1/(1-rho)
SE = sum_i X_i = n e + sum_i (s_i - 1) c_i ,  s_i = sum_{k in N[i]} beta/d_k
```

## Roadmap and status

| Step | Script(s) | Content | Status |
|---|---|---|---|
| 1a | `s01_primitives.py` | primitives on any network; SE identity; `Lambda_i = 1 + s_i - beta/d_i`; closed forms on K_n, C_n, S_n; admissible beta bands | done |
| 1b | `s02_cd_best_response.py` | CD individual problem: best response as a Cobb–Douglas demand `c = b W/theta`; Lambda-irrelevance; SOC; corners; complements | done |
| 1c | `s03_cd_symmetric.py` | CD, common B: K_n, C_n, S_n (general n); ceilings; mixed regime; comparative statics incl. group size; ordering; uniqueness | done |
| 2 | `s04_cd_asymmetric.py` | CD, heterogeneous B_i: K_n (general n, n=2,3), C_n (C_3, C_4, general), S_n (general n, n=3), threshold, mixed regimes | next |
| 3 | `s05_ces_*.py` | CES: individual problem, symmetric and asymmetric equilibria on K_n, C_n, S_n; sigma comparative statics; act vs impact identification | planned |
| 4 | `s06_warmglow.py`, `s07_social_efficiency.py` | baseline-T comparison; SE in equilibrium | planned |
| 5 | `thesis-simulations` (`ch2/`) | all simulations re-derived, seeded, tested against the closed forms | planned |

## New results obtained in Step 1 (not in the 5th-report version)

* Best response in demand form: `c_i = (b_i/theta_i)(e + (beta/d_i) S_i)`; with
  `A+B=1` it is **linear in B_i**: `alpha_i = B_i e/theta_i`, `gamma_i = B_i beta/(d_i - beta)`.
* Group-size effect on K_n: `dc*/dn = -B(1-B) beta e / D_K^2 < 0`.
* Star, common B, **general n**: closed forms for centre and leaf,
  gap `(n-2) B(1-B) beta e / Delta_S`, `Delta_S = (n-beta)(2-beta) - (n-1) B^2 beta^2`.
* **Closed-form leaf ceiling** (was numerical): positive root of
  `beta(n+beta(n-1)) B^2 + 2(n-beta) B - (n-beta)(2-beta) = 0`; equals 1/6 at n=3, beta=1.5.
* Centre vs complete-network member: `(n-1)(n-2) B^2 (1-B) beta^2 e / (D_K Delta_S) > 0`.
* Contraction lemma: `rho(Gamma G) < 1` implies a unique equilibrium **including corners**
  (weighted sup-norm + componentwise truncation).

## Audit of the 5th-report Chapter 2 (items to fix at integration, Step 6)

Confirmed correct by independent derivation: CD symmetric K_n and cycle closed forms,
ceilings, comparative statics in (B, e, beta); star n=3 gap and threshold `B_1*`;
CES network decision coefficient; spectral radius of the star; CES numbers at sigma = 1.5, 2.

To correct or qualify:
1. Ch2 introduction, end of the topologies paragraph: incomplete sentence
   ("The Cobb--Douglas model within a general elasticity-of-substitution family.").
2. Model: `theta_i in (0,1)` needs only `0 < beta < d_i`; `beta > 1` is a separate
   (social-productivity) requirement. The sentence "For theta in (0,1) we require 1 < beta < d_i" conflates the two.
3. CES comparative statics (Prop. `ces_compstat`): "beta raises contributions through both the
   material return and the impact multiplier" holds only for sigma >= 1; under gross complements
   the Lambda channel lowers contributions (Prop. `ces_multiplier_sign`).
4. CES simulations: "contributions decline monotonically in sigma" is parameter-specific.
   `d log kappa~_i / d sigma = log(kappa_i Lambda_i)` (impact-based), so contributions fall in sigma
   iff `kappa_i Lambda_i < 1`; e.g. on the regular graph at beta = 1.5 this fails for B in (0.2, 0.25).
   To be derived formally in Step 3.
5. Simulations, Monte Carlo: the statistic labelled Pr(P > C) is computed as
   Pr(mean of the two leaves > centre); the text should say so.
6. H4 (periphery > centre > regular) is an n = 3 result; for n >= 4 the centre vs cycle-member
   ranking is parameter dependent (Step 1c). The thesis states it only for the experiment, which is fine,
   but the general-n wording in "Comparison across topologies" should be qualified.
7. `milgrom1990` (supermodular games) to be added to the bibliography if the contraction /
   least–greatest equilibrium argument is used.

# Chapter 2 re-derivation (`ch2/`)

Independent re-derivation of every analytical result in Chapter 2
(*Altruism in Networked Public Good Games: Theory and Simulations*).

The older scripts in the repository root **verify** formulas that were typed in.
The scripts here **derive** each object with SymPy (`solve`, `diff`, `limit`),
print the derived expression, compare it with the expression reported in the
thesis, and export the derived closed forms as LaTeX. Small-n solutions are, in
addition, checked against a **formula-free** Nash equilibrium obtained by
maximising the actual utility functions (`netlib.nash_utility`).

```
pip install sympy numpy
cd ch2
python run_all.py          # or any single sNN_*.py
```

Each script writes `results/<script>.txt` (full log: derived expressions and
PASS/FAIL) and `tex/<script>.tex` (derived closed forms as LaTeX), and exits with
a non-zero status if any check fails.

## Model and notation

```
X_i = W_i - theta_i c_i ,  W_i = e + (beta/d_i) S_i ,  S_i = sum_{j in G_i} c_j ,  theta_i = 1 - beta/d_i
Y_i = c_i + lambda_i sum_{j in G_i} (beta/d_j) c_i = Lambda~_i c_i
      Lambda_i  = 1 + sum_{j in G_i} beta/d_j          (impact multiplier)
      Lambda~_i = 1 + lambda_i (Lambda_i - 1)           (effective impact multiplier)
      lambda_i >= 0 impact weight: 0 = act-based, 1 = impact-based
General model (CES):  U_i = [(1-B_i) X_i^rho + B_i Y_i^rho]^{1/rho} ,  sigma = 1/(1-rho),  B_i = prosocial weight
Reference spec (CD):  U_i = X_i^{1-B_i} Y_i^{B_i}   (sigma = 1)
Best response: c_i = min{ e, (pi_i/theta_i) W_i }
               pi_i = 1/(1 + ((1-B_i)/B_i)^sigma p_i^(sigma-1)),  p_i = theta_i/Lambda~_i ;  pi_i = B_i at sigma = 1
Altruism:      a_i = A_i (X_i/Y_i)^(1/sigma),  A_i = lambda_i B_i/(1-B_i)   (willingness to pay for others' benefit)
SE = sum_i X_i = n e + sum_i (s_i - 1) c_i ,  s_i = sum_{k in N[i]} beta/d_k
```

## Scripts

| Script | Content | Checks |
|---|---|---|
| `s01_primitives.py` | primitives on any network; SE identity; `Lambda_i = 1 + s_i - beta/d_i`; K_n, C_n, S_n closed forms; admissible beta bands | 14 |
| `s02_cd_best_response.py` | CD individual problem: demand form `c = B W/theta`, Lambda-irrelevance, SOC, corners, complements | 21 |
| `s03_cd_symmetric.py` | CD, common B, general n: K_n, C_n, S_n; ceilings (closed-form leaf ceiling), mixed regime, group-size effect, ordering, uniqueness | 40 |
| `s04_cd_heterogeneous.py` | CD, heterogeneous B_i, general n: K_n; any network `c* = M alpha`, `dc_i/dB_j = M_ij c_j/B_j`; C_n; S_n; threshold `B_C*(B_P, n)`; active set | 34 |
| `s05_ces_best_response.py` | CES individual problem: share `pi_i`, limits, moral channel (sigma-1), substitution channel threshold `theta/(theta+Lambda)`, act vs impact, Hicks–Slutsky decomposition | 24 |
| `s06_ces_homogeneous.py` | CES, common B: K_n, C_n (closed-form ceilings), S_n with position-specific shares; robustness of the positional ordering | 26 |
| `s07_ces_heterogeneous.py` | CES, heterogeneous B_i: K_n, any network (`dc_i/dB_j` with sigma factor), S_n (+ threshold), C_n; utility-maximisation check | 10 |
| `s08_small_n.py` | Appendix A: explicit K_2, K_3, K_4, C_4, S_3, S_4 (CD and CES, common and heterogeneous); S_2 = K_2, C_3 = K_3; three-route cross-validation | 21 |
| `s09_warmglow.py` | additive-baseline warm-glow benchmark (common and heterogeneous weights) | 10 |
| `s10_impact_altruism.py` | general model with impact weight: moral component as agency-restricted AGH (1998) utility (lambda = alpha/g); general best response; lambda-invariance iff sigma = 1; dc/dlambda on networks; price elasticity of giving; altruism (CFS 2008 WTP, MAT ordering, OVX-type decomposition); identification of (sigma, lambda) from the n = 3 design and the Star rotation; exact leaf-centre condition and its thresholds | 59 |
| `common.py`, `netlib.py` | symbolic helpers; numerical solvers (closed form on the active set, BR iteration, formula-free utility maximisation) | |

## Results that are new relative to the 5th-report version

Cobb–Douglas
* Best response as a demand: `c_i = (B_i/theta_i) W_i` — linear in `B_i`; `alpha_i = B_i e/theta_i`, `gamma_i = B_i beta/(d_i - beta)`.
* Group-size effect on K_n: `dc*/dn = -B(1-B) beta e / D_K^2 < 0`.
* Star with common B for **general n**; closed-form leaf ceiling (1/6 at n=3, beta=1.5); centre vs K_n member `(n-1)(n-2)B^2(1-B)beta^2 e/(D_K Delta_S)`.
* Heterogeneous star for general n: centre depends on leaves only through `Sigma_L`; threshold `B_C* = 2B_P(n-beta)/[n(2-beta) + (n-2)B_P beta]`.
* Any network: `dc_i*/dB_j = M_ij c_j*/B_j > 0` (own elasticity `M_ii >= 1`).
* Cycle C_n, n >= 4: contributions are **not** ordered by B_i alone (counterexample on C_4).
* Contraction lemma: `rho(Gamma G) < 1` gives a unique equilibrium including corners.

CES
* Share form: the whole CD apparatus applies with `B_i -> pi_i`; `p_i = theta_i/Lambda_i` is the price of the moral good.
* Substitution channel: own giving rises with sigma iff `B_i > theta_i/(theta_i + Lambda_i)`.
* Hicks–Slutsky decomposition: `dlog c/dlog Lambda = sigma(1-pi) [substitution] + pi [income] - 1 [conversion] = (sigma-1)(1-pi)`; Cobb–Douglas neutrality = unit price elasticity of the moral good.
* Closed-form ceilings on K_n/C_n: `odds = theta Lambda^((1-sigma)/sigma) beta^(-1/sigma)` (impact), `theta beta^(-1/sigma)` (act).
* Any network: `dc_i*/dB_j = M~_ij c_j* sigma(1-pi_j)/(B_j(1-B_j)) > 0` — H1 robust at every sigma.
* Ordering: at n = 3 leaf > centre at every sigma (both specs); impact-based reversal possible only for n >= 4, low beta, sigma >= 2. Centre > regular fails under impact-based complements at small B (limit `(Lambda_K/Lambda_C)^(1-sigma)`).

## General model with an impact weight (25 Oct 2026)

* CES is the general model, Cobb-Douglas (sigma = 1) the reference specification: the unique member of the CES family whose predictions do not depend on lambda.
* `Y_i = c_i + lambda_i (Lambda_i - 1) c_i` is the part of the Anderson-Goeree-Holt (1998) moral terms that agent i controls; lambda = alpha/g.
* Price elasticity of giving: `eps = -[1 + (sigma-1)(1-pi)]` (CD: -1; Bakija-Heim -0.7 => sigma ~ 0.7).
* Altruism: `a = A (X/Y)^(1/sigma)`, `A = lambda B/(1-B)`; a = 0 iff lambda = 0; at the optimum a* = lambda p; MAT iff A' >= A and odds(B')/odds(B) (l'/l)^(1-1/sigma) >= 1.
* Exact leaf-centre condition on S_n: `c_L > c_C  iff  odds(pi_L)/odds(pi_C) > theta_L/theta_C`; with common B, reversal iff lambda > lbar_n(beta) = 2n/[(n^2+n)(2-beta) - 2 beta] and sigma > sigma* = 1 + log(theta_C/theta_L)/log(p_L/p_C). lbar_3(1.5) = 2.
* Identification: with K_3, centre and leaf observed, (D1, D2) = (sigma-1)(g1(lambda), g2(lambda)) identifies (sigma, lambda) iff sigma != 1.
* Correction: the 5th-report claim "n = 3 leaf > centre at every sigma for every beta in [6/5, 2) (impact-based)" holds only for beta >= 9/7; at beta = 1.2 it reverses for sigma > 9.72 (the earlier grid stopped at sigma = 8).

## Corrections to the 5th-report Chapter 2 (applied in the thesis text)

1. Introduction: incomplete sentence about the CES family.
2. Model: `theta_i in (0,1)` needs only `0 < beta < d_i`; `beta > 1` is a separate requirement.
3. CES: the claim that beta raises contributions through the Lambda channel holds only for sigma >= 1; the total effect can be negative under strong complements (example on K_8).
4. CES simulations: contributions do not decline monotonically in sigma in general (threshold `theta/(theta+Lambda)`; band (0.2, 0.25) on the regular graph at beta = 1.5).
5. CES ordering: "preserved at every sigma over the full admissible grid" is true for n = 3 but not for larger stars (impact-based, sigma >= 2); the centre-regular link fails under impact-based complements at small B.
6. Cycle: the claim that a higher B_i raises i's contribution above an "otherwise identical neighbour" is replaced by the exact statement (ordering by B only on C_3).
7. Monte Carlo: `Pr(P > C)` is the probability that the *mean* periphery contribution exceeds the centre's.
8. Notation: the CES prosocial weight is written B_i (= 1 - delta_i) throughout.
